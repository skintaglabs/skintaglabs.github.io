#!/usr/bin/env python3
"""Export trained MobileNetV3 model to mobile formats (ONNX, Core ML, TFLite).

This script converts the PyTorch MobileNetV3-Large model to formats optimized
for deployment on iOS (Core ML) and Android (TFLite).

Usage:
    python scripts/export_mobile_models.py --model_path models/mobilenet_distilled/mobilenet_v3_large.pt \
        --output_dir models/mobile_exports
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MobileNetClassifier(nn.Module):
    """MobileNetV3-Large with custom classification head."""

    def __init__(self, n_classes=2, dropout=0.2):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class MobileNetWithSoftmax(nn.Module):
    """Wrapper that adds softmax for deployment."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        return torch.softmax(logits, dim=1)


def load_model(model_path, device='cpu'):
    """Load the trained PyTorch model."""
    model = MobileNetClassifier(n_classes=2, dropout=0.2)

    # Try loading state dict first
    try:
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'backbone.features.0.0.weight' in state_dict:
            model.load_state_dict(state_dict)
        else:
            # It might be a full model
            model = torch.load(model_path, map_location=device)
    except:
        model = torch.load(model_path, map_location=device)

    model.eval()
    return model


def export_onnx(model, output_path, image_size=224, opset_version=12):
    """Export model to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Wrap model with softmax for deployment
    model_with_softmax = MobileNetWithSoftmax(model)
    model_with_softmax.eval()

    # Export
    torch.onnx.export(
        model_with_softmax,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['probabilities'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'probabilities': {0: 'batch_size'}
        }
    )

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"  ONNX model exported successfully")
    print(f"  Size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

    return output_path


def export_coreml(onnx_path, output_path, image_size=224):
    """Convert ONNX model to Core ML format for iOS."""
    print(f"\nConverting to Core ML: {output_path}")

    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils

        # Load ONNX model and convert
        model = ct.converters.onnx.convert(
            model=str(onnx_path),
            minimum_deployment_target=ct.target.iOS15,
        )

        # Add metadata
        model.author = "SkinTag"
        model.short_description = "Skin lesion triage model (MobileNetV3-Large)"
        model.version = "1.0"

        # Set input/output descriptions
        model.input_description['image'] = "RGB image of skin lesion (224x224)"
        model.output_description['probabilities'] = "Probability [benign, malignant]"

        # Save FP32 version
        model.save(output_path)
        print(f"  Core ML model (FP32) saved")
        print(f"  Size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")

        # Create FP16 quantized version
        fp16_path = str(output_path).replace('.mlmodel', '_fp16.mlmodel')
        model_fp16 = quantization_utils.quantize_weights(model, nbits=16)
        model_fp16.save(fp16_path)
        print(f"  Core ML model (FP16) saved: {fp16_path}")
        print(f"  Size: {Path(fp16_path).stat().st_size / (1024*1024):.2f} MB")

        return output_path, fp16_path

    except ImportError:
        print("  Warning: coremltools not installed. Skipping Core ML export.")
        print("  Install with: pip install coremltools")
        return None, None
    except Exception as e:
        print(f"  Error converting to Core ML: {e}")
        return None, None


def export_tflite(onnx_path, output_dir, image_size=224, representative_images=None):
    """Convert ONNX model to TFLite format for Android."""
    print(f"\nConverting to TFLite...")

    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        output_dir = Path(output_dir)

        # Step 1: ONNX -> TensorFlow SavedModel
        print("  Step 1: Converting ONNX to TensorFlow...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)

        savedmodel_path = output_dir / "tf_savedmodel"
        tf_rep.export_graph(str(savedmodel_path))
        print(f"  TF SavedModel saved to: {savedmodel_path}")

        # Step 2: TensorFlow -> TFLite (FP32)
        print("  Step 2: Converting to TFLite (FP32)...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_path))
        tflite_model = converter.convert()

        fp32_path = output_dir / "model_fp32.tflite"
        with open(fp32_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  TFLite (FP32) saved: {fp32_path}")
        print(f"  Size: {fp32_path.stat().st_size / (1024*1024):.2f} MB")

        # Step 3: TFLite with FP16 quantization
        print("  Step 3: Converting to TFLite (FP16)...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()

        fp16_path = output_dir / "model_fp16.tflite"
        with open(fp16_path, 'wb') as f:
            f.write(tflite_fp16)
        print(f"  TFLite (FP16) saved: {fp16_path}")
        print(f"  Size: {fp16_path.stat().st_size / (1024*1024):.2f} MB")

        # Step 4: TFLite with INT8 quantization (requires representative dataset)
        if representative_images is not None:
            print("  Step 4: Converting to TFLite (INT8)...")

            def representative_dataset():
                for img in representative_images[:100]:
                    # Ensure correct shape and type
                    img = np.expand_dims(img, axis=0).astype(np.float32)
                    yield [img]

            converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_path))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32

            try:
                tflite_int8 = converter.convert()
                int8_path = output_dir / "model_int8.tflite"
                with open(int8_path, 'wb') as f:
                    f.write(tflite_int8)
                print(f"  TFLite (INT8) saved: {int8_path}")
                print(f"  Size: {int8_path.stat().st_size / (1024*1024):.2f} MB")
            except Exception as e:
                print(f"  Warning: INT8 quantization failed: {e}")
                int8_path = None
        else:
            print("  Skipping INT8 quantization (no representative dataset)")
            int8_path = None

        return fp32_path, fp16_path, int8_path

    except ImportError as e:
        print(f"  Warning: Required packages not installed: {e}")
        print("  Install with: pip install tensorflow onnx-tf")
        return None, None, None
    except Exception as e:
        print(f"  Error converting to TFLite: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def verify_onnx_model(onnx_path, pytorch_model, image_size=224):
    """Verify ONNX model produces same outputs as PyTorch."""
    print("\nVerifying ONNX model...")

    import onnxruntime as ort

    # Create test input
    test_input = torch.randn(1, 3, image_size, image_size)

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = torch.softmax(pytorch_model(test_input), dim=1).numpy()

    # ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_input)[0]

    # Compare
    max_diff = np.abs(pt_output - onnx_output).max()
    print(f"  Max difference between PyTorch and ONNX: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("  Verification PASSED")
        return True
    else:
        print("  Verification FAILED - outputs differ significantly")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export MobileNet to mobile formats")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained PyTorch model")
    parser.add_argument("--output_dir", type=str, default="models/mobile_exports",
                        help="Output directory for exported models")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--skip_coreml", action="store_true",
                        help="Skip Core ML export")
    parser.add_argument("--skip_tflite", action="store_true",
                        help="Skip TFLite export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    print("Model loaded successfully")

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    export_onnx(model, str(onnx_path), args.image_size)

    # Verify ONNX
    verify_onnx_model(onnx_path, model, args.image_size)

    # Export to Core ML (iOS)
    if not args.skip_coreml:
        coreml_path = output_dir / "SkinTagModel.mlmodel"
        export_coreml(onnx_path, str(coreml_path), args.image_size)

    # Export to TFLite (Android)
    if not args.skip_tflite:
        tflite_dir = output_dir / "tflite"
        tflite_dir.mkdir(exist_ok=True)
        export_tflite(onnx_path, tflite_dir, args.image_size)

    # Save export summary
    summary = {
        'source_model': str(args.model_path),
        'image_size': args.image_size,
        'exports': {
            'onnx': str(onnx_path),
            'coreml': str(output_dir / "SkinTagModel.mlmodel") if not args.skip_coreml else None,
            'tflite_fp32': str(output_dir / "tflite" / "model_fp32.tflite") if not args.skip_tflite else None,
            'tflite_fp16': str(output_dir / "tflite" / "model_fp16.tflite") if not args.skip_tflite else None,
        },
        'preprocessing': {
            'resize': [args.image_size, args.image_size],
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
        },
        'output_classes': ['benign', 'malignant'],
    }

    with open(output_dir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("Export complete!")
    print(f"{'='*50}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
