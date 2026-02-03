import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/triage_result.dart';

/// Service for loading and running TFLite model inference
class ModelService {
  Interpreter? _interpreter;

  // Model input specifications
  static const int inputSize = 224;
  static const int numClasses = 2;

  // ImageNet normalization values
  static const List<double> mean = [0.485, 0.456, 0.406];
  static const List<double> std = [0.229, 0.224, 0.225];

  /// Load the TFLite model
  Future<void> loadModel() async {
    try {
      // Try loading from assets
      _interpreter = await Interpreter.fromAsset('assets/models/model_fp16.tflite');
    } catch (e) {
      // Try alternative model path
      try {
        _interpreter = await Interpreter.fromAsset('assets/models/model.tflite');
      } catch (e2) {
        throw Exception('Failed to load model: $e2');
      }
    }
  }

  /// Run inference on an image file
  Future<TriageResult> predict(File imageFile) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Load and preprocess image
    final imageBytes = await imageFile.readAsBytes();
    final input = _preprocessImage(imageBytes);

    // Prepare output buffer [1, 2] for probabilities
    final output = List.filled(1 * numClasses, 0.0).reshape([1, numClasses]);

    // Run inference
    _interpreter!.run(input, output);

    // Extract probabilities
    final probabilities = output[0] as List<double>;

    return TriageResult(
      benignProbability: probabilities[0],
      malignantProbability: probabilities[1],
    );
  }

  /// Run inference on raw bytes (from camera)
  Future<TriageResult> predictFromBytes(Uint8List imageBytes) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    final input = _preprocessImage(imageBytes);

    // Prepare output buffer
    final output = List.filled(1 * numClasses, 0.0).reshape([1, numClasses]);

    // Run inference
    _interpreter!.run(input, output);

    final probabilities = output[0] as List<double>;

    return TriageResult(
      benignProbability: probabilities[0],
      malignantProbability: probabilities[1],
    );
  }

  /// Preprocess image for model input
  List<List<List<List<double>>>> _preprocessImage(Uint8List imageBytes) {
    // Decode image
    final image = img.decodeImage(imageBytes);
    if (image == null) {
      throw Exception('Failed to decode image');
    }

    // Resize to model input size
    final resizedImage = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Create input tensor [1, 224, 224, 3] with normalized RGB values
    final input = List.generate(
      1,
      (_) => List.generate(
        inputSize,
        (y) => List.generate(
          inputSize,
          (x) {
            final pixel = resizedImage.getPixel(x, y);
            // Normalize: (pixel / 255 - mean) / std
            return [
              ((pixel.r / 255.0) - mean[0]) / std[0],
              ((pixel.g / 255.0) - mean[1]) / std[1],
              ((pixel.b / 255.0) - mean[2]) / std[2],
            ];
          },
        ),
      ),
    );

    return input;
  }

  /// Dispose of the interpreter
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}

extension ListReshape<T> on List<T> {
  List<dynamic> reshape(List<int> shape) {
    if (shape.length == 1) {
      return this;
    }

    int totalElements = shape.reduce((a, b) => a * b);
    if (length != totalElements) {
      throw ArgumentError('Cannot reshape list of length $length to shape $shape');
    }

    List<dynamic> result = List.from(this);
    for (int i = shape.length - 1; i > 0; i--) {
      final size = shape[i];
      final newResult = <List<dynamic>>[];
      for (int j = 0; j < result.length; j += size) {
        newResult.add(result.sublist(j, j + size));
      }
      result = newResult;
    }

    return result;
  }
}
