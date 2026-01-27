"""Training script for skin lesion classifier."""

import sys
import yaml
import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier


def load_images_and_labels(data_dir: Path):
    """Load images and infer labels from directory structure.

    Expected structure:
        data_dir/
            benign/
                image1.jpg
            malignant/
                image2.jpg
    """
    images = []
    labels = []
    class_names = ["benign", "malignant"]

    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpg"):
            images.append(Image.open(img_path).convert("RGB"))
            labels.append(label_idx)

    return images, np.array(labels), class_names


def main():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(__file__).parent.parent / "data"
    cache_dir = Path(__file__).parent.parent / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine batch size based on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config["extraction"]["batch_size_gpu"] if device == "cuda" else config["extraction"]["batch_size_cpu"]
    print(f"Device: {device}, Batch size: {batch_size}")

    # Load data
    print("Loading images...")
    images, labels, class_names = load_images_and_labels(data_dir)
    print(f"Loaded {len(images)} images across {len(class_names)} classes")

    if len(images) == 0:
        print(f"No images found. Create data directory with structure:")
        print(f"  {data_dir}/benign/*.jpg")
        print(f"  {data_dir}/malignant/*.jpg")
        return

    # Extract embeddings
    embedding_cache = cache_dir / "embeddings.pt"
    extractor = EmbeddingExtractor(device=device)
    embeddings = extractor.extract_dataset(images, batch_size=batch_size, cache_path=embedding_cache)
    extractor.unload_model()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings.numpy(), labels, test_size=0.2, random_state=config["training"]["seed"], stratify=labels
    )

    # Train classifier
    print(f"Training {config['training']['classifier']} classifier...")
    clf = SklearnClassifier(classifier_type=config["training"]["classifier"])
    clf.fit(X_train, y_train)

    print(f"Train accuracy: {clf.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

    # Save model
    model_path = cache_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

    # Output structured results for CI summary
    print("\n--- RESULTS ---")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Train Accuracy | {clf.score(X_train, y_train):.3f} |")
    print(f"| Test Accuracy | {clf.score(X_test, y_test):.3f} |")
    print(f"| Train Samples | {len(X_train)} |")
    print(f"| Test Samples | {len(X_test)} |")
    print(f"| Embedding Dim | {embeddings.shape[1]} |")


if __name__ == "__main__":
    main()
