"""Training script for skin lesion classifier."""

import sys
from pathlib import Path

# Add project root to path before any src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier
from src.data.loader import load_ham10000, CLASS_NAMES


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

    # Load HAM10000 dataset
    binary = config["data"].get("binary_classification", True)
    print(f"Loading HAM10000 dataset (binary={binary})...")

    try:
        images, labels, metadata = load_ham10000(data_dir, binary=binary)
        labels = np.array(labels)
        class_names = ["benign", "malignant"] if binary else list(CLASS_NAMES.keys())
        print(f"Loaded {len(images)} images across {len(class_names)} classes")
        print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # Save metadata for evaluation
        metadata.to_csv(cache_dir / "metadata.csv", index=False)
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("\nDownload HAM10000 from Kaggle:")
        print("  kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset")
        print("  unzip skin-cancer-dataset.zip -d data/")
        return

    if len(images) == 0:
        print("No images loaded.")
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
