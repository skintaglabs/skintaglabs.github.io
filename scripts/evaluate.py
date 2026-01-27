"""Evaluation script for robustness assessment."""

import sys
from pathlib import Path

# Add project root to path before any src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pickle
import torch
import numpy as np
from PIL import Image

from src.model.embeddings import EmbeddingExtractor
from src.data.augmentations import (
    get_lighting_augmentation,
    get_noise_augmentation,
    get_compression_augmentation,
)
from src.evaluation.metrics import robustness_report


def apply_augmentation(images, augmentation):
    """Apply augmentation to a list of PIL images."""
    augmented = []
    for img in images:
        aug_array = augmentation(image=np.array(img))["image"]
        augmented.append(Image.fromarray(aug_array))
    return augmented


def main():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_dir = Path(__file__).parent.parent / "results" / "cache"

    # Load classifier
    model_path = cache_dir / "classifier.pkl"
    if not model_path.exists():
        print("No trained model found. Run train.py first.")
        return

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Load test embeddings (assuming same split as training)
    embedding_cache = cache_dir / "embeddings.pt"
    if not embedding_cache.exists():
        print("No cached embeddings. Run train.py first.")
        return

    embeddings = torch.load(embedding_cache)
    print(f"Loaded embeddings: {embeddings.shape}")

    # For full robustness testing, you'd re-extract embeddings on augmented images
    # Here we show a simpler approach: evaluate on cached embeddings

    # Placeholder: Load test labels
    # In practice, save train/test split indices during training
    print("\nTo run full robustness evaluation:")
    print("1. Load test images")
    print("2. Apply augmentations (lighting, noise, compression)")
    print("3. Re-extract embeddings for each augmented set")
    print("4. Compare accuracy across conditions")
    print("\nSee notebooks/demo.ipynb for the complete workflow.")


if __name__ == "__main__":
    main()
