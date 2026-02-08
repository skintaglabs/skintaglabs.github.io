"""Training script for skin lesion classifier — supports multi-dataset and 3 model types."""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

import sys
from pathlib import Path

# Add project root to path before any src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier
from src.model.baseline import MajorityClassBaseline, RandomWeightedBaseline
from src.model.deep_classifier import DeepClassifier
from src.data.loader import load_ham10000, load_multi_dataset, CLASS_NAMES
from src.data.schema import samples_to_arrays
from src.data.sampler import compute_domain_balanced_weights, compute_combined_balanced_weights, compute_stratified_split_key


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Use only N samples (0 = all)")
    parser.add_argument("--multi-dataset", action="store_true", help="Use multi-dataset loading")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to load (default: all available)")
    parser.add_argument("--domain-balance", action="store_true", help="Use domain-balanced sample weights")
    parser.add_argument("--model", choices=["all", "logistic", "deep", "baseline"], default="logistic",
                        help="Which model(s) to train")
    args = parser.parse_args()

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

    binary = config["data"].get("binary_classification", True)

    # Load data — multi-dataset or HAM10000-only
    if args.multi_dataset:
        datasets_to_load = args.datasets or config.get("data", {}).get("datasets", None)
        dataset_options = config.get("data", {}).get("dataset_options", {})
        samples = load_multi_dataset(data_dir, datasets=datasets_to_load, dataset_options=dataset_options)
        images, labels, metadata = samples_to_arrays(samples)
        class_names = ["benign", "malignant"]
    else:
        print(f"Loading HAM10000 dataset (binary={binary})...")
        try:
            images, labels, metadata = load_ham10000(data_dir, binary=binary)
            labels = np.array(labels)
            class_names = ["benign", "malignant"] if binary else list(CLASS_NAMES.keys())
        except FileNotFoundError as e:
            print(f"Dataset not found: {e}")
            print("\nDownload HAM10000 from Kaggle:")
            print("  kaggle datasets download -d farjanakabirsamanta/skin-cancer-dataset")
            print("  unzip skin-cancer-dataset.zip -d data/")
            return

    print(f"Loaded {len(images)} images across {len(class_names)} classes")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(images) == 0:
        print("No images loaded.")
        return

    # Save metadata for evaluation
    metadata.to_csv(cache_dir / "metadata.csv", index=False)

    # Sample if requested (for CI/testing)
    if args.sample > 0 and args.sample < len(images):
        print(f"Sampling {args.sample} images...")
        np.random.seed(config["training"]["seed"])
        indices = np.random.choice(len(images), args.sample, replace=False)
        images = [images[i] for i in indices]
        labels = labels[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)
        print(f"Sampled {len(images)} images")

    # Extract embeddings
    embedding_cache = cache_dir / "embeddings.pt"
    extractor = EmbeddingExtractor(device=device)
    embeddings = extractor.extract_dataset(images, batch_size=batch_size, cache_path=embedding_cache)
    extractor.unload_model()

    # Stratified split — use (label, domain) composite key if multi-dataset
    if args.multi_dataset and "domain" in metadata.columns:
        domains = metadata["domain"].values
        stratify_key = compute_stratified_split_key(labels, domains)
        # Fall back to label-only if any group is too small
        unique, counts = np.unique(stratify_key, return_counts=True)
        if counts.min() < 2:
            print("Warning: some (label, domain) groups too small for stratified split, using label only")
            stratify_key = labels
    else:
        stratify_key = labels
        domains = None

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        embeddings.numpy(), labels, metadata,
        test_size=0.2, random_state=config["training"]["seed"], stratify=stratify_key
    )

    # Save test metadata for evaluation
    meta_test.to_csv(cache_dir / "test_metadata.csv", index=False)

    # Compute balanced weights (domain + Fitzpatrick skin tone)
    sample_weights = None
    if args.domain_balance and "domain" in meta_train.columns:
        train_domains = meta_train["domain"].values
        if "fitzpatrick" in meta_train.columns:
            # Combined domain + Fitzpatrick balancing — addresses both domain shift
            # and historical under-sampling of darker skin tones
            train_fitz = meta_train["fitzpatrick"].values
            sample_weights = compute_combined_balanced_weights(train_domains, train_fitz, y_train)
            print(f"Combined domain+Fitzpatrick balanced weights (range: {sample_weights.min():.3f} - {sample_weights.max():.3f})")
        else:
            sample_weights = compute_domain_balanced_weights(train_domains, y_train)
            print(f"Domain-balanced weights (range: {sample_weights.min():.3f} - {sample_weights.max():.3f})")

    # Train model(s)
    results = {}
    models_to_train = ["baseline", "logistic", "deep"] if args.model == "all" else [args.model]

    for model_type in models_to_train:
        print(f"\n--- Training {model_type} ---")

        if model_type == "baseline":
            clf = MajorityClassBaseline()
            clf.fit(X_train, y_train)
        elif model_type == "logistic":
            clf = SklearnClassifier(classifier_type="logistic")
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        elif model_type == "deep":
            clf = DeepClassifier(
                embedding_dim=embeddings.shape[1],
                device=device,
            )
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            continue

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        test_f1 = float(f1_score(y_test, y_pred_test, average='macro', zero_division=0))
        train_f1 = float(f1_score(y_train, y_pred_train, average='macro', zero_division=0))
        test_f1_bin = float(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))

        print(f"  Train accuracy: {train_acc:.3f}  F1 macro: {train_f1:.3f}")
        print(f"  Test accuracy:  {test_acc:.3f}  F1 macro: {test_f1:.3f}  F1 (malignant): {test_f1_bin:.3f}")

        results[model_type] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_f1_macro": train_f1,
            "test_f1_macro": test_f1,
            "test_f1_malignant": test_f1_bin,
        }

        # Save model
        model_path = cache_dir / f"classifier_{model_type}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Model saved to {model_path}")

        # Also save as default classifier.pkl if it's the selected one
        if model_type == config["training"].get("classifier", "logistic") or model_type == "logistic":
            default_path = cache_dir / "classifier.pkl"
            with open(default_path, "wb") as f:
                pickle.dump(clf, f)

    # Save comparison results
    with open(cache_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Output structured results
    print("\n--- RESULTS ---")
    print(f"| Model | Train Acc | Test Acc | F1 Macro | F1 Malignant |")
    print(f"|-------|-----------|----------|----------|--------------|")
    for model_type, res in results.items():
        print(f"| {model_type} | {res['train_accuracy']:.3f} | {res['test_accuracy']:.3f} "
              f"| {res['test_f1_macro']:.3f} | {res['test_f1_malignant']:.3f} |")
    print(f"| Samples | {len(X_train)} train | {len(X_test)} test |")
    print(f"| Embedding Dim | {embeddings.shape[1]} |")


if __name__ == "__main__":
    main()
