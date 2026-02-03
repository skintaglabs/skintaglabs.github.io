"""Train all 3 modeling approaches and save comparison metrics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.model.embeddings import EmbeddingExtractor
from src.model.classifier import SklearnClassifier
from src.model.baseline import MajorityClassBaseline, RandomWeightedBaseline
from src.model.deep_classifier import DeepClassifier
from src.data.loader import load_ham10000, load_multi_dataset, CLASS_NAMES
from src.data.schema import samples_to_arrays
from src.data.sampler import compute_domain_balanced_weights
from src.evaluation.metrics import robustness_report, compare_models
from src.data.loader import get_demographic_groups


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--multi-dataset", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--domain-balance", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / "data"
    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config["extraction"]["batch_size_gpu"] if device == "cuda" else config["extraction"]["batch_size_cpu"]
    seed = config["training"]["seed"]

    # Load data
    if args.multi_dataset:
        datasets_to_load = args.datasets or config.get("data", {}).get("datasets", None)
        dataset_options = config.get("data", {}).get("dataset_options", {})
        samples = load_multi_dataset(data_dir, datasets=datasets_to_load, dataset_options=dataset_options)
        images, labels, metadata = samples_to_arrays(samples)
    else:
        images, labels, metadata = load_ham10000(data_dir, binary=True)
        labels = np.array(labels)

    print(f"Loaded {len(images)} images")

    if args.sample > 0 and args.sample < len(images):
        np.random.seed(seed)
        indices = np.random.choice(len(images), args.sample, replace=False)
        images = [images[i] for i in indices]
        labels = labels[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)

    # Extract embeddings
    cache_path = cache_dir / "embeddings_all_models.pt"
    extractor = EmbeddingExtractor(device=device)
    embeddings = extractor.extract_dataset(images, batch_size=batch_size, cache_path=cache_path)
    extractor.unload_model()

    # Split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        embeddings.numpy(), labels, metadata,
        test_size=0.2, random_state=seed, stratify=labels
    )

    # Domain-balanced weights
    sample_weights = None
    if args.domain_balance and "domain" in meta_train.columns:
        sample_weights = compute_domain_balanced_weights(
            meta_train["domain"].values, y_train
        )

    # Train all models (baseline, classical, gradient boosted, deep)
    models = {
        "majority_baseline": MajorityClassBaseline(),
        "random_baseline": RandomWeightedBaseline(seed=seed),
        "logistic_regression": SklearnClassifier(classifier_type="logistic"),
        "xgboost": SklearnClassifier(classifier_type="xgboost"),
        "deep_mlp": DeepClassifier(embedding_dim=X_train.shape[1], device=device),
    }

    all_results = {}
    test_groups = get_demographic_groups(meta_test)

    for name, clf in models.items():
        print(f"\n--- Training: {name} ---")

        if name in ("majority_baseline", "random_baseline"):
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

        report = robustness_report(
            y_test, y_pred,
            groups=test_groups if test_groups else None,
            class_names=["benign", "malignant"],
            y_proba=y_proba,
        )

        all_results[name] = {
            "train_accuracy": float(clf.score(X_train, y_train)),
            "test_accuracy": float(report["overall_accuracy"]),
            "balanced_accuracy": float(report["balanced_accuracy"]),
            "f1_binary": float(report["f1_binary"]),
            "f1_macro": float(report["f1_macro"]),
            "f1_weighted": float(report["f1_weighted"]),
            "auc": float(report.get("auc", float("nan"))),
        }

        # Save model
        with open(cache_dir / f"classifier_{name}.pkl", "wb") as f:
            pickle.dump(clf, f)

        print(f"  Train acc: {all_results[name]['train_accuracy']:.3f}")
        print(f"  Test acc: {all_results[name]['test_accuracy']:.3f}")
        print(f"  Balanced acc: {all_results[name]['balanced_accuracy']:.3f}")
        print(f"  F1 macro: {all_results[name]['f1_macro']:.3f}  F1 (malignant): {all_results[name]['f1_binary']:.3f}")

    # Comparison
    comparison = compare_models(all_results)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Train':>8} {'Test':>8} {'Bal Acc':>8} {'F1 Mac':>8} {'F1 Mal':>8} {'AUC':>8}")
    print("-" * 80)
    for name, res in all_results.items():
        print(f"{name:<25} {res['train_accuracy']:>8.3f} {res['test_accuracy']:>8.3f} "
              f"{res['balanced_accuracy']:>8.3f} {res['f1_macro']:>8.3f} "
              f"{res['f1_binary']:>8.3f} {res['auc']:>8.3f}")
    print(f"\nBest model: {comparison['best_model']}")

    # Save
    output = {
        "model_results": all_results,
        "comparison": comparison,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    def convert(obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(cache_dir / "model_comparison.json", "w") as f:
        json.dump(json.loads(json.dumps(output, default=convert)), f, indent=2)
    print(f"\nResults saved to {cache_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
