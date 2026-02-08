"""Evaluation script — full fairness report across all demographic axes."""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import pandas as pd

from src.evaluation.metrics import robustness_report, compare_models
from src.data.loader import get_demographic_groups
from src.model.triage import TriageSystem


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["logistic"], help="Models to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / "results" / "cache"
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load test data
    test_meta_path = cache_dir / "test_metadata.csv"
    embeddings_path = cache_dir / "embeddings.pt"

    if not test_meta_path.exists():
        print("No test metadata found. Run train.py first.")
        return
    if not embeddings_path.exists():
        print("No cached embeddings. Run train.py first.")
        return

    test_meta = pd.read_csv(test_meta_path)
    y_test = test_meta["label"].values if "label" in test_meta.columns else None

    if y_test is None:
        print("No label column in test metadata.")
        return

    # Get demographic groups
    groups = get_demographic_groups(test_meta)
    print(f"Test samples: {len(y_test)}")
    print(f"Demographic axes: {list(groups.keys())}")

    # Load test embeddings (we need to reconstruct test split indices)
    import torch
    embeddings = torch.load(embeddings_path)

    # Evaluate each model
    all_results = {}
    triage_system = TriageSystem(config.get("triage", {}))

    for model_name in args.models:
        model_path = cache_dir / f"classifier_{model_name}.pkl"
        if not model_path.exists():
            model_path = cache_dir / "classifier.pkl"
        if not model_path.exists():
            print(f"Model {model_name} not found, skipping")
            continue

        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        # We need test embeddings — use saved test metadata indices
        # For now, re-split using same seed
        from sklearn.model_selection import train_test_split
        all_meta = pd.read_csv(cache_dir / "metadata.csv")
        n_total = len(all_meta)
        indices = np.arange(n_total)

        labels_all = all_meta["label"].values if "label" in all_meta.columns else None
        if labels_all is None:
            # Legacy HAM10000 metadata
            from src.data.loader import BINARY_MAPPING
            if "dx" in all_meta.columns:
                labels_all = np.array([BINARY_MAPPING.get(dx, 0) for dx in all_meta["dx"]])
            else:
                print("Cannot determine labels from metadata")
                continue

        _, test_indices = train_test_split(
            indices, test_size=0.2, random_state=config["training"]["seed"],
            stratify=labels_all
        )

        X_test = embeddings[test_indices].numpy()
        y_test_actual = labels_all[test_indices]

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

        # Get groups for test set
        test_meta_subset = all_meta.iloc[test_indices].reset_index(drop=True)
        test_groups = get_demographic_groups(test_meta_subset)

        # Full robustness report
        report = robustness_report(
            y_test_actual, y_pred,
            groups=test_groups,
            class_names=["benign", "malignant"],
            y_proba=y_proba,
        )

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Overall accuracy:  {report['overall_accuracy']:.3f}")
        print(f"Balanced accuracy: {report['balanced_accuracy']:.3f}")
        print(f"F1 (malignant):    {report['f1_binary']:.3f}")
        print(f"F1 macro:          {report['f1_macro']:.3f}")
        print(f"F1 weighted:       {report['f1_weighted']:.3f}")
        if "auc" in report:
            print(f"AUC:               {report['auc']:.3f}")
        print(f"\n{report['classification_report']}")

        # Print fairness metrics per axis
        for axis in test_groups:
            key = f"per_{axis}"
            if key in report:
                print(f"\n--- {axis.upper()} breakdown ---")
                for group, metrics in report[key].items():
                    print(f"  {group}: acc={metrics['accuracy']:.3f}, "
                          f"f1={metrics['f1']:.3f}, "
                          f"sens={metrics['sensitivity']:.3f}, "
                          f"spec={metrics['specificity']:.3f}, "
                          f"n={metrics['n']}")
                gap_key = f"{axis}_fairness_gap"
                if gap_key in report:
                    print(f"  Fairness gap: {report[gap_key]:.3f}")
                eq_key = f"{axis}_equalized_odds"
                if eq_key in report:
                    eq = report[eq_key]
                    print(f"  Equalized odds gap: sens={eq['sensitivity_gap']:.3f}, spec={eq['specificity_gap']:.3f}")

        # Triage distribution
        if y_proba is not None:
            mal_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            triage_results = triage_system.assess_batch(mal_proba)
            tiers = [t.urgency_tier for t in triage_results]
            print(f"\n--- Triage Distribution ---")
            for tier in ["low", "moderate", "high"]:
                count = tiers.count(tier)
                print(f"  {tier}: {count} ({count/len(tiers)*100:.1f}%)")

        all_results[model_name] = report

    # Compare models
    if len(all_results) > 1:
        summary = {
            name: {
                "accuracy": r["overall_accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "f1_macro": r["f1_macro"],
                "f1_binary": r["f1_binary"],
                "auc": r.get("auc", float("nan")),
            }
            for name, r in all_results.items()
        }
        comparison = compare_models(summary)
        print(f"\n{'='*60}")
        print(f"Best model: {comparison['best_model']} (metric={comparison['best_metric']:.3f})")

    # Save results
    output_path = args.output or str(cache_dir / "evaluation_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
