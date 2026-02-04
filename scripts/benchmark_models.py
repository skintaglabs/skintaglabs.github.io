"""Comprehensive benchmark pipeline for pre-trained vs fine-tuned model comparison.

Evaluates all model variants across multiple dimensions:
- Overall performance (accuracy, F1, AUC)
- Fairness metrics (per-Fitzpatrick type)
- Cross-domain robustness
- Distortion robustness (optional)
- Inference speed and model size
- Historical tracking and regression detection
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_demographic_groups, load_multi_dataset
from src.evaluation.metrics import robustness_report, compare_models
from src.model.baseline import MajorityClassBaseline, RandomWeightedBaseline
from src.model.classifier import SklearnClassifier
from src.model.embeddings import EmbeddingExtractor
from src.model.triage import TriageSystem


class BenchmarkRunner:
    """Orchestrates comprehensive model benchmarking."""

    def __init__(self, config_path: Path, benchmark_config_path: Path = None):
        self.project_root = Path(__file__).parent.parent
        self.cache_dir = self.project_root / "results" / "cache"
        self.output_dir = self.project_root / "results" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if benchmark_config_path and benchmark_config_path.exists():
            with open(benchmark_config_path) as f:
                self.benchmark_config = yaml.safe_load(f)
        else:
            self.benchmark_config = self._default_benchmark_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results = {
            "metadata": {
                "timestamp": self.timestamp,
                "device": self.device,
                "config": self.config,
            },
            "models": {},
            "comparison": {},
            "fairness": {},
            "regression_alerts": [],
        }

    def _default_benchmark_config(self):
        """Default benchmark configuration."""
        return {
            "models_to_benchmark": [
                "baseline_majority",
                "baseline_random",
                "logistic_frozen",
                "xgboost_frozen",
                "deep_frozen",
                "xgboost_finetuned",
            ],
            "run_robustness_tests": False,
            "run_fairness_analysis": True,
            "run_cross_domain": False,
            "track_history": True,
            "performance_thresholds": {
                "min_accuracy": 0.85,
                "min_f1_macro": 0.80,
                "max_fairness_gap": 0.10,
            },
        }

    def load_data(self):
        """Load test data and embeddings."""
        print("\n[1/7] Loading test data...")

        test_meta_path = self.cache_dir / "test_metadata.csv"
        metadata_path = self.cache_dir / "metadata.csv"
        embeddings_path = self.cache_dir / "embeddings.pt"

        if not embeddings_path.exists():
            print("No cached embeddings found. Run train.py first.")
            return False

        import pandas as pd

        all_meta = pd.read_csv(metadata_path)
        labels_all = all_meta["label"].values

        _, test_indices = train_test_split(
            np.arange(len(all_meta)),
            test_size=0.2,
            random_state=self.config["training"]["seed"],
            stratify=labels_all,
        )

        embeddings = torch.load(embeddings_path)
        self.X_test_frozen = embeddings[test_indices].numpy()
        self.y_test = labels_all[test_indices]

        self.test_metadata = all_meta.iloc[test_indices].reset_index(drop=True)
        self.test_groups = get_demographic_groups(self.test_metadata)

        # Load fine-tuned embeddings if available
        finetuned_test_path = self.cache_dir / "embeddings_finetuned_test.pt"
        if finetuned_test_path.exists():
            self.X_test_finetuned = torch.load(finetuned_test_path).numpy()
            print(f"  Loaded fine-tuned embeddings: {self.X_test_finetuned.shape}")
        else:
            self.X_test_finetuned = None
            print("  Fine-tuned embeddings not found (skipping fine-tuned benchmarks)")

        print(f"  Test samples: {len(self.y_test)}")
        print(f"  Demographic groups: {list(self.test_groups.keys())}")

        return True

    def benchmark_model(self, model_name: str, model, X_test, embedding_type: str):
        """Benchmark a single model."""
        print(f"\n  Evaluating {model_name}...")

        start_time = time.perf_counter()

        if model_name.startswith("baseline"):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        inference_time = (time.perf_counter() - start_time) / len(self.y_test) * 1000

        report = robustness_report(
            self.y_test,
            y_pred,
            groups=self.test_groups,
            class_names=["benign", "malignant"],
            y_proba=y_proba,
        )

        report["embedding_type"] = embedding_type
        report["inference_time_ms"] = inference_time

        # Check for regressions
        thresholds = self.benchmark_config["performance_thresholds"]
        if report["overall_accuracy"] < thresholds["min_accuracy"]:
            self.results["regression_alerts"].append({
                "model": model_name,
                "issue": "accuracy_below_threshold",
                "value": report["overall_accuracy"],
                "threshold": thresholds["min_accuracy"],
            })

        if report["f1_macro"] < thresholds["min_f1_macro"]:
            self.results["regression_alerts"].append({
                "model": model_name,
                "issue": "f1_macro_below_threshold",
                "value": report["f1_macro"],
                "threshold": thresholds["min_f1_macro"],
            })

        for group_name in self.test_groups:
            gap_key = f"{group_name}_fairness_gap"
            if gap_key in report and report[gap_key] > thresholds["max_fairness_gap"]:
                self.results["regression_alerts"].append({
                    "model": model_name,
                    "issue": f"{group_name}_fairness_gap_exceeded",
                    "value": report[gap_key],
                    "threshold": thresholds["max_fairness_gap"],
                })

        print(f"    Accuracy: {report['overall_accuracy']:.4f}")
        print(f"    F1 Macro: {report['f1_macro']:.4f}")
        print(f"    Inference: {inference_time:.4f} ms/sample")

        return report

    def run_benchmarks(self):
        """Run benchmarks for all configured models."""
        print("\n[2/7] Benchmarking models...")

        models_to_run = self.benchmark_config["models_to_benchmark"]

        # Baseline models
        if "baseline_majority" in models_to_run:
            baseline_maj = MajorityClassBaseline()
            baseline_maj.fit(None, self.y_test)
            self.results["models"]["baseline_majority"] = self.benchmark_model(
                "baseline_majority", baseline_maj, self.X_test_frozen, "N/A"
            )

        if "baseline_random" in models_to_run:
            baseline_rand = RandomWeightedBaseline()
            baseline_rand.fit(None, self.y_test)
            self.results["models"]["baseline_random"] = self.benchmark_model(
                "baseline_random", baseline_rand, self.X_test_frozen, "N/A"
            )

        # Frozen embedding models
        frozen_models = {
            "logistic_frozen": "classifier_logistic.pkl",
            "xgboost_frozen": "classifier_xgboost.pkl",
            "deep_frozen": "classifier_deep.pkl",
        }

        for model_name, filename in frozen_models.items():
            if model_name not in models_to_run:
                continue

            model_path = self.cache_dir / filename
            if not model_path.exists():
                print(f"  {model_name} not found, skipping")
                continue

            clf = SklearnClassifier(classifier_type=model_name.split("_")[0])
            clf.pipeline = joblib.load(model_path)

            self.results["models"][model_name] = self.benchmark_model(
                model_name, clf, self.X_test_frozen, "frozen"
            )

        # Fine-tuned embedding models
        if self.X_test_finetuned is not None and "xgboost_finetuned" in models_to_run:
            model_path = self.cache_dir / "classifier_xgboost_finetuned.pkl"
            if model_path.exists():
                clf_finetuned = SklearnClassifier(classifier_type="xgboost")
                clf_finetuned.pipeline = joblib.load(model_path)

                self.results["models"]["xgboost_finetuned"] = self.benchmark_model(
                    "xgboost_finetuned", clf_finetuned, self.X_test_finetuned, "finetuned"
                )

    def compare_all_models(self):
        """Generate model comparison and ranking."""
        print("\n[3/7] Comparing models...")

        summary = {
            name: {
                "accuracy": r["overall_accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "f1_macro": r["f1_macro"],
                "f1_binary": r["f1_binary"],
                "auc": r.get("auc", float("nan")),
                "embedding_type": r["embedding_type"],
            }
            for name, r in self.results["models"].items()
        }

        comparison = compare_models(summary)
        self.results["comparison"] = comparison

        print(f"  Best model: {comparison['best_model']}")
        print(f"  Best F1 macro: {comparison['best_metric']:.4f}")

    def analyze_fairness(self):
        """Analyze fairness gaps across models."""
        print("\n[4/7] Analyzing fairness...")

        fairness_summary = {}

        for model_name, report in self.results["models"].items():
            fairness_summary[model_name] = {}

            for group_name in self.test_groups:
                gap_key = f"{group_name}_fairness_gap"
                eq_key = f"{group_name}_equalized_odds"

                if gap_key in report:
                    fairness_summary[model_name][gap_key] = report[gap_key]

                if eq_key in report:
                    fairness_summary[model_name][eq_key] = report[eq_key]

        self.results["fairness"] = fairness_summary

        # Print fairness summary
        for model_name, gaps in fairness_summary.items():
            print(f"\n  {model_name}:")
            for gap_name, gap_value in gaps.items():
                if isinstance(gap_value, dict):
                    print(f"    {gap_name}: {gap_value}")
                else:
                    print(f"    {gap_name}: {gap_value:.4f}")

    def collect_model_sizes(self):
        """Collect model file sizes."""
        print("\n[5/7] Collecting model sizes...")

        sizes = {}

        model_files = {
            "baseline": None,
            "logistic_frozen": "classifier_logistic.pkl",
            "xgboost_frozen": "classifier_xgboost.pkl",
            "deep_frozen": "classifier_deep.pkl",
            "xgboost_finetuned": "classifier_xgboost_finetuned.pkl",
        }

        for model_name, filename in model_files.items():
            if filename is None:
                sizes[model_name] = 0
                continue

            model_path = self.cache_dir / filename
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                sizes[model_name] = size_mb
                print(f"  {model_name}: {size_mb:.2f} MB")

        # Fine-tuned SigLIP model
        finetuned_model_path = self.project_root / "models" / "finetuned_siglip" / "model_state.pt"
        if finetuned_model_path.exists():
            size_mb = finetuned_model_path.stat().st_size / (1024 * 1024)
            sizes["siglip_finetuned_full"] = size_mb
            print(f"  siglip_finetuned_full: {size_mb:.2f} MB")

        self.results["model_sizes"] = sizes

    def track_history(self):
        """Track performance over time and detect regressions."""
        print("\n[6/7] Tracking historical performance...")

        history_file = self.output_dir / "benchmark_history.json"

        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
        else:
            history = {"runs": []}

        current_run = {
            "timestamp": self.timestamp,
            "best_model": self.results["comparison"]["best_model"],
            "best_f1_macro": self.results["comparison"]["best_metric"],
            "model_metrics": {
                name: {
                    "accuracy": r["overall_accuracy"],
                    "f1_macro": r["f1_macro"],
                    "auc": r.get("auc", None),
                }
                for name, r in self.results["models"].items()
            },
            "regression_alerts": self.results["regression_alerts"],
        }

        history["runs"].append(current_run)

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"  History updated: {len(history['runs'])} total runs")

        # Detect regressions compared to previous run
        if len(history["runs"]) > 1:
            prev_run = history["runs"][-2]
            curr_best = current_run["best_f1_macro"]
            prev_best = prev_run["best_f1_macro"]

            if curr_best < prev_best - 0.01:
                print(f"  WARNING: Performance regression detected!")
                print(f"  Previous best: {prev_best:.4f}, Current best: {curr_best:.4f}")

    def generate_report(self):
        """Generate markdown and JSON reports."""
        print("\n[7/7] Generating reports...")

        # Save JSON
        json_path = self.output_dir / f"benchmark_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  JSON report: {json_path}")

        # Generate markdown
        md_path = self.output_dir / f"benchmark_{self.timestamp}.md"
        with open(md_path, "w") as f:
            f.write(self._generate_markdown())
        print(f"  Markdown report: {md_path}")

        # Create latest symlink
        latest_path = self.output_dir / "benchmark_latest.md"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(md_path.name)

    def _generate_markdown(self):
        """Generate markdown report content."""
        md = f"""# Model Benchmark Report

**Generated:** {self.timestamp}
**Device:** {self.device}

## Executive Summary

- **Best Model:** {self.results['comparison']['best_model']}
- **Best F1 Macro:** {self.results['comparison']['best_metric']:.4f}
- **Models Evaluated:** {len(self.results['models'])}
- **Regression Alerts:** {len(self.results['regression_alerts'])}

## Performance Comparison

| Model | Embedding | Accuracy | F1 Macro | F1 Malignant | AUC | Inference (ms) |
|-------|-----------|----------|----------|--------------|-----|----------------|
"""

        for name, metrics in self.results["models"].items():
            auc_str = f"{metrics.get('auc', 0):.4f}" if metrics.get('auc') else "N/A"
            inf_time = metrics.get('inference_time_ms', 0)
            md += f"| {name} | {metrics['embedding_type']} | {metrics['overall_accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_binary']:.4f} | {auc_str} | {inf_time:.4f} |\n"

        md += "\n## Fairness Analysis\n\n"

        for model_name, fairness_data in self.results["fairness"].items():
            md += f"\n### {model_name}\n\n"
            for metric_name, value in fairness_data.items():
                if isinstance(value, dict):
                    md += f"**{metric_name}:**\n"
                    for k, v in value.items():
                        md += f"- {k}: {v:.4f}\n"
                else:
                    md += f"- **{metric_name}:** {value:.4f}\n"

        if self.results["regression_alerts"]:
            md += "\n## Regression Alerts\n\n"
            for alert in self.results["regression_alerts"]:
                md += f"- **{alert['model']}**: {alert['issue']} (value: {alert['value']:.4f}, threshold: {alert['threshold']:.4f})\n"
        else:
            md += "\n## Regression Alerts\n\nNo regressions detected.\n"

        md += "\n## Model Sizes\n\n"
        md += "| Model | Size (MB) |\n"
        md += "|-------|-----------|\n"
        for name, size in self.results.get("model_sizes", {}).items():
            md += f"| {name} | {size:.2f} |\n"

        md += "\n## Per-Demographic Performance\n\n"

        for model_name, report in self.results["models"].items():
            md += f"\n### {model_name}\n\n"

            for group_name in ["fitzpatrick", "domain", "dataset"]:
                key = f"per_{group_name}"
                if key in report:
                    md += f"\n#### {group_name.title()} Breakdown\n\n"
                    md += "| Group | Accuracy | F1 | Sensitivity | Specificity | N |\n"
                    md += "|-------|----------|----|-----------|-----------|-----------|\n"
                    for group, metrics in report[key].items():
                        md += f"| {group} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | {metrics['n']} |\n"

        md += f"\n---\n*Report generated by SkinTag benchmark pipeline*\n"

        return md

    def run(self):
        """Execute full benchmark pipeline."""
        print("="*70)
        print("  SkinTag Model Benchmark Pipeline")
        print("="*70)

        if not self.load_data():
            return False

        self.run_benchmarks()
        self.compare_all_models()

        if self.benchmark_config["run_fairness_analysis"]:
            self.analyze_fairness()

        self.collect_model_sizes()

        if self.benchmark_config["track_history"]:
            self.track_history()

        self.generate_report()

        print("\n" + "="*70)
        print("  Benchmark complete!")
        print("="*70)

        if self.results["regression_alerts"]:
            print(f"\nWARNING: {len(self.results['regression_alerts'])} regression(s) detected!")
            for alert in self.results["regression_alerts"]:
                print(f"  - {alert['model']}: {alert['issue']}")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark pre-trained vs fine-tuned models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config file",
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to benchmark config file",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable historical tracking",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config
    benchmark_config_path = project_root / args.benchmark_config

    runner = BenchmarkRunner(config_path, benchmark_config_path)

    if args.no_history:
        runner.benchmark_config["track_history"] = False

    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
