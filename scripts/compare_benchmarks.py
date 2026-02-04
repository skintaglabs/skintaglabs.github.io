"""Compare two benchmark runs and highlight differences.

Usage:
    python scripts/compare_benchmarks.py \\
        --baseline results/benchmarks/benchmark_2024-01-15.json \\
        --current results/benchmarks/benchmark_2024-01-16.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_benchmark(path: Path):
    """Load benchmark JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_metrics(baseline: dict, current: dict):
    """Compare metrics between two benchmark runs."""
    print("\n" + "="*70)
    print("  METRIC COMPARISON")
    print("="*70)

    baseline_models = baseline["models"]
    current_models = current["models"]

    common_models = set(baseline_models.keys()) & set(current_models.keys())

    if not common_models:
        print("\nNo common models found between runs!")
        return

    print(f"\n{'Model':<25} {'Metric':<15} {'Baseline':<12} {'Current':<12} {'Change':<12}")
    print("-"*70)

    for model_name in sorted(common_models):
        base_metrics = baseline_models[model_name]
        curr_metrics = current_models[model_name]

        metrics_to_compare = [
            ("overall_accuracy", "Accuracy"),
            ("f1_macro", "F1 Macro"),
            ("f1_binary", "F1 Malignant"),
            ("auc", "AUC"),
        ]

        for metric_key, metric_label in metrics_to_compare:
            base_val = base_metrics.get(metric_key)
            curr_val = curr_metrics.get(metric_key)

            if base_val is None or curr_val is None:
                continue

            change = curr_val - base_val
            change_str = f"{change:+.4f}"
            change_pct = (change / base_val * 100) if base_val > 0 else 0

            indicator = ""
            if abs(change) > 0.01:
                if change > 0:
                    indicator = "↑"
                else:
                    indicator = "↓"

            print(f"{model_name:<25} {metric_label:<15} {base_val:<12.4f} {curr_val:<12.4f} {change_str:<8} ({change_pct:+.2f}%) {indicator}")


def compare_fairness(baseline: dict, current: dict):
    """Compare fairness metrics."""
    print("\n" + "="*70)
    print("  FAIRNESS COMPARISON")
    print("="*70)

    baseline_fairness = baseline.get("fairness", {})
    current_fairness = current.get("fairness", {})

    common_models = set(baseline_fairness.keys()) & set(current_fairness.keys())

    if not common_models:
        print("\nNo fairness data found!")
        return

    for model_name in sorted(common_models):
        print(f"\n{model_name}:")
        base_fair = baseline_fairness[model_name]
        curr_fair = current_fairness[model_name]

        for gap_name in base_fair:
            if gap_name not in curr_fair:
                continue

            base_val = base_fair[gap_name]
            curr_val = curr_fair[gap_name]

            if isinstance(base_val, dict) or isinstance(curr_val, dict):
                continue

            change = curr_val - base_val
            indicator = "↑" if change > 0 else ("↓" if change < 0 else "=")

            print(f"  {gap_name}: {base_val:.4f} → {curr_val:.4f} ({change:+.4f}) {indicator}")


def compare_regressions(baseline: dict, current: dict):
    """Compare regression alerts."""
    print("\n" + "="*70)
    print("  REGRESSION ALERTS")
    print("="*70)

    baseline_alerts = baseline.get("regression_alerts", [])
    current_alerts = current.get("regression_alerts", [])

    print(f"\nBaseline: {len(baseline_alerts)} alert(s)")
    print(f"Current:  {len(current_alerts)} alert(s)")

    if current_alerts:
        print("\nCurrent Alerts:")
        for alert in current_alerts:
            print(f"  - {alert['model']}: {alert['issue']}")
            print(f"    Value: {alert['value']:.4f}, Threshold: {alert['threshold']:.4f}")

    # New alerts
    new_alerts = [a for a in current_alerts if a not in baseline_alerts]
    if new_alerts:
        print(f"\nNEW ALERTS ({len(new_alerts)}):")
        for alert in new_alerts:
            print(f"  - {alert['model']}: {alert['issue']}")

    # Resolved alerts
    resolved_alerts = [a for a in baseline_alerts if a not in current_alerts]
    if resolved_alerts:
        print(f"\nRESOLVED ALERTS ({len(resolved_alerts)}):")
        for alert in resolved_alerts:
            print(f"  - {alert['model']}: {alert['issue']}")


def compare_best_models(baseline: dict, current: dict):
    """Compare best model selection."""
    print("\n" + "="*70)
    print("  BEST MODEL COMPARISON")
    print("="*70)

    baseline_best = baseline["comparison"]["best_model"]
    current_best = current["comparison"]["best_model"]

    baseline_metric = baseline["comparison"]["best_metric"]
    current_metric = current["comparison"]["best_metric"]

    print(f"\nBaseline: {baseline_best} (F1 macro: {baseline_metric:.4f})")
    print(f"Current:  {current_best} (F1 macro: {current_metric:.4f})")

    if baseline_best != current_best:
        print("\n⚠️  BEST MODEL CHANGED!")

    metric_change = current_metric - baseline_metric
    if abs(metric_change) > 0.01:
        indicator = "↑ IMPROVED" if metric_change > 0 else "↓ DEGRADED"
        print(f"\nPerformance change: {metric_change:+.4f} {indicator}")


def generate_summary(baseline: dict, current: dict):
    """Generate executive summary of comparison."""
    print("\n" + "="*70)
    print("  EXECUTIVE SUMMARY")
    print("="*70)

    baseline_time = baseline["metadata"]["timestamp"]
    current_time = current["metadata"]["timestamp"]

    print(f"\nBaseline: {baseline_time}")
    print(f"Current:  {current_time}")

    # Count metrics
    baseline_models = baseline["models"]
    current_models = current["models"]

    common_models = set(baseline_models.keys()) & set(current_models.keys())

    improvements = 0
    degradations = 0
    unchanged = 0

    for model_name in common_models:
        base_f1 = baseline_models[model_name].get("f1_macro", 0)
        curr_f1 = current_models[model_name].get("f1_macro", 0)

        change = curr_f1 - base_f1

        if abs(change) < 0.001:
            unchanged += 1
        elif change > 0:
            improvements += 1
        else:
            degradations += 1

    print(f"\nModels compared: {len(common_models)}")
    print(f"  Improved:   {improvements}")
    print(f"  Degraded:   {degradations}")
    print(f"  Unchanged:  {unchanged}")

    # Overall assessment
    current_alerts = len(current.get("regression_alerts", []))
    baseline_alerts = len(baseline.get("regression_alerts", []))

    print(f"\nRegression alerts:")
    print(f"  Baseline: {baseline_alerts}")
    print(f"  Current:  {current_alerts}")

    if improvements > degradations and current_alerts <= baseline_alerts:
        print("\n✅ Overall: IMPROVED")
    elif degradations > improvements or current_alerts > baseline_alerts:
        print("\n⚠️  Overall: DEGRADED")
    else:
        print("\n➡️  Overall: STABLE")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark runs"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline benchmark JSON",
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output markdown file",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}")
        sys.exit(1)

    if not current_path.exists():
        print(f"Current file not found: {current_path}")
        sys.exit(1)

    print("Loading benchmarks...")
    baseline = load_benchmark(baseline_path)
    current = load_benchmark(current_path)

    generate_summary(baseline, current)
    compare_best_models(baseline, current)
    compare_metrics(baseline, current)
    compare_fairness(baseline, current)
    compare_regressions(baseline, current)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
