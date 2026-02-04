"""Analyze benchmark history and trends over time.

Usage:
    python scripts/analyze_benchmark_history.py
    python scripts/analyze_benchmark_history.py --plot
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_history(history_path: Path):
    """Load benchmark history file."""
    if not history_path.exists():
        print(f"No history file found at {history_path}")
        print("Run benchmarks first to generate history.")
        sys.exit(1)

    with open(history_path) as f:
        return json.load(f)


def parse_timestamp(ts_str: str):
    """Parse timestamp string to datetime."""
    return datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")


def analyze_trends(history: dict):
    """Analyze performance trends over time."""
    print("\n" + "="*70)
    print("  PERFORMANCE TRENDS")
    print("="*70)

    runs = history["runs"]

    if len(runs) < 2:
        print("\nInsufficient data for trend analysis (need at least 2 runs)")
        return

    print(f"\nTotal runs: {len(runs)}")
    print(f"First run: {runs[0]['timestamp']}")
    print(f"Latest run: {runs[-1]['timestamp']}")

    # Best model evolution
    print("\n" + "-"*70)
    print("Best Model Evolution")
    print("-"*70)

    best_models = [run["best_model"] for run in runs]
    best_metrics = [run["best_f1_macro"] for run in runs]

    for i, (model, metric, timestamp) in enumerate(zip(best_models, best_metrics, [r["timestamp"] for r in runs])):
        change = ""
        if i > 0:
            prev_metric = best_metrics[i-1]
            delta = metric - prev_metric
            change = f" ({delta:+.4f})"

        print(f"{timestamp}: {model} = {metric:.4f}{change}")

    # Model performance over time
    print("\n" + "-"*70)
    print("Model Performance Over Time")
    print("-"*70)

    all_models = set()
    for run in runs:
        all_models.update(run["model_metrics"].keys())

    for model_name in sorted(all_models):
        print(f"\n{model_name}:")

        f1_scores = []
        timestamps = []

        for run in runs:
            if model_name in run["model_metrics"]:
                f1_scores.append(run["model_metrics"][model_name]["f1_macro"])
                timestamps.append(run["timestamp"])

        if not f1_scores:
            continue

        for i, (ts, f1) in enumerate(zip(timestamps, f1_scores)):
            change = ""
            if i > 0:
                delta = f1 - f1_scores[i-1]
                if abs(delta) > 0.001:
                    change = f" ({delta:+.4f})"

            print(f"  {ts}: {f1:.4f}{change}")

        # Calculate trend
        if len(f1_scores) >= 3:
            first_3_avg = sum(f1_scores[:3]) / 3
            last_3_avg = sum(f1_scores[-3:]) / 3
            trend = last_3_avg - first_3_avg

            if abs(trend) > 0.01:
                direction = "↑ IMPROVING" if trend > 0 else "↓ DEGRADING"
                print(f"  Trend: {direction} ({trend:+.4f})")


def analyze_regressions(history: dict):
    """Analyze regression patterns."""
    print("\n" + "="*70)
    print("  REGRESSION ANALYSIS")
    print("="*70)

    runs = history["runs"]

    total_alerts = sum(len(run["regression_alerts"]) for run in runs)
    runs_with_alerts = sum(1 for run in runs if run["regression_alerts"])

    print(f"\nTotal regression alerts: {total_alerts}")
    print(f"Runs with alerts: {runs_with_alerts}/{len(runs)}")

    # Most common issues
    issue_counts = {}
    model_alert_counts = {}

    for run in runs:
        for alert in run["regression_alerts"]:
            issue = alert["issue"]
            model = alert["model"]

            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            model_alert_counts[model] = model_alert_counts.get(model, 0) + 1

    if issue_counts:
        print("\nMost common issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count}")

        print("\nModels with most alerts:")
        for model, count in sorted(model_alert_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {model}: {count}")


def analyze_stability(history: dict):
    """Analyze model stability."""
    print("\n" + "="*70)
    print("  STABILITY ANALYSIS")
    print("="*70)

    runs = history["runs"]

    if len(runs) < 3:
        print("\nInsufficient data for stability analysis (need at least 3 runs)")
        return

    all_models = set()
    for run in runs:
        all_models.update(run["model_metrics"].keys())

    print(f"\n{'Model':<25} {'Mean F1':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Stability':<12}")
    print("-"*70)

    for model_name in sorted(all_models):
        f1_scores = []

        for run in runs:
            if model_name in run["model_metrics"]:
                f1_scores.append(run["model_metrics"][model_name]["f1_macro"])

        if len(f1_scores) < 2:
            continue

        import statistics

        mean_f1 = statistics.mean(f1_scores)
        std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0
        min_f1 = min(f1_scores)
        max_f1 = max(f1_scores)

        stability = "High" if std_f1 < 0.01 else ("Medium" if std_f1 < 0.02 else "Low")

        print(f"{model_name:<25} {mean_f1:<12.4f} {std_f1:<12.4f} {min_f1:<12.4f} {max_f1:<12.4f} {stability:<12}")


def generate_recommendations(history: dict):
    """Generate recommendations based on historical data."""
    print("\n" + "="*70)
    print("  RECOMMENDATIONS")
    print("="*70)

    runs = history["runs"]

    if len(runs) < 2:
        print("\nInsufficient data for recommendations")
        return

    recommendations = []

    # Check if best model has changed recently
    recent_runs = runs[-5:] if len(runs) >= 5 else runs
    best_models = [run["best_model"] for run in recent_runs]

    if len(set(best_models)) > 1:
        recommendations.append(
            "⚠️  Best model has changed recently. Consider re-evaluating model selection criteria."
        )

    # Check for consistent degradation
    latest_run = runs[-1]
    if len(runs) >= 3:
        prev_3_metrics = [runs[i]["best_f1_macro"] for i in range(-4, -1)]
        if all(latest_run["best_f1_macro"] < m for m in prev_3_metrics):
            recommendations.append(
                "⚠️  Performance has degraded over last 3 runs. Investigate training pipeline or data quality."
            )

    # Check for frequent regressions
    recent_alerts = sum(len(run["regression_alerts"]) for run in recent_runs)
    if recent_alerts > len(recent_runs) * 2:
        recommendations.append(
            "⚠️  High frequency of regression alerts. Consider tightening quality controls or adjusting thresholds."
        )

    # Success indicators
    if len(runs) >= 3:
        first_metric = runs[0]["best_f1_macro"]
        last_metric = runs[-1]["best_f1_macro"]
        improvement = last_metric - first_metric

        if improvement > 0.02:
            recommendations.append(
                f"✅ Overall improvement of {improvement:.4f} since first run. Good progress!"
            )

    if latest_run["regression_alerts"] == []:
        recommendations.append(
            "✅ No regression alerts in latest run. All models meeting performance thresholds."
        )

    if recommendations:
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\nNo specific recommendations at this time.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark history and trends"
    )
    parser.add_argument(
        "--history",
        type=str,
        default="results/benchmarks/benchmark_history.json",
        help="Path to history file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    history_path = project_root / args.history

    print("="*70)
    print("  BENCHMARK HISTORY ANALYSIS")
    print("="*70)

    history = load_history(history_path)

    analyze_trends(history)
    analyze_regressions(history)
    analyze_stability(history)
    generate_recommendations(history)

    print("\n" + "="*70)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            runs = history["runs"]
            timestamps = [r["timestamp"] for r in runs]
            best_metrics = [r["best_f1_macro"] for r in runs]

            plt.figure(figsize=(12, 6))
            plt.plot(range(len(timestamps)), best_metrics, marker='o')
            plt.xlabel("Run Number")
            plt.ylabel("Best F1 Macro")
            plt.title("Best Model Performance Over Time")
            plt.grid(True, alpha=0.3)
            plt.xticks(range(len(timestamps)), [ts.split('_')[0] for ts in timestamps], rotation=45)
            plt.tight_layout()

            plot_path = project_root / "results" / "benchmarks" / "performance_history.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\nPlot saved to {plot_path}")

        except ImportError:
            print("\nMatplotlib not available. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
