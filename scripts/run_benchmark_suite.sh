#!/bin/bash
# Comprehensive benchmark suite runner
# Usage: ./scripts/run_benchmark_suite.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  SkinTag Benchmark Suite"
echo "============================================"
echo ""

MODE="${1:-full}"

case "$MODE" in
  quick)
    echo "Running quick benchmark (frozen models only)..."
    python scripts/benchmark_models.py --benchmark-config configs/benchmark_config.yaml
    ;;

  full)
    echo "Running full benchmark suite..."
    python scripts/benchmark_models.py
    ;;

  compare)
    if [ -z "$2" ] || [ -z "$3" ]; then
      echo "Usage: $0 compare <baseline.json> <current.json>"
      exit 1
    fi
    python scripts/compare_benchmarks.py --baseline "$2" --current "$3"
    ;;

  history)
    echo "Analyzing benchmark history..."
    python scripts/analyze_benchmark_history.py
    ;;

  plot)
    echo "Generating performance plots..."
    python scripts/analyze_benchmark_history.py --plot
    ;;

  latest)
    echo "Showing latest benchmark report..."
    if [ -f "results/benchmarks/benchmark_latest.md" ]; then
      cat "results/benchmarks/benchmark_latest.md"
    else
      echo "No benchmark results found. Run 'full' first."
      exit 1
    fi
    ;;

  clean)
    echo "Cleaning old benchmark results (keeping last 10)..."
    cd results/benchmarks
    ls -t benchmark_*.json | tail -n +11 | xargs -r rm
    ls -t benchmark_*.md | tail -n +11 | xargs -r rm
    echo "Cleanup complete."
    ;;

  *)
    echo "Usage: $0 {quick|full|compare|history|plot|latest|clean}"
    echo ""
    echo "Commands:"
    echo "  quick    - Run benchmark with frozen models only (fast)"
    echo "  full     - Run complete benchmark suite (default)"
    echo "  compare  - Compare two benchmark runs"
    echo "  history  - Analyze performance trends"
    echo "  plot     - Generate performance plots"
    echo "  latest   - Show most recent benchmark report"
    echo "  clean    - Remove old benchmark files (keep 10 most recent)"
    echo ""
    echo "Examples:"
    echo "  $0 full"
    echo "  $0 compare results/benchmarks/benchmark_2024-01-15.json results/benchmarks/benchmark_2024-01-16.json"
    echo "  $0 history"
    exit 1
    ;;
esac

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
