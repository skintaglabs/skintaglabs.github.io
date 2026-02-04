# Model Benchmarking Pipeline

Comprehensive benchmarking system for comparing pre-trained vs fine-tuned SigLIP models across multiple performance dimensions.

## Quick Start

```bash
# Run full benchmark suite
python scripts/benchmark_models.py

# View latest results
cat results/benchmarks/benchmark_latest.md

# Analyze historical trends
python scripts/analyze_benchmark_history.py
```

## What Gets Benchmarked

### Models Compared

1. **Baselines**
   - Majority class predictor
   - Random weighted predictor

2. **Frozen SigLIP Embeddings** (Pre-trained, no fine-tuning)
   - Logistic regression
   - XGBoost classifier
   - Deep MLP

3. **Fine-tuned SigLIP Embeddings** (Last 4 layers unfrozen)
   - XGBoost classifier
   - Deep MLP

4. **End-to-End** (Optional)
   - Full fine-tuned SigLIP model

### Performance Dimensions

#### 1. Overall Performance
- Accuracy
- Balanced accuracy
- F1 macro (primary ranking metric for medical AI)
- F1 malignant (binary classification)
- AUC-ROC

#### 2. Fairness Metrics
- **Per-Fitzpatrick skin type** (I-VI)
  - Accuracy, sensitivity, specificity per type
  - Maximum gap between types
  - Equalized odds gaps

- **Per-domain**
  - Dermoscopic vs clinical vs smartphone

- **Per-dataset**
  - HAM10000, DDI, Fitzpatrick17k, PAD-UFES-20, BCN20000

#### 3. Efficiency Metrics
- Inference time (ms per sample)
- Model size (MB)
- Training time (for reference)

#### 4. Robustness (Optional)
- Performance under image distortions
- Cross-domain generalization

## Configuration

### Benchmark Configuration

Edit `configs/benchmark_config.yaml`:

```yaml
models_to_benchmark:
  - baseline_majority
  - xgboost_frozen
  - xgboost_finetuned

run_fairness_analysis: true
run_robustness_tests: false
track_history: true

performance_thresholds:
  min_accuracy: 0.85
  min_f1_macro: 0.80
  max_fairness_gap: 0.10
```

### Performance Thresholds

Regression alerts are triggered when models fall below:
- **Accuracy**: 0.85
- **F1 Macro**: 0.80
- **AUC**: 0.90
- **Fairness Gap**: 0.10 (10% max difference between groups)
- **Equalized Odds Gap**: 0.15

## Usage Examples

### Basic Benchmark

```bash
python scripts/benchmark_models.py
```

Output:
- `results/benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS.md` (markdown report)
- `results/benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS.json` (structured data)
- `results/benchmarks/benchmark_latest.md` (symlink to latest)

### Custom Configuration

```bash
python scripts/benchmark_models.py \
  --config configs/config.yaml \
  --benchmark-config configs/benchmark_config.yaml
```

### Disable Historical Tracking

```bash
python scripts/benchmark_models.py --no-history
```

### Compare Two Runs

```bash
python scripts/compare_benchmarks.py \
  --baseline results/benchmarks/benchmark_2024-01-15_10-30-00.json \
  --current results/benchmarks/benchmark_2024-01-16_14-20-00.json
```

### Analyze Trends

```bash
# View trends across all runs
python scripts/analyze_benchmark_history.py

# Generate performance plot
python scripts/analyze_benchmark_history.py --plot
```

## Interpreting Results

### Markdown Report Structure

```markdown
# Model Benchmark Report

## Executive Summary
- Best model and primary metric
- Number of models evaluated
- Regression alerts

## Performance Comparison
Table of all models with key metrics

## Fairness Analysis
Per-demographic performance breakdowns

## Regression Alerts
Models falling below thresholds

## Model Sizes
Disk space requirements

## Per-Demographic Performance
Detailed breakdowns by Fitzpatrick type, domain, dataset
```

### JSON Structure

```json
{
  "metadata": {
    "timestamp": "2024-01-16_14-20-00",
    "device": "cuda"
  },
  "models": {
    "xgboost_frozen": {
      "overall_accuracy": 0.9520,
      "f1_macro": 0.9380,
      "embedding_type": "frozen",
      "per_fitzpatrick": {...},
      "fairness_gap": 0.045
    }
  },
  "comparison": {
    "best_model": "xgboost_finetuned",
    "best_metric": 0.9510
  },
  "regression_alerts": []
}
```

## Primary Ranking Metric: F1 Macro

**Why F1 Macro over Accuracy?**

For medical AI systems, especially in dermatology:

1. **Class Imbalance**: Most lesions are benign (~80%), so accuracy can be misleadingly high
2. **Equal Importance**: Missing malignant cases (false negatives) is as critical as over-diagnosing (false positives)
3. **Fairness**: F1 macro ensures good performance on minority class (malignant)

## Fairness Requirements

For clinical deployment, the benchmark enforces:

### Critical Thresholds
- **Sensitivity Gap** < 5% across Fitzpatrick types
- **Specificity Gap** < 10% across Fitzpatrick types
- **F1 Gap** < 10% across domains

### Why This Matters
Medical AI systems that perform poorly on darker skin tones (Fitzpatrick IV-VI) perpetuate healthcare inequities. The benchmark explicitly tracks and alerts on fairness gaps.

## Regression Detection

The pipeline automatically detects:

1. **Performance Regressions**
   - Best F1 macro drops > 1% from previous run
   - Any model falls below threshold

2. **Fairness Regressions**
   - Demographic gaps increase
   - Equalized odds violations

3. **Model Selection Changes**
   - Best model ranking changes
   - Alerts for investigation

## Historical Tracking

Performance metrics are tracked in `results/benchmarks/benchmark_history.json`:

```json
{
  "runs": [
    {
      "timestamp": "2024-01-15_10-30-00",
      "best_model": "xgboost_frozen",
      "best_f1_macro": 0.9380,
      "model_metrics": {...},
      "regression_alerts": []
    }
  ]
}
```

Use `analyze_benchmark_history.py` to:
- View trends over time
- Identify stability issues
- Track model evolution
- Generate performance plots

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Model Benchmark

on:
  pull_request:
    paths:
      - 'src/model/**'
      - 'scripts/train*.py'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run benchmark
        run: python scripts/benchmark_models.py
      - name: Check for regressions
        run: |
          python scripts/analyze_benchmark_history.py
          # Fail if regressions detected
          if [ $(jq '.regression_alerts | length' results/benchmarks/benchmark_*.json | tail -1) -gt 0 ]; then
            echo "Regression detected!"
            exit 1
          fi
```

## Pre-trained vs Fine-tuned Comparison

### Expected Results

**Frozen SigLIP + XGBoost** (Pre-trained)
- Accuracy: ~95%
- F1 Macro: ~93%
- Size: <10 MB
- Inference: Fast
- Use case: Quick deployment, resource-constrained

**Fine-tuned SigLIP + XGBoost**
- Accuracy: ~96-97%
- F1 Macro: ~94-95%
- Size: 3.3 GB (embeddings can be cached)
- Inference: Moderate
- Use case: Best accuracy, worth the size

**Trade-off Analysis**
- Fine-tuning provides ~1-2% absolute improvement
- Critical for fairness (reduces Fitzpatrick gaps)
- Justifies 300x size increase for production medical AI

## Troubleshooting

### No Embeddings Found

```
Error: No cached embeddings found. Run train.py first.
```

**Solution:**
```bash
python scripts/train.py
```

### Missing Fine-tuned Embeddings

```
Warning: Fine-tuned embeddings not found (skipping fine-tuned benchmarks)
```

**Solution:**
```bash
python scripts/comprehensive_evaluation.py  # Generates fine-tuned embeddings
```

### Regression Alerts

Check `results/benchmarks/benchmark_latest.md` for details. Common causes:
- Data distribution shift
- Training instability
- Hyperparameter changes

Compare with previous run:
```bash
python scripts/compare_benchmarks.py --baseline <prev> --current <latest>
```

## Best Practices

1. **Run benchmarks after every model change**
2. **Track history** to detect gradual degradation
3. **Focus on F1 macro** over raw accuracy
4. **Monitor fairness gaps** as primary concern
5. **Compare frozen vs fine-tuned** to justify complexity

## References

- [Equalized Odds for Fair Classification](https://arxiv.org/abs/1610.02413)
- [Fitzpatrick Scale in Medical AI](https://www.nature.com/articles/s41591-021-01592-w)
- [Medical AI Benchmarking Guidelines](https://www.fda.gov/medical-devices/software-medical-device-samd)
