#!/usr/bin/env python3
"""SkinTag: Unified pipeline — data check, embed, train, evaluate, serve.

Run this single file to execute the full pipeline from data through deployment.
Each stage is wrapped in error handling so failures are logged as warnings and
the pipeline continues. Internet is only required on first run (to download the
SigLIP model from HuggingFace); all subsequent runs use cached artifacts.

Usage:
    python run_pipeline.py                   # Full pipeline (multi-dataset, all models)
    python run_pipeline.py --quick           # Quick smoke test (500 samples, logistic only)
    python run_pipeline.py --skip-train      # Skip training, run eval + app on existing models
    python run_pipeline.py --no-app          # Everything except launching the web app
    python run_pipeline.py --app-only        # Just launch the web app
"""

import sys
import os
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_warnings = []
_stage_times = {}


def _banner(msg):
    width = 70
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}\n")


def _warn(stage, msg, exc=None):
    entry = f"[WARNING] {stage}: {msg}"
    if exc:
        entry += f"\n  -> {type(exc).__name__}: {exc}"
    _warnings.append(entry)
    print(entry, file=sys.stderr)


def _run_stage(name, fn, *args, **kwargs):
    """Run a pipeline stage with timing and error capture."""
    _banner(name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        _stage_times[name] = elapsed
        print(f"\n  [{name}] completed in {elapsed:.1f}s")
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        _stage_times[name] = elapsed
        _warn(name, f"Stage failed after {elapsed:.1f}s", exc)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Stage 1: Verify environment & data
# ---------------------------------------------------------------------------

def stage_check_environment():
    """Verify Python packages and data directories."""
    import importlib

    required = [
        "torch", "torchvision", "transformers", "PIL", "numpy",
        "sklearn", "pandas", "yaml", "tqdm",
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  Missing packages: {missing}")
        print(f"  Run: pip install -r requirements.txt")
        raise ImportError(f"Missing required packages: {missing}")
    print("  All required packages available.")

    # Check data
    data_dir = PROJECT_ROOT / "data"
    datasets_found = []
    datasets_missing = []

    checks = {
        "HAM10000": data_dir / "HAM10000_metadata.csv",
        "DDI": data_dir / "ddi" / "ddi_metadata.csv",
        "Fitzpatrick17k": data_dir / "fitzpatrick17k" / "fitzpatrick17k.csv",
        "PAD-UFES-20": data_dir / "pad_ufes" / "metadata.csv",
        "BCN20000": data_dir / "bcn20000" / "bcn20000_metadata.csv",
    }
    for name, path in checks.items():
        if path.exists():
            datasets_found.append(name)
            print(f"  [OK] {name}: {path}")
        else:
            datasets_missing.append(name)
            print(f"  [--] {name}: NOT FOUND ({path})")

    if not datasets_found:
        raise FileNotFoundError("No datasets found. See PLAN.md for download instructions.")

    # Check SigLIP model cache
    import torch
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cached = any(cache_home.rglob("*siglip*")) if cache_home.exists() else False
    if model_cached:
        print(f"  [OK] SigLIP model cached at {cache_home}")
    else:
        print(f"  [!!] SigLIP model NOT cached — will download ~1.6GB on first run")
        print(f"       Internet connection required for first embedding extraction.")

    return {
        "datasets_found": datasets_found,
        "datasets_missing": datasets_missing,
        "model_cached": model_cached,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# ---------------------------------------------------------------------------
# Stage 2: Load data
# ---------------------------------------------------------------------------

def stage_load_data(sample_n=0):
    """Load multi-dataset metadata and return (image_paths, labels, metadata).

    Images are NOT loaded into RAM here — only file paths are collected.
    Actual image I/O happens lazily during embedding extraction.
    """
    import yaml
    import numpy as np
    from src.data.loader import load_multi_dataset
    from src.data.schema import samples_to_arrays

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / "data"
    datasets = config.get("data", {}).get("datasets", None)
    dataset_options = config.get("data", {}).get("dataset_options", {})

    samples = load_multi_dataset(data_dir, datasets=datasets, dataset_options=dataset_options)
    if not samples:
        raise RuntimeError("No samples loaded from any dataset.")

    # Returns paths (not PIL images) since loaders now use lazy loading
    image_paths, labels, metadata = samples_to_arrays(samples)
    print(f"  Total: {len(image_paths)} images (paths only — no RAM usage)")
    print(f"  Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    if "domain" in metadata.columns:
        print(f"  Domains: {metadata['domain'].value_counts().to_dict()}")
    if "dataset" in metadata.columns:
        print(f"  Datasets: {metadata['dataset'].value_counts().to_dict()}")

    # Sub-sample for quick mode
    if sample_n > 0 and sample_n < len(image_paths):
        np.random.seed(42)
        idx = np.random.choice(len(image_paths), sample_n, replace=False)
        image_paths = [image_paths[i] for i in idx]
        labels = labels[idx]
        metadata = metadata.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {len(image_paths)} images (--quick mode)")

    # Save metadata for downstream stages
    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(cache_dir / "metadata.csv", index=False)

    return image_paths, labels, metadata


# ---------------------------------------------------------------------------
# Stage 3: Extract embeddings
# ---------------------------------------------------------------------------

def stage_extract_embeddings(image_paths):
    """Extract SigLIP embeddings (cached to disk).

    Accepts file paths — images are loaded per-batch during extraction,
    so only a few images are in RAM at any time.
    """
    import yaml
    import torch
    from src.model.embeddings import EmbeddingExtractor

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config["extraction"]["batch_size_gpu"] if device == "cuda" else config["extraction"]["batch_size_cpu"]

    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "embeddings.pt"

    print(f"  Device: {device}, Batch size: {batch_size}")
    print(f"  Cache: {cache_path}")
    print(f"  Images: {len(image_paths)} (streaming from disk)")

    extractor = EmbeddingExtractor(device=device)
    embeddings = extractor.extract_dataset(image_paths, batch_size=batch_size, cache_path=cache_path)
    extractor.unload_model()  # free GPU/RAM

    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Stage 4: Train all models
# ---------------------------------------------------------------------------

def stage_train_models(embeddings, labels, metadata):
    """Train baseline, logistic, and deep models. Returns results dict."""
    import yaml
    import pickle
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    from src.model.classifier import SklearnClassifier
    from src.model.baseline import MajorityClassBaseline
    from src.model.deep_classifier import DeepClassifier
    from src.data.sampler import (
        compute_combined_balanced_weights,
        compute_domain_balanced_weights,
        compute_stratified_split_key,
    )

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_dir = PROJECT_ROOT / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed = config["training"]["seed"]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    emb_np = embeddings.numpy() if hasattr(embeddings, "numpy") else np.asarray(embeddings)

    # Stratified split on (label, domain)
    if "domain" in metadata.columns:
        stratify_key = compute_stratified_split_key(labels, metadata["domain"].values)
        unique, counts = np.unique(stratify_key, return_counts=True)
        if counts.min() < 2:
            print("  Warning: small (label,domain) groups — falling back to label-only stratification")
            stratify_key = labels
    else:
        stratify_key = labels

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        emb_np, labels, metadata,
        test_size=0.2, random_state=seed, stratify=stratify_key,
    )
    meta_test.to_csv(cache_dir / "test_metadata.csv", index=False)
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # Domain + Fitzpatrick balanced weights
    sample_weights = None
    if "domain" in meta_train.columns:
        train_domains = meta_train["domain"].values
        if "fitzpatrick" in meta_train.columns:
            sample_weights = compute_combined_balanced_weights(
                train_domains, meta_train["fitzpatrick"].values, y_train
            )
            print(f"  Combined domain+Fitzpatrick balanced weights applied")
        else:
            sample_weights = compute_domain_balanced_weights(train_domains, y_train)
            print(f"  Domain-balanced weights applied")

    # Train each model type
    results = {}
    model_specs = [
        ("baseline", lambda: MajorityClassBaseline()),
        ("logistic", lambda: SklearnClassifier(classifier_type="logistic")),
        ("deep", lambda: DeepClassifier(embedding_dim=emb_np.shape[1], device=device)),
    ]

    for model_type, make_clf in model_specs:
        print(f"\n  --- Training {model_type} ---")
        try:
            clf = make_clf()
            if model_type == "baseline":
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, sample_weight=sample_weights)

            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            train_acc = float(np.mean(y_pred_train == y_train))
            test_acc = float(np.mean(y_pred_test == y_test))
            test_f1 = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
            train_f1 = float(f1_score(y_train, y_pred_train, average="macro", zero_division=0))
            test_f1_bin = float(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))

            print(f"    Train acc={train_acc:.3f}  F1={train_f1:.3f}")
            print(f"    Test  acc={test_acc:.3f}  F1={test_f1:.3f}  F1(malignant)={test_f1_bin:.3f}")

            results[model_type] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_f1_macro": train_f1,
                "test_f1_macro": test_f1,
                "test_f1_malignant": test_f1_bin,
            }

            # Save model
            model_path = cache_dir / f"classifier_{model_type}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)
            print(f"    Saved: {model_path}")

            # Default classifier
            default_type = config["training"].get("classifier", "logistic")
            if model_type == default_type or model_type == "logistic":
                with open(cache_dir / "classifier.pkl", "wb") as f:
                    pickle.dump(clf, f)

        except Exception as exc:
            _warn(f"Train/{model_type}", f"Model training failed", exc)
            traceback.print_exc()

    # Save results JSON
    with open(cache_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n  {'Model':<12} {'Train Acc':>10} {'Test Acc':>10} {'F1 Macro':>10} {'F1 Malig':>10}")
    print(f"  {'-'*54}")
    for name, r in results.items():
        print(f"  {name:<12} {r['train_accuracy']:>10.3f} {r['test_accuracy']:>10.3f} "
              f"{r['test_f1_macro']:>10.3f} {r['test_f1_malignant']:>10.3f}")

    # ------------------------------------------------------------------
    # Condition classification (10-class) — if enabled and labels exist
    # ------------------------------------------------------------------
    train_condition = config.get("training", {}).get("condition_classifier", False)
    has_condition_labels = "condition_label" in metadata.columns and metadata["condition_label"].notna().sum() > 0

    if train_condition and has_condition_labels:
        print(f"\n\n  === Condition Classification (10-class) ===")

        # Extract condition labels from the already-split metadata DataFrames
        cond_train = meta_train["condition_label"].values.astype(float)
        cond_test = meta_test["condition_label"].values.astype(float)

        # Filter out NaN condition labels
        train_mask = ~np.isnan(cond_train)
        test_mask = ~np.isnan(cond_test)

        if train_mask.sum() > 100 and test_mask.sum() > 10:
            X_cond_train = X_train[train_mask]
            y_cond_train = cond_train[train_mask].astype(int)
            X_cond_test = X_test[test_mask]
            y_cond_test = cond_test[test_mask].astype(int)

            n_classes = len(np.unique(y_cond_train))
            print(f"  Condition train: {len(y_cond_train)}, test: {len(y_cond_test)}, classes: {n_classes}")

            cond_results = {}
            cond_model_specs = [
                ("logistic", lambda: SklearnClassifier(classifier_type="logistic")),
                ("deep", lambda: DeepClassifier(
                    embedding_dim=emb_np.shape[1], n_classes=n_classes, device=device
                )),
            ]

            for model_type, make_clf in cond_model_specs:
                print(f"\n  --- Training condition/{model_type} ---")
                try:
                    clf = make_clf()
                    clf.fit(X_cond_train, y_cond_train, sample_weight=sample_weights[train_mask] if sample_weights is not None else None)

                    y_pred_test = clf.predict(X_cond_test)
                    test_acc = float(np.mean(y_pred_test == y_cond_test))
                    test_f1 = float(f1_score(y_cond_test, y_pred_test, average="macro", zero_division=0))

                    print(f"    Test acc={test_acc:.3f}  F1 macro={test_f1:.3f}")

                    cond_results[model_type] = {
                        "test_accuracy": test_acc,
                        "test_f1_macro": test_f1,
                    }

                    # Save model
                    model_path = cache_dir / f"classifier_condition_{model_type}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(clf, f)
                    print(f"    Saved: {model_path}")

                    # Default condition classifier (prefer logistic)
                    if model_type == "logistic":
                        with open(cache_dir / "classifier_condition.pkl", "wb") as f:
                            pickle.dump(clf, f)

                except Exception as exc:
                    _warn(f"Train/condition_{model_type}", f"Condition model training failed", exc)
                    traceback.print_exc()

            # Save condition results
            with open(cache_dir / "condition_training_results.json", "w") as f:
                json.dump(cond_results, f, indent=2)

            # Condition summary table
            print(f"\n  {'Cond Model':<12} {'Test Acc':>10} {'F1 Macro':>10}")
            print(f"  {'-'*34}")
            for name, r in cond_results.items():
                print(f"  {name:<12} {r['test_accuracy']:>10.3f} {r['test_f1_macro']:>10.3f}")
        else:
            print(f"  Skipping condition training: insufficient labeled data "
                  f"(train={train_mask.sum()}, test={test_mask.sum()})")

    return results


# ---------------------------------------------------------------------------
# Stage 5: Evaluate with fairness metrics
# ---------------------------------------------------------------------------

def stage_evaluate():
    """Run fairness evaluation on all trained models."""
    import yaml
    import json
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split

    from src.evaluation.metrics import robustness_report, compare_models
    from src.data.loader import get_demographic_groups
    from src.model.triage import TriageSystem

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_dir = PROJECT_ROOT / "results" / "cache"

    # Check prerequisites
    meta_path = cache_dir / "metadata.csv"
    emb_path = cache_dir / "embeddings.pt"
    if not meta_path.exists() or not emb_path.exists():
        raise FileNotFoundError("Run training first — metadata.csv and embeddings.pt required.")

    all_meta = pd.read_csv(meta_path)
    embeddings = torch.load(emb_path, weights_only=True)
    seed = config["training"]["seed"]

    # Verify embeddings match metadata
    if len(embeddings) != len(all_meta):
        raise RuntimeError(
            f"Embedding/metadata mismatch: {len(embeddings)} embeddings vs {len(all_meta)} metadata rows. "
            f"Re-run training (without --skip-train) to regenerate matching artifacts."
        )

    # Reconstruct test split
    labels_all = all_meta["label"].values if "label" in all_meta.columns else None
    if labels_all is None:
        from src.data.loader import BINARY_MAPPING
        labels_all = np.array([BINARY_MAPPING.get(dx, 0) for dx in all_meta["dx"]])

    indices = np.arange(len(all_meta))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels_all)

    X_test = embeddings[test_idx].numpy()
    y_test = labels_all[test_idx]
    test_meta = all_meta.iloc[test_idx].reset_index(drop=True)
    groups = get_demographic_groups(test_meta)

    triage = TriageSystem(config.get("triage", {}))

    # Evaluate each trained model
    model_names = ["baseline", "logistic", "deep"]
    all_results = {}

    for model_name in model_names:
        model_path = cache_dir / f"classifier_{model_name}.pkl"
        if not model_path.exists():
            print(f"  [--] {model_name}: not found, skipping")
            continue

        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        report = robustness_report(
            y_test, y_pred, groups=groups,
            class_names=["benign", "malignant"], y_proba=y_proba,
        )

        print(f"\n  {model_name}: acc={report['overall_accuracy']:.3f}  "
              f"bal_acc={report['balanced_accuracy']:.3f}  "
              f"F1={report['f1_macro']:.3f}  "
              f"AUC={report.get('auc', float('nan')):.3f}")

        # Fairness summary
        for axis in ["fitzpatrick", "domain"]:
            gap_key = f"{axis}_equalized_odds"
            if gap_key in report:
                eq = report[gap_key]
                print(f"    {axis} eq. odds gap: sens={eq['sensitivity_gap']:.3f} spec={eq['specificity_gap']:.3f}")

        # Triage distribution
        if y_proba is not None:
            mal_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            triage_results = triage.assess_batch(mal_proba)
            tiers = [t.urgency_tier for t in triage_results]
            tier_counts = {t: tiers.count(t) for t in ["low", "moderate", "high"]}
            print(f"    Triage: {tier_counts}")

        all_results[model_name] = report

    # Best model
    if len(all_results) > 1:
        summary = {
            n: {"f1_macro": r["f1_macro"], "auc": r.get("auc", 0)}
            for n, r in all_results.items()
        }
        best = compare_models(summary)
        print(f"\n  Best model: {best['best_model']} (F1 macro={best['best_metric']:.3f})")

    # ------------------------------------------------------------------
    # Condition evaluation (10-class)
    # ------------------------------------------------------------------
    cond_classifier_path = cache_dir / "classifier_condition.pkl"
    if cond_classifier_path.exists() and "condition_label" in test_meta.columns:
        print(f"\n\n  === Condition Evaluation (10-class) ===")
        try:
            with open(cond_classifier_path, "rb") as f:
                cond_clf = pickle.load(f)

            cond_labels = test_meta["condition_label"].values.astype(float)
            cond_mask = ~np.isnan(cond_labels)

            if cond_mask.sum() > 10:
                X_cond = X_test[cond_mask]
                y_cond = cond_labels[cond_mask].astype(int)
                y_cond_pred = cond_clf.predict(X_cond)

                from src.evaluation.metrics import condition_classification_report
                from src.data.taxonomy import CONDITION_NAMES, Condition

                cond_report = condition_classification_report(y_cond, y_cond_pred)
                all_results["condition"] = cond_report

                print(f"  Condition accuracy: {cond_report['accuracy']:.3f}")
                print(f"  Condition F1 macro: {cond_report['f1_macro']:.3f}")
                print(f"  Per-condition F1:")
                for cid, metrics in cond_report.get("per_condition", {}).items():
                    cname = CONDITION_NAMES.get(Condition(int(cid)), f"Class {cid}")
                    print(f"    {cname:<30} F1={metrics['f1']:.3f}  n={metrics['n']}")
        except Exception as exc:
            _warn("Evaluate/condition", "Condition evaluation failed", exc)
            traceback.print_exc()

    # Save
    def _convert(obj):
        import numpy as _np
        if isinstance(obj, (_np.integer,)): return int(obj)
        if isinstance(obj, (_np.floating,)): return float(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=_convert))
    out_path = cache_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Stage 6: Launch web app
# ---------------------------------------------------------------------------

def stage_launch_app():
    """Launch the FastAPI web app."""
    import uvicorn

    print("  Starting SkinTag web app on http://localhost:8000")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SkinTag: Unified pipeline — data, train, evaluate, serve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (500 samples, faster)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing models")
    parser.add_argument("--no-app", action="store_true",
                        help="Run everything except the web app")
    parser.add_argument("--app-only", action="store_true",
                        help="Only launch the web app")
    args = parser.parse_args()

    _banner("SkinTag Pipeline")
    print("  Run with --quick for a fast smoke test (500 samples)")
    print("  Run with --no-app to skip the web app at the end")
    print("  Run with --app-only to just launch the web app")
    print()
    t_start = time.time()

    # App-only shortcut
    if args.app_only:
        stage_launch_app()
        return

    # Stage 1: Environment check
    env = _run_stage("1. Check Environment", stage_check_environment)
    if env is None:
        print("\nEnvironment check failed. Fix issues above and re-run.")
        return

    image_paths, labels, metadata, embeddings = None, None, None, None

    if not args.skip_train:
        # Stage 2: Load data (metadata + paths only, no image I/O)
        sample_n = 500 if args.quick else 0
        result = _run_stage("2. Load Data", stage_load_data, sample_n)
        if result is not None:
            image_paths, labels, metadata = result
        else:
            print("\nData loading failed. Check dataset paths in PLAN.md.")
            _print_summary(t_start)
            return

        # Stage 3: Extract embeddings (images loaded per-batch from paths)
        embeddings = _run_stage("3. Extract Embeddings", stage_extract_embeddings, image_paths)
        if embeddings is None:
            print("\nEmbedding extraction failed. Check SigLIP model / internet connection.")
            _print_summary(t_start)
            return

        # Stage 4: Train models
        _run_stage("4. Train Models", stage_train_models, embeddings, labels, metadata)

    # Stage 5: Evaluate
    _run_stage("5. Evaluate (Fairness)", stage_evaluate)

    # Stage 6: Web app
    if not args.no_app:
        _print_summary(t_start)
        _run_stage("6. Launch Web App", stage_launch_app)
    else:
        _print_summary(t_start)


def _print_summary(t_start):
    elapsed = time.time() - t_start
    _banner("Pipeline Summary")
    print(f"  Total elapsed: {elapsed:.1f}s\n")

    print("  Stage Timings:")
    for name, t in _stage_times.items():
        print(f"    {name:<35} {t:>8.1f}s")

    if _warnings:
        print(f"\n  WARNINGS ({len(_warnings)}):")
        for w in _warnings:
            print(f"    {w}")
        print("\n  Some stages had issues. Re-run with Claude Code to debug.")
    else:
        print("\n  All stages completed successfully.")

    # Check what artifacts exist
    cache_dir = PROJECT_ROOT / "results" / "cache"
    artifacts = [
        ("embeddings.pt", "SigLIP embeddings"),
        ("classifier_baseline.pkl", "Baseline model (binary)"),
        ("classifier_logistic.pkl", "Logistic regression (binary)"),
        ("classifier_deep.pkl", "Deep MLP (binary)"),
        ("classifier.pkl", "Default binary model (for app)"),
        ("classifier_condition_logistic.pkl", "Logistic regression (condition)"),
        ("classifier_condition_deep.pkl", "Deep MLP (condition)"),
        ("classifier_condition.pkl", "Default condition model"),
        ("training_results.json", "Binary training metrics"),
        ("condition_training_results.json", "Condition training metrics"),
        ("evaluation_results.json", "Full evaluation (both targets)"),
        ("metadata.csv", "Dataset metadata"),
        ("test_metadata.csv", "Test split metadata"),
    ]
    print("\n  Artifacts:")
    for fname, desc in artifacts:
        path = cache_dir / fname
        status = "OK" if path.exists() else "--"
        size = f"({path.stat().st_size / 1024 / 1024:.1f}MB)" if path.exists() else ""
        print(f"    [{status}] {desc:<30} {fname} {size}")


if __name__ == "__main__":
    main()
