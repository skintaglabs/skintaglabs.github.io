"""Evaluation metrics for robustness, fairness, and model comparison."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_accuracy(y_true, y_pred):
    """Compute overall accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_per_group_accuracy(y_true, y_pred, groups):
    """Compute accuracy per demographic/condition group.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        groups: Group assignment for each sample

    Returns:
        Dictionary mapping group -> accuracy
    """
    unique_groups = np.unique(groups)
    group_acc = {}
    for group in unique_groups:
        mask = groups == group
        if mask.sum() > 0:
            group_acc[group] = accuracy_score(y_true[mask], y_pred[mask])
    return group_acc


def compute_fairness_gap(group_accuracies: dict):
    """Compute max accuracy gap between groups."""
    if not group_accuracies:
        return 0.0
    accuracies = list(group_accuracies.values())
    return max(accuracies) - min(accuracies)


def compute_per_group_metrics(y_true, y_pred, y_proba, groups):
    """Compute comprehensive metrics per group.

    Returns dict of group -> {accuracy, balanced_accuracy, sensitivity, specificity, auc, n}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray([str(g) for g in groups])  # normalize mixed types
    unique_groups = np.unique(groups)
    results = {}

    for group in unique_groups:
        mask = groups == group
        n = mask.sum()
        if n < 2:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        acc = accuracy_score(yt, yp)
        bal_acc = balanced_accuracy_score(yt, yp) if len(np.unique(yt)) > 1 else acc

        # Sensitivity (recall for malignant=1) and specificity
        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')

        # F1 score — harmonic mean of precision and sensitivity (recall)
        if not (np.isnan(precision) or np.isnan(sensitivity)) and (precision + sensitivity) > 0:
            f1 = 2 * precision * sensitivity / (precision + sensitivity)
        else:
            f1 = float('nan')

        # Macro F1 (via sklearn for robustness with edge cases)
        if len(np.unique(yt)) > 1:
            f1_macro = float(f1_score(yt, yp, average='macro', zero_division=0))
        else:
            f1_macro = f1

        # AUC
        auc = float('nan')
        if y_proba is not None and len(np.unique(yt)) > 1:
            try:
                proba = np.asarray(y_proba)[mask]
                if proba.ndim == 2:
                    proba = proba[:, 1]
                auc = roc_auc_score(yt, proba)
            except (ValueError, IndexError):
                pass

        results[str(group)] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "f1_macro": float(f1_macro),
            "auc": float(auc),
            "n": int(n),
        }

    return results


def compute_equalized_odds_gap(per_group_metrics: dict):
    """Compute equalized odds gap: max difference in sensitivity, specificity, and F1 across groups.

    Returns dict with sensitivity_gap, specificity_gap, f1_gap, max_gap.
    """
    sensitivities = [m["sensitivity"] for m in per_group_metrics.values() if not np.isnan(m["sensitivity"])]
    specificities = [m["specificity"] for m in per_group_metrics.values() if not np.isnan(m["specificity"])]
    f1s = [m["f1"] for m in per_group_metrics.values() if not np.isnan(m["f1"])]

    sens_gap = (max(sensitivities) - min(sensitivities)) if len(sensitivities) >= 2 else 0.0
    spec_gap = (max(specificities) - min(specificities)) if len(specificities) >= 2 else 0.0
    f1_gap = (max(f1s) - min(f1s)) if len(f1s) >= 2 else 0.0

    return {
        "sensitivity_gap": float(sens_gap),
        "specificity_gap": float(spec_gap),
        "f1_gap": float(f1_gap),
        "max_gap": float(max(sens_gap, spec_gap, f1_gap)),
    }


def cross_domain_report(y_true, y_pred, y_proba, domains):
    """Per-domain accuracy report and domain gap.

    Returns dict with per_domain metrics and domain_gap.
    """
    per_domain = compute_per_group_metrics(y_true, y_pred, y_proba, domains)
    accuracies = [m["accuracy"] for m in per_domain.values()]
    domain_gap = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0

    return {
        "per_domain": per_domain,
        "domain_accuracy_gap": float(domain_gap),
    }


def compare_models(model_results: dict):
    """Side-by-side comparison table of multiple model results.

    Args:
        model_results: dict of model_name -> {metric: value}

    Returns:
        dict with comparison data and best_model name
    """
    comparison = {}
    best_model = None
    best_metric = -1

    for name, metrics in model_results.items():
        comparison[name] = metrics
        # Use F1 macro as primary ranking metric (best for imbalanced dermatology data),
        # falling back to balanced accuracy, then test accuracy
        primary = metrics.get("f1_macro", metrics.get("balanced_accuracy", metrics.get("test_accuracy", 0)))
        if primary > best_metric:
            best_metric = primary
            best_model = name

    return {
        "models": comparison,
        "best_model": best_model,
        "best_metric": float(best_metric),
    }


def condition_classification_report(y_true, y_pred):
    """Multi-class condition classification metrics.

    Returns dict with overall accuracy, F1 macro, and per-condition breakdown
    (precision, recall/sensitivity, F1, support count).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-condition metrics
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_condition = {}
    for cls in unique_classes:
        mask_true = y_true == cls
        mask_pred = y_pred == cls
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        n = int(mask_true.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_condition[int(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_val),
            "n": n,
        }

    return {
        "accuracy": float(acc),
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "per_condition": per_condition,
    }


def robustness_report(y_true, y_pred, groups=None, class_names=None, y_proba=None):
    """Generate full robustness evaluation report.

    Extended with Fitzpatrick fairness, equalized odds, and domain breakdowns.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    multi_class = len(np.unique(y_true)) > 1

    report = {
        "overall_accuracy": compute_accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if multi_class else compute_accuracy(y_true, y_pred),
        "f1_binary": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # AUC
    if y_proba is not None and multi_class:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2:
                proba = proba[:, 1]
            report["auc"] = float(roc_auc_score(y_true, proba))
        except (ValueError, IndexError):
            pass

    if groups is not None:
        for group_name, group_values in groups.items():
            group_values = np.asarray(group_values)
            per_group = compute_per_group_metrics(y_true, y_pred, y_proba, group_values)
            report[f"per_{group_name}"] = per_group

            # Fairness gap
            group_acc = {g: m["accuracy"] for g, m in per_group.items()}
            report[f"{group_name}_fairness_gap"] = compute_fairness_gap(group_acc)

            # Equalized odds gap
            eq_odds = compute_equalized_odds_gap(per_group)
            report[f"{group_name}_equalized_odds"] = eq_odds

    return report
