#!/usr/bin/env python3
"""Clinical triage analysis for skin lesion classification.

Evaluates classifier performance with clinical sensitivity/specificity
tradeoffs and proposes appropriate triage tiers for field deployment.

Clinical perspective: A dermatologist triaging patients via local technicians
who are not experts. Goal is to help people who might benefit from dermatology
appointments while minimizing missed malignancies.

Key principles:
- HIGH sensitivity for malignancy (never miss melanoma, accept more false positives)
- Actionable guidance (some "benign" conditions still need treatment)
- Appropriate urgency matching clinical risk
"""

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path BEFORE imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data.taxonomy import CONDITION_BINARY, CONDITION_NAMES, Condition


# =============================================================================
# CLINICAL TRIAGE FRAMEWORK
# =============================================================================

# Clinical triage tiers based on dermatological best practices
CLINICAL_TRIAGE_TIERS = {
    "URGENT": {
        "name": "Urgent Specialist Referral",
        "color": "#dc2626",  # red-600
        "timeframe": "Within 2 weeks",
        "description": "Suspicious for skin cancer. Requires urgent dermatology evaluation and possible biopsy.",
        "conditions": [
            Condition.MELANOMA,
            Condition.SQUAMOUS_CELL_CARCINOMA,
        ],
        "threshold_philosophy": "Maximize sensitivity (>95%), accept lower specificity",
    },
    "PRIORITY": {
        "name": "Priority Referral",
        "color": "#ea580c",  # orange-600
        "timeframe": "Within 4-6 weeks",
        "description": "Needs dermatology evaluation. May require treatment or monitoring.",
        "conditions": [
            Condition.BASAL_CELL_CARCINOMA,
            Condition.ACTINIC_KERATOSIS,
        ],
        "threshold_philosophy": "High sensitivity (>90%), balance with specificity",
    },
    "ROUTINE": {
        "name": "Routine Evaluation",
        "color": "#ca8a04",  # yellow-600
        "timeframe": "Within 3 months",
        "description": "Non-urgent but may benefit from dermatology consultation or primary care treatment.",
        "conditions": [
            Condition.NON_NEOPLASTIC,  # inflammatory, infectious - needs treatment
        ],
        "threshold_philosophy": "Balanced sensitivity/specificity",
    },
    "MONITOR": {
        "name": "Self-Monitor",
        "color": "#16a34a",  # green-600
        "timeframe": "Routine self-checks",
        "description": "Likely benign. Continue monitoring for changes using ABCDE criteria.",
        "conditions": [
            Condition.MELANOCYTIC_NEVUS,
            Condition.SEBORRHEIC_KERATOSIS,
            Condition.DERMATOFIBROMA,
            Condition.VASCULAR_LESION,
            Condition.OTHER_UNKNOWN,
        ],
        "threshold_philosophy": "High confidence required to reassure",
    },
}

# Condition-specific clinical guidance for tooltips
CONDITION_GUIDANCE = {
    Condition.MELANOMA: {
        "urgency": "URGENT",
        "action": "Seek urgent dermatology referral for dermoscopy and possible biopsy",
        "warning": "Melanoma can be life-threatening if not caught early. Any suspicion warrants evaluation.",
        "false_negative_cost": "CRITICAL - missed melanoma can be fatal",
        "target_sensitivity": 0.95,
    },
    Condition.SQUAMOUS_CELL_CARCINOMA: {
        "urgency": "URGENT",
        "action": "Refer to dermatology within 2 weeks for evaluation",
        "warning": "SCC can metastasize if untreated. Early treatment has excellent outcomes.",
        "false_negative_cost": "HIGH - delayed treatment increases metastatic risk",
        "target_sensitivity": 0.93,
    },
    Condition.BASAL_CELL_CARCINOMA: {
        "urgency": "PRIORITY",
        "action": "Schedule dermatology appointment within 4-6 weeks",
        "warning": "BCC rarely metastasizes but grows locally and can cause tissue damage.",
        "false_negative_cost": "MODERATE - delayed treatment may require larger excision",
        "target_sensitivity": 0.90,
    },
    Condition.ACTINIC_KERATOSIS: {
        "urgency": "PRIORITY",
        "action": "Dermatology evaluation recommended. May be treated with cryotherapy or topicals.",
        "warning": "Pre-cancerous lesion. ~10% progress to SCC if untreated.",
        "false_negative_cost": "MODERATE - progression risk over time",
        "target_sensitivity": 0.85,
    },
    Condition.NON_NEOPLASTIC: {
        "urgency": "ROUTINE",
        "action": "May benefit from primary care or dermatology treatment",
        "warning": "Includes inflammatory conditions (eczema, psoriasis, dermatitis), infections (fungal, bacterial). Not cancer but often needs treatment for symptom relief.",
        "false_negative_cost": "LOW - conditions are treatable, not life-threatening",
        "target_sensitivity": 0.75,
    },
    Condition.MELANOCYTIC_NEVUS: {
        "urgency": "MONITOR",
        "action": "Continue routine self-monitoring using ABCDE criteria",
        "warning": "Common moles. Monitor for asymmetry, border irregularity, color changes, diameter >6mm, or evolution.",
        "false_negative_cost": "LOW if monitoring instructions followed",
        "target_sensitivity": 0.70,
    },
    Condition.SEBORRHEIC_KERATOSIS: {
        "urgency": "MONITOR",
        "action": "Benign growth. No treatment needed unless cosmetically bothersome.",
        "warning": "Very common in older adults. Can sometimes mimic melanoma to untrained eye.",
        "false_negative_cost": "MINIMAL",
        "target_sensitivity": 0.70,
    },
    Condition.DERMATOFIBROMA: {
        "urgency": "MONITOR",
        "action": "Benign fibrous growth. No treatment needed.",
        "warning": "Firm nodule, often on legs. May persist indefinitely.",
        "false_negative_cost": "MINIMAL",
        "target_sensitivity": 0.70,
    },
    Condition.VASCULAR_LESION: {
        "urgency": "MONITOR",
        "action": "Benign vascular growth. Treatment only for cosmetic reasons.",
        "warning": "Includes cherry angiomas, hemangiomas. Common and harmless.",
        "false_negative_cost": "MINIMAL",
        "target_sensitivity": 0.70,
    },
    Condition.OTHER_UNKNOWN: {
        "urgency": "ROUTINE",
        "action": "If uncertain, recommend primary care evaluation",
        "warning": "Could not be confidently classified. Consider clinical evaluation if symptomatic.",
        "false_negative_cost": "UNKNOWN - recommend evaluation if concerned",
        "target_sensitivity": 0.50,
    },
}


def load_aligned_test_data():
    """Load test embeddings properly aligned with test metadata."""
    cache_dir = Path("results/cache")

    # Load metadata
    full_meta = pd.read_csv(cache_dir / "metadata.csv")
    test_meta = pd.read_csv(cache_dir / "test_metadata.csv")

    # Load all embeddings
    all_embeddings = torch.load(cache_dir / "embeddings.pt", weights_only=True)

    # Create mapping from sample_id to embedding index
    sample_id_to_idx = {sid: idx for idx, sid in enumerate(full_meta["sample_id"])}

    # Get test embeddings in correct order matching test_meta
    test_indices = [sample_id_to_idx[sid] for sid in test_meta["sample_id"]]
    test_embeddings = all_embeddings[test_indices].numpy()

    return test_embeddings, test_meta


def load_classifier(classifier_type="xgboost"):
    """Load a trained classifier."""
    cache_dir = Path("results/cache")
    clf_path = cache_dir / f"classifier_{classifier_type}.pkl"

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    print(f"Loaded classifier from {clf_path}")
    return clf


def evaluate_binary_classifier(X_test, y_true, clf):
    """Evaluate binary classifier with clinical thresholds."""
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_malignant": float(f1_score(y_true, y_pred, pos_label=1)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }

    # Find thresholds for target sensitivities
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    for target_sens in [0.99, 0.95, 0.90, 0.85, 0.80]:
        idx = np.argmin(np.abs(tpr - target_sens))
        thresh = thresholds[idx] if len(thresholds) > idx else 0.5
        spec = 1 - fpr[idx]
        results[f"threshold_{int(target_sens*100)}_sens"] = float(thresh)
        results[f"specificity_at_{int(target_sens*100)}_sens"] = float(spec)

    # Performance at common thresholds
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        sens = recall_score(y_true, y_pred_thresh)
        spec = recall_score(1 - y_true, 1 - y_pred_thresh)
        results[f"sens_at_{int(thresh*100)}"] = float(sens)
        results[f"spec_at_{int(thresh*100)}"] = float(spec)

    return results, y_prob


def evaluate_per_condition(X_test, y_true_condition, clf_condition):
    """Evaluate condition classifier per-class."""
    y_prob = clf_condition.predict_proba(X_test)
    y_pred = clf_condition.predict(X_test)

    results = {}

    for i, condition in enumerate(Condition):
        name = CONDITION_NAMES[condition]

        # Binary: this condition vs all others
        y_binary_true = (y_true_condition == i).astype(int)
        y_binary_pred = (y_pred == i).astype(int)

        tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
        tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
        fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
        fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))

        n_samples = np.sum(y_binary_true)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        try:
            auc = roc_auc_score(y_binary_true, y_prob[:, i])
        except:
            auc = 0.5

        results[condition] = {
            "name": name,
            "n_samples": int(n_samples),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "auc": float(auc),
        }

    return results, y_prob


def analyze_clinical_tiers(y_true_binary, y_prob_binary, y_true_condition, y_prob_condition, test_meta):
    """Analyze performance across clinical triage tiers."""

    tier_analysis = {}

    for tier_name, tier_info in CLINICAL_TRIAGE_TIERS.items():
        tier_conditions = tier_info["conditions"]
        tier_condition_ids = [c.value for c in tier_conditions]

        # Ground truth: is sample in this tier based on condition?
        y_tier_true = np.isin(y_true_condition, tier_condition_ids).astype(int)

        # Prediction using condition probabilities
        tier_prob = y_prob_condition[:, tier_condition_ids].sum(axis=1)

        n_samples = int(np.sum(y_tier_true))

        if n_samples > 0:
            try:
                auc = roc_auc_score(y_tier_true, tier_prob)
            except:
                auc = 0.5

            # Find optimal threshold for different sensitivities
            fpr, tpr, thresholds = roc_curve(y_tier_true, tier_prob)
            for target in [0.95, 0.90, 0.80]:
                idx = np.argmin(np.abs(tpr - target))
                thresh = thresholds[idx] if len(thresholds) > idx else 0.5
                spec = 1 - fpr[idx]
                tier_analysis[f"{tier_name}_thresh_{int(target*100)}"] = float(thresh)
                tier_analysis[f"{tier_name}_spec_at_{int(target*100)}"] = float(spec)
        else:
            auc = 0.5

        tier_analysis[tier_name] = {
            "name": tier_info["name"],
            "n_samples": n_samples,
            "conditions": [CONDITION_NAMES[c] for c in tier_conditions],
            "auc": float(auc),
        }

    return tier_analysis


def generate_web_app_config(binary_results, condition_results, tier_analysis):
    """Generate configuration for web app integration."""

    # Recommended thresholds based on clinical priorities
    # Use HIGH sensitivity for malignancy screening (95% sens = catch nearly all cancers)
    recommended_threshold = binary_results.get("threshold_95_sens", 0.15)

    config = {
        "version": "2.0",
        "binary_classifier": {
            "recommended_threshold": recommended_threshold,
            "auc": binary_results["auc"],
            "thresholds": {
                "high_sensitivity": {
                    "value": binary_results.get("threshold_95_sens", 0.15),
                    "sensitivity": 0.95,
                    "specificity": binary_results.get("specificity_at_95_sens", 0.70),
                    "use_case": "Screening mode - minimize missed malignancies",
                },
                "balanced": {
                    "value": binary_results.get("threshold_90_sens", 0.20),
                    "sensitivity": 0.90,
                    "specificity": binary_results.get("specificity_at_90_sens", 0.80),
                    "use_case": "Balanced mode - good sensitivity with reasonable specificity",
                },
                "high_specificity": {
                    "value": 0.50,
                    "sensitivity": binary_results.get("sens_at_50", 0.80),
                    "specificity": binary_results.get("spec_at_50", 0.90),
                    "use_case": "Confirmation mode - fewer false positives",
                },
            },
        },
        "triage_tiers": {},
        "condition_guidance": {},
        "ui_colors": {
            "urgent": "#dc2626",
            "priority": "#ea580c",
            "routine": "#ca8a04",
            "monitor": "#16a34a",
        },
    }

    # Add triage tier configs
    for tier_name, tier_info in CLINICAL_TRIAGE_TIERS.items():
        config["triage_tiers"][tier_name] = {
            "name": tier_info["name"],
            "color": tier_info["color"],
            "timeframe": tier_info["timeframe"],
            "description": tier_info["description"],
            "conditions": [CONDITION_NAMES[c] for c in tier_info["conditions"]],
        }

    # Add condition-specific guidance (for tooltips)
    for condition, guidance in CONDITION_GUIDANCE.items():
        config["condition_guidance"][CONDITION_NAMES[condition]] = {
            "urgency": guidance["urgency"],
            "action": guidance["action"],
            "warning": guidance["warning"],
            "clinical_priority": guidance["false_negative_cost"],
        }

    # Add per-condition performance
    config["condition_performance"] = {}
    for condition, metrics in condition_results.items():
        config["condition_performance"][metrics["name"]] = {
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "auc": metrics["auc"],
            "n_test_samples": metrics["n_samples"],
        }

    return config


def print_clinical_summary(binary_results, condition_results, tier_analysis):
    """Print clinical summary for review."""

    print("\n" + "=" * 80)
    print("CLINICAL TRIAGE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n## BINARY CLASSIFIER (Malignant vs Benign)")
    print(f"   AUC: {binary_results['auc']:.3f}")
    print(f"   Accuracy: {binary_results['accuracy']:.1%}")
    print(f"   F1 (Malignant): {binary_results['f1_malignant']:.3f}")

    print("\n## SENSITIVITY/SPECIFICITY TRADEOFFS")
    print("   Threshold | Sensitivity | Specificity | Use Case")
    print("   " + "-" * 60)

    for target in [99, 95, 90, 85, 80]:
        thresh = binary_results.get(f"threshold_{target}_sens", 0)
        spec = binary_results.get(f"specificity_at_{target}_sens", 0)
        print(f"   {thresh:.3f}    |    {target}%     |    {spec:.1%}    | ", end="")
        if target >= 95:
            print("SCREENING - catch nearly all cancers")
        elif target >= 90:
            print("BALANCED - good for general use")
        else:
            print("CONFIRMATORY - fewer false alarms")

    print("\n## RECOMMENDED CLINICAL THRESHOLDS")
    print("   For field screening by non-specialists:")
    thresh_95 = binary_results.get("threshold_95_sens", 0.15)
    spec_95 = binary_results.get("specificity_at_95_sens", 0.70)
    print(f"   -> Use threshold {thresh_95:.3f} (95% sensitivity, {spec_95:.1%} specificity)")
    print(f"   -> {(1-spec_95)*100:.0f}% of benign lesions will be flagged for review")
    print("   -> This is acceptable to ensure we don't miss malignancies")

    print("\n## PER-CONDITION CLASSIFIER PERFORMANCE")
    print(f"   {'Condition':<30} {'N':>6} {'Sens':>8} {'Spec':>8} {'AUC':>8}")
    print("   " + "-" * 65)
    for condition in Condition:
        r = condition_results[condition]
        print(f"   {r['name']:<30} {r['n_samples']:>6} {r['sensitivity']:>7.1%} "
              f"{r['specificity']:>7.1%} {r['auc']:>7.3f}")

    print("\n## CLINICAL RECOMMENDATIONS")
    print("   1. PRIMARY TRIAGE: Use binary classifier (malignant vs benign)")
    print("      - High sensitivity mode by default (threshold ~0.15)")
    print("      - Any score above threshold -> recommend dermatology consult")
    print()
    print("   2. CONDITION ESTIMATION: Show likely conditions as INFORMATIONAL")
    print("      - Condition classifier is less accurate than binary")
    print("      - Display as 'Possible conditions' not 'Diagnosis'")
    print("      - Always include disclaimer about AI limitations")
    print()
    print("   3. NON-NEOPLASTIC CONDITIONS (eczema, psoriasis, infections):")
    print("      - These are NOT malignant but may still need treatment")
    print("      - Flag separately: 'May benefit from medical treatment'")
    print("      - Don't classify as 'benign - no action needed'")
    print()
    print("   4. TOOLTIPS: Include condition-specific guidance")
    print("      - What the condition means")
    print("      - Recommended action")
    print("      - Urgency level")


def main():
    print("=" * 60)
    print("CLINICAL TRIAGE ANALYSIS")
    print("Evaluating classifiers for field deployment")
    print("=" * 60)

    # Load data
    print("\nLoading test data...")
    X_test, test_meta = load_aligned_test_data()
    y_true_binary = test_meta["label"].values
    y_true_condition = test_meta["condition_label"].values

    print(f"Test samples: {len(y_true_binary)}")
    print(f"Malignant: {sum(y_true_binary==1)}, Benign: {sum(y_true_binary==0)}")

    # Load classifiers
    print("\nLoading classifiers...")
    clf_binary = load_classifier("xgboost")
    clf_condition = load_classifier("condition_logistic")

    # Evaluate binary classifier
    print("\nEvaluating binary classifier...")
    binary_results, y_prob_binary = evaluate_binary_classifier(X_test, y_true_binary, clf_binary)

    # Evaluate condition classifier
    print("\nEvaluating condition classifier...")
    condition_results, y_prob_condition = evaluate_per_condition(X_test, y_true_condition, clf_condition)

    # Analyze clinical tiers
    print("\nAnalyzing clinical triage tiers...")
    tier_analysis = analyze_clinical_tiers(
        y_true_binary, y_prob_binary, y_true_condition, y_prob_condition, test_meta
    )

    # Print summary
    print_clinical_summary(binary_results, condition_results, tier_analysis)

    # Generate web app config
    config = generate_web_app_config(binary_results, condition_results, tier_analysis)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    config_path = output_dir / "clinical_triage_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n\nConfiguration saved to: {config_path}")

    # Save detailed results
    detailed_results = {
        "binary_classifier": binary_results,
        "condition_classifier": {CONDITION_NAMES[c]: r for c, r in condition_results.items()},
        "tier_analysis": tier_analysis,
    }

    results_path = output_dir / "clinical_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return binary_results, condition_results, tier_analysis


if __name__ == "__main__":
    main()
