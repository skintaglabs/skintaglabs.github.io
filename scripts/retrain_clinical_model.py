#!/usr/bin/env python3
"""Retrain SigLIP model with clinical triage optimization.

This script implements hierarchical multi-task fine-tuning with clinical class
weights to improve condition-level discrimination, especially for high-priority
conditions (melanoma, SCC).

Run standalone after setting up environment:
    python scripts/retrain_clinical_model.py --epochs 10 --batch_size 32

For quick test run:
    python scripts/retrain_clinical_model.py --quick --epochs 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from tqdm import tqdm

# Clinical triage configuration
CLINICAL_TIERS = {
    "URGENT": [0, 2],      # Melanoma (0), SCC (2) - within 2 weeks
    "PRIORITY": [1, 3],    # BCC (1), Actinic Keratosis (3) - 4-6 weeks
    "ROUTINE": [8],        # Non-neoplastic (8) - 3 months
    "MONITOR": [4, 5, 6, 7, 9],  # Benign lesions - self-monitor
}

# Condition names for reporting
CONDITION_NAMES = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Melanocytic Nevus",
    5: "Seborrheic Keratosis",
    6: "Dermatofibroma",
    7: "Vascular Lesion",
    8: "Non-Neoplastic",
    9: "Other/Unknown",
}

# Clinical class weights - higher for dangerous/rare conditions
CLINICAL_CLASS_WEIGHTS = {
    0: 15.0,   # Melanoma - critical, relatively rare
    1: 3.0,    # BCC - common, priority
    2: 12.0,   # SCC - urgent, less common
    3: 8.0,    # Actinic Keratosis - precancerous
    4: 0.5,    # Melanocytic Nevus - very common
    5: 1.0,    # Seborrheic Keratosis - common
    6: 2.0,    # Dermatofibroma - less common
    7: 2.0,    # Vascular Lesion - less common
    8: 1.5,    # Non-Neoplastic - needs separation
    9: 0.8,    # Other/Unknown - catch-all
}

# Binary mapping
CONDITION_TO_BINARY = {
    0: 1, 1: 1, 2: 1, 3: 1,  # Malignant
    4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,  # Benign
}


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted embeddings."""

    def __init__(self, embeddings, binary_labels, condition_labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        self.condition_labels = torch.tensor(condition_labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "binary_label": self.binary_labels[idx],
            "condition_label": self.condition_labels[idx],
        }


class ClinicalFocalLoss(nn.Module):
    """Focal loss with clinical asymmetry for high-priority conditions."""

    def __init__(self, gamma=2.0, class_weights=None, false_negative_penalty=3.0):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.fn_penalty = false_negative_penalty

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        # Focal weight
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        # Class weights
        if self.class_weights is not None:
            weights = self.class_weights[targets]
        else:
            weights = torch.ones_like(targets, dtype=torch.float32)

        # Extra penalty for false negatives on malignant conditions (0-3)
        is_malignant = targets < 4
        penalty = torch.where(is_malignant, self.fn_penalty, 1.0)

        # Cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Combined loss
        loss = penalty * weights * focal_weight * ce_loss
        return loss.mean()


class HierarchicalClassifier(nn.Module):
    """Multi-task classifier with binary and condition heads."""

    def __init__(self, input_dim=1152, hidden_dim=512, n_conditions=10, dropout=0.3):
        super().__init__()

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Binary head (malignant vs benign)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Condition head (10-class)
        self.condition_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Linear(128, n_conditions),
        )

        # Tier head (4-class: URGENT, PRIORITY, ROUTINE, MONITOR)
        self.tier_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        shared_repr = self.shared(x)
        binary_logits = self.binary_head(shared_repr)
        condition_logits = self.condition_head(shared_repr)
        tier_logits = self.tier_head(shared_repr)
        return binary_logits, condition_logits, tier_logits


def condition_to_tier(condition_labels):
    """Map condition labels to tier labels."""
    tier_labels = torch.zeros_like(condition_labels)
    for tier_idx, (tier_name, conditions) in enumerate(CLINICAL_TIERS.items()):
        for cond in conditions:
            tier_labels[condition_labels == cond] = tier_idx
    return tier_labels


def load_data(cache_dir, quick=False):
    """Load embeddings and metadata."""
    print("Loading data...")

    # Load metadata
    full_meta = pd.read_csv(cache_dir / "metadata.csv")
    test_meta = pd.read_csv(cache_dir / "test_metadata.csv")

    # Load embeddings
    all_embeddings = torch.load(cache_dir / "embeddings.pt", weights_only=True).numpy()

    # Create train/test split indices
    test_ids = set(test_meta["sample_id"])
    train_mask = ~full_meta["sample_id"].isin(test_ids)
    test_mask = full_meta["sample_id"].isin(test_ids)

    X_train = all_embeddings[train_mask]
    X_test = all_embeddings[test_mask]

    y_train_binary = full_meta.loc[train_mask, "label"].values
    y_test_binary = full_meta.loc[test_mask, "label"].values

    y_train_condition = full_meta.loc[train_mask, "condition_label"].values
    y_test_condition = full_meta.loc[test_mask, "condition_label"].values

    if quick:
        # Subsample for quick testing
        n_train = min(5000, len(X_train))
        n_test = min(1000, len(X_test))
        indices_train = np.random.choice(len(X_train), n_train, replace=False)
        indices_test = np.random.choice(len(X_test), n_test, replace=False)

        X_train = X_train[indices_train]
        X_test = X_test[indices_test]
        y_train_binary = y_train_binary[indices_train]
        y_test_binary = y_test_binary[indices_test]
        y_train_condition = y_train_condition[indices_train]
        y_test_condition = y_test_condition[indices_test]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Train malignant: {sum(y_train_binary)}")
    print(f"  Embedding dim: {X_train.shape[1]}")

    return (X_train, y_train_binary, y_train_condition,
            X_test, y_test_binary, y_test_condition)


def create_weighted_sampler(condition_labels, class_weights):
    """Create weighted sampler for balanced training."""
    sample_weights = np.array([class_weights[c] for c in condition_labels])
    sample_weights = sample_weights / sample_weights.sum()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(condition_labels),
        replacement=True,
    )
    return sampler


def train_epoch(model, dataloader, optimizer, binary_criterion, condition_criterion,
                tier_criterion, class_weights_tensor, device, binary_weight=0.4,
                condition_weight=0.4, tier_weight=0.2):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        binary_labels = batch["binary_label"].to(device)
        condition_labels = batch["condition_label"].to(device)
        tier_labels = condition_to_tier(condition_labels).to(device)

        optimizer.zero_grad()

        binary_logits, condition_logits, tier_logits = model(embeddings)

        # Multi-task loss
        loss_binary = binary_criterion(binary_logits, binary_labels)
        loss_condition = condition_criterion(condition_logits, condition_labels)
        loss_tier = tier_criterion(tier_logits, tier_labels)

        loss = (binary_weight * loss_binary +
                condition_weight * loss_condition +
                tier_weight * loss_tier)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.eval()

    all_binary_preds = []
    all_binary_probs = []
    all_binary_labels = []
    all_condition_preds = []
    all_condition_probs = []
    all_condition_labels = []

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch["embedding"].to(device)
            binary_labels = batch["binary_label"]
            condition_labels = batch["condition_label"]

            binary_logits, condition_logits, _ = model(embeddings)

            binary_probs = F.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()
            binary_preds = binary_logits.argmax(dim=1).cpu().numpy()

            condition_probs = F.softmax(condition_logits, dim=1).cpu().numpy()
            condition_preds = condition_logits.argmax(dim=1).cpu().numpy()

            all_binary_preds.extend(binary_preds)
            all_binary_probs.extend(binary_probs)
            all_binary_labels.extend(binary_labels.numpy())
            all_condition_preds.extend(condition_preds)
            all_condition_probs.extend(condition_probs)
            all_condition_labels.extend(condition_labels.numpy())

    # Calculate metrics
    binary_acc = accuracy_score(all_binary_labels, all_binary_preds)
    binary_f1 = f1_score(all_binary_labels, all_binary_preds, pos_label=1)
    binary_auc = roc_auc_score(all_binary_labels, all_binary_probs)

    condition_acc = accuracy_score(all_condition_labels, all_condition_preds)
    condition_f1 = f1_score(all_condition_labels, all_condition_preds, average='macro')

    # Per-condition sensitivity for URGENT conditions
    all_condition_labels = np.array(all_condition_labels)
    all_condition_preds = np.array(all_condition_preds)

    melanoma_mask = all_condition_labels == 0
    melanoma_sens = (all_condition_preds[melanoma_mask] == 0).mean() if melanoma_mask.sum() > 0 else 0

    scc_mask = all_condition_labels == 2
    scc_sens = (all_condition_preds[scc_mask] == 2).mean() if scc_mask.sum() > 0 else 0

    return {
        "binary_accuracy": binary_acc,
        "binary_f1": binary_f1,
        "binary_auc": binary_auc,
        "condition_accuracy": condition_acc,
        "condition_f1_macro": condition_f1,
        "melanoma_sensitivity": melanoma_sens,
        "scc_sensitivity": scc_sens,
    }


def save_model(model, output_dir, metrics, epoch):
    """Save model and training info."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), output_dir / "clinical_classifier.pt")

    # Save config
    config = {
        "model_type": "HierarchicalClassifier",
        "input_dim": 1152,
        "hidden_dim": 512,
        "n_conditions": 10,
        "trained_epochs": epoch,
        "final_metrics": metrics,
        "class_weights": CLINICAL_CLASS_WEIGHTS,
        "clinical_tiers": {k: v for k, v in CLINICAL_TIERS.items()},
        "condition_names": CONDITION_NAMES,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "clinical_classifier_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Retrain model with clinical optimization")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick test with subset of data")
    parser.add_argument("--output_dir", type=str, default="results/cache/clinical_model",
                        help="Output directory for trained model")
    parser.add_argument("--cache_dir", type=str, default="results/cache",
                        help="Directory with embeddings and metadata")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    cache_dir = Path(args.cache_dir)
    (X_train, y_train_binary, y_train_condition,
     X_test, y_test_binary, y_test_condition) = load_data(cache_dir, quick=args.quick)

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train_binary, y_train_condition)
    test_dataset = EmbeddingDataset(X_test, y_test_binary, y_test_condition)

    # Create weighted sampler for balanced training
    sampler = create_weighted_sampler(y_train_condition, CLINICAL_CLASS_WEIGHTS)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = HierarchicalClassifier(input_dim=X_train.shape[1]).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights tensor for focal loss
    class_weights = torch.tensor(
        [CLINICAL_CLASS_WEIGHTS[i] for i in range(10)],
        dtype=torch.float32
    ).to(device)

    # Loss functions
    binary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0], device=device)  # Higher weight for malignant
    )
    condition_criterion = ClinicalFocalLoss(
        gamma=2.0,
        class_weights=class_weights,
        false_negative_penalty=3.0,
    )
    tier_criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING WITH CLINICAL OPTIMIZATION")
    print("=" * 60)

    best_auc = 0
    best_metrics = None

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer,
            binary_criterion, condition_criterion, tier_criterion,
            class_weights, device,
        )

        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Binary: Acc={metrics['binary_accuracy']:.1%}, F1={metrics['binary_f1']:.3f}, AUC={metrics['binary_auc']:.3f}")
        print(f"  Condition: Acc={metrics['condition_accuracy']:.1%}, F1={metrics['condition_f1_macro']:.3f}")
        print(f"  URGENT: Melanoma Sens={metrics['melanoma_sensitivity']:.1%}, SCC Sens={metrics['scc_sensitivity']:.1%}")

        # Save best model
        if metrics['binary_auc'] > best_auc:
            best_auc = metrics['binary_auc']
            best_metrics = metrics
            save_model(model, args.output_dir, metrics, epoch + 1)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Binary AUC: {best_auc:.3f}")
    print(f"Best Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Generate clinical evaluation report
    print("\n" + "=" * 60)
    print("CLINICAL EVALUATION")
    print("=" * 60)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embedding"].to(device)
            binary_labels = batch["binary_label"]

            binary_logits, _, _ = model(embeddings)
            probs = F.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(binary_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Find threshold for 95% sensitivity
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    threshold_95 = thresholds[idx_95]
    specificity_95 = 1 - fpr[idx_95]

    print(f"\nAt 95% sensitivity:")
    print(f"  Threshold: {threshold_95:.3f}")
    print(f"  Specificity: {specificity_95:.1%}")

    # Save clinical thresholds
    clinical_config = {
        "recommended_threshold": float(threshold_95),
        "sensitivity_at_threshold": 0.95,
        "specificity_at_threshold": float(specificity_95),
        "model_path": str(Path(args.output_dir) / "clinical_classifier.pt"),
    }

    with open(Path(args.output_dir) / "clinical_thresholds.json", "w") as f:
        json.dump(clinical_config, f, indent=2)

    print(f"\nClinical config saved to {args.output_dir}/clinical_thresholds.json")


if __name__ == "__main__":
    main()
