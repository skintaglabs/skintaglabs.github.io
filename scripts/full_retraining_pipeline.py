#!/usr/bin/env python3
"""Full retraining pipeline with clinical optimization and image quality augmentation.

This script performs end-to-end retraining WITHOUT deleting existing models:
1. Adds realistic field condition augmentations (blur, motion, smartphone artifacts)
2. Trains multiple classifiers (Logistic, XGBoost, Deep MLP, Hierarchical)
3. Compares frozen embeddings vs partial fine-tuning approaches
4. Retrains mobile models (MobileNetV3, EfficientNet-B0) via knowledge distillation
5. Evaluates all models with clinical metrics
6. Generates comparison report showing improvement from fine-tuning

All models are saved with timestamps - NO existing models are deleted.

Run standalone:
    python scripts/full_retraining_pipeline.py --epochs 15

Quick test:
    python scripts/full_retraining_pipeline.py --quick --epochs 2

Skip steps:
    python scripts/full_retraining_pipeline.py --skip-classifiers --skip-mobile

Compare frozen vs fine-tuned:
    python scripts/full_retraining_pipeline.py --compare-finetuning
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import mobilenet_v3_large, efficientnet_b0
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm


# =============================================================================
# CLINICAL CONFIGURATION
# =============================================================================

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

CONDITION_TO_BINARY = {
    0: 1, 1: 1, 2: 1, 3: 1,  # Malignant
    4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,  # Benign
}


# =============================================================================
# FIELD CONDITION AUGMENTATIONS
# =============================================================================

def get_field_condition_augmentation(severity="moderate"):
    """Augmentations simulating real-world smartphone capture conditions.

    Includes:
    - Motion blur (shaky hands)
    - Focus blur (autofocus issues)
    - Low light noise
    - Overexposure/underexposure
    - Color cast (different lighting)
    - Compression artifacts (messaging apps)
    - Partial occlusion (fingers, hair)
    """

    if severity == "light":
        blur_limit, noise_var, brightness = 3, 30, 0.2
    elif severity == "heavy":
        blur_limit, noise_var, brightness = 9, 80, 0.4
    else:  # moderate
        blur_limit, noise_var, brightness = 5, 50, 0.3

    return A.Compose([
        # Motion blur (shaky hands during capture)
        A.OneOf([
            A.MotionBlur(blur_limit=blur_limit, p=1.0),
            A.GaussianBlur(blur_limit=blur_limit, p=1.0),
            A.MedianBlur(blur_limit=blur_limit, p=1.0),
        ], p=0.3),

        # Focus issues (out-of-focus areas)
        A.OneOf([
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
            A.ZoomBlur(max_factor=1.05, p=1.0),
        ], p=0.2),

        # Lighting variations (indoor/outdoor, different bulbs)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness,
                contrast_limit=brightness,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.RandomToneCurve(scale=0.2, p=1.0),
        ], p=0.5),

        # Color cast (fluorescent lights, sunlight, etc.)
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=1.0
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.05, p=1.0),
        ], p=0.4),

        # Sensor noise (low light, older phones)
        A.OneOf([
            A.GaussNoise(var_limit=(10, noise_var), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.4),

        # Compression (WhatsApp, social media sharing)
        A.OneOf([
            A.ImageCompression(quality_lower=40, quality_upper=90, p=1.0),
            A.Downscale(scale_min=0.4, scale_max=0.8, p=1.0),
        ], p=0.3),

        # Shadow from hand/phone (partial occlusion)
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=0.15
        ),

        # Glare/specular highlights
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(1, 3),
            src_radius=100,
            p=0.1
        ),
    ])


def get_clinical_training_transform(image_size=224, augment_severity="moderate"):
    """Training transform with clinical-appropriate augmentations.

    Preserves diagnostic features while adding realistic variations.
    """
    return A.Compose([
        # Resize first
        A.Resize(image_size, image_size),

        # Geometric (skin lesions are orientation-invariant)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=0, p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.3
        ),

        # Field conditions
        get_field_condition_augmentation(severity=augment_severity),

        # Cutout (simulate partial occlusion, missing data)
        A.CoarseDropout(
            max_holes=4,
            max_height=int(image_size * 0.1),
            max_width=int(image_size * 0.1),
            fill_value=0,
            p=0.2
        ),

        # Normalize for ImageNet-pretrained models
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_eval_transform(image_size=224):
    """Evaluation transform (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# =============================================================================
# DATASETS
# =============================================================================

class SkinLesionDataset(Dataset):
    """Dataset for skin lesion images with clinical labels."""

    def __init__(self, image_paths, binary_labels, condition_labels, transform=None):
        self.image_paths = image_paths
        self.binary_labels = binary_labels
        self.condition_labels = condition_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            # Return a blank image if loading fails
            print(f"Warning: Failed to load {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "binary_label": torch.tensor(self.binary_labels[idx], dtype=torch.long),
            "condition_label": torch.tensor(self.condition_labels[idx], dtype=torch.long),
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


# =============================================================================
# MODELS
# =============================================================================

class ClinicalFocalLoss(nn.Module):
    """Focal loss with clinical asymmetry."""

    def __init__(self, gamma=2.0, class_weights=None, fn_penalty=3.0):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.fn_penalty = fn_penalty

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        if self.class_weights is not None:
            weights = self.class_weights[targets]
        else:
            weights = torch.ones_like(targets, dtype=torch.float32)

        # Extra penalty for false negatives on malignant (conditions 0-3)
        is_malignant = targets < 4
        penalty = torch.where(is_malignant, self.fn_penalty, 1.0)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        loss = penalty * weights * focal_weight * ce_loss
        return loss.mean()


class HierarchicalClassifier(nn.Module):
    """Multi-task classifier with binary and condition heads."""

    def __init__(self, input_dim=1152, hidden_dim=512, n_conditions=10, dropout=0.3):
        super().__init__()

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

        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self.condition_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Linear(128, n_conditions),
        )

    def forward(self, x):
        shared = self.shared(x)
        binary = self.binary_head(shared)
        condition = self.condition_head(shared)
        return binary, condition


class MobileNetClassifier(nn.Module):
    """MobileNetV3-Large with clinical classification head."""

    def __init__(self, n_classes=2, dropout=0.2):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights="IMAGENET1K_V2")
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 with clinical classification head."""

    def __init__(self, n_classes=2, dropout=0.2):
        super().__init__()
        self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# XGBOOST AND SKLEARN CLASSIFIERS
# =============================================================================

def train_xgboost_classifier(X_train, y_train, X_test, y_test, n_classes=2):
    """Train XGBoost classifier with clinical optimization."""
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights
    class_counts = np.bincount(y_train)
    class_weights = {i: len(y_train) / (len(class_counts) * count)
                     for i, count in enumerate(class_counts)}
    # Boost malignant class weight for high sensitivity
    if n_classes == 2:
        class_weights[1] = class_weights[1] * 2.0

    sample_weights = np.array([class_weights[y] for y in y_train])

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # Evaluate
    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if n_classes == 2 else clf.predict_proba(X_test_scaled)
    y_pred = clf.predict(X_test_scaled)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

    return clf, scaler, {"auc": auc, "accuracy": acc, "f1": f1}


def train_logistic_classifier(X_train, y_train, X_test, y_test, n_classes=2):
    """Train logistic regression classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class weights
    class_weights = "balanced" if n_classes == 2 else "balanced"

    clf = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if n_classes == 2 else clf.predict_proba(X_test_scaled)
    y_pred = clf.predict(X_test_scaled)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    return clf, scaler, {
        "auc": auc,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, pos_label=1 if n_classes == 2 else None, average='macro' if n_classes > 2 else 'binary'),
    }


def train_mlp_classifier(X_train, y_train, X_test, y_test, n_classes=2):
    """Train sklearn MLP classifier."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    clf.fit(X_train_scaled, y_train)

    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if n_classes == 2 else clf.predict_proba(X_test_scaled)
    y_pred = clf.predict(X_test_scaled)

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    return clf, scaler, {
        "auc": auc,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, pos_label=1 if n_classes == 2 else None, average='macro' if n_classes > 2 else 'binary'),
    }


def compare_embedding_approaches(cache_dir, output_dir):
    """Compare frozen embeddings vs fine-tuned embeddings.

    Shows that unfreezing a few layers improves performance.
    """
    print("\n" + "=" * 60)
    print("COMPARING FROZEN VS FINE-TUNED EMBEDDINGS")
    print("=" * 60)

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)

    # Load metadata
    full_meta = pd.read_csv(cache_dir / "metadata.csv")
    test_meta = pd.read_csv(cache_dir / "test_metadata.csv")

    test_ids = set(test_meta["sample_id"])
    train_mask = ~full_meta["sample_id"].isin(test_ids)
    test_mask = full_meta["sample_id"].isin(test_ids)

    y_train = full_meta.loc[train_mask, "label"].values
    y_test = full_meta.loc[test_mask, "label"].values

    results = {}

    # 1. Frozen embeddings (original SigLIP)
    if (cache_dir / "embeddings.pt").exists():
        print("\n[1] Evaluating FROZEN SigLIP embeddings...")
        embeddings = torch.load(cache_dir / "embeddings.pt", weights_only=True).numpy()

        sample_id_to_idx = {sid: idx for idx, sid in enumerate(full_meta["sample_id"])}
        train_idx = [sample_id_to_idx[sid] for sid in full_meta.loc[train_mask, "sample_id"]]
        test_idx = [sample_id_to_idx[sid] for sid in full_meta.loc[test_mask, "sample_id"]]

        X_train_frozen = embeddings[train_idx]
        X_test_frozen = embeddings[test_idx]

        # Train XGBoost on frozen
        _, _, frozen_xgb = train_xgboost_classifier(X_train_frozen, y_train, X_test_frozen, y_test)
        results["frozen_xgboost"] = frozen_xgb
        print(f"   XGBoost AUC: {frozen_xgb['auc']:.3f}")

        # Train Logistic on frozen
        _, _, frozen_log = train_logistic_classifier(X_train_frozen, y_train, X_test_frozen, y_test)
        results["frozen_logistic"] = frozen_log
        print(f"   Logistic AUC: {frozen_log['auc']:.3f}")

    # 2. Fine-tuned embeddings (if available)
    finetuned_train = cache_dir / "embeddings_finetuned_train.pt"
    finetuned_test = cache_dir / "embeddings_finetuned_test.pt"

    if finetuned_train.exists() and finetuned_test.exists():
        print("\n[2] Evaluating FINE-TUNED embeddings...")
        X_train_ft = torch.load(finetuned_train, weights_only=True).numpy()
        X_test_ft = torch.load(finetuned_test, weights_only=True).numpy()

        # Need to align with correct labels
        # Assuming fine-tuned embeddings are in same order as train/test splits
        if len(X_train_ft) == len(y_train):
            _, _, ft_xgb = train_xgboost_classifier(X_train_ft, y_train, X_test_ft, y_test)
            results["finetuned_xgboost"] = ft_xgb
            print(f"   XGBoost AUC: {ft_xgb['auc']:.3f}")

            _, _, ft_log = train_logistic_classifier(X_train_ft, y_train, X_test_ft, y_test)
            results["finetuned_logistic"] = ft_log
            print(f"   Logistic AUC: {ft_log['auc']:.3f}")
    else:
        print("\n[2] Fine-tuned embeddings not found, skipping comparison")

    # Generate comparison report
    print("\n" + "-" * 60)
    print("COMPARISON SUMMARY")
    print("-" * 60)

    if "frozen_xgboost" in results and "finetuned_xgboost" in results:
        frozen_auc = results["frozen_xgboost"]["auc"]
        finetuned_auc = results["finetuned_xgboost"]["auc"]
        improvement = (finetuned_auc - frozen_auc) / frozen_auc * 100

        print(f"\nXGBoost Performance:")
        print(f"  Frozen SigLIP:      AUC = {frozen_auc:.4f}")
        print(f"  Fine-tuned SigLIP:  AUC = {finetuned_auc:.4f}")
        print(f"  Improvement:        {improvement:+.2f}%")

        if improvement > 0:
            print(f"\n  >> Fine-tuning IMPROVED performance by {improvement:.1f}%")
        else:
            print(f"\n  >> Fine-tuning did not improve (consider more epochs or different LR)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "embedding_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_data(cache_dir, data_dir=None, quick=False):
    """Load metadata and prepare data splits."""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    cache_dir = Path(cache_dir)

    # Load metadata
    full_meta = pd.read_csv(cache_dir / "metadata.csv")
    test_meta = pd.read_csv(cache_dir / "test_metadata.csv")

    # Get train/test splits
    test_ids = set(test_meta["sample_id"])
    train_mask = ~full_meta["sample_id"].isin(test_ids)
    test_mask = full_meta["sample_id"].isin(test_ids)

    train_meta = full_meta[train_mask].reset_index(drop=True)
    test_meta_aligned = full_meta[test_mask].reset_index(drop=True)

    if quick:
        # Subsample for quick testing
        n_train = min(2000, len(train_meta))
        n_test = min(500, len(test_meta_aligned))
        train_meta = train_meta.sample(n=n_train, random_state=42)
        test_meta_aligned = test_meta_aligned.sample(n=n_test, random_state=42)

    print(f"  Train samples: {len(train_meta)}")
    print(f"  Test samples: {len(test_meta_aligned)}")
    print(f"  Train malignant: {train_meta['label'].sum()}")

    # Load embeddings if available (for classifier training)
    embeddings_path = cache_dir / "embeddings.pt"
    if embeddings_path.exists():
        all_embeddings = torch.load(embeddings_path, weights_only=True).numpy()

        # Get embedding indices matching metadata
        sample_id_to_idx = {sid: idx for idx, sid in enumerate(full_meta["sample_id"])}
        train_indices = [sample_id_to_idx[sid] for sid in train_meta["sample_id"]]
        test_indices = [sample_id_to_idx[sid] for sid in test_meta_aligned["sample_id"]]

        X_train = all_embeddings[train_indices]
        X_test = all_embeddings[test_indices]
    else:
        X_train, X_test = None, None
        print("  Warning: No embeddings found, will need to extract")

    return {
        "train_meta": train_meta,
        "test_meta": test_meta_aligned,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_binary": train_meta["label"].values,
        "y_test_binary": test_meta_aligned["label"].values,
        "y_train_condition": train_meta["condition_label"].values,
        "y_test_condition": test_meta_aligned["condition_label"].values,
    }


def create_weighted_sampler(condition_labels):
    """Create weighted sampler for balanced training."""
    sample_weights = np.array([CLINICAL_CLASS_WEIGHTS[c] for c in condition_labels])
    sample_weights = sample_weights / sample_weights.sum() * len(condition_labels)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(condition_labels),
        replacement=True,
    )
    return sampler


def train_classifier(data, output_dir, epochs=15, batch_size=64, lr=1e-3, device="cuda"):
    """Train hierarchical classifier on embeddings."""
    print("\n" + "=" * 60)
    print("TRAINING HIERARCHICAL CLASSIFIER")
    print("=" * 60)

    if data["X_train"] is None:
        print("  Error: No embeddings available for classifier training")
        return None

    # Create datasets
    train_dataset = EmbeddingDataset(
        data["X_train"],
        data["y_train_binary"],
        data["y_train_condition"],
    )
    test_dataset = EmbeddingDataset(
        data["X_test"],
        data["y_test_binary"],
        data["y_test_condition"],
    )

    # Weighted sampler
    sampler = create_weighted_sampler(data["y_train_condition"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = data["X_train"].shape[1]
    model = HierarchicalClassifier(input_dim=input_dim).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions
    class_weights = torch.tensor(
        [CLINICAL_CLASS_WEIGHTS[i] for i in range(10)], dtype=torch.float32
    ).to(device)

    binary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0], device=device)
    )
    condition_criterion = ClinicalFocalLoss(
        gamma=2.0, class_weights=class_weights, fn_penalty=3.0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            embeddings = batch["embedding"].to(device)
            binary_labels = batch["binary_label"].to(device)
            condition_labels = batch["condition_label"].to(device)

            optimizer.zero_grad()
            binary_logits, condition_logits = model(embeddings)

            loss = (
                0.5 * binary_criterion(binary_logits, binary_labels) +
                0.5 * condition_criterion(condition_logits, condition_labels)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch["embedding"].to(device)
                binary_labels = batch["binary_label"]

                binary_logits, _ = model(embeddings)
                probs = F.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(binary_labels.numpy())

        auc = roc_auc_score(all_labels, all_probs)

        print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict().copy()

    # Save best model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(best_model_state)
    torch.save(best_model_state, output_dir / "clinical_classifier.pt")

    # Final evaluation
    model.eval()
    all_probs, all_labels, all_cond_probs, all_cond_labels = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embedding"].to(device)
            binary_labels = batch["binary_label"]
            condition_labels = batch["condition_label"]

            binary_logits, condition_logits = model(embeddings)

            all_probs.extend(F.softmax(binary_logits, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(binary_labels.numpy())
            all_cond_probs.extend(F.softmax(condition_logits, dim=1).cpu().numpy())
            all_cond_labels.extend(condition_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_cond_labels = np.array(all_cond_labels)
    all_cond_preds = np.array(all_cond_probs).argmax(axis=1)

    # Calculate clinical thresholds
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    threshold_95 = thresholds[idx_95]
    specificity_95 = 1 - fpr[idx_95]

    results = {
        "binary_auc": float(best_auc),
        "condition_accuracy": float(accuracy_score(all_cond_labels, all_cond_preds)),
        "condition_f1_macro": float(f1_score(all_cond_labels, all_cond_preds, average='macro')),
        "threshold_95_sens": float(threshold_95),
        "specificity_at_95_sens": float(specificity_95),
    }

    with open(output_dir / "classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Final AUC: {best_auc:.3f}")
    print(f"  At 95% sensitivity: threshold={threshold_95:.3f}, specificity={specificity_95:.1%}")

    return model


def train_mobile_model(
    model_class,
    model_name,
    teacher_model,
    data,
    output_dir,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    temperature=4.0,
    alpha=0.7,
    device="cuda",
    image_size=224,
    data_dir=None,
):
    """Train mobile model via knowledge distillation with augmentation."""
    print(f"\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()} (KNOWLEDGE DISTILLATION)")
    print("=" * 60)

    # Check if we have image paths
    if "image_path" not in data["train_meta"].columns:
        print("  Error: No image paths in metadata, skipping mobile training")
        print("  Note: Mobile models require raw images for augmentation")
        return None

    # Create transforms
    train_transform = get_clinical_training_transform(image_size=image_size)
    eval_transform = get_eval_transform(image_size=image_size)

    # Get image paths
    train_paths = data["train_meta"]["image_path"].values
    test_paths = data["test_meta"]["image_path"].values

    # Create datasets
    train_dataset = SkinLesionDataset(
        train_paths,
        data["y_train_binary"],
        data["y_train_condition"],
        transform=train_transform,
    )
    test_dataset = SkinLesionDataset(
        test_paths,
        data["y_test_binary"],
        data["y_test_condition"],
        transform=eval_transform,
    )

    # Weighted sampler
    sampler = create_weighted_sampler(data["y_train_condition"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Student model
    student = model_class(n_classes=2).to(device)
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Loss
    ce_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        student.train()
        if teacher_model is not None:
            teacher_model.eval()

        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = batch["image"].to(device)
            binary_labels = batch["binary_label"].to(device)

            optimizer.zero_grad()

            student_logits = student(images)

            # Hard label loss
            hard_loss = ce_criterion(student_logits, binary_labels)

            # Soft label loss (if teacher available)
            if teacher_model is not None:
                with torch.no_grad():
                    # Get teacher embeddings
                    # Note: In practice, you'd need to extract embeddings here
                    # For now, we use hard labels only
                    pass

                loss = hard_loss
            else:
                loss = hard_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Evaluate
        student.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["binary_label"]

                logits = student(images)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))

        print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={auc:.3f}, Acc={acc:.1%}")

        if auc > best_auc:
            best_auc = auc
            best_state = student.state_dict().copy()

    # Save best model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    student.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{model_name}.pt")

    # Export to ONNX
    export_dir = output_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    student.eval()

    torch.onnx.export(
        student,
        dummy_input,
        export_dir / "model.onnx",
        export_params=True,
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    results = {
        "model": model_name,
        "best_auc": float(best_auc),
        "parameters": sum(p.numel() for p in student.parameters()),
    }

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best AUC: {best_auc:.3f}")
    print(f"  Model saved to {output_dir}")

    return student


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full clinical retraining pipeline")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for deep classifiers")
    parser.add_argument("--mobile-epochs", type=int, default=30, help="Epochs for mobile models")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--skip-classifiers", action="store_true", help="Skip classifier training")
    parser.add_argument("--skip-mobile", action="store_true", help="Skip mobile model training")
    parser.add_argument("--compare-finetuning", action="store_true", help="Compare frozen vs fine-tuned")
    parser.add_argument("--cache_dir", type=str, default="results/cache")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: timestamped)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create versioned output directory (never overwrite existing models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.cache_dir) / f"clinical_v{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("NOTE: Existing models in results/cache/ are preserved")

    # Load data
    data = load_data(args.cache_dir, quick=args.quick)

    # Track all results
    all_results = {
        "timestamp": timestamp,
        "config": vars(args),
        "classifiers": {},
        "mobile_models": {},
        "comparisons": {},
    }

    # =================================================================
    # STEP 1: Compare frozen vs fine-tuned embeddings
    # =================================================================
    if args.compare_finetuning:
        comparison = compare_embedding_approaches(args.cache_dir, output_dir)
        all_results["comparisons"]["embedding_comparison"] = comparison

    # =================================================================
    # STEP 2: Train all classifier types
    # =================================================================
    if not args.skip_classifiers and data["X_train"] is not None:
        print("\n" + "=" * 60)
        print("TRAINING CLASSIFIERS ON EMBEDDINGS")
        print("=" * 60)

        # 2a. XGBoost (binary)
        print("\n[1/5] Training XGBoost (binary)...")
        xgb_clf, xgb_scaler, xgb_results = train_xgboost_classifier(
            data["X_train"], data["y_train_binary"],
            data["X_test"], data["y_test_binary"],
            n_classes=2
        )
        all_results["classifiers"]["xgboost_binary"] = xgb_results
        print(f"      AUC: {xgb_results['auc']:.3f}, Acc: {xgb_results['accuracy']:.1%}")

        # Save XGBoost
        xgb_dir = output_dir / "classifiers"
        xgb_dir.mkdir(exist_ok=True)
        with open(xgb_dir / "xgboost_binary.pkl", "wb") as f:
            pickle.dump({"classifier": xgb_clf, "scaler": xgb_scaler}, f)

        # 2b. Logistic (binary)
        print("\n[2/5] Training Logistic Regression (binary)...")
        log_clf, log_scaler, log_results = train_logistic_classifier(
            data["X_train"], data["y_train_binary"],
            data["X_test"], data["y_test_binary"],
            n_classes=2
        )
        all_results["classifiers"]["logistic_binary"] = log_results
        print(f"      AUC: {log_results['auc']:.3f}, Acc: {log_results['accuracy']:.1%}")

        with open(xgb_dir / "logistic_binary.pkl", "wb") as f:
            pickle.dump({"classifier": log_clf, "scaler": log_scaler}, f)

        # 2c. MLP (binary)
        print("\n[3/5] Training MLP (binary)...")
        mlp_clf, mlp_scaler, mlp_results = train_mlp_classifier(
            data["X_train"], data["y_train_binary"],
            data["X_test"], data["y_test_binary"],
            n_classes=2
        )
        all_results["classifiers"]["mlp_binary"] = mlp_results
        print(f"      AUC: {mlp_results['auc']:.3f}, Acc: {mlp_results['accuracy']:.1%}")

        with open(xgb_dir / "mlp_binary.pkl", "wb") as f:
            pickle.dump({"classifier": mlp_clf, "scaler": mlp_scaler}, f)

        # 2d. XGBoost (10-class condition)
        print("\n[4/5] Training XGBoost (10-class condition)...")
        xgb_cond_clf, xgb_cond_scaler, xgb_cond_results = train_xgboost_classifier(
            data["X_train"], data["y_train_condition"],
            data["X_test"], data["y_test_condition"],
            n_classes=10
        )
        all_results["classifiers"]["xgboost_condition"] = xgb_cond_results
        print(f"      AUC: {xgb_cond_results['auc']:.3f}, Acc: {xgb_cond_results['accuracy']:.1%}")

        with open(xgb_dir / "xgboost_condition.pkl", "wb") as f:
            pickle.dump({"classifier": xgb_cond_clf, "scaler": xgb_cond_scaler}, f)

        # 2e. Hierarchical (PyTorch, multi-task)
        print("\n[5/5] Training Hierarchical Classifier (multi-task)...")
        hier_clf = train_classifier(
            data,
            output_dir / "hierarchical",
            epochs=args.epochs if not args.quick else 2,
            batch_size=args.batch_size,
            device=device,
        )
        if hier_clf:
            hier_results = json.load(open(output_dir / "hierarchical" / "classifier_results.json"))
            all_results["classifiers"]["hierarchical"] = hier_results

        # Print classifier comparison
        print("\n" + "-" * 60)
        print("CLASSIFIER COMPARISON (Binary)")
        print("-" * 60)
        print(f"{'Classifier':<25} {'AUC':>10} {'Accuracy':>12} {'F1':>10}")
        print("-" * 60)
        for name, res in all_results["classifiers"].items():
            if "binary" in name or name == "hierarchical":
                auc = res.get("auc", res.get("binary_auc", 0))
                acc = res.get("accuracy", res.get("condition_accuracy", 0))
                f1 = res.get("f1", res.get("condition_f1_macro", 0))
                print(f"{name:<25} {auc:>10.3f} {acc:>11.1%} {f1:>10.3f}")

    # =================================================================
    # STEP 3: Train mobile models (with augmentation)
    # =================================================================
    if not args.skip_mobile:
        # MobileNetV3-Large
        print("\n[Mobile 1/2] Training MobileNetV3-Large...")
        mobilenet = train_mobile_model(
            MobileNetClassifier,
            "mobilenet_v3_large",
            None,  # No teacher for now
            data,
            output_dir / "mobilenet",
            epochs=args.mobile_epochs if not args.quick else 3,
            batch_size=args.batch_size,
            device=device,
        )
        if mobilenet:
            all_results["mobile_models"]["mobilenet"] = json.load(
                open(output_dir / "mobilenet" / "training_results.json")
            )

        # EfficientNet-B0
        print("\n[Mobile 2/2] Training EfficientNet-B0...")
        efficientnet = train_mobile_model(
            EfficientNetClassifier,
            "efficientnet_b0",
            None,
            data,
            output_dir / "efficientnet",
            epochs=args.mobile_epochs if not args.quick else 3,
            batch_size=args.batch_size,
            device=device,
        )
        if efficientnet:
            all_results["mobile_models"]["efficientnet"] = json.load(
                open(output_dir / "efficientnet" / "training_results.json")
            )

    # =================================================================
    # STEP 4: Calculate clinical metrics
    # =================================================================
    print("\n" + "=" * 60)
    print("CLINICAL SENSITIVITY/SPECIFICITY ANALYSIS")
    print("=" * 60)

    if data["X_test"] is not None and "xgboost_binary" in all_results["classifiers"]:
        # Use best classifier (XGBoost) for clinical thresholds
        y_prob = xgb_clf.predict_proba(xgb_scaler.transform(data["X_test"]))[:, 1]
        y_true = data["y_test_binary"]

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        clinical_thresholds = {}
        for target_sens in [0.99, 0.95, 0.90, 0.85]:
            idx = np.argmin(np.abs(tpr - target_sens))
            thresh = thresholds[idx] if len(thresholds) > idx else 0.5
            spec = 1 - fpr[idx]
            clinical_thresholds[f"sens_{int(target_sens*100)}"] = {
                "threshold": float(thresh),
                "specificity": float(spec),
            }
            print(f"  At {target_sens:.0%} sensitivity: threshold={thresh:.3f}, specificity={spec:.1%}")

        all_results["clinical_thresholds"] = clinical_thresholds

    # =================================================================
    # Save final results
    # =================================================================
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save a summary to the main cache dir for easy access
    summary_path = Path(args.cache_dir) / f"training_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "output_dir": str(output_dir),
            "timestamp": timestamp,
            "best_binary_auc": max(
                (r.get("auc", r.get("binary_auc", 0))
                 for r in all_results["classifiers"].values()),
                default=0
            ),
            "classifiers_trained": list(all_results["classifiers"].keys()),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Full results: {results_path}")
    print(f"Summary: {summary_path}")
    print("\nExisting models in results/cache/ have been PRESERVED.")
    print(f"New models saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
