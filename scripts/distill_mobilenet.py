#!/usr/bin/env python3
"""Knowledge distillation from fine-tuned SigLIP to MobileNetV3-Large.

This script trains a lightweight MobileNetV3-Large model to mimic the predictions
of the larger SigLIP teacher model, enabling deployment on mobile devices.

Usage:
    python scripts/distill_mobilenet.py --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_multi_dataset


class SkinLesionDataset(Dataset):
    """Dataset that loads images and labels for distillation training."""

    def __init__(self, image_paths, labels, transform=None, teacher_logits=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        if self.teacher_logits is not None:
            return image, label, self.teacher_logits[idx]
        return image, label


class MobileNetClassifier(nn.Module):
    """MobileNetV3-Large with custom classification head for skin lesion triage."""

    def __init__(self, n_classes=2, dropout=0.2, pretrained=True):
        super().__init__()

        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class DistillationLoss(nn.Module):
    """Combined loss: soft target (KL div) + hard target (CE)."""

    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)

        hard_loss = self.ce_loss(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def load_teacher_model(teacher_dir, device):
    """Load the fine-tuned SigLIP teacher model."""
    from src.model.deep_classifier import EndToEndSigLIP
    from transformers import AutoImageProcessor

    config_path = Path(teacher_dir) / "config.json"
    weights_path = Path(teacher_dir) / "model_state.pt"

    with open(config_path) as f:
        config = json.load(f)

    model = EndToEndSigLIP(
        model_name=config["model_name"],
        hidden_dim=config["hidden_dim"],
        n_classes=config["n_classes"],
        dropout=config["dropout"],
        unfreeze_layers=config["unfreeze_layers"],
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(config["model_name"])

    return model, processor


def compute_teacher_logits(model, processor, image_paths, batch_size, device):
    """Pre-compute teacher logits for all training images."""
    print("Pre-computing teacher logits...")

    all_logits = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            logits = model(pixel_values)

        all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)


def get_transforms(image_size=224):
    """Get train and validation transforms for MobileNet."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        if len(batch) == 3:
            images, labels, teacher_logits = batch
            teacher_logits = teacher_logits.to(device)
        else:
            images, labels = batch
            teacher_logits = None

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        student_logits = model(images)

        if teacher_logits is not None:
            loss = criterion(student_logits, teacher_logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(student_logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = student_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in val_loader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_malignant = f1_score(all_labels, all_preds, pos_label=1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_malignant': f1_malignant,
        'auc': auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation to MobileNetV3")
    parser.add_argument("--teacher_dir", type=str, default="models/finetuned_siglip")
    parser.add_argument("--output_dir", type=str, default="models/mobilenet_distilled")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--no_teacher", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data_dir = Path("data")
    samples = load_multi_dataset(data_dir)

    all_paths = [str(s.image_path) for s in samples]
    all_labels = np.array([s.label for s in samples])

    print(f"Total samples: {len(all_paths)}")
    print(f"Class distribution: {np.bincount(all_labels)}")

    # Train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(all_paths)),
        test_size=0.15,
        stratify=all_labels,
        random_state=42
    )

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = all_labels[train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = all_labels[val_idx]

    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")

    # Load teacher and compute soft labels
    teacher_logits = None
    if not args.no_teacher:
        teacher_dir = Path(args.teacher_dir)
        if teacher_dir.exists():
            print(f"\nLoading teacher model from {teacher_dir}...")
            teacher_model, processor = load_teacher_model(str(teacher_dir), device)

            teacher_logits = compute_teacher_logits(
                teacher_model, processor, train_paths, batch_size=16, device=device
            )
            print(f"Teacher logits shape: {teacher_logits.shape}")
        else:
            print(f"Warning: Teacher model not found at {teacher_dir}")
            print("Training without distillation")

    # Create datasets
    train_transform, val_transform = get_transforms(args.image_size)

    train_dataset = SkinLesionDataset(
        train_paths, train_labels,
        transform=train_transform,
        teacher_logits=teacher_logits
    )
    val_dataset = SkinLesionDataset(
        val_paths, val_labels,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    print("\nCreating MobileNetV3-Large...")
    model = MobileNetClassifier(n_classes=2, dropout=0.2, pretrained=True)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    if teacher_logits is not None:
        criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
        print(f"\nUsing distillation loss (T={args.temperature}, alpha={args.alpha})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard cross-entropy loss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_f1 = 0
    best_state = None
    history = []

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")
        print(f"  Val: acc={val_metrics['accuracy']:.3f}, f1_macro={val_metrics['f1_macro']:.3f}, "
              f"f1_mal={val_metrics['f1_malignant']:.3f}, auc={val_metrics['auc']:.3f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            **val_metrics
        })

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best model! F1 macro: {best_f1:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_metrics = evaluate(model, val_loader, device)
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  F1 Macro: {final_metrics['f1_macro']:.4f}")
    print(f"  F1 Malignant: {final_metrics['f1_malignant']:.4f}")
    print(f"  AUC: {final_metrics['auc']:.4f}")

    # Save model
    print(f"\nSaving model to {output_dir}...")
    torch.save(model.state_dict(), output_dir / "mobilenet_v3_large.pt")
    torch.save(model, output_dir / "mobilenet_v3_large_full.pt")

    config = {
        'model_type': 'MobileNetV3-Large',
        'n_classes': 2,
        'image_size': args.image_size,
        'input_normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'distillation': {
            'teacher_dir': str(args.teacher_dir) if not args.no_teacher else None,
            'temperature': args.temperature,
            'alpha': args.alpha,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
        'final_metrics': final_metrics,
        'total_parameters': total_params,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nDistillation complete!")
    print(f"Model saved to: {output_dir / 'mobilenet_v3_large.pt'}")

    model_size_mb = (output_dir / "mobilenet_v3_large.pt").stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
