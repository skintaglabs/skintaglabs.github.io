"""Fine-tuned deep learning classifier on SigLIP embeddings.

Two modes:
  1. Head-only (default): 2-layer MLP on frozen pre-extracted embeddings.
     Fast, works with cached embeddings.
  2. End-to-end: Unfreezes last N layers of SigLIP backbone and fine-tunes
     jointly with the classification head. Requires raw images, GPU recommended.

Both modes implement the same fit/predict/predict_proba/score interface.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


class DeepClassificationHead(nn.Module):
    """2-layer MLP classification head: embedding_dim -> hidden -> n_classes."""

    def __init__(self, embedding_dim=1152, hidden_dim=256, n_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class EndToEndSigLIP(nn.Module):
    """SigLIP backbone with trainable classification head.

    Optionally unfreezes the last N transformer layers for fine-tuning.
    """

    def __init__(self, model_name, hidden_dim=256, n_classes=2, dropout=0.3, unfreeze_layers=0):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        embedding_dim = self.backbone.config.vision_config.hidden_size
        self.head = DeepClassificationHead(embedding_dim, hidden_dim, n_classes, dropout)

        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N vision encoder layers
        if unfreeze_layers > 0:
            vision_layers = self.backbone.vision_model.encoder.layers
            for layer in vision_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, pixel_values):
        features = self.backbone.get_image_features(pixel_values=pixel_values)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return self.head(features)


class FineTunableSigLIP(nn.Module):
    """SigLIP with unfrozen last N transformer layers for fine-tuning.

    V2 architecture with 3-layer MLP head (LayerNorm + GELU).
    Used by the full_retraining_pipeline for production models.
    """

    def __init__(
        self,
        model_name="google/siglip-so400m-patch14-384",
        hidden_dim=512,
        n_classes=2,
        dropout=0.3,
        unfreeze_layers=4,
    ):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.backbone.config.vision_config.hidden_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0:
            vision_layers = self.backbone.vision_model.encoder.layers
            for layer in vision_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone.vision_model(pixel_values=pixel_values)
        features = outputs.pooler_output
        return self.head(features)

    def extract_embeddings(self, pixel_values):
        """Extract embeddings without classification head."""
        with torch.no_grad():
            outputs = self.backbone.vision_model(pixel_values=pixel_values)
            return outputs.pooler_output


class DeepClassifier:
    """PyTorch-based deep classifier on pre-extracted embeddings.

    Matches the fit/predict/predict_proba/score interface used by
    SklearnClassifier and baseline models.
    """

    def __init__(
        self,
        embedding_dim: int = 1152,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 8,
        device: str = None,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.training_history = []

    def _build_model(self):
        self.model = DeepClassificationHead(
            self.embedding_dim, self.hidden_dim, self.n_classes, self.dropout
        ).to(self.device)

    def fit(self, embeddings, labels, sample_weight=None, val_embeddings=None, val_labels=None):
        """Train the classification head on pre-extracted embeddings.

        Args:
            embeddings: numpy array or torch tensor (N, D)
            labels: numpy array (N,)
            sample_weight: optional per-sample weights (N,)
            val_embeddings: optional validation embeddings for early stopping
            val_labels: optional validation labels
        """
        X = self._to_tensor(embeddings).float()
        y = torch.tensor(np.asarray(labels), dtype=torch.long)

        self._build_model()

        # Class-weighted cross-entropy loss
        class_counts = np.bincount(np.asarray(labels), minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        ce_weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Prepare sample weights
        sw = None
        if sample_weight is not None:
            sw = torch.tensor(np.asarray(sample_weight), dtype=torch.float32)

        # Validation set for early stopping
        has_val = val_embeddings is not None and val_labels is not None
        if has_val:
            X_val = self._to_tensor(val_embeddings).float().to(self.device)
            y_val = torch.tensor(np.asarray(val_labels), dtype=torch.long).to(self.device)

        # Train/val split from training data if no explicit val set
        if not has_val:
            n = len(X)
            perm = torch.randperm(n)
            val_size = max(1, int(0.15 * n))
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
            X_val = X[val_idx].to(self.device)
            y_val = y[val_idx].to(self.device)
            X_train = X[train_idx]
            y_train = y[train_idx]
            sw_train = sw[train_idx] if sw is not None else None
        else:
            X_train = X
            y_train = y
            sw_train = sw

        dataset = TensorDataset(X_train, y_train) if sw_train is None else TensorDataset(X_train, y_train, sw_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.training_history = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in loader:
                if sw_train is not None:
                    bx, by, bw = batch
                else:
                    bx, by = batch
                    bw = None

                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)

                if bw is not None:
                    bw = bw.to(self.device)
                    loss = (loss * bw).mean()
                else:
                    loss = loss.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)

            scheduler.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = nn.CrossEntropyLoss()(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

            self.training_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss / len(X_train),
                'val_loss': val_loss,
                'val_acc': val_acc,
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        return self

    def predict(self, embeddings):
        X = self._to_tensor(embeddings).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
        return logits.argmax(1).cpu().numpy()

    def predict_proba(self, embeddings):
        X = self._to_tensor(embeddings).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def score(self, embeddings, labels):
        preds = self.predict(embeddings)
        labels = np.asarray(labels)
        return (preds == labels).mean()

    def save_head(self, path: str):
        """Save just the classification head weights (lightweight, ~1MB)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_head(self, path: str):
        """Load classification head weights."""
        self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        return self

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(np.asarray(x))


class EndToEndClassifier:
    """Fine-tunes SigLIP backbone (last N layers) + classification head jointly.

    Use this when you want to adapt the vision encoder to dermatology images.
    Requires raw PIL images as input (not pre-extracted embeddings).
    GPU strongly recommended.

    After training, call export_for_inference() to save the full model
    for deployment in the web app.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.3,
        unfreeze_layers: int = 4,
        lr_head: float = 1e-3,
        lr_backbone: float = 1e-5,
        epochs: int = 20,
        batch_size: int = 8,
        patience: int = 5,
        device: str = None,
    ):
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.unfreeze_layers = unfreeze_layers
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.training_history = []

    def _build_model(self):
        from transformers import AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = EndToEndSigLIP(
            self.model_name, self.hidden_dim, self.n_classes,
            self.dropout, self.unfreeze_layers
        ).to(self.device)

    def _prepare_images(self, images):
        """Convert PIL images to pixel_values tensor."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    def fit(self, images, labels, sample_weight=None, val_images=None, val_labels=None):
        """Fine-tune on raw PIL images.

        Args:
            images: list of PIL Images
            labels: array-like of int labels
            sample_weight: optional per-sample weights
            val_images: optional validation images
            val_labels: optional validation labels
        """
        self._build_model()

        labels = np.asarray(labels)
        n = len(images)

        # Class-weighted loss
        class_counts = np.bincount(labels, minlength=self.n_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.n_classes
        ce_weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

        # Separate learning rates for head vs backbone
        head_params = list(self.model.head.parameters())
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": self.lr_head},
            {"params": backbone_params, "lr": self.lr_backbone},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Train/val split
        has_val = val_images is not None and val_labels is not None
        if not has_val:
            perm = np.random.permutation(n)
            val_size = max(1, int(0.15 * n))
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
            val_images_split = [images[i] for i in val_idx]
            val_labels_split = labels[val_idx]
            train_images = [images[i] for i in train_idx]
            train_labels = labels[train_idx]
            train_weights = sample_weight[train_idx] if sample_weight is not None else None
        else:
            train_images = images
            train_labels = labels
            train_weights = sample_weight
            val_images_split = val_images
            val_labels_split = np.asarray(val_labels)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        n_train = len(train_images)

        self.training_history = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            perm = np.random.permutation(n_train)

            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_imgs = [train_images[i] for i in idx]
                batch_labels = torch.tensor(train_labels[idx], dtype=torch.long).to(self.device)

                pixel_values = self._prepare_images(batch_imgs).to(self.device)

                optimizer.zero_grad()
                logits = self.model(pixel_values)
                loss = criterion(logits, batch_labels)

                if train_weights is not None:
                    bw = torch.tensor(train_weights[idx], dtype=torch.float32).to(self.device)
                    loss = (loss * bw).mean()
                else:
                    loss = loss.mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for start in range(0, len(val_images_split), self.batch_size):
                    batch_imgs = val_images_split[start:start + self.batch_size]
                    batch_labels = torch.tensor(
                        val_labels_split[start:start + self.batch_size], dtype=torch.long
                    ).to(self.device)
                    pixel_values = self._prepare_images(batch_imgs).to(self.device)
                    logits = self.model(pixel_values)
                    val_loss += nn.CrossEntropyLoss()(logits, batch_labels).item() * len(batch_labels)
                    val_correct += (logits.argmax(1) == batch_labels).sum().item()

            val_loss /= len(val_images_split)
            val_acc = val_correct / len(val_images_split)

            self.training_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss / n_train,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })
            print(f"  Epoch {epoch}: train_loss={epoch_loss/n_train:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, images):
        """Predict from raw PIL images."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size]
                pixel_values = self._prepare_images(batch).to(self.device)
                logits = self.model(pixel_values)
                all_preds.append(logits.argmax(1).cpu())
        return torch.cat(all_preds).numpy()

    def predict_proba(self, images):
        """Predict probabilities from raw PIL images."""
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size]
                pixel_values = self._prepare_images(batch).to(self.device)
                logits = self.model(pixel_values)
                all_proba.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(all_proba).numpy()

    def score(self, images, labels):
        preds = self.predict(images)
        labels = np.asarray(labels)
        return (preds == labels).mean()

    def export_for_inference(self, save_dir: str):
        """Export the fine-tuned model for deployment.

        Saves:
          - model_state.pt: Full model state dict
          - head_state.pt: Just the classification head (for use with EmbeddingExtractor)
          - config.json: Model configuration

        The app can then load the fine-tuned backbone for better accuracy,
        or use just the head with the standard SigLIP extractor.
        """
        import json
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Full model
        torch.save(self.model.state_dict(), save_dir / "model_state.pt")

        # Head only (lightweight)
        torch.save(self.model.head.state_dict(), save_dir / "head_state.pt")

        # Config
        config = {
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "n_classes": self.n_classes,
            "dropout": self.dropout,
            "unfreeze_layers": self.unfreeze_layers,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Exported to {save_dir}/")
        print(f"  model_state.pt: Full fine-tuned model")
        print(f"  head_state.pt: Classification head only (~1MB)")
        print(f"  config.json: Model configuration")

    @classmethod
    def load_for_inference(cls, save_dir: str, device: str = None):
        """Load a previously exported fine-tuned model.

        Detects v2 models (siglip_finetuned.pt + FineTunableSigLIP architecture)
        vs v1 models (model_state.pt + EndToEndSigLIP architecture).
        """
        import json
        save_dir = Path(save_dir)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Check for v2 model directory (may be nested under siglip_finetuned/)
        v2_dir = save_dir / "siglip_finetuned"
        if (v2_dir / "siglip_finetuned.pt").exists():
            save_dir = v2_dir

        is_v2 = (save_dir / "siglip_finetuned.pt").exists()

        with open(save_dir / "config.json") as f:
            config = json.load(f)

        obj = cls(
            model_name=config["model_name"],
            hidden_dim=config.get("hidden_dim", 512),
            n_classes=config.get("n_classes", 2),
            dropout=config.get("dropout", 0.3),
            unfreeze_layers=config.get("unfreeze_layers", 4),
            device=device,
        )

        if is_v2:
            from transformers import AutoImageProcessor
            obj.processor = AutoImageProcessor.from_pretrained(config["model_name"])
            obj.model = FineTunableSigLIP(
                model_name=config["model_name"],
                hidden_dim=config.get("hidden_dim", 512),
                n_classes=config.get("n_classes", 2),
                dropout=config.get("dropout", 0.3),
                unfreeze_layers=config.get("unfreeze_layers", 4),
            ).to(device)
            state = torch.load(save_dir / "siglip_finetuned.pt", map_location=device)
            obj.model.load_state_dict(state)
            print(f"Loaded v2 FineTunableSigLIP model from {save_dir}")
        else:
            obj._build_model()
            state = torch.load(save_dir / "model_state.pt", map_location=device)
            obj.model.load_state_dict(state)
            print(f"Loaded v1 EndToEndSigLIP model from {save_dir}")

        obj.model.to(device)
        obj.model.eval()
        return obj

    def extract_embeddings(self, images):
        """Extract embeddings from PIL images using the fine-tuned backbone.

        Only available when the underlying model is FineTunableSigLIP.
        """
        if not isinstance(self.model, FineTunableSigLIP):
            return None
        pixel_values = self._prepare_images(images).to(self.device)
        return self.model.extract_embeddings(pixel_values)
