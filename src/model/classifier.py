"""Classifiers for MedSigLIP embeddings — sklearn for speed, PyTorch optional."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _make_xgboost_clf(n_classes: int = 2):
    """Create an XGBoost classifier with tuned hyperparameters for SigLIP embeddings."""
    from xgboost import XGBClassifier
    return XGBClassifier(
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
        tree_method="hist",  # fast CPU, also supports GPU via "gpu_hist"
        random_state=42,
        n_jobs=-1,
    )


class SklearnClassifier:
    """Fast, lightweight classifier using sklearn (or XGBoost). Recommended for hackathons."""

    def __init__(self, classifier_type: str = "logistic", n_classes: int = 2):
        if classifier_type == "logistic":
            clf = LogisticRegression(max_iter=1000)
        elif classifier_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)
        elif classifier_type == "xgboost":
            clf = _make_xgboost_clf(n_classes=n_classes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def fit(self, embeddings, labels, sample_weight=None):
        """Train on pre-extracted embeddings.

        Args:
            embeddings: (N, D) array or tensor
            labels: (N,) array
            sample_weight: optional (N,) per-sample weights for domain balancing
        """
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        fit_params = {}
        if sample_weight is not None:
            fit_params["classifier__sample_weight"] = np.asarray(sample_weight)
        self.pipeline.fit(X, labels, **fit_params)
        return self

    def predict(self, embeddings):
        """Predict class labels."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.pipeline.predict(X)

    def predict_proba(self, embeddings):
        """Predict class probabilities."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.pipeline.predict_proba(X)

    def predict_triage(self, embeddings, triage_system):
        """Predict with triage assessment.

        Args:
            embeddings: (N, D) array or tensor
            triage_system: TriageSystem instance

        Returns:
            List of TriageResult objects
        """
        proba = self.predict_proba(embeddings)
        # Malignant probability is column 1
        mal_proba = proba[:, 1] if proba.ndim == 2 else proba
        return [triage_system.assess(p) for p in mal_proba]

    def score(self, embeddings, labels):
        """Compute accuracy."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.pipeline.score(X, labels)


class ZeroShotClassifier:
    """Zero-shot classification using image-text similarity. No training needed."""

    def __init__(self, extractor, class_descriptions: list[str]):
        self.extractor = extractor
        self.class_descriptions = class_descriptions
        self.class_names = [d.split()[-1] for d in class_descriptions]  # Simple name extraction
        self.text_embeddings = None

    def _encode_classes(self):
        if self.text_embeddings is None:
            self.text_embeddings = self.extractor.extract_text(self.class_descriptions)
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
        return self.text_embeddings

    def predict(self, image_embeddings):
        """Predict class from pre-extracted image embeddings."""
        text_emb = self._encode_classes()
        img_emb = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        similarity = img_emb @ text_emb.T
        return similarity.argmax(dim=-1).numpy()

    def predict_with_scores(self, image_embeddings):
        """Return predictions and similarity scores."""
        text_emb = self._encode_classes()
        img_emb = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        similarity = img_emb @ text_emb.T
        predictions = similarity.argmax(dim=-1).numpy()
        scores = similarity.numpy()
        return predictions, scores
