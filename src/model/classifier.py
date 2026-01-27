"""Classifiers for MedSigLIP embeddings — sklearn for speed, PyTorch optional."""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SklearnClassifier:
    """Fast, lightweight classifier using sklearn. Recommended for hackathons."""

    def __init__(self, classifier_type: str = "logistic"):
        if classifier_type == "logistic":
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif classifier_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def fit(self, embeddings, labels):
        """Train on pre-extracted embeddings."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        self.pipeline.fit(X, labels)
        return self

    def predict(self, embeddings):
        """Predict class labels."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.pipeline.predict(X)

    def predict_proba(self, embeddings):
        """Predict class probabilities."""
        X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        return self.pipeline.predict_proba(X)

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
