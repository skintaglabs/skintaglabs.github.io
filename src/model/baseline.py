"""Naive baseline models for comparison.

Required for three-model comparison: naive baseline, classical ML, deep learning.
Both implement the same fit/predict/predict_proba/score interface as SklearnClassifier.
"""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

import numpy as np


class MajorityClassBaseline:
    """Always predicts the majority class (benign). Simplest possible baseline."""

    def __init__(self):
        self.majority_class = 0
        self.class_priors = None

    def fit(self, embeddings, labels, sample_weight=None):
        labels = np.asarray(labels)
        unique, counts = np.unique(labels, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        total = counts.sum()
        self.class_priors = {c: n / total for c, n in zip(unique, counts)}
        return self

    def predict(self, embeddings):
        n = len(embeddings) if hasattr(embeddings, '__len__') else embeddings.shape[0]
        return np.full(n, self.majority_class, dtype=int)

    def predict_proba(self, embeddings):
        n = len(embeddings) if hasattr(embeddings, '__len__') else embeddings.shape[0]
        n_classes = len(self.class_priors) if self.class_priors else 2
        proba = np.zeros((n, n_classes))
        proba[:, self.majority_class] = 1.0
        return proba

    def score(self, embeddings, labels):
        labels = np.asarray(labels)
        preds = self.predict(embeddings)
        return (preds == labels).mean()


class RandomWeightedBaseline:
    """Predicts proportional to class distribution. Better than majority for AUC."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.class_priors = None
        self.classes = None

    def fit(self, embeddings, labels, sample_weight=None):
        labels = np.asarray(labels)
        unique, counts = np.unique(labels, return_counts=True)
        self.classes = unique
        total = counts.sum()
        self.class_priors = counts / total
        return self

    def predict(self, embeddings):
        n = len(embeddings) if hasattr(embeddings, '__len__') else embeddings.shape[0]
        return self.rng.choice(self.classes, size=n, p=self.class_priors)

    def predict_proba(self, embeddings):
        n = len(embeddings) if hasattr(embeddings, '__len__') else embeddings.shape[0]
        n_classes = len(self.classes)
        proba = np.tile(self.class_priors, (n, 1))
        return proba

    def score(self, embeddings, labels):
        labels = np.asarray(labels)
        preds = self.predict(embeddings)
        return (preds == labels).mean()
