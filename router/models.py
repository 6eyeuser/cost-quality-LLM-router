"""Custom estimators. Kept in an importable module so joblib artifacts can be loaded by the gateway."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SimilarityWeightedRouter(ClassifierMixin, BaseEstimator):
    """RouteLLM-style similarity-weighted (SW) routing, re-implemented here (NOT the official package).

    P(cheap ok | q) is a softmax-similarity-weighted vote of the labels of the k nearest TRAINING
    prompts (cosine similarity on embeddings), shrunk toward the global base rate:

        w_i = exp(sim(q, x_i) / temperature),   p = (sum_i w_i y_i + a * prior) / (sum_i w_i + a)

    A small temperature makes it behave like 1-NN; a large one like the global mean.
    """

    def __init__(self, k: int = 50, temperature: float = 0.05, prior_strength: float = 0.5):
        self.k = k
        self.temperature = temperature
        self.prior_strength = prior_strength

    @staticmethod
    def _unit(X):
        X = np.asarray(X, dtype=np.float32)
        return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.X_ = self._unit(X)
        self.y_ = y.astype(np.float32)
        self.prior_ = float(y.mean())
        return self

    def predict_proba(self, X):
        Q = self._unit(X)
        sims = Q @ self.X_.T                                   # (q, n)
        k = min(self.k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(sims, idx, axis=1)
        w = np.exp((top - top.max(axis=1, keepdims=True)) / self.temperature)
        # weights are relative to the best neighbour; the prior counts as a pseudo-neighbour of weight a
        p1 = ((w * self.y_[idx]).sum(1) + self.prior_strength * self.prior_)/ (w.sum(1) + self.prior_strength)
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
