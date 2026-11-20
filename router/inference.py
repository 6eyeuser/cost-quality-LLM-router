"""Router artifact: one joblib file holding the fitted model + everything serving needs."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import build_features, make_embedder


def save_artifact(path: str | Path, **fields) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fields, path)
    return path


class LoadedRouter:
    def __init__(self, artifact: dict, embedder=None):
        self.artifact = artifact
        self.model = artifact["model"]
        self.mode = artifact["feature_mode"]
        self.threshold = float(artifact["threshold"])
        self.name = artifact.get("name", "router")
        self.embedder = embedder
        if self.embedder is None and self.mode != "rules":
            self.embedder = make_embedder(artifact.get("embedder"))

    def p_cheap(self, prompts: list[str]) -> np.ndarray:
        X = build_features(prompts, self.embedder, self.mode)
        return self.model.predict_proba(X)[:, 1]

    def p_cheap_one(self, prompt: str) -> float:
        return float(self.p_cheap([prompt])[0])

    def warmup(self) -> None:
        self.p_cheap(["warm up the model"])


def load_router(path: str | Path, embedder=None) -> LoadedRouter:
    return LoadedRouter(joblib.load(path), embedder=embedder)
