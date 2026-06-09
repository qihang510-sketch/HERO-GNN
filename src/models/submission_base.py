from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class SubmissionFeatureModel:
    """Small interface wrapper for reproduced submission baselines.

    The heavy lifting for these baselines is in src.training.submission, where
    model-specific graph features are built and trained with a real classifier.
    These wrappers keep a uniform model API for configs and downstream tooling.
    """

    implementation_source = "reproduced"

    def __init__(self, name: str, hidden_dim: int = 64, **kwargs: Any) -> None:
        self.name = name
        self.hidden_dim = int(hidden_dim)
        self.kwargs = dict(kwargs)
        self.classifier: Any | None = None

    def forward(self, batch_or_graph: Any) -> Any:
        if isinstance(batch_or_graph, dict) and "features" in batch_or_graph:
            return batch_or_graph["features"]
        if hasattr(batch_or_graph, "features"):
            return batch_or_graph.features
        return batch_or_graph

    def compute_loss(self, logits: Any, labels: Any, **_: Any) -> Any:
        if torch is not None and isinstance(logits, torch.Tensor):
            target = labels.float() if hasattr(labels, "float") else torch.as_tensor(labels, dtype=torch.float32)
            return nn.functional.binary_cross_entropy_with_logits(logits.float().view(-1), target.view(-1))
        scores = np.asarray(logits, dtype=np.float32).reshape(-1)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        eps = 1e-7
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        return float(-np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)))

    def predict(self, batch_or_graph: Any) -> np.ndarray:
        features = np.asarray(self.forward(batch_or_graph), dtype=np.float32)
        if self.classifier is None:
            raise RuntimeError(f"{self.name} classifier has not been fitted.")
        probabilities = self.classifier.predict_proba(features)
        if probabilities.shape[1] == 1:
            label = int(self.classifier.classes_[0])
            return np.ones(features.shape[0], dtype=np.float32) if label == 1 else np.zeros(features.shape[0], dtype=np.float32)
        class_to_col = {int(label): col for col, label in enumerate(self.classifier.classes_)}
        return probabilities[:, class_to_col.get(1, 0)].astype(np.float32)

    def export_embeddings(self, batch_or_graph: Any) -> np.ndarray:
        return np.asarray(self.forward(batch_or_graph), dtype=np.float32)
