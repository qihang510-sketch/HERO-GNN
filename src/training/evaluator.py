from __future__ import annotations

import numpy as np

from src.utils.metrics import macro_f1, precision_at_k, recall_at_k, safe_auprc, safe_auroc


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 100,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    preds = (scores >= 0.5).astype(np.int64)
    return {
        "macro_f1": macro_f1(labels, preds),
        "auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        f"precision_at_{k}": precision_at_k(labels, scores, k),
        f"recall_at_{k}": recall_at_k(labels, scores, k),
    }
