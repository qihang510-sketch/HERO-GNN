from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(f1_score(labels, preds, average="macro", zero_division=0))


def confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    if labels.size == 0:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if labels.size == 0:
        return 0.0
    top = _top_k(labels, scores, k)
    return float(np.mean(labels[top])) if top.size else 0.0


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    positives = float(np.sum(labels == 1))
    if labels.size == 0 or positives == 0:
        return 0.0
    top = _top_k(labels, scores, k)
    return float(np.sum(labels[top] == 1) / positives) if top.size else 0.0


def _top_k(labels: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, labels.size)
    if k == 0:
        return np.array([], dtype=np.int64)
    return np.argsort(scores)[::-1][:k]
