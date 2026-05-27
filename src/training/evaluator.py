from __future__ import annotations

import numpy as np

from src.utils.metrics import confusion_counts, macro_f1, precision_at_k, recall_at_k, safe_auprc, safe_auroc

THRESHOLD_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    k: int = 100,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    preds = (scores >= threshold).astype(np.int64)
    confusion = confusion_counts(labels, preds)
    return {
        "macro_f1": macro_f1(labels, preds),
        "auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        f"precision_at_{k}": precision_at_k(labels, scores, k),
        f"recall_at_{k}": recall_at_k(labels, scores, k),
        **confusion,
        "pred_positive_rate": float(np.mean(preds)) if preds.size else 0.0,
        "best_threshold": float(threshold),
    }


def tune_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0:
        return 0.5

    best_threshold = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (scores >= float(threshold)).astype(np.int64)
        score = macro_f1(labels, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def split_label_stats(labels_by_split: dict[str, np.ndarray]) -> dict[str, float | int]:
    stats: dict[str, float | int] = {}
    for split_name, labels in labels_by_split.items():
        labels = np.asarray(labels, dtype=np.int64)
        total = int(labels.size)
        positive = int(np.sum(labels == 1)) if total else 0
        stats[f"positive_rate_{split_name}"] = float(positive / total) if total else 0.0
        stats[f"positive_count_{split_name}"] = positive
        stats[f"total_count_{split_name}"] = total
    return stats
