from __future__ import annotations

import numpy as np

from src.utils.metrics import confusion_counts, macro_f1, precision_at_k, recall_at_k, safe_auprc, safe_auroc

THRESHOLD_SEARCH_MODE = "dense+quantile"
DENSE_THRESHOLD_GRID = np.round(np.arange(0.01, 0.991, 0.01), 2)


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
    thresholds: np.ndarray | None = None,
    return_info: bool = False,
) -> float | dict[str, float | str]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if labels.size == 0 or scores.size == 0:
        info = {
            "best_threshold": 0.5,
            "threshold_search_mode": THRESHOLD_SEARCH_MODE,
            "val_best_macro_f1": 0.0,
            "val_pred_positive_rate_at_best_threshold": 0.0,
        }
        return info if return_info else float(info["best_threshold"])

    candidate_thresholds = _threshold_candidates(scores, thresholds)
    positive_rate = float(np.mean(labels == 1))
    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    best_pred_rate = 0.0
    best_rate_gap = float("inf")
    for threshold in candidate_thresholds:
        preds = (scores >= float(threshold)).astype(np.int64)
        score = macro_f1(labels, preds)
        pred_rate = float(np.mean(preds)) if preds.size else 0.0
        rate_gap = abs(pred_rate - positive_rate)
        if score > best_f1 + 1e-12 or (abs(score - best_f1) <= 1e-12 and rate_gap < best_rate_gap):
            best_f1 = score
            best_threshold = float(threshold)
            best_pred_rate = pred_rate
            best_rate_gap = rate_gap
    info = {
        "best_threshold": best_threshold,
        "threshold_search_mode": THRESHOLD_SEARCH_MODE,
        "val_best_macro_f1": float(best_f1),
        "val_pred_positive_rate_at_best_threshold": float(best_pred_rate),
    }
    return info if return_info else best_threshold


def fixed_threshold_diagnostics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    preds = (scores >= threshold).astype(np.int64)
    return {
        "macro_f1_at_05": macro_f1(labels, preds),
        "pred_positive_rate_at_05": float(np.mean(preds)) if preds.size else 0.0,
    }


def _threshold_candidates(scores: np.ndarray, thresholds: np.ndarray | None) -> np.ndarray:
    if thresholds is not None:
        candidates = np.asarray(thresholds, dtype=np.float32)
    else:
        quantiles = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99))).astype(np.float32)
        candidates = np.concatenate([DENSE_THRESHOLD_GRID.astype(np.float32), quantiles])
    candidates = np.unique(candidates[np.isfinite(candidates)])
    if candidates.size == 0:
        return np.array([0.5], dtype=np.float32)
    return candidates


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


def prediction_probability_stats(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return {
            "mean_pred_prob_pos": 0.0,
            "mean_pred_prob_neg": 0.0,
            "std_pred_prob": 0.0,
            "min_pred_prob": 0.0,
            "max_pred_prob": 0.0,
        }
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    return {
        "mean_pred_prob_pos": float(np.mean(pos_scores)) if pos_scores.size else 0.0,
        "mean_pred_prob_neg": float(np.mean(neg_scores)) if neg_scores.size else 0.0,
        "std_pred_prob": float(np.std(scores)),
        "min_pred_prob": float(np.min(scores)),
        "max_pred_prob": float(np.max(scores)),
    }
