from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_risk_cards import TEXT_RICH_DATASETS, build_risk_cards
from src.data import schema
from src.llm.json_utils import label_key, normalize_label, strict_jsonl_label
from src.llm.mock_labeler import label_risk_card_mechanism
from src.training.submission import _submission_metric_payload, processed_ready, resolve_processed_dir
from src.training.trainer import _resolve_hero_config, train_single_experiment


METRICS = ["Macro-F1", "AUROC", "AUPRC"]
REAL_LLM_HINTS = ("qwen", "openai", "gpt", "claude", "gemini", "llm_real", "full_llm")
NOISE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]


def read_jsonl_labels(path: str | Path) -> list[dict[str, Any]]:
    label_path = Path(path)
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            labels.append(normalize_label(json.loads(line)))
    return labels


def write_jsonl_labels(path: str | Path, labels: list[dict[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(strict_jsonl_label(label) + "\n")
    return out


def annotation_stats(
    labels: list[dict[str, Any]],
    candidate_cards: int | None = None,
    annotation_source: str = "",
    annotation_time_seconds: float | None = None,
) -> dict[str, Any]:
    count = int(len(labels))
    risk_values = [int(label.get("risk_relevance", 0)) for label in labels]
    confidences = [float(label.get("confidence", 0.0)) for label in labels]
    cards = int(candidate_cards) if candidate_cards is not None else count
    return {
        "annotation_source": annotation_source,
        "candidate_cards": cards,
        "annotated_cards": count,
        "annotation_coverage": float(count / cards) if cards else 0.0,
        "risk_relevance_positive_rate": float(np.mean(risk_values)) if risk_values else 0.0,
        "mechanism_distribution": json.dumps(_mechanism_distribution(labels), sort_keys=True),
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "annotation_time_seconds": float(annotation_time_seconds) if annotation_time_seconds is not None else pd.NA,
    }


def find_annotation_file(dataset: str, data_root: str | Path = "data", explicit: str | Path | None = None) -> tuple[Path | None, str]:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path, _source_for_annotation(path, explicit=True)
        return None, "explicit_annotation_missing"
    processed_dir = resolve_processed_dir(dataset, data_root)
    if not processed_dir.exists():
        return None, "processed_data_missing"
    real_candidates: list[Path] = []
    for pattern in ["*qwen*.jsonl", "*Qwen*.jsonl", "*openai*.jsonl", "*gpt*.jsonl", "*claude*.jsonl", "*full_llm*.jsonl"]:
        real_candidates.extend(sorted(processed_dir.glob(pattern)))
    if real_candidates:
        return real_candidates[0], "cached_llm"
    fallback_candidates = [
        processed_dir / "llm_labels.jsonl",
        processed_dir / "llm_labels_mock.jsonl",
        processed_dir / "llm_labels_rule.jsonl",
        processed_dir / "rule_labels.jsonl",
        *sorted(processed_dir.glob("*rule*.jsonl")),
        *sorted(processed_dir.glob("*mock*.jsonl")),
    ]
    for path in fallback_candidates:
        if path.exists():
            return path, "proxy_or_rule_cache"
    return None, "annotation_cache_missing"


def ensure_annotation_file(
    dataset: str,
    data_root: str | Path,
    output_dir: str | Path,
    seed: int,
    explicit: str | Path | None = None,
    max_cards: int = 2000,
    source_hint: str = "rule_based_fallback",
) -> tuple[Path | None, str, dict[str, Any]]:
    found, source = find_annotation_file(dataset, data_root=data_root, explicit=explicit)
    if found is not None:
        labels = read_jsonl_labels(found)
        return found, source, annotation_stats(labels, annotation_source=source)
    processed_dir = resolve_processed_dir(dataset, data_root)
    if not processed_ready(processed_dir) or dataset not in TEXT_RICH_DATASETS:
        return None, source, {"annotation_source": source, "status": "missing", "skip_reason": "annotation_cache_and_fallback_unavailable"}
    start = time.perf_counter()
    labels = build_rule_based_labels_from_cards(dataset, processed_dir, seed=seed, max_cards=max_cards)
    elapsed = time.perf_counter() - start
    out_file = Path(output_dir) / "annotations" / "fallback" / dataset / f"{source_hint}_seed_{seed}.jsonl"
    write_jsonl_labels(out_file, labels)
    stats = annotation_stats(labels, candidate_cards=len(labels), annotation_source=source_hint, annotation_time_seconds=elapsed)
    return out_file, source_hint, stats


def build_rule_based_labels_from_cards(dataset: str, processed_dir: Path, seed: int, max_cards: int = 2000) -> list[dict[str, Any]]:
    cards = build_risk_cards(
        dataset=dataset,
        data_dir=processed_dir,
        max_cards=max_cards,
        seed=seed,
        max_candidates_per_node=20,
    )
    labels = []
    for card in cards:
        label = normalize_label(label_risk_card_mechanism(card), risk_card=card)
        label["dataset"] = dataset
        label["labeler"] = "rule_based_fallback"
        labels.append(label)
    return labels


def labels_from_cards(cards: list[dict[str, Any]], labeler: str, seed: int) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    labels: list[dict[str, Any]] = []
    for card in cards:
        if labeler == "random_labeler":
            labels.append(_random_label(card, rng))
        elif labeler == "proxy_labeler":
            labels.append(_proxy_label(card))
        else:
            label = normalize_label(label_risk_card_mechanism(card), risk_card=card)
            label["labeler"] = "rule_based_labeler"
            labels.append(label)
    return labels, time.perf_counter() - start


def load_or_build_risk_cards(
    dataset: str,
    data_root: str | Path,
    output_dir: str | Path,
    seed: int,
    risk_card_file: str | Path | None = None,
    max_cards: int = 2000,
) -> tuple[list[dict[str, Any]], Path | None, str]:
    if risk_card_file:
        path = Path(risk_card_file)
        if not path.exists():
            return [], path, "risk_card_file_missing"
        return _read_jsonl(path), path, "provided_risk_cards"
    processed_dir = resolve_processed_dir(dataset, data_root)
    candidates = [
        processed_dir / "risk_cards.jsonl",
        processed_dir / f"{dataset}_risk_cards.jsonl",
        *sorted(processed_dir.glob("*risk*card*.jsonl")),
    ]
    for path in candidates:
        if path.exists():
            return _read_jsonl(path), path, "cached_risk_cards"
    if not processed_ready(processed_dir) or dataset not in TEXT_RICH_DATASETS:
        return [], None, "risk_cards_unavailable"
    cards = build_risk_cards(dataset, processed_dir, max_cards=max_cards, seed=seed)
    out_file = Path(output_dir) / "annotations" / "risk_cards" / dataset / f"risk_cards_seed_{seed}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, sort_keys=True) + "\n")
    return cards, out_file, "generated_risk_cards"


def train_hero_with_labels(
    dataset: str,
    seed: int,
    label_file: str | Path,
    output_root: str | Path,
    data_root: str | Path,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: str,
    experiment_tag: str,
    labeler: str,
) -> dict[str, Any]:
    data_dir = resolve_processed_dir(dataset, data_root)
    hero_config = _resolve_hero_config("hero_gnn", {"use_llm_annotation": True})
    metrics = train_single_experiment(
        dataset=dataset,
        model_name="hero_gnn",
        seed=int(seed),
        data_dir=data_dir,
        output_root=output_root,
        epochs=int(epochs),
        lr=float(lr),
        hidden_dim=int(hidden_dim),
        llm_label_file=label_file,
        experiment_tag=experiment_tag,
        llm_labeler=labeler,
        disable_llm_fallback=True,
        device=device,
        hero_config=hero_config,
    )
    payload = _submission_metric_payload(metrics, dataset, "hero_gnn", seed, "project")
    payload.update(
        {
            "suite_model": "hero_gnn",
            "experiment_tag": experiment_tag,
            "labeler": labeler,
            "llm_label_file": str(label_file),
            "status": "ok",
        }
    )
    return payload


def metric_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "seed_count"])
    rows: list[dict[str, Any]] = []
    ok = frame[frame.get("status", "ok").astype(str).isin(["ok", "exists"])] if "status" in frame else frame
    if ok.empty:
        return pd.DataFrame(columns=[*group_cols, "seed_count"])
    for keys, group in ok.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float) if metric in group else np.array([])
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else pd.NA
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size >= 2 else pd.NA
            row[f"{metric}_mean_std"] = _paper_mean_std(values)
        rows.append(row)
    return pd.DataFrame(rows)


def write_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def has_full_llm_config() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_QWEN_MODEL_PATH") or os.environ.get("QWEN_MODEL_PATH"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _source_for_annotation(path: Path, explicit: bool = False) -> str:
    if explicit:
        return "explicit_annotation_file"
    stem = path.stem.lower()
    return "cached_llm" if any(hint in stem for hint in REAL_LLM_HINTS) else "proxy_or_rule_cache"


def _mechanism_distribution(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        mechanism = str(label.get("mechanism", "irrelevant_heterophily"))
        counts[mechanism] = counts.get(mechanism, 0) + 1
    return counts


def _random_label(card: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    mechanism = str(rng.choice(schema.EVIDENCE_MECHANISMS))
    return normalize_label(
        {
            "dataset": card.get("dataset", ""),
            "target_id": card.get("target_id", ""),
            "neighbor_id": card.get("neighbor_id", ""),
            "mechanism": mechanism,
            "risk_relevance": int(rng.integers(0, 2)),
            "confidence": float(rng.random()),
            "rationale": "random labeler baseline",
            "labeler": "random_labeler",
        },
        risk_card=card,
    )


def _proxy_label(card: dict[str, Any]) -> dict[str, Any]:
    semantic_similarity = _float(card.get("semantic_similarity", 0.5))
    structural_score = _float(card.get("structural_score", 0.0))
    rating_deviation = _float(card.get("rating_deviation", card.get("numeric_deviation", 0.0)))
    time_deviation = _float(card.get("time_deviation", 0.0))
    score = 0.35 * (1.0 - semantic_similarity) + 0.30 * structural_score + 0.20 * rating_deviation + 0.15 * time_deviation
    relevance = int(score >= 0.45)
    if relevance and rating_deviation >= max(structural_score, time_deviation):
        mechanism = "behavioral_contradiction"
    elif relevance and time_deviation >= 0.4:
        mechanism = "coordinated_burst"
    elif relevance and structural_score >= 0.4:
        mechanism = "counterparty_risk"
    else:
        mechanism = "irrelevant_heterophily"
    return normalize_label(
        {
            "dataset": card.get("dataset", ""),
            "target_id": card.get("target_id", ""),
            "neighbor_id": card.get("neighbor_id", ""),
            "mechanism": mechanism,
            "risk_relevance": relevance,
            "confidence": float(np.clip(score, 0.0, 1.0)),
            "rationale": "proxy label from risk-card structural, semantic, rating, and time signals",
            "labeler": "proxy_labeler",
        },
        risk_card=card,
    )


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return float(np.clip(out, 0.0, 1.0))


def _paper_mean_std(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "missing"
    mean = float(np.mean(values) * 100.0)
    if values.size < 2:
        return f"{mean:.2f} \u00b1 NA"
    std = float(np.std(values, ddof=1) * 100.0)
    return f"{mean:.2f} \u00b1 {std:.2f}"
