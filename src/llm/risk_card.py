from __future__ import annotations

from typing import Any

from src.graph.neighbor_retrieval import HeteroCandidate

RISK_CARD_FIELDS = (
    "dataset",
    "target_id",
    "neighbor_id",
    "target_text",
    "neighbor_text",
    "target_rating",
    "neighbor_rating",
    "rating_deviation",
    "time_deviation",
    "semantic_similarity",
    "feature_distance",
    "structural_score",
    "candidate_reason",
    "edge_type",
)


def format_risk_card(node_id: int | str, score: float, evidence_chain: list[dict]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "risk_score": float(score),
        "evidence_chain": evidence_chain,
    }


def format_candidate_risk_card(
    candidate: HeteroCandidate,
    dataset: str = "",
    node_lookup: dict[str, dict[str, Any]] | None = None,
    rating_lookup: dict[str, float] | None = None,
) -> dict[str, Any]:
    target_rating = _rating_for(candidate.target_id, rating_lookup)
    neighbor_rating = _rating_for(candidate.neighbor_id, rating_lookup)
    rating_deviation = abs(target_rating - neighbor_rating) if target_rating or neighbor_rating else float(candidate.rating_diff)
    card = {
        "dataset": str(dataset),
        "target_id": candidate.target_id,
        "neighbor_id": candidate.neighbor_id,
        "target_text": _node_text(candidate.target_id, node_lookup),
        "neighbor_text": _node_text(candidate.neighbor_id, node_lookup),
        "target_rating": float(target_rating),
        "neighbor_rating": float(neighbor_rating),
        "rating_deviation": float(rating_deviation),
        "time_deviation": float(candidate.time_deviation),
        "semantic_similarity": float(candidate.semantic_similarity),
        "feature_distance": float(candidate.numeric_deviation),
        "structural_score": float(candidate.structural_score),
        "candidate_reason": _candidate_reason(candidate),
        "edge_type": candidate.metapath,
        "metapath": candidate.metapath,
        "has_text_signal": bool(getattr(candidate, "has_text_signal", True)),
        "numeric_deviation": float(candidate.numeric_deviation),
        "same_user": bool(candidate.same_user),
        "same_item_or_business": bool(candidate.same_item_or_business),
        "same_time_window": bool(candidate.time_gap_is_short),
        "rating_diff": float(candidate.rating_diff),
        "burst_score": float(candidate.burst_score),
        "candidate_score": float(candidate.candidate_score),
    }
    return card


def format_pair_risk_card(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": label.get("dataset", ""),
        "target_id": label["target_id"],
        "neighbor_id": label["neighbor_id"],
        "metapath": label.get("metapath", label.get("edge_type", "")),
        "mechanism": label["mechanism"],
        "risk_relevance": int(label["risk_relevance"]),
        "confidence": float(label["confidence"]),
        "rationale": label["rationale"],
    }


def strict_risk_card(card: dict[str, Any]) -> dict[str, Any]:
    return {field: _risk_card_default(card, field) for field in RISK_CARD_FIELDS}


def _risk_card_default(card: dict[str, Any], field: str) -> Any:
    if field in card and card[field] is not None:
        return card[field]
    if field in {
        "target_rating",
        "neighbor_rating",
        "rating_deviation",
        "time_deviation",
        "semantic_similarity",
        "feature_distance",
        "structural_score",
    }:
        return 0.0
    return ""


def _node_text(node_id: str, node_lookup: dict[str, dict[str, Any]] | None) -> str:
    if not node_lookup:
        return ""
    return str(node_lookup.get(str(node_id), {}).get("text", ""))


def _rating_for(node_id: str, rating_lookup: dict[str, float] | None) -> float:
    if not rating_lookup:
        return 0.0
    try:
        return float(rating_lookup.get(str(node_id), 0.0))
    except (TypeError, ValueError):
        return 0.0


def _candidate_reason(candidate: HeteroCandidate) -> str:
    relation = candidate.metapath.replace("-", " ")
    semantic_distance = 1.0 - float(candidate.semantic_similarity)
    signals = [
        f"edge_type={candidate.metapath}",
        f"semantic_distance={semantic_distance:.3f}",
        f"structural_score={float(candidate.structural_score):.3f}",
        f"feature_distance={float(candidate.numeric_deviation):.3f}",
        f"time_deviation={float(candidate.time_deviation):.3f}",
    ]
    if bool(candidate.same_user):
        signals.append("shared_user_context")
    if bool(candidate.same_item_or_business):
        signals.append("shared_item_or_business_context")
    if bool(candidate.time_gap_is_short):
        signals.append("short_time_window")
    return f"heterophilic candidate from {relation}; " + "; ".join(signals)
