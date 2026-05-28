from __future__ import annotations

from typing import Any

from src.graph.neighbor_retrieval import HeteroCandidate


def format_risk_card(node_id: int | str, score: float, evidence_chain: list[dict]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "risk_score": float(score),
        "evidence_chain": evidence_chain,
    }


def format_candidate_risk_card(candidate: HeteroCandidate) -> dict[str, Any]:
    return {
        "target_id": candidate.target_id,
        "neighbor_id": candidate.neighbor_id,
        "metapath": candidate.metapath,
        "semantic_similarity": float(candidate.semantic_similarity),
        "has_text_signal": bool(getattr(candidate, "has_text_signal", True)),
        "structural_score": float(candidate.structural_score),
        "numeric_deviation": float(candidate.numeric_deviation),
        "time_deviation": float(candidate.time_deviation),
        "same_user": bool(candidate.same_user),
        "same_item_or_business": bool(candidate.same_item_or_business),
        "same_time_window": bool(candidate.time_gap_is_short),
        "rating_diff": float(candidate.rating_diff),
        "burst_score": float(candidate.burst_score),
    }


def format_pair_risk_card(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": label["target_id"],
        "neighbor_id": label["neighbor_id"],
        "metapath": label["metapath"],
        "mechanism": label["mechanism"],
        "risk_relevance": int(label["risk_relevance"]),
        "confidence": float(label["confidence"]),
        "rationale": label["rationale"],
    }
