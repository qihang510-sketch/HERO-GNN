from __future__ import annotations

from typing import Any


def format_risk_card(node_id: int | str, score: float, evidence_chain: list[dict]) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "risk_score": float(score),
        "evidence_chain": evidence_chain,
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
