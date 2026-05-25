from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.graph.neighbor_retrieval import HeteroCandidate


def mock_risk_label(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["spam", "fake", "refund", "burst"]):
        return "risk"
    return "normal"


def label_candidate_mechanism(candidate: HeteroCandidate) -> dict[str, Any]:
    mechanism = "irrelevant_heterophily"
    confidence = 0.45
    rationale = "heterophilous neighbor without enough risk context"

    if candidate.semantic_similarity < 0.35 and candidate.same_user and candidate.rating_diff > 0.6:
        mechanism = "behavioral_contradiction"
        confidence = 0.83
        rationale = "same user, low semantic similarity, high rating deviation"
    elif candidate.semantic_similarity < 0.45 and candidate.same_device:
        mechanism = "identity_sharing"
        confidence = 0.80
        rationale = "same device links semantically different reviews"
    elif (
        candidate.semantic_similarity < 0.50
        and candidate.same_item
        and candidate.time_gap_is_short
        and candidate.burst_score > 0.5
    ):
        mechanism = "coordinated_burst"
        confidence = 0.78
        rationale = "same item, short time gap, bursty behavior"
    elif candidate.structural_score > 0.7 and candidate.neighbor_risk_prior > 0.5:
        mechanism = "counterparty_risk"
        confidence = 0.76
        rationale = "close metapath neighbor has high risk prior"
    elif candidate.semantic_similarity < 0.55 and candidate.structural_score > 0.8:
        mechanism = "camouflage_bridge"
        confidence = 0.68
        rationale = "semantically normal-looking review bridges to risky structure"

    risk_relevance = 0 if mechanism == "irrelevant_heterophily" else 1
    return {
        "target_id": candidate.target_id,
        "neighbor_id": candidate.neighbor_id,
        "metapath": candidate.metapath,
        "mechanism": mechanism,
        "risk_relevance": risk_relevance,
        "confidence": confidence,
        "rationale": rationale,
        "candidate": asdict(candidate),
    }
