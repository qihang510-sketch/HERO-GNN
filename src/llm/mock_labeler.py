from __future__ import annotations

from typing import Any

from src.graph.neighbor_retrieval import HeteroCandidate
from src.llm.risk_card import format_candidate_risk_card

MOCK_LABELER_VERSION = "risk_card_v2"


def mock_risk_label(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["spam", "fake", "refund", "burst"]):
        return "risk"
    return "normal"


def label_candidate_mechanism(candidate: HeteroCandidate) -> dict[str, Any]:
    card = format_candidate_risk_card(candidate)
    return label_risk_card_mechanism(card)


def label_risk_card_mechanism(card: dict[str, Any]) -> dict[str, Any]:
    official_mode = not bool(card.get("has_text_signal", True))
    mechanism = "irrelevant_heterophily"
    confidence = 0.35
    rationale = "pair lacks the combined semantic, structural, and behavioral evidence required for risk relevance"

    semantic_dissimilar = True if official_mode else float(card["semantic_similarity"]) <= 0.55
    structurally_close = float(card["structural_score"]) >= 0.60
    numeric_signal = float(card["numeric_deviation"]) >= 0.45 or float(card["rating_diff"]) >= 0.45
    time_signal = bool(card["same_time_window"]) and float(card["time_deviation"]) >= 0.65
    burst_signal = float(card["burst_score"]) >= 0.55
    identity_signal = bool(card["same_user"]) and float(card["rating_diff"]) >= 0.45
    counterparty_signal = bool(card["same_item_or_business"]) and (numeric_signal or burst_signal or time_signal)
    has_behavior_signal = numeric_signal or time_signal or burst_signal or identity_signal or counterparty_signal

    risk_relevance = int(semantic_dissimilar and structurally_close and has_behavior_signal)
    if risk_relevance:
        if bool(card["same_user"]) and numeric_signal:
            mechanism = "behavioral_contradiction"
            confidence = 0.83
            rationale = "same user pair is semantically dissimilar, structurally close, and has rating or numeric contradiction"
        elif bool(card["same_item_or_business"]) and burst_signal and time_signal:
            mechanism = "coordinated_burst"
            confidence = 0.81
            rationale = "same item/business pair combines semantic mismatch, short-time proximity, and burst behavior"
        elif bool(card["same_user"]) and burst_signal:
            mechanism = "identity_sharing"
            confidence = 0.77
            rationale = "shared identity context has semantic mismatch plus burst-like behavior"
        elif bool(card["same_item_or_business"]) and numeric_signal:
            mechanism = "counterparty_risk"
            confidence = 0.75
            rationale = "same item/business neighborhood shows semantic mismatch with numeric or rating deviation"
        elif structurally_close and numeric_signal and time_signal:
            mechanism = "camouflage_bridge"
            confidence = 0.72
            rationale = "structurally close pair bridges semantic mismatch with time and numeric anomalies"
        else:
            mechanism = "camouflage_bridge"
            confidence = 0.68
            rationale = "heterophilous pair has structural support and at least one behavioral anomaly"

    return {
        "target_id": card["target_id"],
        "neighbor_id": card["neighbor_id"],
        "metapath": card["metapath"],
        "mechanism": mechanism,
        "risk_relevance": risk_relevance,
        "confidence": confidence,
        "rationale": rationale,
        "risk_card": card,
        "labeler_mode": "HERO-official" if official_mode else "risk-card",
        "labeler_version": MOCK_LABELER_VERSION,
    }
