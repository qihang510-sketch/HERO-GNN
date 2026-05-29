from __future__ import annotations

import json
from typing import Any

from src.data.schema import EVIDENCE_MECHANISMS
from src.llm.json_utils import STRICT_LABEL_FIELDS

SYSTEM_PROMPT = (
    "You are a fraud-risk mechanism annotator for graph-based fraud detection.\n"
    "You do not classify the target as fraud or normal.\n"
    "Your task is to judge whether a heterophilic neighbor provides risk-relevant evidence for the target.\n"
    "Return JSON only."
)

MECHANISM_DEFINITIONS = {
    "irrelevant_heterophily": "The neighbor is different from the target but does not provide risk-relevant evidence.",
    "behavioral_contradiction": "Shared context with contradictory ratings, timing, or numeric behavior.",
    "coordinated_burst": "Short-time or burst-like activity suggests coordinated behavior around the same context.",
    "identity_sharing": "Shared user, account, device, or identity context makes the neighbor risk-relevant.",
    "camouflage_bridge": "The neighbor bridges normal-looking and suspicious evidence through structural proximity.",
    "counterparty_risk": "Shared item, business, product, or counterparty context carries risk-relevant evidence.",
}


def build_risk_card_prompt(risk_card: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(risk_card)},
    ]


def build_user_prompt(risk_card: dict[str, Any]) -> str:
    schema = {
        "dataset": str(risk_card.get("dataset", "")),
        "target_id": str(risk_card.get("target_id", "")),
        "neighbor_id": str(risk_card.get("neighbor_id", "")),
        "mechanism": "one of: " + ", ".join(EVIDENCE_MECHANISMS),
        "risk_relevance": "0 or 1",
        "confidence": "number from 0.0 to 1.0",
        "rationale": "short reason, one sentence",
    }
    target = {
        "target_id": risk_card.get("target_id", ""),
        "target_text": risk_card.get("target_text", ""),
        "target_rating": risk_card.get("target_rating", 0.0),
    }
    neighbor = {
        "neighbor_id": risk_card.get("neighbor_id", ""),
        "neighbor_text": risk_card.get("neighbor_text", ""),
        "neighbor_rating": risk_card.get("neighbor_rating", 0.0),
    }
    deviations = {
        "rating_deviation": risk_card.get("rating_deviation", 0.0),
        "time_deviation": risk_card.get("time_deviation", 0.0),
        "semantic_similarity": risk_card.get("semantic_similarity", 0.0),
        "feature_distance": risk_card.get("feature_distance", 0.0),
        "structural_score": risk_card.get("structural_score", 0.0),
    }
    relation = {
        "candidate_reason": risk_card.get("candidate_reason", ""),
        "edge_type": risk_card.get("edge_type", risk_card.get("metapath", "")),
    }
    return (
        "Target node evidence:\n"
        f"{json.dumps(target, sort_keys=True)}\n\n"
        "Neighbor node evidence:\n"
        f"{json.dumps(neighbor, sort_keys=True)}\n\n"
        "Pairwise deviations:\n"
        f"{json.dumps(deviations, sort_keys=True)}\n\n"
        "Candidate relation:\n"
        f"{json.dumps(relation, sort_keys=True)}\n\n"
        "Risk card:\n"
        f"{json.dumps(risk_card, sort_keys=True)}\n\n"
        "Mechanism definitions:\n"
        f"{json.dumps(MECHANISM_DEFINITIONS, sort_keys=True)}\n\n"
        "Output JSON schema:\n"
        f"{json.dumps(schema, sort_keys=True)}\n\n"
        "Rules:\n"
        "- Do not output markdown.\n"
        "- Do not output explanatory paragraphs.\n"
        "- Only output valid JSON.\n"
        f"- Output exactly these keys: {', '.join(STRICT_LABEL_FIELDS)}.\n"
        "- Do not classify the target as fraud or normal."
    )
