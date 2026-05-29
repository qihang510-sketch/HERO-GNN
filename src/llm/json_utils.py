from __future__ import annotations

import json
import re
from typing import Any

from src.data.schema import EVIDENCE_MECHANISMS

STRICT_LABEL_FIELDS = (
    "dataset",
    "target_id",
    "neighbor_id",
    "mechanism",
    "risk_relevance",
    "confidence",
    "rationale",
)
LEGACY_LABEL_FIELDS = (
    "target_id",
    "neighbor_id",
    "metapath",
    "mechanism",
    "risk_relevance",
    "confidence",
    "rationale",
)


def parse_json_object(text: str) -> dict[str, Any]:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def normalize_label(payload: dict[str, Any], risk_card: dict[str, Any] | None = None) -> dict[str, Any]:
    risk_card = risk_card or {}
    mechanism = str(payload.get("mechanism", "irrelevant_heterophily"))
    if mechanism not in EVIDENCE_MECHANISMS:
        mechanism = "irrelevant_heterophily"

    label = {
        "target_id": str(payload.get("target_id", risk_card.get("target_id", ""))),
        "neighbor_id": str(payload.get("neighbor_id", risk_card.get("neighbor_id", ""))),
        "metapath": str(payload.get("metapath", payload.get("edge_type", risk_card.get("metapath", risk_card.get("edge_type", ""))))),
        "mechanism": mechanism,
        "risk_relevance": bounded_int(payload.get("risk_relevance", 0)),
        "confidence": bounded_float(payload.get("confidence", 0.0)),
        "rationale": str(payload.get("rationale", "")),
    }
    dataset = payload.get("dataset", risk_card.get("dataset"))
    if dataset is not None and str(dataset) != "":
        label["dataset"] = str(dataset)
    labeler = payload.get("labeler", payload.get("llm_labeler"))
    if labeler is not None and str(labeler) != "":
        label["labeler"] = str(labeler)
    return label


def strict_label(label: dict[str, Any], risk_card: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_label(label, risk_card=risk_card)
    return {
        "dataset": str(normalized.get("dataset", (risk_card or {}).get("dataset", ""))),
        "target_id": str(normalized.get("target_id", "")),
        "neighbor_id": str(normalized.get("neighbor_id", "")),
        "mechanism": str(normalized.get("mechanism", "irrelevant_heterophily")),
        "risk_relevance": int(normalized.get("risk_relevance", 0)),
        "confidence": bounded_float(normalized.get("confidence", 0.0)),
        "rationale": str(normalized.get("rationale", "")),
    }


def fallback_label(risk_card: dict[str, Any], rationale: str = "labeler output could not be parsed") -> dict[str, Any]:
    return strict_label(
        {
            "dataset": risk_card.get("dataset", ""),
            "target_id": risk_card.get("target_id", ""),
            "neighbor_id": risk_card.get("neighbor_id", ""),
            "mechanism": "irrelevant_heterophily",
            "risk_relevance": 0,
            "confidence": 0.0,
            "rationale": rationale,
        },
        risk_card=risk_card,
    )


def label_key(label: dict[str, Any]) -> str:
    target_id = str(label.get("target_id", ""))
    neighbor_id = str(label.get("neighbor_id", ""))
    metapath = str(label.get("metapath", label.get("edge_type", "")))
    if metapath:
        return f"{target_id}|{neighbor_id}|{metapath}"
    return pair_key(target_id, neighbor_id)


def pair_key(target_id: str, neighbor_id: str) -> str:
    return f"{target_id}|{neighbor_id}"


def strict_jsonl_label(label: dict[str, Any], risk_card: dict[str, Any] | None = None) -> str:
    payload = strict_label(label, risk_card=risk_card)
    return json.dumps({field: payload[field] for field in STRICT_LABEL_FIELDS}, sort_keys=True)


def legacy_jsonl_label(label: dict[str, Any]) -> str:
    normalized = normalize_label(label)
    return json.dumps({field: normalized[field] for field in LEGACY_LABEL_FIELDS}, sort_keys=True)


def bounded_int(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except (TypeError, ValueError):
        return 0


def bounded_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric
