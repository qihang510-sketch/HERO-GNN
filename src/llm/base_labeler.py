from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from src.data.schema import EVIDENCE_MECHANISMS

OPTIONAL_REAL_LLM_MESSAGE = "Real LLM labeler is optional. Please set OPENAI_API_KEY or provide local model path."
LABEL_FIELDS = ("target_id", "neighbor_id", "metapath", "mechanism", "risk_relevance", "confidence", "rationale")


class OptionalLabelerUnavailable(RuntimeError):
    pass


class BaseRiskLabeler(ABC):
    name = "base"

    @abstractmethod
    def label_risk_card(self, risk_card: dict[str, Any]) -> dict[str, Any]:
        """Return the shared risk-label schema for one risk card."""


def build_risk_card_prompt(risk_card: dict[str, Any]) -> list[dict[str, str]]:
    mechanisms = ", ".join(EVIDENCE_MECHANISMS)
    schema = {
        "target_id": str(risk_card.get("target_id", "")),
        "neighbor_id": str(risk_card.get("neighbor_id", "")),
        "metapath": str(risk_card.get("metapath", "")),
        "mechanism": "one of the allowed mechanism strings",
        "risk_relevance": 0,
        "confidence": 0.0,
        "rationale": "short reason",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a fraud-graph risk labeling assistant. "
                "Classify one heterophilic neighbor risk card. "
                f"Allowed mechanisms: {mechanisms}. "
                "Return JSON only. Do not include markdown, prose, or code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Is this heterophilic neighbor risk-relevant to the target?\n"
                "Which mechanism best explains the relation?\n"
                "Return valid JSON only with exactly these keys:\n"
                f"{json.dumps(schema, sort_keys=True)}\n"
                "Risk card:\n"
                f"{json.dumps(risk_card, sort_keys=True)}"
            ),
        },
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
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
        "metapath": str(payload.get("metapath", risk_card.get("metapath", ""))),
        "mechanism": mechanism,
        "risk_relevance": int(_bounded_int(payload.get("risk_relevance", 0))),
        "confidence": _bounded_float(payload.get("confidence", 0.0)),
        "rationale": str(payload.get("rationale", "")),
    }
    return label


def label_key(label: dict[str, Any]) -> str:
    return f"{label['target_id']}|{label['neighbor_id']}|{label['metapath']}"


def jsonl_label(label: dict[str, Any]) -> str:
    return json.dumps({field: label[field] for field in LABEL_FIELDS}, sort_keys=True)


def _bounded_int(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _bounded_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric
