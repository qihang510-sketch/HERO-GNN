from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.llm.json_utils import (
    LEGACY_LABEL_FIELDS as LABEL_FIELDS,
    bounded_float,
    bounded_int,
    label_key,
    legacy_jsonl_label,
    normalize_label,
    parse_json_object,
    strict_jsonl_label,
)
from src.llm.prompt_templates import build_risk_card_prompt

OPTIONAL_REAL_LLM_MESSAGE = "Real LLM labeler is optional. Please set OPENAI_API_KEY or provide local model path."


class OptionalLabelerUnavailable(RuntimeError):
    pass


class BaseRiskLabeler(ABC):
    name = "base"

    @abstractmethod
    def label_risk_card(self, risk_card: dict[str, Any]) -> dict[str, Any]:
        """Return the shared risk-label schema for one risk card."""


def jsonl_label(label: dict[str, Any]) -> str:
    return legacy_jsonl_label(label)


__all__ = [
    "BaseRiskLabeler",
    "LABEL_FIELDS",
    "OPTIONAL_REAL_LLM_MESSAGE",
    "OptionalLabelerUnavailable",
    "bounded_float",
    "bounded_int",
    "build_risk_card_prompt",
    "jsonl_label",
    "label_key",
    "normalize_label",
    "parse_json_object",
    "strict_jsonl_label",
]
