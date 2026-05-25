from __future__ import annotations

from typing import Any


def label_candidate_with_openai(candidate: dict[str, Any]) -> dict[str, Any]:
    """Reserved interface for a future OpenAI-backed risk mechanism labeler."""
    raise NotImplementedError(
        "OpenAI labeler is reserved for future use. Return target_id, neighbor_id, "
        "mechanism, risk_relevance, confidence, and rationale in the shared schema."
    )
