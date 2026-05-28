from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from src.llm.base_labeler import (
    OPTIONAL_REAL_LLM_MESSAGE,
    BaseRiskLabeler,
    OptionalLabelerUnavailable,
    build_risk_card_prompt,
    normalize_label,
    parse_json_object,
)


class OpenAIRiskLabeler(BaseRiskLabeler):
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OptionalLabelerUnavailable(OPTIONAL_REAL_LLM_MESSAGE)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.temperature = float(temperature)
        self.timeout = int(timeout)

    def label_risk_card(self, risk_card: dict[str, Any]) -> dict[str, Any]:
        messages = build_risk_card_prompt(risk_card)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI labeler request failed: {exc}") from exc
        content = result["choices"][0]["message"]["content"]
        return normalize_label(parse_json_object(content), risk_card=risk_card)


def label_candidate_with_openai(candidate: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return OpenAIRiskLabeler(**kwargs).label_risk_card(candidate)
