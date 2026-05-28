from __future__ import annotations

import os
from typing import Any

from src.llm.base_labeler import (
    OPTIONAL_REAL_LLM_MESSAGE,
    BaseRiskLabeler,
    OptionalLabelerUnavailable,
    build_risk_card_prompt,
    normalize_label,
    parse_json_object,
)


class LocalQwenRiskLabeler(BaseRiskLabeler):
    name = "local_qwen"

    def __init__(self, model_path: str | None = None, max_new_tokens: int = 256) -> None:
        self.model_path = model_path or os.getenv("LOCAL_QWEN_MODEL_PATH")
        if not self.model_path:
            raise OptionalLabelerUnavailable(OPTIONAL_REAL_LLM_MESSAGE)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalLabelerUnavailable(
                "Real LLM labeler is optional. Please set OPENAI_API_KEY or provide local model path."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True)
        self.max_new_tokens = int(max_new_tokens)

    def label_risk_card(self, risk_card: dict[str, Any]) -> dict[str, Any]:
        messages = build_risk_card_prompt(risk_card)
        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return normalize_label(parse_json_object(text), risk_card=risk_card)

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"


def label_candidate_with_local_qwen(candidate: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return LocalQwenRiskLabeler(**kwargs).label_risk_card(candidate)
