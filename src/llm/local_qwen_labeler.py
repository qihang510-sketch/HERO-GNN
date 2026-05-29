from __future__ import annotations

from typing import Any

from src.llm.base_labeler import (
    OPTIONAL_REAL_LLM_MESSAGE,
    BaseRiskLabeler,
    OptionalLabelerUnavailable,
    build_risk_card_prompt,
    normalize_label,
    parse_json_object,
)

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_LOAD_ERROR = (
    "Failed to load Qwen2.5-7B-Instruct. "
    "Please check HuggingFace access or provide --model_name_or_path. "
    f"{OPTIONAL_REAL_LLM_MESSAGE}"
)


class LocalQwenRiskLabeler(BaseRiskLabeler):
    name = "local_qwen"

    def __init__(self, model_name_or_path: str | None = None, model_path: str | None = None, max_new_tokens: int = 256) -> None:
        self.model_name_or_path = model_name_or_path or model_path or DEFAULT_QWEN_MODEL
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalLabelerUnavailable(QWEN_LOAD_ERROR) from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise OptionalLabelerUnavailable(QWEN_LOAD_ERROR) from exc
        self.max_new_tokens = int(max_new_tokens)

    def label_risk_card(self, risk_card: dict[str, Any]) -> dict[str, Any]:
        messages = build_risk_card_prompt(risk_card)
        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return normalize_label(parse_json_object(text), risk_card=risk_card)

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"


def label_candidate_with_local_qwen(candidate: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return LocalQwenRiskLabeler(**kwargs).label_risk_card(candidate)
