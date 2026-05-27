from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LabelCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def get(self, key: str) -> dict[str, Any] | None:
        return self.values.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.values[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for key in sorted(self.values):
                value = self.values[key]
                payload = {
                    "target_id": value["target_id"],
                    "neighbor_id": value["neighbor_id"],
                    "metapath": value["metapath"],
                    "mechanism": value["mechanism"],
                    "risk_relevance": int(value["risk_relevance"]),
                    "confidence": float(value["confidence"]),
                    "rationale": value["rationale"],
                }
                if "risk_card" in value:
                    payload["risk_card"] = value["risk_card"]
                if "labeler_version" in value:
                    payload["labeler_version"] = value["labeler_version"]
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return
        if text.startswith("{"):
            try:
                raw = json.loads(text)
                if isinstance(raw, dict):
                    self.values = raw
                    return
            except json.JSONDecodeError:
                pass
        for line in text.splitlines():
            payload = json.loads(line)
            key = cache_key(payload["target_id"], payload["neighbor_id"], payload.get("metapath", ""))
            self.values[key] = payload


def cache_key(target_id: str, neighbor_id: str, metapath: str) -> str:
    return f"{target_id}|{neighbor_id}|{metapath}"
