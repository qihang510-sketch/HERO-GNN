from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NODE_ID = "node_id"
NODE_TYPE = "node_type"
TEXT = "text"
LABEL = "label"
SPLIT = "split"
TIMESTAMP = "timestamp"
REVIEW_KIND = "review_kind"

SRC = "src"
DST = "dst"
EDGE_TYPE = "edge_type"

NODE_TYPES = ("review", "user", "item", "device")
REVIEW_KINDS = ("normal", "homophilic_fraud", "heterophilic_fraud")
EDGE_TYPES = (
    "review-user-review",
    "review-item-review",
    "review-device-review",
    "review-time-review",
)
EVIDENCE_MECHANISMS = (
    "irrelevant_heterophily",
    "behavioral_contradiction",
    "coordinated_burst",
    "identity_sharing",
    "camouflage_bridge",
    "counterparty_risk",
)
BASE_NODE_COLUMNS = (NODE_ID, NODE_TYPE, TEXT, LABEL, SPLIT, TIMESTAMP, REVIEW_KIND)
BASE_EDGE_COLUMNS = (SRC, DST, EDGE_TYPE, TIMESTAMP)


@dataclass
class GraphData:
    features: np.ndarray
    edge_index: np.ndarray
    labels: np.ndarray
    evidence: dict[int, list[int]]
    metadata: dict[str, Any]

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence_pairs = np.array(
            [(node, item) for node, items in self.evidence.items() for item in items],
            dtype=np.int64,
        )
        np.savez_compressed(
            path,
            features=self.features,
            edge_index=self.edge_index,
            labels=self.labels,
            evidence_pairs=evidence_pairs,
            metadata=np.array([self.metadata], dtype=object),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "GraphData":
        data = np.load(Path(path), allow_pickle=True)
        evidence: dict[int, list[int]] = {}
        for node, item in data["evidence_pairs"]:
            evidence.setdefault(int(node), []).append(int(item))
        metadata = dict(data["metadata"][0]) if "metadata" in data else {}
        return cls(
            features=data["features"],
            edge_index=data["edge_index"],
            labels=data["labels"],
            evidence=evidence,
            metadata=metadata,
        )
