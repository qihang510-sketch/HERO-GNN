from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import schema
from src.data.schema import GraphData


@dataclass
class ProcessedGraphData:
    features: np.ndarray
    text_features: np.ndarray
    numeric_features: np.ndarray
    labels: np.ndarray
    split: dict[str, np.ndarray]
    edges: pd.DataFrame
    nodes: pd.DataFrame
    node_id_to_idx: dict[str, int]
    edge_index: np.ndarray
    evidence_gt: dict


def load_processed_data(data_dir: str | Path = "data/processed/synthetic") -> ProcessedGraphData:
    data_dir = Path(data_dir)
    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    features_path = data_dir / "features.npz"
    split_path = data_dir / "split.json"
    evidence_path = data_dir / "evidence_gt.json"

    _require_files([nodes_path, edges_path, features_path, split_path])

    nodes = pd.read_csv(nodes_path).fillna("")
    edges = pd.read_csv(edges_path).fillna("")
    feature_payload = np.load(features_path, allow_pickle=True)
    node_ids = feature_payload["node_ids"].astype(str).tolist()
    features = feature_payload["features"].astype(np.float32)
    text_features = feature_payload["text_features"].astype(np.float32)
    numeric_features = feature_payload["numeric_features"].astype(np.float32)

    node_id_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}
    labels_by_id = nodes.set_index(schema.NODE_ID)[schema.LABEL].to_dict()
    labels = np.array([int(labels_by_id.get(node_id, -1)) for node_id in node_ids], dtype=np.int64)

    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    split = {
        name: np.array([node_id_to_idx[node_id] for node_id in ids if node_id in node_id_to_idx], dtype=np.int64)
        for name, ids in split_ids.items()
    }

    edge_pairs = [
        (node_id_to_idx[src], node_id_to_idx[dst])
        for src, dst in edges[[schema.SRC, schema.DST]].itertuples(index=False, name=None)
        if src in node_id_to_idx and dst in node_id_to_idx
    ]
    edge_index = np.array(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64)
    evidence_gt = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    return ProcessedGraphData(
        features=features,
        text_features=text_features,
        numeric_features=numeric_features,
        labels=labels,
        split=split,
        edges=edges,
        nodes=nodes,
        node_id_to_idx=node_id_to_idx,
        edge_index=edge_index,
        evidence_gt=evidence_gt,
    )


def load_graph(path: str | Path) -> GraphData | ProcessedGraphData:
    path = Path(path)
    if path.is_dir():
        return load_processed_data(path)
    return GraphData.load_npz(path)


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed data file(s): {missing}")
