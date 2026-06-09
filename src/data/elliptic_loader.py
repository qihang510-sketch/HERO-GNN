from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import schema


ELLIPTIC_FEATURES = "elliptic_txs_features.csv"
ELLIPTIC_EDGES = "elliptic_txs_edgelist.csv"
ELLIPTIC_CLASSES = "elliptic_txs_classes.csv"


def prepare_elliptic_dataset(
    raw_dir: str | Path = "data/raw/elliptic",
    output_dir: str | Path = "data/processed/elliptic",
    seed: int = 42,
    force: bool = False,
) -> Path:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    required_outputs = [output_dir / "nodes.csv", output_dir / "edges.csv", output_dir / "features.npz", output_dir / "split.json"]
    if not force and all(path.exists() for path in required_outputs):
        return output_dir

    feature_path = raw_dir / ELLIPTIC_FEATURES
    edge_path = raw_dir / ELLIPTIC_EDGES
    class_path = raw_dir / ELLIPTIC_CLASSES
    _require_files([feature_path, edge_path, class_path])

    features = pd.read_csv(feature_path, header=None)
    if features.shape[1] < 3:
        raise ValueError("Elliptic feature file must contain txId, time_step, and feature columns.")
    tx_ids = features.iloc[:, 0].astype(str)
    time_step = pd.to_numeric(features.iloc[:, 1], errors="coerce").fillna(-1).astype(int)
    feature_matrix = features.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    classes = pd.read_csv(class_path)
    id_col = _first_existing(classes, ["txId", "tx_id", "node_id"])
    class_col = _first_existing(classes, ["class", "label"])
    labels_by_id = {
        str(row[id_col]): _elliptic_label(row[class_col])
        for _, row in classes.iterrows()
    }
    labels = np.asarray([labels_by_id.get(tx_id, -1) for tx_id in tx_ids], dtype=np.int64)

    nodes = pd.DataFrame(
        {
            schema.NODE_ID: tx_ids,
            schema.NODE_TYPE: "transaction",
            schema.TEXT: "",
            schema.LABEL: labels,
            schema.SPLIT: "",
            schema.TIMESTAMP: time_step,
            "time_step": time_step,
        }
    )
    split = _labeled_split(nodes[schema.NODE_ID].to_numpy(dtype=str), labels, seed=seed)
    for split_name, ids in split.items():
        nodes.loc[nodes[schema.NODE_ID].isin(ids), schema.SPLIT] = split_name

    raw_edges = pd.read_csv(edge_path)
    src_col = _first_existing(raw_edges, ["txId1", "src", "source", "from"])
    dst_col = _first_existing(raw_edges, ["txId2", "dst", "target", "to"])
    known_ids = set(tx_ids.tolist())
    edges = raw_edges[[src_col, dst_col]].rename(columns={src_col: schema.SRC, dst_col: schema.DST})
    edges[schema.SRC] = edges[schema.SRC].astype(str)
    edges[schema.DST] = edges[schema.DST].astype(str)
    edges = edges[edges[schema.SRC].isin(known_ids) & edges[schema.DST].isin(known_ids)].copy()
    edges[schema.EDGE_TYPE] = "transaction-transfer"
    edges[schema.TIMESTAMP] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges[[schema.SRC, schema.DST, schema.EDGE_TYPE, schema.TIMESTAMP]].to_csv(output_dir / "edges.csv", index=False)
    np.savez_compressed(
        output_dir / "features.npz",
        node_ids=tx_ids.to_numpy(dtype=str),
        features=feature_matrix.astype(np.float32),
        text_features=np.zeros((feature_matrix.shape[0], 0), dtype=np.float32),
        numeric_features=feature_matrix.astype(np.float32),
    )
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    _write_report(output_dir, raw_dir, nodes, edges, feature_matrix)
    return output_dir


def _elliptic_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "illicit", "fraud", "true"}:
        return 1
    if text in {"2", "licit", "benign", "false", "0"}:
        return 0
    return -1


def _labeled_split(node_ids: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for label in [0, 1]:
        labeled = np.asarray([idx for idx, value in enumerate(labels.tolist()) if int(value) == label], dtype=np.int64)
        rng.shuffle(labeled)
        n = int(labeled.size)
        train_end = int(round(n * 0.6))
        val_end = train_end + int(round(n * 0.2))
        split["train"].extend(node_ids[labeled[:train_end]].astype(str).tolist())
        split["val"].extend(node_ids[labeled[train_end:val_end]].astype(str).tolist())
        split["test"].extend(node_ids[labeled[val_end:]].astype(str).tolist())
    for ids in split.values():
        ids.sort()
    return split


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of columns {names}, found {list(frame.columns)}")


def _write_report(output_dir: Path, raw_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, features: np.ndarray) -> None:
    payload = {
        "dataset": "elliptic",
        "raw_dir": str(raw_dir),
        "num_nodes": int(nodes.shape[0]),
        "num_edges": int(edges.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_labeled": int((nodes[schema.LABEL] >= 0).sum()),
        "num_unknown": int((nodes[schema.LABEL] < 0).sum()),
        "label_source": "official",
        "unknown_label_policy": "kept_as_graph_context_excluded_from_loss",
    }
    (output_dir / "preprocess_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files. This is expected on local VSCode. "
            f"Please run on AutoDL or provide data path. Missing: {missing}"
        )
