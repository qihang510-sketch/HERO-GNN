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
LABEL_MAPPING = {"licit": 0, "illicit": 1, "unknown": -1}


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

    features = _read_feature_frame(feature_path)
    if features.shape[1] < 3:
        raise ValueError("Elliptic feature file must contain txId, time_step, and feature columns.")
    tx_ids = features.iloc[:, 0].map(_normalize_id).astype(str)
    time_step = pd.to_numeric(features.iloc[:, 1], errors="coerce").fillna(-1).astype(int)
    feature_matrix = features.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    classes = _read_class_frame(class_path)
    id_col = _first_existing(classes, ["txId", "tx_id", "node_id"])
    class_col = _first_existing(classes, ["class", "label"])
    labels_by_id = {
        _normalize_id(row[id_col]): _elliptic_label(row[class_col])
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
    split_stats = _split_stats(split, {str(node_id): int(label) for node_id, label in zip(tx_ids, labels)})
    for split_name, ids in split.items():
        nodes.loc[nodes[schema.NODE_ID].isin(ids), schema.SPLIT] = split_name

    raw_edges = _read_edge_frame(edge_path)
    src_col = _first_existing(raw_edges, ["txId1", "src", "source", "from"])
    dst_col = _first_existing(raw_edges, ["txId2", "dst", "target", "to"])
    known_ids = set(tx_ids.tolist())
    edges = raw_edges[[src_col, dst_col]].rename(columns={src_col: schema.SRC, dst_col: schema.DST})
    edges[schema.SRC] = edges[schema.SRC].map(_normalize_id).astype(str)
    edges[schema.DST] = edges[schema.DST].map(_normalize_id).astype(str)
    edges = edges[edges[schema.SRC].isin(known_ids) & edges[schema.DST].isin(known_ids)].copy()
    edges[schema.EDGE_TYPE] = "transaction-transfer"
    edges[schema.TIMESTAMP] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges[[schema.SRC, schema.DST, schema.EDGE_TYPE, schema.TIMESTAMP]].to_csv(output_dir / "edges.csv", index=False)
    node_id_to_idx = {node_id: index for index, node_id in enumerate(tx_ids.astype(str).tolist())}
    edge_pairs = [
        (node_id_to_idx[src], node_id_to_idx[dst])
        for src, dst in edges[[schema.SRC, schema.DST]].itertuples(index=False, name=None)
        if src in node_id_to_idx and dst in node_id_to_idx
    ]
    edge_index = np.array(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64)
    np.savez_compressed(
        output_dir / "features.npz",
        node_ids=tx_ids.to_numpy(dtype=str),
        features=feature_matrix.astype(np.float32),
        text_features=np.zeros((feature_matrix.shape[0], 0), dtype=np.float32),
        numeric_features=feature_matrix.astype(np.float32),
    )
    np.save(output_dir / "features.npy", feature_matrix.astype(np.float32))
    np.save(output_dir / "edge_index.npy", edge_index)
    np.save(output_dir / "labels.npy", labels)
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    _write_report(output_dir, raw_dir, nodes, edges, feature_matrix, split_stats)
    return output_dir


def _elliptic_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "illicit", "fraud", "true"}:
        return 1
    if text in {"2", "2.0", "licit", "benign", "false", "0", "0.0"}:
        return 0
    return -1


def _normalize_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.lstrip("-").isdigit():
            return head
    return text


def _labeled_split(node_ids: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    if int(np.sum(labels >= 0)) < 3:
        raise ValueError("Elliptic split failed: fewer than 3 labeled nodes are available.")
    if int(np.sum(labels == 0)) == 0 or int(np.sum(labels == 1)) == 0:
        raise ValueError("Elliptic split failed: both licit and illicit labeled nodes are required.")
    for label in [0, 1]:
        labeled = np.asarray([idx for idx, value in enumerate(labels.tolist()) if int(value) == label], dtype=np.int64)
        rng.shuffle(labeled)
        n = int(labeled.size)
        if n >= 3:
            train_end = max(1, int(round(n * 0.6)))
            val_size = max(1, int(round(n * 0.2)))
            if train_end + val_size >= n:
                train_end = max(1, n - 2)
                val_size = 1
            val_end = train_end + val_size
        else:
            train_end = 1
            val_end = min(n, 2)
        split["train"].extend(node_ids[labeled[:train_end]].astype(str).tolist())
        split["val"].extend(node_ids[labeled[train_end:val_end]].astype(str).tolist())
        split["test"].extend(node_ids[labeled[val_end:]].astype(str).tolist())
    for ids in split.values():
        ids.sort()
    if not all(split[name] for name in ["train", "val", "test"]):
        raise ValueError("Elliptic split failed: train/val/test must each contain labeled nodes.")
    return split


def check_elliptic_processed(output_dir: str | Path = "data/processed/elliptic") -> dict[str, Any]:
    output_dir = Path(output_dir)
    nodes_path = output_dir / "nodes.csv"
    split_path = output_dir / "split.json"
    _require_files([nodes_path, split_path])
    nodes = pd.read_csv(nodes_path).fillna("")
    labels = pd.to_numeric(nodes[schema.LABEL], errors="coerce").fillna(-1).astype(int)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    label_by_id = nodes.set_index(schema.NODE_ID)[schema.LABEL].astype(int).to_dict()
    payload = {
        "num_nodes": int(nodes.shape[0]),
        "num_labeled": int((labels >= 0).sum()),
        "num_licit": int((labels == 0).sum()),
        "num_illicit": int((labels == 1).sum()),
        "num_unknown": int((labels < 0).sum()),
        "splits": _split_stats(split, {str(k): int(v) for k, v in label_by_id.items()}),
    }
    return payload


def format_elliptic_check_report(payload: dict[str, Any]) -> str:
    lines = [
        f"labeled node count: {payload['num_labeled']}",
        f"licit count: {payload['num_licit']}",
        f"illicit count: {payload['num_illicit']}",
        f"unknown count: {payload['num_unknown']}",
    ]
    for split_name in ["train", "val", "test"]:
        stats = payload["splits"].get(split_name, {})
        lines.append(
            f"{split_name} labeled count: {stats.get('labeled_count', 0)} "
            f"(licit={stats.get('licit', 0)}, illicit={stats.get('illicit', 0)}, unknown={stats.get('unknown', 0)})"
        )
    return "\n".join(lines)


def _read_feature_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    first = str(frame.iloc[0, 0]).strip().lower() if not frame.empty else ""
    if first in {"txid", "tx_id", "node_id"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


def _read_class_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.empty:
        raise ValueError("Elliptic class file is empty.")
    first = [str(value).strip() for value in raw.iloc[0].tolist()]
    if any(value.lower() in {"txid", "tx_id", "node_id"} for value in first):
        columns = first
        frame = raw.iloc[1:].reset_index(drop=True)
        frame.columns = columns
        return frame
    frame = raw.iloc[:, :2].copy()
    frame.columns = ["txId", "class"]
    return frame


def _read_edge_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return pd.DataFrame(columns=["txId1", "txId2"])
    first = [str(value).strip() for value in raw.iloc[0].tolist()]
    if any(value.lower() in {"txid1", "tx_id1", "src", "source", "from"} for value in first):
        columns = first
        frame = raw.iloc[1:].reset_index(drop=True)
        frame.columns = columns
        return frame
    frame = raw.iloc[:, :2].copy()
    frame.columns = ["txId1", "txId2"]
    return frame


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of columns {names}, found {list(frame.columns)}")


def _split_stats(split: dict[str, list[str]], labels_by_id: dict[str, int]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split_name, ids in split.items():
        labels = [int(labels_by_id.get(str(node_id), -1)) for node_id in ids]
        out[split_name] = {
            "labeled_count": int(sum(label >= 0 for label in labels)),
            "licit": int(sum(label == 0 for label in labels)),
            "illicit": int(sum(label == 1 for label in labels)),
            "unknown": int(sum(label < 0 for label in labels)),
        }
    return out


def _write_report(output_dir: Path, raw_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, features: np.ndarray, split_stats: dict[str, dict[str, int]]) -> None:
    labels = pd.to_numeric(nodes[schema.LABEL], errors="coerce").fillna(-1).astype(int)
    payload = {
        "dataset": "elliptic",
        "dataset_type": "transaction_graph",
        "raw_dir": str(raw_dir),
        "num_nodes": int(nodes.shape[0]),
        "num_edges": int(edges.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_labeled": int((labels >= 0).sum()),
        "num_licit": int((labels == 0).sum()),
        "num_illicit": int((labels == 1).sum()),
        "num_unknown": int((labels < 0).sum()),
        "train_labeled_count": int(split_stats["train"]["labeled_count"]),
        "val_labeled_count": int(split_stats["val"]["labeled_count"]),
        "test_labeled_count": int(split_stats["test"]["labeled_count"]),
        "label_mapping": LABEL_MAPPING,
        "label_source": "official",
        "split_strategy": "stratified_labeled_nodes_60_20_20_min_one_per_split",
        "split_class_distribution": split_stats,
        "unknown_label_policy": "kept_as_graph_context_excluded_from_loss",
    }
    (output_dir / "preprocess_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files. This is expected on local VSCode. "
            f"Please run on AutoDL or provide data path. Missing: {missing}"
        )
