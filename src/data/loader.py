from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import schema
from src.data.schema import GraphData

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class ProcessedGraphData:
    features: np.ndarray
    text_features: np.ndarray
    numeric_features: np.ndarray
    labels: np.ndarray
    split: dict[str, np.ndarray]
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    edges: pd.DataFrame
    nodes: pd.DataFrame
    node_id_to_idx: dict[str, int]
    edge_index: np.ndarray
    evidence_gt: dict
    preprocess_report: dict


def load_processed_data(data_dir: str | Path = "data/processed/synthetic") -> ProcessedGraphData:
    data_dir = Path(data_dir)
    if _is_elliptic_dir(data_dir):
        return _load_elliptic_processed_data(data_dir)

    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    features_path = data_dir / "features.npz"
    split_path = data_dir / "split.json"
    evidence_path = data_dir / "evidence_gt.json"
    report_path = data_dir / "preprocess_report.json"
    metadata_path = data_dir / "metadata.json"

    _require_files([nodes_path, edges_path, features_path, split_path])

    nodes = _read_nodes(nodes_path)
    edges = _read_edges(edges_path)
    feature_payload = np.load(features_path, allow_pickle=True)
    node_ids = [_normalize_node_id(value) for value in feature_payload["node_ids"].tolist()]
    features = feature_payload["features"].astype(np.float32)
    text_features = feature_payload["text_features"].astype(np.float32)
    numeric_features = feature_payload["numeric_features"].astype(np.float32)

    node_id_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}
    labels = _labels_from_nodes(nodes, node_ids)

    split = _split_from_json(split_path, node_id_to_idx, num_nodes=len(node_ids))
    train_mask, val_mask, test_mask = _masks_from_split(split, len(node_ids))

    edge_index = _edge_index_from_edges(edges, node_id_to_idx)
    evidence_gt = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    if report_path.exists():
        preprocess_report = json.loads(report_path.read_text(encoding="utf-8"))
    elif metadata_path.exists():
        preprocess_report = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        preprocess_report = {}
    return ProcessedGraphData(
        features=features,
        text_features=text_features,
        numeric_features=numeric_features,
        labels=labels,
        split=split,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edges=edges,
        nodes=nodes,
        node_id_to_idx=node_id_to_idx,
        edge_index=edge_index,
        evidence_gt=evidence_gt,
        preprocess_report=preprocess_report,
    )


def load_graph(path: str | Path) -> GraphData | ProcessedGraphData:
    path = Path(path)
    if path.is_dir():
        return load_processed_data(path)
    return GraphData.load_npz(path)


def validate_labeled_split(data: ProcessedGraphData) -> dict[str, Any]:
    labels = np.asarray(data.labels, dtype=np.int64)
    masks = {
        "train": _mask_from_data(data, "train"),
        "val": _mask_from_data(data, "val"),
        "test": _mask_from_data(data, "test"),
    }
    result: dict[str, Any] = {
        "valid": True,
        "num_labeled": int(np.sum(labels >= 0)),
        "label_distribution": _label_distribution(labels),
    }
    for split_name, mask in masks.items():
        labeled_mask = mask & (labels >= 0)
        count = int(np.sum(labeled_mask))
        result[f"{split_name}_count"] = count
        result[f"{split_name}_class_distribution"] = _label_distribution(labels[labeled_mask])
        if count <= 0:
            result["valid"] = False
    if not result["valid"]:
        result["warning"] = "processed split has no labeled train/val/test nodes"
    return result


def processed_dataset_diagnostics(
    data: ProcessedGraphData,
    processed_dir: str | Path,
    dataset: str | None = None,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    split_status = validate_labeled_split(data)
    return {
        "dataset": dataset or data.preprocess_report.get("dataset", processed_dir.name),
        "dataset_type": data.preprocess_report.get("dataset_type", "unknown"),
        "processed_dir": str(processed_dir),
        "num_nodes": int(data.features.shape[0]),
        "num_edges": int(data.edge_index.shape[1]) if data.edge_index.ndim == 2 else 0,
        "feature_shape": list(data.features.shape),
        "label_shape": list(data.labels.shape),
        "num_labeled": split_status["num_labeled"],
        "label_distribution": split_status["label_distribution"],
        "train_count": split_status["train_count"],
        "val_count": split_status["val_count"],
        "test_count": split_status["test_count"],
        "train_class_distribution": split_status["train_class_distribution"],
        "val_class_distribution": split_status["val_class_distribution"],
        "test_class_distribution": split_status["test_class_distribution"],
        "available_files": sorted(path.name for path in processed_dir.iterdir()) if processed_dir.exists() else [],
        "required_fields_status": _required_fields_status(data),
    }


def _load_elliptic_processed_data(data_dir: Path) -> ProcessedGraphData:
    report_path = data_dir / "preprocess_report.json"
    metadata_path = data_dir / "metadata.json"
    preprocess_report = _read_json(report_path) or _read_json(metadata_path) or {"dataset": "elliptic", "dataset_type": "transaction_graph"}

    features, node_ids, text_features, numeric_features = _load_feature_block(data_dir)
    num_nodes = int(features.shape[0])
    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    nodes = _read_nodes(nodes_path) if nodes_path.exists() else _make_nodes(node_ids, np.full(num_nodes, -1, dtype=np.int64))
    if node_ids is None:
        node_ids = _node_ids_from_nodes_or_range(nodes, num_nodes)
    node_ids = [_normalize_node_id(value) for value in node_ids]
    if len(node_ids) != num_nodes:
        raise ValueError(f"Elliptic node id count {len(node_ids)} does not match features rows {num_nodes}.")
    node_id_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}

    labels = _load_labels(data_dir, nodes, node_ids, num_nodes)
    nodes = _ensure_nodes_frame(nodes, node_ids, labels)

    edge_index = _load_edge_index(data_dir, node_id_to_idx)
    edges = _read_edges(edges_path) if edges_path.exists() else _make_edges(edge_index, node_ids)

    train_mask, val_mask, test_mask = _load_masks(data_dir, node_id_to_idx, num_nodes, nodes)
    _validate_mask_shapes(num_nodes, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    split = {
        "train": np.flatnonzero(train_mask).astype(np.int64),
        "val": np.flatnonzero(val_mask).astype(np.int64),
        "test": np.flatnonzero(test_mask).astype(np.int64),
    }
    nodes[schema.SPLIT] = _split_column_from_masks(num_nodes, train_mask, val_mask, test_mask)
    preprocess_report.setdefault("dataset", "elliptic")
    preprocess_report.setdefault("dataset_type", "transaction_graph")
    preprocess_report.setdefault("label_source", "official")

    return ProcessedGraphData(
        features=features.astype(np.float32),
        text_features=text_features.astype(np.float32),
        numeric_features=numeric_features.astype(np.float32),
        labels=labels.astype(np.int64),
        split=split,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edges=edges,
        nodes=nodes,
        node_id_to_idx=node_id_to_idx,
        edge_index=edge_index.astype(np.int64),
        evidence_gt={},
        preprocess_report=preprocess_report,
    )


def _is_elliptic_dir(data_dir: Path) -> bool:
    if data_dir.name.lower() == "elliptic":
        return True
    metadata = _read_json(data_dir / "metadata.json")
    return str((metadata or {}).get("dataset", "")).lower() == "elliptic"


def _load_feature_block(data_dir: Path) -> tuple[np.ndarray, list[str] | None, np.ndarray, np.ndarray]:
    npz_path = data_dir / "features.npz"
    if npz_path.exists():
        payload = np.load(npz_path, allow_pickle=True)
        features = payload["features"].astype(np.float32)
        node_ids = [_normalize_node_id(value) for value in payload["node_ids"].tolist()] if "node_ids" in payload else None
        text_features = payload["text_features"].astype(np.float32) if "text_features" in payload else np.zeros((features.shape[0], 0), dtype=np.float32)
        numeric_features = payload["numeric_features"].astype(np.float32) if "numeric_features" in payload else features.astype(np.float32)
        return features, node_ids, text_features, numeric_features
    features = _load_first_array(data_dir, ["features.npy", "features.pt"], "features")
    features = np.asarray(features, dtype=np.float32)
    return features, None, np.zeros((features.shape[0], 0), dtype=np.float32), features.astype(np.float32)


def _load_labels(data_dir: Path, nodes: pd.DataFrame, node_ids: list[str], num_nodes: int) -> np.ndarray:
    for name in ["labels.npy", "labels.pt"]:
        path = data_dir / name
        if path.exists():
            labels = np.asarray(_load_array(path), dtype=np.int64).reshape(-1)
            if labels.shape[0] != num_nodes:
                raise ValueError(f"Elliptic labels length {labels.shape[0]} does not match num_nodes={num_nodes}.")
            return labels
    if schema.LABEL in nodes:
        return _labels_from_nodes(nodes, node_ids)
    raise FileNotFoundError(f"Missing Elliptic labels file in {data_dir}; expected labels.npy or labels.pt.")


def _load_edge_index(data_dir: Path, node_id_to_idx: dict[str, int]) -> np.ndarray:
    for name in ["edge_index.npy", "edge_index.pt"]:
        path = data_dir / name
        if path.exists():
            edge_index = np.asarray(_load_array(path), dtype=np.int64)
            if edge_index.ndim != 2:
                raise ValueError(f"Elliptic edge_index must be 2-D, found shape={edge_index.shape}.")
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.T
            if edge_index.shape[0] != 2:
                raise ValueError(f"Elliptic edge_index must have shape [2, num_edges], found {edge_index.shape}.")
            return edge_index
    edges_path = data_dir / "edges.csv"
    if edges_path.exists():
        return _edge_index_from_edges(_read_edges(edges_path), node_id_to_idx)
    return np.zeros((2, 0), dtype=np.int64)


def _load_masks(
    data_dir: Path,
    node_id_to_idx: dict[str, int],
    num_nodes: int,
    nodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    individual = _load_individual_masks(data_dir, num_nodes)
    if individual is not None:
        return individual
    packed = _load_packed_masks(data_dir, num_nodes)
    if packed is not None:
        return packed
    split_path = data_dir / "split.json"
    if split_path.exists():
        split = _split_from_json(split_path, node_id_to_idx, num_nodes=num_nodes)
        return _masks_from_split(split, num_nodes)
    if schema.SPLIT in nodes:
        return _masks_from_split_column(nodes[schema.SPLIT].astype(str).tolist(), num_nodes)
    raise FileNotFoundError(f"Missing Elliptic masks in {data_dir}; expected train/val/test masks, masks.pt, or split.json.")


def _load_individual_masks(data_dir: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    masks = []
    for split_name in ["train", "val", "test"]:
        loaded = None
        for suffix in [".npy", ".pt"]:
            path = data_dir / f"{split_name}_mask{suffix}"
            if path.exists():
                loaded = _as_bool_mask(_load_array(path), num_nodes, f"{split_name}_mask")
                break
        if loaded is None:
            return None
        masks.append(loaded)
    return masks[0], masks[1], masks[2]


def _load_packed_masks(data_dir: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for name in ["masks.pt", "masks.npy", "masks.npz"]:
        path = data_dir / name
        if not path.exists():
            continue
        payload = _load_array(path)
        if isinstance(payload, np.lib.npyio.NpzFile):
            return (
                _as_bool_mask(payload["train_mask"], num_nodes, "train_mask"),
                _as_bool_mask(payload["val_mask"], num_nodes, "val_mask"),
                _as_bool_mask(payload["test_mask"], num_nodes, "test_mask"),
            )
        if isinstance(payload, dict):
            return (
                _as_bool_mask(payload.get("train_mask", payload.get("train")), num_nodes, "train_mask"),
                _as_bool_mask(payload.get("val_mask", payload.get("val")), num_nodes, "val_mask"),
                _as_bool_mask(payload.get("test_mask", payload.get("test")), num_nodes, "test_mask"),
            )
        array = np.asarray(payload)
        if array.dtype == object and array.shape == ():
            obj = array.item()
            if isinstance(obj, dict):
                return (
                    _as_bool_mask(obj.get("train_mask", obj.get("train")), num_nodes, "train_mask"),
                    _as_bool_mask(obj.get("val_mask", obj.get("val")), num_nodes, "val_mask"),
                    _as_bool_mask(obj.get("test_mask", obj.get("test")), num_nodes, "test_mask"),
                )
    return None


def _load_first_array(data_dir: Path, names: list[str], label: str) -> np.ndarray:
    for name in names:
        path = data_dir / name
        if path.exists():
            return np.asarray(_load_array(path))
    raise FileNotFoundError(f"Missing {label} file in {data_dir}; expected one of {names}.")


def _load_array(path: Path) -> Any:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".pt":
        if torch is None:
            raise ImportError(f"Cannot read {path}; torch is not installed.")
        value = torch.load(path, map_location="cpu")
        return _torch_to_numpy(value)
    raise ValueError(f"Unsupported array file: {path}")


def _torch_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _torch_to_numpy(val) for key, val in value.items()}
    if torch is not None and hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _read_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path).fillna("")
    if schema.NODE_ID in nodes:
        nodes[schema.NODE_ID] = nodes[schema.NODE_ID].map(_normalize_node_id)
    return nodes


def _read_edges(path: Path) -> pd.DataFrame:
    edges = pd.read_csv(path).fillna("")
    for col in [schema.SRC, schema.DST]:
        if col in edges:
            edges[col] = edges[col].map(_normalize_node_id)
    if schema.EDGE_TYPE not in edges:
        edges[schema.EDGE_TYPE] = "edge"
    if schema.TIMESTAMP not in edges:
        edges[schema.TIMESTAMP] = ""
    return edges


def _normalize_node_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.lstrip("-").isdigit():
            return head
    return text


def _labels_from_nodes(nodes: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    normalized = nodes.copy()
    normalized[schema.NODE_ID] = normalized[schema.NODE_ID].map(_normalize_node_id)
    labels_by_id = pd.to_numeric(normalized.set_index(schema.NODE_ID)[schema.LABEL], errors="coerce").fillna(-1).astype(int).to_dict()
    return np.array([int(labels_by_id.get(node_id, -1)) for node_id in node_ids], dtype=np.int64)


def _split_from_json(split_path: Path, node_id_to_idx: dict[str, int], num_nodes: int) -> dict[str, np.ndarray]:
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    split = {}
    for name, ids in split_ids.items():
        values = []
        for value in ids:
            node_id = _normalize_node_id(value)
            if node_id in node_id_to_idx:
                values.append(node_id_to_idx[node_id])
                continue
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_nodes:
                values.append(idx)
        split[name] = np.array(values, dtype=np.int64)
    return split


def _masks_from_split(split: dict[str, np.ndarray], num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = []
    for split_name in ["train", "val", "test"]:
        mask = np.zeros(num_nodes, dtype=bool)
        indices = np.asarray(split.get(split_name, np.array([], dtype=np.int64)), dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < num_nodes)]
        mask[indices] = True
        masks.append(mask)
    return masks[0], masks[1], masks[2]


def _masks_from_split_column(values: list[str], num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) != num_nodes:
        raise ValueError(f"Split column length {len(values)} does not match num_nodes={num_nodes}.")
    lower = np.asarray([value.strip().lower() for value in values], dtype=object)
    return lower == "train", lower == "val", lower == "test"


def _edge_index_from_edges(edges: pd.DataFrame, node_id_to_idx: dict[str, int]) -> np.ndarray:
    if not {schema.SRC, schema.DST}.issubset(edges.columns):
        return np.zeros((2, 0), dtype=np.int64)
    edge_pairs = [
        (node_id_to_idx[src], node_id_to_idx[dst])
        for src, dst in edges[[schema.SRC, schema.DST]].itertuples(index=False, name=None)
        if src in node_id_to_idx and dst in node_id_to_idx
    ]
    return np.array(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64)


def _node_ids_from_nodes_or_range(nodes: pd.DataFrame, num_nodes: int) -> list[str]:
    if schema.NODE_ID in nodes and nodes.shape[0] == num_nodes:
        return nodes[schema.NODE_ID].map(_normalize_node_id).astype(str).tolist()
    return [str(index) for index in range(num_nodes)]


def _make_nodes(node_ids: list[str] | None, labels: np.ndarray) -> pd.DataFrame:
    node_ids = node_ids or [str(index) for index in range(labels.shape[0])]
    return pd.DataFrame(
        {
            schema.NODE_ID: node_ids,
            schema.NODE_TYPE: "transaction",
            schema.TEXT: "",
            schema.LABEL: labels.astype(np.int64),
            schema.SPLIT: "",
            schema.TIMESTAMP: "",
        }
    )


def _ensure_nodes_frame(nodes: pd.DataFrame, node_ids: list[str], labels: np.ndarray) -> pd.DataFrame:
    if nodes.shape[0] != len(node_ids) or schema.NODE_ID not in nodes:
        return _make_nodes(node_ids, labels)
    nodes = nodes.copy()
    nodes[schema.NODE_ID] = nodes[schema.NODE_ID].map(_normalize_node_id)
    nodes[schema.LABEL] = labels.astype(np.int64)
    if schema.NODE_TYPE not in nodes:
        nodes[schema.NODE_TYPE] = "transaction"
    if schema.TEXT not in nodes:
        nodes[schema.TEXT] = ""
    if schema.SPLIT not in nodes:
        nodes[schema.SPLIT] = ""
    if schema.TIMESTAMP not in nodes:
        nodes[schema.TIMESTAMP] = ""
    return nodes


def _make_edges(edge_index: np.ndarray, node_ids: list[str]) -> pd.DataFrame:
    if edge_index.size == 0:
        return pd.DataFrame(columns=[schema.SRC, schema.DST, schema.EDGE_TYPE, schema.TIMESTAMP])
    src = [node_ids[int(idx)] for idx in edge_index[0]]
    dst = [node_ids[int(idx)] for idx in edge_index[1]]
    return pd.DataFrame({schema.SRC: src, schema.DST: dst, schema.EDGE_TYPE: "transaction-transfer", schema.TIMESTAMP: ""})


def _as_bool_mask(value: Any, num_nodes: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing {name}.")
    array = np.asarray(value)
    if array.dtype != bool:
        array = array.astype(bool)
    array = array.reshape(-1)
    if array.shape[0] != num_nodes:
        raise ValueError(f"{name} length {array.shape[0]} does not match num_nodes={num_nodes}.")
    return array


def _validate_mask_shapes(num_nodes: int, **masks: np.ndarray) -> None:
    for name, mask in masks.items():
        if mask.shape != (num_nodes,):
            raise ValueError(f"{name} shape {mask.shape} does not match [{num_nodes}].")


def _split_column_from_masks(num_nodes: int, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> list[str]:
    values = [""] * num_nodes
    for idx in np.flatnonzero(train_mask):
        values[int(idx)] = "train"
    for idx in np.flatnonzero(val_mask):
        values[int(idx)] = "val"
    for idx in np.flatnonzero(test_mask):
        values[int(idx)] = "test"
    return values


def _mask_from_data(data: ProcessedGraphData, split_name: str) -> np.ndarray:
    direct = getattr(data, f"{split_name}_mask", None)
    if direct is not None:
        return np.asarray(direct, dtype=bool)
    mask = np.zeros(data.labels.shape[0], dtype=bool)
    indices = np.asarray(data.split.get(split_name, np.array([], dtype=np.int64)), dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < mask.shape[0])]
    mask[indices] = True
    return mask


def _label_distribution(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        "licit": int(np.sum(labels == 0)),
        "illicit": int(np.sum(labels == 1)),
        "unknown": int(np.sum(labels < 0)),
    }


def _required_fields_status(data: ProcessedGraphData) -> dict[str, bool]:
    return {
        "features": data.features.ndim == 2 and data.features.shape[0] == data.labels.shape[0],
        "edge_index": data.edge_index.ndim == 2 and data.edge_index.shape[0] == 2,
        "labels": data.labels.ndim == 1 and data.labels.shape[0] == data.features.shape[0],
        "train_mask": data.train_mask.shape == data.labels.shape,
        "val_mask": data.val_mask.shape == data.labels.shape,
        "test_mask": data.test_mask.shape == data.labels.shape,
        "metadata": bool(data.preprocess_report),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed data file(s): {missing}")
