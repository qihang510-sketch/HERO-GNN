from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import schema
from src.data.split import split_ids

DGL_REQUIRED_MESSAGE = (
    "DGL is required for this preprocessing script.\n"
    "Please install DGL following the official installation guide."
)

DGL_FRAUD_DATASETS = {
    "fraud_yelp_official": {
        "class_name": "FraudYelpDataset",
        "node_type": "review",
    },
    "fraud_amazon_official": {
        "class_name": "FraudAmazonDataset",
        "node_type": "user",
    },
}


def preprocess_dgl_fraud_dataset(
    dataset: str,
    output_dir: str | Path,
    seed: int = 0,
    raw_dir: str | Path | None = None,
    train_size: float = 0.7,
    val_size: float = 0.1,
) -> Path:
    graph = _load_dgl_fraud_graph(
        dataset=dataset,
        raw_dir=raw_dir,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
    )
    return preprocess_dgl_fraud_graph(graph=graph, dataset=dataset, output_dir=output_dir, seed=seed)


def preprocess_dgl_fraud_graph(
    graph: Any,
    dataset: str,
    output_dir: str | Path,
    seed: int = 0,
) -> Path:
    if dataset not in DGL_FRAUD_DATASETS:
        raise ValueError(f"Unknown DGL fraud dataset: {dataset}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_type = DGL_FRAUD_DATASETS[dataset]["node_type"]
    dgl_node_type = _resolve_graph_node_type(graph, node_type)
    features = _extract_features(graph, dgl_node_type)
    labels = _extract_labels(graph, dgl_node_type, expected_rows=features.shape[0])
    node_ids = [f"{node_type}_{index}" for index in range(features.shape[0])]

    splits, split_source, labeled_mask = _extract_split(node_ids, graph, dgl_node_type, seed=seed)
    node_labels = labels.astype(np.int64, copy=True)
    if split_source == "official_mask":
        node_labels[~labeled_mask] = -1

    nodes = pd.DataFrame(
        {
            schema.NODE_ID: node_ids,
            schema.NODE_TYPE: node_type,
            schema.TEXT: "",
            schema.LABEL: node_labels,
            schema.SPLIT: [splits.get(node_id, "unlabeled") for node_id in node_ids],
            schema.TIMESTAMP: 0,
        }
    )
    edges = _extract_edges(graph, node_ids)

    _write_processed(output_dir, nodes, edges, features)
    _write_report(output_dir, dataset, nodes, edges, features, split_source)
    return output_dir


def _load_dgl_fraud_graph(
    dataset: str,
    raw_dir: str | Path | None,
    seed: int,
    train_size: float,
    val_size: float,
) -> Any:
    if dataset not in DGL_FRAUD_DATASETS:
        raise ValueError(f"Unknown DGL fraud dataset: {dataset}")
    try:
        import dgl.data as dgl_data
    except ImportError as exc:
        raise ImportError(DGL_REQUIRED_MESSAGE) from exc

    dataset_class = getattr(dgl_data, DGL_FRAUD_DATASETS[dataset]["class_name"])
    kwargs = _supported_constructor_kwargs(
        dataset_class,
        {
            "raw_dir": str(raw_dir) if raw_dir is not None else None,
            "random_seed": int(seed),
            "train_size": float(train_size),
            "val_size": float(val_size),
            "verbose": True,
        },
    )
    dgl_dataset = dataset_class(**kwargs)
    return dgl_dataset[0]


def _supported_constructor_kwargs(dataset_class: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(dataset_class)
    except (TypeError, ValueError):
        return {key: value for key, value in values.items() if value is not None}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in values.items() if value is not None}
    supported = set(signature.parameters)
    return {
        key: value
        for key, value in values.items()
        if key in supported and value is not None
    }


def _resolve_graph_node_type(graph: Any, preferred_node_type: str) -> str | None:
    node_types = list(getattr(graph, "ntypes", []) or [])
    if preferred_node_type in node_types:
        return preferred_node_type
    if len(node_types) == 1:
        return str(node_types[0])
    return preferred_node_type if node_types else None


def _extract_features(graph: Any, node_type: str | None) -> np.ndarray:
    features = _extract_node_data(graph, node_type, ("feature", "feat", "features"))
    if features is None:
        raise ValueError("DGL fraud graph must provide node features under 'feature'.")
    values = _to_numpy(features).astype(np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _extract_labels(graph: Any, node_type: str | None, expected_rows: int) -> np.ndarray:
    labels = _extract_node_data(graph, node_type, ("label", "labels"))
    if labels is None:
        raise ValueError("DGL fraud graph must provide node labels under 'label'.")
    values = _to_numpy(labels).reshape(-1).astype(np.int64)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"Label count ({values.shape[0]}) does not match feature rows ({expected_rows})."
        )
    return values


def _extract_split(
    node_ids: list[str],
    graph: Any,
    node_type: str | None,
    seed: int,
) -> tuple[dict[str, str], str, np.ndarray]:
    masks = {
        name: _extract_mask(graph, node_type, f"{name}_mask", len(node_ids))
        for name in ("train", "val", "test")
    }
    if all(mask is not None for mask in masks.values()):
        assigned = np.zeros(len(node_ids), dtype=bool)
        lookup: dict[str, str] = {}
        for name in ("train", "val", "test"):
            mask = masks[name]
            assert mask is not None
            assigned |= mask
            for node_id in np.asarray(node_ids, dtype=object)[mask].tolist():
                lookup[str(node_id)] = name
        if masks["train"].any() and masks["test"].any():
            return lookup, "official_mask", assigned

    split = split_ids(node_ids, seed=seed, train=0.6, val=0.2)
    lookup = {node_id: name for name, values in split.items() for node_id in values}
    return lookup, "generated_60_20_20", np.ones(len(node_ids), dtype=bool)


def _extract_mask(graph: Any, node_type: str | None, key: str, expected_rows: int) -> np.ndarray | None:
    mask = _extract_node_data(graph, node_type, (key,))
    if mask is None:
        return None
    values = _to_numpy(mask).reshape(-1).astype(bool)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"Mask {key} length ({values.shape[0]}) does not match node count ({expected_rows})."
        )
    return values


def _extract_node_data(graph: Any, node_type: str | None, keys: tuple[str, ...]) -> Any | None:
    nodes_accessor = getattr(graph, "nodes", None)
    if nodes_accessor is not None and node_type is not None:
        for key in keys:
            try:
                node_view = nodes_accessor[node_type]
                if key in node_view.data:
                    return node_view.data[key]
            except (AttributeError, KeyError, TypeError):
                pass

    ndata = getattr(graph, "ndata", None)
    for key in keys:
        try:
            value = ndata[key]
        except (TypeError, KeyError, AttributeError):
            continue
        if isinstance(value, Mapping):
            if node_type in value:
                return value[node_type]
            if len(value) == 1:
                return next(iter(value.values()))
        return value
    return None


def _extract_edges(graph: Any, node_ids: list[str]) -> pd.DataFrame:
    frames = []
    canonical_etypes = list(getattr(graph, "canonical_etypes", []) or [])
    if canonical_etypes:
        for index, etype in enumerate(canonical_etypes):
            src, dst = _graph_edges(graph, etype)
            frames.append(_edge_frame(src, dst, node_ids, _relation_name(etype, index)))
    else:
        src, dst = _graph_edges(graph, None)
        frames.append(_edge_frame(src, dst, node_ids, "rel_0"))
    if not frames:
        return pd.DataFrame(columns=[schema.SRC, schema.DST, schema.EDGE_TYPE])
    return pd.concat(frames, ignore_index=True)


def _graph_edges(graph: Any, etype: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if etype is None:
        src, dst = graph.edges()
    else:
        try:
            src, dst = graph.edges(etype=etype)
        except (TypeError, KeyError):
            src, dst = graph.edges(etype=etype[1] if isinstance(etype, tuple) and len(etype) > 1 else etype)
    return _to_numpy(src).reshape(-1).astype(np.int64), _to_numpy(dst).reshape(-1).astype(np.int64)


def _relation_name(etype: Any, index: int) -> str:
    if isinstance(etype, tuple) and len(etype) >= 2 and etype[1]:
        return str(etype[1])
    if etype:
        return str(etype)
    return f"rel_{index}"


def _edge_frame(src: np.ndarray, dst: np.ndarray, node_ids: list[str], edge_type: str) -> pd.DataFrame:
    if src.size == 0:
        return pd.DataFrame(columns=[schema.SRC, schema.DST, schema.EDGE_TYPE])
    node_id_array = np.asarray(node_ids, dtype=object)
    valid = (src >= 0) & (dst >= 0) & (src < len(node_ids)) & (dst < len(node_ids))
    src = src[valid]
    dst = dst[valid]
    return pd.DataFrame(
        {
            schema.SRC: node_id_array[src],
            schema.DST: node_id_array[dst],
            schema.EDGE_TYPE: edge_type,
        }
    )


def _write_processed(output_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, features: np.ndarray) -> None:
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)
    text_features = np.zeros((features.shape[0], 1), dtype=np.float32)
    numeric_features = features.astype(np.float32, copy=False)
    np.savez_compressed(
        output_dir / "features.npz",
        node_ids=nodes[schema.NODE_ID].to_numpy(dtype=object),
        features=numeric_features,
        text_features=text_features,
        numeric_features=numeric_features,
        numeric_columns=np.array([f"dgl_feat_{index}" for index in range(numeric_features.shape[1])], dtype=object),
    )
    split = {
        name: nodes.loc[nodes[schema.SPLIT] == name, schema.NODE_ID].tolist()
        for name in ("train", "val", "test")
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(
    output_dir: Path,
    dataset: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    features: np.ndarray,
    split_source: str,
) -> None:
    labels = nodes[schema.LABEL].to_numpy(dtype=np.int64)
    labeled = labels >= 0
    num_positive = int(np.sum(labels[labeled] == 1))
    num_negative = int(np.sum(labels[labeled] == 0))
    denominator = max(num_positive + num_negative, 1)
    report = {
        "dataset": dataset,
        "label_source": "official",
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "positive_rate": float(num_positive / denominator),
        "feature_dim": int(features.shape[1]),
        "split_source": split_source,
        "num_unlabeled": int(np.sum(~labeled)),
    }
    (output_dir / "preprocess_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)
