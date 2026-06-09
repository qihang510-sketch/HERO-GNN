from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import schema
from src.data.loader import ProcessedGraphData, load_processed_data, validate_labeled_split
from src.training.evaluator import (
    binary_classification_metrics,
    fixed_threshold_diagnostics,
    prediction_probability_stats,
    tune_threshold,
)
from src.training.trainer import train_single_experiment
from src.utils.io import write_json
from src.utils.seed import set_seed


TEXT_RICH_DATASETS = ("yelp_academic", "amazon_video")
OFFICIAL_DATASETS = ("fraud_yelp", "fraud_amazon")
TRANSACTION_DATASETS = ("elliptic",)
SUBMISSION_DATASETS = (*TEXT_RICH_DATASETS, *OFFICIAL_DATASETS, *TRANSACTION_DATASETS)

TEXT_RICH_MODELS = (
    "mlp",
    "gcn",
    "gat",
    "graphsage",
    "care_gnn",
    "graphconsis",
    "pc_gnn",
    "bwgnn",
    "linkx",
    "dgp",
    "mled",
    "hero_gnn",
)
OFFICIAL_MODELS = (
    "mlp",
    "gcn",
    "gat",
    "graphsage",
    "care_gnn",
    "graphconsis",
    "pc_gnn",
    "bwgnn",
    "linkx",
    "hero_official",
)
TRANSACTION_MODELS = (
    "mlp",
    "gcn",
    "gat",
    "graphsage",
    "bwgnn",
    "linkx",
    "hogrl",
    "rgtan",
    "hero_official",
)
DATASET_MODEL_MATRIX = {
    "yelp_academic": TEXT_RICH_MODELS,
    "amazon_video": TEXT_RICH_MODELS,
    "fraud_yelp": OFFICIAL_MODELS,
    "fraud_amazon": OFFICIAL_MODELS,
    "elliptic": TRANSACTION_MODELS,
}

MODEL_ALIASES = {
    "MLP": "mlp",
    "GCN": "gcn",
    "GAT": "gat",
    "GraphSAGE": "graphsage",
    "CARE-GNN": "care_gnn",
    "CARE_GNN": "care_gnn",
    "GraphConsis": "graphconsis",
    "PC-GNN": "pc_gnn",
    "PC_GNN": "pc_gnn",
    "BWGNN": "bwgnn",
    "LINKX": "linkx",
    "DGP": "dgp",
    "MLED": "mled",
    "HOGRL": "hogrl",
    "RGTAN": "rgtan",
    "HERO-GNN": "hero_gnn",
    "HERO-official": "hero_official",
    "HERO_OFFICIAL": "hero_official",
}
FORBIDDEN_SUBMISSION_NAMES = {
    "care_gnn_lite",
    "graphconsis_lite",
    "pc_gnn_lite",
    "bwgnn_lite",
    "dgp_lite",
    "mled_lite",
    "hogrl_lite",
    "rgtan_lite",
    "flag_lite",
    "sec_gfd_lite",
    "dga_gnn_lite",
}
MODEL_IMPLEMENTATION_SOURCE = {
    "mlp": "benchmark",
    "gcn": "benchmark",
    "gat": "benchmark",
    "graphsage": "benchmark",
    "care_gnn": "reproduced",
    "graphconsis": "reproduced",
    "pc_gnn": "reproduced",
    "bwgnn": "reproduced",
    "linkx": "reproduced",
    "dgp": "reproduced",
    "mled": "reproduced",
    "hogrl": "reproduced",
    "rgtan": "reproduced",
    "hero_gnn": "project",
    "hero_official": "project",
}
HERO_MODELS = {"hero_gnn", "hero_official"}


@dataclass
class SubmissionResult:
    dataset: str
    model: str
    seed: int
    status: str
    path: Path
    reason: str = ""


def normalize_model_name(name: str) -> str:
    text = str(name).strip()
    return MODEL_ALIASES.get(text, text.lower().replace("-", "_"))


def normalize_dataset_name(name: str) -> str:
    text = str(name).strip().lower()
    aliases = {
        "fraud_yelp_official": "fraud_yelp",
        "fraudamazon": "fraud_amazon",
        "fraudyelp": "fraud_yelp",
        "fraud_amazon_official": "fraud_amazon",
    }
    return aliases.get(text, text)


def default_models_for_dataset(dataset: str) -> tuple[str, ...]:
    dataset = normalize_dataset_name(dataset)
    if dataset not in DATASET_MODEL_MATRIX:
        raise ValueError(f"Unknown submission dataset: {dataset}")
    return DATASET_MODEL_MATRIX[dataset]


def run_submission_experiment(
    dataset: str,
    model: str,
    seed: int,
    output_dir: str | Path,
    data_root: str | Path = "data",
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
    top_k: int = 10,
    overwrite: bool = False,
    device: str = "auto",
    llm_label_file: str | Path | None = None,
) -> SubmissionResult:
    dataset = normalize_dataset_name(dataset)
    model = normalize_model_name(model)
    result_dir = Path(output_dir) / dataset / model / f"seed_{seed}"
    metrics_path = result_dir / "metrics.json"
    skip_path = result_dir / "skip_reason.json"
    if metrics_path.exists() and not overwrite:
        return SubmissionResult(dataset, model, seed, "exists", metrics_path)
    if skip_path.exists() and not overwrite:
        return SubmissionResult(dataset, model, seed, "skipped", skip_path, "skip_reason_already_exists")
    if model in FORBIDDEN_SUBMISSION_NAMES or "lite" in model.lower():
        return write_skip(result_dir, dataset, model, seed, "forbidden_lite_baseline_name")
    if dataset not in DATASET_MODEL_MATRIX:
        return write_skip(result_dir, dataset, model, seed, "unknown_dataset")
    if model not in DATASET_MODEL_MATRIX[dataset]:
        return write_skip(result_dir, dataset, model, seed, "model_not_applicable_to_dataset")
    data_dir = resolve_processed_dir(dataset, data_root=data_root)
    if not processed_ready(data_dir):
        return write_skip(
            result_dir,
            dataset,
            model,
            seed,
            "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.",
            {"data_dir": str(data_dir)},
        )
    try:
        graph = load_processed_data(data_dir)
        split_status = validate_labeled_split(graph)
        if not split_status["valid"]:
            return write_skip(
                result_dir,
                dataset,
                model,
                seed,
                "processed split has no labeled train/val/test nodes",
                {"data_dir": str(data_dir), **_split_validation_extra(split_status)},
            )
        if model in HERO_MODELS:
            return _run_project_hero(
                dataset=dataset,
                model=model,
                seed=seed,
                data_dir=data_dir,
                result_dir=result_dir,
                output_dir=Path(output_dir),
                epochs=epochs,
                lr=lr,
                hidden_dim=hidden_dim,
                top_k=top_k,
                device=device,
                llm_label_file=llm_label_file,
            )
        return _run_reproduced_baseline(
            graph=graph,
            dataset=dataset,
            model=model,
            seed=seed,
            data_dir=data_dir,
            result_dir=result_dir,
            epochs=epochs,
            lr=lr,
            top_k=top_k,
            llm_label_file=llm_label_file,
        )
    except FileNotFoundError as exc:
        return write_skip(result_dir, dataset, model, seed, str(exc), {"data_dir": str(data_dir)})
    except Exception as exc:  # keep AutoDL sweeps moving while preserving the failure.
        return write_skip(result_dir, dataset, model, seed, f"experiment_failed: {type(exc).__name__}: {exc}")


def resolve_processed_dir(dataset: str, data_root: str | Path = "data") -> Path:
    data_root = Path(data_root)
    candidates = {
        "yelp_academic": [data_root / "processed" / "yelp_academic"],
        "amazon_video": [data_root / "processed" / "amazon_video"],
        "fraud_yelp": [data_root / "processed" / "fraud_yelp", data_root / "processed" / "fraud_yelp_official"],
        "fraud_amazon": [data_root / "processed" / "fraud_amazon", data_root / "processed" / "fraud_amazon_official"],
        "elliptic": [data_root / "processed" / "elliptic"],
    }.get(normalize_dataset_name(dataset), [data_root / "processed" / dataset])
    for path in candidates:
        if processed_ready(path):
            return path
    return candidates[0]


def processed_ready(path: str | Path) -> bool:
    path = Path(path)
    if path.name.lower() == "elliptic" or _metadata_dataset(path) == "elliptic":
        has_features = any((path / name).exists() for name in ["features.npz", "features.npy", "features.pt"])
        has_edges = any((path / name).exists() for name in ["edge_index.npy", "edge_index.pt", "edges.csv"])
        has_labels = any((path / name).exists() for name in ["labels.npy", "labels.pt", "nodes.csv"])
        has_masks = (
            all((path / f"{split_name}_mask.npy").exists() or (path / f"{split_name}_mask.pt").exists() for split_name in ["train", "val", "test"])
            or any((path / name).exists() for name in ["masks.pt", "masks.npy", "masks.npz", "split.json"])
        )
        return has_features and has_edges and has_labels and has_masks
    return all((path / name).exists() for name in ["nodes.csv", "edges.csv", "features.npz", "split.json"])


def write_skip(
    result_dir: str | Path,
    dataset: str,
    model: str,
    seed: int,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> SubmissionResult:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "status": "skipped",
        "skip_reason": reason,
        "timestamp": _timestamp(),
    }
    if extra:
        payload.update(extra)
    path = result_dir / "skip_reason.json"
    write_json(path, payload)
    _write_log(result_dir, [f"SKIPPED {dataset}/{model}/seed_{seed}: {reason}"])
    return SubmissionResult(dataset, model, seed, "skipped", path, reason)


def _run_project_hero(
    dataset: str,
    model: str,
    seed: int,
    data_dir: Path,
    result_dir: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    hidden_dim: int,
    top_k: int,
    device: str,
    llm_label_file: str | Path | None,
) -> SubmissionResult:
    trainer_dataset = {"fraud_yelp": "fraud_yelp_official", "fraud_amazon": "fraud_amazon_official"}.get(dataset, dataset)
    internal_root = output_dir / "_project_runs"
    metrics = train_single_experiment(
        dataset=trainer_dataset,
        model_name=model,
        seed=seed,
        data_dir=data_dir,
        output_root=internal_root,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        top_k=top_k,
        llm_label_file=llm_label_file,
        device=device,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    converted = _submission_metric_payload(metrics, dataset, model, seed, MODEL_IMPLEMENTATION_SOURCE[model])
    converted["trainer_dataset"] = trainer_dataset
    converted["data_dir"] = str(data_dir)
    write_json(result_dir / "metrics.json", converted)
    prediction_file = metrics.get("predictions_file")
    if prediction_file and Path(str(prediction_file)).exists():
        shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
    _write_config(result_dir, dataset, model, seed, data_dir, MODEL_IMPLEMENTATION_SOURCE[model], {"source": "train_single_experiment"})
    _write_log(result_dir, [f"Completed {dataset}/{model}/seed_{seed} via project trainer."])
    return SubmissionResult(dataset, model, seed, "ok", result_dir / "metrics.json")


def _run_reproduced_baseline(
    graph: ProcessedGraphData,
    dataset: str,
    model: str,
    seed: int,
    data_dir: Path,
    result_dir: Path,
    epochs: int,
    lr: float,
    top_k: int,
    llm_label_file: str | Path | None,
) -> SubmissionResult:
    set_seed(seed)
    feature_result = build_submission_features(graph, dataset, model, seed=seed, top_k=top_k, llm_label_file=llm_label_file, data_dir=data_dir)
    if feature_result.skip_reason:
        return write_skip(result_dir, dataset, model, seed, feature_result.skip_reason, {"data_dir": str(data_dir)})
    features = np.asarray(feature_result.features, dtype=np.float32)
    labels = np.asarray(graph.labels, dtype=np.int64)
    split_status = validate_labeled_split(graph)
    if not split_status["valid"]:
        return write_skip(
            result_dir,
            dataset,
            model,
            seed,
            "processed split has no labeled train/val/test nodes",
            {"data_dir": str(data_dir), **_split_validation_extra(split_status)},
        )
    train_idx = _valid_split(graph, "train")
    val_idx = _valid_split(graph, "val")
    test_idx = _valid_split(graph, "test")
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        return write_skip(result_dir, dataset, model, seed, "processed split has no labeled train/val/test nodes")

    train_x, train_y = features[train_idx], labels[train_idx]
    if model == "pc_gnn":
        train_x, train_y = _oversample_positive(train_x, train_y, seed)
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    val_x_scaled = scaler.transform(features[val_idx]) if val_idx.size else train_x_scaled
    test_x_scaled = scaler.transform(features[test_idx])
    if len(np.unique(train_y)) < 2:
        classifier = DummyClassifier(strategy="prior")
    else:
        classifier = LogisticRegression(class_weight="balanced", max_iter=max(int(epochs) * 20, 500), random_state=seed, solver="liblinear", C=max(float(lr) * 1000.0, 0.01))
    start = time.perf_counter()
    classifier.fit(train_x_scaled, train_y)
    training_time = time.perf_counter() - start
    val_scores = _positive_scores(classifier, val_x_scaled)
    test_scores = _positive_scores(classifier, test_x_scaled)
    threshold_labels = labels[val_idx] if val_idx.size else train_y
    threshold_info = tune_threshold(threshold_labels, val_scores, return_info=True)
    best_threshold = float(threshold_info["best_threshold"])
    metrics = binary_classification_metrics(labels[test_idx], test_scores, k=100, threshold=best_threshold)
    metrics.update(threshold_info)
    metrics.update(fixed_threshold_diagnostics(labels[test_idx], test_scores, threshold=0.5))
    metrics.update(prediction_probability_stats(labels[test_idx], test_scores))
    metrics.update(feature_result.diagnostics)
    metrics["Training time"] = float(training_time)
    metrics["time_training_sec"] = float(training_time)
    metrics["Best epoch"] = int(epochs)
    metrics["best_epoch"] = int(epochs)

    result_dir.mkdir(parents=True, exist_ok=True)
    payload = _submission_metric_payload(metrics, dataset, model, seed, MODEL_IMPLEMENTATION_SOURCE[model])
    payload["data_dir"] = str(data_dir)
    payload["feature_builder"] = feature_result.name
    write_json(result_dir / "metrics.json", payload)
    np.save(result_dir / "predictions.npy", test_scores.astype(np.float32))
    _write_config(result_dir, dataset, model, seed, data_dir, MODEL_IMPLEMENTATION_SOURCE[model], feature_result.diagnostics)
    _write_log(result_dir, [f"Completed {dataset}/{model}/seed_{seed}.", f"metrics={json.dumps(payload, sort_keys=True)}"])
    return SubmissionResult(dataset, model, seed, "ok", result_dir / "metrics.json")


@dataclass
class FeatureResult:
    name: str
    features: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None
    skip_reason: str = ""


def build_submission_features(
    graph: ProcessedGraphData,
    dataset: str,
    model: str,
    seed: int,
    top_k: int,
    data_dir: Path,
    llm_label_file: str | Path | None = None,
) -> FeatureResult:
    x = np.asarray(graph.features, dtype=np.float32)
    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    diagnostics = {"implementation_source": MODEL_IMPLEMENTATION_SOURCE.get(model, "reproduced")}
    if model == "mlp":
        return FeatureResult(model, x, diagnostics)
    if model == "graphsage":
        return FeatureResult(model, _concat([x, _neighbor_mean(x, edge_index)]), diagnostics)
    if model == "gcn":
        return FeatureResult(model, _concat([x, _gcn_smooth(x, edge_index)]), diagnostics)
    if model == "gat":
        return FeatureResult(model, _concat([x, _attention_neighbor_mean(x, edge_index)]), diagnostics)
    if model == "care_gnn":
        values = _care_relation_features(graph, top_k=max(int(top_k), 1), train_idx=_valid_split(graph, "train"))
        return FeatureResult(model, _concat([x, values]), {**diagnostics, "neighbor_budget": int(top_k)})
    if model == "graphconsis":
        values = _graphconsis_features(graph)
        return FeatureResult(model, _concat([x, values]), diagnostics)
    if model == "pc_gnn":
        return FeatureResult(model, _concat([x, _neighbor_mean(x, edge_index), _degree_features(edge_index, x.shape[0])]), {**diagnostics, "class_imbalance_aware": True})
    if model == "bwgnn":
        waves = _multi_scale_wavelet_features(x, edge_index)
        return FeatureResult(model, waves, diagnostics)
    if model == "linkx":
        structure = _linkx_structure_features(edge_index, x.shape[0], seed=seed)
        return FeatureResult(model, _concat([x, structure]), diagnostics)
    if model == "dgp":
        if dataset not in TEXT_RICH_DATASETS:
            return FeatureResult(model, skip_reason="dgp_requires_text_rich_dataset")
        values = _dgp_features(graph)
        return FeatureResult(model, values, {**diagnostics, "prompt_budget_control": True})
    if model == "mled":
        if dataset not in TEXT_RICH_DATASETS:
            return FeatureResult(model, skip_reason="mled_requires_text_rich_dataset")
        labels_path = Path(llm_label_file) if llm_label_file else _find_annotation_file(data_dir, ["qwen", "llm_labels.jsonl"])
        if labels_path is None:
            return FeatureResult(model, skip_reason="mled_requires_existing_llm_or_risk_card_annotations")
        return FeatureResult(model, _mled_features(graph, labels_path), {**diagnostics, "llm_annotation_file": str(labels_path)})
    if model == "hogrl":
        if dataset != "elliptic":
            return FeatureResult(model, skip_reason="hogrl_only_required_for_elliptic")
        return FeatureResult(model, _hogrl_features(x, edge_index), diagnostics)
    if model == "rgtan":
        if dataset != "elliptic":
            return FeatureResult(model, skip_reason="rgtan_only_required_for_elliptic")
        return FeatureResult(model, _rgtan_features(graph), diagnostics)
    return FeatureResult(model, skip_reason=f"unsupported_submission_model:{model}")


def _submission_metric_payload(metrics: dict[str, Any], dataset: str, model: str, seed: int, implementation_source: str) -> dict[str, Any]:
    macro_f1 = float(metrics.get("macro_f1", metrics.get("Macro-F1", 0.0)))
    auroc = float(metrics.get("auroc", metrics.get("AUROC", 0.0)))
    auprc = float(metrics.get("auprc", metrics.get("AUPRC", 0.0)))
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))
    denom = max(tn + fp + fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    payload = {
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "implementation_source": implementation_source,
        "split_id": "default",
        "Macro-F1": macro_f1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": float((tp + tn) / denom),
        "Precision": float(precision),
        "Recall": float(recall),
        "Best epoch": int(metrics.get("Best epoch", metrics.get("best_epoch", 0))),
        "Training time": float(metrics.get("Training time", metrics.get("time_training_sec", 0.0))),
        "timestamp": _timestamp(),
        "macro_f1": macro_f1,
        "auroc": auroc,
        "auprc": auprc,
    }
    for key, value in metrics.items():
        if key not in payload:
            payload[key] = _json_safe(value)
    return payload


def _valid_split(graph: ProcessedGraphData, split_name: str) -> np.ndarray:
    mask = np.asarray(getattr(graph, f"{split_name}_mask", np.zeros(graph.labels.shape[0], dtype=bool)), dtype=bool)
    if mask.shape != graph.labels.shape:
        mask = np.zeros(graph.labels.shape[0], dtype=bool)
        indices = np.asarray(graph.split.get(split_name, np.array([], dtype=np.int64)), dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < graph.labels.shape[0])]
        mask[indices] = True
    return np.flatnonzero(mask & (graph.labels >= 0)).astype(np.int64)


def _split_validation_extra(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_count": int(status.get("train_count", 0)),
        "val_count": int(status.get("val_count", 0)),
        "test_count": int(status.get("test_count", 0)),
        "num_labeled": int(status.get("num_labeled", 0)),
        "label_distribution": status.get("label_distribution", {}),
        "train_class_distribution": status.get("train_class_distribution", {}),
        "val_class_distribution": status.get("val_class_distribution", {}),
        "test_class_distribution": status.get("test_class_distribution", {}),
    }


def _metadata_dataset(path: Path) -> str:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return ""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(payload.get("dataset", "")).lower()


def _positive_scores(classifier: Any, features: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    if probabilities.shape[1] == 1:
        label = int(classifier.classes_[0])
        return np.ones(features.shape[0], dtype=np.float32) if label == 1 else np.zeros(features.shape[0], dtype=np.float32)
    class_to_col = {int(label): col for col, label in enumerate(classifier.classes_)}
    return probabilities[:, class_to_col.get(1, 0)].astype(np.float32)


def _oversample_positive(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if pos.size == 0 or neg.size == 0 or pos.size >= neg.size:
        return features, labels
    rng = np.random.default_rng(seed)
    sampled = rng.choice(pos, size=neg.size - pos.size, replace=True)
    idx = np.concatenate([np.arange(labels.size), sampled])
    return features[idx], labels[idx]


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays if arr is not None and arr.size]
    return np.concatenate(arrays, axis=1).astype(np.float32)


def _neighbor_mean(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    agg = np.zeros_like(features, dtype=np.float32)
    degree = np.zeros((features.shape[0], 1), dtype=np.float32)
    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        np.add.at(agg, src, features[dst])
        np.add.at(degree, src, 1.0)
    return agg / np.maximum(degree, 1.0)


def _gcn_smooth(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    return 0.5 * features + 0.5 * _neighbor_mean(features, edge_index)


def _attention_neighbor_mean(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    out = np.zeros_like(features, dtype=np.float32)
    if not edge_index.size:
        return out
    src, dst = edge_index[0], edge_index[1]
    scores = _cosine_rows(features[src], features[dst])
    order = np.argsort(src)
    src_sorted, dst_sorted, scores_sorted = src[order], dst[order], scores[order]
    start = 0
    while start < src_sorted.size:
        end = start + 1
        while end < src_sorted.size and src_sorted[end] == src_sorted[start]:
            end += 1
        local_scores = scores_sorted[start:end]
        weights = _softmax(local_scores)
        out[src_sorted[start]] = weights @ features[dst_sorted[start:end]]
        start = end
    return out.astype(np.float32)


def _care_relation_features(graph: ProcessedGraphData, top_k: int, train_idx: np.ndarray) -> np.ndarray:
    x = np.asarray(graph.features, dtype=np.float32)
    labels = np.asarray(graph.labels, dtype=np.int64)
    fraud_proto = x[train_idx[labels[train_idx] == 1]].mean(axis=0) if np.any(labels[train_idx] == 1) else x[train_idx].mean(axis=0)
    normal_proto = x[train_idx[labels[train_idx] == 0]].mean(axis=0) if np.any(labels[train_idx] == 0) else x[train_idx].mean(axis=0)
    relation_names = sorted(set(graph.edges.get(schema.EDGE_TYPE, pd.Series(["edge"])).astype(str).tolist()))[:4]
    relation_to_col = {name: idx for idx, name in enumerate(relation_names)}
    per_node: list[list[tuple[float, int, int]]] = [[] for _ in range(x.shape[0])]
    for src_id, dst_id, edge_type in graph.edges[[schema.SRC, schema.DST, schema.EDGE_TYPE]].itertuples(index=False, name=None):
        if src_id not in graph.node_id_to_idx or dst_id not in graph.node_id_to_idx:
            continue
        src_idx = graph.node_id_to_idx[str(src_id)]
        dst_idx = graph.node_id_to_idx[str(dst_id)]
        rel = relation_to_col.get(str(edge_type), 0)
        fraud_score = _cosine_vector(x[dst_idx], fraud_proto) - _cosine_vector(x[dst_idx], normal_proto)
        pred_similarity = 1.0 - abs(_cosine_vector(x[src_idx], fraud_proto) - _cosine_vector(x[dst_idx], fraud_proto))
        per_node[src_idx].append((float(fraud_score + pred_similarity), dst_idx, rel))
    blocks = []
    for rel in range(max(len(relation_names), 1)):
        block = np.zeros_like(x, dtype=np.float32)
        for idx, candidates in enumerate(per_node):
            chosen = [item for item in candidates if item[2] == rel]
            chosen = sorted(chosen, key=lambda item: item[0], reverse=True)[:top_k]
            if chosen:
                block[idx] = x[[item[1] for item in chosen]].mean(axis=0)
        blocks.append(block)
    return _concat(blocks)


def _graphconsis_features(graph: ProcessedGraphData) -> np.ndarray:
    x = np.asarray(graph.features, dtype=np.float32)
    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    if not edge_index.size:
        return np.zeros((x.shape[0], x.shape[1] * 2 + 2), dtype=np.float32)
    src, dst = edge_index[0], edge_index[1]
    weights = np.clip((_cosine_rows(x[src], x[dst]) + 1.0) / 2.0, 0.0, 1.0)
    agg = np.zeros_like(x, dtype=np.float32)
    diff = np.zeros_like(x, dtype=np.float32)
    degree = np.zeros((x.shape[0], 1), dtype=np.float32)
    weighted_degree = np.zeros((x.shape[0], 1), dtype=np.float32)
    np.add.at(agg, src, x[dst] * weights.reshape(-1, 1))
    np.add.at(diff, src, np.abs(x[src] - x[dst]))
    np.add.at(degree, src, 1.0)
    np.add.at(weighted_degree, src, weights.reshape(-1, 1))
    agg = agg / np.maximum(weighted_degree, 1e-6)
    diff = diff / np.maximum(degree, 1.0)
    return _concat([agg, diff, degree, weighted_degree / np.maximum(degree, 1.0)])


def _degree_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    out_deg = np.zeros((num_nodes, 1), dtype=np.float32)
    in_deg = np.zeros((num_nodes, 1), dtype=np.float32)
    if edge_index.size:
        np.add.at(out_deg, edge_index[0], 1.0)
        np.add.at(in_deg, edge_index[1], 1.0)
    return np.log1p(np.concatenate([out_deg, in_deg], axis=1)).astype(np.float32)


def _multi_scale_wavelet_features(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    one = _gcn_smooth(features, edge_index)
    two = _gcn_smooth(one, edge_index)
    three = _gcn_smooth(two, edge_index)
    low = 0.5 * one + 0.3 * two + 0.2 * three
    high1 = features - one
    high2 = one - two
    return _concat([features, low, high1, high2])


def _linkx_structure_features(edge_index: np.ndarray, num_nodes: int, seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchors = rng.normal(0.0, 1.0, size=(num_nodes, dim)).astype(np.float32)
    projection = np.zeros((num_nodes, dim), dtype=np.float32)
    degree = np.zeros((num_nodes, 1), dtype=np.float32)
    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        np.add.at(projection, src, anchors[dst])
        np.add.at(degree, src, 1.0)
    projection = projection / np.maximum(degree, 1.0)
    return _concat([projection, _degree_features(edge_index, num_nodes)])


def _dgp_features(graph: ProcessedGraphData) -> np.ndarray:
    target_text = np.asarray(graph.text_features, dtype=np.float32)
    if target_text.size == 0:
        target_text = np.asarray(graph.features, dtype=np.float32)
    neighbor_text = _neighbor_mean(target_text, np.asarray(graph.edge_index, dtype=np.int64))
    prompt_stats = _text_prompt_stats(graph)
    return _concat([target_text, neighbor_text, np.abs(target_text - neighbor_text), prompt_stats])


def _mled_features(graph: ProcessedGraphData, label_file: Path) -> np.ndarray:
    base = _dgp_features(graph)
    mechanism = np.zeros((graph.features.shape[0], 8), dtype=np.float32)
    labels = _read_label_file(label_file)
    for label in labels:
        target_id = str(label.get("target_id", ""))
        if target_id not in graph.node_id_to_idx:
            continue
        idx = graph.node_id_to_idx[target_id]
        mechanism[idx, 0] += float(label.get("risk_relevance", 0))
        mechanism[idx, 1] += float(label.get("confidence", 0.0))
        mechanism[idx, 2 + (abs(hash(str(label.get("mechanism", "")))) % 6)] += 1.0
    counts = np.maximum(mechanism[:, :1], 1.0)
    mechanism[:, 1:] = mechanism[:, 1:] / counts
    return _concat([base, mechanism])


def _hogrl_features(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    hop1 = _neighbor_mean(features, edge_index)
    hop2 = _neighbor_mean(hop1, edge_index)
    hop3 = _neighbor_mean(hop2, edge_index)
    return _concat([features, hop1, hop2, hop3, np.abs(features - hop2)])


def _rgtan_features(graph: ProcessedGraphData) -> np.ndarray:
    x = np.asarray(graph.features, dtype=np.float32)
    edge_index = np.asarray(graph.edge_index, dtype=np.int64)
    timestamps = _node_timestamps(graph)
    time_norm = _normalize_column(timestamps.reshape(-1, 1))
    temporal_neighbor = np.zeros_like(x, dtype=np.float32)
    degree = np.zeros((x.shape[0], 1), dtype=np.float32)
    if edge_index.size:
        src, dst = edge_index[0], edge_index[1]
        delta = np.abs(timestamps[src] - timestamps[dst])
        weights = np.exp(-_normalize_array(delta))
        np.add.at(temporal_neighbor, src, x[dst] * weights.reshape(-1, 1))
        np.add.at(degree, src, weights.reshape(-1, 1))
    temporal_neighbor = temporal_neighbor / np.maximum(degree, 1e-6)
    return _concat([x, temporal_neighbor, time_norm, degree])


def _text_prompt_stats(graph: ProcessedGraphData) -> np.ndarray:
    texts = graph.nodes.get(schema.TEXT, pd.Series([""] * graph.features.shape[0])).astype(str).tolist()
    lengths = np.asarray([len(text.split()) for text in texts], dtype=np.float32).reshape(-1, 1)
    chars = np.asarray([len(text) for text in texts], dtype=np.float32).reshape(-1, 1)
    return _concat([_normalize_column(lengths), _normalize_column(chars)])


def _node_timestamps(graph: ProcessedGraphData) -> np.ndarray:
    for col in ["time_step", schema.TIMESTAMP, "timestamp"]:
        if col in graph.nodes:
            return pd.to_numeric(graph.nodes[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return np.arange(graph.features.shape[0], dtype=np.float32)


def _find_annotation_file(data_dir: Path, preferred_tokens: list[str]) -> Path | None:
    candidates = sorted(data_dir.glob("*.jsonl"))
    for token in preferred_tokens:
        for path in candidates:
            if token in path.name.lower():
                return path
    return None


def _read_label_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denom, 1e-8)


def _cosine_vector(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-8))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return (exp / np.maximum(exp.sum(), 1e-8)).astype(np.float32)


def _normalize_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - np.nanmean(values, axis=0, keepdims=True)) / np.maximum(np.nanstd(values, axis=0, keepdims=True), 1e-6)


def _normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - float(np.min(values))) / max(float(np.max(values) - np.min(values)), 1e-6)


def _write_config(result_dir: Path, dataset: str, model: str, seed: int, data_dir: Path, implementation_source: str, extra: dict[str, Any]) -> None:
    payload = {
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "data_dir": str(data_dir),
        "implementation_source": implementation_source,
        "extra": _json_safe(extra),
    }
    (result_dir / "config_resolved.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_log(result_dir: Path, lines: list[str]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join([_timestamp(), *lines, ""])
    (result_dir / "run.log").write_text(text, encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
