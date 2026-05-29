from __future__ import annotations

import logging
import pickle
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.data import schema
from src.data.loader import ProcessedGraphData, load_processed_data
from src.graph.heterophily_scoring import mechanism_id, risk_heterophily_score
from src.graph.neighbor_retrieval import (
    filter_rule_hetero_edges,
    filter_topk_semantic_edges,
    retrieve_hetero_candidates,
)
from src.llm.label_cache import LabelCache, cache_key
from src.llm.base_labeler import normalize_label
from src.llm.mock_labeler import MOCK_LABELER_VERSION, label_candidate_mechanism
from src.llm.risk_card import format_candidate_risk_card
from src.training.evaluator import (
    binary_classification_metrics,
    fixed_threshold_diagnostics,
    prediction_probability_stats,
    split_label_stats,
    tune_threshold,
)
from src.utils.io import write_json
from src.utils.seed import set_seed

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

BASELINE_MODEL_NAMES = (
    "mlp",
    "graphsage",
    "semsim_gnn",
    "rulehetero_gnn",
    "sec_gfd_lite",
    "dga_gnn_lite",
    "flag_lite",
)
HERO_MODEL_NAMES = ("hero_gnn", "hero_wo_chain", "hero_wo_hetero", "hero_wo_mechanism")
HERO_OFFICIAL_MODEL_NAMES = (
    "hero_official",
    "hero_official_wo_hetero",
    "hero_official_wo_relation",
    "hero_official_wo_feature_deviation",
)
MODEL_NAMES = (*BASELINE_MODEL_NAMES, *HERO_MODEL_NAMES, *HERO_OFFICIAL_MODEL_NAMES)
LITE_MODEL_NAMES = ("sec_gfd_lite", "dga_gnn_lite", "flag_lite")


def _hero_variant_flags(model_name: str) -> dict[str, bool]:
    if model_name == "hero_gnn":
        return {
            "use_hetero": True,
            "use_chain": True,
            "use_mechanism": True,
            "use_chain_encoder": True,
            "use_mock_llm_mechanism": True,
        }
    if model_name == "hero_wo_chain":
        return {
            "use_hetero": True,
            "use_chain": False,
            "use_mechanism": True,
            "use_chain_encoder": False,
            "use_mock_llm_mechanism": True,
        }
    if model_name == "hero_wo_hetero":
        return {
            "use_hetero": False,
            "use_chain": False,
            "use_mechanism": False,
            "use_chain_encoder": False,
            "use_mock_llm_mechanism": False,
        }
    if model_name == "hero_wo_mechanism":
        return {
            "use_hetero": True,
            "use_chain": True,
            "use_mechanism": False,
            "use_chain_encoder": True,
            "use_mock_llm_mechanism": False,
        }
    return {
        "use_hetero": False,
        "use_chain": False,
        "use_mechanism": False,
        "use_chain_encoder": False,
        "use_mock_llm_mechanism": False,
    }


def _hero_branch_masks(model_name: str) -> dict[str, int]:
    if model_name == "hero_gnn":
        values = (1, 1, 1, 1, 1)
    elif model_name == "hero_wo_chain":
        values = (1, 1, 1, 1, 0)
    elif model_name == "hero_wo_hetero":
        values = (1, 1, 0, 0, 0)
    elif model_name == "hero_wo_mechanism":
        values = (1, 1, 1, 0, 1)
    else:
        values = (0, 0, 0, 0, 0)
    return {
        "branch_mask_target": values[0],
        "branch_mask_homo": values[1],
        "branch_mask_hetero": values[2],
        "branch_mask_mechanism": values[3],
        "branch_mask_chain": values[4],
    }


def _default_chain_diagnostics() -> dict[str, float | int]:
    return {
        "avg_chain_quality": 0.0,
        "avg_chain_quality_pos": 0.0,
        "avg_chain_quality_neg": 0.0,
        "num_raw_chains": 0,
        "num_filtered_chains": 0,
        "chain_filter_keep_rate": 0.0,
        "avg_chain_gate": 0.0,
        "avg_chain_gate_pos": 0.0,
        "avg_chain_gate_neg": 0.0,
        "lambda_chain_pos": 0.0,
        "lambda_chain_neg": 0.0,
        "chain_pos_loss": 0.0,
        "chain_neg_loss": 0.0,
    }


def _model_metadata(model_name: str) -> dict[str, Any]:
    if model_name in HERO_OFFICIAL_MODEL_NAMES:
        return {
            "model_family": "hero_official",
        }
    if model_name == "sec_gfd_lite":
        return {
            "model_family": "sec_gfd_lite",
            "uses_heterophily_filter": True,
        }
    if model_name == "dga_gnn_lite":
        return {
            "model_family": "dga_gnn_lite",
            "num_attribute_groups": 4,
            "num_neighbor_groups": 4,
        }
    if model_name == "flag_lite":
        return {
            "model_family": "flag_lite",
            "uses_semantic_enhancement": True,
        }
    return {}


def _official_variant_flags(model_name: str) -> dict[str, bool]:
    return {
        "use_official_hetero": model_name != "hero_official_wo_hetero",
        "use_official_relation": model_name != "hero_official_wo_relation",
        "use_official_feature_deviation": model_name != "hero_official_wo_feature_deviation",
    }


def _default_official_diagnostics() -> dict[str, float | int | bool]:
    return {
        "official_mode": False,
        "use_official_chain": False,
        "avg_relation_gate": 0.0,
        "avg_feature_deviation_gate": 0.0,
        "avg_official_hetero_gate": 0.0,
        "official_avg_feature_distance_selected": 0.0,
        "official_avg_homo_similarity_selected": 0.0,
        "official_avg_relation_rarity": 0.0,
        "official_num_relation_types": 0,
        "official_topk_homo": 0,
        "official_topk_hetero": 0,
    }


def _is_official_graph(graph: ProcessedGraphData, dataset: str) -> bool:
    return graph.preprocess_report.get("label_source") == "official" or dataset in {"fraud_yelp_official", "fraud_amazon_official"}


def _resolve_device_name(device: str | None) -> tuple[str, bool]:
    requested = device or "auto"
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if requested == "auto":
        return ("cuda" if cuda_available else "cpu"), cuda_available
    if requested == "cuda" and not cuda_available:
        print("[WARNING] CUDA requested but not available; fallback to CPU")
        return "cpu", cuda_available
    return requested, cuda_available


def train_single_experiment(
    dataset: str = "synthetic",
    model_name: str = "mlp",
    seed: int = 0,
    data_dir: str | Path | None = None,
    output_root: str | Path = "outputs",
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
    top_k: int = 10,
    max_target_nodes: int | None = None,
    max_candidates_per_node: int = 20,
    homophilic_topk: int = 5,
    heterophilic_topk: int = 5,
    topk_chains: int = 3,
    max_chain_length: int = 2,
    min_chain_quality: float = 0.45,
    lambda_chain_pos: float = 0.03,
    lambda_chain_neg: float = 0.01,
    llm_label_file: str | Path | None = None,
    experiment_tag: str | None = None,
    llm_labeler: str | None = None,
    eval_target_file: str | Path | None = None,
    disable_llm_fallback: bool = False,
    enable_official_chain: bool = False,
    device: str | None = "auto",
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {MODEL_NAMES}.")

    set_seed(seed)
    data_dir = Path(data_dir or f"data/processed/{dataset}")
    graph = load_processed_data(data_dir)
    official_mode = _is_official_graph(graph, dataset)
    device_name, cuda_available = _resolve_device_name(device)
    paths = _experiment_paths(output_root, dataset, model_name, seed, experiment_tag=experiment_tag)
    logger = _make_logger(paths["log"], dataset, model_name, seed)
    time_total_start = time.perf_counter()
    stage_times = {
        "time_retrieval_sec": 0.0,
        "time_mock_labeling_sec": 0.0,
        "time_evidence_chain_sec": 0.0,
        "time_training_sec": 0.0,
    }
    start_message = f"[START] dataset={dataset} model={model_name} seed={seed}"
    print(start_message)
    logger.info(start_message)
    logger.info("Starting experiment dataset=%s model=%s seed=%s", dataset, model_name, seed)
    logger.info("Training knobs epochs=%s lr=%s hidden_dim=%s top_k=%s", epochs, lr, hidden_dim, top_k)
    print(f"[DEVICE] dataset={dataset} model={model_name} seed={seed} device={device_name}")

    train_idx = _valid_label_indices(graph.split.get("train", np.array([], dtype=np.int64)), graph.labels)
    val_idx = _valid_label_indices(graph.split.get("val", np.array([], dtype=np.int64)), graph.labels)
    test_idx = _valid_label_indices(graph.split.get("test", np.array([], dtype=np.int64)), graph.labels)
    if official_mode and max_target_nodes is not None:
        train_idx, val_idx, test_idx = _limit_official_split_indices(train_idx, val_idx, test_idx, max_target_nodes)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Processed data must contain labeled train and test review nodes.")
    eval_target_indices = _load_eval_target_indices(eval_target_file, graph, test_idx) if eval_target_file is not None else None

    class_stats = _split_class_stats(
        train_labels=graph.labels[train_idx],
        val_labels=graph.labels[val_idx],
        test_labels=graph.labels[test_idx],
    )
    pos_weight = _pos_weight_from_counts(class_stats["train_num_pos"], class_stats["train_num_neg"])
    print(f"[TRAIN-INFO] dataset={dataset} model={model_name} seed={seed}")
    print(f"[TRAIN-INFO] train_pos={class_stats['train_num_pos']} train_neg={class_stats['train_num_neg']} pos_weight={pos_weight:.6f}")
    variant_flags = _hero_variant_flags(model_name)
    branch_masks = _hero_branch_masks(model_name)
    if official_mode and model_name in HERO_MODEL_NAMES:
        raise ValueError(f"{model_name} is a text-rich HERO model. Use hero_official variants for official fraud datasets.")
    if not official_mode and model_name in HERO_OFFICIAL_MODEL_NAMES:
        raise ValueError(f"{model_name} is for official non-text fraud datasets. Use text-rich HERO variants for {dataset}.")

    if model_name in HERO_MODEL_NAMES:
        variant_message = (
            f"[VARIANT] model={model_name} "
            f"use_hetero={variant_flags['use_hetero']} "
            f"use_chain={variant_flags['use_chain']} "
            f"use_mechanism={variant_flags['use_mechanism']}"
        )
        print(variant_message)
        logger.info(variant_message)
    logger.info(
        "TRAIN-INFO train_pos=%s train_neg=%s pos_weight=%s",
        class_stats["train_num_pos"],
        class_stats["train_num_neg"],
        pos_weight,
    )

    if model_name in HERO_MODEL_NAMES:
        features, features_without_chains, hero_artifacts = _prepare_hero_features(
            graph=graph,
            data_dir=data_dir,
            model_name=model_name,
            target_indices=np.concatenate([train_idx, val_idx, test_idx]),
            homophilic_topk=homophilic_topk,
            heterophilic_topk=heterophilic_topk,
            max_target_nodes=max_target_nodes,
            max_candidates_per_node=max_candidates_per_node,
            topk_chains=topk_chains,
            max_chain_length=max_chain_length,
            min_chain_quality=min_chain_quality,
            llm_label_file=llm_label_file,
            experiment_tag=experiment_tag,
            llm_labeler=llm_labeler,
            coverage_target_indices=eval_target_indices,
            disable_llm_fallback=disable_llm_fallback,
        )
        stage_times.update(hero_artifacts.get("time", {}))
        print(f"[TRAIN] {model_name}")
        train_start = time.perf_counter()
        val_scores, test_scores, scores_without_chains, checkpoint = _fit_torch_feature_model(
            features=features,
            features_without_chains=features_without_chains,
            labels=graph.labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            pos_weight=pos_weight,
            model_name=model_name,
            feature_dims=hero_artifacts.get("feature_dims", {}),
            use_hetero=bool(hero_artifacts.get("use_hetero", False)),
            use_mechanism=bool(hero_artifacts.get("use_mechanism", False)),
            use_chain=bool(hero_artifacts.get("use_chain", False)),
            lambda_chain_pos=float(lambda_chain_pos),
            lambda_chain_neg=float(lambda_chain_neg),
            min_chain_quality=float(min_chain_quality),
            device=device_name,
        )
        stage_times["time_training_sec"] += time.perf_counter() - train_start
        checkpoint["hero_artifacts"] = hero_artifacts
        hero_artifacts["model_diagnostics"] = checkpoint.get("diagnostics", {})
    elif model_name in HERO_OFFICIAL_MODEL_NAMES:
        train_start = time.perf_counter()
        val_scores, test_scores, checkpoint, official_artifacts = _fit_hero_official_model(
            graph=graph,
            model_name=model_name,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            pos_weight=pos_weight,
            homophilic_topk=homophilic_topk,
            heterophilic_topk=heterophilic_topk,
            max_candidates_per_node=max_candidates_per_node,
            enable_official_chain=enable_official_chain,
            device=device_name,
        )
        stage_times["time_retrieval_sec"] = float(official_artifacts.get("time_retrieval_sec", 0.0))
        stage_times["time_training_sec"] = time.perf_counter() - train_start - stage_times["time_retrieval_sec"]
    elif torch is not None:
        train_start = time.perf_counter()
        val_scores, test_scores, checkpoint = _fit_torch_model(
            graph=graph,
            model_name=model_name,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            top_k=top_k,
            pos_weight=pos_weight,
            device=device_name,
        )
        stage_times["time_training_sec"] = time.perf_counter() - train_start
    else:
        train_start = time.perf_counter()
        features = _prepare_features(graph, model_name, top_k=top_k)
        val_scores, test_scores, _scores_without, checkpoint = _fit_numpy_feature_model(
            features=features,
            features_without_chains=features,
            labels=graph.labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            pos_weight=pos_weight,
        )
        stage_times["time_training_sec"] = time.perf_counter() - train_start

    eval_test_idx = test_idx
    eval_test_scores = test_scores
    if eval_target_indices is not None:
        eval_positions = _subset_positions(test_idx, eval_target_indices)
        if eval_positions.size == 0:
            raise ValueError(f"No eval target ids from {eval_target_file} overlap the test split.")
        eval_test_idx = test_idx[eval_positions]
        eval_test_scores = test_scores[eval_positions]
        if model_name in HERO_MODEL_NAMES:
            scores_without_chains = scores_without_chains[eval_positions]

    threshold_labels = graph.labels[val_idx] if val_idx.size else graph.labels[train_idx]
    threshold_info = tune_threshold(threshold_labels, val_scores, return_info=True)
    best_threshold = float(threshold_info["best_threshold"])
    metrics = binary_classification_metrics(graph.labels[eval_test_idx], eval_test_scores, k=100, threshold=best_threshold)
    metrics.update(threshold_info)
    metrics.update(fixed_threshold_diagnostics(graph.labels[eval_test_idx], eval_test_scores, threshold=0.5))
    metrics.update(prediction_probability_stats(graph.labels[eval_test_idx], eval_test_scores))
    metrics.update(class_stats)
    if eval_target_file is not None:
        metrics["eval_target_file"] = str(eval_target_file)
        metrics["num_eval_target_nodes"] = int(eval_test_idx.size)
    if disable_llm_fallback:
        metrics["disable_llm_fallback"] = True
    metrics["pos_weight"] = float(pos_weight)
    metrics["official_mode"] = bool(official_mode)
    metrics["device"] = device_name
    metrics["cuda_available"] = bool(cuda_available)
    metrics.update(_model_metadata(model_name))
    metrics.update({key: bool(value) for key, value in variant_flags.items()})
    metrics.update(branch_masks)
    metrics.update(
        split_label_stats(
            {
                "train": graph.labels[train_idx],
                "val": graph.labels[val_idx],
                "test": graph.labels[eval_test_idx],
            }
        )
    )
    print(f"[EVAL-INFO] best_threshold={best_threshold:.6f} pred_positive_rate={metrics['pred_positive_rate']:.6f}")
    print(
        "[EVAL-INFO] "
        f"mean_pred_prob_pos={metrics['mean_pred_prob_pos']:.6f} "
        f"mean_pred_prob_neg={metrics['mean_pred_prob_neg']:.6f}"
    )
    if model_name in HERO_MODEL_NAMES:
        explanation_metrics = _write_hero_explanations(
            graph=graph,
            dataset=dataset,
            model_name=model_name,
            seed=seed,
            output_root=output_root,
            test_idx=eval_test_idx,
            scores=eval_test_scores,
            scores_without_chains=scores_without_chains,
            hero_artifacts=hero_artifacts,
        )
        metrics.update(explanation_metrics)
        metrics["avg_selected_neighbors"] = _hero_avg_selected_neighbors(hero_artifacts)
    elif model_name in HERO_OFFICIAL_MODEL_NAMES:
        official_metrics = dict(official_artifacts.get("diagnostics", {}))
        metrics.update(official_metrics)
        metrics["avg_selected_neighbors"] = float(official_metrics.get("avg_selected_neighbors", 0.0))
    else:
        metrics["avg_selected_neighbors"] = _baseline_avg_selected_neighbors(graph, model_name, eval_test_idx, top_k)
    for key, value in _default_chain_diagnostics().items():
        metrics.setdefault(key, value)
    for key, value in _default_official_diagnostics().items():
        metrics.setdefault(key, value)
    payload = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        **metrics,
    }
    if llm_label_file is not None or metrics.get("llm_label_file") or experiment_tag:
        payload["llm_label_file"] = str(llm_label_file) if llm_label_file is not None else str(metrics.get("llm_label_file", ""))
        payload["experiment_tag"] = str(experiment_tag or metrics.get("experiment_tag", ""))
        payload["llm_labeler"] = str(llm_labeler or metrics.get("llm_labeler", "mock"))
        payload["llm_label_coverage_rate"] = float(metrics.get("llm_label_coverage_rate", 0.0))
    if eval_target_file is not None:
        payload["eval_target_file"] = str(eval_target_file)
        payload["num_eval_target_nodes"] = int(metrics.get("num_eval_target_nodes", 0))
    if disable_llm_fallback:
        payload["disable_llm_fallback"] = True
    payload.update(stage_times)
    payload["time_total_sec"] = time.perf_counter() - time_total_start

    write_json(paths["metrics"], payload)
    _write_checkpoint(paths["checkpoint"], checkpoint, payload)
    done_message = f"[DONE] dataset={dataset} model={model_name} seed={seed} metrics={payload}"
    print(done_message)
    print(f"[TIME] model={model_name} total={payload['time_total_sec']:.3f}")
    print(f"[TIME] retrieval={payload['time_retrieval_sec']:.3f}")
    print(f"[TIME] mock_labeling={payload['time_mock_labeling_sec']:.3f}")
    print(f"[TIME] evidence_chain={payload['time_evidence_chain_sec']:.3f}")
    print(f"[TIME] training={payload['time_training_sec']:.3f}")
    logger.info(done_message)
    logger.info("Finished experiment metrics=%s", payload)
    return payload


def write_placeholder_result(output_dir: str | Path, experiment_name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{experiment_name}.json"
    path.write_text('{"status": "placeholder", "metric": null}\n', encoding="utf-8")
    return path


def _split_class_stats(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, int]:
    return {
        "train_num_pos": _num_pos(train_labels),
        "train_num_neg": _num_neg(train_labels),
        "val_num_pos": _num_pos(val_labels),
        "val_num_neg": _num_neg(val_labels),
        "test_num_pos": _num_pos(test_labels),
        "test_num_neg": _num_neg(test_labels),
    }


def _num_pos(labels: np.ndarray) -> int:
    labels = np.asarray(labels, dtype=np.int64)
    return int(np.sum(labels == 1))


def _num_neg(labels: np.ndarray) -> int:
    labels = np.asarray(labels, dtype=np.int64)
    return int(np.sum(labels == 0))


def _pos_weight_from_counts(num_pos: int, num_neg: int) -> float:
    return float(num_neg / max(num_pos, 1))


def _prepare_features(graph: ProcessedGraphData, model_name: str, top_k: int) -> np.ndarray:
    if model_name == "mlp":
        return graph.features
    if model_name == "graphsage":
        return _graphsage_features(graph.features, graph.edge_index)
    if model_name == "semsim_gnn":
        filtered = filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=top_k)
        return _graphsage_features(graph.features, filtered)
    if model_name == "rulehetero_gnn":
        filtered = filter_rule_hetero_edges(
            graph.edge_index,
            graph.text_features,
            graph.numeric_features,
            top_k=top_k,
        )
        return _graphsage_features(graph.features, filtered)
    if model_name == "sec_gfd_lite":
        low_pass = _neighbor_mean_features(graph.features, graph.edge_index)
        high_pass = _neighbor_mean_abs_diff_features(graph.features, graph.edge_index)
        return np.concatenate([graph.features, low_pass, high_pass], axis=1).astype(np.float32)
    if model_name == "dga_gnn_lite":
        return _dga_lite_features(graph)
    if model_name == "flag_lite":
        return _flag_lite_features(graph, top_k=top_k)
    raise ValueError(f"Unknown model_name={model_name}")


def _prepare_hero_features(
    graph: ProcessedGraphData,
    data_dir: Path,
    model_name: str,
    target_indices: np.ndarray,
    homophilic_topk: int,
    heterophilic_topk: int,
    max_target_nodes: int | None,
    max_candidates_per_node: int,
    topk_chains: int,
    max_chain_length: int,
    min_chain_quality: float,
    llm_label_file: str | Path | None = None,
    experiment_tag: str | None = None,
    llm_labeler: str | None = None,
    coverage_target_indices: np.ndarray | None = None,
    disable_llm_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    print(f"[START] {model_name}")
    timings = {
        "time_retrieval_sec": 0.0,
        "time_mock_labeling_sec": 0.0,
        "time_evidence_chain_sec": 0.0,
        "time_training_sec": 0.0,
    }
    target_indices = _limit_target_indices(target_indices, max_target_nodes)
    variant_flags = _hero_variant_flags(model_name)
    use_hetero = bool(variant_flags["use_hetero"])
    use_chain = bool(variant_flags["use_chain"])
    use_mechanism = bool(variant_flags["use_mechanism"])
    use_mock_llm_mechanism = bool(variant_flags["use_mock_llm_mechanism"])
    homo_edges = filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=homophilic_topk)
    homo_agg = _neighbor_mean_features(graph.features, homo_edges)
    feature_dims = {
        "target_dim": int(graph.features.shape[1]),
        "homo_dim": int(homo_agg.shape[1]),
        "hetero_dim": int(graph.features.shape[1]),
        "mechanism_dim": int(len(schema.EVIDENCE_MECHANISMS)),
        "chain_dim": int(graph.features.shape[1] + len(schema.EVIDENCE_MECHANISMS) + 2),
    }
    zero_hetero = np.zeros_like(graph.features, dtype=np.float32)
    zero_mechanism = np.zeros((graph.features.shape[0], len(schema.EVIDENCE_MECHANISMS)), dtype=np.float32)
    zero_chain = np.zeros((graph.features.shape[0], graph.features.shape[1] + len(schema.EVIDENCE_MECHANISMS) + 2), dtype=np.float32)
    base_debug = {
        **variant_flags,
        **_hero_branch_masks(model_name),
        "num_homophilic_neighbors_used": int(homo_edges.shape[1]) if homo_edges.size else 0,
        "num_heterophilic_neighbors_used": 0,
        "num_chains_used": 0,
        "num_mechanism_labels_used": 0,
        "num_raw_chains": 0,
        "num_filtered_chains": 0,
        "chain_filter_keep_rate": 0.0,
    }
    if llm_label_file is not None or experiment_tag:
        base_debug.update(
            {
                "llm_label_file": str(llm_label_file) if llm_label_file is not None else "",
                "experiment_tag": str(experiment_tag or ""),
                "llm_labeler": str(llm_labeler or ("external" if llm_label_file is not None else "mock")),
                "llm_label_coverage_rate": 0.0,
                "llm_label_total_pairs": 0,
                "num_external_llm_labels_used": 0,
                "num_missing_llm_pairs": 0,
                "disable_llm_fallback": bool(disable_llm_fallback),
            }
        )

    if not use_hetero:
        features = np.concatenate([graph.features, homo_agg, zero_hetero, zero_mechanism, zero_chain], axis=1).astype(np.float32)
        return features, features.copy(), {
            "chains_by_idx": {},
            "labels_by_target": {},
            "candidates_by_target": {},
            "time": timings,
            "feature_dims": feature_dims,
            "variant_debug": base_debug,
            **variant_flags,
        }

    print("[BUILD] hetero candidates")
    retrieval_start = time.perf_counter()
    candidates_by_target = _load_or_build_hetero_candidates(
        data_dir=data_dir,
        graph=graph,
        target_indices=target_indices,
        max_target_nodes=max_target_nodes,
        max_candidates_per_node=max_candidates_per_node,
    )
    candidates_by_target = _trim_candidates(candidates_by_target, heterophilic_topk)
    timings["time_retrieval_sec"] = time.perf_counter() - retrieval_start

    if use_mock_llm_mechanism and llm_label_file is not None:
        if disable_llm_fallback:
            print("[BUILD] external LLM labels without mock fallback")
        else:
            print("[BUILD] external LLM labels with mock fallback")
    else:
        print("[BUILD] mock LLM labels" if use_mock_llm_mechanism else "[BUILD] rule hetero labels")
    labels_by_target, label_time, label_stats = _label_candidates(
        data_dir,
        candidates_by_target,
        use_mechanism=use_mock_llm_mechanism,
        llm_label_file=llm_label_file,
        experiment_tag=experiment_tag,
        llm_labeler=llm_labeler,
        coverage_target_indices=coverage_target_indices,
        disable_llm_fallback=disable_llm_fallback,
    )
    timings["time_mock_labeling_sec"] = label_time
    hetero_features, mechanism_features = _hetero_feature_matrix(
        graph=graph,
        labels_by_target=labels_by_target,
        use_mechanism=use_mechanism,
    )
    usage_debug = {
        **base_debug,
        "num_heterophilic_neighbors_used": int(sum(len(labels) for labels in labels_by_target.values())),
        "num_mechanism_labels_used": int(sum(len(labels) for labels in labels_by_target.values())) if use_mechanism else 0,
        **label_stats,
    }

    if not use_chain:
        features = np.concatenate([graph.features, homo_agg, hetero_features, mechanism_features, zero_chain], axis=1).astype(np.float32)
        return features, features.copy(), {
            "chains_by_idx": {},
            "labels_by_target": labels_by_target,
            "candidates_by_target": candidates_by_target,
            "time": timings,
            "feature_dims": feature_dims,
            "variant_debug": usage_debug,
            **variant_flags,
        }

    flat_labels = [label for labels in labels_by_target.values() for label in labels]
    print("[BUILD] evidence chains")
    chain_start = time.perf_counter()
    raw_chains_by_idx = _load_or_build_evidence_chains(
        data_dir=data_dir,
        graph=graph,
        labels_by_target=labels_by_target,
        flat_labels=flat_labels,
        topk_chains=topk_chains,
        max_chain_length=max_chain_length,
        use_cache=use_mock_llm_mechanism,
    )
    chains_by_idx = _filter_chains_by_quality(raw_chains_by_idx, min_chain_quality=min_chain_quality, topk_chains=topk_chains)
    timings["time_evidence_chain_sec"] = time.perf_counter() - chain_start
    chain_features = _chain_feature_matrix(graph, chains_by_idx, use_mechanism=use_mechanism)
    features = np.concatenate([graph.features, homo_agg, hetero_features, mechanism_features, chain_features], axis=1).astype(np.float32)
    features_without_chains = np.concatenate([graph.features, homo_agg, hetero_features, mechanism_features, zero_chain], axis=1).astype(np.float32)
    raw_count = int(sum(len(chains) for chains in raw_chains_by_idx.values()))
    filtered_count = int(sum(len(chains) for chains in chains_by_idx.values()))
    usage_debug["num_chains_used"] = filtered_count
    usage_debug["num_raw_chains"] = raw_count
    usage_debug["num_filtered_chains"] = filtered_count
    usage_debug["chain_filter_keep_rate"] = float(filtered_count / raw_count) if raw_count else 0.0
    return features, features_without_chains, {
        "chains_by_idx": chains_by_idx,
        "raw_chains_by_idx": raw_chains_by_idx,
        "labels_by_target": labels_by_target,
        "candidates_by_target": candidates_by_target,
        "time": timings,
        "feature_dims": feature_dims,
        "min_chain_quality": float(min_chain_quality),
        "variant_debug": usage_debug,
        **variant_flags,
    }


def _limit_target_indices(target_indices: np.ndarray, max_target_nodes: int | None) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if max_target_nodes is None or max_target_nodes <= 0 or target_indices.size <= max_target_nodes:
        return target_indices
    return target_indices[: int(max_target_nodes)]


def _limit_official_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    max_target_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    limit = int(max_target_nodes)
    if limit <= 0:
        return train_idx, val_idx, test_idx
    total = train_idx.size + val_idx.size + test_idx.size
    if total <= limit:
        return train_idx, val_idx, test_idx
    train_quota = max(1, int(limit * 0.6))
    val_quota = max(1, int(limit * 0.2)) if val_idx.size else 0
    test_quota = max(1, limit - train_quota - val_quota)
    return train_idx[:train_quota], val_idx[:val_quota], test_idx[:test_quota]


def _load_or_build_hetero_candidates(
    data_dir: Path,
    graph: ProcessedGraphData,
    target_indices: np.ndarray,
    max_target_nodes: int | None,
    max_candidates_per_node: int,
) -> dict[int, list[Any]]:
    path = data_dir / "hetero_candidates.pkl"
    expected = {
        "num_nodes": int(graph.features.shape[0]),
        "num_edges": int(graph.edge_index.shape[1]),
        "max_target_nodes": int(max_target_nodes) if max_target_nodes is not None else None,
        "max_candidates_per_node": int(max_candidates_per_node),
    }
    if path.exists():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        candidates = payload.get("candidates_by_target", payload) if isinstance(payload, dict) else payload
        if _hetero_cache_is_compatible(metadata, expected):
            print("[CACHE] loading hetero_candidates.pkl")
            return _filter_candidate_targets(candidates, target_indices)
        print("[BUILD] building hetero candidates")
    else:
        print("[BUILD] building hetero candidates")
    candidates = retrieve_hetero_candidates(
        edge_index=graph.edge_index,
        edges=graph.edges,
        nodes=graph.nodes,
        node_id_to_idx=graph.node_id_to_idx,
        text_features=graph.text_features,
        numeric_features=graph.numeric_features,
        target_indices=target_indices,
        max_candidates_per_node=max_candidates_per_node,
    )
    with path.open("wb") as handle:
        pickle.dump({"metadata": expected, "candidates_by_target": candidates}, handle)
    return candidates


def _hetero_cache_is_compatible(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    if metadata.get("num_nodes") != expected["num_nodes"] or metadata.get("num_edges") != expected["num_edges"]:
        return False
    cached_candidates = metadata.get("max_candidates_per_node")
    if cached_candidates is not None and int(cached_candidates) < int(expected["max_candidates_per_node"]):
        return False
    cached_targets = metadata.get("max_target_nodes")
    expected_targets = expected.get("max_target_nodes")
    if expected_targets is None:
        return cached_targets is None
    if cached_targets is None:
        return True
    return int(cached_targets) >= int(expected_targets)


def _filter_candidate_targets(candidates_by_target: dict[int, list[Any]], target_indices: np.ndarray) -> dict[int, list[Any]]:
    target_set = {int(index) for index in target_indices}
    return {target: candidates for target, candidates in candidates_by_target.items() if int(target) in target_set}


def _trim_candidates(candidates_by_target: dict[int, list[Any]], limit: int) -> dict[int, list[Any]]:
    if limit <= 0:
        return {target: [] for target in candidates_by_target}
    return {target: candidates[:limit] for target, candidates in candidates_by_target.items()}


def _load_or_build_evidence_chains(
    data_dir: Path,
    graph: ProcessedGraphData,
    labels_by_target: dict[int, list[dict[str, Any]]],
    flat_labels: list[dict[str, Any]],
    topk_chains: int,
    max_chain_length: int,
    use_cache: bool,
) -> dict[int, list[dict[str, Any]]]:
    path = data_dir / "evidence_chains.jsonl"
    if use_cache and path.exists():
        print("[CACHE] loading evidence_chains.jsonl")
        cached = _read_evidence_chains(path)
        if all(0 <= target_idx < graph.features.shape[0] for target_idx in cached):
            enriched = _enrich_cached_chain_signals(cached, flat_labels)
            upgraded = _ensure_chain_quality(cached)
            if enriched or upgraded:
                _write_evidence_chains(path, cached)
            return cached
        print("[BUILD] building evidence chains")
    else:
        print("[BUILD] building evidence chains")
    chains_by_idx = _build_evidence_chains_for_targets(
        graph=graph,
        labels_by_target=labels_by_target,
        flat_labels=flat_labels,
        topk_chains=topk_chains,
        max_chain_length=max_chain_length,
    )
    if use_cache:
        _write_evidence_chains(path, chains_by_idx)
    return chains_by_idx


def _build_evidence_chains_for_targets(
    graph: ProcessedGraphData,
    labels_by_target: dict[int, list[dict[str, Any]]],
    flat_labels: list[dict[str, Any]],
    topk_chains: int,
    max_chain_length: int,
) -> dict[int, list[dict[str, Any]]]:
    idx_to_id = _idx_to_node_id(graph)
    include_two_hop = max_chain_length >= 2
    labels_by_target_id: dict[str, list[dict[str, Any]]] = {}
    for label in flat_labels:
        if label.get("risk_relevance", 0) == 1:
            labels_by_target_id.setdefault(label["target_id"], []).append(label)
    chains_by_idx: dict[int, list[dict[str, Any]]] = {}
    iterator = tqdm(labels_by_target.items(), desc="evidence_chain building", unit="node")
    for target_idx, labels in iterator:
        target_id = labels[0]["target_id"] if labels else idx_to_id[target_idx]
        chains_by_idx[target_idx] = _build_evidence_chains_indexed(
            target_id=target_id,
            labels=labels,
            labels_by_target_id=labels_by_target_id,
            top_k=topk_chains,
            include_two_hop=include_two_hop,
        )
    return chains_by_idx


def _build_evidence_chains_indexed(
    target_id: str,
    labels: list[dict[str, Any]],
    labels_by_target_id: dict[str, list[dict[str, Any]]],
    top_k: int,
    include_two_hop: bool,
) -> list[dict[str, Any]]:
    direct = [label for label in labels if label.get("risk_relevance", 0) == 1]
    chains = [_one_hop_chain_from_label(label) for label in direct]
    if include_two_hop:
        for first in direct:
            for second in labels_by_target_id.get(first["neighbor_id"], [])[:2]:
                if second["neighbor_id"] == target_id:
                    continue
                chains.append(_two_hop_chain_from_labels(first, second))
    for chain in chains:
        _with_chain_quality(chain)
    return sorted(chains, key=lambda item: (item.get("chain_quality", 0.0), item.get("chain_score", 0.0)), reverse=True)[:top_k]


def _one_hop_chain_from_label(label: dict[str, Any]) -> dict[str, Any]:
    candidate = label.get("candidate", {})
    score = float(label.get("risk_score", label.get("confidence", 0.0)))
    signals = _chain_signals_from_label(label)
    return _with_chain_quality({
        "target_id": label["target_id"],
        "chain_nodes": [label["target_id"], label["neighbor_id"]],
        "chain_edges": [label["metapath"]],
        "mechanism": label["mechanism"],
        "risk_relevance": int(label.get("risk_relevance", 0)),
        "chain_score": score,
        "confidence": float(label.get("confidence", 0.0)),
        "rationale": label["rationale"],
        "neighbor_idx": candidate.get("neighbor_idx"),
        **signals,
    })


def _two_hop_chain_from_labels(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_score = float(first.get("risk_score", first.get("confidence", 0.0)))
    second_score = float(second.get("risk_score", second.get("confidence", 0.0)))
    signals = _average_chain_signals(first, second)
    return _with_chain_quality({
        "target_id": first["target_id"],
        "chain_nodes": [first["target_id"], first["neighbor_id"], second["neighbor_id"]],
        "chain_edges": [first["metapath"], second["metapath"]],
        "mechanism": first["mechanism"],
        "risk_relevance": int(first.get("risk_relevance", 0)),
        "chain_score": (first_score + second_score) / 2.0,
        "confidence": float(first.get("confidence", 0.0)),
        "rationale": f"{first['rationale']}; then {second['rationale']}",
        "neighbor_idx": first.get("candidate", {}).get("neighbor_idx"),
        **signals,
    })


def _ensure_chain_quality(chains_by_idx: dict[int, list[dict[str, Any]]]) -> bool:
    upgraded = False
    for chains in chains_by_idx.values():
        for chain in chains:
            if "chain_quality" not in chain:
                upgraded = True
            _with_chain_quality(chain)
    return upgraded


def _enrich_cached_chain_signals(
    chains_by_idx: dict[int, list[dict[str, Any]]],
    flat_labels: list[dict[str, Any]],
) -> bool:
    label_by_edge = {
        (label.get("target_id"), label.get("neighbor_id"), label.get("metapath")): label
        for label in flat_labels
    }
    enriched = False
    for chains in chains_by_idx.values():
        for chain in chains:
            chain_nodes = chain.get("chain_nodes", [])
            chain_edges = chain.get("chain_edges", [])
            signal_rows = []
            for offset, metapath in enumerate(chain_edges):
                if offset + 1 >= len(chain_nodes):
                    continue
                label = label_by_edge.get((chain_nodes[offset], chain_nodes[offset + 1], metapath))
                if label is not None:
                    signal_rows.append(_chain_signals_from_label(label))
            if not signal_rows:
                continue
            for key in ("structural_score", "numeric_deviation", "time_deviation", "semantic_dissimilarity"):
                value = float(np.mean([row.get(key, 0.0) for row in signal_rows]))
                if abs(float(chain.get(key, -1.0)) - value) > 1e-8:
                    chain[key] = value
                    enriched = True
            if "chain_quality" in chain:
                previous_quality = float(chain.get("chain_quality", 0.0))
                _with_chain_quality(chain)
                enriched = enriched or abs(previous_quality - float(chain.get("chain_quality", 0.0))) > 1e-8
    return enriched


def _filter_chains_by_quality(
    chains_by_idx: dict[int, list[dict[str, Any]]],
    min_chain_quality: float,
    topk_chains: int,
) -> dict[int, list[dict[str, Any]]]:
    filtered: dict[int, list[dict[str, Any]]] = {}
    threshold = float(min_chain_quality)
    limit = max(int(topk_chains), 0)
    for target_idx, chains in chains_by_idx.items():
        prepared = [_with_chain_quality(dict(chain)) for chain in chains]
        kept = [chain for chain in prepared if float(chain.get("chain_quality", 0.0)) >= threshold]
        kept = sorted(kept, key=lambda item: (item.get("chain_quality", 0.0), item.get("chain_score", 0.0)), reverse=True)
        filtered[int(target_idx)] = kept[:limit] if limit > 0 else []
    return filtered


def _with_chain_quality(chain: dict[str, Any]) -> dict[str, Any]:
    chain["confidence"] = _bounded_float(chain.get("confidence", 0.0))
    chain["risk_relevance"] = int(chain.get("risk_relevance", 0))
    chain["chain_score"] = _bounded_float(chain.get("chain_score", 0.0))
    chain["structural_score"] = _bounded_float(chain.get("structural_score", 0.0))
    chain["numeric_deviation"] = _bounded_float(chain.get("numeric_deviation", 0.0))
    chain["time_deviation"] = _bounded_float(chain.get("time_deviation", 0.0))
    chain["semantic_dissimilarity"] = _bounded_float(chain.get("semantic_dissimilarity", 0.0))
    chain["chain_quality"] = _bounded_float(
        0.30 * chain["confidence"]
        + 0.25 * float(chain["risk_relevance"])
        + 0.20 * chain["chain_score"]
        + 0.15 * chain["structural_score"]
        + 0.10 * chain["numeric_deviation"]
    )
    return chain


def _chain_signals_from_label(label: dict[str, Any]) -> dict[str, float]:
    candidate = label.get("candidate", {}) or {}
    risk_card = label.get("risk_card", {}) or {}
    semantic_similarity = _first_numeric(candidate, risk_card, "semantic_similarity", default=1.0)
    return {
        "structural_score": _first_numeric(candidate, risk_card, "structural_score", default=0.0),
        "numeric_deviation": _first_numeric(candidate, risk_card, "numeric_deviation", default=0.0),
        "time_deviation": _first_numeric(candidate, risk_card, "time_deviation", default=0.0),
        "semantic_dissimilarity": _first_numeric(candidate, risk_card, "semantic_distance", default=1.0 - semantic_similarity),
    }


def _average_chain_signals(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    left = _chain_signals_from_label(first)
    right = _chain_signals_from_label(second)
    return {key: float((left.get(key, 0.0) + right.get(key, 0.0)) / 2.0) for key in left}


def _first_numeric(primary: dict[str, Any], secondary: dict[str, Any], key: str, default: float = 0.0) -> float:
    for container in (primary, secondary):
        if key in container and container[key] is not None:
            return _bounded_float(container[key])
    return _bounded_float(default)


def _bounded_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    return float(min(max(numeric, 0.0), 1.0))


def _write_evidence_chains(path: Path, chains_by_idx: dict[int, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for target_idx in sorted(chains_by_idx):
            payload = {"target_idx": int(target_idx), "chains": chains_by_idx[target_idx]}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_evidence_chains(path: Path) -> dict[int, list[dict[str, Any]]]:
    chains_by_idx: dict[int, list[dict[str, Any]]] = {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return chains_by_idx
    for line in text.splitlines():
        payload = json.loads(line)
        chains_by_idx[int(payload["target_idx"])] = payload.get("chains", [])
    return chains_by_idx


def _coverage_pair_count(pairs: list[tuple[int, Any]], coverage_target_set: set[int] | None) -> int:
    if coverage_target_set is None:
        return len(pairs)
    return int(sum(1 for target_idx, _candidate in pairs if int(target_idx) in coverage_target_set))


def _missing_llm_label(candidate: Any) -> dict[str, Any]:
    return {
        "target_id": candidate.target_id,
        "neighbor_id": candidate.neighbor_id,
        "metapath": candidate.metapath,
        "mechanism": "irrelevant_heterophily",
        "risk_relevance": 0,
        "confidence": 0.0,
        "rationale": "missing external LLM label; fallback disabled",
        "labeler_version": "external_missing_no_fallback",
    }


def _label_candidates(
    data_dir: Path,
    candidates_by_target: dict[int, list[Any]],
    use_mechanism: bool,
    llm_label_file: str | Path | None = None,
    experiment_tag: str | None = None,
    llm_labeler: str | None = None,
    coverage_target_indices: np.ndarray | None = None,
    disable_llm_fallback: bool = False,
) -> tuple[dict[int, list[dict[str, Any]]], float, dict[str, Any]]:
    label_start = time.perf_counter()
    custom_label_path = Path(llm_label_file) if llm_label_file is not None else None
    if custom_label_path is not None and not custom_label_path.exists():
        raise FileNotFoundError(f"Missing LLM label file: {custom_label_path}")
    cache_path = custom_label_path or (data_dir / "llm_labels.jsonl")
    cache = LabelCache(cache_path) if use_mechanism else None
    custom_readonly = custom_label_path is not None
    labels_by_target: dict[int, list[dict[str, Any]]] = {}
    if cache is not None and cache.path.exists():
        print(f"[CACHE] loading {cache.path}")
    pairs = [
        (target_idx, candidate)
        for target_idx, candidates in candidates_by_target.items()
        for candidate in candidates
    ]
    base_by_pair: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    coverage_target_set = {int(value) for value in coverage_target_indices} if coverage_target_indices is not None else None
    stats = {
        "external_llm_label_file": str(custom_label_path) if custom_label_path is not None else "",
        "external_llm_labels_used": 0,
        "mock_llm_labels_used": 0,
    }
    if custom_label_path is not None or experiment_tag:
        stats.update(
            {
                "llm_label_file": str(custom_label_path) if custom_label_path is not None else "",
                "experiment_tag": str(experiment_tag or ""),
                "llm_labeler": str(llm_labeler or ("external" if custom_label_path is not None else "mock")),
                "llm_label_total_pairs": _coverage_pair_count(pairs, coverage_target_set),
                "llm_label_coverage_rate": 0.0,
                "num_external_llm_labels_used": 0,
                "num_missing_llm_pairs": 0,
                "disable_llm_fallback": bool(disable_llm_fallback),
            }
        )
    if use_mechanism:
        labeler_desc = "external_llm_labels" if custom_readonly else "mock_labeler"
        for target_idx, candidate in tqdm(pairs, desc=labeler_desc, unit="pair"):
            is_coverage_pair = coverage_target_set is None or int(target_idx) in coverage_target_set
            key = cache_key(candidate.target_id, candidate.neighbor_id, candidate.metapath)
            cached_label = cache.get(key) if cache is not None else None
            if custom_readonly and cached_label is None and cache is not None:
                cached_label = cache.get_pair(candidate.target_id, candidate.neighbor_id)
            if custom_readonly and cached_label is not None:
                base_label = normalize_label(cached_label, risk_card=format_candidate_risk_card(candidate))
                stats["external_llm_labels_used"] += 1
                if is_coverage_pair and "num_external_llm_labels_used" in stats:
                    stats["num_external_llm_labels_used"] += 1
            elif custom_readonly and disable_llm_fallback:
                base_label = _missing_llm_label(candidate)
                if is_coverage_pair and "num_missing_llm_pairs" in stats:
                    stats["num_missing_llm_pairs"] += 1
            elif cached_label is None or cached_label.get("labeler_version") != MOCK_LABELER_VERSION:
                base_label = label_candidate_mechanism(candidate)
                stats["mock_llm_labels_used"] += 1
                if custom_readonly and is_coverage_pair and "num_missing_llm_pairs" in stats:
                    stats["num_missing_llm_pairs"] += 1
                if cache is not None and not custom_readonly:
                    cache.set(key, base_label)
            else:
                base_label = dict(cached_label)
            base_by_pair[(target_idx, candidate.target_id, candidate.neighbor_id, candidate.metapath)] = base_label
        if custom_label_path is not None or experiment_tag:
            coverage_total = int(stats.get("llm_label_total_pairs", 0))
            stats["llm_label_coverage_rate"] = (
                float(stats.get("num_external_llm_labels_used", 0) / coverage_total) if custom_readonly and coverage_total else 0.0
            )

    for target_idx, candidate in tqdm(pairs, desc="risk heterophily scoring", unit="pair"):
        if use_mechanism:
            label = dict(base_by_pair[(target_idx, candidate.target_id, candidate.neighbor_id, candidate.metapath)])
        else:
            label = {
                "target_id": candidate.target_id,
                "neighbor_id": candidate.neighbor_id,
                "metapath": candidate.metapath,
                "mechanism": "irrelevant_heterophily",
                "risk_relevance": int(candidate.candidate_score >= 0.55),
                "confidence": float(candidate.candidate_score),
                "rationale": "rule score only; mechanism labels disabled",
            }
        label["candidate"] = asdict(candidate)
        label["risk_card"] = format_candidate_risk_card(candidate)
        score, mechanism_logits = risk_heterophily_score(
            candidate,
            mechanism=label.get("mechanism"),
            risk_relevance_label=int(label.get("risk_relevance", 0)),
            use_mechanism=use_mechanism,
        )
        label = dict(label)
        label["risk_score"] = score
        label["mechanism_logits"] = mechanism_logits.tolist()
        if use_mechanism and cache is not None and not custom_readonly:
            cache.set(cache_key(candidate.target_id, candidate.neighbor_id, candidate.metapath), label)
        labels_by_target.setdefault(target_idx, []).append(label)

    for target_idx in candidates_by_target:
        target_labels = labels_by_target.get(target_idx, [])
        labels_by_target[target_idx] = sorted(target_labels, key=lambda item: item["risk_score"], reverse=True)
    if cache is not None and not custom_readonly:
        cache.save()
    return labels_by_target, time.perf_counter() - label_start, stats


def _chain_feature_matrix(
    graph: ProcessedGraphData,
    chains_by_idx: dict[int, list[dict[str, Any]]],
    use_mechanism: bool,
) -> np.ndarray:
    dim = graph.features.shape[1]
    out_dim = dim + len(schema.EVIDENCE_MECHANISMS) + 2
    features = np.zeros((graph.features.shape[0], out_dim), dtype=np.float32)
    node_id_to_idx = graph.node_id_to_idx
    for target_idx, chains in chains_by_idx.items():
        if not chains:
            continue
        reps = []
        weights = []
        for chain in chains:
            chain = _with_chain_quality(dict(chain))
            indices = [node_id_to_idx[node_id] for node_id in chain["chain_nodes"] if node_id in node_id_to_idx]
            if indices:
                node_repr = graph.features[indices].mean(axis=0)
            else:
                node_repr = np.zeros(dim, dtype=np.float32)
            mechanism = np.zeros(len(schema.EVIDENCE_MECHANISMS), dtype=np.float32)
            if use_mechanism:
                mechanism[mechanism_id(chain.get("mechanism", "irrelevant_heterophily"))] = 1.0
            score = np.array([float(chain.get("chain_score", 0.0))], dtype=np.float32)
            quality = np.array([float(chain.get("chain_quality", 0.0))], dtype=np.float32)
            reps.append(np.concatenate([node_repr, mechanism, score, quality]))
            weights.append(max(float(chain.get("chain_quality", 0.0)), 1e-3))
        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / np.maximum(float(np.sum(weight_array)), 1e-6)
        features[target_idx] = np.sum(np.asarray(reps, dtype=np.float32) * weight_array[:, None], axis=0)
    return features


def _hetero_feature_matrix(
    graph: ProcessedGraphData,
    labels_by_target: dict[int, list[dict[str, Any]]],
    use_mechanism: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hetero = np.zeros_like(graph.features, dtype=np.float32)
    mechanisms = np.zeros((graph.features.shape[0], len(schema.EVIDENCE_MECHANISMS)), dtype=np.float32)
    for target_idx, labels in labels_by_target.items():
        node_reps = []
        mechanism_reps = []
        weights = []
        for label in labels:
            candidate = label.get("candidate", {})
            neighbor_idx = candidate.get("neighbor_idx")
            if neighbor_idx is None:
                neighbor_id = label.get("neighbor_id")
                neighbor_idx = graph.node_id_to_idx.get(neighbor_id)
            if neighbor_idx is None or not (0 <= int(neighbor_idx) < graph.features.shape[0]):
                continue
            score = float(label.get("risk_score", label.get("confidence", 0.0)))
            weight = max(score, 1e-3)
            node_reps.append(graph.features[int(neighbor_idx)])
            weights.append(weight)
            if use_mechanism:
                mechanism = np.zeros(len(schema.EVIDENCE_MECHANISMS), dtype=np.float32)
                mechanism[mechanism_id(label.get("mechanism", "irrelevant_heterophily"))] = 1.0
                mechanism_reps.append(mechanism)
        if not node_reps:
            continue
        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / np.maximum(float(np.sum(weight_array)), 1e-6)
        hetero[int(target_idx)] = np.sum(np.asarray(node_reps, dtype=np.float32) * weight_array[:, None], axis=0)
        if use_mechanism and mechanism_reps:
            mechanisms[int(target_idx)] = np.sum(np.asarray(mechanism_reps, dtype=np.float32) * weight_array[:, None], axis=0)
    return hetero, mechanisms


def _edge_index_for_model(graph: ProcessedGraphData, model_name: str, top_k: int) -> np.ndarray:
    if model_name in {"mlp", "graphsage", "sec_gfd_lite", "dga_gnn_lite"}:
        return graph.edge_index
    if model_name == "semsim_gnn":
        return filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=top_k)
    if model_name == "rulehetero_gnn":
        return filter_rule_hetero_edges(graph.edge_index, graph.text_features, graph.numeric_features, top_k=top_k)
    if model_name == "flag_lite":
        return _flag_neighbor_edges(graph, top_k=top_k)
    raise ValueError(f"Unknown model_name={model_name}")


def _neighbor_mean_features(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    agg = np.zeros_like(features, dtype=np.float32)
    degree = np.zeros((features.shape[0], 1), dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        np.add.at(agg, src, features[dst])
        np.add.at(degree, src, 1.0)
    return agg / np.maximum(degree, 1.0)


def _neighbor_mean_abs_diff_features(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    agg = np.zeros_like(features, dtype=np.float32)
    degree = np.zeros((features.shape[0], 1), dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        np.add.at(agg, src, np.abs(features[src] - features[dst]))
        np.add.at(degree, src, 1.0)
    return agg / np.maximum(degree, 1.0)


def _flag_lite_features(graph: ProcessedGraphData, top_k: int) -> np.ndarray:
    edge_index = _flag_neighbor_edges(graph, top_k=top_k)
    semantic_neighbor_mean = _neighbor_mean_features(graph.features, edge_index)
    return np.concatenate([graph.features, graph.text_features, semantic_neighbor_mean], axis=1).astype(np.float32)


def _flag_neighbor_edges(graph: ProcessedGraphData, top_k: int) -> np.ndarray:
    if _has_text_signal_matrix(graph.text_features):
        return filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=top_k)
    return _filter_topk_similarity_edges(graph.edge_index, graph.features, top_k=top_k)


def _has_text_signal_matrix(features: np.ndarray, eps: float = 1e-8) -> bool:
    return bool(features.size and features.shape[1] > 0 and np.any(np.linalg.norm(features, axis=1) > eps))


def _filter_topk_similarity_edges(edge_index: np.ndarray, features: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    scores = _cosine_edge_scores(edge_index, features)
    selected: list[int] = []
    for src in np.unique(edge_index[0]):
        positions = np.flatnonzero(edge_index[0] == src)
        ranked = positions[np.argsort(scores[positions])[::-1]]
        selected.extend(ranked[:top_k].tolist())
    return edge_index[:, np.array(selected, dtype=np.int64)] if selected else np.zeros((2, 0), dtype=np.int64)


def _cosine_edge_scores(edge_index: np.ndarray, features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if edge_index.size == 0:
        return np.array([], dtype=np.float32)
    src = features[edge_index[0]]
    dst = features[edge_index[1]]
    denom = np.linalg.norm(src, axis=1) * np.linalg.norm(dst, axis=1)
    scores = np.zeros(edge_index.shape[1], dtype=np.float32)
    valid = denom > eps
    scores[valid] = np.sum(src[valid] * dst[valid], axis=1) / denom[valid]
    return scores


def _dga_lite_features(graph: ProcessedGraphData, num_attribute_groups: int = 4, num_neighbor_groups: int = 4) -> np.ndarray:
    attribute_groups = _attribute_groups(graph.features, num_attribute_groups)
    relation_groups = _relation_groups_for_graph(graph, num_neighbor_groups)
    distance_groups = _edge_distance_groups(graph.features, graph.edge_index, num_neighbor_groups)
    neighbor_groups = (relation_groups[: graph.edge_index.shape[1]] + distance_groups) % num_neighbor_groups
    attr_contexts = _group_mean_by_edge_group(
        features=graph.features,
        edge_index=graph.edge_index,
        group_ids=attribute_groups,
        num_groups=num_attribute_groups,
        use_neighbor_group=True,
    )
    neighbor_contexts = _group_mean_by_edge_group(
        features=graph.features,
        edge_index=graph.edge_index,
        group_ids=neighbor_groups,
        num_groups=num_neighbor_groups,
        use_neighbor_group=False,
    )
    return np.concatenate([graph.features, attr_contexts.reshape(graph.features.shape[0], -1), neighbor_contexts.reshape(graph.features.shape[0], -1)], axis=1).astype(np.float32)


def _attribute_groups(features: np.ndarray, num_groups: int = 4) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    scores = np.mean(features, axis=1)
    order = np.argsort(scores)
    groups = np.zeros(features.shape[0], dtype=np.int64)
    ranks = np.arange(features.shape[0], dtype=np.int64)
    groups[order] = np.minimum(ranks * int(num_groups) // max(features.shape[0], 1), int(num_groups) - 1)
    return groups


def _relation_groups_for_graph(graph: ProcessedGraphData, num_groups: int = 4) -> np.ndarray:
    if graph.edge_index.size == 0:
        return np.zeros(0, dtype=np.int64)
    relation_names = []
    for src, dst, edge_type in graph.edges[[schema.SRC, schema.DST, schema.EDGE_TYPE]].itertuples(index=False, name=None):
        if src in graph.node_id_to_idx and dst in graph.node_id_to_idx:
            relation_names.append(str(edge_type))
    mapping = {name: index % int(num_groups) for index, name in enumerate(sorted(set(relation_names)))}
    values = [mapping.get(name, 0) for name in relation_names]
    if len(values) < graph.edge_index.shape[1]:
        values.extend([0] * (graph.edge_index.shape[1] - len(values)))
    return np.asarray(values[: graph.edge_index.shape[1]], dtype=np.int64)


def _edge_distance_groups(features: np.ndarray, edge_index: np.ndarray, num_groups: int = 4) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros(0, dtype=np.int64)
    distances = np.linalg.norm(features[edge_index[0]] - features[edge_index[1]], axis=1)
    span = max(float(np.max(distances) - np.min(distances)), 1e-8)
    normalized = (distances - float(np.min(distances))) / span
    return np.minimum((normalized * int(num_groups)).astype(np.int64), int(num_groups) - 1)


def _group_mean_by_edge_group(
    features: np.ndarray,
    edge_index: np.ndarray,
    group_ids: np.ndarray,
    num_groups: int,
    use_neighbor_group: bool,
) -> np.ndarray:
    out = np.zeros((features.shape[0], int(num_groups), features.shape[1]), dtype=np.float32)
    degree = np.zeros((features.shape[0], int(num_groups), 1), dtype=np.float32)
    if edge_index.size == 0:
        return out
    src = edge_index[0]
    dst = edge_index[1]
    edge_groups = group_ids[dst] if use_neighbor_group else group_ids[: edge_index.shape[1]]
    for group in range(int(num_groups)):
        mask = edge_groups == group
        if not np.any(mask):
            continue
        np.add.at(out[:, group, :], src[mask], features[dst[mask]])
        np.add.at(degree[:, group, :], src[mask], 1.0)
    return out / np.maximum(degree, 1.0)


def _prepare_official_features(
    graph: ProcessedGraphData,
    homophilic_topk: int,
    heterophilic_topk: int,
    max_candidates_per_node: int,
    target_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, float | int]]:
    x = graph.features.astype(np.float32, copy=False)
    num_nodes, feature_dim = x.shape
    edge_index = graph.edge_index
    relation_names = _edge_relation_names_for_graph(graph)
    relation_values = sorted(set(relation_names)) or ["rel_0"]
    relation_to_idx = {name: idx for idx, name in enumerate(relation_values)}
    relation_dim = max(len(relation_values), 1)
    relation_counts = {name: relation_names.count(name) for name in relation_values}
    max_relation_count = max(relation_counts.values()) if relation_counts else 1
    relation_rarity = np.asarray(
        [1.0 - relation_counts.get(name, 0) / max(max_relation_count, 1) for name in relation_names],
        dtype=np.float32,
    )
    degree = _undirected_degree(edge_index, num_nodes)
    degree_dev = _edge_degree_deviation(edge_index, degree)
    similarity = _cosine_edge_scores(edge_index, x)
    feature_distance = np.clip(1.0 - similarity, 0.0, 1.0).astype(np.float32)
    local_z = _local_feature_zscores(edge_index, feature_distance)
    hetero_score = (
        0.45 * feature_distance
        + 0.25 * relation_rarity
        + 0.20 * local_z
        + 0.10 * degree_dev
    ).astype(np.float32)

    homo = np.zeros_like(x, dtype=np.float32)
    hetero = np.zeros_like(x, dtype=np.float32)
    feature_deviation = np.zeros_like(x, dtype=np.float32)
    relation_type = np.zeros((num_nodes, relation_dim), dtype=np.float32)
    selected_feature_distances: list[float] = []
    selected_homo_similarities: list[float] = []
    selected_relation_rarity: list[float] = []
    selected_counts: list[int] = []
    max_pool = max(int(max_candidates_per_node), int(heterophilic_topk), 1)
    positions_by_src: dict[int, list[int]] = {}
    if edge_index.size:
        for position, src in enumerate(edge_index[0]):
            positions_by_src.setdefault(int(src), []).append(int(position))
    if target_indices is None:
        source_nodes = sorted(positions_by_src)
    else:
        source_nodes = [
            int(index)
            for index in np.asarray(target_indices, dtype=np.int64)
            if 0 <= int(index) < num_nodes
        ]
    for src in source_nodes:
        src = int(src)
        positions = np.asarray(positions_by_src.get(src, []), dtype=np.int64)
        if positions.size == 0:
            selected_counts.append(0)
            continue
        homo_positions = positions[np.argsort(similarity[positions])[::-1]][: max(int(homophilic_topk), 0)]
        hetero_pool = positions[np.argsort(hetero_score[positions])[::-1]][:max_pool]
        hetero_positions = hetero_pool[: max(int(heterophilic_topk), 0)]
        if homo_positions.size:
            dst = edge_index[1, homo_positions]
            weights = _normalize_positive(similarity[homo_positions])
            homo[src] = np.sum(x[dst] * weights[:, None], axis=0)
            selected_homo_similarities.extend(similarity[homo_positions].astype(float).tolist())
        if hetero_positions.size:
            dst = edge_index[1, hetero_positions]
            weights = _normalize_positive(hetero_score[hetero_positions])
            hetero[src] = np.sum(x[dst] * weights[:, None], axis=0)
            feature_deviation[src] = np.sum(np.abs(x[src] - x[dst]) * weights[:, None], axis=0)
            selected_feature_distances.extend(feature_distance[hetero_positions].astype(float).tolist())
            selected_relation_rarity.extend(relation_rarity[hetero_positions].astype(float).tolist())
        relation_positions = np.concatenate([homo_positions, hetero_positions]) if homo_positions.size or hetero_positions.size else np.array([], dtype=np.int64)
        if relation_positions.size:
            for position in relation_positions:
                relation_type[src, relation_to_idx.get(relation_names[int(position)], 0)] += 1.0
            relation_type[src] /= max(float(np.sum(relation_type[src])), 1.0)
        selected_counts.append(int(homo_positions.size + hetero_positions.size))

    features = np.concatenate([x, homo, hetero, feature_deviation, relation_type], axis=1).astype(np.float32)
    feature_dims = {
        "target_dim": int(feature_dim),
        "homo_dim": int(feature_dim),
        "hetero_dim": int(feature_dim),
        "feature_deviation_dim": int(feature_dim),
        "relation_dim": int(relation_dim),
    }
    diagnostics = {
        "official_avg_feature_distance_selected": float(np.mean(selected_feature_distances)) if selected_feature_distances else 0.0,
        "official_avg_homo_similarity_selected": float(np.mean(selected_homo_similarities)) if selected_homo_similarities else 0.0,
        "official_avg_relation_rarity": float(np.mean(selected_relation_rarity)) if selected_relation_rarity else 0.0,
        "official_num_relation_types": int(relation_dim),
        "official_topk_homo": int(homophilic_topk),
        "official_topk_hetero": int(heterophilic_topk),
        "avg_selected_neighbors": float(np.mean(selected_counts)) if selected_counts else 0.0,
    }
    return features, feature_dims, diagnostics


def _edge_relation_names_for_graph(graph: ProcessedGraphData) -> list[str]:
    names: list[str] = []
    for src, dst, edge_type in graph.edges[[schema.SRC, schema.DST, schema.EDGE_TYPE]].itertuples(index=False, name=None):
        if src in graph.node_id_to_idx and dst in graph.node_id_to_idx:
            names.append(str(edge_type))
    if len(names) < graph.edge_index.shape[1]:
        names.extend(["rel_0"] * (graph.edge_index.shape[1] - len(names)))
    return names[: graph.edge_index.shape[1]]


def _undirected_degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    degree = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.size:
        np.add.at(degree, edge_index[0], 1.0)
        np.add.at(degree, edge_index[1], 1.0)
    return degree


def _edge_degree_deviation(edge_index: np.ndarray, degree: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros(0, dtype=np.float32)
    values = np.abs(np.log1p(degree[edge_index[0]]) - np.log1p(degree[edge_index[1]])).astype(np.float32)
    max_value = float(np.max(values)) if values.size else 0.0
    return values / max(max_value, 1e-8)


def _local_feature_zscores(edge_index: np.ndarray, distances: np.ndarray) -> np.ndarray:
    out = np.zeros_like(distances, dtype=np.float32)
    if edge_index.size == 0:
        return out
    for src in np.unique(edge_index[0]):
        positions = np.flatnonzero(edge_index[0] == src)
        values = distances[positions]
        if values.size <= 1:
            out[positions] = values
            continue
        std = float(np.std(values))
        z = (values - float(np.mean(values))) / max(std, 1e-8)
        out[positions] = np.clip((z + 2.0) / 4.0, 0.0, 1.0)
    return out


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float32)
    weights = np.maximum(weights, 0.0) + 1e-6
    return weights / max(float(np.sum(weights)), 1e-6)


def _baseline_avg_selected_neighbors(
    graph: ProcessedGraphData,
    model_name: str,
    target_indices: np.ndarray,
    top_k: int,
) -> float:
    if model_name == "mlp":
        return 0.0
    edge_index = _edge_index_for_model(graph, model_name, top_k)
    return _average_out_degree(edge_index, target_indices)


def _hero_avg_selected_neighbors(hero_artifacts: dict[str, Any]) -> float:
    chains_by_idx = hero_artifacts.get("chains_by_idx", {})
    if chains_by_idx:
        return float(np.mean([len(chains) for chains in chains_by_idx.values()]))
    candidates_by_target = hero_artifacts.get("candidates_by_target", {})
    if candidates_by_target:
        return float(np.mean([len(candidates) for candidates in candidates_by_target.values()]))
    return 0.0


def _average_out_degree(edge_index: np.ndarray, target_indices: np.ndarray) -> float:
    if edge_index.size == 0 or len(target_indices) == 0:
        return 0.0
    counts = []
    for index in target_indices:
        counts.append(float(np.sum(edge_index[0] == int(index))))
    return float(np.mean(counts)) if counts else 0.0


def _idx_to_node_id(graph: ProcessedGraphData) -> dict[int, str]:
    return {idx: node_id for node_id, idx in graph.node_id_to_idx.items()}


def _write_hero_explanations(
    graph: ProcessedGraphData,
    dataset: str,
    model_name: str,
    seed: int,
    output_root: str | Path,
    test_idx: np.ndarray,
    scores: np.ndarray,
    scores_without_chains: np.ndarray,
    hero_artifacts: dict[str, Any],
) -> dict[str, float]:
    variant_debug = hero_artifacts.get("variant_debug", {})
    experiment_tag = str(variant_debug.get("experiment_tag", ""))
    if experiment_tag:
        path = Path(output_root) / "explanations_llm_comparison" / dataset / experiment_tag / model_name / f"seed_{seed}" / "examples.jsonl"
    else:
        path = Path(output_root) / "explanations" / dataset / model_name / f"seed_{seed}" / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_id = _idx_to_node_id(graph)
    chains_by_idx = hero_artifacts.get("chains_by_idx", {})
    necessities_all: list[float] = []
    necessities_pos: list[float] = []
    necessities_neg: list[float] = []
    chain_counts: list[int] = []
    evidence_recalls: list[float] = []
    full_probs_pos: list[float] = []
    no_chain_probs_pos: list[float] = []
    full_probs_neg: list[float] = []
    no_chain_probs_neg: list[float] = []
    for local_pos, node_idx in enumerate(test_idx):
        node_idx = int(node_idx)
        chains = chains_by_idx.get(node_idx, [])
        full_prob = float(scores[local_pos])
        no_chain_prob = float(scores_without_chains[local_pos])
        score_drop = full_prob - no_chain_prob
        necessities_all.append(score_drop)
        if int(graph.labels[node_idx]) == 1:
            necessities_pos.append(score_drop)
            full_probs_pos.append(full_prob)
            no_chain_probs_pos.append(no_chain_prob)
        else:
            necessities_neg.append(score_drop)
            full_probs_neg.append(full_prob)
            no_chain_probs_neg.append(no_chain_prob)
        chain_counts.append(len(chains))
        evidence_recalls.append(_evidence_recall_for_node(idx_to_id[node_idx], chains, graph.evidence_gt))

    with path.open("w", encoding="utf-8") as handle:
        for local_pos, node_idx in enumerate(test_idx[:50]):
            node_idx = int(node_idx)
            chains = chains_by_idx.get(node_idx, [])
            pred_prob = float(scores[local_pos])
            score_without = float(scores_without_chains[local_pos])
            score_drop = pred_prob - score_without
            payload = {
                "target_id": idx_to_id[node_idx],
                "label": int(graph.labels[node_idx]),
                "pred_prob": pred_prob,
                "pred_without_chains": score_without,
                "score_drop": score_drop,
                "top_chains": [
                    {
                        "chain_nodes": chain["chain_nodes"],
                        "mechanism": chain["mechanism"],
                        "risk_relevance": int(chain.get("risk_relevance", 0)),
                        "chain_score": float(chain["chain_score"]),
                        "chain_quality": float(chain.get("chain_quality", 0.0)),
                        "rationale": chain["rationale"],
                    }
                    for chain in chains
                ],
                "score_without_chains": score_without,
                "necessity": score_drop,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    necessity_all = float(np.mean(necessities_all)) if necessities_all else 0.0
    necessity_pos = float(np.mean(necessities_pos)) if necessities_pos else 0.0
    necessity_neg = float(np.mean(necessities_neg)) if necessities_neg else 0.0
    necessity_gap = necessity_pos - necessity_neg
    avg_num_chains = float(np.mean(chain_counts)) if chain_counts else 0.0
    model_diagnostics = hero_artifacts.get("model_diagnostics", {})
    metrics = {
        "use_hetero": bool(hero_artifacts.get("use_hetero", False)),
        "use_chain": bool(hero_artifacts.get("use_chain", False)),
        "use_mechanism": bool(hero_artifacts.get("use_mechanism", False)),
        "use_chain_encoder": bool(hero_artifacts.get("use_chain_encoder", False)),
        "use_mock_llm_mechanism": bool(hero_artifacts.get("use_mock_llm_mechanism", False)),
        "branch_mask_target": int(variant_debug.get("branch_mask_target", 0)),
        "branch_mask_homo": int(variant_debug.get("branch_mask_homo", 0)),
        "branch_mask_hetero": int(variant_debug.get("branch_mask_hetero", 0)),
        "branch_mask_mechanism": int(variant_debug.get("branch_mask_mechanism", 0)),
        "branch_mask_chain": int(variant_debug.get("branch_mask_chain", 0)),
        "num_homophilic_neighbors_used": int(variant_debug.get("num_homophilic_neighbors_used", 0)),
        "num_heterophilic_neighbors_used": int(variant_debug.get("num_heterophilic_neighbors_used", 0)),
        "num_chains_used": int(variant_debug.get("num_chains_used", 0)),
        "num_mechanism_labels_used": int(variant_debug.get("num_mechanism_labels_used", 0)),
        "external_llm_label_file": str(variant_debug.get("external_llm_label_file", "")),
        "external_llm_labels_used": int(variant_debug.get("external_llm_labels_used", 0)),
        "mock_llm_labels_used": int(variant_debug.get("mock_llm_labels_used", 0)),
        "num_external_llm_labels_used": int(variant_debug.get("num_external_llm_labels_used", variant_debug.get("external_llm_labels_used", 0))),
        "num_missing_llm_pairs": int(variant_debug.get("num_missing_llm_pairs", 0)),
        "disable_llm_fallback": bool(variant_debug.get("disable_llm_fallback", False)),
        "num_raw_chains": int(variant_debug.get("num_raw_chains", 0)),
        "num_filtered_chains": int(variant_debug.get("num_filtered_chains", 0)),
        "chain_filter_keep_rate": float(variant_debug.get("chain_filter_keep_rate", 0.0)),
        "evidence_recall_proxy": float(np.mean(evidence_recalls)) if evidence_recalls else 0.0,
        "evidence_necessity": necessity_all,
        "evidence_necessity_score": necessity_all,
        "avg_evidence_necessity_all": necessity_all,
        "avg_evidence_necessity_pos": necessity_pos,
        "avg_evidence_necessity_neg": necessity_neg,
        "evidence_necessity_gap": necessity_gap,
        "avg_num_chains": avg_num_chains,
        "avg_full_prob_pos": float(np.mean(full_probs_pos)) if full_probs_pos else 0.0,
        "avg_no_chain_prob_pos": float(np.mean(no_chain_probs_pos)) if no_chain_probs_pos else 0.0,
        "avg_full_prob_neg": float(np.mean(full_probs_neg)) if full_probs_neg else 0.0,
        "avg_no_chain_prob_neg": float(np.mean(no_chain_probs_neg)) if no_chain_probs_neg else 0.0,
        "avg_chain_gate": float(model_diagnostics.get("avg_chain_gate", 0.0)),
        "avg_chain_gate_pos": float(model_diagnostics.get("avg_chain_gate_pos", 0.0)),
        "avg_chain_gate_neg": float(model_diagnostics.get("avg_chain_gate_neg", 0.0)),
        "avg_homo_gate": float(model_diagnostics.get("avg_homo_gate", 0.0)),
        "avg_target_gate": float(model_diagnostics.get("avg_target_gate", 0.0)),
        "avg_chain_quality": float(model_diagnostics.get("avg_chain_quality", 0.0)),
        "avg_chain_quality_pos": float(model_diagnostics.get("avg_chain_quality_pos", 0.0)),
        "avg_chain_quality_neg": float(model_diagnostics.get("avg_chain_quality_neg", 0.0)),
        "target_repr_norm": float(model_diagnostics.get("target_repr_norm", 0.0)),
        "homo_repr_norm": float(model_diagnostics.get("homo_repr_norm", 0.0)),
        "hetero_repr_norm": float(model_diagnostics.get("hetero_repr_norm", 0.0)),
        "mechanism_repr_norm": float(model_diagnostics.get("mechanism_repr_norm", 0.0)),
        "chain_repr_norm": float(model_diagnostics.get("chain_repr_norm", 0.0)),
        "final_repr_norm": float(model_diagnostics.get("final_repr_norm", 0.0)),
        "delta_zero_hetero": float(model_diagnostics.get("delta_zero_hetero", 0.0)),
        "delta_zero_chain": float(model_diagnostics.get("delta_zero_chain", 0.0)),
        "delta_zero_mechanism": float(model_diagnostics.get("delta_zero_mechanism", 0.0)),
        "lambda_chain_pos": float(model_diagnostics.get("lambda_chain_pos", 0.0)),
        "lambda_chain_neg": float(model_diagnostics.get("lambda_chain_neg", 0.0)),
        "chain_pos_loss": float(model_diagnostics.get("chain_pos_loss", 0.0)),
        "chain_neg_loss": float(model_diagnostics.get("chain_neg_loss", 0.0)),
    }
    if variant_debug.get("llm_label_file") or variant_debug.get("experiment_tag"):
        metrics.update(
            {
                "llm_label_file": str(variant_debug.get("llm_label_file", variant_debug.get("external_llm_label_file", ""))),
                "experiment_tag": str(variant_debug.get("experiment_tag", "")),
                "llm_labeler": str(variant_debug.get("llm_labeler", "mock")),
                "llm_label_coverage_rate": float(variant_debug.get("llm_label_coverage_rate", 0.0)),
                "llm_label_total_pairs": int(variant_debug.get("llm_label_total_pairs", 0)),
            }
        )
    _write_hero_diagnostics(
        output_root=output_root,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        metrics={**metrics, "avg_selected_neighbors": _hero_avg_selected_neighbors(hero_artifacts)},
    )
    _write_variant_debug(
        output_root=output_root,
        dataset=dataset,
        model_name=model_name,
        seed=seed,
        metrics={**metrics, "avg_selected_neighbors": _hero_avg_selected_neighbors(hero_artifacts)},
    )
    return metrics


def _write_hero_diagnostics(
    output_root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    metrics: dict[str, float],
) -> None:
    experiment_tag = str(metrics.get("experiment_tag", ""))
    if experiment_tag:
        path = Path(output_root) / "diagnostics_llm_comparison" / dataset / experiment_tag / model_name / f"seed_{seed}" / "hero_diagnostics.json"
    else:
        path = Path(output_root) / "diagnostics" / dataset / model_name / f"seed_{seed}" / "hero_diagnostics.json"
    payload = {
        "avg_chain_gate": float(metrics.get("avg_chain_gate", 0.0)),
        "avg_chain_gate_pos": float(metrics.get("avg_chain_gate_pos", 0.0)),
        "avg_chain_gate_neg": float(metrics.get("avg_chain_gate_neg", 0.0)),
        "avg_chain_quality": float(metrics.get("avg_chain_quality", 0.0)),
        "avg_chain_quality_pos": float(metrics.get("avg_chain_quality_pos", 0.0)),
        "avg_chain_quality_neg": float(metrics.get("avg_chain_quality_neg", 0.0)),
        "num_raw_chains": int(metrics.get("num_raw_chains", 0)),
        "num_filtered_chains": int(metrics.get("num_filtered_chains", 0)),
        "chain_filter_keep_rate": float(metrics.get("chain_filter_keep_rate", 0.0)),
        "avg_num_chains": float(metrics.get("avg_num_chains", 0.0)),
        "avg_selected_neighbors": float(metrics.get("avg_selected_neighbors", 0.0)),
        "avg_full_prob_pos": float(metrics.get("avg_full_prob_pos", 0.0)),
        "avg_no_chain_prob_pos": float(metrics.get("avg_no_chain_prob_pos", 0.0)),
        "avg_full_prob_neg": float(metrics.get("avg_full_prob_neg", 0.0)),
        "avg_no_chain_prob_neg": float(metrics.get("avg_no_chain_prob_neg", 0.0)),
        "evidence_necessity": float(metrics.get("evidence_necessity", 0.0)),
        "evidence_necessity_gap": float(metrics.get("evidence_necessity_gap", 0.0)),
    }
    write_json(path, payload)


def _write_variant_debug(
    output_root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    metrics: dict[str, Any],
) -> None:
    experiment_tag = str(metrics.get("experiment_tag", ""))
    if experiment_tag:
        path = Path(output_root) / "diagnostics_llm_comparison" / dataset / experiment_tag / model_name / f"seed_{seed}" / "variant_debug.json"
    else:
        path = Path(output_root) / "diagnostics" / dataset / model_name / f"seed_{seed}" / "variant_debug.json"
    payload = {
        "dataset": dataset,
        "model": model_name,
        "seed": int(seed),
        "use_hetero": bool(metrics.get("use_hetero", False)),
        "use_chain": bool(metrics.get("use_chain", False)),
        "use_mechanism": bool(metrics.get("use_mechanism", False)),
        "branch_mask_target": int(metrics.get("branch_mask_target", 0)),
        "branch_mask_homo": int(metrics.get("branch_mask_homo", 0)),
        "branch_mask_hetero": int(metrics.get("branch_mask_hetero", 0)),
        "branch_mask_mechanism": int(metrics.get("branch_mask_mechanism", 0)),
        "branch_mask_chain": int(metrics.get("branch_mask_chain", 0)),
        "num_homophilic_neighbors_used": int(metrics.get("num_homophilic_neighbors_used", 0)),
        "num_heterophilic_neighbors_used": int(metrics.get("num_heterophilic_neighbors_used", 0)),
        "num_chains_used": int(metrics.get("num_chains_used", 0)),
        "num_mechanism_labels_used": int(metrics.get("num_mechanism_labels_used", 0)),
        "external_llm_label_file": str(metrics.get("external_llm_label_file", "")),
        "llm_label_file": str(metrics.get("llm_label_file", metrics.get("external_llm_label_file", ""))),
        "experiment_tag": str(metrics.get("experiment_tag", "")),
        "llm_labeler": str(metrics.get("llm_labeler", "mock")),
        "llm_label_coverage_rate": float(metrics.get("llm_label_coverage_rate", 0.0)),
        "llm_label_total_pairs": int(metrics.get("llm_label_total_pairs", 0)),
        "num_external_llm_labels_used": int(metrics.get("num_external_llm_labels_used", metrics.get("external_llm_labels_used", 0))),
        "num_missing_llm_pairs": int(metrics.get("num_missing_llm_pairs", 0)),
        "disable_llm_fallback": bool(metrics.get("disable_llm_fallback", False)),
        "external_llm_labels_used": int(metrics.get("external_llm_labels_used", 0)),
        "mock_llm_labels_used": int(metrics.get("mock_llm_labels_used", 0)),
        "num_raw_chains": int(metrics.get("num_raw_chains", 0)),
        "num_filtered_chains": int(metrics.get("num_filtered_chains", 0)),
        "chain_filter_keep_rate": float(metrics.get("chain_filter_keep_rate", 0.0)),
        "avg_chain_gate": float(metrics.get("avg_chain_gate", 0.0)),
        "avg_chain_gate_pos": float(metrics.get("avg_chain_gate_pos", 0.0)),
        "avg_chain_gate_neg": float(metrics.get("avg_chain_gate_neg", 0.0)),
        "avg_chain_quality": float(metrics.get("avg_chain_quality", 0.0)),
        "avg_chain_quality_pos": float(metrics.get("avg_chain_quality_pos", 0.0)),
        "avg_chain_quality_neg": float(metrics.get("avg_chain_quality_neg", 0.0)),
        "avg_selected_neighbors": float(metrics.get("avg_selected_neighbors", 0.0)),
        "avg_num_chains": float(metrics.get("avg_num_chains", 0.0)),
        "target_repr_norm": float(metrics.get("target_repr_norm", 0.0)),
        "homo_repr_norm": float(metrics.get("homo_repr_norm", 0.0)),
        "hetero_repr_norm": float(metrics.get("hetero_repr_norm", 0.0)),
        "mechanism_repr_norm": float(metrics.get("mechanism_repr_norm", 0.0)),
        "chain_repr_norm": float(metrics.get("chain_repr_norm", 0.0)),
        "final_repr_norm": float(metrics.get("final_repr_norm", 0.0)),
        "delta_zero_hetero": float(metrics.get("delta_zero_hetero", 0.0)),
        "delta_zero_chain": float(metrics.get("delta_zero_chain", 0.0)),
        "delta_zero_mechanism": float(metrics.get("delta_zero_mechanism", 0.0)),
    }
    write_json(path, payload)


def _evidence_recall_for_node(target_id: str, chains: list[dict[str, Any]], evidence_gt: dict) -> float:
    gt = evidence_gt.get(target_id, {})
    gt_neighbors = set(gt.get("evidence_neighbors", []))
    if not gt_neighbors:
        return 1.0 if chains else 0.0
    predicted = {
        node_id
        for chain in chains
        for node_id in chain.get("chain_nodes", [])[1:]
    }
    return float(len(gt_neighbors & predicted) / len(gt_neighbors))


def _fit_hero_official_model(
    graph: ProcessedGraphData,
    model_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    pos_weight: float,
    homophilic_topk: int,
    heterophilic_topk: int,
    max_candidates_per_node: int,
    enable_official_chain: bool,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    retrieval_start = time.perf_counter()
    features, feature_dims, diagnostics = _prepare_official_features(
        graph=graph,
        homophilic_topk=homophilic_topk,
        heterophilic_topk=heterophilic_topk,
        max_candidates_per_node=max_candidates_per_node,
        target_indices=np.concatenate([train_idx, val_idx, test_idx]),
    )
    retrieval_time = time.perf_counter() - retrieval_start
    diagnostics.update(
        {
            "official_mode": True,
            "use_official_chain": bool(enable_official_chain),
            "lambda_chain_pos": 0.0,
            "lambda_chain_neg": 0.0,
            "num_chains_used": 0,
            "num_raw_chains": 0,
            "num_filtered_chains": 0,
        }
    )
    variant_flags = _official_variant_flags(model_name)
    if torch is None:
        val_scores, test_scores, _without, checkpoint = _fit_numpy_feature_model(
            features=features,
            features_without_chains=features,
            labels=graph.labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            pos_weight=pos_weight,
        )
        checkpoint["official_feature_dims"] = feature_dims
        return val_scores, test_scores, checkpoint, {"diagnostics": diagnostics, "time_retrieval_sec": retrieval_time}

    from src.models.hero_official import HEROOfficial

    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(graph.labels, dtype=torch.float32, device=device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_eval_idx = val_idx if val_idx.size else train_idx
    val_tensor = torch.tensor(val_eval_idx, dtype=torch.long, device=device)
    test_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)
    target_x, homo_x, hetero_x, deviation_x, relation_x = _split_official_features_torch(x, feature_dims)
    model = HEROOfficial(
        input_dim=int(feature_dims["target_dim"]),
        relation_dim=int(feature_dims["relation_dim"]),
        hidden_dim=hidden_dim,
        output_dim=1,
        use_hetero=variant_flags["use_official_hetero"],
        use_relation=variant_flags["use_official_relation"],
        use_feature_deviation=variant_flags["use_official_feature_deviation"],
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device))
    best_state = None
    best_score = -1.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(target_x, homo_x, hetero_x, deviation_x, relation_x)
        loss = criterion(logits[train_tensor], y[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(target_x, homo_x, hetero_x, deviation_x, relation_x)
            val_scores = torch.sigmoid(logits[val_tensor]).detach().cpu().numpy()
        val_metrics = binary_classification_metrics(graph.labels[val_eval_idx], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, details = model(target_x, homo_x, hetero_x, deviation_x, relation_x, return_details=True)
        val_scores = torch.sigmoid(logits[val_tensor]).detach().cpu().numpy()
        test_scores = torch.sigmoid(logits[test_tensor]).detach().cpu().numpy()
        diagnostics.update(_official_gate_diagnostics(details, test_tensor))
    return val_scores, test_scores, {"model_state_dict": best_state, "model_name": model_name, "pos_weight": float(pos_weight)}, {
        "diagnostics": diagnostics,
        "time_retrieval_sec": retrieval_time,
    }


def _split_official_features_torch(x, feature_dims: dict[str, int]):
    target_dim = int(feature_dims["target_dim"])
    homo_dim = int(feature_dims["homo_dim"])
    hetero_dim = int(feature_dims["hetero_dim"])
    deviation_dim = int(feature_dims["feature_deviation_dim"])
    relation_dim = int(feature_dims["relation_dim"])
    target_start = 0
    homo_start = target_start + target_dim
    hetero_start = homo_start + homo_dim
    deviation_start = hetero_start + hetero_dim
    relation_start = deviation_start + deviation_dim
    return (
        x[:, target_start:homo_start],
        x[:, homo_start:hetero_start],
        x[:, hetero_start:deviation_start],
        x[:, deviation_start:relation_start],
        x[:, relation_start : relation_start + relation_dim],
    )


def _official_gate_diagnostics(details: dict[str, Any], indices) -> dict[str, float]:
    diagnostics = {
        "avg_relation_gate": 0.0,
        "avg_feature_deviation_gate": 0.0,
        "avg_official_hetero_gate": 0.0,
    }
    mapping = {
        "avg_relation_gate": "relation_gate",
        "avg_feature_deviation_gate": "feature_deviation_gate",
        "avg_official_hetero_gate": "official_hetero_gate",
    }
    for metric, key in mapping.items():
        value = details.get(key)
        if value is not None and value.numel():
            diagnostics[metric] = float(value[indices].detach().cpu().mean().item())
    return diagnostics


def _fit_torch_model(
    graph: ProcessedGraphData,
    model_name: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    top_k: int,
    pos_weight: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from src.models.dga_gnn_lite import DGAGNNLite
    from src.models.flag_lite import FLAGLite
    from src.models.graphsage import GraphSAGE
    from src.models.mlp import MLP
    from src.models.rulehetero_gnn import RuleHeteroGNN
    from src.models.sec_gfd_lite import SECGFDLite
    from src.models.semsim_gnn import SemSimGNN

    torch.manual_seed(seed)
    model_cls = {
        "mlp": MLP,
        "graphsage": GraphSAGE,
        "semsim_gnn": SemSimGNN,
        "rulehetero_gnn": RuleHeteroGNN,
        "sec_gfd_lite": SECGFDLite,
        "dga_gnn_lite": DGAGNNLite,
        "flag_lite": FLAGLite,
    }[model_name]
    model_kwargs = {"input_dim": graph.features.shape[1], "hidden_dim": hidden_dim, "output_dim": 1}
    if model_name == "flag_lite":
        model_kwargs["text_dim"] = graph.text_features.shape[1]
    model = model_cls(**model_kwargs).to(device)
    x = torch.tensor(graph.features, dtype=torch.float32, device=device)
    text_x = torch.tensor(graph.text_features, dtype=torch.float32, device=device)
    labels = torch.tensor(graph.labels, dtype=torch.float32, device=device)
    edge_index = torch.tensor(_edge_index_for_model(graph, model_name, top_k), dtype=torch.long, device=device)
    attribute_groups = torch.tensor(_attribute_groups(graph.features, 4), dtype=torch.long, device=device)
    relation_groups = torch.tensor(_relation_groups_for_graph(graph, 4), dtype=torch.long, device=device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_eval_idx = val_idx if val_idx.size else train_idx
    val_tensor = torch.tensor(val_eval_idx, dtype=torch.long, device=device)
    test_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device))
    best_state = None
    best_score = -1.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = _model_logits(model, model_name, x, edge_index, text_x, attribute_groups, relation_groups)
        loss = criterion(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = _model_logits(model, model_name, x, edge_index, text_x, attribute_groups, relation_groups)
            val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        val_metrics = binary_classification_metrics(graph.labels[val_eval_idx], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = _model_logits(model, model_name, x, edge_index, text_x, attribute_groups, relation_groups)
        val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        test_scores = torch.sigmoid(logits[test_tensor]).cpu().numpy()
    return val_scores, test_scores, {"model_state_dict": best_state, "model_name": model_name, "pos_weight": float(pos_weight)}


def _model_logits(model, model_name: str, x, edge_index, text_x=None, attribute_groups=None, relation_groups=None):
    if model_name == "mlp":
        logits = model(x)
    elif model_name == "flag_lite":
        logits = model(x, edge_index, text_x)
    elif model_name == "dga_gnn_lite":
        logits = model(x, edge_index, attribute_groups=attribute_groups, relation_groups=relation_groups)
    else:
        logits = model(x, edge_index)
    return logits.squeeze(-1)


def _split_hero_features_torch(x, feature_dims: dict[str, int]):
    target_dim = int(feature_dims["target_dim"])
    homo_dim = int(feature_dims["homo_dim"])
    hetero_dim = int(feature_dims["hetero_dim"])
    mechanism_dim = int(feature_dims["mechanism_dim"])
    chain_dim = int(feature_dims["chain_dim"])
    target_x = x[:, :target_dim]
    homo_x = x[:, target_dim : target_dim + homo_dim]
    hetero_start = target_dim + homo_dim
    mechanism_start = hetero_start + hetero_dim
    chain_start = mechanism_start + mechanism_dim
    hetero_x = x[:, hetero_start:mechanism_start]
    mechanism_x = x[:, mechanism_start:chain_start]
    chain_x = x[:, chain_start : chain_start + chain_dim]
    return target_x, homo_x, hetero_x, mechanism_x, chain_x


def _torch_gate_diagnostics(gates: dict[str, Any], indices, labels: torch.Tensor | None = None) -> dict[str, float]:
    diagnostics = {}
    for name, key in [
        ("avg_chain_gate", "chain_gate"),
        ("avg_homo_gate", "homo_gate"),
        ("avg_target_gate", "target_gate"),
    ]:
        value = gates.get(key)
        if value is None:
            diagnostics[name] = 0.0
            continue
        selected = value[indices]
        diagnostics[name] = float(selected.detach().cpu().mean().item()) if selected.numel() else 0.0
    chain_gate = gates.get("chain_gate")
    diagnostics["avg_chain_gate_pos"] = 0.0
    diagnostics["avg_chain_gate_neg"] = 0.0
    if labels is not None and chain_gate is not None:
        selected_gate = chain_gate[indices]
        selected_labels = labels[indices]
        pos_mask = selected_labels == 1.0
        neg_mask = selected_labels == 0.0
        if torch.any(pos_mask):
            diagnostics["avg_chain_gate_pos"] = float(selected_gate[pos_mask].detach().cpu().mean().item())
        if torch.any(neg_mask):
            diagnostics["avg_chain_gate_neg"] = float(selected_gate[neg_mask].detach().cpu().mean().item())
    return diagnostics


def _torch_chain_quality_diagnostics(chain_x, labels: torch.Tensor, indices) -> dict[str, float]:
    diagnostics = {
        "avg_chain_quality": 0.0,
        "avg_chain_quality_pos": 0.0,
        "avg_chain_quality_neg": 0.0,
    }
    if chain_x.numel() == 0 or chain_x.shape[1] == 0:
        return diagnostics
    selected_quality = chain_x[indices, -1].detach()
    selected_labels = labels[indices]
    if selected_quality.numel():
        diagnostics["avg_chain_quality"] = float(selected_quality.cpu().mean().item())
    pos_mask = selected_labels == 1.0
    neg_mask = selected_labels == 0.0
    if torch.any(pos_mask):
        diagnostics["avg_chain_quality_pos"] = float(selected_quality[pos_mask].cpu().mean().item())
    if torch.any(neg_mask):
        diagnostics["avg_chain_quality_neg"] = float(selected_quality[neg_mask].cpu().mean().item())
    return diagnostics


def _torch_repr_diagnostics(details: dict[str, Any], indices) -> dict[str, float]:
    diagnostics = {}
    for metric_name, detail_key in [
        ("target_repr_norm", "target_repr"),
        ("homo_repr_norm", "homo_repr"),
        ("hetero_repr_norm", "hetero_repr"),
        ("mechanism_repr_norm", "mechanism_repr"),
        ("chain_repr_norm", "chain_repr"),
        ("final_repr_norm", "final_repr"),
    ]:
        value = details.get(detail_key)
        if value is None:
            diagnostics[metric_name] = 0.0
            continue
        selected = value[indices]
        diagnostics[metric_name] = float(torch.linalg.norm(selected, dim=1).mean().detach().cpu().item()) if selected.numel() else 0.0
    return diagnostics


def _torch_branch_delta_diagnostics(
    model,
    target_x,
    homo_x,
    hetero_x,
    mechanism_x,
    chain_x,
    test_tensor,
    full_logits,
) -> dict[str, float]:
    prob_full = torch.sigmoid(full_logits[test_tensor])
    zero_hetero = torch.sigmoid(model(target_x, homo_x, hetero_x, mechanism_x, chain_x, zero_hetero=True)[test_tensor])
    zero_chain = torch.sigmoid(model(target_x, homo_x, hetero_x, mechanism_x, chain_x, zero_chain=True)[test_tensor])
    zero_mechanism = torch.sigmoid(model(target_x, homo_x, hetero_x, mechanism_x, chain_x, zero_mechanism=True)[test_tensor])
    return {
        "delta_zero_hetero": float(torch.mean(torch.abs(prob_full - zero_hetero)).detach().cpu().item()) if prob_full.numel() else 0.0,
        "delta_zero_chain": float(torch.mean(torch.abs(prob_full - zero_chain)).detach().cpu().item()) if prob_full.numel() else 0.0,
        "delta_zero_mechanism": float(torch.mean(torch.abs(prob_full - zero_mechanism)).detach().cpu().item()) if prob_full.numel() else 0.0,
    }


def _fit_torch_feature_model(
    features: np.ndarray,
    features_without_chains: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
    pos_weight: float,
    model_name: str,
    feature_dims: dict[str, int],
    use_hetero: bool,
    use_mechanism: bool,
    use_chain: bool,
    lambda_chain_pos: float,
    lambda_chain_neg: float,
    min_chain_quality: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if torch is None:
        return _fit_numpy_feature_model(
            features=features,
            features_without_chains=features_without_chains,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            epochs=epochs,
            lr=lr,
            pos_weight=pos_weight,
            feature_dims=feature_dims,
            use_hetero=use_hetero,
            use_mechanism=use_mechanism,
            use_chain=use_chain,
            lambda_chain_pos=lambda_chain_pos if model_name == "hero_gnn" else 0.0,
            lambda_chain_neg=lambda_chain_neg if model_name == "hero_gnn" else 0.0,
            min_chain_quality=min_chain_quality,
        )

    from src.models.hero_gnn import HEROGNN

    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_eval_idx = val_idx if val_idx.size else train_idx
    val_tensor = torch.tensor(val_eval_idx, dtype=torch.long, device=device)
    test_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)
    target_x, homo_x, hetero_x, mechanism_x, chain_x = _split_hero_features_torch(x, feature_dims)
    model = HEROGNN(
        input_dim=int(feature_dims["target_dim"]),
        hidden_dim=hidden_dim,
        output_dim=1,
        num_mechanisms=len(schema.EVIDENCE_MECHANISMS),
        use_heterophily=use_hetero,
        use_mechanism=use_mechanism,
        use_chain=use_chain,
        hetero_input_dim=int(feature_dims["hetero_dim"]),
        mechanism_input_dim=int(feature_dims["mechanism_dim"]),
        chain_input_dim=int(feature_dims["chain_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device))
    best_state = None
    best_score = -1.0
    last_chain_pos_loss = 0.0
    last_chain_neg_loss = 0.0
    active_lambda_pos = float(lambda_chain_pos) if model_name == "hero_gnn" and use_chain else 0.0
    active_lambda_neg = float(lambda_chain_neg) if model_name == "hero_gnn" and use_chain else 0.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(target_x, homo_x, hetero_x, mechanism_x, chain_x)
        loss = criterion(logits[train_tensor], y[train_tensor])
        chain_pos_loss = torch.tensor(0.0, dtype=loss.dtype)
        chain_neg_loss = torch.tensor(0.0, dtype=loss.dtype)
        if active_lambda_pos > 0.0 or active_lambda_neg > 0.0:
            no_chain_logits = model(target_x, homo_x, hetero_x, mechanism_x, chain_x, force_no_chain=True)
            chain_quality = chain_x[train_tensor, -1] if chain_x.shape[1] else torch.zeros_like(y[train_tensor])
            quality_mask = chain_quality >= float(min_chain_quality)
            pos_mask = (y[train_tensor] == 1.0) & quality_mask
            neg_mask = (y[train_tensor] == 0.0) & quality_mask
            if active_lambda_pos > 0.0 and torch.any(pos_mask):
                full_prob = torch.sigmoid(logits[train_tensor][pos_mask])
                no_chain_prob = torch.sigmoid(no_chain_logits[train_tensor][pos_mask])
                chain_pos_loss = torch.relu(no_chain_prob - full_prob).mean()
                loss = loss + active_lambda_pos * chain_pos_loss
            if active_lambda_neg > 0.0 and torch.any(neg_mask):
                full_prob = torch.sigmoid(logits[train_tensor][neg_mask])
                no_chain_prob = torch.sigmoid(no_chain_logits[train_tensor][neg_mask])
                chain_neg_loss = torch.relu(full_prob - no_chain_prob).mean()
                loss = loss + active_lambda_neg * chain_neg_loss
        last_chain_pos_loss = float(chain_pos_loss.detach().cpu().item())
        last_chain_neg_loss = float(chain_neg_loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(target_x, homo_x, hetero_x, mechanism_x, chain_x)
            val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        val_metrics = binary_classification_metrics(labels[val_eval_idx], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, details = model(target_x, homo_x, hetero_x, mechanism_x, chain_x, return_details=True)
        logits_without = model(target_x, homo_x, hetero_x, mechanism_x, chain_x, zero_chain=True)
        val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        test_scores = torch.sigmoid(logits[test_tensor]).cpu().numpy()
        scores_without_chains = torch.sigmoid(logits_without[test_tensor]).cpu().numpy()
        diagnostics = _torch_gate_diagnostics(details, test_tensor, labels=y)
        diagnostics.update(_torch_chain_quality_diagnostics(chain_x, labels=y, indices=test_tensor))
        diagnostics.update(_torch_repr_diagnostics(details, test_tensor))
        diagnostics.update(
            _torch_branch_delta_diagnostics(
                model=model,
                target_x=target_x,
                homo_x=homo_x,
                hetero_x=hetero_x,
                mechanism_x=mechanism_x,
                chain_x=chain_x,
                test_tensor=test_tensor,
                full_logits=logits,
            )
        )
    diagnostics.update(
        {
            "lambda_chain_pos": float(active_lambda_pos),
            "lambda_chain_neg": float(active_lambda_neg),
            "chain_pos_loss": float(last_chain_pos_loss),
            "chain_neg_loss": float(last_chain_neg_loss),
        }
    )
    return val_scores, test_scores, scores_without_chains, {
        "model_state_dict": best_state,
        "model_name": "hero_gnn_fusion",
        "pos_weight": float(pos_weight),
        "diagnostics": diagnostics,
    }


def _fit_numpy_feature_model(
    features: np.ndarray,
    features_without_chains: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    pos_weight: float,
    feature_dims: dict[str, int] | None = None,
    use_hetero: bool = False,
    use_mechanism: bool = False,
    use_chain: bool = False,
    lambda_chain_pos: float = 0.0,
    lambda_chain_neg: float = 0.0,
    min_chain_quality: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    active_lambda_pos = float(lambda_chain_pos) if use_chain else 0.0
    active_lambda_neg = float(lambda_chain_neg) if use_chain else 0.0
    if feature_dims:
        x, x_without, gate_diagnostics = _numpy_hero_fused_features(
            features=features,
            features_without_chains=features_without_chains,
            feature_dims=feature_dims,
            use_hetero=use_hetero,
            use_mechanism=use_mechanism,
            use_chain=use_chain,
        )
    else:
        x = np.asarray(features, dtype=np.float32)
        x_without = np.asarray(features_without_chains, dtype=np.float32)
        gate_diagnostics = {
            "avg_chain_gate": 0.0,
            "avg_homo_gate": 0.0,
            "avg_target_gate": 0.0,
        }
    y = np.asarray(labels, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx_eval = np.asarray(val_idx if val_idx.size else train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean) / std
    x_without_scaled = (x_without - mean) / std
    if feature_dims:
        x_no_chain_scaled = _zero_hero_branch_scaled(x_scaled, feature_dims, "chain")
    else:
        x_no_chain_scaled = x_without_scaled
    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = np.float32(0.0)
    train_y = y[train_idx]
    sample_weights = np.where(train_y == 1.0, float(pos_weight), 1.0).astype(np.float32)
    step = float(lr) if lr > 0 else 0.001
    chain_pos_loss = 0.0
    chain_neg_loss = 0.0
    # This is the same weighted logistic objective as BCEWithLogitsLoss(pos_weight);
    # it keeps the repo runnable in environments where torch is unavailable.
    for _epoch in range(max(1, epochs)):
        logits, grad_features = _numpy_full_logits_and_grad_features(
            x_full=x_scaled[train_idx],
            x_without=x_no_chain_scaled[train_idx],
            weights=weights,
            bias=float(bias),
            use_chain=use_chain,
        )
        probs = _sigmoid_np(logits)
        errors = (probs - train_y) * sample_weights
        denom = max(float(train_idx.size), 1.0)
        grad_w = (grad_features.T @ errors) / denom + 1e-4 * weights
        grad_b = np.sum(errors) / denom
        if active_lambda_pos > 0.0:
            chain_pos_loss, chain_grad_w, chain_grad_b = _numpy_chain_loss_grad(
                weights=weights,
                bias=float(bias),
                x_full=x_scaled,
                x_without=x_no_chain_scaled,
                raw_features=features,
                labels=y,
                train_idx=train_idx,
                use_chain=use_chain,
                feature_dims=feature_dims,
                min_chain_quality=min_chain_quality,
                positive_constraint=True,
            )
            grad_w += active_lambda_pos * chain_grad_w
            grad_b += active_lambda_pos * chain_grad_b
        if active_lambda_neg > 0.0:
            chain_neg_loss, chain_grad_w, chain_grad_b = _numpy_chain_loss_grad(
                weights=weights,
                bias=float(bias),
                x_full=x_scaled,
                x_without=x_no_chain_scaled,
                raw_features=features,
                labels=y,
                train_idx=train_idx,
                use_chain=use_chain,
                feature_dims=feature_dims,
                min_chain_quality=min_chain_quality,
                positive_constraint=False,
            )
            grad_w += active_lambda_neg * chain_grad_w
            grad_b += active_lambda_neg * chain_grad_b
        weights -= step * grad_w.astype(np.float32)
        bias = np.float32(bias - step * grad_b)

    scores = _numpy_predict_from_scaled(x_scaled, weights, float(bias), use_chain=use_chain, feature_dims=feature_dims).astype(np.float32)
    scores_without = _numpy_predict_from_scaled(
        x_no_chain_scaled,
        weights,
        float(bias),
        use_chain=use_chain,
        feature_dims=feature_dims,
    ).astype(np.float32)
    diagnostics = _gate_diagnostics_for_indices(gate_diagnostics, test_idx, labels=y)
    diagnostics.update(_numpy_chain_quality_diagnostics(features, feature_dims, labels=y, indices=test_idx))
    diagnostics.update(_numpy_repr_diagnostics(x_scaled, feature_dims, test_idx))
    diagnostics.update(_numpy_branch_delta_diagnostics(x_scaled, weights, float(bias), use_chain, feature_dims, test_idx))
    diagnostics.update(
        {
            "lambda_chain_pos": float(active_lambda_pos),
            "lambda_chain_neg": float(active_lambda_neg),
            "chain_pos_loss": float(chain_pos_loss),
            "chain_neg_loss": float(chain_neg_loss),
        }
    )
    return (
        scores[val_idx_eval],
        scores[test_idx],
        scores_without[test_idx],
        {
            "model_name": "numpy_weighted_logistic",
            "weights": weights,
            "bias": float(bias),
            "pos_weight": float(pos_weight),
            "diagnostics": diagnostics,
        },
    )


def _numpy_hero_fused_features(
    features: np.ndarray,
    features_without_chains: np.ndarray,
    feature_dims: dict[str, int],
    use_hetero: bool,
    use_mechanism: bool,
    use_chain: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    target, homo, hetero, mechanism, chain = _split_hero_features_np(features, feature_dims)
    target_without, homo_without, hetero_without, mechanism_without, _chain_without = _split_hero_features_np(features_without_chains, feature_dims)
    target_gate = np.ones((features.shape[0], 1), dtype=np.float32)
    homo_gate = np.ones((features.shape[0], 1), dtype=np.float32)
    hetero_gate = np.ones((features.shape[0], 1), dtype=np.float32) if use_hetero else np.zeros((features.shape[0], 1), dtype=np.float32)
    mechanism_gate = np.ones((features.shape[0], 1), dtype=np.float32) if use_mechanism else np.zeros((features.shape[0], 1), dtype=np.float32)
    chain_gate = _chain_gate_np(chain) if use_chain else np.zeros((features.shape[0], 1), dtype=np.float32)
    gated_hetero = hetero * hetero_gate
    gated_mechanism = mechanism * mechanism_gate
    gated_chain = chain * chain_gate
    zero_hetero = np.zeros_like(gated_hetero, dtype=np.float32)
    zero_mechanism = np.zeros_like(gated_mechanism, dtype=np.float32)
    zero_chain = np.zeros_like(gated_chain, dtype=np.float32)
    fused = np.concatenate([target * target_gate, homo * homo_gate, gated_hetero, gated_mechanism, gated_chain], axis=1).astype(np.float32)
    fused_without = np.concatenate(
        [
            target_without * target_gate,
            homo_without * homo_gate,
            hetero_without * hetero_gate if use_hetero else zero_hetero,
            mechanism_without * mechanism_gate if use_mechanism else zero_mechanism,
            zero_chain,
        ],
        axis=1,
    ).astype(np.float32)
    return fused, fused_without, {
        "target_gate": target_gate,
        "homo_gate": homo_gate,
        "hetero_gate": hetero_gate,
        "mechanism_gate": mechanism_gate,
        "chain_gate": chain_gate,
    }


def _split_hero_features_np(features: np.ndarray, feature_dims: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_dim = int(feature_dims["target_dim"])
    homo_dim = int(feature_dims["homo_dim"])
    hetero_dim = int(feature_dims["hetero_dim"])
    mechanism_dim = int(feature_dims["mechanism_dim"])
    chain_dim = int(feature_dims["chain_dim"])
    target = features[:, :target_dim]
    homo = features[:, target_dim : target_dim + homo_dim]
    hetero_start = target_dim + homo_dim
    mechanism_start = hetero_start + hetero_dim
    chain_start = mechanism_start + mechanism_dim
    hetero = features[:, hetero_start:mechanism_start]
    mechanism = features[:, mechanism_start:chain_start]
    chain = features[:, chain_start : chain_start + chain_dim]
    return target, homo, hetero, mechanism, chain


def _chain_gate_np(chain: np.ndarray) -> np.ndarray:
    if chain.size == 0:
        return np.zeros((chain.shape[0], 1), dtype=np.float32)
    has_chain = np.linalg.norm(chain, axis=1, keepdims=True) > 1e-8
    chain_quality = chain[:, -1:].astype(np.float32)
    gate = _sigmoid_np(-0.5 + 2.0 * chain_quality).reshape(-1, 1).astype(np.float32)
    return np.where(has_chain, gate, 0.0).astype(np.float32)


def _gate_diagnostics_for_indices(
    gates: dict[str, np.ndarray],
    indices: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    indices = np.asarray(indices, dtype=np.int64)
    diagnostics = {}
    for name, key in [
        ("avg_chain_gate", "chain_gate"),
        ("avg_homo_gate", "homo_gate"),
        ("avg_target_gate", "target_gate"),
    ]:
        values = gates.get(key)
        diagnostics[name] = float(np.mean(values[indices])) if values is not None and indices.size else 0.0
    diagnostics["avg_chain_gate_pos"] = 0.0
    diagnostics["avg_chain_gate_neg"] = 0.0
    chain_gate = gates.get("chain_gate")
    if labels is not None and chain_gate is not None and indices.size:
        selected_gate = chain_gate[indices]
        selected_labels = np.asarray(labels)[indices]
        pos_mask = selected_labels == 1.0
        neg_mask = selected_labels == 0.0
        if np.any(pos_mask):
            diagnostics["avg_chain_gate_pos"] = float(np.mean(selected_gate[pos_mask]))
        if np.any(neg_mask):
            diagnostics["avg_chain_gate_neg"] = float(np.mean(selected_gate[neg_mask]))
    return diagnostics


def _numpy_chain_quality_diagnostics(
    features: np.ndarray,
    feature_dims: dict[str, int] | None,
    labels: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    diagnostics = {
        "avg_chain_quality": 0.0,
        "avg_chain_quality_pos": 0.0,
        "avg_chain_quality_neg": 0.0,
    }
    indices = np.asarray(indices, dtype=np.int64)
    if not feature_dims or indices.size == 0:
        return diagnostics
    quality = _chain_quality_from_feature_matrix(features, feature_dims)
    selected_quality = quality[indices]
    selected_labels = np.asarray(labels)[indices]
    diagnostics["avg_chain_quality"] = float(np.mean(selected_quality)) if selected_quality.size else 0.0
    pos_mask = selected_labels == 1.0
    neg_mask = selected_labels == 0.0
    if np.any(pos_mask):
        diagnostics["avg_chain_quality_pos"] = float(np.mean(selected_quality[pos_mask]))
    if np.any(neg_mask):
        diagnostics["avg_chain_quality_neg"] = float(np.mean(selected_quality[neg_mask]))
    return diagnostics


def _chain_quality_from_feature_matrix(features: np.ndarray, feature_dims: dict[str, int]) -> np.ndarray:
    _target, _homo, _hetero, _mechanism, chain = _split_hero_features_np(features, feature_dims)
    if chain.size == 0 or chain.shape[1] == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    return chain[:, -1].astype(np.float32)


def _numpy_repr_diagnostics(
    x_scaled: np.ndarray,
    feature_dims: dict[str, int] | None,
    indices: np.ndarray,
) -> dict[str, float]:
    diagnostics = {
        "target_repr_norm": 0.0,
        "homo_repr_norm": 0.0,
        "hetero_repr_norm": 0.0,
        "mechanism_repr_norm": 0.0,
        "chain_repr_norm": 0.0,
        "final_repr_norm": 0.0,
    }
    indices = np.asarray(indices, dtype=np.int64)
    if not feature_dims or indices.size == 0:
        if indices.size:
            diagnostics["final_repr_norm"] = float(np.linalg.norm(x_scaled[indices], axis=1).mean())
        return diagnostics
    slices = _hero_branch_slices(feature_dims)
    selected = x_scaled[indices]
    diagnostics["target_repr_norm"] = _mean_row_norm(selected[:, slices["target"]])
    diagnostics["homo_repr_norm"] = _mean_row_norm(selected[:, slices["homo"]])
    diagnostics["hetero_repr_norm"] = _mean_row_norm(selected[:, slices["hetero"]])
    diagnostics["mechanism_repr_norm"] = _mean_row_norm(selected[:, slices["mechanism"]])
    diagnostics["chain_repr_norm"] = _mean_row_norm(selected[:, slices["chain"]])
    diagnostics["final_repr_norm"] = _mean_row_norm(selected)
    return diagnostics


def _numpy_branch_delta_diagnostics(
    x_scaled: np.ndarray,
    weights: np.ndarray,
    bias: float,
    use_chain: bool,
    feature_dims: dict[str, int] | None,
    indices: np.ndarray,
) -> dict[str, float]:
    diagnostics = {
        "delta_zero_hetero": 0.0,
        "delta_zero_chain": 0.0,
        "delta_zero_mechanism": 0.0,
    }
    indices = np.asarray(indices, dtype=np.int64)
    if not feature_dims or indices.size == 0:
        return diagnostics
    prob_full = _numpy_predict_from_scaled(x_scaled, weights, bias, use_chain=use_chain, feature_dims=feature_dims)[indices]
    for branch_name, metric_name in [
        ("hetero", "delta_zero_hetero"),
        ("chain", "delta_zero_chain"),
        ("mechanism", "delta_zero_mechanism"),
    ]:
        x_zeroed = _zero_hero_branch_scaled(x_scaled, feature_dims, branch_name)
        prob_zeroed = _numpy_predict_from_scaled(x_zeroed, weights, bias, use_chain=use_chain, feature_dims=feature_dims)[indices]
        diagnostics[metric_name] = float(np.mean(np.abs(prob_full - prob_zeroed))) if prob_full.size else 0.0
    return diagnostics


def _numpy_predict_from_scaled(
    x_scaled: np.ndarray,
    weights: np.ndarray,
    bias: float,
    use_chain: bool,
    feature_dims: dict[str, int] | None,
) -> np.ndarray:
    if feature_dims and use_chain:
        x_without_chain = _zero_hero_branch_scaled(x_scaled, feature_dims, "chain")
        return _sigmoid_np(_numpy_full_logits(x_scaled, x_without_chain, weights, bias, use_chain=True))
    return _sigmoid_np(_numpy_no_chain_logits(x_scaled, weights, bias))


def _zero_hero_branch_scaled(x_scaled: np.ndarray, feature_dims: dict[str, int], branch_name: str) -> np.ndarray:
    out = np.array(x_scaled, copy=True)
    out[:, _hero_branch_slices(feature_dims)[branch_name]] = 0.0
    return out


def _hero_branch_slices(feature_dims: dict[str, int]) -> dict[str, slice]:
    target_dim = int(feature_dims["target_dim"])
    homo_dim = int(feature_dims["homo_dim"])
    hetero_dim = int(feature_dims["hetero_dim"])
    mechanism_dim = int(feature_dims["mechanism_dim"])
    chain_dim = int(feature_dims["chain_dim"])
    target_start = 0
    homo_start = target_start + target_dim
    hetero_start = homo_start + homo_dim
    mechanism_start = hetero_start + hetero_dim
    chain_start = mechanism_start + mechanism_dim
    return {
        "target": slice(target_start, homo_start),
        "homo": slice(homo_start, hetero_start),
        "hetero": slice(hetero_start, mechanism_start),
        "mechanism": slice(mechanism_start, chain_start),
        "chain": slice(chain_start, chain_start + chain_dim),
    }


def _mean_row_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=1).mean())


def _numpy_chain_loss_grad(
    weights: np.ndarray,
    bias: float,
    x_full: np.ndarray,
    x_without: np.ndarray,
    raw_features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    use_chain: bool,
    feature_dims: dict[str, int] | None,
    min_chain_quality: float,
    positive_constraint: bool,
) -> tuple[float, np.ndarray, float]:
    if not feature_dims:
        return 0.0, np.zeros_like(weights, dtype=np.float32), 0.0
    quality = _chain_quality_from_feature_matrix(raw_features, feature_dims)
    target_label = 1.0 if positive_constraint else 0.0
    selected_idx = train_idx[(labels[train_idx] == target_label) & (quality[train_idx] >= float(min_chain_quality))]
    if selected_idx.size == 0:
        return 0.0, np.zeros_like(weights, dtype=np.float32), 0.0
    full_logits, full_grad_features = _numpy_full_logits_and_grad_features(
        x_full=x_full[selected_idx],
        x_without=x_without[selected_idx],
        weights=weights,
        bias=bias,
        use_chain=use_chain,
    )
    no_chain_logits = _numpy_no_chain_logits(x_without[selected_idx], weights, bias)
    full_prob = _sigmoid_np(full_logits)
    no_chain_prob = _sigmoid_np(no_chain_logits)
    diff = no_chain_prob - full_prob if positive_constraint else full_prob - no_chain_prob
    active = diff > 0.0
    loss = float(np.mean(np.maximum(diff, 0.0)))
    if not np.any(active):
        return loss, np.zeros_like(weights, dtype=np.float32), 0.0
    denom = max(float(selected_idx.size), 1.0)
    if positive_constraint:
        grad_full_logits = np.where(active, -full_prob * (1.0 - full_prob), 0.0) / denom
        grad_no_chain_logits = np.where(active, no_chain_prob * (1.0 - no_chain_prob), 0.0) / denom
    else:
        grad_full_logits = np.where(active, full_prob * (1.0 - full_prob), 0.0) / denom
        grad_no_chain_logits = np.where(active, -no_chain_prob * (1.0 - no_chain_prob), 0.0) / denom
    grad_w = full_grad_features.T @ grad_full_logits + x_without[selected_idx].T @ grad_no_chain_logits
    grad_b = float(np.sum(grad_full_logits + grad_no_chain_logits))
    return loss, grad_w.astype(np.float32), grad_b


def _numpy_full_logits_and_grad_features(
    x_full: np.ndarray,
    x_without: np.ndarray,
    weights: np.ndarray,
    bias: float,
    use_chain: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_chain:
        return _numpy_no_chain_logits(x_full, weights, bias), x_full
    no_chain_logits = _numpy_no_chain_logits(x_without, weights, bias)
    delta_features = x_full - x_without
    raw_delta = delta_features @ weights
    has_chain = (np.linalg.norm(delta_features, axis=1) > 1e-8).astype(np.float32)
    shifted = raw_delta - 2.0
    chain_delta = 0.2 * _softplus_np(shifted) * has_chain
    chain_grad = (0.2 * _sigmoid_np(shifted) * has_chain).reshape(-1, 1)
    grad_features = x_without + chain_grad * delta_features
    return (no_chain_logits + chain_delta).astype(np.float32), grad_features.astype(np.float32)


def _numpy_full_logits(
    x_full: np.ndarray,
    x_without: np.ndarray,
    weights: np.ndarray,
    bias: float,
    use_chain: bool,
) -> np.ndarray:
    logits, _grad_features = _numpy_full_logits_and_grad_features(
        x_full=x_full,
        x_without=x_without,
        weights=weights,
        bias=bias,
        use_chain=use_chain,
    )
    return logits


def _numpy_no_chain_logits(x_without: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return (x_without @ weights + bias).astype(np.float32)


def _softplus_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def _sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _graphsage_features(features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    agg = np.zeros_like(features, dtype=np.float32)
    degree = np.zeros((features.shape[0], 1), dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        np.add.at(agg, src, features[dst])
        np.add.at(degree, src, 1.0)
    degree = np.maximum(degree, 1.0)
    agg = agg / degree
    return np.concatenate([features, agg], axis=1).astype(np.float32)


def _fit_classifier(features: np.ndarray, labels: np.ndarray, seed: int):
    if len(np.unique(labels)) < 2:
        classifier = DummyClassifier(strategy="prior")
    else:
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=seed,
            solver="liblinear",
        )
    return classifier.fit(features, labels)


def _positive_scores(classifier, features: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(features)
    if probabilities.shape[1] == 1:
        label = int(classifier.classes_[0])
        return np.ones(features.shape[0], dtype=np.float32) if label == 1 else np.zeros(features.shape[0], dtype=np.float32)
    class_to_col = {int(label): col for col, label in enumerate(classifier.classes_)}
    return probabilities[:, class_to_col.get(1, 0)].astype(np.float32)


def _valid_label_indices(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    return indices[labels[indices] >= 0]


def _load_eval_target_indices(path: str | Path, graph: ProcessedGraphData, test_idx: np.ndarray) -> np.ndarray:
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing eval target file: {target_path}")
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        target_ids = payload
    elif isinstance(payload, dict):
        target_ids = payload.get("target_ids", payload.get("targets", []))
    else:
        raise ValueError("eval_target_file must be a JSON list or an object with target_ids.")
    target_set = {str(value) for value in target_ids}
    test_set = {int(value) for value in np.asarray(test_idx, dtype=np.int64).tolist()}
    indices = [
        int(graph.node_id_to_idx[target_id])
        for target_id in target_set
        if target_id in graph.node_id_to_idx and int(graph.node_id_to_idx[target_id]) in test_set
    ]
    if not indices:
        return np.array([], dtype=np.int64)
    ordered = [int(idx) for idx in test_idx.tolist() if int(idx) in set(indices)]
    return np.asarray(ordered, dtype=np.int64)


def _subset_positions(full_indices: np.ndarray, subset_indices: np.ndarray) -> np.ndarray:
    subset = {int(value) for value in np.asarray(subset_indices, dtype=np.int64).tolist()}
    return np.asarray([pos for pos, value in enumerate(full_indices.tolist()) if int(value) in subset], dtype=np.int64)


def _experiment_paths(
    output_root: str | Path,
    dataset: str,
    model_name: str,
    seed: int,
    experiment_tag: str | None = None,
) -> dict[str, Path]:
    output_root = Path(output_root)
    if experiment_tag:
        tag = str(experiment_tag)
        result_dir = output_root / "results_llm_comparison" / dataset / tag / model_name / f"seed_{seed}"
        checkpoint_dir = output_root / "checkpoints_llm_comparison" / dataset / tag / model_name / f"seed_{seed}"
        log_dir = output_root / "logs_llm_comparison" / dataset / tag / model_name
    else:
        result_dir = output_root / "results" / dataset / model_name / f"seed_{seed}"
        checkpoint_dir = output_root / "checkpoints" / dataset / model_name / f"seed_{seed}"
        log_dir = output_root / "logs" / dataset / model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "metrics": result_dir / "metrics.json",
        "checkpoint": checkpoint_dir / "best.pt",
        "log": log_dir / f"seed_{seed}.log",
    }


def _write_checkpoint(path: Path, checkpoint: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**checkpoint, "metrics": metrics}
    if torch is not None and "model_state_dict" in checkpoint:
        torch.save(payload, path)
        return
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _make_logger(log_path: Path, dataset: str, model_name: str, seed: int) -> logging.Logger:
    logger = logging.getLogger(f"hero_gnn.{dataset}.{model_name}.{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
