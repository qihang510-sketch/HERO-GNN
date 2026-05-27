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
    candidates_to_edge_index,
    filter_rule_hetero_edges,
    filter_topk_semantic_edges,
    retrieve_hetero_candidates,
)
from src.llm.label_cache import LabelCache, cache_key
from src.llm.mock_labeler import MOCK_LABELER_VERSION, label_candidate_mechanism
from src.llm.risk_card import format_candidate_risk_card
from src.training.evaluator import binary_classification_metrics, prediction_probability_stats, split_label_stats, tune_threshold
from src.utils.io import write_json
from src.utils.seed import set_seed

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

BASELINE_MODEL_NAMES = ("mlp", "graphsage", "semsim_gnn", "rulehetero_gnn")
HERO_MODEL_NAMES = ("hero_gnn", "hero_wo_hetero", "hero_wo_mechanism", "hero_wo_chain")
MODEL_NAMES = (*BASELINE_MODEL_NAMES, *HERO_MODEL_NAMES)


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
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {MODEL_NAMES}.")

    set_seed(seed)
    data_dir = Path(data_dir or f"data/processed/{dataset}")
    graph = load_processed_data(data_dir)
    paths = _experiment_paths(output_root, dataset, model_name, seed)
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

    train_idx = _valid_label_indices(graph.split.get("train", np.array([], dtype=np.int64)), graph.labels)
    val_idx = _valid_label_indices(graph.split.get("val", np.array([], dtype=np.int64)), graph.labels)
    test_idx = _valid_label_indices(graph.split.get("test", np.array([], dtype=np.int64)), graph.labels)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Processed data must contain labeled train and test review nodes.")

    class_stats = _split_class_stats(
        train_labels=graph.labels[train_idx],
        val_labels=graph.labels[val_idx],
        test_labels=graph.labels[test_idx],
    )
    pos_weight = _pos_weight_from_counts(class_stats["train_num_pos"], class_stats["train_num_neg"])
    print(f"[TRAIN-INFO] dataset={dataset} model={model_name} seed={seed}")
    print(f"[TRAIN-INFO] train_pos={class_stats['train_num_pos']} train_neg={class_stats['train_num_neg']} pos_weight={pos_weight:.6f}")
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
        )
        stage_times["time_training_sec"] += time.perf_counter() - train_start
        checkpoint["hero_artifacts"] = hero_artifacts
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

    threshold_labels = graph.labels[val_idx] if val_idx.size else graph.labels[train_idx]
    best_threshold = tune_threshold(threshold_labels, val_scores)
    metrics = binary_classification_metrics(graph.labels[test_idx], test_scores, k=100, threshold=best_threshold)
    metrics.update(prediction_probability_stats(graph.labels[test_idx], test_scores))
    metrics.update(class_stats)
    metrics["pos_weight"] = float(pos_weight)
    metrics.update(
        split_label_stats(
            {
                "train": graph.labels[train_idx],
                "val": graph.labels[val_idx],
                "test": graph.labels[test_idx],
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
            test_idx=test_idx,
            scores=test_scores,
            scores_without_chains=scores_without_chains,
            hero_artifacts=hero_artifacts,
        )
        metrics.update(explanation_metrics)
        metrics["avg_selected_neighbors"] = _hero_avg_selected_neighbors(hero_artifacts)
    else:
        metrics["avg_selected_neighbors"] = _baseline_avg_selected_neighbors(graph, model_name, test_idx, top_k)
    payload = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        **metrics,
    }
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    print(f"[START] {model_name}")
    timings = {
        "time_retrieval_sec": 0.0,
        "time_mock_labeling_sec": 0.0,
        "time_evidence_chain_sec": 0.0,
        "time_training_sec": 0.0,
    }
    target_indices = _limit_target_indices(target_indices, max_target_nodes)
    homo_edges = filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=homophilic_topk)
    homo_agg = _neighbor_mean_features(graph.features, homo_edges)
    zero_chain = np.zeros((graph.features.shape[0], graph.features.shape[1] + len(schema.EVIDENCE_MECHANISMS) + 1), dtype=np.float32)

    if model_name == "hero_wo_hetero":
        features = np.concatenate([graph.features, homo_agg, zero_chain], axis=1).astype(np.float32)
        return features, features.copy(), {"chains_by_idx": {}, "labels_by_target": {}, "candidates_by_target": {}, "time": timings}

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
    if model_name == "hero_wo_chain":
        risk_edges = candidates_to_edge_index(candidates_by_target)
        risk_agg = _neighbor_mean_features(graph.features, risk_edges)
        features = np.concatenate([graph.features, homo_agg, risk_agg], axis=1).astype(np.float32)
        return features, features.copy(), {"chains_by_idx": {}, "labels_by_target": {}, "candidates_by_target": candidates_by_target, "time": timings}

    use_mechanism = model_name != "hero_wo_mechanism"
    print("[BUILD] mock LLM labels")
    labels_by_target, label_time = _label_candidates(data_dir, candidates_by_target, use_mechanism=use_mechanism)
    timings["time_mock_labeling_sec"] = label_time
    flat_labels = [label for labels in labels_by_target.values() for label in labels]
    print("[BUILD] evidence chains")
    chain_start = time.perf_counter()
    chains_by_idx = _load_or_build_evidence_chains(
        data_dir=data_dir,
        graph=graph,
        labels_by_target=labels_by_target,
        flat_labels=flat_labels,
        topk_chains=topk_chains,
        max_chain_length=max_chain_length,
        use_cache=use_mechanism,
    )
    timings["time_evidence_chain_sec"] = time.perf_counter() - chain_start
    chain_features = _chain_feature_matrix(graph, chains_by_idx, use_mechanism=use_mechanism)
    features = np.concatenate([graph.features, homo_agg, chain_features], axis=1).astype(np.float32)
    features_without_chains = np.concatenate([graph.features, homo_agg, zero_chain], axis=1).astype(np.float32)
    return features, features_without_chains, {
        "chains_by_idx": chains_by_idx,
        "labels_by_target": labels_by_target,
        "candidates_by_target": candidates_by_target,
        "time": timings,
    }


def _limit_target_indices(target_indices: np.ndarray, max_target_nodes: int | None) -> np.ndarray:
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if max_target_nodes is None or max_target_nodes <= 0 or target_indices.size <= max_target_nodes:
        return target_indices
    return target_indices[: int(max_target_nodes)]


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
    return sorted(chains, key=lambda item: item["chain_score"], reverse=True)[:top_k]


def _one_hop_chain_from_label(label: dict[str, Any]) -> dict[str, Any]:
    candidate = label.get("candidate", {})
    score = float(label.get("risk_score", label.get("confidence", 0.0)))
    return {
        "target_id": label["target_id"],
        "chain_nodes": [label["target_id"], label["neighbor_id"]],
        "chain_edges": [label["metapath"]],
        "mechanism": label["mechanism"],
        "risk_relevance": int(label.get("risk_relevance", 0)),
        "chain_score": score,
        "confidence": float(label.get("confidence", 0.0)),
        "rationale": label["rationale"],
        "neighbor_idx": candidate.get("neighbor_idx"),
    }


def _two_hop_chain_from_labels(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_score = float(first.get("risk_score", first.get("confidence", 0.0)))
    second_score = float(second.get("risk_score", second.get("confidence", 0.0)))
    return {
        "target_id": first["target_id"],
        "chain_nodes": [first["target_id"], first["neighbor_id"], second["neighbor_id"]],
        "chain_edges": [first["metapath"], second["metapath"]],
        "mechanism": first["mechanism"],
        "risk_relevance": int(first.get("risk_relevance", 0)),
        "chain_score": (first_score + second_score) / 2.0,
        "confidence": float(first.get("confidence", 0.0)),
        "rationale": f"{first['rationale']}; then {second['rationale']}",
        "neighbor_idx": first.get("candidate", {}).get("neighbor_idx"),
    }


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


def _label_candidates(
    data_dir: Path,
    candidates_by_target: dict[int, list[Any]],
    use_mechanism: bool,
) -> tuple[dict[int, list[dict[str, Any]]], float]:
    label_start = time.perf_counter()
    cache = LabelCache(data_dir / "llm_labels.jsonl")
    labels_by_target: dict[int, list[dict[str, Any]]] = {}
    if cache.path.exists():
        print("[CACHE] loading llm_labels.jsonl")
    pairs = [
        (target_idx, candidate)
        for target_idx, candidates in candidates_by_target.items()
        for candidate in candidates
    ]
    base_by_pair: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for target_idx, candidate in tqdm(pairs, desc="mock_labeler", unit="pair"):
        key = cache_key(candidate.target_id, candidate.neighbor_id, candidate.metapath)
        cached_label = cache.get(key)
        if cached_label is None or cached_label.get("labeler_version") != MOCK_LABELER_VERSION:
            base_label = label_candidate_mechanism(candidate)
            cache.set(key, base_label)
        else:
            base_label = dict(cached_label)
        base_by_pair[(target_idx, candidate.target_id, candidate.neighbor_id, candidate.metapath)] = base_label

    for target_idx, candidate in tqdm(pairs, desc="risk heterophily scoring", unit="pair"):
        label = dict(base_by_pair[(target_idx, candidate.target_id, candidate.neighbor_id, candidate.metapath)])
        label["candidate"] = asdict(candidate)
        label["risk_card"] = format_candidate_risk_card(candidate)
        if not use_mechanism:
            label = dict(label)
            label["mechanism"] = "irrelevant_heterophily"
            label["risk_relevance"] = int(candidate.candidate_score >= 0.55)
            label["confidence"] = float(candidate.candidate_score)
            label["rationale"] = "rule score only; mechanism labels disabled"
        score, mechanism_logits = risk_heterophily_score(
            candidate,
            mechanism=label.get("mechanism"),
            risk_relevance_label=int(label.get("risk_relevance", 0)),
            use_mechanism=use_mechanism,
        )
        label = dict(label)
        label["risk_score"] = score
        label["mechanism_logits"] = mechanism_logits.tolist()
        if use_mechanism:
            cache.set(cache_key(candidate.target_id, candidate.neighbor_id, candidate.metapath), label)
        labels_by_target.setdefault(target_idx, []).append(label)

    for target_idx in candidates_by_target:
        target_labels = labels_by_target.get(target_idx, [])
        labels_by_target[target_idx] = sorted(target_labels, key=lambda item: item["risk_score"], reverse=True)
    cache.save()
    return labels_by_target, time.perf_counter() - label_start


def _chain_feature_matrix(
    graph: ProcessedGraphData,
    chains_by_idx: dict[int, list[dict[str, Any]]],
    use_mechanism: bool,
) -> np.ndarray:
    dim = graph.features.shape[1]
    out_dim = dim + len(schema.EVIDENCE_MECHANISMS) + 1
    features = np.zeros((graph.features.shape[0], out_dim), dtype=np.float32)
    node_id_to_idx = graph.node_id_to_idx
    for target_idx, chains in chains_by_idx.items():
        if not chains:
            continue
        reps = []
        for chain in chains:
            indices = [node_id_to_idx[node_id] for node_id in chain["chain_nodes"] if node_id in node_id_to_idx]
            if indices:
                node_repr = graph.features[indices].mean(axis=0)
            else:
                node_repr = np.zeros(dim, dtype=np.float32)
            mechanism = np.zeros(len(schema.EVIDENCE_MECHANISMS), dtype=np.float32)
            if use_mechanism:
                mechanism[mechanism_id(chain.get("mechanism", "irrelevant_heterophily"))] = 1.0
            score = np.array([float(chain.get("chain_score", 0.0))], dtype=np.float32)
            reps.append(np.concatenate([node_repr, mechanism, score]))
        features[target_idx] = np.mean(reps, axis=0)
    return features


def _edge_index_for_model(graph: ProcessedGraphData, model_name: str, top_k: int) -> np.ndarray:
    if model_name in {"mlp", "graphsage"}:
        return graph.edge_index
    if model_name == "semsim_gnn":
        return filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=top_k)
    if model_name == "rulehetero_gnn":
        return filter_rule_hetero_edges(graph.edge_index, graph.text_features, graph.numeric_features, top_k=top_k)
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
    path = Path(output_root) / "explanations" / dataset / model_name / f"seed_{seed}" / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_id = _idx_to_node_id(graph)
    chains_by_idx = hero_artifacts.get("chains_by_idx", {})
    necessities_all: list[float] = []
    necessities_pos: list[float] = []
    necessities_neg: list[float] = []
    chain_counts: list[int] = []
    evidence_recalls: list[float] = []
    for local_pos, node_idx in enumerate(test_idx):
        node_idx = int(node_idx)
        chains = chains_by_idx.get(node_idx, [])
        score_drop = float(scores[local_pos] - scores_without_chains[local_pos])
        necessities_all.append(score_drop)
        if int(graph.labels[node_idx]) == 1:
            necessities_pos.append(score_drop)
        else:
            necessities_neg.append(score_drop)
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
    return {
        "evidence_recall_proxy": float(np.mean(evidence_recalls)) if evidence_recalls else 0.0,
        "evidence_necessity": necessity_all,
        "evidence_necessity_score": necessity_all,
        "avg_evidence_necessity_all": necessity_all,
        "avg_evidence_necessity_pos": necessity_pos,
        "avg_evidence_necessity_neg": necessity_neg,
        "evidence_necessity_gap": necessity_gap,
        "avg_num_chains": float(np.mean(chain_counts)) if chain_counts else 0.0,
    }


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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from src.models.graphsage import GraphSAGE
    from src.models.mlp import MLP
    from src.models.rulehetero_gnn import RuleHeteroGNN
    from src.models.semsim_gnn import SemSimGNN

    torch.manual_seed(seed)
    model_cls = {
        "mlp": MLP,
        "graphsage": GraphSAGE,
        "semsim_gnn": SemSimGNN,
        "rulehetero_gnn": RuleHeteroGNN,
    }[model_name]
    model = model_cls(input_dim=graph.features.shape[1], hidden_dim=hidden_dim, output_dim=1)
    x = torch.tensor(graph.features, dtype=torch.float32)
    labels = torch.tensor(graph.labels, dtype=torch.float32)
    edge_index = torch.tensor(_edge_index_for_model(graph, model_name, top_k), dtype=torch.long)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx if val_idx.size else train_idx, dtype=torch.long)
    test_tensor = torch.tensor(test_idx, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32))
    best_state = None
    best_score = -1.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = _model_logits(model, model_name, x, edge_index)
        loss = criterion(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = _model_logits(model, model_name, x, edge_index)
            val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        val_metrics = binary_classification_metrics(graph.labels[val_tensor.numpy()], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = _model_logits(model, model_name, x, edge_index)
        val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        test_scores = torch.sigmoid(logits[test_tensor]).cpu().numpy()
    return val_scores, test_scores, {"model_state_dict": best_state, "model_name": model_name, "pos_weight": float(pos_weight)}


def _model_logits(model, model_name: str, x, edge_index):
    logits = model(x) if model_name == "mlp" else model(x, edge_index)
    return logits.squeeze(-1)


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
        )

    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    x_without = torch.tensor(features_without_chains, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx if val_idx.size else train_idx, dtype=torch.long)
    test_tensor = torch.tensor(test_idx, dtype=torch.long)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32))
    best_state = None
    best_score = -1.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = criterion(logits[train_tensor], y[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        val_metrics = binary_classification_metrics(labels[val_tensor.numpy()], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        logits_without = model(x_without).squeeze(-1)
        val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
        test_scores = torch.sigmoid(logits[test_tensor]).cpu().numpy()
        scores_without_chains = torch.sigmoid(logits_without[test_tensor]).cpu().numpy()
    return val_scores, test_scores, scores_without_chains, {
        "model_state_dict": best_state,
        "model_name": "hero_feature_mlp",
        "pos_weight": float(pos_weight),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float32)
    x_without = np.asarray(features_without_chains, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx_eval = np.asarray(val_idx if val_idx.size else train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean) / std
    x_without_scaled = (x_without - mean) / std
    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = np.float32(0.0)
    train_y = y[train_idx]
    sample_weights = np.where(train_y == 1.0, float(pos_weight), 1.0).astype(np.float32)
    step = float(lr) if lr > 0 else 0.001
    # This is the same weighted logistic objective as BCEWithLogitsLoss(pos_weight);
    # it keeps the repo runnable in environments where torch is unavailable.
    for _epoch in range(max(1, epochs)):
        logits = x_scaled[train_idx] @ weights + bias
        probs = _sigmoid_np(logits)
        errors = (probs - train_y) * sample_weights
        denom = max(float(train_idx.size), 1.0)
        grad_w = (x_scaled[train_idx].T @ errors) / denom + 1e-4 * weights
        grad_b = np.sum(errors) / denom
        weights -= step * grad_w.astype(np.float32)
        bias = np.float32(bias - step * grad_b)

    scores = _sigmoid_np(x_scaled @ weights + bias).astype(np.float32)
    scores_without = _sigmoid_np(x_without_scaled @ weights + bias).astype(np.float32)
    return (
        scores[val_idx_eval],
        scores[test_idx],
        scores_without[test_idx],
        {
            "model_name": "numpy_weighted_logistic",
            "weights": weights,
            "bias": float(bias),
            "pos_weight": float(pos_weight),
        },
    )


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


def _experiment_paths(output_root: str | Path, dataset: str, model_name: str, seed: int) -> dict[str, Path]:
    output_root = Path(output_root)
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
