from __future__ import annotations

import logging
import pickle
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.data import schema
from src.data.loader import ProcessedGraphData, load_processed_data
from src.graph.evidence_chain import build_evidence_chains
from src.graph.heterophily_scoring import mechanism_id, risk_heterophily_score
from src.graph.neighbor_retrieval import (
    candidates_to_edge_index,
    filter_rule_hetero_edges,
    filter_topk_semantic_edges,
    retrieve_hetero_candidates,
)
from src.llm.label_cache import LabelCache, cache_key
from src.llm.mock_labeler import label_candidate_mechanism
from src.training.evaluator import binary_classification_metrics
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
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name={model_name}. Expected one of {MODEL_NAMES}.")

    set_seed(seed)
    data_dir = Path(data_dir or f"data/processed/{dataset}")
    graph = load_processed_data(data_dir)
    paths = _experiment_paths(output_root, dataset, model_name, seed)
    logger = _make_logger(paths["log"], dataset, model_name, seed)
    logger.info("Starting experiment dataset=%s model=%s seed=%s", dataset, model_name, seed)
    logger.info("Training knobs epochs=%s lr=%s hidden_dim=%s top_k=%s", epochs, lr, hidden_dim, top_k)

    train_idx = _valid_label_indices(graph.split.get("train", np.array([], dtype=np.int64)), graph.labels)
    val_idx = _valid_label_indices(graph.split.get("val", np.array([], dtype=np.int64)), graph.labels)
    test_idx = _valid_label_indices(graph.split.get("test", np.array([], dtype=np.int64)), graph.labels)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Processed data must contain labeled train and test review nodes.")

    if model_name in HERO_MODEL_NAMES:
        features, features_without_chains, hero_artifacts = _prepare_hero_features(
            graph=graph,
            data_dir=data_dir,
            model_name=model_name,
            target_indices=np.concatenate([train_idx, val_idx, test_idx]),
            top_k=top_k,
        )
        classifier = _fit_classifier(features[train_idx], graph.labels[train_idx], seed=seed)
        scores = _positive_scores(classifier, features[test_idx])
        scores_without_chains = _positive_scores(classifier, features_without_chains[test_idx])
        checkpoint = {"classifier": classifier, "hero_artifacts": hero_artifacts}
    elif torch is not None:
        scores, checkpoint = _fit_torch_model(
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
        )
    else:
        features = _prepare_features(graph, model_name, top_k=top_k)
        classifier = _fit_classifier(features[train_idx], graph.labels[train_idx], seed=seed)
        scores = _positive_scores(classifier, features[test_idx])
        checkpoint = {"classifier": classifier}

    metrics = binary_classification_metrics(graph.labels[test_idx], scores, k=100)
    if model_name in HERO_MODEL_NAMES:
        explanation_metrics = _write_hero_explanations(
            graph=graph,
            dataset=dataset,
            model_name=model_name,
            seed=seed,
            output_root=output_root,
            test_idx=test_idx,
            scores=scores,
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

    write_json(paths["metrics"], payload)
    _write_checkpoint(paths["checkpoint"], checkpoint, payload)
    logger.info("Finished experiment metrics=%s", payload)
    return payload


def write_placeholder_result(output_dir: str | Path, experiment_name: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{experiment_name}.json"
    path.write_text('{"status": "placeholder", "metric": null}\n', encoding="utf-8")
    return path


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
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    homo_edges = filter_topk_semantic_edges(graph.edge_index, graph.text_features, top_k=top_k)
    homo_agg = _neighbor_mean_features(graph.features, homo_edges)
    zero_chain = np.zeros((graph.features.shape[0], graph.features.shape[1] + len(schema.EVIDENCE_MECHANISMS) + 1), dtype=np.float32)

    if model_name == "hero_wo_hetero":
        features = np.concatenate([graph.features, homo_agg, zero_chain], axis=1).astype(np.float32)
        return features, features.copy(), {"chains_by_idx": {}, "labels_by_target": {}, "candidates_by_target": {}}

    candidates_by_target = retrieve_hetero_candidates(
        edge_index=graph.edge_index,
        edges=graph.edges,
        nodes=graph.nodes,
        node_id_to_idx=graph.node_id_to_idx,
        text_features=graph.text_features,
        numeric_features=graph.numeric_features,
        target_indices=target_indices,
        top_k=top_k,
    )
    if model_name == "hero_wo_chain":
        risk_edges = candidates_to_edge_index(candidates_by_target)
        risk_agg = _neighbor_mean_features(graph.features, risk_edges)
        features = np.concatenate([graph.features, homo_agg, risk_agg], axis=1).astype(np.float32)
        return features, features.copy(), {"chains_by_idx": {}, "labels_by_target": {}, "candidates_by_target": candidates_by_target}

    use_mechanism = model_name != "hero_wo_mechanism"
    labels_by_target = _label_candidates(data_dir, candidates_by_target, use_mechanism=use_mechanism)
    flat_labels = [label for labels in labels_by_target.values() for label in labels]
    chains_by_idx: dict[int, list[dict[str, Any]]] = {}
    for target_idx, labels in labels_by_target.items():
        target_id = labels[0]["target_id"] if labels else _idx_to_node_id(graph)[target_idx]
        chains_by_idx[target_idx] = build_evidence_chains(
            target_id=target_id,
            labels=labels,
            all_labels=flat_labels,
            top_k=3,
            include_two_hop=True,
        )
    chain_features = _chain_feature_matrix(graph, chains_by_idx, use_mechanism=use_mechanism)
    features = np.concatenate([graph.features, homo_agg, chain_features], axis=1).astype(np.float32)
    features_without_chains = np.concatenate([graph.features, homo_agg, zero_chain], axis=1).astype(np.float32)
    return features, features_without_chains, {
        "chains_by_idx": chains_by_idx,
        "labels_by_target": labels_by_target,
        "candidates_by_target": candidates_by_target,
    }


def _label_candidates(
    data_dir: Path,
    candidates_by_target: dict[int, list[Any]],
    use_mechanism: bool,
) -> dict[int, list[dict[str, Any]]]:
    cache = LabelCache(data_dir / "llm_labels.jsonl")
    labels_by_target: dict[int, list[dict[str, Any]]] = {}
    for target_idx, candidates in candidates_by_target.items():
        target_labels = []
        for candidate in candidates:
            key = cache_key(candidate.target_id, candidate.neighbor_id, candidate.metapath)
            label = cache.get(key)
            if label is None or "candidate" not in label:
                label = label_candidate_mechanism(candidate)
            if not use_mechanism:
                label = dict(label)
                label["mechanism"] = "irrelevant_heterophily"
                label["risk_relevance"] = int(candidate.candidate_score >= 0.55)
                label["confidence"] = float(candidate.candidate_score)
                label["rationale"] = "rule score only; mechanism labels disabled"
                label["candidate"] = getattr(candidate, "__dict__", label.get("candidate", {}))
            score, mechanism_logits = risk_heterophily_score(
                candidate,
                mechanism=label.get("mechanism"),
                risk_relevance_label=int(label.get("risk_relevance", 0)),
                use_mechanism=use_mechanism,
            )
            label = dict(label)
            label["risk_score"] = score
            label["mechanism_logits"] = mechanism_logits.tolist()
            label["candidate"] = getattr(candidate, "__dict__", label.get("candidate", {}))
            cache.set(key, label)
            target_labels.append(label)
        labels_by_target[target_idx] = sorted(target_labels, key=lambda item: item["risk_score"], reverse=True)
    cache.save()
    return labels_by_target


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
    necessities: list[float] = []
    chain_counts: list[int] = []
    evidence_recalls: list[float] = []
    with path.open("w", encoding="utf-8") as handle:
        for local_pos, node_idx in enumerate(test_idx[:50]):
            node_idx = int(node_idx)
            chains = chains_by_idx.get(node_idx, [])
            pred_prob = float(scores[local_pos])
            score_without = float(scores_without_chains[local_pos])
            necessity = max(pred_prob - score_without, 0.0)
            necessities.append(necessity)
            chain_counts.append(len(chains))
            evidence_recalls.append(_evidence_recall_for_node(idx_to_id[node_idx], chains, graph.evidence_gt))
            payload = {
                "target_id": idx_to_id[node_idx],
                "label": int(graph.labels[node_idx]),
                "pred_prob": pred_prob,
                "top_chains": [
                    {
                        "chain_nodes": chain["chain_nodes"],
                        "mechanism": chain["mechanism"],
                        "chain_score": float(chain["chain_score"]),
                        "rationale": chain["rationale"],
                    }
                    for chain in chains
                ],
                "score_without_chains": score_without,
                "necessity": necessity,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return {
        "evidence_recall_proxy": float(np.mean(evidence_recalls)) if evidence_recalls else 0.0,
        "evidence_necessity_score": float(np.mean(necessities)) if necessities else 0.0,
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
) -> tuple[np.ndarray, dict[str, Any]]:
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
    model = model_cls(input_dim=graph.features.shape[1], hidden_dim=hidden_dim, output_dim=2)
    x = torch.tensor(graph.features, dtype=torch.float32)
    labels = torch.tensor(graph.labels, dtype=torch.long)
    edge_index = torch.tensor(_edge_index_for_model(graph, model_name, top_k), dtype=torch.long)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx if val_idx.size else train_idx, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_score = -1.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(x) if model_name == "mlp" else model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(logits[train_tensor], labels[train_tensor])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x) if model_name == "mlp" else model(x, edge_index)
            val_scores = torch.softmax(logits[val_tensor], dim=1)[:, 1].cpu().numpy()
        val_metrics = binary_classification_metrics(graph.labels[val_tensor.numpy()], val_scores, k=100)
        if val_metrics["auprc"] >= best_score:
            best_score = val_metrics["auprc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x) if model_name == "mlp" else model(x, edge_index)
        scores = torch.softmax(logits[torch.tensor(test_idx, dtype=torch.long)], dim=1)[:, 1].cpu().numpy()
    return scores, {"model_state_dict": best_state, "model_name": model_name}


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
