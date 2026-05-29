from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_processed_data  # noqa: E402
from src.graph.neighbor_retrieval import retrieve_hetero_candidates  # noqa: E402
from src.llm.risk_card import format_candidate_risk_card, strict_risk_card  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

TEXT_RICH_DATASETS = {"yelp_academic", "amazon_video"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-neighbor risk cards for small-scale LLM mechanism annotation.")
    parser.add_argument("--dataset", required=True, choices=sorted(TEXT_RICH_DATASETS), help="Text-rich dataset name.")
    parser.add_argument("--data_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--max_cards", type=int, default=2000, help="Maximum risk cards to write.")
    parser.add_argument("--out_file", required=True, help="Output JSONL risk-card path.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--max_candidates_per_node", type=int, default=20, help="Candidate cap per sampled target.")
    parser.add_argument("--max_target_nodes", type=int, default=None, help="Optional target node cap before candidate retrieval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = build_risk_cards(
        dataset=args.dataset,
        data_dir=Path(args.data_dir or f"data/processed/{args.dataset}"),
        max_cards=args.max_cards,
        seed=args.seed,
        max_candidates_per_node=args.max_candidates_per_node,
        max_target_nodes=args.max_target_nodes,
    )
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, sort_keys=True) + "\n")
    print(f"Wrote {len(cards)} risk cards to {out_file}")


def build_risk_cards(
    dataset: str,
    data_dir: Path,
    max_cards: int,
    seed: int = 0,
    max_candidates_per_node: int = 20,
    max_target_nodes: int | None = None,
) -> list[dict[str, Any]]:
    if dataset not in TEXT_RICH_DATASETS:
        raise ValueError(f"Real LLM risk cards are only supported for text-rich datasets: {sorted(TEXT_RICH_DATASETS)}")
    set_seed(seed)
    graph = load_processed_data(data_dir)
    node_lookup = _node_lookup(graph.nodes)
    rating_lookup = _rating_lookup(data_dir, graph.nodes)
    target_indices = _balanced_target_indices(graph, seed=seed, max_cards=max_cards, max_candidates_per_node=max_candidates_per_node)
    if max_target_nodes is not None:
        target_indices = target_indices[: max(0, int(max_target_nodes))]
    candidates_by_target = _load_cached_candidates(data_dir)
    if candidates_by_target is None:
        candidates_by_target = retrieve_hetero_candidates(
            edge_index=graph.edge_index,
            edges=graph.edges,
            nodes=graph.nodes,
            node_id_to_idx=graph.node_id_to_idx,
            text_features=graph.text_features,
            numeric_features=graph.numeric_features,
            target_indices=target_indices,
            max_candidates_per_node=max_candidates_per_node,
        )
    candidates_by_target = _filter_targets(candidates_by_target, target_indices)
    if not _has_candidates(candidates_by_target):
        candidates_by_target = retrieve_hetero_candidates(
            edge_index=graph.edge_index,
            edges=graph.edges,
            nodes=graph.nodes,
            node_id_to_idx=graph.node_id_to_idx,
            text_features=graph.text_features,
            numeric_features=graph.numeric_features,
            target_indices=target_indices,
            max_candidates_per_node=max_candidates_per_node,
        )
    candidates = _balanced_candidates(candidates_by_target, graph.labels, seed=seed)
    cards: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in tqdm(candidates, desc="risk_cards", unit="card"):
        pair = (str(candidate.target_id), str(candidate.neighbor_id))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        card = format_candidate_risk_card(
            candidate,
            dataset=dataset,
            node_lookup=node_lookup,
            rating_lookup=rating_lookup,
        )
        cards.append(strict_risk_card(card))
        if len(cards) >= max(0, int(max_cards)):
            break
    return cards


def _load_cached_candidates(data_dir: Path) -> dict[int, list[Any]] | None:
    path = data_dir / "hetero_candidates.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    candidates = payload.get("candidates_by_target", payload) if isinstance(payload, dict) else payload
    if not isinstance(candidates, dict):
        return None
    print("[CACHE] loading hetero_candidates.pkl")
    return {int(target): list(values) for target, values in candidates.items()}


def _balanced_target_indices(graph: Any, seed: int, max_cards: int, max_candidates_per_node: int) -> np.ndarray:
    train_val = []
    for split_name in ("train", "val"):
        train_val.extend(graph.split.get(split_name, np.array([], dtype=np.int64)).tolist())
    indices = np.asarray(sorted(set(int(value) for value in train_val)), dtype=np.int64)
    if indices.size == 0:
        indices = np.asarray(sorted(set(int(value) for value in graph.node_id_to_idx.values())), dtype=np.int64)
    labeled = indices[graph.labels[indices] >= 0] if indices.size else indices
    positives = labeled[graph.labels[labeled] == 1] if labeled.size else np.array([], dtype=np.int64)
    negatives = labeled[graph.labels[labeled] == 0] if labeled.size else np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    target_limit = max(1, int(np.ceil(max_cards / max(max_candidates_per_node, 1))) * 3)
    if positives.size and negatives.size:
        half = max(1, target_limit // 2)
        selected = np.concatenate([positives[:half], negatives[: target_limit - half]])
    else:
        selected = labeled.copy()
        rng.shuffle(selected)
        selected = selected[:target_limit]
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _balanced_candidates(candidates_by_target: dict[int, list[Any]], labels: np.ndarray, seed: int) -> list[Any]:
    positives: list[Any] = []
    negatives: list[Any] = []
    unlabeled: list[Any] = []
    for target, candidates in candidates_by_target.items():
        ordered = sorted(candidates, key=lambda item: float(getattr(item, "candidate_score", 0.0)), reverse=True)
        label = int(labels[int(target)]) if 0 <= int(target) < labels.shape[0] else -1
        if label == 1:
            positives.extend(ordered)
        elif label == 0:
            negatives.extend(ordered)
        else:
            unlabeled.extend(ordered)
    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    merged: list[Any] = []
    for left, right in zip(positives, negatives):
        merged.extend([left, right])
    merged.extend(positives[len(negatives) :])
    merged.extend(negatives[len(positives) :])
    merged.extend(unlabeled)
    return merged


def _filter_targets(candidates_by_target: dict[int, list[Any]], target_indices: np.ndarray) -> dict[int, list[Any]]:
    target_set = {int(index) for index in target_indices}
    return {int(target): candidates for target, candidates in candidates_by_target.items() if int(target) in target_set}


def _has_candidates(candidates_by_target: dict[int, list[Any]]) -> bool:
    return any(bool(candidates) for candidates in candidates_by_target.values())


def _node_lookup(nodes) -> dict[str, dict[str, Any]]:
    return {str(row["node_id"]): dict(row) for _, row in nodes.iterrows()}


def _rating_lookup(data_dir: Path, nodes) -> dict[str, float]:
    feature_path = data_dir / "features.npz"
    if not feature_path.exists():
        return {}
    payload = np.load(feature_path, allow_pickle=True)
    numeric_columns = [str(value) for value in payload["numeric_columns"].tolist()] if "numeric_columns" in payload.files else []
    rating_names = ("stars", "overall", "rating")
    rating_index = next((idx for idx, name in enumerate(numeric_columns) if name in rating_names), None)
    if rating_index is None:
        return {}
    column = f"feat_{rating_index}"
    if column not in nodes.columns:
        return {}
    return {str(row["node_id"]): _safe_float(row[column]) for _, row in nodes[["node_id", column]].iterrows()}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
