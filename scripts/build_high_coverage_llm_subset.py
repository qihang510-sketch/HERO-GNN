from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_risk_cards import TEXT_RICH_DATASETS, _node_lookup, _rating_lookup  # noqa: E402
from src.data.loader import load_processed_data  # noqa: E402
from src.graph.neighbor_retrieval import retrieve_hetero_candidates  # noqa: E402
from src.llm.risk_card import format_candidate_risk_card, strict_risk_card  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a high-coverage real-LLM target subset and dense risk cards.")
    parser.add_argument("--dataset", required=True, choices=sorted(TEXT_RICH_DATASETS), help="Text-rich dataset name.")
    parser.add_argument("--data_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--num_target_nodes", type=int, default=500, help="Number of evaluation target nodes.")
    parser.add_argument("--heterophilic_topk", type=int, default=10, help="Top-k heterophilic candidates per target.")
    parser.add_argument("--out_target_file", required=True, help="Output JSON target-id list.")
    parser.add_argument("--out_risk_card_file", required=True, help="Output JSONL risk-card file.")
    parser.add_argument("--seed", type=int, default=0, help="Subset sampling seed.")
    parser.add_argument("--report_dir", default="outputs/summary_llm", help="Directory for subset report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_high_coverage_subset(
        dataset=args.dataset,
        data_dir=Path(args.data_dir or f"data/processed/{args.dataset}"),
        num_target_nodes=args.num_target_nodes,
        heterophilic_topk=args.heterophilic_topk,
        seed=args.seed,
    )
    target_path = Path(args.out_target_file)
    card_path = Path(args.out_risk_card_file)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        target_path,
        {
            "dataset": args.dataset,
            "split": result["split"],
            "num_target_nodes": len(result["target_ids"]),
            "target_ids": result["target_ids"],
        },
    )
    with card_path.open("w", encoding="utf-8") as handle:
        for card in result["risk_cards"]:
            handle.write(json.dumps(card, sort_keys=True) + "\n")
    report = {
        "dataset": args.dataset,
        "split": result["split"],
        "num_requested_target_nodes": int(args.num_target_nodes),
        "num_selected_target_nodes": len(result["target_ids"]),
        "heterophilic_topk": int(args.heterophilic_topk),
        "num_risk_cards": len(result["risk_cards"]),
        "avg_cards_per_target": float(len(result["risk_cards"]) / max(len(result["target_ids"]), 1)),
        "num_positive_targets": int(result["num_positive_targets"]),
        "num_negative_targets": int(result["num_negative_targets"]),
        "out_target_file": str(target_path),
        "out_risk_card_file": str(card_path),
    }
    report_path = Path(args.report_dir) / f"highcov_subset_report_{args.dataset}.json"
    write_json(report_path, report)
    print(f"Wrote {len(result['target_ids'])} target ids to {target_path}")
    print(f"Wrote {len(result['risk_cards'])} risk cards to {card_path}")
    print(f"Wrote high-coverage subset report to {report_path}")


def build_high_coverage_subset(
    dataset: str,
    data_dir: Path,
    num_target_nodes: int,
    heterophilic_topk: int,
    seed: int = 0,
) -> dict[str, Any]:
    if dataset not in TEXT_RICH_DATASETS:
        raise ValueError(f"High-coverage LLM subset only supports {sorted(TEXT_RICH_DATASETS)}")
    set_seed(seed)
    graph = load_processed_data(data_dir)
    split_name, target_pool = _preferred_eval_pool(graph)
    candidate_cap = max(int(heterophilic_topk), 20)
    target_pool = _balanced_pool(target_pool, graph.labels, seed=seed, limit=max(int(num_target_nodes) * 4, int(num_target_nodes)))
    candidates_by_target = _load_cached_candidates(data_dir)
    candidates_by_target = _ensure_candidates(
        graph=graph,
        cached=candidates_by_target,
        target_indices=target_pool,
        max_candidates_per_node=candidate_cap,
    )
    selected_targets = _select_targets_with_candidates(
        target_pool=target_pool,
        labels=graph.labels,
        candidates_by_target=candidates_by_target,
        num_target_nodes=int(num_target_nodes),
        seed=seed,
    )
    node_lookup = _node_lookup(graph.nodes)
    rating_lookup = _rating_lookup(data_dir, graph.nodes)
    idx_to_node_id = {idx: node_id for node_id, idx in graph.node_id_to_idx.items()}
    risk_cards: list[dict[str, Any]] = []
    target_ids: list[str] = []
    for target_idx in selected_targets:
        target_id = idx_to_node_id.get(int(target_idx))
        if target_id is None:
            continue
        target_ids.append(str(target_id))
        candidates = _unique_pair_candidates(sorted(
            candidates_by_target.get(int(target_idx), []),
            key=lambda item: float(getattr(item, "candidate_score", 0.0)),
            reverse=True,
        ))[: max(0, int(heterophilic_topk))]
        for candidate in candidates:
            card = format_candidate_risk_card(
                candidate,
                dataset=dataset,
                node_lookup=node_lookup,
                rating_lookup=rating_lookup,
            )
            risk_cards.append(strict_risk_card(card))
    selected_labels = graph.labels[np.asarray(selected_targets, dtype=np.int64)] if selected_targets else np.array([], dtype=np.int64)
    return {
        "split": split_name,
        "target_ids": target_ids,
        "target_indices": [int(value) for value in selected_targets],
        "risk_cards": risk_cards,
        "num_positive_targets": int(np.sum(selected_labels == 1)),
        "num_negative_targets": int(np.sum(selected_labels == 0)),
    }


def _unique_pair_candidates(candidates: list[Any]) -> list[Any]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for candidate in candidates:
        key = (str(candidate.target_id), str(candidate.neighbor_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _preferred_eval_pool(graph: Any) -> tuple[str, np.ndarray]:
    for split_name in ("test", "eval", "val"):
        values = graph.split.get(split_name, np.array([], dtype=np.int64))
        values = np.asarray(values, dtype=np.int64)
        values = values[graph.labels[values] >= 0] if values.size else values
        if values.size:
            return split_name, values
    values = np.asarray(sorted(set(int(value) for value in graph.node_id_to_idx.values())), dtype=np.int64)
    values = values[graph.labels[values] >= 0] if values.size else values
    return "all_labeled", values


def _balanced_pool(indices: np.ndarray, labels: np.ndarray, seed: int, limit: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    positives = indices[labels[indices] == 1]
    negatives = indices[labels[indices] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if positives.size and negatives.size:
        half = max(1, int(limit) // 2)
        selected = np.concatenate([positives[:half], negatives[: max(0, int(limit) - half)]])
    else:
        selected = indices.copy()
        rng.shuffle(selected)
        selected = selected[: int(limit)]
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _select_targets_with_candidates(
    target_pool: np.ndarray,
    labels: np.ndarray,
    candidates_by_target: dict[int, list[Any]],
    num_target_nodes: int,
    seed: int,
) -> list[int]:
    with_candidates = [int(idx) for idx in target_pool if candidates_by_target.get(int(idx))]
    positives = [idx for idx in with_candidates if int(labels[idx]) == 1]
    negatives = [idx for idx in with_candidates if int(labels[idx]) == 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected: list[int] = []
    if positives and negatives:
        target_pos = min(len(positives), max(1, num_target_nodes // 2))
        target_neg = min(len(negatives), max(0, num_target_nodes - target_pos))
        selected.extend(positives[:target_pos])
        selected.extend(negatives[:target_neg])
        if len(selected) < num_target_nodes:
            leftovers = positives[target_pos:] + negatives[target_neg:]
            rng.shuffle(leftovers)
            selected.extend(leftovers[: num_target_nodes - len(selected)])
    else:
        selected = with_candidates[:num_target_nodes]
    rng.shuffle(selected)
    return selected[:num_target_nodes]


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


def _ensure_candidates(
    graph: Any,
    cached: dict[int, list[Any]] | None,
    target_indices: np.ndarray,
    max_candidates_per_node: int,
) -> dict[int, list[Any]]:
    target_set = {int(value) for value in target_indices}
    if cached is not None:
        filtered = {target: values for target, values in cached.items() if int(target) in target_set}
        if sum(1 for values in filtered.values() if values) >= max(1, len(target_set) // 2):
            return filtered
    print("[BUILD] retrieving heterophilic candidates for high-coverage subset")
    return retrieve_hetero_candidates(
        edge_index=graph.edge_index,
        edges=graph.edges,
        nodes=graph.nodes,
        node_id_to_idx=graph.node_id_to_idx,
        text_features=graph.text_features,
        numeric_features=graph.numeric_features,
        target_indices=target_indices,
        max_candidates_per_node=max_candidates_per_node,
    )


if __name__ == "__main__":
    main()
