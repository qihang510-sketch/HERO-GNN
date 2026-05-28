from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_processed_data  # noqa: E402
from src.graph.neighbor_retrieval import retrieve_hetero_candidates  # noqa: E402
from src.llm.base_labeler import (  # noqa: E402
    OPTIONAL_REAL_LLM_MESSAGE,
    OptionalLabelerUnavailable,
    jsonl_label,
    normalize_label,
)
from src.llm.local_qwen_labeler import LocalQwenRiskLabeler  # noqa: E402
from src.llm.mock_labeler import label_risk_card_mechanism  # noqa: E402
from src.llm.openai_labeler import OpenAIRiskLabeler  # noqa: E402
from src.llm.risk_card import format_candidate_risk_card  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small-sample LLM mechanism labels from HERO risk cards.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--labeler", required=True, choices=["mock", "openai", "local_qwen"], help="Labeler backend.")
    parser.add_argument("--max_cards", type=int, default=500, help="Maximum risk cards to label.")
    parser.add_argument("--out_file", required=True, help="Output JSONL label file.")
    parser.add_argument("--seed", type=int, default=0, help="Risk card sampling seed.")
    parser.add_argument("--max_candidates_per_node", type=int, default=20, help="Candidate cap per sampled target.")
    parser.add_argument("--max_target_nodes", type=int, default=None, help="Optional target node cap.")
    parser.add_argument("--openai_model", default=None, help="Optional OpenAI model override.")
    parser.add_argument("--local_model_path", default=None, help="Optional local Qwen model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.labeler in {"openai", "local_qwen"} and args.max_cards > 2000:
        raise SystemExit("Real LLM labeler is optional and intended for 500 to 2000 cards. Please lower --max_cards.")
    try:
        labeler = _make_labeler(args)
        cards = _build_risk_cards(
            dataset=args.dataset,
            data_dir=args.data_dir,
            seed=args.seed,
            max_cards=args.max_cards,
            max_candidates_per_node=args.max_candidates_per_node,
            max_target_nodes=args.max_target_nodes,
        )
        out_file = Path(args.out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as handle:
            for card in tqdm(cards, desc=f"{args.labeler}_labeler", unit="card"):
                label = labeler(card)
                handle.write(jsonl_label(label) + "\n")
    except OptionalLabelerUnavailable as exc:
        raise SystemExit(str(exc) or OPTIONAL_REAL_LLM_MESSAGE) from exc
    print(f"Wrote {len(cards)} {args.labeler} labels to {args.out_file}")


def _make_labeler(args: argparse.Namespace):
    if args.labeler == "mock":
        return lambda card: normalize_label(label_risk_card_mechanism(card), risk_card=card)
    if args.labeler == "openai":
        openai = OpenAIRiskLabeler(model=args.openai_model)
        return openai.label_risk_card
    if args.labeler == "local_qwen":
        qwen = LocalQwenRiskLabeler(model_path=args.local_model_path)
        return qwen.label_risk_card
    raise ValueError(f"Unknown labeler={args.labeler}")


def _build_risk_cards(
    dataset: str,
    data_dir: str | None,
    seed: int,
    max_cards: int,
    max_candidates_per_node: int,
    max_target_nodes: int | None,
) -> list[dict]:
    graph = load_processed_data(data_dir or f"data/processed/{dataset}")
    target_indices = _sample_target_indices(graph, seed=seed, max_cards=max_cards, max_candidates_per_node=max_candidates_per_node, max_target_nodes=max_target_nodes)
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
    flat = [candidate for values in candidates.values() for candidate in values]
    flat = sorted(flat, key=lambda candidate: candidate.candidate_score, reverse=True)
    return [format_candidate_risk_card(candidate) for candidate in flat[: max(0, int(max_cards))]]


def _sample_target_indices(graph, seed: int, max_cards: int, max_candidates_per_node: int, max_target_nodes: int | None) -> np.ndarray:
    values = []
    for split_name in ("train", "val", "test"):
        values.extend(graph.split.get(split_name, np.array([], dtype=np.int64)).tolist())
    unique = np.asarray(sorted(set(int(value) for value in values)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    if max_target_nodes is not None:
        return unique[: max(0, int(max_target_nodes))]
    target_limit = max(1, int(np.ceil(max_cards / max(max_candidates_per_node, 1))) * 2)
    return unique[: min(target_limit, unique.size)]


if __name__ == "__main__":
    main()
