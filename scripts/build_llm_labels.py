from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_risk_cards import build_risk_cards  # noqa: E402
from src.data.loader import load_processed_data  # noqa: E402
from src.graph.neighbor_retrieval import retrieve_hetero_candidates  # noqa: E402
from src.llm.base_labeler import (  # noqa: E402
    OPTIONAL_REAL_LLM_MESSAGE,
    OptionalLabelerUnavailable,
    jsonl_label,
    normalize_label,
)
from src.llm.json_utils import fallback_label, label_key, strict_jsonl_label  # noqa: E402
from src.llm.local_qwen_labeler import DEFAULT_QWEN_MODEL, LocalQwenRiskLabeler  # noqa: E402
from src.llm.mock_labeler import label_risk_card_mechanism  # noqa: E402
from src.llm.openai_labeler import OpenAIRiskLabeler  # noqa: E402
from src.llm.risk_card import format_candidate_risk_card  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM mechanism labels from target-neighbor risk cards.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--labeler", required=True, choices=["mock", "openai", "local_qwen"], help="Labeler backend.")
    parser.add_argument("--risk_card_file", default=None, help="Input risk_cards.jsonl path.")
    parser.add_argument("--max_cards", type=int, default=None, help="Optional card cap. Used for legacy card construction too.")
    parser.add_argument("--out_file", required=True, help="Output JSONL label file.")
    parser.add_argument("--seed", type=int, default=0, help="Risk card sampling seed.")
    parser.add_argument("--max_candidates_per_node", type=int, default=20, help="Candidate cap per sampled target.")
    parser.add_argument("--max_target_nodes", type=int, default=None, help="Optional target node cap for legacy construction.")
    parser.add_argument("--openai_model", default="gpt-5.4-mini", help="OpenAI model name.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_QWEN_MODEL, help="Qwen model name or local path.")
    parser.add_argument("--local_model_path", default=None, help="Deprecated alias for --model_name_or_path.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation cap for local Qwen.")
    parser.add_argument("--report_dir", default="outputs/summary_llm", help="Directory for build report JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.labeler in {"openai", "local_qwen"} and args.max_cards is not None and args.max_cards > 2000:
        raise SystemExit("Real LLM labeler is optional and intended for 500 to 2000 cards. Please lower --max_cards.")
    try:
        labeler = _make_labeler(args)
        report = build_labels(args=args, labeler=labeler)
    except OptionalLabelerUnavailable as exc:
        raise SystemExit(str(exc) or OPTIONAL_REAL_LLM_MESSAGE) from exc
    report_path = Path(args.report_dir) / f"llm_label_build_report_{args.dataset}_{args.labeler}.json"
    write_json(report_path, report)
    print(f"Wrote {report['num_written']} {args.labeler} labels to {args.out_file}")
    print(f"Wrote LLM label build report to {report_path}")


def build_labels(args: argparse.Namespace, labeler) -> dict[str, Any]:
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    processed_keys = _read_existing_keys(out_file)
    parse_error_count = 0
    fallback_count = 0
    num_seen = 0
    num_written = 0
    num_skipped_existing = 0
    mechanism_counts: dict[str, int] = {}
    risk_values: list[int] = []
    confidence_values: list[float] = []
    strict_output = bool(args.risk_card_file) or args.dataset in {"yelp_academic", "amazon_video"}

    with out_file.open("a", encoding="utf-8") as handle:
        for card in tqdm(_iter_risk_cards(args), desc=f"{args.labeler}_labeler", unit="card"):
            num_seen += 1
            key = _card_key(card)
            if key in processed_keys:
                num_skipped_existing += 1
                continue
            label, parse_errors, used_fallback = _label_with_retries(labeler, card)
            parse_error_count += parse_errors
            fallback_count += int(used_fallback)
            payload = _strict_output_label(label, card)
            if strict_output:
                handle.write(strict_jsonl_label(payload, risk_card=card) + "\n")
            else:
                handle.write(jsonl_label(normalize_label(label, risk_card=card)) + "\n")
            handle.flush()
            processed_keys.add(key)
            num_written += 1
            mechanism = str(payload["mechanism"])
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            risk_values.append(int(payload["risk_relevance"]))
            confidence_values.append(float(payload["confidence"]))

    return {
        "dataset": args.dataset,
        "labeler": args.labeler,
        "risk_card_file": str(args.risk_card_file or ""),
        "out_file": str(out_file),
        "model_name_or_path": str(args.model_name_or_path if args.labeler == "local_qwen" else ""),
        "openai_model": str(args.openai_model if args.labeler == "openai" else ""),
        "num_seen": int(num_seen),
        "num_written": int(num_written),
        "num_skipped_existing": int(num_skipped_existing),
        "parse_error_count": int(parse_error_count),
        "fallback_count": int(fallback_count),
        "risk_relevance_rate": float(np.mean(risk_values)) if risk_values else 0.0,
        "avg_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "mechanism_distribution": mechanism_counts,
    }


def _make_labeler(args: argparse.Namespace):
    if args.labeler == "mock":
        return lambda card: normalize_label(label_risk_card_mechanism(card), risk_card=card)
    if args.labeler == "openai":
        openai = OpenAIRiskLabeler(model=args.openai_model)
        return openai.label_risk_card
    if args.labeler == "local_qwen":
        model_name_or_path = args.local_model_path or args.model_name_or_path
        qwen = LocalQwenRiskLabeler(model_name_or_path=model_name_or_path, max_new_tokens=args.max_new_tokens)
        return qwen.label_risk_card
    raise ValueError(f"Unknown labeler={args.labeler}")


def _label_with_retries(labeler, card: dict[str, Any]) -> tuple[dict[str, Any], int, bool]:
    parse_errors = 0
    for _attempt in range(3):
        try:
            return normalize_label(labeler(card), risk_card=card), parse_errors, False
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1
    return fallback_label(card), parse_errors, True


def _strict_output_label(label: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_label(label, risk_card=card)
    return {
        "dataset": str(normalized.get("dataset", card.get("dataset", ""))),
        "target_id": str(normalized.get("target_id", card.get("target_id", ""))),
        "neighbor_id": str(normalized.get("neighbor_id", card.get("neighbor_id", ""))),
        "mechanism": str(normalized.get("mechanism", "irrelevant_heterophily")),
        "risk_relevance": int(normalized.get("risk_relevance", 0)),
        "confidence": float(normalized.get("confidence", 0.0)),
        "rationale": str(normalized.get("rationale", "")),
    }


def _iter_risk_cards(args: argparse.Namespace):
    if args.risk_card_file:
        risk_card_file = Path(args.risk_card_file)
        if not risk_card_file.exists():
            raise FileNotFoundError(f"Missing risk card file: {risk_card_file}")
        limit = args.max_cards if args.max_cards is not None else None
        count = 0
        with risk_card_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                card = json.loads(line)
                card.setdefault("dataset", args.dataset)
                yield card
                count += 1
                if limit is not None and count >= limit:
                    break
        return

    # Backward-compatible path used by existing smoke tests. New real-LLM studies
    # should call scripts/build_risk_cards.py explicitly and pass --risk_card_file.
    max_cards = int(args.max_cards if args.max_cards is not None else 500)
    if args.dataset in {"yelp_academic", "amazon_video"}:
        for card in build_risk_cards(
            dataset=args.dataset,
            data_dir=Path(args.data_dir or f"data/processed/{args.dataset}"),
            max_cards=max_cards,
            seed=args.seed,
            max_candidates_per_node=args.max_candidates_per_node,
            max_target_nodes=args.max_target_nodes,
        ):
            yield card
        return
    for card in _legacy_build_risk_cards(
        dataset=args.dataset,
        data_dir=args.data_dir,
        seed=args.seed,
        max_cards=max_cards,
        max_candidates_per_node=args.max_candidates_per_node,
        max_target_nodes=args.max_target_nodes,
    ):
        yield card


def _read_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = normalize_label(payload)
            keys.add(label_key(normalized))
            keys.add(label_key({"target_id": normalized.get("target_id", ""), "neighbor_id": normalized.get("neighbor_id", "")}))
    return keys


def _card_key(card: dict[str, Any]) -> str:
    return label_key(
        {
            "target_id": card.get("target_id", ""),
            "neighbor_id": card.get("neighbor_id", ""),
        }
    )


def _legacy_build_risk_cards(
    dataset: str,
    data_dir: str | None,
    seed: int,
    max_cards: int,
    max_candidates_per_node: int,
    max_target_nodes: int | None,
) -> list[dict[str, Any]]:
    graph = load_processed_data(data_dir or f"data/processed/{dataset}")
    target_indices = _sample_target_indices(
        graph,
        seed=seed,
        max_cards=max_cards,
        max_candidates_per_node=max_candidates_per_node,
        max_target_nodes=max_target_nodes,
    )
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
