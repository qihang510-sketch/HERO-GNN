from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_processed_data  # noqa: E402


DEFAULT_DATASETS = ["yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"]
TEXT_RICH_DATASETS = {"yelp_academic", "amazon_video"}
DATASET_DIR_ALIASES = {
    "yelp_academic": ["yelp_academic"],
    "amazon_video": ["amazon_video"],
    "fraud_yelp": ["fraud_yelp", "fraud_yelp_official"],
    "fraud_amazon": ["fraud_amazon", "fraud_amazon_official"],
    "elliptic": ["elliptic"],
}
FORBIDDEN_SOURCE_MARKERS = {
    "forecast",
    "planning_only",
    "not_for_paper",
    "mock_test",
    "highcov_test",
    "test_strict",
}
NA = "N/A"

CASE_FIELDS = [
    "dataset",
    "case_id",
    "target_id",
    "target_label",
    "neighbor_id",
    "neighbor_label",
    "relation_type",
    "hop_distance",
    "split",
    "target_text_summary",
    "neighbor_text_summary",
    "target_rating",
    "neighbor_rating",
    "target_time",
    "neighbor_time",
    "target_item_or_business",
    "neighbor_item_or_business",
    "target_user_or_account",
    "neighbor_user_or_account",
    "raw_edge_type",
    "raw_edge_weight",
    "raw_path_example",
    "structural_proximity",
    "common_neighbor_count",
    "jaccard_similarity",
    "degree_target",
    "degree_neighbor",
    "neighbor_fraud_ratio",
    "suspicious_path_count",
    "text_similarity",
    "rating_gap",
    "time_gap",
    "same_item_or_business",
    "same_user_or_counterparty",
    "burst_indicator",
    "behavior_conflict_score",
    "risk_relevance",
    "mechanism_candidate",
    "mechanism_description",
    "confidence",
    "risk_summary",
    "keep_or_downweight",
]

TRACE_COLUMNS = [
    "dataset",
    "case_id",
    "field_name",
    "source_column_or_file",
    "raw_value_target",
    "raw_value_neighbor",
    "computation_rule",
    "computed_value",
    "threshold_or_normalization",
    "risk_card_slot",
    "risk_interpretation",
]


@dataclass
class ExtractionResult:
    cases: list[dict[str, Any]]
    field_traces: list[dict[str, Any]]
    dataset_reports: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract representative risk-card construction cases.")
    parser.add_argument("--data_root", default="data", help="Project data directory.")
    parser.add_argument("--output_dir", default="outputs/risk_card_cases", help="Output directory.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to extract.")
    parser.add_argument("--source_outputs", nargs="*", default=[], help="Existing output roots to search for HERO artifacts.")
    parser.add_argument("--top_candidates_per_dataset", type=int, default=3, help="Candidate rows kept in selected_cases.jsonl.")
    parser.add_argument("--top_cases_per_dataset", type=int, default=1, help="Cases marked as selected for table construction.")
    parser.add_argument("--strict", action="store_true", help="Fail if any selected case has unavailable fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = extract_representative_risk_cards(
        data_root=args.data_root,
        datasets=args.datasets,
        source_outputs=args.source_outputs,
        top_candidates_per_dataset=args.top_candidates_per_dataset,
        top_cases_per_dataset=args.top_cases_per_dataset,
        strict=args.strict,
    )
    write_raw_outputs(result, args.output_dir)
    print(f"Wrote {len(result.cases)} candidate case rows to {Path(args.output_dir) / 'raw' / 'selected_cases.jsonl'}")
    print(f"Wrote {len(result.field_traces)} field trace rows to {Path(args.output_dir) / 'raw' / 'risk_card_field_traces.jsonl'}")


def extract_representative_risk_cards(
    data_root: str | Path = "data",
    datasets: Iterable[str] = DEFAULT_DATASETS,
    source_outputs: Iterable[str | Path] | None = None,
    top_candidates_per_dataset: int = 3,
    top_cases_per_dataset: int = 1,
    strict: bool = False,
) -> ExtractionResult:
    data_root = Path(data_root)
    source_roots = [Path(path) for path in (source_outputs or [])]
    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            dataset_cases, dataset_traces, report = _extract_dataset_cases(
                dataset=dataset,
                data_root=data_root,
                source_outputs=source_roots,
                top_candidates_per_dataset=max(1, int(top_candidates_per_dataset)),
                top_cases_per_dataset=max(1, int(top_cases_per_dataset)),
            )
        except Exception as exc:
            processed_dir = _resolve_processed_dir(data_root, dataset)
            reason = f"Risk-card case extraction failed for {dataset}: {type(exc).__name__}: {exc}"
            dataset_cases, dataset_traces, report = _unavailable_dataset(dataset, reason, processed_dir)
        if strict:
            missing = [
                (case.get("dataset"), case.get("case_id"), field)
                for case in dataset_cases
                if _is_selected(case)
                for field in case.get("unavailable_fields", [])
            ]
            if missing:
                raise ValueError(f"Strict mode found unavailable selected-case fields: {missing[:10]}")
        cases.extend(dataset_cases)
        traces.extend(dataset_traces)
        reports.append(report)
    return ExtractionResult(cases=cases, field_traces=traces, dataset_reports=reports)


def write_raw_outputs(result: ExtractionResult, output_dir: str | Path) -> None:
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw_dir / "selected_cases.jsonl", result.cases)
    _write_jsonl(raw_dir / "risk_card_field_traces.jsonl", result.field_traces)


def _extract_dataset_cases(
    dataset: str,
    data_root: Path,
    source_outputs: list[Path],
    top_candidates_per_dataset: int,
    top_cases_per_dataset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    processed_dir = _resolve_processed_dir(data_root, dataset)
    if not _has_processed_graph(processed_dir):
        reason = f"Processed graph files are unavailable under {processed_dir}."
        return _unavailable_dataset(dataset, reason, processed_dir)

    try:
        graph = load_processed_data(processed_dir)
    except Exception as exc:  # pragma: no cover - defensive path for incomplete external datasets.
        reason = f"Processed graph could not be loaded from {processed_dir}: {exc}"
        return _unavailable_dataset(dataset, reason, processed_dir)

    context = _graph_context(graph, processed_dir)
    roots = _source_search_roots(source_outputs, processed_dir)
    evidence_index, evidence_files = _load_evidence_index(dataset, roots)
    candidates = _load_annotation_candidates(dataset, roots, evidence_index)
    if not candidates:
        candidates = _load_risk_card_candidates(dataset, roots, evidence_index)
    if not candidates:
        candidates = _load_hetero_candidate_cache(dataset, processed_dir, evidence_index)
    if not candidates:
        candidates = _build_graph_only_candidates(dataset, context, evidence_index)

    if not candidates:
        reason = "No real annotation, risk-card cache, evidence chain, hetero candidate, or graph edge candidates were found."
        return _unavailable_dataset(dataset, reason, processed_dir)

    candidates = _attach_context_and_score(candidates, context, evidence_index)
    selected_candidates = candidates[:top_candidates_per_dataset]
    selected_ids = {
        str(candidate.get("target_id"))
        for candidate in selected_candidates
        if _has_value(candidate.get("target_id"))
    }
    selected_ids.update(
        str(candidate.get("neighbor_id"))
        for candidate in selected_candidates
        if _has_value(candidate.get("neighbor_id"))
    )
    raw_lookup = _load_raw_review_lookup(dataset, data_root / "raw", selected_ids)

    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        candidate = dict(candidate)
        candidate["selection_rank"] = rank
        candidate["is_selected"] = bool(rank <= top_cases_per_dataset)
        case = _build_case_record(dataset, candidate, context, raw_lookup, evidence_index)
        cases.append(case)
        if _is_selected(case):
            traces.extend(_build_field_traces(case, context, raw_lookup))

    report = _dataset_report(dataset, processed_dir, cases, traces, evidence_files)
    return cases, traces, report


def _resolve_processed_dir(data_root: Path, dataset: str) -> Path:
    processed_root = data_root / "processed"
    aliases = DATASET_DIR_ALIASES.get(dataset, [dataset])
    for alias in aliases:
        path = processed_root / alias
        if _has_processed_graph(path):
            return path
    for alias in aliases:
        path = processed_root / alias
        if path.exists():
            return path
    return processed_root / aliases[0]


def _has_processed_graph(path: Path) -> bool:
    if not path.exists():
        return False
    standard = ["nodes.csv", "edges.csv", "features.npz", "split.json"]
    if all((path / name).exists() for name in standard):
        return True
    elliptic_files = ["features.npz", "labels.npy", "edge_index.npy"]
    return path.name.lower() == "elliptic" and any((path / name).exists() for name in elliptic_files)


def _source_search_roots(source_outputs: list[Path], processed_dir: Path) -> list[Path]:
    roots = [processed_dir]
    roots.extend(path for path in source_outputs if path.exists())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _graph_context(graph: Any, processed_dir: Path) -> dict[str, Any]:
    nodes = graph.nodes.copy()
    edges = graph.edges.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    if "src" in edges:
        edges["src"] = edges["src"].astype(str)
    if "dst" in edges:
        edges["dst"] = edges["dst"].astype(str)
    node_lookup = {str(row["node_id"]): row.to_dict() for _, row in nodes.iterrows()}
    idx_to_node_id = {int(idx): str(node_id) for node_id, idx in graph.node_id_to_idx.items()}
    adjacency = _build_undirected_adjacency(graph.edge_index, len(idx_to_node_id))
    edge_types = _edge_type_lookup(edges, graph.node_id_to_idx)
    labels = np.asarray(graph.labels, dtype=np.int64)
    feature_scale = _max_edge_l1(graph.numeric_features, graph.edge_index)
    return {
        "graph": graph,
        "processed_dir": processed_dir,
        "nodes": nodes,
        "edges": edges,
        "node_lookup": node_lookup,
        "node_id_to_idx": graph.node_id_to_idx,
        "idx_to_node_id": idx_to_node_id,
        "adjacency": adjacency,
        "edge_types": edge_types,
        "labels": labels,
        "feature_scale": feature_scale,
        "nodes_source": _rel_source(processed_dir / "nodes.csv"),
        "edges_source": _rel_source(processed_dir / "edges.csv"),
        "features_source": _rel_source(processed_dir / "features.npz"),
        "preprocess_source": _rel_source(processed_dir / "preprocess_report.json"),
    }


def _load_evidence_index(dataset: str, roots: list[Path]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    files = _discover_files(dataset, roots, ["evidence_chains.jsonl", "*evidence_chains*.jsonl", "evidence_chains.csv", "*evidence_chains*.csv"])
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in files:
        if path.suffix.lower() == ".csv":
            rows = _read_csv_records(path)
        else:
            rows = _read_jsonl_records(path)
        for record in rows:
            for item in _evidence_records_from_payload(record, path):
                key = (str(item["target_id"]), str(item["neighbor_id"]))
                previous = index.get(key)
                if previous is None or _as_float(item.get("chain_quality"), 0.0) > _as_float(previous.get("chain_quality"), 0.0):
                    index[key] = item
    return index, [_rel_source(path) for path in files]


def _evidence_records_from_payload(record: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(record.get("chains"), list):
        for chain in record.get("chains", []):
            records.extend(_evidence_records_from_chain(chain, path))
        return records
    if "chain_nodes" in record:
        return _evidence_records_from_chain(record, path)
    target = _first_present(record, ["target_id", "target", "node_id", "src"])
    neighbor = _first_present(record, ["neighbor_id", "neighbor", "dst"])
    if not (_has_value(target) and _has_value(neighbor)):
        return []
    chain_nodes = [str(target), str(neighbor)]
    chain_edges = [_first_present(record, ["chain_edges", "edge_type", "relation_type", "metapath"], "edge")]
    records.append(
        {
            "target_id": str(target),
            "neighbor_id": str(neighbor),
            "chain_nodes": chain_nodes,
            "chain_edges": chain_edges,
            "chain_quality": _first_present(record, ["chain_quality", "chain_score", "risk_relevance_score", "score", "confidence"], NA),
            "chain_score": _first_present(record, ["chain_score", "score"], NA),
            "confidence": _first_present(record, ["confidence", "confidence_score"], NA),
            "risk_relevance": _first_present(record, ["risk_relevance", "risk_relevance_score"], NA),
            "mechanism": _first_present(record, ["mechanism", "mechanism_label", "relation_type"], NA),
            "rationale": _first_present(record, ["rationale", "evidence", "key_evidence", "reason"], NA),
            "source_file": _rel_source(path),
        }
    )
    return records


def _evidence_records_from_chain(chain: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    chain_nodes = _jsonish_list(chain.get("chain_nodes"))
    if len(chain_nodes) < 2:
        return []
    chain_edges = _jsonish_list(chain.get("chain_edges"))
    target = str(chain.get("target_id", chain_nodes[0]))
    neighbor = str(chain_nodes[1])
    return [
        {
            "target_id": target,
            "neighbor_id": neighbor,
            "chain_nodes": [str(value) for value in chain_nodes],
            "chain_edges": [str(value) for value in chain_edges],
            "chain_quality": _first_present(chain, ["chain_quality", "chain_score", "confidence"], NA),
            "chain_score": _first_present(chain, ["chain_score", "score"], NA),
            "confidence": _first_present(chain, ["confidence"], NA),
            "risk_relevance": _first_present(chain, ["risk_relevance"], NA),
            "mechanism": _first_present(chain, ["mechanism"], NA),
            "rationale": _first_present(chain, ["rationale"], NA),
            "structural_proximity": _first_present(chain, ["structural_score"], NA),
            "behavior_conflict_score": _first_present(chain, ["numeric_deviation", "semantic_dissimilarity"], NA),
            "source_file": _rel_source(path),
        }
    ]


def _load_annotation_candidates(
    dataset: str,
    roots: list[Path],
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    patterns = [
        "annotations.jsonl",
        "qwen_annotations.jsonl",
        "cached_llm_annotations.jsonl",
        "mechanism_annotations.jsonl",
        "llm_labels.jsonl",
        "llm_labels_*.jsonl",
        "*annotations*.jsonl",
    ]
    files = _discover_files(dataset, roots, patterns)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for path in files:
        for record in _read_jsonl_records(path):
            candidate = _candidate_from_annotation(dataset, record, path, evidence_index)
            if candidate:
                _merge_candidate(candidates, candidate)
    return list(candidates.values())


def _load_risk_card_candidates(
    dataset: str,
    roots: list[Path],
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    files = _discover_files(dataset, roots, ["*risk_card*.jsonl", "*risk_cards*.jsonl"])
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for path in files:
        for record in _read_jsonl_records(path):
            payload = record.get("risk_card", record)
            if not isinstance(payload, dict):
                continue
            candidate = _candidate_from_risk_card(dataset, payload, path, evidence_index)
            if candidate:
                _merge_candidate(candidates, candidate)
    return list(candidates.values())


def _load_hetero_candidate_cache(
    dataset: str,
    processed_dir: Path,
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    path = processed_dir / "hetero_candidates.pkl"
    if not path.exists() or _is_forbidden_path(path):
        return []
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return []
    values = payload.get("candidates_by_target", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, dict):
        return []
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for target_candidates in values.values():
        for item in list(target_candidates):
            target_id = getattr(item, "target_id", None)
            neighbor_id = getattr(item, "neighbor_id", None)
            if not (_has_value(target_id) and _has_value(neighbor_id)):
                continue
            candidate = {
                "dataset": dataset,
                "target_id": str(target_id),
                "neighbor_id": str(neighbor_id),
                "relation_type": _value_or_na(getattr(item, "metapath", NA)),
                "risk_weight": _as_optional_float(getattr(item, "candidate_score", None)),
                "risk_weight_source": f"{_rel_source(path)}:candidate_score",
                "risk_relevance": NA,
                "confidence": NA,
                "mechanism_candidate": NA,
                "mechanism_description": NA,
                "structural_proximity": _as_optional_float(getattr(item, "structural_score", None)),
                "text_similarity": _as_optional_float(getattr(item, "semantic_similarity", None)),
                "behavior_conflict_score": _behavior_conflict_from_mapping(
                    {
                        "numeric_deviation": getattr(item, "numeric_deviation", None),
                        "rating_deviation": getattr(item, "rating_diff", None),
                        "semantic_similarity": getattr(item, "semantic_similarity", None),
                        "burst_score": getattr(item, "burst_score", None),
                    }
                ),
                "burst_score": _as_optional_float(getattr(item, "burst_score", None)),
                "same_item_or_business": bool(getattr(item, "same_item_or_business", False)),
                "same_user_or_counterparty": bool(getattr(item, "same_user", False)),
                "selection_source": "hetero_candidate_cache",
                "reconstruction_source": "hetero_candidates_cache",
                "cached_annotation_used": False,
                "qwen_annotation_used": False,
                "hero_risk_weight_used": True,
                "evidence_chain_used": False,
                "source_files": [_rel_source(path)],
            }
            _apply_evidence(candidate, evidence_index)
            _merge_candidate(candidates, candidate)
    return list(candidates.values())


def _candidate_from_annotation(
    dataset: str,
    record: dict[str, Any],
    path: Path,
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    risk_card = record.get("risk_card", {})
    if not isinstance(risk_card, dict):
        risk_card = {}
    target_id = _first_present(record, ["target_id"], _first_present(risk_card, ["target_id"], NA))
    neighbor_id = _first_present(record, ["neighbor_id"], _first_present(risk_card, ["neighbor_id"], NA))
    if not (_has_value(target_id) and _has_value(neighbor_id)):
        return None
    source = _rel_source(path)
    labeler = str(record.get("labeler_version", path.stem)).lower()
    candidate = {
        "dataset": dataset,
        "target_id": str(target_id),
        "neighbor_id": str(neighbor_id),
        "relation_type": _first_present(record, ["metapath", "edge_type", "relation_type"], _first_present(risk_card, ["metapath", "edge_type"], NA)),
        "risk_weight": _first_float(record, ["risk_score", "risk_weight"], _first_float(risk_card, ["candidate_score", "risk_score", "risk_weight"], None)),
        "risk_weight_source": f"{source}:risk_card.candidate_score" if _has_any(risk_card, ["candidate_score"]) else f"{source}:risk_score/confidence",
        "risk_relevance": _first_present(record, ["risk_relevance", "risk_relevance_score"], NA),
        "confidence": _first_present(record, ["confidence", "confidence_score"], NA),
        "mechanism_candidate": _first_present(record, ["mechanism", "mechanism_candidate", "mechanism_label"], NA),
        "mechanism_description": _first_present(record, ["rationale", "mechanism_description", "explanation"], NA),
        "structural_proximity": _first_float(risk_card, ["structural_score", "structural_proximity"], None),
        "text_similarity": _first_float(risk_card, ["semantic_similarity", "text_similarity"], None),
        "rating_gap_from_card": _first_float(risk_card, ["rating_deviation", "rating_diff"], None),
        "time_gap_score_from_card": _first_float(risk_card, ["time_deviation"], None),
        "behavior_conflict_score": _behavior_conflict_from_mapping(risk_card),
        "burst_score": _first_float(risk_card, ["burst_score"], None),
        "same_item_or_business": _first_present(risk_card, ["same_item_or_business"], NA),
        "same_user_or_counterparty": _first_present(risk_card, ["same_user"], NA),
        "candidate_reason": _first_present(risk_card, ["candidate_reason"], NA),
        "selection_source": "qwen_annotation" if "qwen" in path.stem.lower() or "qwen" in labeler else "cached_annotation",
        "reconstruction_source": "cached_model_output",
        "cached_annotation_used": True,
        "qwen_annotation_used": bool("qwen" in path.stem.lower() or "qwen" in labeler),
        "hero_risk_weight_used": _has_any(risk_card, ["candidate_score", "risk_score", "risk_weight"]),
        "evidence_chain_used": False,
        "source_files": [source],
        "annotation_labeler_version": _value_or_na(record.get("labeler_version", NA)),
        "risk_card_payload": risk_card,
    }
    _apply_evidence(candidate, evidence_index)
    return candidate


def _candidate_from_risk_card(
    dataset: str,
    risk_card: dict[str, Any],
    path: Path,
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    target_id = _first_present(risk_card, ["target_id"], NA)
    neighbor_id = _first_present(risk_card, ["neighbor_id"], NA)
    if not (_has_value(target_id) and _has_value(neighbor_id)):
        return None
    source = _rel_source(path)
    candidate = {
        "dataset": dataset,
        "target_id": str(target_id),
        "neighbor_id": str(neighbor_id),
        "relation_type": _first_present(risk_card, ["metapath", "edge_type", "relation_type"], NA),
        "risk_weight": _first_float(risk_card, ["candidate_score", "risk_score", "risk_weight"], None),
        "risk_weight_source": f"{source}:candidate_score",
        "risk_relevance": NA,
        "confidence": NA,
        "mechanism_candidate": NA,
        "mechanism_description": _first_present(risk_card, ["candidate_reason"], NA),
        "structural_proximity": _first_float(risk_card, ["structural_score", "structural_proximity"], None),
        "text_similarity": _first_float(risk_card, ["semantic_similarity", "text_similarity"], None),
        "rating_gap_from_card": _first_float(risk_card, ["rating_deviation", "rating_diff"], None),
        "time_gap_score_from_card": _first_float(risk_card, ["time_deviation"], None),
        "behavior_conflict_score": _behavior_conflict_from_mapping(risk_card),
        "burst_score": _first_float(risk_card, ["burst_score"], None),
        "same_item_or_business": _first_present(risk_card, ["same_item_or_business"], NA),
        "same_user_or_counterparty": _first_present(risk_card, ["same_user"], NA),
        "candidate_reason": _first_present(risk_card, ["candidate_reason"], NA),
        "selection_source": "risk_card_cache",
        "reconstruction_source": "risk_card_cache",
        "cached_annotation_used": False,
        "qwen_annotation_used": False,
        "hero_risk_weight_used": True,
        "evidence_chain_used": False,
        "source_files": [source],
        "risk_card_payload": risk_card,
    }
    _apply_evidence(candidate, evidence_index)
    return candidate


def _build_graph_only_candidates(
    dataset: str,
    context: dict[str, Any],
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    graph = context["graph"]
    edge_index = graph.edge_index
    if edge_index.size == 0:
        return []
    labels = context["labels"]
    adjacency = context["adjacency"]
    idx_to_node_id = context["idx_to_node_id"]
    numeric_features = np.asarray(graph.numeric_features, dtype=np.float32)
    max_l1 = context["feature_scale"]
    candidates: list[dict[str, Any]] = []
    source_files = [context["nodes_source"], context["edges_source"], context["features_source"]]
    seen_pairs: set[tuple[str, str]] = set()
    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        target_id = idx_to_node_id.get(src_i)
        neighbor_id = idx_to_node_id.get(dst_i)
        if not target_id or not neighbor_id:
            continue
        key = (target_id, neighbor_id)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        relation = context["edge_types"].get((src_i, dst_i), "edge")
        target_label = _label_at(labels, src_i)
        neighbor_label = _label_at(labels, dst_i)
        label_diff = 1.0 if target_label in {0, 1} and neighbor_label in {0, 1} and target_label != neighbor_label else 0.0
        target_risk = 1.0 if target_label == 1 else 0.0
        neighbor_ratio = _neighbor_fraud_ratio(labels, adjacency.get(src_i, set()))
        numeric_gap = _feature_l1(numeric_features, src_i, dst_i, max_l1)
        structural = _metapath_proximity(str(relation))
        suspicious_count = _suspicious_path_count(labels, adjacency.get(src_i, set()), adjacency.get(dst_i, set()))
        neighbor_ratio_score = safe_float(neighbor_ratio, default=0.0)
        numeric_gap_score = safe_float(numeric_gap, default=0.0)
        structural_score = safe_float(structural, default=0.0)
        suspicious_count_score = safe_float(suspicious_count, default=0.0)
        suspicious_score = min(suspicious_count_score / 3.0, 1.0)
        heuristic_score = _bounded(
            0.30 * label_diff
            + 0.25 * target_risk
            + 0.15 * structural_score
            + 0.15 * numeric_gap_score
            + 0.15 * neighbor_ratio_score
        )
        candidate = {
            "dataset": dataset,
            "target_id": target_id,
            "neighbor_id": neighbor_id,
            "relation_type": relation,
            "risk_weight": heuristic_score,
            "risk_weight_source": "scripts/extract_representative_risk_cards.py:_build_graph_only_candidates",
            "risk_relevance": NA,
            "confidence": NA,
            "mechanism_candidate": _infer_mechanism(dataset, relation, numeric_gap_score, int(suspicious_count_score), neighbor_ratio_score),
            "mechanism_description": "Heuristic-only reconstruction from processed graph structure and node features.",
            "structural_proximity": structural_score,
            "behavior_conflict_score": numeric_gap_score,
            "suspicious_path_score": suspicious_score,
            "suspicious_path_count": suspicious_count,
            "neighbor_fraud_ratio": neighbor_ratio,
            "selection_source": "graph_only_heuristic",
            "reconstruction_source": "processed_graph_only",
            "annotation_source": "unavailable",
            "evidence_source": "processed_graph_only",
            "cached_annotation_used": False,
            "qwen_annotation_used": False,
            "hero_risk_weight_used": False,
            "evidence_chain_used": False,
            "source_files": source_files,
        }
        _apply_evidence(candidate, evidence_index)
        candidates.append(candidate)
    return candidates


def _attach_context_and_score(
    candidates: list[dict[str, Any]],
    context: dict[str, Any],
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    for candidate in candidates:
        _apply_evidence(candidate, evidence_index)
        _attach_graph_features(candidate, context)
    _normalize_risk_weights(candidates)
    for candidate in candidates:
        candidate["case_score"] = _case_score(candidate)
        candidate["label_priority"] = 1 if candidate.get("target_label") == 1 else 0
        candidate["selection_reason"] = _selection_reason(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            int(item.get("label_priority", 0)),
            _as_float(item.get("case_score"), 0.0),
            _as_float(item.get("confidence"), -1.0),
        ),
        reverse=True,
    )


def _attach_graph_features(candidate: dict[str, Any], context: dict[str, Any]) -> None:
    node_id_to_idx = context["node_id_to_idx"]
    labels = context["labels"]
    adjacency = context["adjacency"]
    graph = context["graph"]
    target_id = str(candidate.get("target_id"))
    neighbor_id = str(candidate.get("neighbor_id"))
    target_idx = node_id_to_idx.get(target_id)
    neighbor_idx = node_id_to_idx.get(neighbor_id)
    candidate["target_idx"] = target_idx if target_idx is not None else NA
    candidate["neighbor_idx"] = neighbor_idx if neighbor_idx is not None else NA
    candidate["target_label"] = _label_at(labels, target_idx)
    candidate["neighbor_label"] = _label_at(labels, neighbor_idx)
    target_row = context["node_lookup"].get(target_id, {})
    neighbor_row = context["node_lookup"].get(neighbor_id, {})
    candidate["split"] = _value_or_na(target_row.get("split", NA))
    if target_idx is None or neighbor_idx is None:
        candidate.setdefault("common_neighbor_count", NA)
        candidate.setdefault("jaccard_similarity", NA)
        candidate.setdefault("degree_target", NA)
        candidate.setdefault("degree_neighbor", NA)
        candidate.setdefault("neighbor_fraud_ratio", NA)
        candidate.setdefault("suspicious_path_count", NA)
        return
    target_adj = adjacency.get(int(target_idx), set())
    neighbor_adj = adjacency.get(int(neighbor_idx), set())
    common = target_adj & neighbor_adj
    union = target_adj | neighbor_adj
    candidate["common_neighbor_count"] = int(len(common))
    candidate["jaccard_similarity"] = float(len(common) / len(union)) if union else 0.0
    candidate["degree_target"] = int(len(target_adj))
    candidate["degree_neighbor"] = int(len(neighbor_adj))
    candidate["neighbor_fraud_ratio"] = _neighbor_fraud_ratio(labels, target_adj)
    candidate["suspicious_path_count"] = _suspicious_path_count(labels, target_adj, neighbor_adj)
    if not _has_value(candidate.get("relation_type")):
        candidate["relation_type"] = context["edge_types"].get((int(target_idx), int(neighbor_idx)), NA)
    if not _has_value(candidate.get("structural_proximity")):
        candidate["structural_proximity"] = _metapath_proximity(str(candidate.get("relation_type", "")))
    if not _has_value(candidate.get("text_similarity")):
        candidate["text_similarity"] = _text_similarity(graph.text_features, int(target_idx), int(neighbor_idx))
    if not _has_value(candidate.get("behavior_conflict_score")):
        candidate["behavior_conflict_score"] = _feature_l1(graph.numeric_features, int(target_idx), int(neighbor_idx), context["feature_scale"])


def _normalize_risk_weights(candidates: list[dict[str, Any]]) -> None:
    values = [_as_optional_float(candidate.get("risk_weight")) for candidate in candidates]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        for candidate in candidates:
            candidate["normalized_risk_weight"] = NA
        return
    min_value = min(finite)
    max_value = max(finite)
    for candidate, value in zip(candidates, values):
        if value is None or not math.isfinite(value):
            candidate["normalized_risk_weight"] = NA
        elif max_value > min_value:
            candidate["normalized_risk_weight"] = _bounded((value - min_value) / (max_value - min_value))
        else:
            candidate["normalized_risk_weight"] = _bounded(value)


def _case_score(candidate: dict[str, Any]) -> float:
    weighted_fields = [
        ("normalized_risk_weight", 0.30),
        ("risk_relevance", 0.25),
        ("confidence", 0.15),
        ("structural_proximity", 0.10),
        ("behavior_conflict_score", 0.10),
        ("suspicious_path_score", 0.10),
    ]
    numerator = 0.0
    denominator = 0.0
    for field, weight in weighted_fields:
        value = _as_optional_float(candidate.get(field))
        if value is None:
            continue
        numerator += weight * _bounded(value)
        denominator += weight
    return float(numerator / denominator) if denominator > 0 else 0.0


def _build_case_record(
    dataset: str,
    candidate: dict[str, Any],
    context: dict[str, Any],
    raw_lookup: dict[str, dict[str, Any]],
    evidence_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    case = {field: NA for field in CASE_FIELDS}
    target_id = str(candidate.get("target_id", NA))
    neighbor_id = str(candidate.get("neighbor_id", NA))
    target_row = context["node_lookup"].get(target_id, {})
    neighbor_row = context["node_lookup"].get(neighbor_id, {})
    target_raw = raw_lookup.get(target_id, {})
    neighbor_raw = raw_lookup.get(neighbor_id, {})
    relation = _value_or_na(candidate.get("relation_type", NA))
    edge_source = context["edges_source"]
    evidence = evidence_index.get((target_id, neighbor_id), {})

    case.update(
        {
            "dataset": dataset,
            "case_id": f"{dataset}_case_{int(candidate.get('selection_rank', 1)):02d}",
            "target_id": target_id,
            "target_label": _value_or_na(candidate.get("target_label", NA)),
            "neighbor_id": neighbor_id,
            "neighbor_label": _value_or_na(candidate.get("neighbor_label", NA)),
            "relation_type": relation,
            "hop_distance": _hop_distance(evidence),
            "split": _value_or_na(candidate.get("split", NA)),
            "target_text_summary": _text_summary(_first_present(target_raw, ["text"], target_row.get("text", NA))),
            "neighbor_text_summary": _text_summary(_first_present(neighbor_raw, ["text"], neighbor_row.get("text", NA))),
            "target_rating": _raw_rating(dataset, target_raw),
            "neighbor_rating": _raw_rating(dataset, neighbor_raw),
            "target_time": _raw_time(dataset, target_raw, target_row),
            "neighbor_time": _raw_time(dataset, neighbor_raw, neighbor_row),
            "target_item_or_business": _raw_item(dataset, target_raw),
            "neighbor_item_or_business": _raw_item(dataset, neighbor_raw),
            "target_user_or_account": _raw_user(dataset, target_raw),
            "neighbor_user_or_account": _raw_user(dataset, neighbor_raw),
            "raw_edge_type": relation,
            "raw_edge_weight": _edge_weight(context, target_id, neighbor_id),
            "raw_path_example": _path_example(target_id, neighbor_id, relation, evidence),
            "structural_proximity": _rounded_or_na(candidate.get("structural_proximity")),
            "common_neighbor_count": _value_or_na(candidate.get("common_neighbor_count", NA)),
            "jaccard_similarity": _rounded_or_na(candidate.get("jaccard_similarity")),
            "degree_target": _value_or_na(candidate.get("degree_target", NA)),
            "degree_neighbor": _value_or_na(candidate.get("degree_neighbor", NA)),
            "neighbor_fraud_ratio": _rounded_or_na(candidate.get("neighbor_fraud_ratio")),
            "suspicious_path_count": _value_or_na(candidate.get("suspicious_path_count", NA)),
            "text_similarity": _rounded_or_na(candidate.get("text_similarity")),
            "same_item_or_business": _same_context(
                case_value=None,
                left=_raw_item(dataset, target_raw),
                right=_raw_item(dataset, neighbor_raw),
                fallback=candidate.get("same_item_or_business"),
                relation=relation,
                relation_markers=("item", "business", "product", "rating", "week", "month"),
            ),
            "same_user_or_counterparty": _same_context(
                case_value=None,
                left=_raw_user(dataset, target_raw),
                right=_raw_user(dataset, neighbor_raw),
                fallback=candidate.get("same_user_or_counterparty"),
                relation=relation,
                relation_markers=("user", "reviewer", "counterparty", "transaction"),
            ),
            "burst_indicator": _burst_indicator(candidate, relation),
            "behavior_conflict_score": _rounded_or_na(candidate.get("behavior_conflict_score")),
            "risk_relevance": _value_or_na(candidate.get("risk_relevance", NA)),
            "mechanism_candidate": _mechanism_or_infer(dataset, candidate, relation),
            "mechanism_description": _mechanism_description(candidate),
            "confidence": _rounded_or_na(candidate.get("confidence")),
            "keep_or_downweight": _keep_or_downweight(candidate),
        }
    )
    case["rating_gap"] = _rating_gap(case, candidate)
    case["time_gap"] = _time_gap(case, candidate)
    case["risk_summary"] = _risk_summary(case, candidate)
    case["status"] = "ok"
    case["selection_rank"] = int(candidate.get("selection_rank", 1))
    case["is_selected"] = bool(candidate.get("is_selected", False))
    case["selection_source"] = _value_or_na(candidate.get("selection_source", NA))
    case["selection_score"] = _rounded_or_na(candidate.get("case_score"))
    case["normalized_risk_weight"] = _rounded_or_na(candidate.get("normalized_risk_weight"))
    case["selection_reason"] = _value_or_na(candidate.get("selection_reason", NA))
    case["reconstruction_source"] = _value_or_na(candidate.get("reconstruction_source", NA))
    case["cached_annotation_used"] = bool(candidate.get("cached_annotation_used", False))
    case["qwen_annotation_used"] = bool(candidate.get("qwen_annotation_used", False))
    case["annotation_source"] = _annotation_source(candidate)
    case["evidence_source"] = _value_or_na(candidate.get("evidence_source", "processed_graph_only" if case["reconstruction_source"] == "processed_graph_only" else evidence.get("source_file", NA)))
    case["hero_risk_weight_used"] = bool(candidate.get("hero_risk_weight_used", False))
    case["evidence_chain_used"] = bool(candidate.get("evidence_chain_used", False))
    case["annotation_labeler_version"] = _value_or_na(candidate.get("annotation_labeler_version", NA))
    case["target_text_full"] = _value_or_na(_first_present(target_raw, ["text"], target_row.get("text", NA)))
    case["neighbor_text_full"] = _value_or_na(_first_present(neighbor_raw, ["text"], neighbor_row.get("text", NA)))
    case["source_files"] = _unique_sources(
        list(candidate.get("source_files", []))
        + [context["nodes_source"], edge_source, context["features_source"]]
        + [_value_or_na(target_raw.get("_source_file", NA)), _value_or_na(neighbor_raw.get("_source_file", NA))]
        + [_value_or_na(evidence.get("source_file", NA))]
    )
    case["field_sources"] = _field_sources(dataset, case, context, target_raw, neighbor_raw, candidate, evidence)
    case["unavailable_fields"] = [field for field in CASE_FIELDS if _is_na(case.get(field))]
    return _fill_missing(case)


def _field_sources(
    dataset: str,
    case: dict[str, Any],
    context: dict[str, Any],
    target_raw: dict[str, Any],
    neighbor_raw: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, str]:
    raw_source = _value_or_na(target_raw.get("_source_file", neighbor_raw.get("_source_file", NA)))
    rating_col = "stars" if dataset == "yelp_academic" else "overall" if dataset == "amazon_video" else NA
    user_col = "user_id" if dataset == "yelp_academic" else "reviewerID" if dataset == "amazon_video" else NA
    item_col = "business_id" if dataset == "yelp_academic" else "asin" if dataset == "amazon_video" else NA
    time_col = "date" if dataset == "yelp_academic" else "unixReviewTime" if dataset == "amazon_video" else "timestamp"
    risk_source = _value_or_na(candidate.get("risk_weight_source", NA))
    annotation_source = _first_source(candidate.get("source_files", []))
    evidence_source = _value_or_na(evidence.get("source_file", NA))
    sources = {
        "target_label": f"{context['nodes_source']}:label",
        "neighbor_label": f"{context['nodes_source']}:label",
        "split": f"{context['nodes_source']}:split",
        "relation_type": f"{context['edges_source']}:edge_type",
        "raw_edge_type": f"{context['edges_source']}:edge_type",
        "raw_edge_weight": f"{context['edges_source']}:edge_weight",
        "raw_path_example": evidence_source if _has_value(evidence_source) else f"{context['edges_source']}:src,dst,edge_type",
        "target_text_summary": f"{raw_source}:text" if _has_value(raw_source) else f"{context['nodes_source']}:text",
        "neighbor_text_summary": f"{raw_source}:text" if _has_value(raw_source) else f"{context['nodes_source']}:text",
        "target_rating": f"{raw_source}:{rating_col}" if _has_value(raw_source) and _has_value(rating_col) else "unavailable",
        "neighbor_rating": f"{raw_source}:{rating_col}" if _has_value(raw_source) and _has_value(rating_col) else "unavailable",
        "target_time": f"{raw_source}:{time_col}" if _has_value(raw_source) else f"{context['nodes_source']}:timestamp",
        "neighbor_time": f"{raw_source}:{time_col}" if _has_value(raw_source) else f"{context['nodes_source']}:timestamp",
        "target_item_or_business": f"{raw_source}:{item_col}" if _has_value(raw_source) and _has_value(item_col) else "unavailable",
        "neighbor_item_or_business": f"{raw_source}:{item_col}" if _has_value(raw_source) and _has_value(item_col) else "unavailable",
        "target_user_or_account": f"{raw_source}:{user_col}" if _has_value(raw_source) and _has_value(user_col) else "unavailable",
        "neighbor_user_or_account": f"{raw_source}:{user_col}" if _has_value(raw_source) and _has_value(user_col) else "unavailable",
        "structural_proximity": f"{annotation_source}:risk_card.structural_score" if bool(candidate.get("hero_risk_weight_used")) and _has_value(annotation_source) else "src.graph.neighbor_retrieval._metapath_proximity",
        "common_neighbor_count": "scripts/extract_representative_risk_cards.py:_attach_graph_features",
        "jaccard_similarity": "scripts/extract_representative_risk_cards.py:_attach_graph_features",
        "degree_target": "scripts/extract_representative_risk_cards.py:_attach_graph_features",
        "degree_neighbor": "scripts/extract_representative_risk_cards.py:_attach_graph_features",
        "neighbor_fraud_ratio": "scripts/extract_representative_risk_cards.py:_neighbor_fraud_ratio",
        "suspicious_path_count": "scripts/extract_representative_risk_cards.py:_suspicious_path_count",
        "text_similarity": f"{annotation_source}:risk_card.semantic_similarity" if bool(candidate.get("hero_risk_weight_used")) and _has_value(annotation_source) else f"{context['features_source']}:text_features",
        "rating_gap": "scripts/extract_representative_risk_cards.py:_rating_gap",
        "time_gap": "scripts/extract_representative_risk_cards.py:_time_gap",
        "same_item_or_business": "raw review IDs or relation_type markers",
        "same_user_or_counterparty": "raw review IDs or relation_type markers",
        "burst_indicator": "cached risk_card.burst_score or temporal relation_type",
        "behavior_conflict_score": f"{annotation_source}:risk_card numeric/semantic deviation fields" if bool(candidate.get("hero_risk_weight_used")) and _has_value(annotation_source) else "processed numeric feature L1 distance",
        "risk_relevance": annotation_source if bool(candidate.get("cached_annotation_used")) else "unavailable",
        "mechanism_candidate": annotation_source if bool(candidate.get("cached_annotation_used")) else "heuristic inference from relation/cues",
        "mechanism_description": annotation_source if bool(candidate.get("cached_annotation_used")) else "heuristic inference from relation/cues",
        "confidence": annotation_source if bool(candidate.get("cached_annotation_used")) else "unavailable",
        "risk_summary": "scripts/extract_representative_risk_cards.py:_risk_summary",
        "keep_or_downweight": "scripts/extract_representative_risk_cards.py:_keep_or_downweight",
    }
    for key in list(sources):
        if _is_na(case.get(key)):
            sources[key] = "unavailable"
    return sources


def _build_field_traces(case: dict[str, Any], context: dict[str, Any], raw_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = _trace_fields_for_dataset(str(case["dataset"]))
    traces = []
    for field in fields:
        source = case.get("field_sources", {}).get(field, "unavailable")
        raw_target, raw_neighbor = _trace_raw_values(field, case)
        traces.append(
            _fill_missing(
                {
                    "dataset": case["dataset"],
                    "case_id": case["case_id"],
                    "field_name": field,
                    "source_column_or_file": source,
                    "raw_value_target": raw_target,
                    "raw_value_neighbor": raw_neighbor,
                    "computation_rule": _trace_rule(field),
                    "computed_value": case.get(field, NA),
                    "threshold_or_normalization": _trace_threshold(field),
                    "risk_card_slot": _risk_card_slot(field),
                    "risk_interpretation": _trace_interpretation(field, case),
                }
            )
        )
    return traces


def _trace_fields_for_dataset(dataset: str) -> list[str]:
    common = [
        "target_label",
        "neighbor_label",
        "relation_type",
        "structural_proximity",
        "common_neighbor_count",
        "neighbor_fraud_ratio",
        "suspicious_path_count",
        "risk_relevance",
        "confidence",
        "mechanism_candidate",
    ]
    if dataset == "yelp_academic":
        return [
            "target_text_summary",
            "neighbor_text_summary",
            "target_rating",
            "neighbor_rating",
            "rating_gap",
            "time_gap",
            "same_item_or_business",
            "text_similarity",
        ] + common
    if dataset == "amazon_video":
        return [
            "target_text_summary",
            "neighbor_text_summary",
            "target_rating",
            "neighbor_rating",
            "rating_gap",
            "time_gap",
            "same_item_or_business",
            "burst_indicator",
        ] + common
    if dataset == "fraud_yelp":
        return [
            "raw_edge_type",
            "target_label",
            "neighbor_label",
            "degree_target",
            "degree_neighbor",
            "neighbor_fraud_ratio",
            "common_neighbor_count",
            "jaccard_similarity",
            "structural_proximity",
            "suspicious_path_count",
        ]
    if dataset == "fraud_amazon":
        return [
            "raw_edge_type",
            "target_label",
            "neighbor_label",
            "degree_target",
            "degree_neighbor",
            "neighbor_fraud_ratio",
            "common_neighbor_count",
            "jaccard_similarity",
            "structural_proximity",
            "behavior_conflict_score",
        ]
    if dataset == "elliptic":
        return [
            "target_time",
            "neighbor_time",
            "raw_edge_type",
            "hop_distance",
            "neighbor_fraud_ratio",
            "time_gap",
            "raw_path_example",
            "structural_proximity",
            "suspicious_path_count",
            "mechanism_candidate",
        ]
    return common


def _trace_raw_values(field: str, case: dict[str, Any]) -> tuple[Any, Any]:
    paired = {
        "target_label": ("target_label", "neighbor_label"),
        "neighbor_label": ("target_label", "neighbor_label"),
        "target_text_summary": ("target_text_summary", "neighbor_text_summary"),
        "neighbor_text_summary": ("target_text_summary", "neighbor_text_summary"),
        "target_rating": ("target_rating", "neighbor_rating"),
        "neighbor_rating": ("target_rating", "neighbor_rating"),
        "target_time": ("target_time", "neighbor_time"),
        "neighbor_time": ("target_time", "neighbor_time"),
        "target_item_or_business": ("target_item_or_business", "neighbor_item_or_business"),
        "neighbor_item_or_business": ("target_item_or_business", "neighbor_item_or_business"),
        "target_user_or_account": ("target_user_or_account", "neighbor_user_or_account"),
        "neighbor_user_or_account": ("target_user_or_account", "neighbor_user_or_account"),
        "rating_gap": ("target_rating", "neighbor_rating"),
        "time_gap": ("target_time", "neighbor_time"),
        "same_item_or_business": ("target_item_or_business", "neighbor_item_or_business"),
        "same_user_or_counterparty": ("target_user_or_account", "neighbor_user_or_account"),
    }
    if field in paired:
        left, right = paired[field]
        return case.get(left, NA), case.get(right, NA)
    if field in {"degree_target", "degree_neighbor"}:
        return case.get("degree_target", NA), case.get("degree_neighbor", NA)
    if field in {"relation_type", "raw_edge_type", "raw_path_example", "hop_distance"}:
        return case.get("target_id", NA), case.get("neighbor_id", NA)
    return case.get("target_id", NA), case.get("neighbor_id", NA)


def _trace_rule(field: str) -> str:
    rules = {
        "target_label": "Read target node label from processed nodes.",
        "neighbor_label": "Read neighbor node label from processed nodes.",
        "relation_type": "Read direct edge type or first evidence-chain edge.",
        "raw_edge_type": "Read direct edge type or first evidence-chain edge.",
        "raw_path_example": "Serialize the direct target-neighbor edge or cached evidence chain.",
        "target_text_summary": "Read raw/processed text and truncate to 30 words for table display.",
        "neighbor_text_summary": "Read raw/processed text and truncate to 30 words for table display.",
        "target_rating": "Read raw rating if the dataset exposes a rating column.",
        "neighbor_rating": "Read raw rating if the dataset exposes a rating column.",
        "target_time": "Read raw review time or processed timestamp.",
        "neighbor_time": "Read raw review time or processed timestamp.",
        "rating_gap": "Absolute difference between target and neighbor ratings when both are available.",
        "time_gap": "Absolute target-neighbor time difference in days when timestamps are available.",
        "same_item_or_business": "Compare raw item/business IDs; fall back to relation-type semantics.",
        "same_user_or_counterparty": "Compare raw user/account IDs; fall back to relation-type semantics.",
        "burst_indicator": "Mark burst when cached burst_score >= 0.5 or relation is a temporal co-review edge.",
        "structural_proximity": "Use cached structural score or metapath proximity mapping.",
        "common_neighbor_count": "Count intersection of undirected target and neighbor adjacency sets.",
        "jaccard_similarity": "common_neighbors / union_neighbors over undirected adjacency sets.",
        "degree_target": "Count undirected neighbors of the target node.",
        "degree_neighbor": "Count undirected neighbors of the neighbor node.",
        "neighbor_fraud_ratio": "Share of labeled fraudulent/high-risk nodes in the target ego neighborhood.",
        "suspicious_path_count": "Count common neighbors with fraud/high-risk labels.",
        "text_similarity": "Use cached semantic similarity or cosine over processed text features.",
        "behavior_conflict_score": "Use cached behavior conflict proxy or normalized numeric feature distance.",
        "risk_relevance": "Read cached annotation risk_relevance if available.",
        "confidence": "Read cached annotation confidence if available.",
        "mechanism_candidate": "Read cached mechanism label or infer from relation and structural cues.",
    }
    return rules.get(field, "Read or compute the field from the recorded provenance.")


def _trace_threshold(field: str) -> str:
    thresholds = {
        "structural_proximity": "metapath proximity in [0,1]",
        "jaccard_similarity": "common / union, [0,1]",
        "neighbor_fraud_ratio": "fraudulent labeled neighbors / labeled neighbors",
        "text_similarity": "cosine similarity in [-1,1], usually [0,1] for TF-IDF",
        "behavior_conflict_score": "normalized to [0,1]",
        "risk_relevance": "cached binary label; missing kept as N/A",
        "confidence": "cached score in [0,1]; missing kept as N/A",
        "burst_indicator": "burst_score >= 0.5 or temporal relation",
        "rating_gap": "absolute rating difference; no imputation",
        "time_gap": "absolute day gap; no imputation",
    }
    return thresholds.get(field, "N/A")


def _risk_card_slot(field: str) -> str:
    if field in {
        "dataset",
        "case_id",
        "target_id",
        "target_label",
        "neighbor_id",
        "neighbor_label",
        "relation_type",
        "hop_distance",
        "split",
    }:
        return "basic identifiers"
    if field in {
        "structural_proximity",
        "common_neighbor_count",
        "jaccard_similarity",
        "degree_target",
        "degree_neighbor",
        "neighbor_fraud_ratio",
        "suspicious_path_count",
    }:
        return "derived structural cues"
    if field in {
        "text_similarity",
        "rating_gap",
        "time_gap",
        "same_item_or_business",
        "same_user_or_counterparty",
        "burst_indicator",
        "behavior_conflict_score",
    }:
        return "derived behavioral/textual cues"
    if field.startswith("target_") or field.startswith("neighbor_") or field.startswith("raw_"):
        return "raw evidence"
    return "risk card outputs"


def _trace_interpretation(field: str, case: dict[str, Any]) -> str:
    if _is_na(case.get(field)):
        return "not available in this dataset"
    interpretations = {
        "target_label": "identifies whether the target is labeled fraud/high-risk",
        "neighbor_label": "shows label agreement or heterogeneity with the target",
        "relation_type": "defines why the two nodes are structurally linked",
        "raw_edge_type": "defines the observed graph relation used as evidence",
        "raw_path_example": "shows the evidence path used by the risk card",
        "target_text_summary": "summarizes the target review text evidence",
        "neighbor_text_summary": "summarizes the neighbor review text evidence",
        "target_rating": "anchors behavioral rating evidence for the target",
        "neighbor_rating": "anchors behavioral rating evidence for the neighbor",
        "rating_gap": "large gaps can signal behavioral contradiction",
        "time_gap": "short gaps can support burst or temporal propagation mechanisms",
        "same_item_or_business": "same entity context makes cross-review comparison meaningful",
        "same_user_or_counterparty": "same account/counterparty context indicates exposure risk",
        "burst_indicator": "flags dense temporal co-review behavior",
        "structural_proximity": "higher values indicate closer graph relation",
        "common_neighbor_count": "shared context can indicate coordinated ego-network exposure",
        "jaccard_similarity": "normalizes shared context by ego-network size",
        "degree_target": "describes target ego-network scale",
        "degree_neighbor": "describes neighbor ego-network scale",
        "neighbor_fraud_ratio": "higher values indicate riskier local exposure",
        "suspicious_path_count": "counts high-risk common-neighbor paths",
        "text_similarity": "low similarity in a close context can indicate semantic conflict",
        "behavior_conflict_score": "higher values indicate stronger behavioral mismatch",
        "risk_relevance": "annotation says whether the pair is risk-relevant",
        "confidence": "annotation confidence in the risk relevance/mechanism",
        "mechanism_candidate": "human-readable hypothesis for why the pair matters",
    }
    return interpretations.get(field, "supports the risk-card construction for this case")


def _unavailable_dataset(dataset: str, reason: str, processed_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case = {field: NA for field in CASE_FIELDS}
    case.update(
        {
            "dataset": dataset,
            "case_id": f"{dataset}_unavailable",
            "status": "unavailable",
            "is_selected": True,
            "selection_rank": 1,
            "selection_source": "unavailable",
            "selection_score": NA,
            "selection_reason": reason,
            "reconstruction_source": "unavailable",
            "cached_annotation_used": False,
            "qwen_annotation_used": False,
            "annotation_source": "unavailable",
            "hero_risk_weight_used": False,
            "evidence_chain_used": False,
            "source_files": [_rel_source(processed_dir) if processed_dir else "unavailable"],
            "field_sources": {field: "unavailable" for field in CASE_FIELDS},
            "unavailable_reason": reason,
        }
    )
    case["unavailable_fields"] = [field for field in CASE_FIELDS if _is_na(case.get(field))]
    trace_fields = [
        "target_label",
        "neighbor_label",
        "relation_type",
        "target_text_summary",
        "target_rating",
        "structural_proximity",
        "neighbor_fraud_ratio",
        "risk_relevance",
    ]
    traces = [
        _fill_missing(
            {
                "dataset": dataset,
                "case_id": case["case_id"],
                "field_name": field,
                "source_column_or_file": "unavailable",
                "raw_value_target": NA,
                "raw_value_neighbor": NA,
                "computation_rule": "No real processed graph or cached model output was available.",
                "computed_value": NA,
                "threshold_or_normalization": "N/A",
                "risk_card_slot": _risk_card_slot(field),
                "risk_interpretation": "not available in this dataset",
            }
        )
        for field in trace_fields
    ]
    report = {
        "dataset": dataset,
        "status": "unavailable",
        "processed_dir": _rel_source(processed_dir),
        "selected_case_id": case["case_id"],
        "target_id": NA,
        "neighbor_id": NA,
        "reason": reason,
        "source_files": case["source_files"],
        "unavailable_fields": case["unavailable_fields"],
        "cached_annotation_used": False,
        "qwen_annotation_used": False,
        "hero_risk_weight_used": False,
        "evidence_chain_used": False,
        "reconstruction_source": "unavailable",
    }
    return [case], traces, report


def _dataset_report(
    dataset: str,
    processed_dir: Path,
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    evidence_files: list[str],
) -> dict[str, Any]:
    selected = next((case for case in cases if _is_selected(case)), cases[0])
    return {
        "dataset": dataset,
        "status": selected.get("status", "ok"),
        "processed_dir": _rel_source(processed_dir),
        "selected_case_id": selected.get("case_id", NA),
        "target_id": selected.get("target_id", NA),
        "target_label": selected.get("target_label", NA),
        "neighbor_id": selected.get("neighbor_id", NA),
        "neighbor_label": selected.get("neighbor_label", NA),
        "relation_type": selected.get("relation_type", NA),
        "selection_reason": selected.get("selection_reason", NA),
        "selection_source": selected.get("selection_source", NA),
        "selection_score": selected.get("selection_score", NA),
        "source_files": selected.get("source_files", []),
        "unavailable_fields": selected.get("unavailable_fields", []),
        "cached_annotation_used": bool(selected.get("cached_annotation_used", False)),
        "qwen_annotation_used": bool(selected.get("qwen_annotation_used", False)),
        "hero_risk_weight_used": bool(selected.get("hero_risk_weight_used", False)),
        "evidence_chain_used": bool(selected.get("evidence_chain_used", False)),
        "evidence_files": evidence_files,
        "reconstruction_source": selected.get("reconstruction_source", NA),
        "num_candidate_cases_kept": len(cases),
        "num_field_trace_rows": len(traces),
    }


def _discover_files(dataset: str, roots: list[Path], patterns: list[str]) -> list[Path]:
    aliases = set(DATASET_DIR_ALIASES.get(dataset, [dataset]))
    aliases.add(dataset)
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern) if root.is_dir() else []:
                if not path.is_file() or _is_forbidden_path(path):
                    continue
                if root.name in aliases or any(part in aliases for part in path.parts) or _looks_dataset_neutral(path):
                    files.append(path)
    return _sort_source_files(files)


def _sort_source_files(files: list[Path]) -> list[Path]:
    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name == "llm_labels.jsonl":
            p = 0
        elif "qwen" in name:
            p = 1
        elif "annotation" in name:
            p = 2
        elif "evidence_chains" in name:
            p = 3
        elif "risk_card" in name:
            p = 4
        else:
            p = 5
        smoke = 1 if any(part.lower().startswith("tmp_") or "smoke" in part.lower() for part in path.parts) else 0
        return (smoke, p, str(path))

    return sorted(dict.fromkeys(files), key=priority)


def _looks_dataset_neutral(path: Path) -> bool:
    return path.parent.name in {"raw", "annotations", "summary", "tables"} or path.name in {
        "llm_labels.jsonl",
        "evidence_chains.jsonl",
        "risk_cards.jsonl",
    }


def _is_forbidden_path(path: Path) -> bool:
    text = str(path).lower().replace("\\", "/")
    return any(marker in text for marker in FORBIDDEN_SOURCE_MARKERS)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError:
        return []
    return records


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return []
    return frame.to_dict(orient="records")


def _load_raw_review_lookup(dataset: str, raw_root: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    if not ids or dataset not in TEXT_RICH_DATASETS:
        return {}
    if dataset == "yelp_academic":
        return _load_yelp_raw_lookup(raw_root / "yelp_academic", ids)
    if dataset == "amazon_video":
        return _load_amazon_raw_lookup(raw_root / "amazon_video", ids)
    return {}


def _load_yelp_raw_lookup(raw_dir: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    path = raw_dir / "yelp_academic_dataset_review.json"
    if not path.exists():
        return {}
    remaining = set(ids)
    found: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                review_id = str(record.get("review_id", ""))
                if review_id not in remaining:
                    continue
                found[review_id] = {
                    "text": _value_or_na(record.get("text", NA)),
                    "rating": _value_or_na(record.get("stars", NA)),
                    "time": _value_or_na(record.get("date", NA)),
                    "item": _value_or_na(record.get("business_id", NA)),
                    "user": _value_or_na(record.get("user_id", NA)),
                    "_source_file": _rel_source(path),
                }
                remaining.remove(review_id)
                if not remaining:
                    break
    except Exception:
        return found
    return found


def _load_amazon_raw_lookup(raw_dir: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    path = _resolve_amazon_raw_path(raw_dir)
    if path is None:
        return {}
    index_by_id = {_amazon_index(node_id): node_id for node_id in ids if _amazon_index(node_id) is not None}
    if not index_by_id:
        return {}
    max_index = max(index_by_id)
    found: dict[str, dict[str, Any]] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index > max_index:
                    break
                if index not in index_by_id or not line.strip():
                    continue
                record = _parse_jsonish(line)
                node_id = index_by_id[index]
                summary = str(record.get("summary", ""))
                review_text = str(record.get("reviewText", ""))
                text = f"{summary} {review_text}".strip()
                found[node_id] = {
                    "text": _value_or_na(text),
                    "rating": _value_or_na(record.get("overall", NA)),
                    "time": _value_or_na(record.get("unixReviewTime", NA)),
                    "item": _value_or_na(record.get("asin", NA)),
                    "user": _value_or_na(record.get("reviewerID", NA)),
                    "_source_file": _rel_source(path),
                }
    except Exception:
        return found
    return found


def _resolve_amazon_raw_path(raw_dir: Path) -> Path | None:
    for name in ["reviews.json.gz", "Video_Games.json.gz", "reviews_Amazon_Instant_Video_5.json.gz"]:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def _parse_jsonish(line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        import ast

        value = ast.literal_eval(line)
        return value if isinstance(value, dict) else {}


def _amazon_index(node_id: str) -> int | None:
    text = str(node_id)
    if not text.startswith("ar_"):
        return None
    try:
        return int(text.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _apply_evidence(candidate: dict[str, Any], evidence_index: dict[tuple[str, str], dict[str, Any]]) -> None:
    key = (str(candidate.get("target_id")), str(candidate.get("neighbor_id")))
    evidence = evidence_index.get(key)
    if not evidence:
        candidate.setdefault("suspicious_path_score", 0.0)
        return
    candidate["evidence_chain_used"] = True
    candidate["suspicious_path_score"] = _bounded(_first_float(evidence, ["chain_quality", "chain_score", "confidence"], 1.0))
    candidate["evidence_chain"] = evidence
    candidate["source_files"] = _unique_sources(list(candidate.get("source_files", [])) + [evidence.get("source_file", NA)])
    if not _has_value(candidate.get("mechanism_candidate")):
        candidate["mechanism_candidate"] = evidence.get("mechanism", NA)
    if not _has_value(candidate.get("mechanism_description")):
        candidate["mechanism_description"] = evidence.get("rationale", NA)
    if not _has_value(candidate.get("structural_proximity")) and _has_value(evidence.get("structural_proximity")):
        candidate["structural_proximity"] = evidence.get("structural_proximity")


def _annotation_source(candidate: dict[str, Any]) -> str:
    if not bool(candidate.get("cached_annotation_used", False)):
        return "unavailable"
    for source in candidate.get("source_files", []):
        text = str(source)
        lower = text.lower()
        if "annotation" in lower or "llm_label" in lower or "qwen" in lower:
            return text
    return _first_source(candidate.get("source_files", []))


def _merge_candidate(candidates: dict[tuple[str, str], dict[str, Any]], candidate: dict[str, Any]) -> None:
    key = (str(candidate["target_id"]), str(candidate["neighbor_id"]))
    previous = candidates.get(key)
    if previous is None or _candidate_priority(candidate) > _candidate_priority(previous):
        candidates[key] = candidate


def _candidate_priority(candidate: dict[str, Any]) -> tuple[int, float, float]:
    source = str(candidate.get("selection_source", ""))
    source_priority = {
        "qwen_annotation": 5,
        "cached_annotation": 4,
        "risk_card_cache": 3,
        "hetero_candidate_cache": 2,
        "graph_only_heuristic": 1,
        "heuristic_only": 1,
    }.get(source, 0)
    risk = _as_float(candidate.get("risk_weight"), 0.0)
    confidence = _as_float(candidate.get("confidence"), 0.0)
    return source_priority, risk, confidence


def _selection_reason(candidate: dict[str, Any]) -> str:
    signals = [
        f"selection_source={candidate.get('selection_source', NA)}",
        f"case_score={_rounded_or_na(candidate.get('case_score'))}",
        f"target_label={candidate.get('target_label', NA)}",
        f"risk_relevance={candidate.get('risk_relevance', NA)}",
        f"confidence={_rounded_or_na(candidate.get('confidence'))}",
        f"structural_proximity={_rounded_or_na(candidate.get('structural_proximity'))}",
        f"behavior_conflict_score={_rounded_or_na(candidate.get('behavior_conflict_score'))}",
        f"evidence_chain_used={bool(candidate.get('evidence_chain_used', False))}",
    ]
    return "; ".join(signals)


def _raw_rating(dataset: str, raw: dict[str, Any]) -> Any:
    if dataset not in TEXT_RICH_DATASETS:
        return NA
    return _value_or_na(raw.get("rating", NA))


def _raw_time(dataset: str, raw: dict[str, Any], node_row: dict[str, Any]) -> Any:
    if _has_value(raw.get("time")):
        return raw.get("time")
    return _value_or_na(node_row.get("timestamp", NA))


def _raw_item(dataset: str, raw: dict[str, Any]) -> Any:
    if dataset not in TEXT_RICH_DATASETS:
        return NA
    return _value_or_na(raw.get("item", NA))


def _raw_user(dataset: str, raw: dict[str, Any]) -> Any:
    if dataset not in TEXT_RICH_DATASETS:
        return NA
    return _value_or_na(raw.get("user", NA))


def _rating_gap(case: dict[str, Any], candidate: dict[str, Any]) -> Any:
    target = _as_optional_float(case.get("target_rating"))
    neighbor = _as_optional_float(case.get("neighbor_rating"))
    if target is not None and neighbor is not None:
        return _round(abs(target - neighbor))
    return _rounded_or_na(candidate.get("rating_gap_from_card"))


def _time_gap(case: dict[str, Any], candidate: dict[str, Any]) -> Any:
    left = _timestamp_value(case.get("target_time"))
    right = _timestamp_value(case.get("neighbor_time"))
    if left is not None and right is not None:
        return _round(abs(left - right) / 86400.0)
    return _rounded_or_na(candidate.get("time_gap_score_from_card"))


def _timestamp_value(value: Any) -> float | None:
    if not _has_value(value):
        return None
    numeric = _as_optional_float(value)
    if numeric is not None:
        return numeric
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed.timestamp())


def _same_context(
    case_value: Any,
    left: Any,
    right: Any,
    fallback: Any,
    relation: str,
    relation_markers: tuple[str, ...],
) -> Any:
    if _has_value(left) and _has_value(right):
        return bool(str(left) == str(right))
    if isinstance(fallback, bool):
        return fallback
    if str(fallback).lower() in {"true", "false"}:
        return str(fallback).lower() == "true"
    relation_lower = str(relation).lower()
    if any(marker in relation_lower for marker in relation_markers):
        return True
    return NA


def _burst_indicator(candidate: dict[str, Any], relation: str) -> Any:
    burst = _as_optional_float(candidate.get("burst_score"))
    if burst is not None:
        return bool(burst >= 0.5)
    relation_lower = str(relation).lower()
    if "week" in relation_lower or "month" in relation_lower or "time" in relation_lower:
        return True
    return NA


def _keep_or_downweight(candidate: dict[str, Any]) -> str:
    risk = _as_optional_float(candidate.get("risk_relevance"))
    confidence = _as_optional_float(candidate.get("confidence"))
    score = _as_float(candidate.get("case_score"), 0.0)
    if risk is not None and risk >= 1 and (confidence is None or confidence >= 0.5):
        return "keep"
    if risk is not None and risk <= 0:
        return "downweight"
    if str(candidate.get("selection_source")) in {"heuristic_only", "graph_only_heuristic"}:
        return "keep (graph_only_heuristic)" if score >= 0.5 else "downweight (graph_only_heuristic)"
    return "keep" if score >= 0.5 else "downweight"


def _mechanism_or_infer(dataset: str, candidate: dict[str, Any], relation: str) -> str:
    mechanism = candidate.get("mechanism_candidate", NA)
    if _has_value(mechanism):
        return str(mechanism)
    return _infer_mechanism(
        dataset,
        relation,
        _as_float(candidate.get("behavior_conflict_score"), 0.0),
        int(_as_float(candidate.get("suspicious_path_count"), 0.0)),
        _as_float(candidate.get("neighbor_fraud_ratio"), 0.0),
    )


def _mechanism_description(candidate: dict[str, Any]) -> str:
    description = candidate.get("mechanism_description", NA)
    if _has_value(description):
        return str(description)
    reason = candidate.get("candidate_reason", NA)
    if _has_value(reason):
        return str(reason)
    return "Mechanism inferred from graph proximity, behavior conflict, and local risk exposure."


def _risk_summary(case: dict[str, Any], candidate: dict[str, Any]) -> str:
    pieces = [
        f"{case['target_id']} -> {case['neighbor_id']} via {case['relation_type']}",
        f"mechanism={case['mechanism_candidate']}",
        f"structural={case['structural_proximity']}",
        f"behavior_conflict={case['behavior_conflict_score']}",
    ]
    if _has_value(case.get("risk_relevance")):
        pieces.append(f"risk_relevance={case['risk_relevance']}")
    if bool(candidate.get("evidence_chain_used")):
        pieces.append("evidence_chain=yes")
    if str(candidate.get("selection_source")) in {"heuristic_only", "graph_only_heuristic"}:
        pieces.append("graph_only_heuristic")
    return "; ".join(pieces)


def _infer_mechanism(dataset: str, relation: str, behavior_gap: float, suspicious_count: int, neighbor_ratio: float) -> str:
    relation_lower = str(relation).lower()
    if dataset == "elliptic":
        return "Transaction Chain Risk"
    if dataset == "fraud_yelp":
        if suspicious_count > 0 or neighbor_ratio > 0:
            return "Risky Neighbor Exposure"
        return "Suspicious Co-review Pattern"
    if dataset == "fraud_amazon":
        if suspicious_count > 0 or neighbor_ratio > 0:
            return "Risky Counterparty"
        return "Coordinated Review Pattern"
    if "rating" in relation_lower or behavior_gap >= 0.5:
        return "Rating Conflict"
    if "week" in relation_lower or "month" in relation_lower or "time" in relation_lower:
        return "Burst Behavior"
    if "user" in relation_lower:
        return "Counterparty Risk"
    return "Behavioral Contradiction"


def _edge_weight(context: dict[str, Any], target_id: str, neighbor_id: str) -> Any:
    edges = context["edges"]
    if "edge_weight" not in edges.columns and "weight" not in edges.columns:
        return NA
    weight_col = "edge_weight" if "edge_weight" in edges.columns else "weight"
    match = edges[(edges["src"].astype(str) == str(target_id)) & (edges["dst"].astype(str) == str(neighbor_id))]
    if match.empty:
        return NA
    return _value_or_na(match.iloc[0][weight_col])


def _path_example(target_id: str, neighbor_id: str, relation: str, evidence: dict[str, Any]) -> str:
    nodes = evidence.get("chain_nodes")
    edges = evidence.get("chain_edges")
    if isinstance(nodes, list) and len(nodes) >= 2:
        edge_values = edges if isinstance(edges, list) else []
        parts = [str(nodes[0])]
        for index, node in enumerate(nodes[1:]):
            edge = str(edge_values[index]) if index < len(edge_values) else "edge"
            parts.append(f"-[{edge}]-> {node}")
        return " ".join(parts)
    return f"{target_id} -[{relation}]-> {neighbor_id}"


def _hop_distance(evidence: dict[str, Any]) -> Any:
    nodes = evidence.get("chain_nodes")
    if isinstance(nodes, list) and len(nodes) >= 2:
        return len(nodes) - 1
    return 1


def _build_undirected_adjacency(edge_index: np.ndarray, size: int) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(size)}
    if edge_index.size == 0:
        return adjacency
    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        adjacency.setdefault(src_i, set()).add(dst_i)
        adjacency.setdefault(dst_i, set()).add(src_i)
    return adjacency


def _edge_type_lookup(edges: pd.DataFrame, node_id_to_idx: dict[str, int]) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    if not {"src", "dst", "edge_type"}.issubset(edges.columns):
        return lookup
    for src, dst, edge_type in edges[["src", "dst", "edge_type"]].itertuples(index=False, name=None):
        src_s = str(src)
        dst_s = str(dst)
        if src_s in node_id_to_idx and dst_s in node_id_to_idx:
            lookup[(int(node_id_to_idx[src_s]), int(node_id_to_idx[dst_s]))] = str(edge_type)
    return lookup


def _neighbor_fraud_ratio(labels: np.ndarray, neighbors: set[int]) -> Any:
    if not neighbors:
        return 0.0
    values = np.asarray([_label_at(labels, idx) for idx in neighbors], dtype=np.int64)
    labeled = values[values >= 0]
    if labeled.size == 0:
        return NA
    return float(np.mean(labeled == 1))


def _suspicious_path_count(labels: np.ndarray, left: set[int], right: set[int]) -> int:
    common = left & right
    if not common:
        return 0
    return int(sum(1 for idx in common if _label_at(labels, idx) == 1))


def _label_at(labels: np.ndarray, idx: int | None | Any) -> int | str:
    try:
        idx_i = int(idx)
    except (TypeError, ValueError):
        return NA
    if 0 <= idx_i < labels.shape[0]:
        return int(labels[idx_i])
    return NA


def _text_similarity(text_features: np.ndarray, left: int, right: int) -> Any:
    if text_features.size == 0 or text_features.shape[1] == 0:
        return NA
    a = text_features[left]
    b = text_features[right]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return NA
    return float(np.dot(a, b) / denom)


def _feature_l1(features: np.ndarray, left: int, right: int, scale: float) -> float:
    if features.size == 0 or features.shape[1] == 0:
        return 0.0
    return _bounded(float(np.sum(np.abs(features[left] - features[right])) / max(float(scale), 1e-8)))


def _max_edge_l1(features: np.ndarray, edge_index: np.ndarray) -> float:
    if features.size == 0 or features.shape[1] == 0 or edge_index.size == 0:
        return 1.0
    deltas = np.sum(np.abs(features[edge_index[0]] - features[edge_index[1]]), axis=1)
    return float(max(np.max(deltas), 1.0))


def _metapath_proximity(metapath: str) -> float:
    return {
        "review-user-review": 0.95,
        "review-device-review": 0.90,
        "review-time-review": 0.75,
        "review-week-review": 0.75,
        "review-month-review": 0.75,
        "review-item-review": 0.65,
        "review-business-review": 0.65,
        "review-product-review": 0.65,
        "review-rating-review": 0.60,
        "net_rur": 0.95,
        "net_rsr": 0.65,
        "net_rtr": 0.75,
        "net_upu": 0.65,
        "net_usu": 0.60,
        "net_uvu": 0.60,
        "transaction-transfer": 0.70,
        "edge": 0.50,
    }.get(str(metapath), 0.50)


def _behavior_conflict_from_mapping(values: dict[str, Any]) -> Any:
    candidates: list[float] = []
    for key in ["numeric_deviation", "feature_distance", "behavior_conflict_score", "burst_score"]:
        value = _as_optional_float(values.get(key))
        if value is not None:
            candidates.append(_bounded(value))
    rating = _as_optional_float(values.get("rating_deviation", values.get("rating_diff")))
    if rating is not None:
        candidates.append(_bounded(rating / 5.0))
    similarity = _as_optional_float(values.get("semantic_similarity", values.get("text_similarity")))
    if similarity is not None:
        candidates.append(_bounded(1.0 - similarity))
    semantic_distance = _as_optional_float(values.get("semantic_distance"))
    if semantic_distance is not None:
        candidates.append(_bounded(semantic_distance))
    return max(candidates) if candidates else NA


def _jsonish_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _first_present(record: dict[str, Any], keys: list[str], default: Any = NA) -> Any:
    for key in keys:
        if key in record and _has_value(record[key]):
            return record[key]
    return default


def _first_float(record: dict[str, Any], keys: list[str], default: Any = None) -> float | None:
    for key in keys:
        value = _as_optional_float(record.get(key))
        if value is not None:
            return value
    return default


def _has_any(record: dict[str, Any], keys: list[str]) -> bool:
    return any(_has_value(record.get(key)) for key in keys)


def safe_float(x: Any, default: float = 0.0) -> float:
    if x is None or x is pd.NA:
        return float(default)
    if isinstance(x, str):
        text = x.strip()
        if text == "" or text.lower() in {"n/a", "na", "nan", "none", "unavailable", "<na>"}:
            return float(default)
        try:
            numeric = float(text)
        except ValueError:
            return float(default)
        return numeric if math.isfinite(numeric) else float(default)
    if isinstance(x, np.ndarray):
        values = x.reshape(-1).tolist()
        return safe_float(values, default=default)
    if isinstance(x, (list, tuple, set)):
        numeric_values: list[float] = []
        for item in x:
            value = _as_optional_float(item)
            if value is not None:
                numeric_values.append(float(value))
        if not numeric_values:
            return float(default)
        return float(np.mean(numeric_values))
    if isinstance(x, (np.integer, np.floating)):
        numeric = float(x)
        return numeric if math.isfinite(numeric) else float(default)
    if isinstance(x, (int, float)):
        numeric = float(x)
        return numeric if math.isfinite(numeric) else float(default)
    try:
        numeric = float(x)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return safe_float(value, default=0.0)
    if isinstance(value, (list, tuple, set)):
        if not value:
            return None
        return safe_float(value, default=0.0)
    if not _has_value(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _as_float(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default=default)


def _bounded(value: Any) -> float:
    return float(min(max(_as_float(value, 0.0), 0.0), 1.0))


def _round(value: float) -> float:
    return float(f"{float(value):.4f}")


def _rounded_or_na(value: Any) -> Any:
    numeric = _as_optional_float(value)
    if numeric is None:
        return NA
    return _round(numeric)


def _value_or_na(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "unavailable":
        return "unavailable"
    if not _has_value(value):
        return NA
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    return value


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"n/a", "na", "nan", "none", "unavailable", "<na>"}


def _has_value(value: Any) -> bool:
    return not _is_na(value)


def _fill_missing(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _value_or_na(value) for key, value in record.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return NA
    return value


def _text_summary(text: Any, max_words: int = 30) -> str:
    if not _has_value(text):
        return NA
    return _truncate_words(str(text).replace("\n", " "), max_words=max_words)


def _truncate_words(text: str, max_words: int = 30) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _rel_source(path: str | Path) -> str:
    if not _has_value(path):
        return "unavailable"
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _unique_sources(sources: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for source in sources:
        text = str(source)
        if not _has_value(text) or text == "unavailable":
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result or ["unavailable"]


def _first_source(sources: Any) -> str:
    if isinstance(sources, list):
        for source in sources:
            if _has_value(source):
                return str(source)
    if _has_value(sources):
        return str(sources)
    return "unavailable"


def _is_selected(case: dict[str, Any]) -> bool:
    value = case.get("is_selected", False)
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


if __name__ == "__main__":
    main()
