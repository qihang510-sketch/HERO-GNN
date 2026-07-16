from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


NA = "N/A"
UNAVAILABLE = "unavailable"
INTEGRITY_STATEMENT = "Cost values are extracted from logged runtime files and cached artifacts. Missing fields are reported as N/A rather than estimated."

DEFAULT_DATASETS = ["yelp_academic", "amazon_video"]
DEFAULT_SUITES = ["main", "main_text_rich"]
DEFAULT_MODELS = [
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
    "hero",
    "hero_full",
    "hero_gnn",
]
MODEL_ORDER = ["MLP", "GCN", "GAT", "GraphSAGE", "CARE-GNN", "GraphConsis", "PC-GNN", "BWGNN", "LINKX", "DGP", "MLED", "HERO"]
MODEL_ALIASES = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "graphsage": "GraphSAGE",
    "graph_sage": "GraphSAGE",
    "care_gnn": "CARE-GNN",
    "care-gnn": "CARE-GNN",
    "caregnn": "CARE-GNN",
    "graphconsis": "GraphConsis",
    "graph_consis": "GraphConsis",
    "pc_gnn": "PC-GNN",
    "pc-gnn": "PC-GNN",
    "pcgnn": "PC-GNN",
    "bwgnn": "BWGNN",
    "linkx": "LINKX",
    "dgp": "DGP",
    "mled": "MLED",
    "hero": "HERO",
    "hero_full": "HERO",
    "hero_gnn": "HERO",
    "hero-official": "HERO",
    "hero_official": "HERO",
}
QUICK_SUITES = {"main_quick", "main_text_rich_quick"}
SUPPLEMENT_SUITES = {"transfer", "ablation", "robustness", "faithfulness", "sensitivity", "labeler_comparison", "cost"}
SUCCESS_STATUSES = {"ok", "exists", "success", "completed"}
FAILED_STATUSES = {"failed", "error", "crashed", "timeout"}
SKIPPED_STATUSES = {"skipped", "skip", "missing", "not_applicable", "unavailable"}
FORBIDDEN_TERMS = ["fake", "example", "forecast", "planning_only", "not_for_paper"]

DETAIL_COLUMNS = [
    "suite",
    "dataset",
    "model",
    "seed_count",
    "run_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "train_time_sec_mean",
    "train_time_sec_std",
    "train_time_sec_min",
    "train_time_sec_max",
    "inference_time_sec_mean",
    "inference_time_sec_std",
    "total_runtime_sec_mean",
    "total_runtime_sec_std",
    "total_runtime_sec_sum",
    "gpu_memory_mb_mean",
    "gpu_memory_mb_max",
    "cpu_memory_mb_mean",
    "cpu_memory_mb_max",
    "annotation_time_sec_mean",
    "annotation_time_sec_sum",
    "risk_card_time_sec_mean",
    "risk_card_time_sec_sum",
    "evidence_chain_time_sec_mean",
    "evidence_chain_time_sec_sum",
    "annotation_cards",
    "annotated_cards",
    "annotation_coverage",
    "annotation_source",
    "annotation_model",
    "annotation_deployment",
    "cache_size_mb",
    "num_nodes",
    "num_edges",
    "num_features",
    "num_train",
    "num_val",
    "num_test",
    "cost_source",
    "missing_fields",
    "status",
    "notes",
]

COMPACT_COLUMNS = [
    "Dataset",
    "Model",
    "#Seeds",
    "Train Time / Seed",
    "Total Runtime",
    "Peak GPU Mem.",
    "LLM Annotation Cost",
    "Cache Size",
    "Cost Source",
    "Status",
]

RAW_COLUMNS = [
    "source_output",
    "suite",
    "dataset",
    "model",
    "model_raw",
    "seed",
    "status",
    "original_status",
    "excluded_reason",
    "run_dir",
    "source_files",
    "train_time_sec",
    "inference_time_sec",
    "total_runtime_sec",
    "gpu_memory_mb",
    "cpu_memory_mb",
    "annotation_time_sec",
    "risk_card_time_sec",
    "evidence_chain_time_sec",
    "annotation_source",
    "annotation_model",
    "annotation_deployment",
    "annotation_cards",
    "annotated_cards",
    "annotation_coverage",
    "cache_size_mb",
    "num_nodes",
    "num_edges",
    "num_features",
    "num_train",
    "num_val",
    "num_test",
    "record_source",
]

TIME_ALIASES = {
    "train_time_sec": ["train_time_sec", "training_time_sec", "time_training_sec", "time_train_sec", "Training time"],
    "inference_time_sec": ["inference_time_sec", "time_inference_sec", "eval_time_sec", "test_time_sec", "time_eval_sec"],
    "total_runtime_sec": ["total_runtime_sec", "time_total_sec", "runtime_seconds", "duration_sec", "elapsed_sec", "wall_time_sec"],
    "gpu_memory_mb": ["gpu_memory_mb", "peak_gpu_memory_mb", "max_gpu_memory_mb", "gpu_mem_mb", "max_memory_allocated_mb"],
    "cpu_memory_mb": ["cpu_memory_mb", "peak_cpu_memory_mb", "peak_memory_mb", "max_rss_mb", "rss_memory_mb"],
    "annotation_time_sec": ["annotation_time_sec", "annotation_time_seconds", "time_annotation_sec", "time_llm_annotation_sec", "time_mock_labeling_sec"],
    "risk_card_time_sec": ["risk_card_time_sec", "time_risk_card_sec", "time_risk_card_construction_sec", "risk_card_construction_time_sec"],
    "evidence_chain_time_sec": ["evidence_chain_time_sec", "time_evidence_chain_sec", "evidence_time_sec", "time_chain_sec"],
}
DATASET_ALIASES = {
    "num_nodes": ["num_nodes", "n_nodes", "node_count"],
    "num_edges": ["num_edges", "n_edges", "edge_count"],
    "num_features": ["num_features", "n_features", "feature_dim"],
    "num_train": ["num_train", "train_count", "total_count_train"],
    "num_val": ["num_val", "val_count", "total_count_val"],
    "num_test": ["num_test", "test_count", "total_count_test"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect runtime and annotation cost for HERO main experiments only.")
    parser.add_argument("--source_outputs", nargs="*", default=[], help="Existing main experiment output roots.")
    parser.add_argument("--output_dir", default="outputs/main_cost", help="Output directory for main cost artifacts.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to summarize.")
    parser.add_argument("--suite_names", nargs="+", default=DEFAULT_SUITES, help="Formal main suites to include.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to summarize.")
    parser.add_argument("--data_root", default="data", help="Optional data root for processed graph statistics.")
    parser.add_argument("--include_quick_test", action="store_true", help="Include main_quick/main_text_rich_quick rows.")
    parser.add_argument("--include_failed", action="store_true", default=True, help="Keep failed/skipped runs in raw counts.")
    parser.add_argument("--exclude_failed", action="store_true", help="Drop failed/skipped raw runs before aggregation.")
    parser.add_argument("--write_latex", action="store_true", help="Write LaTeX tables.")
    parser.add_argument("--write_markdown", action="store_true", help="Write Markdown tables.")
    parser.add_argument("--copy_to_final_artifacts", action="store_true", help="Copy tables/report into final_artifacts.")
    parser.add_argument("--final_artifacts_dir", default="outputs/final_artifacts", help="Final artifacts directory.")
    parser.add_argument("--strict", action="store_true", help="Fail when runtime fields are unavailable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_main_experiment_cost(args)
    print(f"Detailed cost table: {paths['detail_csv']}")
    print(f"Compact cost table: {paths['compact_csv']}")
    print(f"Report: {paths['report']}")


def collect_main_experiment_cost(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    dirs = _ensure_dirs(output_dir)
    sources = [Path(path) for path in getattr(args, "source_outputs", [])]
    source_warnings = [f"missing_source_output: {path}" for path in sources if not path.exists()]
    existing_sources = [path for path in sources if path.exists()]
    if not existing_sources:
        source_warnings.append("no_existing_source_outputs")

    requested_models = _requested_models(getattr(args, "models", DEFAULT_MODELS))
    suite_names = [str(item) for item in getattr(args, "suite_names", DEFAULT_SUITES)]
    datasets = [str(item) for item in getattr(args, "datasets", DEFAULT_DATASETS)]
    include_failed = bool(getattr(args, "include_failed", True)) and not bool(getattr(args, "exclude_failed", False))

    records: list[dict[str, Any]] = []
    for root in existing_sources:
        records.extend(
            _collect_records_from_root(
                root=root,
                suite_names=suite_names,
                datasets=datasets,
                requested_models=requested_models,
                include_quick_test=bool(getattr(args, "include_quick_test", False)),
            )
        )
    records = _dedupe_records(records)
    if not include_failed:
        records = [record for record in records if _is_success(record.get("original_status", record.get("status")))]

    annotation_index = _annotation_index(existing_sources, datasets)
    dataset_stats = _dataset_stats_index(Path(getattr(args, "data_root", "data")), existing_sources, datasets)
    for record in records:
        dataset = str(record.get("dataset", ""))
        if _is_hero(record.get("model")):
            _attach_annotation_info(record, annotation_index.get(dataset, {}))
        _attach_dataset_stats(record, dataset_stats.get(dataset, {}))

    raw_frame = _raw_frame(records)
    detail = build_detail_table(records, datasets=datasets, suite_names=suite_names, models=_dedupe_texts(requested_models.values()), annotation_index=annotation_index, dataset_stats=dataset_stats)
    compact = build_compact_table(detail, datasets=datasets)

    if bool(getattr(args, "strict", False)) and _has_runtime_unavailable(detail):
        raise RuntimeError("Main cost table contains runtime_unavailable rows under --strict.")

    paths: dict[str, Path] = {}
    paths["raw_csv"] = _write_csv(dirs["raw"] / "main_cost_raw_runs.csv", raw_frame)
    paths["sources_jsonl"] = _write_sources_jsonl(dirs["raw"] / "main_cost_runtime_sources.jsonl", records, source_warnings)
    paths["detail_csv"] = _write_csv(dirs["tables_csv"] / "supp_table_main_experiment_cost.csv", detail)
    paths["compact_csv"] = _write_csv(dirs["tables_csv"] / "supp_table_main_experiment_cost_compact.csv", compact)
    if bool(getattr(args, "write_latex", False)):
        paths["detail_latex"] = _write_latex(
            dirs["tables_latex"] / "supp_table_main_experiment_cost.tex",
            detail,
            caption="Detailed runtime and annotation cost of main experiments.",
            label="tab:main_cost_detail",
            compact=False,
        )
        paths["compact_latex"] = _write_latex(
            dirs["tables_latex"] / "supp_table_main_experiment_cost_compact.tex",
            compact,
            caption="Computational cost of main experiments.",
            label="tab:main_cost_compact",
            compact=True,
        )
    if bool(getattr(args, "write_markdown", False)):
        paths["detail_markdown"] = _write_markdown(dirs["tables_markdown"] / "supp_table_main_experiment_cost.md", detail)
        paths["compact_markdown"] = _write_markdown(dirs["tables_markdown"] / "supp_table_main_experiment_cost_compact.md", compact)
    paths["report"] = _write_report(
        dirs["reports"] / "MAIN_EXPERIMENT_COST_REPORT.md",
        source_outputs=sources,
        source_warnings=source_warnings,
        suite_names=suite_names,
        datasets=datasets,
        models=list(requested_models.values()),
        detail=detail,
        paths=paths,
    )
    if bool(getattr(args, "copy_to_final_artifacts", False)):
        paths["final_artifacts_dir"] = _copy_to_final_artifacts(paths, Path(getattr(args, "final_artifacts_dir", "outputs/final_artifacts")))
    return paths


def build_detail_table(
    records: list[dict[str, Any]],
    datasets: list[str],
    suite_names: list[str],
    models: list[str],
    annotation_index: dict[str, dict[str, Any]] | None = None,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    annotation_index = annotation_index or {}
    dataset_stats = dataset_stats or {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("suite", "main")), str(record.get("dataset", "")), str(record.get("model", "")))
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (suite, dataset, model), group in sorted(groups.items(), key=lambda item: (_suite_order(item[0][0], suite_names), datasets.index(item[0][1]) if item[0][1] in datasets else 999, _model_order(item[0][2]), item[0][2])):
        rows.append(_aggregate_group(suite, dataset, model, group, annotation_index.get(dataset, {}), dataset_stats.get(dataset, {})))

    first_suite = suite_names[0] if suite_names else "main"
    existing = {(str(row["dataset"]), str(row["model"])) for row in rows if str(row.get("status")) != "excluded_quick_or_mock"}
    for dataset in datasets:
        for model in models:
            if (dataset, model) in existing:
                continue
            rows.append(_unavailable_cost_row(first_suite, dataset, model, annotation_index.get(dataset, {}), dataset_stats.get(dataset, {})))
            existing.add((dataset, model))

    frame = pd.DataFrame(rows)
    for column in DETAIL_COLUMNS:
        if column not in frame:
            frame[column] = NA
    frame = frame[DETAIL_COLUMNS]
    return frame.fillna(NA).replace("", NA)


def build_compact_table(detail: pd.DataFrame, datasets: list[str] | None = None) -> pd.DataFrame:
    datasets = datasets or DEFAULT_DATASETS
    rows = []
    if detail.empty:
        return pd.DataFrame(columns=COMPACT_COLUMNS)
    subset = detail[detail["dataset"].astype(str).isin(datasets)].copy()
    subset = subset[subset["status"].astype(str) != "excluded_quick_or_mock"]
    subset["_model_order"] = subset["model"].astype(str).map(lambda item: _model_order(item))
    subset["_dataset_order"] = subset["dataset"].astype(str).map(lambda item: datasets.index(item) if item in datasets else 999)
    subset = subset.sort_values(["_dataset_order", "_model_order", "model"], kind="stable")
    for _, row in subset.iterrows():
        model = str(row["model"])
        rows.append(
            {
                "Dataset": _display_dataset(row["dataset"]),
                "Model": model,
                "#Seeds": _display_seed_count(row.get("seed_count")),
                "Train Time / Seed": _mean_std_time(row.get("train_time_sec_mean"), row.get("train_time_sec_std")),
                "Total Runtime": _format_time(row.get("total_runtime_sec_sum")),
                "Peak GPU Mem.": _format_memory(row.get("gpu_memory_mb_max")),
                "LLM Annotation Cost": _compact_annotation_cost(row),
                "Cache Size": _format_cache(row.get("cache_size_mb")) if model == "HERO" else NA,
                "Cost Source": _compact_source(row.get("cost_source")),
                "Status": row.get("status", NA),
            }
        )
    return pd.DataFrame(rows, columns=COMPACT_COLUMNS).fillna(NA).replace("", NA)


def _collect_records_from_root(
    root: Path,
    suite_names: list[str],
    datasets: list[str],
    requested_models: dict[str, str],
    include_quick_test: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_records_from_run_dirs(root))
    records.extend(_records_from_manifests(root))
    records.extend(_records_from_summary(root))
    records.extend(_records_from_logs(root))
    filtered: list[dict[str, Any]] = []
    for record in records:
        record["source_output"] = _source_root_label(root)
        suite = _normalize_suite(record.get("suite", "main"))
        if suite in SUPPLEMENT_SUITES:
            continue
        if suite not in set(suite_names) | QUICK_SUITES:
            continue
        dataset = str(record.get("dataset", ""))
        if datasets and dataset not in datasets:
            continue
        model = _canonical_model(record.get("model_raw", record.get("model")))
        if model not in set(requested_models.values()):
            continue
        record["suite"] = suite
        record["model"] = model
        record["status"] = _normalize_status(record.get("status"))
        record["original_status"] = record["status"]
        record["excluded_reason"] = ""
        if (suite in QUICK_SUITES and not include_quick_test) or _uses_mock_fallback(record):
            record["status"] = "excluded_quick_or_mock"
            record["excluded_reason"] = "quick_test" if suite in QUICK_SUITES and not include_quick_test else "mock_fallback"
        filtered.append(record)
    return filtered


def _records_from_run_dirs(root: Path) -> list[dict[str, Any]]:
    search_roots = [root / "raw", root]
    run_dirs: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for marker_name in ["runtime.json", "metrics.json", "config.json", "skip_reason.json"]:
            for marker in search_root.rglob(marker_name):
                if _skip_path(marker):
                    continue
                run_dirs.add(marker.parent)
    records = []
    for run_dir in sorted(run_dirs):
        record = _record_from_run_dir(root, run_dir, source="runtime_files", priority=1)
        if record is not None:
            records.append(record)
    return records


def _record_from_run_dir(root: Path, run_dir: Path, source: str, priority: int) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    source_files: list[str] = []
    field_sources: dict[str, str] = {}
    for filename in ["config.json", "runtime.json", "metrics.json", "skip_reason.json"]:
        path = run_dir / filename
        data = _read_json(path)
        if not data:
            continue
        rel = _rel_to_root(root, path)
        source_files.append(rel)
        for key, value in data.items():
            if value is None:
                continue
            payload[key] = value
            field_sources[key] = rel
    dataset, model, seed, suite = _infer_identity(root, run_dir, payload)
    if not dataset or not model or seed is None:
        return None
    record = _record_from_payload(root, payload, dataset, model, seed, suite, source, priority)
    record["run_dir"] = _rel_to_root(root, run_dir)
    record["source_files"] = source_files
    record["field_sources"] = field_sources
    return record


def _records_from_manifests(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({root / "run_manifest.json", root / "summary" / "run_manifest.json"}):
        if not path.exists():
            continue
        payload = _read_json_or_list(path)
        rows = _manifest_rows(payload)
        for row in rows:
            run_dir_value = row.get("run_dir") or row.get("output_dir") or row.get("path")
            if run_dir_value:
                run_dir = _resolve_under_root(root, run_dir_value)
                run_record = _record_from_run_dir(root, run_dir, source="run_manifest", priority=2) if run_dir.exists() else None
                if run_record is not None:
                    run_record["source_files"] = _dedupe_texts([*run_record.get("source_files", []), _rel_to_root(root, path)])
                    run_record["record_source"] = "run_manifest+runtime_files"
                    records.append(run_record)
                    continue
            dataset, model, seed, suite = _identity_from_payload_or_empty(row)
            if not dataset or not model or seed is None:
                continue
            record = _record_from_payload(root, row, dataset, model, seed, suite, "run_manifest", 2)
            record["source_files"] = [_rel_to_root(root, path)]
            record["field_sources"] = {key: _rel_to_root(root, path) for key in row}
            records.append(record)
    return records


def _records_from_summary(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted([root / "summary" / "all_raw_runs.csv", root / "tables" / "all_raw_runs.csv"]):
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, keep_default_na=False)
        except Exception:
            continue
        for row in frame.to_dict(orient="records"):
            dataset, model, seed, suite = _identity_from_payload_or_empty(row)
            if not dataset or not model or seed is None:
                continue
            record = _record_from_payload(root, row, dataset, model, seed, suite, "all_raw_runs", 3)
            record["source_files"] = [_rel_to_root(root, path)]
            record["field_sources"] = {key: _rel_to_root(root, path) for key in row}
            records.append(record)
    return records


def _records_from_logs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "logs").rglob("*")) if (root / "logs").exists() else []:
        if not path.is_file() or path.suffix.lower() not in {".log", ".txt"} or _skip_path(path):
            continue
        payload = _parse_log_times(path)
        if not payload:
            continue
        dataset, model, seed, suite = _infer_identity(root, path.parent, payload)
        dataset = dataset or _match_known(path, DEFAULT_DATASETS)
        model = model or _match_model_from_text(str(path))
        if not dataset or not model:
            continue
        record = _record_from_payload(root, payload, dataset, model, seed if seed is not None else 0, suite or "main", "logs", 4)
        record["source_files"] = [_rel_to_root(root, path)]
        record["field_sources"] = {key: _rel_to_root(root, path) for key in payload}
        records.append(record)
    return records


def _record_from_payload(root: Path, payload: dict[str, Any], dataset: str, model: str, seed: int, suite: str, source: str, priority: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_output": _source_root_label(root),
        "suite": _normalize_suite(suite or payload.get("suite", "main")),
        "dataset": str(dataset),
        "model_raw": str(model),
        "model": _canonical_model(model),
        "seed": int(seed),
        "status": _status_from_payload(payload),
        "run_dir": "",
        "source_files": [],
        "field_sources": {},
        "record_source": source,
        "_priority": priority,
    }
    for target, aliases in TIME_ALIASES.items():
        row[target] = _first_float(payload, aliases)
    for target, aliases in DATASET_ALIASES.items():
        row[target] = _first_float(payload, aliases)
    if not _has_value(row.get("num_train")):
        train_pos = _first_float(payload, ["train_num_pos", "positive_count_train"])
        train_neg = _first_float(payload, ["train_num_neg"])
        if train_pos is not None and train_neg is not None:
            row["num_train"] = train_pos + train_neg
    if not _has_value(row.get("num_val")):
        val_pos = _first_float(payload, ["val_num_pos", "positive_count_val"])
        val_neg = _first_float(payload, ["val_num_neg"])
        if val_pos is not None and val_neg is not None:
            row["num_val"] = val_pos + val_neg
    if not _has_value(row.get("num_test")):
        test_pos = _first_float(payload, ["test_num_pos", "positive_count_test"])
        test_neg = _first_float(payload, ["test_num_neg"])
        if test_pos is not None and test_neg is not None:
            row["num_test"] = test_pos + test_neg
    for key in ["annotation_source", "annotation_model", "annotation_deployment"]:
        row[key] = _first_present(payload, [key, key.replace("annotation_", "llm_")], "")
    row["_payload_text"] = json.dumps(_json_ready(payload), ensure_ascii=False).lower()
    return row


def _aggregate_group(suite: str, dataset: str, model: str, group: list[dict[str, Any]], annotation: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    excluded_only = all(str(row.get("status")) == "excluded_quick_or_mock" for row in group)
    usable = [] if excluded_only else [row for row in group if str(row.get("status")) != "excluded_quick_or_mock"]
    successes = [row for row in usable if _is_success(row.get("status"))]
    run_count = len(group if excluded_only else usable)
    failed_count = sum(1 for row in usable if _is_failed(row.get("status")))
    skipped_count = sum(1 for row in usable if _is_skipped(row.get("status")))
    seed_count = len({int(row["seed"]) for row in successes if _safe_int(row.get("seed")) is not None})
    source = _cost_source(group)
    row: dict[str, Any] = {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "seed_count": seed_count,
        "run_count": run_count,
        "success_count": len(successes),
        "failed_count": failed_count,
        "skipped_count": skipped_count,
    }
    for key in ["train_time_sec", "inference_time_sec", "total_runtime_sec", "gpu_memory_mb", "cpu_memory_mb", "annotation_time_sec", "risk_card_time_sec", "evidence_chain_time_sec"]:
        values = _success_values(successes, key)
        if key in {"train_time_sec", "total_runtime_sec"}:
            row[f"{key}_mean"] = _stat_mean(values)
            row[f"{key}_std"] = _stat_std(values, seed_count)
            if key == "train_time_sec":
                row["train_time_sec_min"] = _stat_min(values)
                row["train_time_sec_max"] = _stat_max(values)
            else:
                row["total_runtime_sec_sum"] = _stat_sum(values)
        elif key == "inference_time_sec":
            row["inference_time_sec_mean"] = _stat_mean(values)
            row["inference_time_sec_std"] = _stat_std(values, seed_count)
        elif key == "gpu_memory_mb":
            row["gpu_memory_mb_mean"] = _stat_mean(values)
            row["gpu_memory_mb_max"] = _stat_max(values)
        elif key == "cpu_memory_mb":
            row["cpu_memory_mb_mean"] = _stat_mean(values)
            row["cpu_memory_mb_max"] = _stat_max(values)
        elif key in {"annotation_time_sec", "risk_card_time_sec", "evidence_chain_time_sec"}:
            if _is_hero(model) or any(_has_value(item.get(key)) for item in successes):
                row[f"{key}_mean"] = _stat_mean(values)
                row[f"{key}_sum"] = _stat_sum(values)
            else:
                row[f"{key}_mean"] = NA
                row[f"{key}_sum"] = NA

    if _is_hero(model):
        row["annotation_cards"] = _first_available(annotation.get("annotation_cards"), _first_present(successes, ["annotation_cards"], NA))
        row["annotated_cards"] = _first_available(annotation.get("annotated_cards"), _first_present(successes, ["annotated_cards"], NA))
        row["annotation_coverage"] = _first_available(annotation.get("annotation_coverage"), _coverage(row.get("annotated_cards"), row.get("annotation_cards")))
        row["annotation_source"] = _annotation_source_from_group(annotation, successes)
        row["annotation_model"] = _first_available(annotation.get("annotation_model"), _first_present(successes, ["annotation_model"], UNAVAILABLE))
        row["annotation_deployment"] = _first_available(annotation.get("annotation_deployment"), _first_present(successes, ["annotation_deployment"], UNAVAILABLE))
        row["cache_size_mb"] = _first_available(annotation.get("cache_size_mb"), _first_present(successes, ["cache_size_mb"], NA))
    else:
        row["annotation_cards"] = NA
        row["annotated_cards"] = NA
        row["annotation_coverage"] = NA
        row["annotation_source"] = NA
        row["annotation_model"] = NA
        row["annotation_deployment"] = NA
        row["cache_size_mb"] = NA

    stat_source = _first_success_or_group(successes, usable or group)
    for key in ["num_nodes", "num_edges", "num_features", "num_train", "num_val", "num_test"]:
        row[key] = _first_available(stats.get(key), stat_source.get(key) if stat_source else NA)
    row["cost_source"] = source
    row["missing_fields"] = _missing_fields(row, model)
    row["status"] = _group_status(row, excluded_only)
    row["notes"] = _group_notes(group, row)
    return _finalize_detail_row(row)


def _unavailable_cost_row(suite: str, dataset: str, model: str, annotation: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    row = {column: NA for column in DETAIL_COLUMNS}
    row.update(
        {
            "suite": suite,
            "dataset": dataset,
            "model": model,
            "seed_count": 0,
            "run_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "status": "runtime_unavailable",
            "cost_source": UNAVAILABLE,
            "missing_fields": "train_time_sec,inference_time_sec,total_runtime_sec,gpu_memory_mb,cpu_memory_mb",
            "notes": "No matching main experiment runtime, metrics, manifest, summary, or log entry was found.",
        }
    )
    if _is_hero(model):
        row["annotation_cards"] = _value_or_na(annotation.get("annotation_cards"))
        row["annotated_cards"] = _value_or_na(annotation.get("annotated_cards"))
        row["annotation_coverage"] = _value_or_na(annotation.get("annotation_coverage"))
        row["annotation_source"] = _value_or_na(annotation.get("annotation_source", UNAVAILABLE))
        row["annotation_model"] = _value_or_na(annotation.get("annotation_model", UNAVAILABLE))
        row["annotation_deployment"] = _value_or_na(annotation.get("annotation_deployment", UNAVAILABLE))
        row["cache_size_mb"] = _value_or_na(annotation.get("cache_size_mb"))
    for key in ["num_nodes", "num_edges", "num_features", "num_train", "num_val", "num_test"]:
        row[key] = _value_or_na(stats.get(key))
    return _finalize_detail_row(row)


def _annotation_index(roots: list[Path], datasets: list[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        files = _annotation_files(roots, dataset)
        cards = 0
        annotated = 0
        models: list[str] = []
        deployments: list[str] = []
        source_text: list[str] = []
        size = 0
        for path in files:
            size += path.stat().st_size if path.exists() else 0
            source_text.append(path.name.lower())
            file_count, annotated_count, file_models, file_deployments, file_text = _annotation_file_stats(path)
            cards += file_count
            annotated += annotated_count
            models.extend(file_models)
            deployments.extend(file_deployments)
            source_text.extend(file_text)
        source = _infer_annotation_source(" ".join(source_text), files)
        index[dataset] = {
            "annotation_cards": cards if files else NA,
            "annotated_cards": annotated if files else NA,
            "annotation_coverage": float(annotated / cards) if cards else NA,
            "annotation_source": source,
            "annotation_model": _first_text(models, UNAVAILABLE if files else UNAVAILABLE),
            "annotation_deployment": _first_text(deployments, UNAVAILABLE if files else UNAVAILABLE),
            "cache_size_mb": round(size / (1024.0 * 1024.0), 4) if files else NA,
            "source_files": [_compact_path(path) for path in files],
        }
    return index


def _annotation_files(roots: list[Path], dataset: str) -> list[Path]:
    names = [
        "annotations.jsonl",
        "qwen_annotations.jsonl",
        "cached_llm_annotations.jsonl",
        "mechanism_annotations.jsonl",
        "llm_labels.jsonl",
        "llm_labels_*.jsonl",
    ]
    files: list[Path] = []
    for root in roots:
        for pattern in names:
            files.extend(sorted(root.rglob(pattern)) if root.exists() else [])
    candidates = []
    for path in files:
        text = str(path).lower()
        if _skip_path(path) or ("mock" in text and "qwen" not in text):
            continue
        if dataset in text or _annotation_contains_dataset(path, dataset):
            candidates.append(path)
    return _dedupe_paths(candidates)


def _annotation_file_stats(path: Path) -> tuple[int, int, list[str], list[str], list[str]]:
    cards = 0
    annotated = 0
    models: list[str] = []
    deployments: list[str] = []
    text_signals: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cards += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if any(_has_value(record.get(key)) for key in ["risk_relevance", "confidence", "mechanism", "mechanism_candidate"]):
                    annotated += 1
                for key in ["annotation_model", "llm_model", "model_name", "labeler", "labeler_version"]:
                    if _has_value(record.get(key)):
                        models.append(str(record.get(key)))
                        text_signals.append(str(record.get(key)).lower())
                for key in ["annotation_deployment", "deployment", "backend", "labeler_source"]:
                    if _has_value(record.get(key)):
                        deployments.append(str(record.get(key)))
                        text_signals.append(str(record.get(key)).lower())
    except OSError:
        return 0, 0, [], [], []
    return cards, annotated, models, deployments, text_signals


def _dataset_stats_index(data_root: Path, roots: list[Path], datasets: list[str]) -> dict[str, dict[str, Any]]:
    return {dataset: _dataset_stats(data_root, roots, dataset) for dataset in datasets}


def _dataset_stats(data_root: Path, roots: list[Path], dataset: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    candidates = [
        data_root / "processed" / dataset,
        data_root / dataset / "processed",
        data_root / dataset,
    ]
    for root in roots:
        candidates.extend([root / "processed" / dataset, root / dataset])
    for processed in candidates:
        if not processed.exists():
            continue
        nodes = processed / "nodes.csv"
        edges = processed / "edges.csv"
        features = processed / "features.npz"
        split = processed / "split.json"
        if nodes.exists():
            try:
                frame = pd.read_csv(nodes)
                stats["num_nodes"] = int(len(frame))
                if "split" in frame.columns:
                    counts = frame["split"].astype(str).value_counts()
                    stats.setdefault("num_train", int(counts.get("train", 0)))
                    stats.setdefault("num_val", int(counts.get("val", counts.get("valid", 0))))
                    stats.setdefault("num_test", int(counts.get("test", 0)))
            except Exception:
                pass
        if edges.exists():
            try:
                stats["num_edges"] = int(len(pd.read_csv(edges)))
            except Exception:
                pass
        if features.exists():
            try:
                payload = np.load(features, allow_pickle=True)
                for key in ["features", "x", "numeric_features"]:
                    if key in payload:
                        arr = payload[key]
                        if getattr(arr, "ndim", 0) >= 2:
                            stats["num_features"] = int(arr.shape[1])
                            break
            except Exception:
                pass
        if split.exists():
            try:
                split_payload = json.loads(split.read_text(encoding="utf-8"))
                for key, out_key in [("train", "num_train"), ("val", "num_val"), ("valid", "num_val"), ("test", "num_test")]:
                    if key in split_payload and isinstance(split_payload[key], list):
                        stats[out_key] = int(len(split_payload[key]))
            except Exception:
                pass
        if stats:
            break
    return stats


def _raw_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {column: _raw_cell(record.get(column, NA)) for column in RAW_COLUMNS}
        if isinstance(record.get("source_files"), list):
            row["source_files"] = ";".join(str(item) for item in record.get("source_files", [])) or NA
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = NA
    return frame[RAW_COLUMNS].fillna(NA).replace("", NA)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.fillna(NA).replace("", NA).to_csv(path, index=False)
    return path


def _write_sources_jsonl(path: Path, records: list[dict[str, Any]], warnings: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for warning in warnings:
            handle.write(json.dumps({"status": "warning", "warning": warning}, ensure_ascii=False, sort_keys=True) + "\n")
        for record in records:
            payload = {
                "suite": record.get("suite", NA),
                "dataset": record.get("dataset", NA),
                "model": record.get("model", NA),
                "seed": record.get("seed", NA),
                "status": record.get("status", NA),
                "original_status": record.get("original_status", record.get("status", NA)),
                "excluded_reason": record.get("excluded_reason", ""),
                "source_output": record.get("source_output", NA),
                "source_files": record.get("source_files", []),
                "field_sources": record.get("field_sources", {}),
                "record_source": record.get("record_source", NA),
            }
            handle.write(json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _write_markdown(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.fillna(NA).replace("", NA)
    lines = ["| " + " | ".join(str(col) for col in table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[col]) for col in table.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_latex(path: Path, frame: pd.DataFrame, caption: str, label: str, compact: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dataframe_to_latex(frame, caption=caption, label=label, compact=compact), encoding="utf-8")
    return path


def _dataframe_to_latex(frame: pd.DataFrame, caption: str, label: str, compact: bool) -> str:
    table = frame.fillna(NA).replace("", NA).copy()
    if compact:
        colspec = "p{0.105\\textwidth}p{0.070\\textwidth}p{0.045\\textwidth}p{0.105\\textwidth}p{0.090\\textwidth}p{0.085\\textwidth}p{0.115\\textwidth}p{0.080\\textwidth}p{0.085\\textwidth}p{0.070\\textwidth}"
        size = "\\scriptsize"
        word_limit = 10
    else:
        display_columns = [
            "suite",
            "dataset",
            "model",
            "seed_count",
            "run_count",
            "train_time_sec_mean",
            "train_time_sec_std",
            "total_runtime_sec_sum",
            "gpu_memory_mb_max",
            "annotation_source",
            "cache_size_mb",
            "status",
        ]
        table = table[[column for column in display_columns if column in table.columns]]
        colspec = "p{0.075\\textwidth}p{0.095\\textwidth}p{0.075\\textwidth}p{0.050\\textwidth}p{0.050\\textwidth}p{0.085\\textwidth}p{0.080\\textwidth}p{0.085\\textwidth}p{0.085\\textwidth}p{0.110\\textwidth}p{0.070\\textwidth}p{0.100\\textwidth}"
        size = "\\scriptsize"
        word_limit = 9
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        size,
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(_latex_escape(str(col)) for col in table.columns) + r" \\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        cells = [_latex_escape(_truncate(row[col], word_limit)) for col in table.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _write_report(
    path: Path,
    source_outputs: list[Path],
    source_warnings: list[str],
    suite_names: list[str],
    datasets: list[str],
    models: list[str],
    detail: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Main Experiment Cost Report",
        "",
        INTEGRITY_STATEMENT,
        "",
        "## Scope",
        "",
        f"- source_outputs: {', '.join(_path_for_report(path) for path in source_outputs) or UNAVAILABLE}",
        f"- suite_names: {', '.join(suite_names)}",
        f"- datasets: {', '.join(datasets)}",
        f"- models: {', '.join(models)}",
        "",
    ]
    if source_warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in source_warnings)
        lines.append("")
    lines.extend(["## Artifact Paths", ""])
    for key in ["raw_csv", "sources_jsonl", "detail_csv", "compact_csv", "detail_latex", "compact_latex", "detail_markdown", "compact_markdown"]:
        if key in paths:
            lines.append(f"- {key}: `{_rel_path(paths[key])}`")
    lines.extend(["", "## Run Counts", ""])
    if detail.empty:
        lines.append("- unavailable: no cost rows generated")
    else:
        for _, row in detail.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['model']}: success={row['success_count']}, failed={row['failed_count']}, skipped={row['skipped_count']}, status={row['status']}"
            )
    lines.extend(["", "## Field Availability", ""])
    available = []
    missing = []
    for column in DETAIL_COLUMNS:
        if column in detail.columns and detail[column].astype(str).map(_has_value).any():
            available.append(column)
        else:
            missing.append(column)
    lines.append(f"- available_fields: {', '.join(available) or UNAVAILABLE}")
    lines.append(f"- missing_fields: {', '.join(missing) or NA}")
    lines.extend(["", "## HERO Annotation Cache", ""])
    hero = detail[detail["model"].astype(str) == "HERO"] if not detail.empty else pd.DataFrame()
    if hero.empty:
        lines.append("- HERO rows unavailable.")
    else:
        for _, row in hero.iterrows():
            lines.append(
                f"- {row['dataset']}: annotation_source={row['annotation_source']}, annotation_cards={row['annotation_cards']}, annotated_cards={row['annotated_cards']}, cache_size_mb={row['cache_size_mb']}"
            )
    lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _copy_to_final_artifacts(paths: dict[str, Path], final_dir: Path) -> Path:
    mapping = {
        "detail_csv": final_dir / "tables_csv" / "supp_table_main_experiment_cost.csv",
        "compact_csv": final_dir / "tables_csv" / "supp_table_main_experiment_cost_compact.csv",
        "detail_latex": final_dir / "tables_latex" / "supp_table_main_experiment_cost.tex",
        "compact_latex": final_dir / "tables_latex" / "supp_table_main_experiment_cost_compact.tex",
        "report": final_dir / "reports" / "MAIN_EXPERIMENT_COST_REPORT.md",
    }
    for key, target in mapping.items():
        source = paths.get(key)
        if not isinstance(source, Path) or not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return final_dir


def _ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "raw": output_dir / "raw",
        "tables_csv": output_dir / "tables_csv",
        "tables_latex": output_dir / "tables_latex",
        "tables_markdown": output_dir / "tables_markdown",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _requested_models(models: Iterable[str]) -> dict[str, str]:
    result = {}
    for model in models:
        result[str(model)] = _canonical_model(model)
    return result


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        seed = _safe_int(record.get("seed"))
        if seed is None:
            continue
        key = (str(record.get("source_output", "")), str(record.get("suite", "")), str(record.get("dataset", "")), str(record.get("model", "")), seed)
        grouped.setdefault(key, []).append(record)
    merged = []
    for group in grouped.values():
        group = sorted(group, key=lambda item: int(item.get("_priority", 99)))
        base = dict(group[0])
        base["source_files"] = []
        base["field_sources"] = {}
        record_sources = []
        for record in group:
            record_sources.append(str(record.get("record_source", "")))
            base["source_files"] = _dedupe_texts([*base.get("source_files", []), *record.get("source_files", [])])
            base["field_sources"] = {**record.get("field_sources", {}), **base.get("field_sources", {})}
            for key, value in record.items():
                if key.startswith("_"):
                    continue
                if not _has_value(base.get(key)) and _has_value(value):
                    base[key] = value
        base["record_source"] = "+".join(_dedupe_texts(record_sources))
        merged.append(base)
    return merged


def _skip_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & {"_project_runs", "final_artifacts", "main_cost", "risk_card_cases"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_or_list(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ["runs", "records", "results", "expected_runs"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _infer_identity(root: Path, run_dir: Path, payload: dict[str, Any]) -> tuple[str, str, int | None, str]:
    dataset, model, seed, suite = _identity_from_payload_or_empty(payload)
    if dataset and model and seed is not None:
        return dataset, model, seed, suite
    try:
        parts = run_dir.relative_to(root / "raw").parts if (root / "raw") in run_dir.parents or run_dir == root / "raw" else run_dir.relative_to(root).parts
    except ValueError:
        parts = run_dir.parts
    if parts and parts[0] in set(DEFAULT_SUITES) | QUICK_SUITES | SUPPLEMENT_SUITES | {"main"}:
        suite = suite or parts[0]
        parts = parts[1:]
    if len(parts) >= 3:
        dataset = dataset or parts[0]
        model = model or parts[1]
        seed = seed if seed is not None else _seed_from_text(parts[2])
    return str(dataset or ""), str(model or ""), seed, str(suite or "main")


def _identity_from_payload_or_empty(payload: dict[str, Any]) -> tuple[str, str, int | None, str]:
    dataset = _safe_str(payload.get("dataset"))
    model = _safe_str(payload.get("model", payload.get("suite_model", payload.get("variant"))))
    seed = _safe_int(payload.get("seed"))
    suite = _safe_str(payload.get("suite")) or "main"
    return dataset, model, seed, suite


def _seed_from_text(text: Any) -> int | None:
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else None


def _status_from_payload(payload: dict[str, Any]) -> str:
    status = _normalize_status(payload.get("status"))
    if status and status != "unknown":
        return status
    if payload.get("skip_reason") or payload.get("reason"):
        return "skipped"
    if any(_has_value(payload.get(key)) for key in ["Macro-F1", "macro_f1", "AUROC", "auroc", "AUPRC", "auprc", "time_total_sec", "total_runtime_sec"]):
        return "ok"
    return "unknown"


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if text in {"exist"}:
        return "exists"
    if text in {"skipped", "skip", "missing"}:
        return "skipped"
    return text


def _normalize_suite(value: Any) -> str:
    text = str(value or "main").strip()
    return text or "main"


def _canonical_model(value: Any) -> str:
    text = str(value or "").strip()
    key = text.lower().replace("-", "_")
    return MODEL_ALIASES.get(key, MODEL_ALIASES.get(text.lower(), text))


def _is_hero(model: Any) -> bool:
    return _canonical_model(model) == "HERO"


def _uses_mock_fallback(record: dict[str, Any]) -> bool:
    text = str(record.get("_payload_text", "")).lower()
    if any(signal in text for signal in ['"use_mock_llm_mechanism": true', '"mock_fallback": true', '"llm_labeler": "mock"', '"labeler_source": "mock"']):
        return True
    return False


def _parse_log_times(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    payload: dict[str, Any] = {}
    patterns = {
        "train_time_sec": [r"(?:time_training_sec|train_time_sec|training)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "inference_time_sec": [r"(?:time_inference_sec|inference_time_sec|inference)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "total_runtime_sec": [r"(?:time_total_sec|total_runtime_sec|total)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "gpu_memory_mb": [r"(?:peak_gpu_memory_mb|gpu_memory_mb|max_gpu_memory_mb)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "cpu_memory_mb": [r"(?:peak_cpu_memory_mb|cpu_memory_mb|max_rss_mb)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "annotation_time_sec": [r"(?:annotation_time_sec|time_annotation_sec|mock_labeling)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
        "evidence_chain_time_sec": [r"(?:evidence_chain_time_sec|time_evidence_chain_sec)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"],
    }
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, flags=re.IGNORECASE)
            if match:
                payload[key] = float(match.group(1))
                break
    return payload


def _match_known(path: Path, candidates: list[str]) -> str:
    text = str(path).lower()
    for candidate in candidates:
        if candidate.lower() in text:
            return candidate
    return ""


def _match_model_from_text(text: str) -> str:
    lower = text.lower()
    for alias, display in MODEL_ALIASES.items():
        if alias in lower:
            return display
    return ""


def _resolve_under_root(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _first_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        value = safe_float(payload[key], default=None)
        if value is not None:
            return value
    return None


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or value is pd.NA:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"n/a", "na", "nan", "none", "unavailable", "<na>"}:
            return default
        try:
            number = float(text)
        except ValueError:
            return default
        return number if math.isfinite(number) else default
    if isinstance(value, (list, tuple, set, np.ndarray)):
        arr = []
        for item in list(value):
            number = safe_float(item, default=None)
            if number is not None:
                arr.append(number)
        return float(np.mean(arr)) if arr else default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _success_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = safe_float(record.get(key), default=None)
        if value is not None:
            values.append(value)
    return values


def _stat_mean(values: list[float]) -> Any:
    return round(float(np.mean(values)), 2) if values else NA


def _stat_std(values: list[float], seed_count: int) -> Any:
    return round(float(np.std(values, ddof=1)), 2) if len(values) >= 2 and seed_count >= 2 else NA


def _stat_min(values: list[float]) -> Any:
    return round(float(np.min(values)), 2) if values else NA


def _stat_max(values: list[float]) -> Any:
    return round(float(np.max(values)), 2) if values else NA


def _stat_sum(values: list[float]) -> Any:
    return round(float(np.sum(values)), 2) if values else NA


def _coverage(annotated: Any, cards: Any) -> Any:
    a = safe_float(annotated, default=None)
    c = safe_float(cards, default=None)
    return round(float(a / c), 4) if a is not None and c and c > 0 else NA


def _group_status(row: dict[str, Any], excluded_only: bool) -> str:
    if excluded_only:
        return "excluded_quick_or_mock"
    if int(row.get("success_count", 0)) == 0 and (int(row.get("failed_count", 0)) > 0 or int(row.get("skipped_count", 0)) > 0):
        return "failed_or_skipped"
    if int(row.get("success_count", 0)) == 0:
        return "runtime_unavailable"
    if not _has_value(row.get("train_time_sec_mean")) and not _has_value(row.get("total_runtime_sec_mean")):
        return "runtime_unavailable"
    return "ok"


def _group_notes(group: list[dict[str, Any]], row: dict[str, Any]) -> str:
    excluded = [item for item in group if str(item.get("status")) == "excluded_quick_or_mock"]
    notes = []
    if excluded:
        reasons = sorted({str(item.get("excluded_reason", "excluded")) for item in excluded})
        notes.append(f"excluded_runs={len(excluded)}:{'/'.join(reasons)}")
    if str(row.get("status")) == "runtime_unavailable":
        notes.append("runtime_unavailable")
    return "; ".join(notes) if notes else NA


def _missing_fields(row: dict[str, Any], model: str) -> str:
    fields = [
        "train_time_sec_mean",
        "inference_time_sec_mean",
        "total_runtime_sec_mean",
        "gpu_memory_mb_mean",
        "cpu_memory_mb_mean",
    ]
    if _is_hero(model):
        fields.extend(["annotation_time_sec_mean", "risk_card_time_sec_mean", "evidence_chain_time_sec_mean", "annotation_source", "cache_size_mb"])
    missing = [field for field in fields if not _has_value(row.get(field))]
    return ",".join(missing) if missing else NA


def _cost_source(group: list[dict[str, Any]]) -> str:
    source_types = sorted({str(item.get("record_source", "")) for item in group if _has_value(item.get("record_source"))})
    source_files = []
    for item in group:
        source_files.extend(item.get("source_files", []) if isinstance(item.get("source_files"), list) else [])
    compact_files = [_truncate(path, 4) for path in _dedupe_texts(source_files)[:4]]
    pieces = source_types or [UNAVAILABLE]
    if compact_files:
        pieces.append("files=" + ";".join(compact_files))
    return _truncate("; ".join(pieces), 24)


def _annotation_source_from_group(annotation: dict[str, Any], successes: list[dict[str, Any]]) -> str:
    values = [annotation.get("annotation_source")] + [row.get("annotation_source") for row in successes]
    text = " ".join(str(value).lower() for value in values if _has_value(value))
    if "qwen" in text or "local_qwen" in text:
        return "cached_local_qwen"
    if "proxy" in text:
        return "proxy"
    if _has_value(annotation.get("annotation_source")):
        return str(annotation.get("annotation_source"))
    if any(_has_value(row.get("annotation_source")) for row in successes):
        return "cached_annotation"
    return UNAVAILABLE


def _infer_annotation_source(text: str, files: list[Path]) -> str:
    lower = " ".join([text, " ".join(path.name.lower() for path in files)])
    if not files:
        return UNAVAILABLE
    if "qwen" in lower or "local_qwen" in lower:
        return "cached_local_qwen"
    if "proxy" in lower:
        return "proxy"
    return "cached_annotation"


def _attach_annotation_info(record: dict[str, Any], annotation: dict[str, Any]) -> None:
    for key in ["annotation_cards", "annotated_cards", "annotation_coverage", "annotation_source", "annotation_model", "annotation_deployment", "cache_size_mb"]:
        if not _has_value(record.get(key)) and _has_value(annotation.get(key)):
            record[key] = annotation.get(key)


def _attach_dataset_stats(record: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        if not _has_value(record.get(key)):
            record[key] = value


def _first_success_or_group(successes: list[dict[str, Any]], group: list[dict[str, Any]]) -> dict[str, Any]:
    return (successes or group or [{}])[0]


def _first_available(*values: Any) -> Any:
    for value in values:
        if _has_value(value):
            return value
    return NA


def _first_present(payload: Any, keys: list[str], default: Any = NA) -> Any:
    if isinstance(payload, list):
        for row in payload:
            value = _first_present(row, keys, default=None)
            if _has_value(value):
                return value
        return default
    if isinstance(payload, dict):
        for key in keys:
            if _has_value(payload.get(key)):
                return payload.get(key)
    return default


def _finalize_detail_row(row: dict[str, Any]) -> dict[str, Any]:
    finalized = {}
    for column in DETAIL_COLUMNS:
        finalized[column] = _value_or_na(row.get(column, NA))
    return finalized


def _raw_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return _value_or_na(value)


def _value_or_na(value: Any) -> Any:
    if str(value).strip().lower() == UNAVAILABLE:
        return UNAVAILABLE
    if not _has_value(value):
        return NA
    if isinstance(value, float):
        return round(value, 4)
    return value


def _has_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"n/a", "na", "nan", "none", "<na>", "unavailable"}


def _is_success(status: Any) -> bool:
    return str(status or "").lower() in SUCCESS_STATUSES


def _is_failed(status: Any) -> bool:
    return str(status or "").lower() in FAILED_STATUSES


def _is_skipped(status: Any) -> bool:
    return str(status or "").lower() in SKIPPED_STATUSES


def _has_runtime_unavailable(frame: pd.DataFrame) -> bool:
    return not frame.empty and frame["status"].astype(str).eq("runtime_unavailable").any()


def _compact_annotation_cost(row: pd.Series) -> str:
    if str(row.get("Model", row.get("model", ""))) != "HERO" and str(row.get("model", "")) != "HERO":
        return NA
    source = str(row.get("annotation_source", UNAVAILABLE))
    if source == "cached_local_qwen":
        return "local_qwen, offline"
    if source == "cached_annotation":
        return "cached, offline"
    if source == "proxy":
        return "proxy, offline"
    if source == UNAVAILABLE:
        return UNAVAILABLE
    return "cached, offline"


def _mean_std_time(mean: Any, std: Any) -> str:
    if not _has_value(mean):
        return NA
    mean_text = _format_time(mean)
    if not _has_value(std):
        return f"{mean_text} ± N/A"
    return f"{mean_text} ± {_format_time(std)}"


def _format_time(value: Any) -> str:
    number = safe_float(value, default=None)
    if number is None:
        return NA
    if number < 60:
        return f"{number:.1f}s"
    if number < 3600:
        return f"{number / 60.0:.1f} min"
    return f"{number / 3600.0:.1f} h"


def _format_memory(value: Any) -> str:
    number = safe_float(value, default=None)
    if number is None:
        return NA
    if number >= 1024:
        return f"{number / 1024.0:.2f} GB"
    return f"{number:.0f} MB"


def _format_cache(value: Any) -> str:
    number = safe_float(value, default=None)
    if number is None:
        return NA
    return f"{number:.2f} MB"


def _display_seed_count(value: Any) -> Any:
    number = _safe_int(value)
    return number if number is not None else NA


def _compact_source(value: Any) -> str:
    if not _has_value(value):
        return UNAVAILABLE
    text = str(value)
    if "runtime_files" in text:
        return "runtime/metrics"
    if "all_raw_runs" in text:
        return "summary"
    if "run_manifest" in text:
        return "manifest"
    if "logs" in text:
        return "logs"
    return _truncate(text, 4)


def _display_dataset(value: Any) -> str:
    text = str(value)
    return {
        "yelp_academic": "Yelp Academic",
        "amazon_video": "Amazon Video",
    }.get(text, text)


def _suite_order(suite: str, suite_names: list[str]) -> int:
    return suite_names.index(suite) if suite in suite_names else 999


def _model_order(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else 999


def _safe_str(value: Any) -> str:
    return "" if not _has_value(value) else str(value)


def _safe_int(value: Any) -> int | None:
    try:
        if not _has_value(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _annotation_contains_dataset(path: Path, dataset: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _, line in zip(range(10), handle):
                if dataset in line:
                    return True
    except OSError:
        return False
    return False


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _dedupe_texts(items: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if _has_value(item)))


def _source_root_label(root: Path) -> str:
    return root.name or str(root)


def _rel_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def _compact_path(path: Path) -> str:
    parts = path.parts[-4:]
    return "/".join(parts)


def _path_for_report(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _first_text(items: list[str], default: str) -> str:
    for item in items:
        if _has_value(item):
            return str(item)
    return default


def _markdown_cell(value: Any) -> str:
    return _truncate(value, 24).replace("|", "\\|")


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _truncate(value: Any, max_words: int) -> str:
    text = str(_value_or_na(value)).replace("\n", " ").strip()
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    keep = max(1, max_words - 1)
    return " ".join(words[:keep]) + " ..."


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
