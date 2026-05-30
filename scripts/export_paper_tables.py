from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _mean_std_columns(metrics: list[str]) -> list[str]:
    columns: list[str] = []
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    return columns


PROXY_DATASETS = {"yelp_academic", "amazon_video"}
OFFICIAL_DATASETS = {"fraud_yelp_official", "fraud_amazon_official"}
PROXY_MAIN_METHODS = ["mlp", "graphsage", "semsim_gnn", "rulehetero_gnn", "hero_gnn"]
PROXY_ABLATIONS = ["hero_gnn", "hero_wo_hetero", "hero_wo_mechanism", "hero_wo_chain"]
OFFICIAL_MAIN_METHODS = ["mlp", "graphsage", "dga_gnn_lite", "hero_official"]
OFFICIAL_ABLATIONS = [
    "hero_official",
    "hero_official_wo_hetero",
    "hero_official_wo_relation",
    "hero_official_wo_feature_deviation",
]
FORMAL_FEASIBILITY_TAG = "qwen2p5_7b_500"
FORMAL_FEASIBILITY_DATASETS = {
    "yelp_academic": "summary_llm_yelp",
    "amazon_video": "summary_llm_amazon",
}
FORMAL_HIGHCOV_TAGS = ["mock_highcov_300x20", "qwen2p5_7b_highcov_300x20"]
SMOKE_PATTERNS = ["mock_test", "mock_highcov_test", "risk_cards_test", "llm_labels_mock_test", "test_strict"]

METRICS = ["macro_f1", "auroc", "auprc"]
PROXY_EXTRA_METRICS = ["evidence_recall_proxy", "evidence_necessity", "evidence_necessity_gap"]
OFFICIAL_EXTRA_METRICS = [
    "avg_selected_neighbors",
    "avg_official_hetero_gate",
    "official_avg_feature_distance_selected",
    "official_avg_homo_similarity_selected",
    "official_avg_relation_rarity",
]
LLM_EXTRA_METRICS = ["evidence_recall_proxy", "evidence_necessity_gap"]
CONTROL_COLUMNS = ["coverage", "parse_error_count", "risk_relevance_rate"]

MAIN_PROXY_COLUMNS = (
    ["dataset", "method"]
    + _mean_std_columns(METRICS + ["evidence_recall_proxy"])
    + CONTROL_COLUMNS
)
ABLATION_PROXY_COLUMNS = (
    ["dataset", "variant"]
    + _mean_std_columns(METRICS + PROXY_EXTRA_METRICS)
    + CONTROL_COLUMNS
)
OFFICIAL_BENCHMARK_COLUMNS = (
    ["dataset", "method"]
    + _mean_std_columns(METRICS + OFFICIAL_EXTRA_METRICS)
    + CONTROL_COLUMNS
)
OFFICIAL_ABLATION_COLUMNS = (
    ["dataset", "variant"]
    + _mean_std_columns(METRICS + OFFICIAL_EXTRA_METRICS)
    + CONTROL_COLUMNS
)
REAL_LLM_FEASIBILITY_COLUMNS = (
    [
        "dataset",
        "labeler",
        "experiment_tag",
        "num_cards",
        "num_risk_cards",
        "parse_error_count",
        "fallback_count",
        "risk_relevance_rate",
        "avg_confidence",
        "agreement_with_mock",
        "mechanism_distribution",
        "coverage",
        "llm_label_coverage_rate",
    ]
    + _mean_std_columns(METRICS + LLM_EXTRA_METRICS)
)
REAL_LLM_HIGHCOV_COLUMNS = (
    [
        "dataset",
        "labeler",
        "experiment_tag",
        "num_eval_target_nodes",
        "num_cards",
        "num_risk_cards",
        "coverage",
        "llm_label_coverage_rate",
        "parse_error_count",
        "risk_relevance_rate",
        "avg_confidence",
        "mechanism_distribution",
    ]
    + _mean_std_columns(METRICS + LLM_EXTRA_METRICS)
)
CASE_STUDY_COLUMNS = [
    "case_id",
    "target_id",
    "neighbor_id",
    "mechanism",
    "llm_decision",
    "confidence",
    "evidence_pattern",
    "interpretation",
]

LEGACY_MAIN_COLUMNS = ["dataset", "method", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]
LEGACY_ABLATION_COLUMNS = [
    "dataset",
    "variant",
    "macro_f1_mean",
    "auroc_mean",
    "auprc_mean",
    "evidence_recall_proxy",
    "evidence_necessity",
    "evidence_necessity_gap",
]
LEGACY_NEIGHBOR_COLUMNS = ["dataset", "strategy", "auprc", "auroc", "macro_f1", "avg_selected_neighbors", "evidence_recall_proxy"]
LEGACY_SIGNIFICANCE_COLUMNS = ["dataset", "metric", "hero_mean", "baseline_name", "baseline_mean", "delta", "p_value", "test_type"]
LEGACY_RUNTIME_COLUMNS = ["dataset", "method", "time_total_mean", "time_total_std", "time_retrieval_mean", "time_training_mean"]
LEGACY_LABELER_COLUMNS = [
    "dataset",
    "labeler",
    "num_cards",
    "risk_relevance_rate",
    "mechanism_distribution",
    "avg_confidence",
    "agreement_with_mock",
    "macro_f1",
    "auroc",
    "auprc",
    "evidence_necessity_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final paper-ready CSV and Markdown tables.")
    parser.add_argument("--summary_dir", default="outputs/summary", help="Directory containing proxy/official summary CSV files.")
    parser.add_argument("--out_dir", default="outputs/paper_tables", help="Paper table output directory.")
    parser.add_argument("--outputs_root", default="outputs", help="Primary outputs root to scan.")
    parser.add_argument("--archive_root", default="archive", help="Archived outputs root to scan when present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_paper_tables(
        Path(args.summary_dir),
        Path(args.out_dir),
        outputs_root=Path(args.outputs_root),
        archive_root=Path(args.archive_root),
    )


def export_paper_tables(
    summary_dir: Path,
    out_dir: Path,
    outputs_root: Path | None = None,
    archive_root: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_roots = _source_roots(summary_dir, outputs_root, archive_root)

    result_rows = _read_metric_rows(source_roots, "results")
    llm_rows = _read_metric_rows(source_roots, "results_llm_comparison")
    formal_llm_rows = _read_metric_rows_from_relatives(
        source_roots,
        [
            "results",
            "results_qwen2p5_7b_500",
            "results_llm_comparison",
            "archive_real_llm_qwen_500_final/results",
            "archive_real_llm_qwen_500_final/results_qwen2p5_7b_500",
            "archive_real_llm_qwen_500_final/results_llm_comparison",
        ],
    )

    tables = {
        "table_main_proxy": _build_main_proxy(result_rows, source_roots, summary_dir),
        "table_ablation_proxy": _build_ablation_proxy(result_rows, source_roots, summary_dir),
        "table_official_benchmark": _build_official_benchmark(result_rows, source_roots, summary_dir),
        "table_official_ablation": _build_official_ablation(result_rows, source_roots, summary_dir),
        "table_real_llm_feasibility": _build_real_llm_feasibility(formal_llm_rows, source_roots),
        "table_real_llm_highcov": _build_real_llm_highcov(llm_rows, source_roots),
        "table_case_study": _build_case_study(source_roots),
    }
    for stem, frame in tables.items():
        _write_table(frame, out_dir / f"{stem}.csv")

    _write_readme(out_dir, source_roots)
    _write_legacy_tables(summary_dir, out_dir, tables)
    print(f"Wrote paper tables to {out_dir}")
    _print_self_checks(tables)


def _source_roots(summary_dir: Path, outputs_root: Path | None, archive_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if outputs_root is not None:
        roots.append(outputs_root)
    elif summary_dir.name == "summary":
        roots.append(summary_dir.parent)
    else:
        roots.append(Path("outputs"))
    if archive_root is not None:
        roots.append(archive_root)
    elif Path("archive").exists():
        roots.append(Path("archive"))
    return _dedupe_paths(roots)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _candidate_dir(root: Path, relative: str) -> list[Path]:
    return _dedupe_paths([root / relative, root / "outputs" / relative])


def _candidate_files(source_roots: list[Path], relative: str) -> list[Path]:
    paths: list[Path] = []
    for root in source_roots:
        paths.extend(_candidate_dir(root, str(Path(relative).parent)))
    name = Path(relative).name
    return _dedupe_paths([path / name for path in paths])


def _read_metric_rows(source_roots: list[Path], result_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if result_kind == "results_llm_comparison":
        pattern = "*/*/*/seed_*/metrics.json"
    else:
        pattern = "*/*/seed_*/metrics.json"
    for root in source_roots:
        for result_dir in _candidate_dir(root, result_kind):
            if not result_dir.exists():
                continue
            for path in sorted(result_dir.glob(pattern)):
                payload = _read_json(path)
                if not payload:
                    continue
                payload["_source_file"] = str(path)
                rows.append(payload)
    return rows


def _read_metric_rows_from_relatives(source_roots: list[Path], relatives: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in source_roots:
        for relative in relatives:
            for result_dir in _candidate_dir(root, relative):
                if not result_dir.exists():
                    continue
                for path in sorted(result_dir.glob("**/metrics.json")):
                    key = str(path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    payload = _read_json(path)
                    if not payload:
                        continue
                    payload["_source_file"] = str(path)
                    rows.append(payload)
    return rows


def _build_main_proxy(rows: list[dict[str, Any]], source_roots: list[Path], summary_dir: Path) -> pd.DataFrame:
    allowed = set(PROXY_MAIN_METHODS)
    raw = _aggregate_metric_rows(
        [
            row
            for row in rows
            if _is_proxy_row(row)
            and str(row.get("model", row.get("method", ""))) in allowed
            and not _is_llm_or_highcov_row(row)
        ],
        ["dataset", "method"],
        METRICS + ["evidence_recall_proxy"],
    )
    fallback = _summary_main(source_roots, summary_dir, PROXY_DATASETS, MAIN_PROXY_COLUMNS)
    fallback = _filter_methods(fallback, "method", allowed)
    table = _prefer_first(raw, fallback, ["dataset", "method"])
    return _finalize(table, MAIN_PROXY_COLUMNS)


def _build_ablation_proxy(rows: list[dict[str, Any]], source_roots: list[Path], summary_dir: Path) -> pd.DataFrame:
    allowed = set(PROXY_ABLATIONS)
    raw_rows = [
        row
        for row in rows
        if _is_proxy_row(row) and str(row.get("model", row.get("method", ""))) in allowed and not _is_llm_or_highcov_row(row)
    ]
    raw = _aggregate_metric_rows(raw_rows, ["dataset", "variant"], METRICS + PROXY_EXTRA_METRICS, variant_from_model=True)
    fallback = _summary_ablation(source_roots, summary_dir, PROXY_DATASETS, ABLATION_PROXY_COLUMNS)
    fallback = _filter_methods(fallback, "variant", allowed)
    table = _prefer_first(raw, fallback, ["dataset", "variant"])
    return _finalize(table, ABLATION_PROXY_COLUMNS)


def _build_official_benchmark(rows: list[dict[str, Any]], source_roots: list[Path], summary_dir: Path) -> pd.DataFrame:
    allowed = set(OFFICIAL_MAIN_METHODS)
    raw = _aggregate_metric_rows(
        [
            row
            for row in rows
            if str(row.get("dataset", "")) in OFFICIAL_DATASETS
            and str(row.get("model", row.get("method", ""))) in allowed
        ],
        ["dataset", "method"],
        METRICS + OFFICIAL_EXTRA_METRICS,
    )
    fallback = _summary_main(source_roots, summary_dir, OFFICIAL_DATASETS, OFFICIAL_BENCHMARK_COLUMNS)
    fallback = _filter_methods(fallback, "method", allowed)
    table = _prefer_first(raw, fallback, ["dataset", "method"])
    return _finalize(table, OFFICIAL_BENCHMARK_COLUMNS)


def _build_official_ablation(rows: list[dict[str, Any]], source_roots: list[Path], summary_dir: Path) -> pd.DataFrame:
    allowed = set(OFFICIAL_ABLATIONS)
    raw_rows = [
        row
        for row in rows
        if str(row.get("dataset", "")) in OFFICIAL_DATASETS
        and str(row.get("model", row.get("method", ""))) in allowed
    ]
    raw = _aggregate_metric_rows(raw_rows, ["dataset", "variant"], METRICS + OFFICIAL_EXTRA_METRICS, variant_from_model=True)
    fallback = _summary_ablation(source_roots, summary_dir, OFFICIAL_DATASETS, OFFICIAL_ABLATION_COLUMNS)
    fallback_main = _summary_main(source_roots, summary_dir, OFFICIAL_DATASETS, OFFICIAL_BENCHMARK_COLUMNS)
    if not fallback_main.empty and "method" in fallback_main.columns:
        fallback_main = fallback_main.rename(columns={"method": "variant"})
        fallback_main = _finalize(fallback_main, OFFICIAL_ABLATION_COLUMNS)
    fallback = _concat([fallback, fallback_main])
    fallback = _filter_methods(fallback, "variant", allowed)
    table = _prefer_first(raw, fallback, ["dataset", "variant"])
    return _finalize(table, OFFICIAL_ABLATION_COLUMNS)


def _build_real_llm_feasibility(rows: list[dict[str, Any]], source_roots: list[Path]) -> pd.DataFrame:
    table_rows: list[dict[str, Any]] = []
    for dataset, summary_dir_name in FORMAL_FEASIBILITY_DATASETS.items():
        row = _empty_feasibility_row(dataset)
        _merge_into(row, _feasibility_summary_values(source_roots, dataset, summary_dir_name))
        report, label_file = _feasibility_report_and_label(source_roots, dataset)
        _merge_into(row, _report_values(report))
        if label_file is not None:
            _merge_into(row, _label_file_stats(label_file))
        metric_rows = [
            metric
            for metric in rows
            if _matches_formal_feasibility_metric(metric, dataset)
        ]
        _merge_into(row, _metric_summary_values(metric_rows, METRICS + LLM_EXTRA_METRICS))
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    for metric in METRICS + LLM_EXTRA_METRICS:
        _add_mean_std_display(table, metric)
    return _finalize(table, REAL_LLM_FEASIBILITY_COLUMNS)


def _build_real_llm_highcov(rows: list[dict[str, Any]], source_roots: list[Path]) -> pd.DataFrame:
    table_rows: list[dict[str, Any]] = []
    for tag in FORMAL_HIGHCOV_TAGS:
        row = _empty_highcov_row(tag)
        for values in _highcov_summary_values(source_roots, tag):
            _merge_into(row, values)
        metric_rows = [metric for metric in rows if _matches_highcov_metric(metric, tag)]
        _merge_into(row, _metric_summary_values(metric_rows, METRICS + LLM_EXTRA_METRICS))
        if not row["labeler"]:
            row["labeler"] = _infer_highcov_labeler(tag)
        table_rows.append(row)
    table = pd.DataFrame(table_rows).drop_duplicates(subset=["experiment_tag"], keep="first")
    for metric in METRICS + LLM_EXTRA_METRICS:
        _add_mean_std_display(table, metric)
    return _finalize(table, REAL_LLM_HIGHCOV_COLUMNS)


def _build_case_study(source_roots: list[Path]) -> pd.DataFrame:
    candidates: list[Path] = []
    candidates.extend(_candidate_files(source_roots, "summary_llm_highcov/final_case_study_table.csv"))
    candidates.extend(_candidate_files(source_roots, "archive_case_study_final/final_case_study_table.csv"))
    candidates.append(Path("outputs/archive_case_study_final/final_case_study_table.csv"))
    for path in _dedupe_paths(candidates):
        if path.exists():
            frame = pd.read_csv(path, dtype=str).fillna("")
            return _finalize_case_study(frame)
    return _finalize_case_study(pd.DataFrame(columns=CASE_STUDY_COLUMNS))


def _finalize_case_study(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CASE_STUDY_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result = result[CASE_STUDY_COLUMNS]
    for column in result.columns:
        if column == "confidence":
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        else:
            result[column] = result[column].fillna("").astype(str)
    return result


def _empty_feasibility_row(dataset: str) -> dict[str, Any]:
    row = _empty_metric_output_row(METRICS + LLM_EXTRA_METRICS)
    row.update(
        {
            "dataset": dataset,
            "labeler": "local_qwen",
            "experiment_tag": FORMAL_FEASIBILITY_TAG,
            "num_cards": 0,
            "num_risk_cards": 0,
            "parse_error_count": 0,
            "fallback_count": 0,
            "risk_relevance_rate": 0.0,
            "avg_confidence": 0.0,
            "agreement_with_mock": 0.0,
            "mechanism_distribution": "",
            "coverage": 0.0,
            "llm_label_coverage_rate": 0.0,
        }
    )
    return row


def _empty_highcov_row(tag: str) -> dict[str, Any]:
    row = _empty_metric_output_row(METRICS + LLM_EXTRA_METRICS)
    row.update(
        {
            "dataset": "yelp_academic",
            "labeler": _infer_highcov_labeler(tag),
            "experiment_tag": tag,
            "num_eval_target_nodes": 0,
            "num_cards": 0,
            "num_risk_cards": 0,
            "coverage": 0.0,
            "llm_label_coverage_rate": 0.0,
            "parse_error_count": 0,
            "risk_relevance_rate": 0.0,
            "avg_confidence": 0.0,
            "mechanism_distribution": "",
        }
    )
    return row


def _empty_metric_output_row(metrics: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for metric in metrics:
        row[f"{metric}_mean"] = 0.0
        row[f"{metric}_std"] = 0.0
        row[f"{metric}_mean_std"] = "0.0000±0.0000"
    return row


def _merge_into(target: dict[str, Any], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and not value:
            continue
        if _is_numeric_like(value) and _is_numeric_like(target.get(key, 0.0)):
            numeric_value = float(value)
            current_value = float(target.get(key, 0.0) or 0.0)
            if numeric_value == 0.0 and current_value != 0.0 and key not in {"parse_error_count", "fallback_count"}:
                continue
        target[key] = value


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _feasibility_summary_values(source_roots: list[Path], dataset: str, summary_dir_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in _candidate_files(source_roots, f"{summary_dir_name}/labeler_comparison_table.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            if _is_smoke_row(row, path):
                continue
            if "dataset" in row and str(row.get("dataset", dataset)) not in {"", dataset}:
                continue
            labeler = str(row.get("labeler", ""))
            if labeler and not _is_local_qwen_labeler(labeler):
                continue
            num_cards = _as_float(row.get("num_cards", row.get("num_risk_cards", 0.0)))
            if num_cards and int(round(num_cards)) != 500:
                continue
            values.update(_series_values(row, REAL_LLM_FEASIBILITY_COLUMNS))
            values["dataset"] = dataset
            values["labeler"] = "local_qwen"
            values["experiment_tag"] = FORMAL_FEASIBILITY_TAG
            if "num_risk_cards" not in values and "num_cards" in values:
                values["num_risk_cards"] = values["num_cards"]
    return values


def _feasibility_report_and_label(source_roots: list[Path], dataset: str) -> tuple[dict[str, Any], Path | None]:
    report_name = f"llm_label_build_report_{dataset}_local_qwen.json"
    preferred_roots = _real_llm_archive_roots(source_roots) + source_roots
    for path in _report_candidates(preferred_roots, report_name):
        report = _read_json(path)
        if not report or _is_smoke_text(str(path), report) or _is_highcov_report(report):
            continue
        label_file = _resolve_label_file(report, preferred_roots)
        stats = _label_file_stats(label_file) if label_file is not None else {}
        num_cards = int(
            round(
                _first_positive_numeric(
                    [
                        report.get("num_cards"),
                        report.get("num_risk_cards"),
                        report.get("num_seen"),
                        report.get("num_written"),
                        stats.get("num_cards"),
                    ],
                    default=0.0,
                )
            )
        )
        if num_cards and num_cards != 500:
            continue
        return report, label_file
    return {}, None


def _real_llm_archive_roots(source_roots: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for root in source_roots:
        roots.extend(
            [
                root / "archive_real_llm_qwen_500_final",
                root / "outputs" / "archive_real_llm_qwen_500_final",
            ]
        )
    roots.append(Path("outputs/archive_real_llm_qwen_500_final"))
    return [path for path in _dedupe_paths(roots) if path.exists()]


def _report_candidates(roots: list[Path], report_name: str) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / report_name,
                root / "summary_llm" / report_name,
                root / "outputs" / "summary_llm" / report_name,
                root / "summary_llm_yelp" / report_name,
                root / "summary_llm_amazon" / report_name,
            ]
        )
        if "archive_real_llm_qwen_500_final" in str(root):
            candidates.extend(root.rglob(report_name))
    return _dedupe_paths(candidates)


def _report_values(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    num_cards = _first_positive_numeric(
        [
            report.get("num_cards"),
            report.get("num_risk_cards"),
            report.get("num_seen"),
            report.get("num_outputs"),
            report.get("num_labels"),
            report.get("num_written"),
            report.get("total_cards"),
        ],
        default=0.0,
    )
    mechanism_distribution = report.get("mechanism_distribution", "")
    if isinstance(mechanism_distribution, dict):
        mechanism_distribution = json.dumps(mechanism_distribution, sort_keys=True)
    return {
        "num_cards": int(round(num_cards)) if num_cards else 0,
        "num_risk_cards": int(round(num_cards)) if num_cards else 0,
        "parse_error_count": int(round(_first_numeric([report.get("parse_error_count")], default=0.0))),
        "fallback_count": int(round(_first_numeric([report.get("fallback_count")], default=0.0))),
        "risk_relevance_rate": _first_numeric([report.get("risk_relevance_rate")], default=0.0),
        "avg_confidence": _first_numeric([report.get("avg_confidence")], default=0.0),
        "mechanism_distribution": mechanism_distribution,
    }


def _resolve_label_file(report: dict[str, Any], roots: list[Path]) -> Path | None:
    raw_path = str(report.get("out_file", report.get("label_file", report.get("llm_label_file", ""))))
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path)
        if not path.is_absolute():
            for root in roots:
                candidates.append(root / path)
                candidates.append(root / "outputs" / path)
        filename = path.name
        for root in roots:
            candidates.extend(root.rglob(filename))
    for path in _dedupe_paths(candidates):
        if path.exists() and not _is_smoke_text(str(path)):
            return path
    return None


def _label_file_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    labels: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels.append(payload)
    if not labels:
        return {}
    mechanisms: dict[str, int] = {}
    risk_values = []
    confidence_values = []
    for label in labels:
        risk_values.append(_risk_relevance_value(label))
        confidence_values.append(_as_float(label.get("confidence", 0.0)))
        mechanism = str(label.get("mechanism", label.get("risk_mechanism", "")))
        if mechanism:
            mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    return {
        "num_cards": len(labels),
        "num_risk_cards": len(labels),
        "risk_relevance_rate": sum(risk_values) / len(risk_values) if risk_values else 0.0,
        "avg_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "mechanism_distribution": json.dumps(mechanisms, sort_keys=True),
    }


def _risk_relevance_value(label: dict[str, Any]) -> int:
    value = label.get("risk_relevance", label.get("is_risk_relevant", label.get("risk_label", 0)))
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "risk", "relevant"} else 0
    return int(bool(value))


def _metric_summary_values(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    rows = [row for row in rows if str(row.get("model", "hero_gnn")) == "hero_gnn"]
    if not rows:
        return {}
    values: dict[str, Any] = {}
    for metric in metrics:
        metric_values = [_as_float(row.get(metric, 0.0)) for row in rows]
        values[f"{metric}_mean"] = sum(metric_values) / len(metric_values) if metric_values else 0.0
        values[f"{metric}_std"] = _sample_std(metric_values)
    for column in ["llm_label_coverage_rate", "coverage", "num_eval_target_nodes", "num_cards", "num_risk_cards"]:
        column_values = [_as_float(row.get(column, 0.0)) for row in rows if row.get(column) is not None]
        if not column_values:
            continue
        if column == "num_eval_target_nodes":
            values[column] = int(round(max(column_values)))
        else:
            values[column] = sum(column_values) / len(column_values)
    if "coverage" not in values and "llm_label_coverage_rate" in values:
        values["coverage"] = values["llm_label_coverage_rate"]
    if "llm_label_coverage_rate" not in values and "coverage" in values:
        values["llm_label_coverage_rate"] = values["coverage"]
    return values


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _matches_formal_feasibility_metric(row: dict[str, Any], dataset: str) -> bool:
    if str(row.get("dataset", "")) != dataset:
        return False
    if str(row.get("model", "hero_gnn")) != "hero_gnn":
        return False
    text = _row_text(row)
    if _is_smoke_text(text) or "highcov" in text or "high_coverage" in text:
        return False
    return FORMAL_FEASIBILITY_TAG in text or "results_qwen2p5_7b_500" in text


def _matches_highcov_metric(row: dict[str, Any], tag: str) -> bool:
    if str(row.get("model", "hero_gnn")) != "hero_gnn":
        return False
    text = _row_text(row)
    if _is_smoke_text(text):
        return False
    return tag in text


def _highcov_summary_values(source_roots: list[Path], tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _candidate_files(source_roots, "summary_llm_highcov/real_llm_highcov_table.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            if _is_smoke_row(row, path):
                continue
            if str(row.get("experiment_tag", "")) != tag:
                continue
            values = _series_values(row, REAL_LLM_HIGHCOV_COLUMNS)
            values["experiment_tag"] = tag
            values["labeler"] = str(values.get("labeler", "")) or _infer_highcov_labeler(tag)
            if "coverage" not in values and "llm_label_coverage_rate" in values:
                values["coverage"] = values["llm_label_coverage_rate"]
            if "llm_label_coverage_rate" not in values and "coverage" in values:
                values["llm_label_coverage_rate"] = values["coverage"]
            if "num_cards" not in values and "num_risk_cards" in values:
                values["num_cards"] = values["num_risk_cards"]
            rows.append(values)
    return rows


def _series_values(row: pd.Series, output_columns: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in output_columns:
        if column in row and not _is_missing(row[column]):
            values[column] = row[column]
    for metric in METRICS + LLM_EXTRA_METRICS:
        if metric in row and f"{metric}_mean" not in values:
            values[f"{metric}_mean"] = row[metric]
        if f"{metric}_std" not in values:
            values[f"{metric}_std"] = 0.0
    if "num_risk_cards" not in values and "num_cards" in values:
        values["num_risk_cards"] = values["num_cards"]
    return values


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _is_smoke_row(row: pd.Series, path: Path) -> bool:
    return _is_smoke_text(str(path), row.to_dict())


def _is_smoke_text(text: str, payload: dict[str, Any] | None = None) -> bool:
    merged = text.lower()
    if payload:
        merged += " " + json.dumps(payload, ensure_ascii=False, default=str).lower()
    return any(pattern in merged for pattern in SMOKE_PATTERNS)


def _is_highcov_report(report: dict[str, Any]) -> bool:
    text = json.dumps(report, ensure_ascii=False, default=str).lower()
    return "highcov" in text or "high_coverage" in text


def _is_local_qwen_labeler(labeler: str) -> bool:
    return labeler in {"local_qwen", "qwen", "qwen2p5_7b", "qwen2.5-7b"}


def _infer_highcov_labeler(tag: str) -> str:
    if tag.startswith("mock"):
        return "mock"
    if "qwen" in tag:
        return "local_qwen"
    return ""


def _row_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, default=str).lower()


def _first_numeric(values: list[Any], default: float = 0.0) -> float:
    for value in values:
        numeric = _as_float(value, default=None)
        if numeric is not None:
            return numeric
    return default


def _first_positive_numeric(values: list[Any], default: float = 0.0) -> float:
    for value in values:
        numeric = _as_float(value, default=None)
        if numeric is not None and numeric > 0:
            return numeric
    return default


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_metric_rows(
    rows: list[dict[str, Any]],
    group_columns: list[str],
    metric_columns: list[str],
    variant_from_model: bool = False,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "model" in frame.columns and "method" not in frame.columns:
        frame["method"] = frame["model"]
    if variant_from_model:
        frame["variant"] = frame.get("method", frame.get("model", ""))
    frame["coverage"] = _coverage_series(frame)
    for column in metric_columns + ["coverage", "risk_relevance_rate", "parse_error_count"]:
        _ensure_numeric(frame, column)

    aggregated = frame.groupby(group_columns, dropna=False).agg(**_agg_spec(metric_columns)).reset_index()
    for column in metric_columns:
        _add_mean_std_display(aggregated, column)

    controls = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            coverage=("coverage", "mean"),
            parse_error_count=("parse_error_count", "max"),
            risk_relevance_rate=("risk_relevance_rate", "mean"),
        )
        .reset_index()
    )
    return aggregated.merge(controls, on=group_columns, how="left")


def _aggregate_llm_rows(rows: list[dict[str, Any]], output_columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "model" in frame.columns:
        frame = frame[frame["model"].fillna("") == "hero_gnn"].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["labeler"] = frame.apply(_labeler_value, axis=1)
    frame["experiment_tag"] = frame.apply(_experiment_tag_value, axis=1)
    frame["coverage"] = _coverage_series(frame)
    frame["llm_label_coverage_rate"] = frame["coverage"]
    for column in METRICS + LLM_EXTRA_METRICS + [
        "coverage",
        "llm_label_coverage_rate",
        "risk_relevance_rate",
        "parse_error_count",
        "num_cards",
        "num_risk_cards",
        "num_eval_target_nodes",
        "avg_confidence",
        "agreement_with_mock",
    ]:
        _ensure_numeric(frame, column)

    group_columns = ["dataset", "labeler", "experiment_tag"]
    aggregated = frame.groupby(group_columns, dropna=False).agg(**_agg_spec(METRICS + LLM_EXTRA_METRICS)).reset_index()
    for column in METRICS + LLM_EXTRA_METRICS:
        _add_mean_std_display(aggregated, column)
    controls = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            num_eval_target_nodes=("num_eval_target_nodes", "max"),
            num_cards=("num_cards", "max"),
            num_risk_cards=("num_risk_cards", "max"),
            coverage=("coverage", "mean"),
            llm_label_coverage_rate=("llm_label_coverage_rate", "mean"),
            parse_error_count=("parse_error_count", "max"),
            risk_relevance_rate=("risk_relevance_rate", "mean"),
            avg_confidence=("avg_confidence", "mean"),
            agreement_with_mock=("agreement_with_mock", "mean"),
        )
        .reset_index()
    )
    table = aggregated.merge(controls, on=group_columns, how="left")
    table["mechanism_distribution"] = ""
    return _finalize(table, output_columns)


def _agg_spec(metric_columns: list[str]) -> dict[str, tuple[str, str]]:
    spec: dict[str, tuple[str, str]] = {}
    for column in metric_columns:
        spec[f"{column}_mean"] = (column, "mean")
        spec[f"{column}_std"] = (column, "std")
    return spec


def _summary_main(source_roots: list[Path], summary_dir: Path, datasets: set[str], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _summary_candidates(source_roots, summary_dir, "main_table.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "dataset" in frame.columns:
            frame = frame[frame["dataset"].isin(datasets)].copy()
        frames.append(_normalize_summary_metrics(frame, columns, ["dataset", "method"], METRICS))
    return _concat(frames)


def _summary_ablation(source_roots: list[Path], summary_dir: Path, datasets: set[str], columns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _summary_candidates(source_roots, summary_dir, "ablation_table.csv"):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "dataset" in frame.columns:
            frame = frame[frame["dataset"].isin(datasets)].copy()
        frames.append(_normalize_summary_metrics(frame, columns, ["dataset", "variant"], METRICS + PROXY_EXTRA_METRICS + OFFICIAL_EXTRA_METRICS))
    return _concat(frames)


def _summary_candidates(source_roots: list[Path], summary_dir: Path, name: str) -> list[Path]:
    paths = [summary_dir / name]
    for root in source_roots:
        paths.extend([root / "summary" / name, root / "outputs" / "summary" / name])
    return _dedupe_paths(paths)


def _filter_methods(frame: pd.DataFrame, column: str, allowed: set[str]) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    return frame[frame[column].astype(str).isin(allowed)].copy()


def _normalize_summary_metrics(frame: pd.DataFrame, columns: list[str], keys: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    if "model" in result.columns and "method" not in result.columns:
        result = result.rename(columns={"model": "method"})
    for metric in metrics:
        if f"{metric}_mean" not in result.columns:
            if metric in result.columns:
                result[f"{metric}_mean"] = result[metric]
            else:
                result[f"{metric}_mean"] = 0.0
        if f"{metric}_std" not in result.columns:
            result[f"{metric}_std"] = 0.0
        _ensure_numeric(result, f"{metric}_mean")
        _ensure_numeric(result, f"{metric}_std")
        _add_mean_std_display(result, metric)
    for control in CONTROL_COLUMNS:
        if control not in result.columns:
            result[control] = 0.0
    for key in keys:
        if key not in result.columns:
            result[key] = ""
    return _finalize(result, columns)


def _read_llm_summary_tables(source_roots: list[Path], highcov: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if highcov:
        relatives = ["summary_llm_highcov/real_llm_highcov_table.csv"]
    else:
        relatives = ["summary_llm/labeler_comparison_table.csv", "summary_llm_test/labeler_comparison_table.csv"]
    for relative in relatives:
        for path in _candidate_files(source_roots, relative):
            if path.exists():
                frames.append(_normalize_llm_summary(pd.read_csv(path), highcov=highcov))
    return _concat(frames)


def _normalize_llm_summary(frame: pd.DataFrame, highcov: bool) -> pd.DataFrame:
    columns = REAL_LLM_HIGHCOV_COLUMNS if highcov else REAL_LLM_FEASIBILITY_COLUMNS
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    if "experiment_tag" not in result.columns:
        result["experiment_tag"] = ""
    if "num_risk_cards" not in result.columns and "num_cards" in result.columns:
        result["num_risk_cards"] = result["num_cards"]
    if "num_cards" not in result.columns and "num_risk_cards" in result.columns:
        result["num_cards"] = result["num_risk_cards"]
    if "coverage" not in result.columns:
        result["coverage"] = result.get("llm_label_coverage_rate", 0.0)
    if "llm_label_coverage_rate" not in result.columns:
        result["llm_label_coverage_rate"] = result["coverage"]
    for metric in METRICS + LLM_EXTRA_METRICS:
        if f"{metric}_mean" not in result.columns:
            result[f"{metric}_mean"] = result[metric] if metric in result.columns else 0.0
        if f"{metric}_std" not in result.columns:
            result[f"{metric}_std"] = 0.0
        _ensure_numeric(result, f"{metric}_mean")
        _ensure_numeric(result, f"{metric}_std")
        _add_mean_std_display(result, metric)
    for column in [
        "num_eval_target_nodes",
        "num_cards",
        "num_risk_cards",
        "coverage",
        "llm_label_coverage_rate",
        "parse_error_count",
        "risk_relevance_rate",
        "avg_confidence",
        "agreement_with_mock",
    ]:
        if column not in result.columns:
            result[column] = 0.0
    if "mechanism_distribution" not in result.columns:
        result["mechanism_distribution"] = ""
    return _finalize(result, columns)


def _coverage_series(frame: pd.DataFrame) -> pd.Series:
    if "coverage" in frame.columns:
        return pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.0)
    if "llm_label_coverage_rate" in frame.columns:
        return pd.to_numeric(frame["llm_label_coverage_rate"], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(frame), index=frame.index)


def _labeler_value(row: pd.Series) -> str:
    value = row.get("llm_labeler", row.get("labeler", ""))
    if value:
        return str(value)
    label_file = str(row.get("llm_label_file", row.get("external_llm_label_file", ""))).lower()
    if "qwen" in label_file:
        return "local_qwen"
    if "openai" in label_file:
        return "openai"
    if "mock" in label_file:
        return "mock"
    return "external"


def _experiment_tag_value(row: pd.Series) -> str:
    value = row.get("experiment_tag", "")
    if value:
        return str(value)
    label_file = Path(str(row.get("llm_label_file", row.get("external_llm_label_file", ""))))
    return label_file.stem


def _is_proxy_row(row: dict[str, Any]) -> bool:
    return str(row.get("dataset", "")) in PROXY_DATASETS


def _is_llm_or_highcov_row(row: dict[str, Any]) -> bool:
    return bool(row.get("llm_label_file") or row.get("external_llm_label_file") or row.get("experiment_tag") or _is_highcov_row(row))


def _is_highcov_row(row: dict[str, Any] | pd.Series) -> bool:
    tag = str(row.get("experiment_tag", "")).lower()
    if "highcov" in tag or "high_coverage" in tag:
        return True
    if row.get("eval_target_file"):
        return True
    try:
        return int(row.get("num_eval_target_nodes", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _ensure_numeric(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = 0.0
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _add_mean_std_display(frame: pd.DataFrame, metric: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    display_col = f"{metric}_mean_std"
    if mean_col not in frame.columns:
        frame[mean_col] = 0.0
    if std_col not in frame.columns:
        frame[std_col] = 0.0
    _ensure_numeric(frame, mean_col)
    _ensure_numeric(frame, std_col)
    frame[display_col] = [
        f"{float(mean):.4f}\u00b1{float(std):.4f}"
        for mean, std in zip(frame[mean_col], frame[std_col])
    ]


def _prefer_first(primary: pd.DataFrame, fallback: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame = _concat([primary, fallback])
    if frame.empty:
        return frame
    present_keys = [key for key in keys if key in frame.columns]
    if not present_keys:
        return frame
    return frame.drop_duplicates(subset=present_keys, keep="first").reset_index(drop=True)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _finalize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy() if frame is not None else pd.DataFrame()
    for column in columns:
        if column not in result.columns:
            result[column] = 0.0 if _is_numeric_column(column) else ""
    result = result[columns]
    for column in result.columns:
        if _is_numeric_column(column):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        else:
            result[column] = result[column].fillna("")
    return result


def _is_numeric_column(column: str) -> bool:
    if column.endswith("_mean_std"):
        return False
    if column in {"dataset", "method", "variant", "labeler", "experiment_tag", "mechanism_distribution", "case_id", "target_node", "label", "prediction", "risk_mechanism", "evidence_chain"}:
        return False
    return True


def _write_table(frame: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_markdown(frame, csv_path.with_suffix(".md"))


def _write_markdown(frame: pd.DataFrame, md_path: Path) -> None:
    columns = [str(column) for column in frame.columns]
    with md_path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        handle.write("| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for _, row in frame.iterrows():
            values = [_escape_markdown_cell(_format_cell(row[column])) for column in frame.columns]
            handle.write("| " + " | ".join(values) + " |\n")


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_readme(out_dir: Path, source_roots: list[Path]) -> None:
    lines = [
        "# Paper Tables",
        "",
        "This directory contains final CSV and Markdown exports. Missing fields are filled with blank strings for text fields and 0 for numeric fields.",
        "",
        "Scanned source roots:",
    ]
    for root in source_roots:
        status = "exists" if root.exists() else "missing"
        lines.append(f"- `{root}` ({status})")
    lines.extend(
        [
            "",
            "| table | source directories | notes |",
            "| --- | --- | --- |",
            "| `table_main_proxy` | `outputs/results`, `outputs/summary`, `archive/results`, `archive/summary` | Text-rich proxy-label benchmark only. |",
            "| `table_ablation_proxy` | `outputs/results`, `outputs/summary`, `archive/results`, `archive/summary` | Text-rich HERO ablations only. |",
            "| `table_official_benchmark` | `outputs/results`, `outputs/summary`, `archive/results`, `archive/summary` | Official DGL fraud datasets only. |",
            "| `table_official_ablation` | `outputs/results`, `outputs/summary`, `archive/results`, `archive/summary` | HERO-official ablations only. |",
            "| `table_real_llm_feasibility` | `outputs/summary_llm_yelp`, `outputs/summary_llm_amazon`, `outputs/summary_llm`, `outputs/results_qwen2p5_7b_500`, `outputs/results`, `outputs/archive_real_llm_qwen_500_final` | Formal 500-card Qwen feasibility rows only. |",
            "| `table_real_llm_highcov` | `outputs/summary_llm_highcov`, `outputs/results_llm_comparison`, and archive mirrors | Formal `mock_highcov_300x20` and `qwen2p5_7b_highcov_300x20` rows only. |",
            "| `table_case_study` | `outputs/summary_llm_highcov/final_case_study_table.csv`, then `outputs/archive_case_study_final/final_case_study_table.csv` | Direct case-study export when the source file exists. |",
            "",
            "Each table is written as both `.csv` and `.md` with matching stems.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _print_self_checks(tables: dict[str, pd.DataFrame]) -> None:
    _check_or_warn(
        "table_main_proxy: no ablation variants",
        tables["table_main_proxy"].empty
        or not set(tables["table_main_proxy"].get("method", pd.Series(dtype=str)).astype(str))
        & {"hero_wo_chain", "hero_wo_hetero", "hero_wo_mechanism"},
    )
    _check_or_warn(
        "table_official_benchmark: no official ablation variants",
        tables["table_official_benchmark"].empty
        or not set(tables["table_official_benchmark"].get("method", pd.Series(dtype=str)).astype(str))
        & {"hero_official_wo_feature_deviation", "hero_official_wo_hetero", "hero_official_wo_relation"},
    )
    feasibility = tables["table_real_llm_feasibility"]
    feasibility_text = "\n".join(feasibility.astype(str).to_numpy().ravel().tolist()).lower() if not feasibility.empty else ""
    no_smoke = not any(pattern in feasibility_text for pattern in SMOKE_PATTERNS) and not (feasibility.get("num_cards", pd.Series(dtype=float)) == 20).any()
    _check_or_warn("table_real_llm_feasibility: no 20-card smoke test", no_smoke)
    if not feasibility.empty and (feasibility.get("num_cards", pd.Series(dtype=float)).fillna(0).astype(float) != 500).any():
        print("[WARNING] table_real_llm_feasibility: formal 500-card fields are incomplete or missing")

    highcov = tables["table_real_llm_highcov"]
    highcov_tags = highcov.get("experiment_tag", pd.Series(dtype=str)).astype(str)
    highcov_ok = "mock_highcov_test" not in set(highcov_tags) and not highcov_tags.duplicated().any()
    _check_or_warn("table_real_llm_highcov: no mock_highcov_test and no duplicate experiment_tag", highcov_ok)
    unexpected_tags = set(highcov_tags) - set(FORMAL_HIGHCOV_TAGS)
    if unexpected_tags:
        print(f"[WARNING] table_real_llm_highcov: unexpected experiment_tag values {sorted(unexpected_tags)}")

    case_study = tables["table_case_study"]
    string_fields = ["target_id", "neighbor_id", "mechanism", "llm_decision", "evidence_pattern", "interpretation"]
    zero_filled = False
    for field in string_fields:
        if field in case_study.columns and case_study[field].astype(str).str.fullmatch(r"0(?:\.0+)?").any():
            zero_filled = True
            break
    _check_or_warn("table_case_study: no zero-filled string fields", not zero_filled)
    if case_study.empty:
        print("[WARNING] table_case_study: final_case_study_table.csv was not found or is empty")


def _check_or_warn(message: str, ok: bool) -> None:
    prefix = "[CHECK]" if ok else "[WARNING]"
    print(f"{prefix} {message}")


def _write_legacy_tables(summary_dir: Path, out_dir: Path, new_tables: dict[str, pd.DataFrame]) -> None:
    _copy_or_empty(summary_dir / "main_table.csv", out_dir / "table_main.csv", LEGACY_MAIN_COLUMNS)
    _copy_or_empty(summary_dir / "ablation_table.csv", out_dir / "table_ablation.csv", LEGACY_ABLATION_COLUMNS)
    _copy_or_empty(summary_dir / "neighbor_strategy_table.csv", out_dir / "table_neighbor_strategy.csv", LEGACY_NEIGHBOR_COLUMNS)
    _copy_or_empty(summary_dir / "significance_table.csv", out_dir / "table_significance.csv", LEGACY_SIGNIFICANCE_COLUMNS)
    _copy_or_empty(summary_dir / "runtime_table.csv", out_dir / "table_runtime.csv", LEGACY_RUNTIME_COLUMNS)
    _copy_or_empty(summary_dir / "labeler_comparison_table.csv", out_dir / "table_labeler_comparison.csv", LEGACY_LABELER_COLUMNS)
    new_tables["table_official_benchmark"].to_csv(out_dir / "table_official_fraud.csv", index=False)
    _copy_or_empty(summary_dir / "scalability_table.csv", out_dir / "table_scalability.csv", LEGACY_RUNTIME_COLUMNS)


def _copy_or_empty(src: Path, dst: Path, columns: list[str]) -> None:
    if src.exists():
        frame = pd.read_csv(src)
    else:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(dst, index=False)


if __name__ == "__main__":
    main()
