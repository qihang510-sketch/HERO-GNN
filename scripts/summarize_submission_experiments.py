from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import DATASET_MODEL_MATRIX, FORBIDDEN_SUBMISSION_NAMES


METRICS = ["Macro-F1", "AUROC", "AUPRC", "Accuracy", "Precision", "Recall"]
ABLATION_DATASETS = ["yelp_academic", "amazon_video"]
ABLATION_VARIANTS = [
    "hero_gnn",
    "wo_risk_relevant_heterophily",
    "wo_mechanism_annotation",
    "wo_evidence_chain",
    "wo_llm_annotation",
    "wo_heterophily_filter",
    "wo_dual_branch_encoder",
    "wo_gated_fusion",
]
TABLE_SPECS = {
    "table_text_rich_main": {"datasets": ["yelp_academic", "amazon_video"]},
    "table_official_benchmark": {"datasets": ["fraud_yelp", "fraud_amazon"]},
    "table_transaction_benchmark": {"datasets": ["elliptic"]},
}
EMPTY_TABLES = [
    "table_significance_tests",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize submission experiment metrics into paper tables.")
    parser.add_argument("--input_dir", default="outputs/submission_experiments")
    parser.add_argument("--ablation_dir", default=None, help="Directory from run_ablation_experiments.py. Defaults to outputs/submission_experiments_ablation.")
    parser.add_argument("--llm_labeler_dir", default=None, help="Directory from run_labeler_comparison.py.")
    parser.add_argument("--llm_coverage_dir", default=None, help="Directory from run_llm_coverage_sensitivity.py.")
    parser.add_argument("--output_dir", default="outputs/paper_tables_submission")
    parser.add_argument("--min_seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_metrics(input_dir)
    skipped = _read_skips(input_dir)
    warnings: list[str] = []
    rows = [row for row in rows if not _is_forbidden(row.get("model", ""))]
    all_table = _all_results_table(rows, skipped, min_seeds=args.min_seeds, warnings=warnings)
    _write_table(all_table, output_dir / "table_all_results")
    for stem, spec in TABLE_SPECS.items():
        table = all_table[all_table["dataset"].isin(spec["datasets"])].copy()
        table = _rank_table(table)
        _write_table(table, output_dir / stem)
    ablation_dir = _resolve_ablation_dir(input_dir, args.ablation_dir)
    ablation_table = _ablation_table(ablation_dir, min_seeds=args.min_seeds, warnings=warnings)
    if not ablation_table.empty:
        _write_table(ablation_table, output_dir / "table_ablation")
    elif not (output_dir / "table_ablation.csv").exists():
        _write_table(pd.DataFrame(columns=["status", "warning"]), output_dir / "table_ablation")
    llm_labeler_dir = _resolve_llm_dir(input_dir, args.llm_labeler_dir, "llm_labeler", "outputs/submission_llm_labeler")
    labeler_table = _llm_labeler_table(llm_labeler_dir, warnings=warnings)
    _write_table(labeler_table if not labeler_table.empty else _warning_table(f"missing_or_empty_llm_labeler_results:{llm_labeler_dir}"), output_dir / "table_llm_labeler_comparison")
    llm_coverage_dir = _resolve_llm_dir(input_dir, args.llm_coverage_dir, "llm_coverage", "outputs/submission_llm_coverage")
    coverage_table = _llm_coverage_table(llm_coverage_dir, warnings=warnings)
    _write_table(coverage_table if not coverage_table.empty else _warning_table(f"missing_or_empty_llm_coverage_results:{llm_coverage_dir}"), output_dir / "table_llm_coverage_sensitivity")
    for stem in EMPTY_TABLES:
        path = output_dir / f"{stem}.csv"
        if not path.exists():
            _write_table(pd.DataFrame(columns=["status", "warning"]), output_dir / stem)
    (output_dir / "warnings.json").write_text(json.dumps({"warnings": warnings}, indent=2), encoding="utf-8")
    print(f"Wrote submission tables to {output_dir}")


def _read_metrics(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/seed_*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _read_skips(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/*/seed_*/skip_reason.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _resolve_ablation_dir(input_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if "ablation" in input_dir.name.lower():
        return input_dir
    return Path("outputs/submission_experiments_ablation")


def _resolve_llm_dir(input_dir: Path, explicit: str | None, lightcheck_name: str, formal_default: str) -> Path:
    if explicit:
        return Path(explicit)
    sibling = input_dir.parent / lightcheck_name
    if sibling.exists():
        return sibling
    return Path(formal_default)


def _llm_labeler_table(labeler_dir: Path, warnings: list[str]) -> pd.DataFrame:
    rows = _read_llm_labeler_metrics(labeler_dir)
    if not rows:
        legacy_csv = labeler_dir / "labeler_comparison_table.csv"
        if legacy_csv.exists():
            legacy = pd.read_csv(legacy_csv)
            return _aggregate_llm_frame(legacy, group_cols=["dataset", "labeler"], min_seeds=1, warnings=warnings, table_name="llm_labeler")
        warnings.append(f"llm_labeler_dir contains no metrics: {labeler_dir}")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return _aggregate_llm_frame(frame, group_cols=["dataset", "labeler"], min_seeds=1, warnings=warnings, table_name="llm_labeler")


def _llm_coverage_table(coverage_dir: Path, warnings: list[str]) -> pd.DataFrame:
    rows = _read_llm_coverage_metrics(coverage_dir)
    if not rows:
        legacy_csv = coverage_dir / "coverage_sensitivity_table.csv"
        if legacy_csv.exists():
            legacy = pd.read_csv(legacy_csv)
            return _aggregate_llm_frame(legacy, group_cols=["dataset", "coverage"], min_seeds=1, warnings=warnings, table_name="llm_coverage")
        warnings.append(f"llm_coverage_dir contains no metrics: {coverage_dir}")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "coverage" in frame:
        frame["coverage"] = pd.to_numeric(frame["coverage"], errors="coerce")
    return _aggregate_llm_frame(frame, group_cols=["dataset", "coverage"], min_seeds=1, warnings=warnings, table_name="llm_coverage")


def _read_llm_labeler_metrics(labeler_dir: Path) -> list[dict[str, Any]]:
    if not labeler_dir.exists():
        return []
    rows = []
    for path in sorted(labeler_dir.glob("*/*/seed_*/metrics.json")):
        payload = _read_metric_payload(path)
        if not payload:
            continue
        payload.setdefault("dataset", path.parents[2].name)
        payload.setdefault("labeler", path.parents[1].name)
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _read_llm_coverage_metrics(coverage_dir: Path) -> list[dict[str, Any]]:
    if not coverage_dir.exists():
        return []
    rows = []
    canonical = sorted(coverage_dir.glob("*/*/seed_*/metrics.json"))
    legacy = sorted(coverage_dir.glob("coverage_*/*/hero_gnn/seed_*/metrics.json")) if not canonical else []
    for path in [*canonical, *legacy]:
        payload = _read_metric_payload(path)
        if not payload:
            continue
        if "dataset" not in payload:
            payload["dataset"] = path.parents[2].name if "coverage_" not in path.parents[3].name else path.parents[2].name
        if "coverage" not in payload:
            payload["coverage"] = _coverage_from_path(path)
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def _read_metric_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _coverage_from_path(path: Path) -> Any:
    for part in path.parts:
        if part.startswith("coverage_"):
            try:
                return float(part.split("_", 1)[1]) / 100.0
            except (IndexError, ValueError):
                return pd.NA
    return pd.NA


def _warning_table(message: str) -> pd.DataFrame:
    return pd.DataFrame([{"status": "warning", "warning": message}])


def _aggregate_llm_frame(
    frame: pd.DataFrame,
    group_cols: list[str],
    min_seeds: int,
    warnings: list[str],
    table_name: str,
) -> pd.DataFrame:
    if frame.empty or any(col not in frame for col in group_cols):
        warnings.append(f"{table_name}: missing required columns {group_cols}")
        return pd.DataFrame()
    table_rows = []
    sort_frame = frame.copy()
    for col in group_cols:
        if col == "coverage":
            sort_frame[col] = pd.to_numeric(sort_frame[col], errors="coerce")
    grouped = sort_frame.groupby(group_cols, dropna=False)
    for keys, subset in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = {col: key_values[idx] for idx, col in enumerate(group_cols)}
        seed_count = int(subset["seed"].nunique()) if "seed" in subset else int(subset.shape[0])
        row["seed_count"] = seed_count
        row["missing_seed_count"] = max(int(min_seeds) - seed_count, 0)
        row["status"] = "ok" if seed_count else "missing"
        row["warning"] = ""
        if seed_count and seed_count < min_seeds:
            row["warning"] = f"insufficient_seeds:{seed_count}/{min_seeds}"
        for metric in METRICS:
            values = _metric_values(subset, metric)
            row[f"{metric}_mean"] = float(np.mean(values)) if values else pd.NA
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else pd.NA
            row[f"{metric}_mean_std"] = _mean_std_display(values)
        for col in ["coverage", "parse_error_count", "risk_relevance_rate", "avg_confidence", "llm_label_coverage_rate", "num_cards", "num_annotations"]:
            if col in subset and col not in row:
                numeric = pd.to_numeric(subset[col], errors="coerce").dropna()
                row[col] = float(numeric.mean()) if not numeric.empty else pd.NA
        table_rows.append(row)
    return pd.DataFrame(table_rows)


def _ablation_table(ablation_dir: Path, min_seeds: int, warnings: list[str]) -> pd.DataFrame:
    if not ablation_dir.exists():
        warnings.append(f"ablation_dir missing: {ablation_dir}")
        return pd.DataFrame()
    rows = _read_metrics(ablation_dir)
    skipped = _read_skips(ablation_dir)
    frame = pd.DataFrame(rows)
    skip_frame = pd.DataFrame(skipped)
    if frame.empty and skip_frame.empty:
        warnings.append(f"ablation_dir contains no metrics or skip files: {ablation_dir}")
        return pd.DataFrame()
    table_rows = []
    for dataset in ABLATION_DATASETS:
        for variant in ABLATION_VARIANTS:
            subset = frame[(frame.get("dataset", pd.Series(dtype=str)) == dataset) & (frame.get("model", pd.Series(dtype=str)) == variant)] if not frame.empty else pd.DataFrame()
            skip_subset = skip_frame[(skip_frame.get("dataset", pd.Series(dtype=str)) == dataset) & (skip_frame.get("model", pd.Series(dtype=str)) == variant)] if not skip_frame.empty else pd.DataFrame()
            row = {
                "dataset": dataset,
                "variant": variant,
                "model": variant,
                "ablation": _ablation_display_name(variant),
                "seed_count": int(subset["seed"].nunique()) if "seed" in subset else 0,
                "missing_seed_count": max(int(min_seeds) - (int(subset["seed"].nunique()) if "seed" in subset else 0), 0),
                "skip_count": int(skip_subset.shape[0]),
                "status": "ok" if not subset.empty else ("skipped" if not skip_subset.empty else "NA"),
                "warning": "",
            }
            if row["seed_count"] and row["seed_count"] < min_seeds:
                row["warning"] = f"insufficient_seeds:{row['seed_count']}/{min_seeds}"
                warnings.append(f"ablation {dataset}/{variant} has only {row['seed_count']} seed(s).")
            if row["status"] != "ok":
                row["warning"] = _first_value(skip_subset, "skip_reason") or "missing_results"
                warnings.append(f"ablation {dataset}/{variant}: {row['warning']}")
            for metric in METRICS:
                values = _metric_values(subset, metric)
                row[f"{metric}_mean"] = float(np.mean(values)) if values else pd.NA
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else pd.NA
                row[f"{metric}_mean_std"] = _mean_std_display(values)
            table_rows.append(row)
    return pd.DataFrame(table_rows)


def _ablation_display_name(variant: str) -> str:
    names = {
        "hero_gnn": "HERO-GNN",
        "wo_risk_relevant_heterophily": "w/o Risk-relevant Heterophily",
        "wo_mechanism_annotation": "w/o Mechanism Annotation",
        "wo_evidence_chain": "w/o Evidence Chain",
        "wo_llm_annotation": "w/o LLM Annotation",
        "wo_heterophily_filter": "w/o Heterophily Filter",
        "wo_dual_branch_encoder": "w/o Dual-Branch Encoder",
        "wo_gated_fusion": "w/o Gated Fusion",
    }
    return names.get(variant, variant)


def _all_results_table(rows: list[dict[str, Any]], skipped: list[dict[str, Any]], min_seeds: int, warnings: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    table_rows = []
    skip_frame = pd.DataFrame(skipped)
    for dataset, models in DATASET_MODEL_MATRIX.items():
        for model in models:
            if _is_forbidden(model):
                warnings.append(f"Forbidden lite baseline excluded: {dataset}/{model}")
                continue
            subset = frame[(frame.get("dataset", pd.Series(dtype=str)) == dataset) & (frame.get("model", pd.Series(dtype=str)) == model)] if not frame.empty else pd.DataFrame()
            skip_subset = skip_frame[(skip_frame.get("dataset", pd.Series(dtype=str)) == dataset) & (skip_frame.get("model", pd.Series(dtype=str)) == model)] if not skip_frame.empty else pd.DataFrame()
            row = {
                "dataset": dataset,
                "model": model,
                "seed_count": int(subset["seed"].nunique()) if "seed" in subset else 0,
                "missing_seed_count": max(int(min_seeds) - (int(subset["seed"].nunique()) if "seed" in subset else 0), 0),
                "skip_count": int(skip_subset.shape[0]),
                "implementation_source": _first_value(subset, "implementation_source"),
                "status": "ok" if not subset.empty else ("skipped" if not skip_subset.empty else "NA"),
                "warning": "",
            }
            if row["seed_count"] and row["seed_count"] < min_seeds:
                row["warning"] = f"insufficient_seeds:{row['seed_count']}/{min_seeds}"
                warnings.append(f"{dataset}/{model} has only {row['seed_count']} seed(s).")
            if row["status"] != "ok":
                row["warning"] = _first_value(skip_subset, "skip_reason") or "missing_results"
                warnings.append(f"{dataset}/{model}: {row['warning']}")
            for metric in METRICS:
                values = _metric_values(subset, metric)
                row[f"{metric}_mean"] = float(np.mean(values)) if values else pd.NA
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else pd.NA
                row[f"{metric}_mean_std"] = _mean_std_display(values)
            table_rows.append(row)
    return pd.DataFrame(table_rows)


def _metric_values(frame: pd.DataFrame, metric: str) -> list[float]:
    if frame.empty:
        return []
    columns = [metric, metric.lower(), metric.replace("-", "_").lower()]
    for col in columns:
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").dropna().tolist()
            return [float(value) for value in values]
    return []


def _mean_std_display(values: list[float]) -> str:
    if not values:
        return "NA"
    mean = float(np.mean(values))
    if len(values) < 2:
        return f"{mean:.4f}+/-NA"
    std = float(np.std(values, ddof=1))
    return f"{mean:.4f}+/-{std:.4f}"


def _rank_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    ranked = table.copy()
    for metric in ["Macro-F1", "AUPRC"]:
        col = f"{metric}_mean"
        if col not in ranked:
            continue
        ranked[f"{metric}_rank"] = ranked.groupby("dataset")[col].rank(ascending=False, method="min", na_option="bottom")
        ranked[f"{metric}_mark"] = ranked[f"{metric}_rank"].map(lambda value: "best" if value == 1 else ("second" if value == 2 else ""))
    return ranked


def _first_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""


def _write_table(frame: pd.DataFrame, stem_path: Path) -> None:
    frame.to_csv(stem_path.with_suffix(".csv"), index=False)
    stem_path.with_suffix(".md").write_text(_to_markdown(frame), encoding="utf-8")
    stem_path.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")


def _is_forbidden(model: Any) -> bool:
    text = str(model).lower()
    return text in FORBIDDEN_SUBMISSION_NAMES or "lite" in text


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    if not columns:
        return "\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
