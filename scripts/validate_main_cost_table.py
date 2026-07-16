from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASETS = ["yelp_academic", "amazon_video"]
FORBIDDEN_STRICT = ["forecast", "planning_only", "not_for_paper"]
FORBIDDEN_DATA = ["fake", "manual_example", "fake_example", "example_id", "hard_coded_example"]
MISSING_OK = {"N/A", "unavailable", "runtime_unavailable"}
ALLOWED_HERO_ANNOTATION = {"cached_local_qwen", "cached_annotation", "local_qwen", "proxy", "unavailable", "N/A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate main experiment cost artifacts.")
    parser.add_argument("--cost_dir", default="outputs/main_cost")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_main_cost_table(args.cost_dir, args.datasets)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)
    print(f"[OK] Main experiment cost artifacts validated for {len(args.datasets)} datasets.")


def validate_main_cost_table(cost_dir: str | Path, datasets: list[str]) -> list[str]:
    root = Path(cost_dir)
    errors: list[str] = []
    detail_path = root / "tables_csv" / "supp_table_main_experiment_cost.csv"
    compact_path = root / "tables_csv" / "supp_table_main_experiment_cost_compact.csv"
    raw_path = root / "raw" / "main_cost_raw_runs.csv"
    source_path = root / "raw" / "main_cost_runtime_sources.jsonl"
    report_path = root / "reports" / "MAIN_EXPERIMENT_COST_REPORT.md"

    detail = _read_csv(detail_path, errors)
    compact = _read_csv(compact_path, errors)
    _require_exists(raw_path, errors)
    _require_exists(source_path, errors)
    _require_exists(report_path, errors)

    if detail is not None:
        _validate_detail(detail, datasets, errors)
        _validate_no_blank(detail, "detailed cost table", errors)
        _validate_forbidden_frame(detail, "detailed cost table", errors)
    if compact is not None:
        _validate_compact(compact, datasets, errors)
        _validate_no_blank(compact, "compact cost table", errors)
        _validate_forbidden_frame(compact, "compact cost table", errors)
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in FORBIDDEN_STRICT:
            if term in text:
                errors.append(f"Forbidden term '{term}' found in report.")
        if "cost values are extracted from logged runtime files and cached artifacts" not in text:
            errors.append("MAIN_EXPERIMENT_COST_REPORT.md missing integrity statement.")
    return errors


def _validate_detail(detail: pd.DataFrame, datasets: list[str], errors: list[str]) -> None:
    required = {
        "suite",
        "dataset",
        "model",
        "seed_count",
        "run_count",
        "success_count",
        "failed_count",
        "skipped_count",
        "train_time_sec_mean",
        "total_runtime_sec_mean",
        "gpu_memory_mb_max",
        "annotation_source",
        "cost_source",
        "missing_fields",
        "status",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        errors.append(f"Detailed cost table missing columns: {missing}")
        return
    present = set(detail["dataset"].astype(str))
    missing_datasets = sorted(set(datasets) - present)
    if missing_datasets:
        errors.append(f"Detailed cost table missing datasets: {missing_datasets}")
    models = set(detail["model"].astype(str))
    if "HERO" not in models:
        errors.append("Detailed cost table missing HERO rows.")
    baselines = models - {"HERO"}
    if not baselines:
        errors.append("Detailed cost table missing baseline rows.")
    hero = detail[detail["model"].astype(str) == "HERO"]
    if not hero.empty:
        invalid = sorted(set(hero["annotation_source"].astype(str)) - ALLOWED_HERO_ANNOTATION)
        if invalid:
            errors.append(f"HERO annotation_source contains unsupported values: {invalid}")


def _validate_compact(compact: pd.DataFrame, datasets: list[str], errors: list[str]) -> None:
    required = {
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
    }
    missing = sorted(required - set(compact.columns))
    if missing:
        errors.append(f"Compact cost table missing columns: {missing}")
        return
    display_to_raw = {"Yelp Academic": "yelp_academic", "Amazon Video": "amazon_video"}
    present = {display_to_raw.get(str(value), str(value)) for value in compact["Dataset"].astype(str)}
    missing_datasets = sorted(set(datasets) - present)
    if missing_datasets:
        errors.append(f"Compact cost table missing datasets: {missing_datasets}")
    text = compact.to_csv(index=False).lower()
    if "quick" in text or "mock_fallback" in text:
        errors.append("quick_test/mock_fallback rows should not enter compact table.")
    baseline = compact[compact["Model"].astype(str) != "HERO"]
    if not baseline.empty:
        invalid = baseline[baseline["LLM Annotation Cost"].astype(str) != "N/A"]
        if not invalid.empty:
            errors.append("Baseline rows must have LLM Annotation Cost=N/A.")
    statuses = set(compact["Status"].astype(str))
    allowed_status = {"ok", "runtime_unavailable", "failed_or_skipped", "unavailable", "N/A"}
    invalid_status = sorted(statuses - allowed_status)
    if invalid_status:
        errors.append(f"Compact cost table contains invalid statuses: {invalid_status}")


def _validate_no_blank(frame: pd.DataFrame, name: str, errors: list[str]) -> None:
    for column in frame.columns:
        blanks = int(frame[column].astype(str).str.strip().eq("").sum())
        if blanks:
            errors.append(f"{name} column {column} contains {blanks} blank cells; use N/A or unavailable.")


def _validate_forbidden_frame(frame: pd.DataFrame, name: str, errors: list[str]) -> None:
    provenance = {"source_output", "source_files", "run_dir", "cost_source"}
    for column in frame.columns:
        text = "\n".join(frame[column].astype(str).tolist()).lower()
        for term in FORBIDDEN_STRICT:
            if term in text:
                errors.append(f"Forbidden term '{term}' found in {name} column {column}.")
        if column in provenance:
            continue
        for term in FORBIDDEN_DATA:
            if term in text:
                errors.append(f"Forbidden data term '{term}' found in {name} column {column}.")


def _read_csv(path: Path, errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        errors.append(f"Missing CSV artifact: {path}")
        return None
    try:
        return pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        errors.append(f"Could not read {path}: {exc}")
        return None


def _require_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing artifact: {path}")


if __name__ == "__main__":
    main()
