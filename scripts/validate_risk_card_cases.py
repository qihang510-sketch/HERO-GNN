from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASETS = ["yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"]
FORBIDDEN_TERMS = ["forecast", "planning_only", "not_for_paper"]
FORBIDDEN_DATA_TERMS = ["fake", "manual_example", "fake_example", "example_id", "hard_coded_example"]
FAKE_MARKERS = ["fake_id", "dummy_id", "placeholder_id", "hard_coded_fake", "manual_example", "fake_example"]
MISSING_OK = {"N/A", "unavailable"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate representative risk-card case artifacts.")
    parser.add_argument("--case_dir", default="outputs/risk_card_cases", help="Risk-card case artifact directory.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Required datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_risk_card_cases(args.case_dir, args.datasets)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)
    print(f"[OK] Risk-card case artifacts validated for {len(args.datasets)} datasets.")


def validate_risk_card_cases(case_dir: str | Path, datasets: list[str]) -> list[str]:
    root = Path(case_dir)
    errors: list[str] = []
    compact_path = root / "tables_csv" / "table_risk_card_cases_compact.csv"
    trace_path = root / "tables_csv" / "table_risk_card_field_trace.csv"
    cases_path = root / "raw" / "selected_cases.jsonl"
    report_path = root / "reports" / "RISK_CARD_CASE_REPORT.md"

    compact = _read_csv(compact_path, errors)
    trace = _read_csv(trace_path, errors)
    cases = _read_jsonl(cases_path, errors)
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if not report_path.exists():
        errors.append(f"Missing report: {report_path}")

    if compact is not None:
        _validate_compact(compact, datasets, errors)
    if trace is not None:
        _validate_trace(trace, datasets, errors)
    _validate_cases(cases, datasets, report, errors)
    _validate_forbidden_terms(root, errors)
    return errors


def _validate_compact(compact: pd.DataFrame, datasets: list[str], errors: list[str]) -> None:
    if "dataset" not in compact.columns:
        errors.append("Compact table missing dataset column.")
        return
    present = set(compact["dataset"].astype(str))
    missing = sorted(set(datasets) - present)
    if missing:
        errors.append(f"Compact table missing datasets: {missing}")
    for dataset in datasets:
        count = int((compact["dataset"].astype(str) == dataset).sum())
        if count != 1:
            errors.append(f"Compact table should contain exactly one row for {dataset}, found {count}.")
    _validate_no_blank_cells(compact, "compact table", errors)


def _validate_trace(trace: pd.DataFrame, datasets: list[str], errors: list[str]) -> None:
    required = {"dataset", "case_id", "field_name", "source_column_or_file", "computation_rule"}
    missing_columns = sorted(required - set(trace.columns))
    if missing_columns:
        errors.append(f"Field trace table missing columns: {missing_columns}")
        return
    present = set(trace["dataset"].astype(str))
    missing = sorted(set(datasets) - present)
    if missing:
        errors.append(f"Field trace table missing datasets: {missing}")
    for dataset in datasets:
        count = int((trace["dataset"].astype(str) == dataset).sum())
        if count < 8:
            errors.append(f"Field trace table has fewer than 8 rows for {dataset}: {count}.")
    _validate_no_blank_cells(trace, "field trace table", errors)


def _validate_cases(cases: list[dict[str, Any]], datasets: list[str], report: str, errors: list[str]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_dataset.setdefault(str(case.get("dataset", "")), []).append(case)
    for dataset in datasets:
        rows = by_dataset.get(dataset, [])
        if not rows:
            errors.append(f"selected_cases.jsonl has no row for {dataset}.")
            continue
        selected = [row for row in rows if _truthy(row.get("is_selected"))]
        if not selected:
            errors.append(f"selected_cases.jsonl has no selected case for {dataset}.")
        for row in selected:
            status = str(row.get("status", "ok"))
            sources = row.get("source_files", [])
            if status == "unavailable":
                if dataset not in report or "unavailable" not in _report_section(report, dataset).lower():
                    errors.append(f"{dataset} is unavailable but report does not mark it unavailable.")
                continue
            if not sources or sources == ["unavailable"]:
                errors.append(f"{dataset} selected case has no real source_files provenance.")
            if str(row.get("selection_source", "")).lower() in {"hard_coded", "manual_example", "fake_example"}:
                errors.append(f"{dataset} selected case uses forbidden selection_source={row.get('selection_source')}.")
            if str(row.get("reconstruction_source", "")).lower() == "planning_only":
                errors.append(f"{dataset} selected case uses planning_only reconstruction.")
    serialized = json.dumps(cases, ensure_ascii=False).lower()
    for marker in FAKE_MARKERS:
        if marker in serialized:
            errors.append(f"selected_cases.jsonl contains fake-example marker: {marker}")


def _validate_no_blank_cells(frame: pd.DataFrame, name: str, errors: list[str]) -> None:
    for column in frame.columns:
        values = frame[column].astype(str)
        blank_count = int(values.str.strip().eq("").sum())
        if blank_count:
            errors.append(f"{name} column {column} contains {blank_count} blank cells; use N/A or unavailable.")


def _validate_forbidden_terms(root: Path, errors: list[str]) -> None:
    files = [
        root / "tables_csv" / "table_risk_card_cases_compact.csv",
        root / "tables_csv" / "table_risk_card_field_trace.csv",
        root / "tables_latex" / "table_risk_card_cases_compact.tex",
        root / "tables_latex" / "table_risk_card_field_trace.tex",
        root / "tables_markdown" / "table_risk_card_cases_compact.md",
        root / "tables_markdown" / "table_risk_card_field_trace.md",
        root / "raw" / "selected_cases.jsonl",
        root / "raw" / "risk_card_field_traces.jsonl",
        root / "reports" / "RISK_CARD_CASE_REPORT.md",
    ]
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for term in FORBIDDEN_TERMS:
            if term in text:
                errors.append(f"Forbidden term '{term}' found in {path}.")
        if path.suffix.lower() in {".csv", ".jsonl", ".md"}:
            for term in FORBIDDEN_DATA_TERMS:
                if term in text:
                    errors.append(f"Forbidden data term '{term}' found in {path}.")


def _read_csv(path: Path, errors: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        errors.append(f"Missing CSV artifact: {path}")
        return None
    try:
        return pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        errors.append(f"Could not read {path}: {exc}")
        return None


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        errors.append(f"Missing JSONL artifact: {path}")
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Invalid JSON in {path}:{line_number}: {exc}")
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception as exc:
        errors.append(f"Could not read {path}: {exc}")
    return rows


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _report_section(report: str, dataset: str) -> str:
    marker = f"### {dataset}"
    start = report.find(marker)
    if start < 0:
        return ""
    next_start = report.find("### ", start + len(marker))
    return report[start:] if next_start < 0 else report[start:next_start]


if __name__ == "__main__":
    main()
