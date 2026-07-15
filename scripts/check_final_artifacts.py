from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import read_csv_or_empty  # noqa: E402


REQUIRED_TABLES = {
    "main text-rich": "tables_csv/table2_main_text_rich.csv",
    "transfer": "tables_csv/table3_transfer.csv",
    "ablation": "tables_csv/table4_ablation.csv",
    "significance": "tables_csv/supp_table_significance.csv",
    "robustness": "tables_csv/supp_table_llm_robustness.csv",
    "labeler comparison": "tables_csv/supp_table_labeler_comparison.csv",
    "faithfulness": "tables_csv/supp_table_faithfulness.csv",
    "cost/scalability": "tables_csv/supp_table_cost_scalability.csv",
    "missing/skipped": "tables_csv/supp_table_missing_or_skipped_runs.csv",
}
REQUIRED_LATEX = {
    "main text-rich": "tables_latex/table2_main_text_rich.tex",
    "transfer": "tables_latex/table3_transfer.tex",
    "ablation": "tables_latex/table4_ablation.tex",
    "evidence cases": "tables_latex/table5_evidence_cases.tex",
    "significance": "tables_latex/supp_table_significance.tex",
    "robustness": "tables_latex/supp_table_llm_robustness.tex",
    "labeler comparison": "tables_latex/supp_table_labeler_comparison.tex",
    "faithfulness": "tables_latex/supp_table_faithfulness.tex",
    "cost/scalability": "tables_latex/supp_table_cost_scalability.tex",
}
REQUIRED_FIGURES = {
    "robustness AUPRC pdf": "figures_pdf/fig_llm_robustness_auprc.pdf",
    "robustness AUPRC png": "figures_png/fig_llm_robustness_auprc.png",
    "labeler AUPRC pdf": "figures_pdf/fig_labeler_comparison_auprc.pdf",
    "labeler AUPRC png": "figures_png/fig_labeler_comparison_auprc.png",
    "faithfulness pdf": "figures_pdf/fig_evidence_faithfulness.pdf",
    "faithfulness png": "figures_png/fig_evidence_faithfulness.png",
    "cost annotation time pdf": "figures_pdf/fig_cost_annotation_time.pdf",
    "cost annotation time png": "figures_png/fig_cost_annotation_time.png",
}
REQUIRED_FIGURE_DATA = {
    "robustness curve": "figure_data/robustness_curve_data.csv",
    "labeler comparison": "figure_data/labeler_comparison_data.csv",
    "faithfulness bar": "figure_data/faithfulness_bar_data.csv",
    "seed stability": "figure_data/seed_stability_data.csv",
    "cost scalability": "figure_data/cost_scalability_data.csv",
}
RISK_CARD_DATASETS = {"yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"}
RISK_CARD_TABLES = {
    "risk card compact": "tables_csv/supp_table_risk_card_cases_compact.csv",
    "risk card field trace": "tables_csv/supp_table_risk_card_field_trace.csv",
}
RISK_CARD_LATEX = {
    "risk card compact latex": "tables_latex/supp_table_risk_card_cases_compact.tex",
    "risk card field trace latex": "tables_latex/supp_table_risk_card_field_trace.tex",
}
RISK_CARD_REPORTS = {
    "risk card report": "reports/RISK_CARD_CASE_REPORT.md",
}
FORBIDDEN_RISK_CARD_TERMS = ["planning_only", "forecast", "not_for_paper"]
FORBIDDEN_RISK_CARD_DATA_TERMS = ["fake", "manual_example", "fake_example", "example_id", "hard_coded_example"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check final HERO paper artifacts and write a final experiment report.")
    parser.add_argument("--output_root", default=None, help="Review rerun root containing stage subdirectories.")
    parser.add_argument("--output_dir", default=None, help="Alias for --final_dir.")
    parser.add_argument("--final_dir", default=None, help="Final artifacts directory. Defaults to <output_root>/final_artifacts.")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if required artifacts are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir and not args.final_dir:
        args.final_dir = args.output_dir
    report_path, rows = check_final_artifacts(args.output_root, args.final_dir, allow_missing=args.allow_missing)
    missing = [row for row in rows if row["status"] in {"missing", "failed"}]
    print(f"Wrote final experiment report to {report_path}")
    if args.strict and missing:
        raise SystemExit(1)


def check_final_artifacts(
    output_root: str | Path | None,
    final_dir: str | Path | None = None,
    allow_missing: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(output_root) if output_root else None
    final = Path(final_dir) if final_dir else (root / "final_artifacts" if root else Path("outputs/final_artifacts"))
    report_dir = final / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    checks.extend(_check_csvs(final, REQUIRED_TABLES, required_fields=True))
    checks.extend(_check_text_files(final, REQUIRED_LATEX, kind="latex"))
    checks.extend(_check_binary_files(final, REQUIRED_FIGURES, kind="figure"))
    checks.extend(_check_csvs(final, REQUIRED_FIGURE_DATA, required_fields=False, kind="figure_data"))
    checks.extend(_check_risk_card_artifacts(final))

    status_counts = _status_counts(root)
    tables = _existing_paths(final, "tables_csv", "*.csv")
    figures = _existing_paths(final, "figures_pdf", "*.pdf") + _existing_paths(final, "figures_png", "*.png")
    best_configs = _hero_best_configs(root)
    significance = _table_summary(final / "tables_csv" / "supp_table_significance.csv", ["status", "significant_0_05"])
    robustness = _table_summary(final / "tables_csv" / "supp_table_llm_robustness.csv", ["dataset", "noise_type", "status"])
    faithfulness = _table_summary(final / "tables_csv" / "supp_table_faithfulness.csv", ["dataset", "setting", "status"])
    cost = _table_summary(final / "tables_csv" / "supp_table_cost_scalability.csv", ["dataset", "model", "status"])
    risk_cards = _table_summary(final / "tables_csv" / "supp_table_risk_card_cases_compact.csv", ["dataset", "status", "mechanism_candidate"])
    missing = _missing_or_unavailable(final)

    report_path = report_dir / "FINAL_EXPERIMENT_REPORT.md"
    report_path.write_text(
        "\n".join(
            [
                "# Final Experiment Report",
                "",
                "This report is generated from real run manifests, summary CSVs, LaTeX files, figure files, and final artifact outputs. It does not fabricate metrics.",
                "",
                "## Run Counts",
                f"- experiments_total: {status_counts.get('total', 0)}",
                f"- succeeded: {status_counts.get('ok', 0) + status_counts.get('exists', 0)}",
                f"- failed: {status_counts.get('failed', 0)}",
                f"- skipped: {status_counts.get('skipped', 0)}",
                f"- unavailable: {status_counts.get('unavailable', 0)}",
                f"- missing: {status_counts.get('missing', 0)}",
                "",
                "## Artifact Checks",
                *_format_checks(checks, allow_missing=allow_missing),
                "",
                "## Tables",
                *[f"- `{path}`" for path in tables],
                "",
                "## Figures",
                *[f"- `{path}`" for path in figures],
                "",
                "## HERO Best Configs",
                *(best_configs or ["- missing: tuning/best_configs not found"]),
                "",
                "## Significance Summary",
                *significance,
                "",
                "## Robustness Summary",
                *robustness,
                "",
                "## Faithfulness Summary",
                *faithfulness,
                "",
                "## Cost Summary",
                *cost,
                "",
                "## Risk Card Case Summary",
                *risk_cards,
                "",
                "## Missing Or Unavailable",
                *missing,
                "",
                "## Recommended Placement",
                "- Main text: table2_main_text_rich, table3_transfer, table4_ablation, table5_evidence_cases if evidence chains are available.",
                "- Supplement: significance, robustness, labeler comparison, faithfulness, sensitivity, cost/scalability, seed stability, missing/skipped runs.",
                "",
                "## Integrity Notes",
                "- Do not edit metric CSVs by hand.",
                "- Do not tune configs on test metrics.",
                "- Missing or unavailable rows should remain marked rather than filled with invented values.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(checks).to_csv(report_dir / "final_artifact_checks.csv", index=False)
    return report_path, checks


def _check_risk_card_artifacts(final: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, rel in RISK_CARD_TABLES.items():
        path = final / rel
        frame = read_csv_or_empty(path)
        status = "ok"
        reason = ""
        if not path.exists():
            status = "missing"
            reason = "file_not_found"
        elif frame.empty:
            status = "missing"
            reason = "empty_csv"
        else:
            text = frame.to_csv(index=False).lower()
            forbidden = _forbidden_terms(text, include_data_terms=True)
            if forbidden:
                status = "failed"
                reason = f"forbidden_terms={','.join(forbidden)}"
            elif "dataset" not in frame.columns:
                status = "failed"
                reason = "dataset_column_missing"
            else:
                present = set(frame["dataset"].astype(str))
                missing = sorted(RISK_CARD_DATASETS - present)
                if missing:
                    status = "missing"
                    reason = f"missing_datasets={missing}"
                elif _contains_missing_markers(frame):
                    status = "warning"
                    reason = "contains_NA_or_unavailable_fields"
        rows.append({"kind": "risk_card", "label": label, "path": str(path), "status": status, "reason": reason, "rows": len(frame)})
    for label, rel in RISK_CARD_LATEX.items():
        path = final / rel
        status = "ok"
        reason = ""
        if not path.exists():
            status = "missing"
            reason = "file_not_found"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            forbidden = _forbidden_terms(text, include_data_terms=False)
            if forbidden:
                status = "failed"
                reason = f"forbidden_terms={','.join(forbidden)}"
            elif "\\begin{tabular}" not in text:
                status = "missing"
                reason = "latex_table_not_detected"
            elif "n/a" in text or "unavailable" in text:
                status = "warning"
                reason = "contains_NA_or_unavailable_fields"
        rows.append({"kind": "risk_card", "label": label, "path": str(path), "status": status, "reason": reason, "rows": ""})
    for label, rel in RISK_CARD_REPORTS.items():
        path = final / rel
        status = "ok"
        reason = ""
        if not path.exists():
            status = "missing"
            reason = "file_not_found"
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            forbidden = _forbidden_terms(text, include_data_terms=False)
            if forbidden:
                status = "failed"
                reason = f"forbidden_terms={','.join(forbidden)}"
            elif "all risk-card cases are extracted from real data or cached model outputs" not in text:
                status = "warning"
                reason = "integrity_statement_missing"
            elif "n/a" in text or "unavailable" in text:
                status = "warning"
                reason = "contains_NA_or_unavailable_fields"
        rows.append({"kind": "risk_card", "label": label, "path": str(path), "status": status, "reason": reason, "rows": ""})
    return rows


def _check_csvs(final: Path, mapping: dict[str, str], required_fields: bool, kind: str = "table") -> list[dict[str, Any]]:
    rows = []
    for label, rel in mapping.items():
        path = final / rel
        frame = read_csv_or_empty(path)
        status = "ok"
        reason = ""
        if not path.exists():
            status = "missing"
            reason = "file_not_found"
        elif frame.empty:
            status = "missing"
            reason = "empty_csv"
        elif required_fields and _important_fields_empty(frame):
            status = "unavailable"
            reason = "important_fields_empty_or_unavailable"
        rows.append({"kind": kind, "label": label, "path": str(path), "status": status, "reason": reason, "rows": len(frame)})
    return rows


def _check_text_files(final: Path, mapping: dict[str, str], kind: str) -> list[dict[str, Any]]:
    rows = []
    for label, rel in mapping.items():
        path = final / rel
        status = "ok"
        reason = ""
        if not path.exists():
            status = "missing"
            reason = "file_not_found"
        else:
            try:
                text = path.read_text(encoding="utf-8")
                if "\\begin{tabular}" not in text:
                    status = "unavailable"
                    reason = "latex_table_not_detected"
            except UnicodeDecodeError:
                status = "failed"
                reason = "cannot_read_utf8"
        rows.append({"kind": kind, "label": label, "path": str(path), "status": status, "reason": reason, "rows": ""})
    return rows


def _check_binary_files(final: Path, mapping: dict[str, str], kind: str) -> list[dict[str, Any]]:
    rows = []
    for label, rel in mapping.items():
        path = final / rel
        status = "ok" if path.exists() and path.stat().st_size > 0 else "missing"
        reason = "" if status == "ok" else "file_not_found_or_empty"
        rows.append({"kind": kind, "label": label, "path": str(path), "status": status, "reason": reason, "rows": ""})
    return rows


def _forbidden_terms(text: str, include_data_terms: bool) -> list[str]:
    terms = list(FORBIDDEN_RISK_CARD_TERMS)
    if include_data_terms:
        terms.extend(FORBIDDEN_RISK_CARD_DATA_TERMS)
    return [term for term in terms if term in text]


def _contains_missing_markers(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    text = frame.fillna("").astype(str).to_csv(index=False).lower()
    return "n/a" in text or "unavailable" in text


def _status_counts(root: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {"total": 0}
    if root is None or not root.exists():
        return counts
    for manifest in root.rglob("run_manifest.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for row in payload.get("runs", []):
            status = str(row.get("status", "missing") or "missing")
            counts[status] = counts.get(status, 0) + 1
            counts["total"] = counts.get("total", 0) + 1
    return counts


def _existing_paths(root: Path, folder: str, pattern: str) -> list[str]:
    base = root / folder
    if not base.exists():
        return []
    return [str(path) for path in sorted(base.glob(pattern))]


def _hero_best_configs(root: Path | None) -> list[str]:
    if root is None:
        return []
    rows = []
    summary = read_csv_or_empty(root / "tuning" / "summary" / "best_hero_configs.csv")
    if not summary.empty:
        for _, row in summary.iterrows():
            rows.append(f"- {row.get('dataset', '')}: status={row.get('status', '')}, config_hash={row.get('config_hash', '')}, selected_by={row.get('selected_by', '')}")
    config_dir = root / "tuning" / "best_configs"
    if config_dir.exists():
        rows.extend(f"- `{path}`" for path in sorted(config_dir.glob("hero_full_*.yaml")))
    return rows


def _table_summary(path: Path, columns: list[str]) -> list[str]:
    frame = read_csv_or_empty(path)
    if frame.empty:
        return [f"- unavailable: `{path}` missing or empty"]
    rows = [f"- source: `{path}`, rows={len(frame)}"]
    for column in columns:
        if column in frame:
            counts = frame[column].fillna("N/A").astype(str).value_counts(dropna=False).head(5)
            rows.append("- " + column + ": " + ", ".join(f"{idx}={value}" for idx, value in counts.items()))
    return rows


def _missing_or_unavailable(final: Path) -> list[str]:
    rows = []
    for csv in sorted((final / "tables_csv").glob("*.csv")) if (final / "tables_csv").exists() else []:
        frame = read_csv_or_empty(csv)
        if frame.empty:
            rows.append(f"- `{csv}`: empty")
            continue
        if "status" in frame:
            subset = frame[frame["status"].astype(str).isin(["missing", "unavailable", "warning", "insufficient_seeds", "not_applicable"])]
            if not subset.empty:
                rows.append(f"- `{csv}`: {len(subset)} warning/missing/unavailable rows")
    return rows or ["- none detected in final tables"]


def _important_fields_empty(frame: pd.DataFrame) -> bool:
    non_status = [column for column in frame.columns if column not in {"status", "reason", "source_dir"}]
    if not non_status:
        return True
    if "status" in frame and frame["status"].astype(str).isin(["missing", "unavailable"]).all():
        return True
    return frame[non_status].replace("", pd.NA).dropna(how="all").empty


def _format_checks(checks: list[dict[str, Any]], allow_missing: bool) -> list[str]:
    lines = []
    for row in checks:
        status = row["status"]
        if allow_missing and status in {"missing", "unavailable"}:
            status = f"{status}_allowed"
        reason = f", reason={row['reason']}" if row.get("reason") else ""
        lines.append(f"- {row['kind']} {row['label']}: {status}{reason}, path=`{row['path']}`")
    return lines


if __name__ == "__main__":
    main()
