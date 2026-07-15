from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_representative_risk_cards import (  # noqa: E402
    DEFAULT_DATASETS,
    NA,
    TRACE_COLUMNS,
    ExtractionResult,
    extract_representative_risk_cards,
    write_raw_outputs,
)


COMPACT_COLUMNS = [
    "dataset",
    "target_id",
    "target_label",
    "neighbor_id",
    "neighbor_label",
    "relation_or_path",
    "key_raw_fields",
    "derived_cues",
    "risk_card_summary",
    "mechanism_candidate",
    "risk_relevance",
    "confidence",
    "keep_or_downweight",
    "field_provenance_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build representative risk-card case tables.")
    parser.add_argument("--data_root", default="data", help="Data directory.")
    parser.add_argument("--output_dir", default="outputs/risk_card_cases", help="Risk-card case output directory.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to include.")
    parser.add_argument("--source_outputs", nargs="*", default=[], help="Existing HERO output directories to search.")
    parser.add_argument("--top_cases_per_dataset", type=int, default=1, help="Selected cases per dataset in raw outputs.")
    parser.add_argument("--include_field_trace", action="store_true", help="Write field-level provenance table.")
    parser.add_argument("--write_latex", action="store_true", help="Write LaTeX tables.")
    parser.add_argument("--write_markdown", action="store_true", help="Write Markdown tables.")
    parser.add_argument("--strict", action="store_true", help="Fail if selected real cases have unavailable fields.")
    parser.add_argument("--copy_to_final_artifacts", action="store_true", help="Copy outputs to outputs/final_artifacts/risk_card_cases.")
    parser.add_argument("--final_artifacts_dir", default="outputs/final_artifacts", help="Final artifacts directory used with --copy_to_final_artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_risk_card_case_table(args)


def build_risk_card_case_table(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    dirs = _ensure_dirs(output_dir)
    result = extract_representative_risk_cards(
        data_root=args.data_root,
        datasets=args.datasets,
        source_outputs=args.source_outputs,
        top_candidates_per_dataset=max(3, int(args.top_cases_per_dataset)),
        top_cases_per_dataset=max(1, int(args.top_cases_per_dataset)),
        strict=bool(args.strict),
    )
    write_raw_outputs(result, output_dir)

    compact = _compact_table(result, args.datasets)
    field_trace = _field_trace_table(result)

    paths: dict[str, Path] = {}
    paths["selected_cases"] = dirs["raw"] / "selected_cases.jsonl"
    paths["field_traces_raw"] = dirs["raw"] / "risk_card_field_traces.jsonl"
    paths["compact_csv"] = _write_csv(dirs["tables_csv"] / "table_risk_card_cases_compact.csv", compact)
    if bool(args.include_field_trace):
        paths["field_trace_csv"] = _write_csv(dirs["tables_csv"] / "table_risk_card_field_trace.csv", field_trace)
    if bool(args.write_markdown):
        paths["compact_markdown"] = _write_markdown(dirs["tables_markdown"] / "table_risk_card_cases_compact.md", compact)
        if bool(args.include_field_trace):
            paths["field_trace_markdown"] = _write_markdown(dirs["tables_markdown"] / "table_risk_card_field_trace.md", field_trace)
    if bool(args.write_latex):
        paths["compact_latex"] = _write_latex(
            dirs["tables_latex"] / "table_risk_card_cases_compact.tex",
            compact,
            caption="Representative examples of risk-card construction on five datasets.",
            label="tab:risk_card_cases",
            table_kind="compact",
        )
        if bool(args.include_field_trace):
            paths["field_trace_latex"] = _write_latex(
                dirs["tables_latex"] / "table_risk_card_field_trace.tex",
                field_trace,
                caption="Field-level provenance and computation rules for risk-card construction.",
                label="tab:risk_card_field_trace",
                table_kind="field_trace",
            )
    paths["report"] = _write_report(dirs["reports"] / "RISK_CARD_CASE_REPORT.md", result, paths, args.datasets)
    if bool(args.copy_to_final_artifacts):
        paths["final_artifacts_dir"] = _copy_to_final_artifacts(paths, Path(args.final_artifacts_dir))
    print(f"Compact table: {paths['compact_csv']}")
    if bool(args.include_field_trace):
        print(f"Field trace table: {paths['field_trace_csv']}")
    print(f"Report: {paths['report']}")
    return paths


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


def _compact_table(result: ExtractionResult, datasets: Iterable[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        case = _selected_case(result.cases, dataset)
        rows.append(
            _fill_row(
                {
                    "dataset": dataset,
                    "target_id": case.get("target_id", NA),
                    "target_label": case.get("target_label", NA),
                    "neighbor_id": case.get("neighbor_id", NA),
                    "neighbor_label": case.get("neighbor_label", NA),
                    "relation_or_path": _truncate(case.get("raw_path_example", case.get("relation_type", NA)), 28),
                    "key_raw_fields": _key_raw_fields(case),
                    "derived_cues": _derived_cues(case),
                    "risk_card_summary": _truncate(case.get("risk_summary", NA), 30),
                    "mechanism_candidate": case.get("mechanism_candidate", NA),
                    "risk_relevance": case.get("risk_relevance", NA),
                    "confidence": case.get("confidence", NA),
                    "keep_or_downweight": case.get("keep_or_downweight", NA),
                    "field_provenance_summary": _provenance_summary(case),
                },
                COMPACT_COLUMNS,
            )
        )
    return pd.DataFrame(rows, columns=COMPACT_COLUMNS)


def _field_trace_table(result: ExtractionResult) -> pd.DataFrame:
    rows = [_fill_row(row, TRACE_COLUMNS) for row in result.field_traces]
    return pd.DataFrame(rows, columns=TRACE_COLUMNS)


def _selected_case(cases: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    dataset_cases = [case for case in cases if str(case.get("dataset")) == str(dataset)]
    for case in dataset_cases:
        if _truthy(case.get("is_selected")):
            return case
    return dataset_cases[0] if dataset_cases else {"dataset": dataset, "status": "unavailable"}


def _key_raw_fields(case: dict[str, Any]) -> str:
    parts = [
        f"target_text={_truncate(case.get('target_text_summary', NA), 18)}",
        f"neighbor_text={_truncate(case.get('neighbor_text_summary', NA), 18)}",
        f"rating={case.get('target_rating', NA)}->{case.get('neighbor_rating', NA)}",
        f"time={case.get('target_time', NA)}->{case.get('neighbor_time', NA)}",
        f"item/business={case.get('target_item_or_business', NA)}->{case.get('neighbor_item_or_business', NA)}",
        f"user/account={case.get('target_user_or_account', NA)}->{case.get('neighbor_user_or_account', NA)}",
    ]
    return _truncate("; ".join(parts), 30)


def _derived_cues(case: dict[str, Any]) -> str:
    parts = [
        f"structural={case.get('structural_proximity', NA)}",
        f"common={case.get('common_neighbor_count', NA)}",
        f"jaccard={case.get('jaccard_similarity', NA)}",
        f"neighbor_fraud_ratio={case.get('neighbor_fraud_ratio', NA)}",
        f"rating_gap={case.get('rating_gap', NA)}",
        f"time_gap={case.get('time_gap', NA)}",
        f"behavior_conflict={case.get('behavior_conflict_score', NA)}",
        f"suspicious_paths={case.get('suspicious_path_count', NA)}",
    ]
    return _truncate("; ".join(parts), 30)


def _provenance_summary(case: dict[str, Any]) -> str:
    if str(case.get("status", "ok")) == "unavailable":
        return _truncate(f"unavailable: {case.get('unavailable_reason', case.get('selection_reason', NA))}", 24)
    sources = case.get("source_files", [])
    if not isinstance(sources, list):
        sources = [sources]
    used = [
        f"selection={case.get('selection_source', NA)}",
        f"reconstruction={case.get('reconstruction_source', NA)}",
        f"cached_annotation={case.get('cached_annotation_used', False)}",
        f"HERO_weight={case.get('hero_risk_weight_used', False)}",
        f"evidence_chain={case.get('evidence_chain_used', False)}",
        "sources=" + ", ".join(str(item) for item in sources[:3]),
    ]
    return _truncate("; ".join(used), 30)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.fillna(NA).replace("", NA).to_csv(path, index=False)
    return path


def _write_markdown(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dataframe_to_markdown(frame), encoding="utf-8")
    return path


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    table = frame.fillna(NA).replace("", NA)
    lines = [
        "| " + " | ".join(str(col) for col in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[col]) for col in table.columns) + " |")
    return "\n".join(lines) + "\n"


def _write_latex(path: Path, frame: pd.DataFrame, caption: str, label: str, table_kind: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dataframe_to_latex(frame, caption=caption, label=label, table_kind=table_kind), encoding="utf-8")
    return path


def _dataframe_to_latex(frame: pd.DataFrame, caption: str, label: str, table_kind: str) -> str:
    table = frame.fillna(NA).replace("", NA)
    if table_kind == "compact":
        colspec = (
            "p{0.050\\textwidth}p{0.060\\textwidth}p{0.030\\textwidth}"
            "p{0.060\\textwidth}p{0.030\\textwidth}p{0.065\\textwidth}"
            "p{0.115\\textwidth}p{0.105\\textwidth}p{0.105\\textwidth}"
            "p{0.065\\textwidth}p{0.035\\textwidth}p{0.035\\textwidth}"
            "p{0.050\\textwidth}p{0.095\\textwidth}"
        )
    else:
        colspec = (
            "p{0.055\\textwidth}p{0.070\\textwidth}p{0.065\\textwidth}"
            "p{0.120\\textwidth}p{0.080\\textwidth}p{0.080\\textwidth}"
            "p{0.130\\textwidth}p{0.070\\textwidth}p{0.085\\textwidth}"
            "p{0.075\\textwidth}p{0.115\\textwidth}"
        )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
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
        cells = [_latex_escape(_truncate(row[col], 24 if table_kind == "field_trace" else 22)) for col in table.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _write_report(path: Path, result: ExtractionResult, artifact_paths: dict[str, Path], datasets: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Risk Card Case Report",
        "",
        "All risk-card cases are extracted from real data or cached model outputs. Missing fields are marked as N/A rather than imputed.",
        "",
        "## Artifact Paths",
        "",
    ]
    for key in [
        "selected_cases",
        "field_traces_raw",
        "compact_csv",
        "field_trace_csv",
        "compact_latex",
        "field_trace_latex",
        "compact_markdown",
        "field_trace_markdown",
    ]:
        if key in artifact_paths:
            lines.append(f"- {key}: `{_rel(artifact_paths[key])}`")
    lines.extend(["", "## Dataset Cases", ""])
    report_by_dataset = {row["dataset"]: row for row in result.dataset_reports}
    for dataset in datasets:
        report = report_by_dataset.get(dataset, {"dataset": dataset, "status": "unavailable"})
        unavailable = report.get("unavailable_fields", [])
        if isinstance(unavailable, list):
            unavailable_text = ", ".join(str(item) for item in unavailable[:24])
            if len(unavailable) > 24:
                unavailable_text += ", ..."
        else:
            unavailable_text = str(unavailable)
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- status: {report.get('status', NA)}",
                f"- selected target-neighbor pair: `{report.get('target_id', NA)}` -> `{report.get('neighbor_id', NA)}`",
                f"- target_label / neighbor_label: {report.get('target_label', NA)} / {report.get('neighbor_label', NA)}",
                f"- relation/path: {report.get('relation_type', NA)}",
                f"- why selected: {report.get('selection_reason', report.get('reason', NA))}",
                f"- field sources: {', '.join(str(item) for item in report.get('source_files', ['unavailable']))}",
                f"- unavailable fields: {unavailable_text or NA}",
                f"- used Qwen/local LLM annotation: {bool(report.get('qwen_annotation_used', False))}",
                f"- used cached annotations: {bool(report.get('cached_annotation_used', False))}",
                f"- used HERO risk weights: {bool(report.get('hero_risk_weight_used', False))}",
                f"- used evidence chains: {bool(report.get('evidence_chain_used', False))}",
                f"- reconstruction_source: {report.get('reconstruction_source', NA)}",
                f"- processed_dir: `{report.get('processed_dir', NA)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _copy_to_final_artifacts(paths: dict[str, Path], final_dir: Path) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "compact_csv": final_dir / "tables_csv" / "supp_table_risk_card_cases_compact.csv",
        "field_trace_csv": final_dir / "tables_csv" / "supp_table_risk_card_field_trace.csv",
        "compact_latex": final_dir / "tables_latex" / "supp_table_risk_card_cases_compact.tex",
        "field_trace_latex": final_dir / "tables_latex" / "supp_table_risk_card_field_trace.tex",
        "report": final_dir / "reports" / "RISK_CARD_CASE_REPORT.md",
    }
    for key, target in mapping.items():
        source = paths.get(key)
        if not isinstance(source, Path) or not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return final_dir


def _fill_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {column: _cell(row.get(column, NA)) for column in columns}


def _cell(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float) and pd.isna(value):
        return NA
    text = str(value)
    if text.strip() == "" or text.lower() in {"nan", "none", "<na>"}:
        return NA
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _truncate(value: Any, max_words: int = 30) -> str:
    if value is None:
        return NA
    text = str(value).replace("\n", " ").strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return NA
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _markdown_cell(value: Any) -> str:
    return _truncate(value, 34).replace("|", "\\|")


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


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path.as_posix()


if __name__ == "__main__":
    main()
