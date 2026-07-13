from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import read_csv_or_empty, write_csv, write_latex  # noqa: E402


OUTPUT_COLUMNS = ["dataset", "target", "neighbor", "mechanism", "risk_relevance", "confidence", "key_evidence", "status"]
CASE_PATTERNS = ["table_evidence_cases.csv", "table_evidence_chain_case.csv", "*evidence*case*.csv"]
CHAIN_PATTERNS = ["evidence_chains.csv", "*evidence_chains*.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence-case table from real evidence chain artifacts.")
    parser.add_argument("--input_dir", default="outputs/submission_unified")
    parser.add_argument("--output_dir", default=None, help="Defaults to --input_dir.")
    parser.add_argument("--max_cases", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_evidence_case_table(args.input_dir, output_dir=args.output_dir, max_cases=args.max_cases)


def build_evidence_case_table(input_dir: str | Path, output_dir: str | Path | None = None, max_cases: int = 20) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    table = _find_existing_case_table(input_dir)
    if table.empty:
        table = _build_from_evidence_chains(input_dir, max_cases=max_cases)
    if table.empty:
        table = pd.DataFrame(
            [
                {
                    "dataset": "",
                    "target": "",
                    "neighbor": "",
                    "mechanism": "",
                    "risk_relevance": pd.NA,
                    "confidence": pd.NA,
                    "key_evidence": "",
                    "status": "unavailable",
                    "reason": "No evidence_chains.csv or evidence case raw file found.",
                }
            ]
        )
    table = _canonical_table(table, max_cases=max_cases)
    write_csv(output_dir / "summary" / "table_evidence_cases.csv", table)
    write_latex(output_dir / "summary" / "table_evidence_cases.tex", table)
    write_csv(output_dir / "tables" / "table_evidence_cases.csv", table)
    write_latex(output_dir / "tables" / "table_evidence_cases.tex", table)
    return table


def _find_existing_case_table(root: Path) -> pd.DataFrame:
    for pattern in CASE_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            if path.name == "table_evidence_cases.csv" and "summary" in path.parts:
                continue
            frame = read_csv_or_empty(path)
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _build_from_evidence_chains(root: Path, max_cases: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pattern in CHAIN_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            frame = read_csv_or_empty(path)
            if frame.empty:
                continue
            dataset = _infer_dataset(path, frame)
            for _, record in frame.head(max_cases - len(rows)).iterrows():
                rows.append(_row_from_chain_record(record.to_dict(), dataset))
                if len(rows) >= max_cases:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _row_from_chain_record(record: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "dataset": _first(record, ["dataset"], default=dataset),
        "target": _first(record, ["target", "target_id", "node", "node_id", "dst", "dst_id"]),
        "neighbor": _first(record, ["neighbor", "neighbor_id", "source", "source_id", "src", "src_id"]),
        "mechanism": _first(record, ["mechanism", "mechanism_label", "label", "relation_type"]),
        "risk_relevance": _first(record, ["risk_relevance", "relevance", "risk_relevance_score", "edge_score"]),
        "confidence": _first(record, ["confidence", "confidence_score", "annotation_confidence"]),
        "key_evidence": _first(record, ["key_evidence", "evidence", "evidence_text", "reason", "explanation"]),
        "status": _first(record, ["status"], default="ok"),
    }


def _canonical_table(table: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    table = table.copy()
    for column in OUTPUT_COLUMNS:
        if column not in table:
            table[column] = pd.NA if column in {"risk_relevance", "confidence"} else ""
    if "status" not in table or table["status"].astype(str).eq("").all():
        table["status"] = "ok"
    keep = OUTPUT_COLUMNS + [column for column in ["reason", "source_file"] if column in table]
    return table[keep].head(max_cases)


def _infer_dataset(path: Path, frame: pd.DataFrame) -> str:
    if "dataset" in frame and not frame["dataset"].dropna().empty:
        return str(frame["dataset"].dropna().iloc[0])
    known = {"yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"}
    for part in path.parts:
        if part in known:
            return part
    return ""


def _first(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in record and pd.notna(record[key]):
            return record[key]
    return default


if __name__ == "__main__":
    main()
