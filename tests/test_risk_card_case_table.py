from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_risk_card_case_table import build_risk_card_case_table
from scripts.validate_risk_card_cases import validate_risk_card_cases


DATASETS = ["yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"]
REPO = Path(__file__).resolve().parents[1]


def test_synthetic_yelp_case_generates_compact_and_field_trace(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_processed_yelp(data_root, with_text=True, with_annotation=True)
    out = tmp_path / "risk_cards"
    final = tmp_path / "final"

    build_risk_card_case_table(_args(data_root, out, final))

    compact = pd.read_csv(out / "tables_csv" / "table_risk_card_cases_compact.csv", keep_default_na=False)
    trace = pd.read_csv(out / "tables_csv" / "table_risk_card_field_trace.csv", keep_default_na=False)

    assert set(compact["dataset"]) == set(DATASETS)
    yelp = compact[compact["dataset"] == "yelp_academic"].iloc[0]
    assert yelp["target_id"] == "yr_target"
    assert yelp["neighbor_id"] == "yr_neighbor"
    assert "planning_only" not in compact.to_csv(index=False).lower()
    assert "forecast" not in compact.to_csv(index=False).lower()
    assert {"source_column_or_file", "computation_rule"}.issubset(trace.columns)
    assert (trace.groupby("dataset").size() >= 8).all()
    assert (final / "tables_csv" / "supp_table_risk_card_cases_compact.csv").exists()
    assert (final / "tables_latex" / "supp_table_risk_card_field_trace.tex").exists()


def test_missing_text_field_outputs_na(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_processed_yelp(data_root, with_text=False, with_annotation=True)
    out = tmp_path / "risk_cards"

    build_risk_card_case_table(_args(data_root, out, tmp_path / "final"))

    compact = pd.read_csv(out / "tables_csv" / "table_risk_card_cases_compact.csv", keep_default_na=False)
    yelp = compact[compact["dataset"] == "yelp_academic"].iloc[0]
    assert "target_text=N/A" in yelp["key_raw_fields"]


def test_missing_annotation_source_is_unavailable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_processed_yelp(data_root, with_text=True, with_annotation=False)
    out = tmp_path / "risk_cards"

    build_risk_card_case_table(_args(data_root, out, tmp_path / "final"))

    selected = _read_jsonl(out / "raw" / "selected_cases.jsonl")
    yelp = next(row for row in selected if row["dataset"] == "yelp_academic")
    assert yelp["annotation_source"] == "unavailable"
    assert yelp["selection_source"] == "heuristic_only"


def test_validator_accepts_generated_unavailable_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_processed_yelp(data_root, with_text=True, with_annotation=True)
    out = tmp_path / "risk_cards"
    build_risk_card_case_table(_args(data_root, out, tmp_path / "final"))

    errors = validate_risk_card_cases(out, DATASETS)

    assert errors == []


def test_validator_flags_forbidden_data_terms(tmp_path: Path) -> None:
    case_dir = tmp_path / "risk_cards"
    _write_minimal_invalid_case_dir(case_dir, forbidden_value="forecast")

    errors = validate_risk_card_cases(case_dir, DATASETS)

    assert any("forecast" in error for error in errors)


def test_validator_flags_manual_case_markers(tmp_path: Path) -> None:
    case_dir = tmp_path / "risk_cards"
    _write_minimal_invalid_case_dir(case_dir, forbidden_value="manual_example")

    errors = validate_risk_card_cases(case_dir, DATASETS)

    assert any("manual_example" in error or "example" in error for error in errors)


def test_run_stage_risk_card_cases_dry_run_outputs_commands(tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    smoke = subprocess.run(["bash", "--version"], capture_output=True, check=False)
    if smoke.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")
    result = subprocess.run(
        [
            "bash",
            "scripts/run_stage.sh",
            "--stage",
            "risk_card_cases",
            "--device",
            "cuda",
            "--output_root",
            str(tmp_path / "stage_root"),
            "--dry_run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/build_risk_card_case_table.py" in result.stdout
    assert "scripts/validate_risk_card_cases.py" in result.stdout
    assert "--copy_to_final_artifacts" in result.stdout


def _args(data_root: Path, output_dir: Path, final_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(data_root),
        output_dir=str(output_dir),
        datasets=DATASETS,
        source_outputs=[],
        top_cases_per_dataset=1,
        include_field_trace=True,
        write_latex=True,
        write_markdown=True,
        strict=False,
        copy_to_final_artifacts=True,
        final_artifacts_dir=str(final_dir),
    )


def _write_processed_yelp(data_root: Path, with_text: bool, with_annotation: bool) -> None:
    processed = data_root / "processed" / "yelp_academic"
    processed.mkdir(parents=True)
    node_rows = [
        {
            "node_id": "yr_target",
            "node_type": "review",
            "label": 1,
            "split": "train",
            "timestamp": 1000,
            "feat_0": 1.0,
        },
        {
            "node_id": "yr_neighbor",
            "node_type": "review",
            "label": 0,
            "split": "train",
            "timestamp": 1010,
            "feat_0": 5.0,
        },
    ]
    if with_text:
        node_rows[0]["text"] = "target review reports a severe delivery failure"
        node_rows[1]["text"] = "neighbor review praises the same account context"
    pd.DataFrame(node_rows).to_csv(processed / "nodes.csv", index=False)
    pd.DataFrame(
        [
            {"src": "yr_target", "dst": "yr_neighbor", "edge_type": "review-user-review", "timestamp": 1000},
            {"src": "yr_neighbor", "dst": "yr_target", "edge_type": "review-user-review", "timestamp": 1000},
        ]
    ).to_csv(processed / "edges.csv", index=False)
    np.savez_compressed(
        processed / "features.npz",
        node_ids=np.array(["yr_target", "yr_neighbor"], dtype=object),
        features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        text_features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        numeric_features=np.array([[1.0], [5.0]], dtype=np.float32),
        numeric_columns=np.array(["feat_0"], dtype=object),
    )
    (processed / "split.json").write_text(json.dumps({"train": ["yr_target", "yr_neighbor"], "val": [], "test": []}), encoding="utf-8")
    (processed / "preprocess_report.json").write_text(json.dumps({"dataset": "yelp_academic"}), encoding="utf-8")
    if with_annotation:
        label = {
            "target_id": "yr_target",
            "neighbor_id": "yr_neighbor",
            "metapath": "review-user-review",
            "mechanism": "behavioral_contradiction",
            "risk_relevance": 1,
            "confidence": 0.9,
            "rationale": "cached annotation supports this pair",
            "labeler_version": "risk_card_v2",
            "risk_card": {
                "target_id": "yr_target",
                "neighbor_id": "yr_neighbor",
                "candidate_score": 0.95,
                "structural_score": 0.95,
                "semantic_similarity": 0.1,
                "numeric_deviation": 0.8,
                "burst_score": 0.0,
                "same_user": True,
                "same_item_or_business": False,
            },
        }
        (processed / "llm_labels.jsonl").write_text(json.dumps(label) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_minimal_invalid_case_dir(case_dir: Path, forbidden_value: str) -> None:
    (case_dir / "tables_csv").mkdir(parents=True)
    (case_dir / "raw").mkdir(parents=True)
    (case_dir / "reports").mkdir(parents=True)
    compact = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "target_id": "yr_target",
                "target_label": 1,
                "neighbor_id": "yr_neighbor",
                "neighbor_label": 0,
                "relation_or_path": "review-user-review",
                "key_raw_fields": forbidden_value if dataset == "yelp_academic" else "N/A",
                "derived_cues": "N/A",
                "risk_card_summary": "N/A",
                "mechanism_candidate": "N/A",
                "risk_relevance": "N/A",
                "confidence": "N/A",
                "keep_or_downweight": "N/A",
                "field_provenance_summary": "N/A",
            }
            for dataset in DATASETS
        ]
    )
    compact.to_csv(case_dir / "tables_csv" / "table_risk_card_cases_compact.csv", index=False)
    trace_rows = []
    for dataset in DATASETS:
        for idx in range(8):
            trace_rows.append(
                {
                    "dataset": dataset,
                    "case_id": f"{dataset}_case",
                    "field_name": f"field_{idx}",
                    "source_column_or_file": "unavailable",
                    "raw_value_target": "N/A",
                    "raw_value_neighbor": "N/A",
                    "computation_rule": "N/A",
                    "computed_value": "N/A",
                    "threshold_or_normalization": "N/A",
                    "risk_card_slot": "raw evidence",
                    "risk_interpretation": "not available in this dataset",
                }
            )
    pd.DataFrame(trace_rows).to_csv(case_dir / "tables_csv" / "table_risk_card_field_trace.csv", index=False)
    selected = [
        {"dataset": dataset, "case_id": f"{dataset}_case", "is_selected": True, "status": "unavailable", "source_files": ["unavailable"]}
        for dataset in DATASETS
    ]
    (case_dir / "raw" / "selected_cases.jsonl").write_text("\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8")
    (case_dir / "raw" / "risk_card_field_traces.jsonl").write_text("\n".join(json.dumps(row) for row in trace_rows) + "\n", encoding="utf-8")
    (case_dir / "reports" / "RISK_CARD_CASE_REPORT.md").write_text(
        "All risk-card cases are extracted from real data or cached model outputs. Missing fields are marked as N/A rather than imputed.\n",
        encoding="utf-8",
    )
