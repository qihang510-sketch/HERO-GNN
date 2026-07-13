from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts import run_experiment_suite as suite


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(tmp_path / "data"),
        epochs=1,
        lr=0.001,
        hidden_dim=16,
        top_k=2,
        device="cpu",
        llm_label_file=None,
        config=None,
        tuned_config_dir=None,
        hero_gnn_config=None,
        hero_official_config=None,
        skip_existing=False,
        continue_on_error=True,
        save_predictions=False,
        save_embeddings=False,
        save_evidence=False,
    )


def test_failed_run_writes_error_log(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "suite"
    suite._prepare_output_dirs(output_dir)

    def boom(**_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(suite, "run_submission_experiment", boom)
    row = suite._run_main_case(_args(tmp_path), output_dir, "main", "yelp_academic", "mlp", 0)
    assert row["status"] == "failed"
    assert (Path(row["run_dir"]) / "error.log").exists()


def test_skipped_rows_are_not_rewritten_as_failed(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    suite._prepare_output_dirs(output_dir)
    run_dir = output_dir / "raw" / "fraud_yelp" / "dgp" / "seed_0"
    row = suite._ensure_run_artifacts(
        result_dir=run_dir,
        output_dir=output_dir,
        suite="transfer",
        dataset="fraud_yelp",
        model="dgp",
        seed=0,
        status="skipped",
        reason="model_not_applicable_to_dataset",
        started_at="start",
        ended_at="end",
        log_text="",
        args=_args(tmp_path),
    )
    assert row["status"] == "skipped"
    suite._write_failed_runs(output_dir, [row])
    with (output_dir / "failed_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "skipped"
