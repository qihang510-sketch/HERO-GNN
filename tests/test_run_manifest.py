from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_actual_skipped_run_writes_required_output_structure(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_suite.py",
            "--suite",
            "transfer",
            "--datasets",
            "fraud_yelp",
            "--models",
            "dgp",
            "--seeds",
            "0",
            "--output_dir",
            str(output_dir),
            "--data_root",
            str(tmp_path / "missing_data"),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for child in ["raw", "logs", "summary", "tables", "figures", "figure_data", "configs", "reports"]:
        assert (output_dir / child).is_dir()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_recorded_runs"] == 1
    assert manifest["runs"][0]["status"] == "skipped"
    assert (output_dir / "failed_runs.csv").exists()


def test_skipped_run_has_per_run_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_suite.py",
            "--suite",
            "transfer",
            "--datasets",
            "fraud_yelp",
            "--models",
            "dgp",
            "--seeds",
            "0",
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    run_dir = output_dir / "raw" / "fraud_yelp" / "dgp" / "seed_0"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "runtime.json").exists()
    assert (run_dir / "log.txt").exists()
