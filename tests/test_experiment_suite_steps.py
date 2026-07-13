from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.training.submission import run_submission_experiment

REPO = Path(__file__).resolve().parents[1]


def test_every_suite_dry_run_exits_zero() -> None:
    suites = [
        "main",
        "transfer",
        "ablation",
        "robustness",
        "labeler_comparison",
        "faithfulness",
        "sensitivity",
        "cost",
        "final",
        "all",
    ]
    for suite in suites:
        result = subprocess.run(
            [sys.executable, "scripts/run_experiment_suite.py", "--suite", suite, "--quick_test", "--dry_run"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "planned_runs=" in result.stdout


def test_transfer_dry_run_keeps_non_applicable_models_in_plan() -> None:
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
            "mled",
            "hero",
            "--seeds",
            "0",
            "--dry_run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model=dgp" in result.stdout
    assert "model=mled" in result.stdout
    assert "model=hero_full" in result.stdout


def test_quick_transfer_uses_transfer_dataset() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment_suite.py", "--suite", "transfer", "--quick_test", "--dry_run"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "transfer dataset=fraud_yelp" in result.stdout
    assert "transfer dataset=yelp_academic" not in result.stdout


def test_hero_full_applicable_to_main_and_transfer_datasets(tmp_path: Path) -> None:
    for dataset in ["yelp_academic", "fraud_yelp"]:
        result = run_submission_experiment(
            dataset=dataset,
            model="hero_full",
            seed=0,
            output_dir=tmp_path / "raw",
            data_root=tmp_path / "data",
            epochs=1,
            device="cpu",
        )
        assert result.reason != "model_not_applicable_to_dataset"
