from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
