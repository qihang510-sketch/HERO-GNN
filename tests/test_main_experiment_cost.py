from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.collect_main_experiment_cost import collect_main_experiment_cost
from scripts.validate_main_cost_table import validate_main_cost_table


DATASETS = ["yelp_academic", "amazon_video"]
REPO = Path(__file__).resolve().parents[1]


def test_runtime_json_generates_cost_tables_and_stats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_run(source, "main", "yelp_academic", "gcn", 0, train=10.0, inference=2.0, total=20.0, gpu=1000.0)
    _write_run(source, "main", "yelp_academic", "gcn", 1, train=14.0, inference=4.0, total=30.0, gpu=1200.0)
    _write_run(source, "main", "yelp_academic", "hero_gnn", 0, train=30.0, inference=5.0, total=50.0, gpu=None)
    _write_run(source, "main_quick", "yelp_academic", "gcn", 99, train=1.0, inference=1.0, total=2.0, gpu=10.0)
    _write_run(source, "transfer", "amazon_video", "gcn", 0, train=99.0, inference=1.0, total=100.0, gpu=99.0)
    out = tmp_path / "main_cost"
    final = tmp_path / "final"

    collect_main_experiment_cost(_args(source, out, final))

    detail = pd.read_csv(out / "tables_csv" / "supp_table_main_experiment_cost.csv", keep_default_na=False)
    compact = pd.read_csv(out / "tables_csv" / "supp_table_main_experiment_cost_compact.csv", keep_default_na=False)

    gcn = detail[(detail["dataset"] == "yelp_academic") & (detail["model"] == "GCN") & (detail["status"] == "ok")].iloc[0]
    assert float(gcn["train_time_sec_mean"]) == 12.0
    assert float(gcn["train_time_sec_std"]) == 2.83
    assert float(gcn["total_runtime_sec_sum"]) == 50.0
    assert float(gcn["gpu_memory_mb_max"]) == 1200.0
    assert int(gcn["seed_count"]) == 2

    hero = detail[(detail["dataset"] == "yelp_academic") & (detail["model"] == "HERO")].iloc[0]
    assert hero["annotation_source"] == "unavailable"
    assert hero["gpu_memory_mb_mean"] == "N/A"

    baseline_compact = compact[(compact["Dataset"] == "Yelp Academic") & (compact["Model"] == "GCN")].iloc[0]
    assert baseline_compact["LLM Annotation Cost"] == "N/A"
    assert "quick" not in compact.to_csv(index=False).lower()

    assert (out / "tables_latex" / "supp_table_main_experiment_cost.tex").exists()
    assert (out / "tables_latex" / "supp_table_main_experiment_cost_compact.tex").exists()
    assert (out / "tables_markdown" / "supp_table_main_experiment_cost.md").exists()
    assert (out / "tables_markdown" / "supp_table_main_experiment_cost_compact.md").exists()
    assert (final / "tables_csv" / "supp_table_main_experiment_cost.csv").exists()
    assert (final / "tables_latex" / "supp_table_main_experiment_cost_compact.tex").exists()
    assert validate_main_cost_table(out, DATASETS) == []


def test_single_seed_std_is_na_and_missing_gpu_does_not_fail(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_run(source, "main", "amazon_video", "mlp", 0, train=8.0, inference=None, total=10.0, gpu=None)
    out = tmp_path / "main_cost"

    collect_main_experiment_cost(_args(source, out, tmp_path / "final"))

    detail = pd.read_csv(out / "tables_csv" / "supp_table_main_experiment_cost.csv", keep_default_na=False)
    mlp = detail[(detail["dataset"] == "amazon_video") & (detail["model"] == "MLP")].iloc[0]
    assert mlp["train_time_sec_std"] == "N/A"
    assert mlp["gpu_memory_mb_mean"] == "N/A"
    assert mlp["status"] == "ok"


def test_qwen_cache_is_only_reported_when_cache_indicates_qwen(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_run(source, "main", "yelp_academic", "hero_gnn", 0, train=20.0, inference=2.0, total=30.0, gpu=512.0)
    annotations = source / "annotations" / "yelp_academic" / "qwen_annotations.jsonl"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(
        json.dumps({"dataset": "yelp_academic", "risk_relevance": 1, "confidence": 0.9, "labeler": "local_qwen"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "main_cost"

    collect_main_experiment_cost(_args(source, out, tmp_path / "final"))

    detail = pd.read_csv(out / "tables_csv" / "supp_table_main_experiment_cost.csv", keep_default_na=False)
    hero = detail[(detail["dataset"] == "yelp_academic") & (detail["model"] == "HERO")].iloc[0]
    assert hero["annotation_source"] == "cached_local_qwen"
    assert int(float(hero["annotation_cards"])) == 1
    compact = pd.read_csv(out / "tables_csv" / "supp_table_main_experiment_cost_compact.csv", keep_default_na=False)
    hero_compact = compact[(compact["Dataset"] == "Yelp Academic") & (compact["Model"] == "HERO")].iloc[0]
    assert hero_compact["LLM Annotation Cost"] == "local_qwen, offline"


def test_validator_rejects_forecast_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_run(source, "main", "yelp_academic", "gcn", 0, train=1.0, inference=1.0, total=2.0, gpu=64.0)
    out = tmp_path / "main_cost"
    collect_main_experiment_cost(_args(source, out, tmp_path / "final"))
    compact_path = out / "tables_csv" / "supp_table_main_experiment_cost_compact.csv"
    compact = pd.read_csv(compact_path, keep_default_na=False)
    compact.loc[0, "Status"] = "forecast"
    compact.to_csv(compact_path, index=False)

    errors = validate_main_cost_table(out, DATASETS)

    assert any("forecast" in error for error in errors)


def test_run_stage_main_cost_dry_run_outputs_commands(tmp_path: Path) -> None:
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
            "main_cost",
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
    assert "scripts/collect_main_experiment_cost.py" in result.stdout
    assert "scripts/validate_main_cost_table.py" in result.stdout
    assert "--copy_to_final_artifacts" in result.stdout


def _args(source: Path, out: Path, final: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_outputs=[str(source)],
        output_dir=str(out),
        datasets=DATASETS,
        suite_names=["main", "main_text_rich"],
        models=["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero", "hero_full", "hero_gnn"],
        data_root=str(source / "data"),
        include_quick_test=False,
        include_failed=True,
        exclude_failed=False,
        write_latex=True,
        write_markdown=True,
        strict=False,
        copy_to_final_artifacts=True,
        final_artifacts_dir=str(final),
    )


def _write_run(
    source: Path,
    suite: str,
    dataset: str,
    model: str,
    seed: int,
    train: float | None,
    inference: float | None,
    total: float | None,
    gpu: float | None,
) -> None:
    run_dir = source / "raw" / suite / dataset / model / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"suite": suite, "dataset": dataset, "model": model, "seed": seed}), encoding="utf-8")
    runtime = {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "status": "ok",
        "train_time_sec": train,
        "inference_time_sec": inference,
        "total_runtime_sec": total,
        "peak_gpu_memory_mb": gpu,
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    metrics = {"suite": suite, "dataset": dataset, "model": model, "seed": seed, "status": "ok", "Macro-F1": 0.7, "AUROC": 0.8, "AUPRC": 0.6}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
