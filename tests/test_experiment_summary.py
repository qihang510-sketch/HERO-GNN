import json

import numpy as np

from scripts.check_experiment_completeness import completeness_table
from scripts.summarize_experiment_suite import summarize_suite


def test_suite_summary_mean_std_and_missing_runs(tmp_path):
    output_dir = tmp_path / "suite"
    expected = [
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 0},
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 1},
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 2},
    ]
    _write_json(output_dir / "config.json", {"expected_runs": expected})
    _write_run(output_dir, seed=0, macro_f1=0.70, auroc=0.80, auprc=0.90)
    _write_run(output_dir, seed=1, macro_f1=0.90, auroc=0.70, auprc=0.80)

    outputs = summarize_suite(output_dir)
    main = outputs["table_main_mean_std"]
    row = main[(main["dataset"] == "yelp_academic") & (main["model"] == "mlp")].iloc[0]

    assert row["seed_count"] == 2
    assert row["Macro-F1_mean"] == 0.80
    assert np.isclose(row["Macro-F1_std"], np.std([0.70, 0.90], ddof=1))
    assert row["Macro-F1_mean_std"] == "80.00 ± 14.14"
    assert (output_dir / "summary" / "all_raw_runs.csv").exists()
    assert (output_dir / "tables" / "table_main_mean_std.csv").exists()

    missing = outputs["table_missing_runs"]
    assert len(missing) == 1
    assert int(missing.iloc[0]["seed"]) == 2
    assert missing.iloc[0]["status"] == "missing"


def test_completeness_table_detects_absent_raw_run(tmp_path):
    output_dir = tmp_path / "suite"
    _write_run(output_dir, seed=0, macro_f1=0.70, auroc=0.80, auprc=0.90)
    expected = [
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 0},
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 1},
    ]

    missing = completeness_table(output_dir, expected=expected)

    assert len(missing) == 1
    assert int(missing.iloc[0]["seed"]) == 1
    assert missing.iloc[0]["reason"] == "raw_result_absent"


def _write_run(output_dir, seed: int, macro_f1: float, auroc: float, auprc: float) -> None:
    run_dir = output_dir / "raw" / "yelp_academic" / "mlp" / f"seed_{seed}"
    _write_json(
        run_dir / "config.json",
        {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": seed},
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "suite": "main",
            "dataset": "yelp_academic",
            "model": "mlp",
            "seed": seed,
            "status": "ok",
            "Macro-F1": macro_f1,
            "AUROC": auroc,
            "AUPRC": auprc,
        },
    )


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
