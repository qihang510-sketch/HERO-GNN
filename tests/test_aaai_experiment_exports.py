import json

import pandas as pd

from scripts.export_paper_tables import export_paper_tables
from scripts.run_significance_test import significance_rows
from scripts.summarize_results import main as summarize_main


def test_export_paper_tables_tolerates_missing_inputs(tmp_path):
    summary_dir = tmp_path / "summary"
    out_dir = tmp_path / "paper_tables"
    export_paper_tables(summary_dir, out_dir)

    expected = [
        "table_main.csv",
        "table_ablation.csv",
        "table_neighbor_strategy.csv",
        "table_official_fraud.csv",
        "table_scalability.csv",
        "table_significance.csv",
        "table_runtime.csv",
        "table_labeler_comparison.csv",
    ]
    for name in expected:
        assert (out_dir / name).exists()


def test_summarize_results_writes_runtime_table(tmp_path, monkeypatch):
    result_dir = tmp_path / "results"
    metrics_path = result_dir / "synthetic" / "mlp" / "seed_0" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "model": "mlp",
                "seed": 0,
                "macro_f1": 0.5,
                "auroc": 0.6,
                "auprc": 0.7,
                "time_total_sec": 3.0,
                "time_retrieval_sec": 1.0,
                "time_training_sec": 2.0,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "summary"
    monkeypatch.setattr("sys.argv", ["summarize_results.py", "--result_dir", str(result_dir), "--out_dir", str(out_dir)])
    summarize_main()

    runtime = pd.read_csv(out_dir / "runtime_table.csv")
    assert runtime.loc[0, "time_total_mean"] == 3.0
    assert runtime.loc[0, "time_retrieval_mean"] == 1.0
    assert runtime.loc[0, "time_training_mean"] == 2.0


def test_significance_rows_compare_hero_to_best_baseline():
    frame = pd.DataFrame(
        [
            {"dataset": "d", "model": "hero_gnn", "seed": 0, "auprc": 0.8},
            {"dataset": "d", "model": "hero_gnn", "seed": 1, "auprc": 0.9},
            {"dataset": "d", "model": "mlp", "seed": 0, "auprc": 0.4},
            {"dataset": "d", "model": "mlp", "seed": 1, "auprc": 0.5},
            {"dataset": "d", "model": "flag_lite", "seed": 0, "auprc": 0.7},
            {"dataset": "d", "model": "flag_lite", "seed": 1, "auprc": 0.75},
        ]
    )
    rows = significance_rows(frame, ["auprc"])

    assert rows[0]["baseline_name"] == "flag_lite"
    assert rows[0]["delta"] > 0
    assert 0.0 <= rows[0]["p_value"] <= 1.0
