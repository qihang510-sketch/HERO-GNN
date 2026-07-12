import numpy as np
import pandas as pd

from scripts.run_significance_tests import _write_table, significance_rows


def test_significance_rows_use_matched_seeds_only():
    frame = pd.DataFrame(
        [
            {"suite": "main", "dataset": "yelp_academic", "model": "hero_gnn", "seed": 0, "status": "ok", "Macro-F1": 0.60},
            {"suite": "main", "dataset": "yelp_academic", "model": "hero_gnn", "seed": 1, "status": "ok", "Macro-F1": 0.70},
            {"suite": "main", "dataset": "yelp_academic", "model": "hero_gnn", "seed": 2, "status": "ok", "Macro-F1": 0.80},
            {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 0, "status": "ok", "Macro-F1": 0.10},
            {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 2, "status": "ok", "Macro-F1": 0.20},
            {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 3, "status": "ok", "Macro-F1": 0.90},
            {"suite": "main", "dataset": "yelp_academic", "model": "gcn", "seed": 0, "status": "ok", "Macro-F1": 0.10},
            {"suite": "main", "dataset": "yelp_academic", "model": "gcn", "seed": 1, "status": "ok", "Macro-F1": 0.20},
            {"suite": "main", "dataset": "yelp_academic", "model": "gcn", "seed": 2, "status": "ok", "Macro-F1": 0.30},
        ]
    )

    rows = pd.DataFrame(significance_rows(frame, metrics=["Macro-F1"], min_paired_seeds=3))
    mlp = rows[(rows["baseline_model"] == "mlp") & (rows["metric"] == "Macro-F1")].iloc[0]
    gcn = rows[(rows["baseline_model"] == "gcn") & (rows["metric"] == "Macro-F1")].iloc[0]

    assert int(mlp["paired_seed_count"]) == 2
    assert mlp["status"] == "insufficient_seeds"
    assert int(gcn["paired_seed_count"]) == 3
    assert gcn["status"] == "ok"
    assert np.isclose(gcn["delta"], 0.50)


def test_one_seed_significance_is_insufficient_not_crash():
    frame = pd.DataFrame(
        [
            {"suite": "main", "dataset": "yelp_academic", "model": "hero_gnn", "seed": 0, "status": "ok", "AUPRC": 0.7},
            {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 0, "status": "ok", "AUPRC": 0.6},
        ]
    )

    rows = pd.DataFrame(significance_rows(frame, metrics=["AUPRC"], min_paired_seeds=3))
    row = rows[(rows["baseline_model"] == "mlp") & (rows["metric"] == "AUPRC")].iloc[0]

    assert int(row["paired_seed_count"]) == 1
    assert row["status"] == "insufficient_seeds"
    assert row["p_ttest"] == "NA"
    assert row["p_wilcoxon"] == "NA"


def test_significance_markdown_does_not_require_tabulate(tmp_path, monkeypatch):
    table = pd.DataFrame([{"dataset": "x", "metric": "AUPRC", "status": "ok"}])

    def fail_to_markdown(self, *args, **kwargs):
        raise ImportError("tabulate missing")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", fail_to_markdown, raising=False)
    _write_table(table, tmp_path / "table_significance")

    assert (tmp_path / "table_significance.csv").exists()
    assert (tmp_path / "table_significance.md").exists()
    assert (tmp_path / "table_significance.tex").exists()
