import numpy as np
import pandas as pd

from scripts.run_significance_tests import significance_rows


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
