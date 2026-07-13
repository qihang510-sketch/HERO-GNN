from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.tune_hero_hyperparams import _best_configs  # noqa: E402


def _candidate(lr: float) -> dict:
    return {
        "learning_rate": lr,
        "weight_decay": 5e-5,
        "hidden_dim": 128,
        "dropout": 0.4,
        "num_layers": 2,
        "neighbor_k": 10,
        "lambda_rel": 0.3,
        "lambda_chain": 0.1,
        "confidence_threshold": 0.3,
        "risk_weight_temperature": 1.0,
        "optimizer": "adamw",
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 30,
        "use_class_weight": True,
    }


def test_tuning_selects_by_validation_metrics_not_test_metrics() -> None:
    candidates = [_candidate(0.001), _candidate(0.003)]
    results = pd.DataFrame(
        [
            {
                "dataset": "yelp_academic",
                "trial_id": 0,
                "seed": 0,
                "status": "ok",
                "val_AUPRC": 0.20,
                "val_AUROC": 0.90,
                "val_Macro-F1": 0.80,
                "AUPRC": 0.99,
            },
            {
                "dataset": "yelp_academic",
                "trial_id": 1,
                "seed": 0,
                "status": "ok",
                "val_AUPRC": 0.30,
                "val_AUROC": 0.50,
                "val_Macro-F1": 0.40,
                "AUPRC": 0.10,
            },
        ]
    )
    best = _best_configs(results, candidates)
    row = best.iloc[0]
    assert int(row["trial_id"]) == 1
    assert row["selected_by"] == "validation_AUPRC"
    assert row["selection_metric"] == "val_AUPRC"
