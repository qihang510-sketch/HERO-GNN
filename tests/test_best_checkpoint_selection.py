from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training.trainer import (  # noqa: E402
    _checkpoint_training_metrics,
    _pos_weight_from_strategy,
    _validation_monitor_score,
)


def test_validation_monitor_prefers_requested_validation_metric() -> None:
    metrics = {"auprc": 0.42, "auroc": 0.91, "macro_f1": 0.55}
    assert _validation_monitor_score(metrics, "AUPRC") == 0.42
    assert _validation_monitor_score(metrics, "AUROC") == 0.91
    assert _validation_monitor_score(metrics, "Macro-F1") == 0.55


def test_checkpoint_training_metrics_exports_validation_selection_fields() -> None:
    checkpoint = {
        "diagnostics": {
            "best_epoch": 7,
            "best_val_AUPRC": 0.61,
            "best_val_AUROC": 0.82,
            "selected_by": "validation_AUPRC",
            "early_stopped": True,
        }
    }
    exported = _checkpoint_training_metrics(checkpoint)
    assert exported["best_epoch"] == 7
    assert exported["best_val_AUPRC"] == 0.61
    assert exported["selected_by"] == "validation_AUPRC"
    assert exported["early_stopped"] is True


def test_class_weight_uses_train_label_counts() -> None:
    assert _pos_weight_from_strategy(num_pos=2, num_neg=8, strategy="inverse_frequency") == 4.0
    assert _pos_weight_from_strategy(num_pos=2, num_neg=8, strategy="none") == 1.0
