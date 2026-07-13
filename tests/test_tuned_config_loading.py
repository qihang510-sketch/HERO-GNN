from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_experiment_suite import _tuned_config_for_dataset  # noqa: E402
from src.training.submission import _load_hero_runtime_config  # noqa: E402


def test_tuned_yaml_hero_block_is_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "hero_full_yelp_academic.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "selection": {"selected_by": "validation_AUPRC", "best_epoch": 3},
                "trainer": {"lr": 0.003, "hidden_dim": 128, "top_k": 15},
                "hero": {
                    "optimizer": "adamw",
                    "scheduler": "reduce_on_plateau",
                    "early_stopping_patience": 12,
                    "use_class_weight": True,
                    "risk_weight_temperature": 0.5,
                    "weight_decay": 5e-5,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    hero_config, trainer_overrides, source = _load_hero_runtime_config(config_path, base_model="hero_gnn", variant="hero_full")
    assert hero_config["optimizer"] == "adamw"
    assert hero_config["early_stopping_patience"] == 12
    assert hero_config["use_class_weight"] is True
    assert hero_config["risk_weight_temperature"] == 0.5
    assert hero_config["weight_decay"] == 5e-5
    assert trainer_overrides["lr"] == 0.003
    assert trainer_overrides["hidden_dim"] == 128
    assert trainer_overrides["top_k"] == 15
    assert str(config_path) in source


def test_tuned_config_for_dataset_finds_best_config(tmp_path: Path) -> None:
    best_dir = tmp_path / "best_configs"
    best_dir.mkdir()
    expected = best_dir / "hero_full_yelp_academic.yaml"
    expected.write_text("hero: {}\n", encoding="utf-8")
    found = _tuned_config_for_dataset(str(tmp_path), "yelp_academic", "hero_full")
    assert found == expected
