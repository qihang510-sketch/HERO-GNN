import argparse
import json

import numpy as np
import pandas as pd

from scripts.collect_cost_scalability import collect_cost_scalability
from scripts.run_llm_annotation_robustness import perturb_labels, robustness_plot_data


def test_robustness_perturbations_are_deterministic():
    labels = [
        {
            "dataset": "yelp_academic",
            "target_id": f"t{i}",
            "neighbor_id": f"n{i}",
            "mechanism": "behavioral_contradiction",
            "risk_relevance": 1,
            "confidence": 0.8,
            "rationale": "",
        }
        for i in range(10)
    ]

    flipped = perturb_labels(labels, "relevance_flip", 0.2, seed=7)
    shuffled = perturb_labels(labels, "mechanism_shuffle", 0.2, seed=7)
    noisy = perturb_labels(labels, "confidence_gaussian", 0.1, seed=7)

    assert sum(label["risk_relevance"] == 0 for label in flipped) == 2
    assert sum(label["mechanism"] != "behavioral_contradiction" for label in shuffled) == 2
    assert any(label["confidence"] != 0.8 for label in noisy)
    assert all(0.0 <= label["confidence"] <= 1.0 for label in noisy)


def test_robustness_plot_data_computes_drop_from_zero_noise():
    raw = pd.DataFrame(
        [
            {"dataset": "yelp_academic", "seed": 0, "noise_type": "relevance_flip", "noise_ratio": 0.0, "annotation_source": "cached_llm", "status": "ok", "Macro-F1": 0.8, "AUROC": 0.9, "AUPRC": 0.7},
            {"dataset": "yelp_academic", "seed": 0, "noise_type": "relevance_flip", "noise_ratio": 0.2, "annotation_source": "cached_llm", "status": "ok", "Macro-F1": 0.6, "AUROC": 0.8, "AUPRC": 0.5},
        ]
    )

    plot = robustness_plot_data(raw)
    row = plot[plot["noise_ratio"] == 0.2].iloc[0]

    assert np.isclose(row["Macro-F1_drop"], 0.2)
    assert np.isclose(row["AUPRC_drop"], 0.2)


def test_collect_cost_scalability_reads_real_metrics_and_cache_counts(tmp_path):
    output_dir = tmp_path / "outputs"
    data_root = tmp_path / "data"
    run_dir = output_dir / "raw" / "yelp_academic" / "hero_gnn" / "seed_0"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metrics.json",
        {
            "dataset": "yelp_academic",
            "model": "hero_gnn",
            "seed": 0,
            "status": "ok",
            "Training time": 12.0,
            "time_mock_labeling_sec": 3.0,
            "device": "cpu",
        },
    )
    processed = data_root / "processed" / "yelp_academic"
    processed.mkdir(parents=True)
    _write_jsonl(processed / "risk_cards.jsonl", [{"target_id": "a", "neighbor_id": "b"}, {"target_id": "c", "neighbor_id": "d"}])
    _write_jsonl(processed / "llm_labels_qwen.jsonl", [{"target_id": "a", "neighbor_id": "b"}, {"target_id": "c", "neighbor_id": "d"}])

    table = collect_cost_scalability(
        argparse.Namespace(
            input_dir=str(output_dir),
            output_dir=str(output_dir),
            datasets=["yelp_academic"],
            models=["hero_gnn"],
            data_root=str(data_root),
        )
    )

    row = table.iloc[0]
    assert row["candidate_cards"] == 2
    assert row["annotated_cards"] == 2
    assert row["annotation_coverage"] == 1.0
    assert row["train_time_seconds_per_seed"] == 12.0
    assert (output_dir / "summary" / "table_cost_scalability.csv").exists()


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
