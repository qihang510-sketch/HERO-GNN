import json

import pandas as pd

from src.data.synthetic_builder import generate_synthetic_dataset


def test_synthetic_dataset_generation(tmp_path):
    raw_dir = tmp_path / "synthetic"
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=raw_dir,
        processed_dir=processed_dir,
        num_reviews=120,
        num_users=30,
        num_items=20,
        num_devices=25,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=7,
        text_dim=32,
    )

    assert (raw_dir / "nodes.csv").exists()
    assert (raw_dir / "edges.csv").exists()
    assert (raw_dir / "evidence_gt.json").exists()
    assert (processed_dir / "nodes.csv").exists()
    assert (processed_dir / "edges.csv").exists()
    assert (processed_dir / "features.npz").exists()
    assert (processed_dir / "split.json").exists()
    assert (processed_dir / "evidence_gt.json").exists()

    nodes = pd.read_csv(raw_dir / "nodes.csv")
    review_kinds = set(nodes.loc[nodes["node_type"] == "review", "review_kind"])
    assert {"normal", "homophilic_fraud", "heterophilic_fraud"}.issubset(review_kinds)

    evidence = json.loads((raw_dir / "evidence_gt.json").read_text(encoding="utf-8"))
    assert evidence
    assert all("evidence_neighbors" in value for value in evidence.values())
