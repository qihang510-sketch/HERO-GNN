import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_real_suite import _is_official_dataset
from src.data.loader import load_processed_data
from src.training.trainer import _is_official_graph, train_single_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_official_dataset(base_dir: Path, dataset: str = "fraud_amazon_official") -> Path:
    data_dir = base_dir / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    node_ids = [f"user_{idx}" for idx in range(12)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    split_names = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_type": ["user"] * len(node_ids),
            "text": [""] * len(node_ids),
            "label": labels,
            "split": split_names,
        }
    )
    edges = pd.DataFrame(
        {
            "src": [node_ids[idx] for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 2, 4]],
            "dst": [node_ids[idx] for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6, 8, 10]],
            "edge_type": ["rel_a", "rel_b", "rel_a", "rel_c", "rel_b", "rel_a", "rel_c", "rel_b", "rel_a", "rel_c", "rel_b", "rel_c", "rel_b", "rel_a"],
        }
    )
    features = np.asarray(
        [
            [idx / 12.0, (idx % 3) / 3.0, float(idx % 2), np.sin(idx)]
            for idx in range(len(node_ids))
        ],
        dtype=np.float32,
    )
    nodes.to_csv(data_dir / "nodes.csv", index=False)
    edges.to_csv(data_dir / "edges.csv", index=False)
    np.savez_compressed(
        data_dir / "features.npz",
        node_ids=np.asarray(node_ids),
        features=features,
        text_features=np.zeros((len(node_ids), 1), dtype=np.float32),
        numeric_features=features,
    )
    (data_dir / "split.json").write_text(
        json.dumps(
            {
                "train": node_ids[:8],
                "val": node_ids[8:10],
                "test": node_ids[10:],
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "preprocess_report.json").write_text(
        json.dumps({"dataset": dataset, "label_source": "official"}),
        encoding="utf-8",
    )
    return data_dir


def test_fraud_amazon_official_is_detected_from_preprocess_report(tmp_path):
    data_dir = _write_tiny_official_dataset(tmp_path)
    graph = load_processed_data(data_dir)

    assert _is_official_graph(graph, "fraud_amazon_official") is True
    assert _is_official_dataset("fraud_amazon_official", data_dir) is True


def test_run_real_suite_invokes_hero_official_and_methods_limit(tmp_path):
    data_dir = _write_tiny_official_dataset(tmp_path / "processed")
    output_root = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_real_suite.py",
            "--dataset",
            "fraud_amazon_official",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--data_dir",
            str(data_dir),
            "--output_root",
            str(output_root),
            "--seeds",
            "0",
            "--methods",
            "hero_official",
            "--epochs",
            "1",
            "--hidden_dim",
            "8",
            "--max_candidates_per_node",
            "4",
            "--homophilic_topk",
            "2",
            "--heterophilic_topk",
            "2",
            "--device",
            "auto",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics_path = output_root / "results" / "fraud_amazon_official" / "hero_official" / "seed_0" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["official_mode"] is True
    assert payload["use_official_chain"] is False
    assert payload["official_topk_homo"] == 2
    assert payload["official_topk_hetero"] == 2
    assert "avg_relation_gate" in payload
    assert not (output_root / "results" / "fraud_amazon_official" / "mlp").exists()


def test_device_cuda_request_falls_back_or_uses_cuda(tmp_path):
    data_dir = _write_tiny_official_dataset(tmp_path / "processed")

    metrics = train_single_experiment(
        dataset="fraud_amazon_official",
        model_name="mlp",
        seed=0,
        data_dir=data_dir,
        output_root=tmp_path / "outputs",
        epochs=1,
        hidden_dim=8,
        device="cuda",
    )

    assert metrics["device"] in {"cpu", "cuda"}
    if not bool(metrics["cuda_available"]):
        assert metrics["device"] == "cpu"


def test_summarize_results_includes_official_diagnostics(tmp_path):
    metrics_dir = tmp_path / "results" / "fraud_amazon_official" / "hero_official" / "seed_0"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": "fraud_amazon_official",
                "model": "hero_official",
                "seed": 0,
                "macro_f1": 0.5,
                "auroc": 0.6,
                "auprc": 0.7,
                "official_mode": True,
                "use_official_chain": False,
                "avg_relation_gate": 0.25,
                "avg_feature_deviation_gate": 0.30,
                "avg_official_hetero_gate": 0.35,
                "official_avg_feature_distance_selected": 0.40,
                "official_avg_homo_similarity_selected": 0.50,
                "official_avg_relation_rarity": 0.20,
                "official_num_relation_types": 3,
                "official_topk_homo": 20,
                "official_topk_hetero": 20,
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "summary"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--result_dir",
            str(tmp_path / "results"),
            "--out_dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    diagnostics = pd.read_csv(out_dir / "diagnostic_table.csv")
    for column in [
        "official_mode",
        "use_official_chain",
        "avg_relation_gate",
        "avg_feature_deviation_gate",
        "avg_official_hetero_gate",
        "official_avg_feature_distance_selected",
        "official_avg_homo_similarity_selected",
        "official_avg_relation_rarity",
        "official_num_relation_types",
        "official_topk_homo",
        "official_topk_hetero",
    ]:
        assert column in diagnostics.columns
    assert float(diagnostics.loc[0, "avg_relation_gate"]) == 0.25
