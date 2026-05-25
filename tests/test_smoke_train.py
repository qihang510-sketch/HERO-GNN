from src.data.loader import load_processed_data
from src.data.synthetic_builder import generate_synthetic_dataset
from src.training.trainer import train_single_experiment


def test_loader_reads_processed_synthetic_data(tmp_path):
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=tmp_path / "synthetic",
        processed_dir=processed_dir,
        num_reviews=80,
        num_users=20,
        num_items=10,
        num_devices=15,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=3,
        text_dim=16,
    )

    graph = load_processed_data(processed_dir)
    assert graph.features.shape[0] == len(graph.node_id_to_idx)
    assert graph.labels.shape[0] == graph.features.shape[0]
    assert set(graph.split) == {"train", "val", "test"}
    assert graph.edge_index.shape[0] == 2
    assert graph.evidence_gt


def test_minimal_training_writes_metrics(tmp_path):
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=tmp_path / "synthetic",
        processed_dir=processed_dir,
        num_reviews=120,
        num_users=25,
        num_items=12,
        num_devices=18,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=4,
        text_dim=16,
    )

    metrics = train_single_experiment(
        dataset="synthetic",
        model_name="graphsage",
        seed=4,
        data_dir=processed_dir,
        output_root=tmp_path / "outputs",
        top_k=3,
    )
    assert metrics["dataset"] == "synthetic"
    assert metrics["model"] == "graphsage"
    assert (tmp_path / "outputs" / "results" / "synthetic" / "graphsage" / "seed_4" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "checkpoints" / "synthetic" / "graphsage" / "seed_4" / "best.pt").exists()


def test_hero_training_writes_explanations(tmp_path):
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=tmp_path / "synthetic",
        processed_dir=processed_dir,
        num_reviews=140,
        num_users=30,
        num_items=14,
        num_devices=20,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=5,
        text_dim=16,
    )
    metrics = train_single_experiment(
        dataset="synthetic",
        model_name="hero_gnn",
        seed=5,
        data_dir=processed_dir,
        output_root=tmp_path / "outputs",
        top_k=3,
    )
    assert "evidence_recall_proxy" in metrics
    assert (processed_dir / "llm_labels.jsonl").exists()
    assert (tmp_path / "outputs" / "explanations" / "synthetic" / "hero_gnn" / "seed_5" / "examples.jsonl").exists()
