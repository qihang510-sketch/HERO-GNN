import json

import numpy as np
import pandas as pd

from src.data.dgl_fraud_preprocess import preprocess_dgl_fraud_graph
from src.data.loader import load_processed_data
from src.training.trainer import train_single_experiment


class _NodeView:
    def __init__(self, data):
        self.data = data


class _Nodes:
    def __init__(self, data_by_type):
        self._data_by_type = data_by_type

    def __getitem__(self, node_type):
        return _NodeView(self._data_by_type[node_type])


class _FakeHeteroGraph:
    def __init__(self, node_type, node_data, edges_by_type):
        self.ntypes = [node_type]
        self.nodes = _Nodes({node_type: node_data})
        self.ndata = node_data
        self.canonical_etypes = [(node_type, edge_type, node_type) for edge_type in edges_by_type]
        self._edges_by_type = edges_by_type

    def edges(self, etype=None):
        edge_type = etype[1] if isinstance(etype, tuple) else etype
        return self._edges_by_type[edge_type]


class _FakeHomogeneousGraph:
    def __init__(self, node_type, node_data, edges):
        self.ntypes = [node_type]
        self.nodes = _Nodes({node_type: node_data})
        self.ndata = node_data
        self.canonical_etypes = []
        self._edges = edges

    def edges(self):
        return self._edges


def test_dgl_fraud_preprocess_mock_heterograph_with_official_masks(tmp_path):
    out_dir = tmp_path / "fraud_yelp_official"
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    graph = _FakeHeteroGraph(
        "review",
        {
            "feature": features,
            "label": np.array([0, 1, 0, 1, 0, 1]),
            "train_mask": np.array([True, True, False, False, False, False]),
            "val_mask": np.array([False, False, True, False, False, False]),
            "test_mask": np.array([False, False, False, True, True, False]),
        },
        {
            "net_rur": (np.array([0, 1, 3]), np.array([1, 0, 4])),
            "net_rsr": (np.array([2, 4]), np.array([5, 3])),
        },
    )

    preprocess_dgl_fraud_graph(graph, "fraud_yelp_official", out_dir, seed=0)

    for name in ["nodes.csv", "edges.csv", "features.npz", "split.json", "preprocess_report.json"]:
        assert (out_dir / name).exists()

    nodes = pd.read_csv(out_dir / "nodes.csv")
    edges = pd.read_csv(out_dir / "edges.csv")
    payload = np.load(out_dir / "features.npz", allow_pickle=True)
    split = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    loaded = load_processed_data(out_dir)

    assert set(nodes["node_type"]) == {"review"}
    assert nodes["text"].fillna("").eq("").all()
    assert set(edges["edge_type"]) == {"net_rur", "net_rsr"}
    np.testing.assert_array_equal(payload["features"], features)
    np.testing.assert_array_equal(payload["numeric_features"], features)
    assert np.all(payload["text_features"] == 0.0)
    assert split == {
        "train": ["review_0", "review_1"],
        "val": ["review_2"],
        "test": ["review_3", "review_4"],
    }
    assert report["dataset"] == "fraud_yelp_official"
    assert report["label_source"] == "official"
    assert report["split_source"] == "official_mask"
    assert report["feature_dim"] == 3
    assert report["num_positive"] == 2
    assert report["num_negative"] == 3
    assert loaded.features.shape == (6, 3)


def test_dgl_fraud_preprocess_falls_back_to_project_split(tmp_path):
    out_dir = tmp_path / "fraud_amazon_official"
    graph = _FakeHomogeneousGraph(
        "user",
        {
            "feature": np.ones((6, 2), dtype=np.float32),
            "label": np.array([0, 1, 0, 1, 0, 1]),
        },
        (np.array([0, 1, 2]), np.array([1, 2, 3])),
    )

    preprocess_dgl_fraud_graph(graph, "fraud_amazon_official", out_dir, seed=7)

    nodes = pd.read_csv(out_dir / "nodes.csv")
    edges = pd.read_csv(out_dir / "edges.csv")
    split = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "preprocess_report.json").read_text(encoding="utf-8"))

    assert set(nodes["node_type"]) == {"user"}
    assert set(nodes["split"]) == {"train", "val", "test"}
    assert {name: len(values) for name, values in split.items()} == {"train": 3, "val": 1, "test": 2}
    assert edges["edge_type"].tolist() == ["rel_0", "rel_0", "rel_0"]
    assert report["split_source"] == "generated_60_20_20"
    assert report["num_unlabeled"] == 0


def test_official_no_text_data_trains_mlp_graphsage_and_hero(tmp_path):
    processed_dir = tmp_path / "processed" / "fraud_yelp_official"
    features = np.stack(
        [
            np.linspace(0.0, 1.0, 12),
            np.linspace(1.0, 0.0, 12),
            np.arange(12) % 3,
            np.arange(12) % 2,
        ],
        axis=1,
    ).astype(np.float32)
    graph = _FakeHeteroGraph(
        "review",
        {
            "feature": features,
            "label": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            "train_mask": np.array([True, True, True, True, True, True, True, True, False, False, False, False]),
            "val_mask": np.array([False, False, False, False, False, False, False, False, True, True, False, False]),
            "test_mask": np.array([False, False, False, False, False, False, False, False, False, False, True, True]),
        },
        {
            "net_rur": (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
            "net_rsr": (np.array([10, 9, 8, 7, 6, 5]), np.array([11, 10, 9, 8, 7, 6])),
        },
    )
    preprocess_dgl_fraud_graph(graph, "fraud_yelp_official", processed_dir, seed=0)

    for model_name in ["mlp", "graphsage", "sec_gfd_lite", "dga_gnn_lite", "flag_lite", "hero_official"]:
        metrics = train_single_experiment(
            dataset="fraud_yelp_official",
            model_name=model_name,
            seed=0,
            data_dir=processed_dir,
            output_root=tmp_path / "outputs",
            epochs=1,
            hidden_dim=8,
            top_k=2,
            homophilic_topk=2,
            heterophilic_topk=2,
            max_candidates_per_node=2,
            topk_chains=1,
            max_chain_length=1,
            min_chain_quality=0.0,
        )
        assert metrics["dataset"] == "fraud_yelp_official"
        assert metrics["model"] == model_name
        if model_name == "hero_official":
            assert metrics["official_mode"] is True
            assert metrics["use_official_chain"] is False
