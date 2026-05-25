import numpy as np

from src.graph.neighbor_retrieval import (
    build_adjacency,
    filter_rule_hetero_edges,
    filter_topk_semantic_edges,
    outgoing_neighbors,
    retrieve_hetero_candidates,
)


def test_outgoing_neighbors():
    edge_index = np.array([[0, 0, 1], [1, 2, 2]])
    assert outgoing_neighbors(edge_index, 0).tolist() == [1, 2]
    assert build_adjacency(edge_index)[0] == [1, 2]


def test_semantic_and_rule_filters_keep_topk_per_source():
    edge_index = np.array([[0, 0, 0, 1], [1, 2, 3, 2]])
    text_features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    numeric_features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [3.0, 3.0],
            [0.2, 0.2],
        ],
        dtype=np.float32,
    )

    sem_edges = filter_topk_semantic_edges(edge_index, text_features, top_k=1)
    rule_edges = filter_rule_hetero_edges(edge_index, text_features, numeric_features, top_k=1)

    assert sem_edges.shape == (2, 2)
    assert rule_edges.shape == (2, 2)
    assert sem_edges[:, 0].tolist() == [0, 1]
    assert rule_edges[:, 0].tolist() == [0, 2]


def test_hetero_candidate_recall_requires_context():
    import pandas as pd

    edge_index = np.array([[0, 0], [1, 2]])
    edges = pd.DataFrame(
        [
            {"src": "r0", "dst": "r1", "edge_type": "review-user-review", "timestamp": 10},
            {"src": "r0", "dst": "r2", "edge_type": "review-item-review", "timestamp": 10_000},
        ]
    )
    nodes = pd.DataFrame(
        [
            {"node_id": "r0", "node_type": "review", "label": 0, "timestamp": 10, "feat_1": 2.0},
            {"node_id": "r1", "node_type": "review", "label": 1, "timestamp": 11, "feat_1": 2.0},
            {"node_id": "r2", "node_type": "review", "label": 0, "timestamp": 10_000, "feat_1": 0.0},
        ]
    )
    text_features = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    numeric_features = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    candidates = retrieve_hetero_candidates(
        edge_index=edge_index,
        edges=edges,
        nodes=nodes,
        node_id_to_idx={"r0": 0, "r1": 1, "r2": 2},
        text_features=text_features,
        numeric_features=numeric_features,
        target_indices=[0],
        top_k=2,
        min_context_score=0.8,
    )
    assert [candidate.neighbor_id for candidate in candidates[0]] == ["r1"]
