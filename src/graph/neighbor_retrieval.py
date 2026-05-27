from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import schema


def outgoing_neighbors(edge_index: np.ndarray, node_id: int) -> np.ndarray:
    mask = edge_index[0] == node_id
    return edge_index[1, mask]


def build_adjacency(edge_index: np.ndarray) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {}
    for src, dst in edge_index.T:
        adjacency.setdefault(int(src), []).append(int(dst))
    return adjacency


def filter_topk_semantic_edges(edge_index: np.ndarray, text_features: np.ndarray, top_k: int = 10) -> np.ndarray:
    return _filter_edges_by_score(edge_index, _cosine_similarity_scores(edge_index, text_features), top_k)


def filter_rule_hetero_edges(
    edge_index: np.ndarray,
    text_features: np.ndarray,
    numeric_features: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    semantic_similarity = _cosine_similarity_scores(edge_index, text_features)
    numeric_delta = np.linalg.norm(
        numeric_features[edge_index[0]] - numeric_features[edge_index[1]],
        axis=1,
    )
    if numeric_delta.size and numeric_delta.max() > 0:
        numeric_delta = numeric_delta / numeric_delta.max()
    scores = (1.0 - semantic_similarity) + numeric_delta
    return _filter_edges_by_score(edge_index, scores, top_k)


@dataclass
class HeteroCandidate:
    target_idx: int
    neighbor_idx: int
    target_id: str
    neighbor_id: str
    metapath: str
    semantic_similarity: float
    semantic_distance: float
    structural_score: float
    numeric_deviation: float
    time_deviation: float
    candidate_score: float
    rating_diff: float
    same_user: bool
    same_device: bool
    same_item: bool
    same_item_or_business: bool
    time_gap_is_short: bool
    burst_score: float
    neighbor_risk_prior: float


def retrieve_hetero_candidates(
    edge_index: np.ndarray,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    node_id_to_idx: dict[str, int],
    text_features: np.ndarray,
    numeric_features: np.ndarray,
    target_indices: np.ndarray | list[int] | None = None,
    top_k: int = 10,
    max_candidates_per_node: int | None = None,
    w_struct: float = 0.35,
    w_numeric: float = 0.25,
    w_time: float = 0.20,
    w_semantic: float = 0.20,
    min_semantic_distance: float = 0.20,
    min_context_score: float = 0.30,
) -> dict[int, list[HeteroCandidate]]:
    """Recall structurally grounded heterophilous candidates.

    Semantic distance alone is not enough: a candidate is kept only when it also
    has structural proximity, numeric deviation, or short-time behavioral signal.
    """
    if target_indices is None:
        target_indices = np.unique(edge_index[0]) if edge_index.size else np.array([], dtype=np.int64)
    top_k = int(max_candidates_per_node if max_candidates_per_node is not None else top_k)
    target_set = {int(index) for index in target_indices}
    idx_to_node_id = {idx: node_id for node_id, idx in node_id_to_idx.items()}
    timestamps = _timestamps(nodes, node_id_to_idx, len(node_id_to_idx))
    max_time_gap = max(float(np.ptp(timestamps)), 1.0)
    max_numeric = _max_pair_l1(numeric_features, edge_index)
    edge_type_by_pair = _edge_type_lookup(edges, node_id_to_idx)
    burst_by_idx = _normalized_out_degree(edge_index, len(node_id_to_idx))

    by_target: dict[int, list[HeteroCandidate]] = {target: [] for target in target_set}
    for src, dst in tqdm(edge_index.T, desc="neighbor_retrieval", unit="edge"):
        src = int(src)
        dst = int(dst)
        if src not in target_set:
            continue
        target_id = idx_to_node_id.get(src, "")
        neighbor_id = idx_to_node_id.get(dst, "")
        if not target_id or not neighbor_id:
            continue
        metapath = edge_type_by_pair.get((src, dst), "review-item-review")
        semantic_similarity = _pair_cosine(text_features[src], text_features[dst])
        semantic_distance = 1.0 - semantic_similarity
        structural_score = _metapath_proximity(metapath)
        numeric_deviation = _normalized_l1(numeric_features[src], numeric_features[dst], max_numeric)
        time_deviation = _time_gap_score(timestamps[src], timestamps[dst], max_time_gap)
        context_score = max(structural_score, numeric_deviation, time_deviation)
        if semantic_distance < min_semantic_distance or context_score < min_context_score:
            continue
        candidate_score = (
            w_struct * structural_score
            + w_numeric * numeric_deviation
            + w_time * time_deviation
            + w_semantic * semantic_distance
        )
        same_item_or_business = metapath in {
            "review-item-review",
            "review-business-review",
            "review-product-review",
            "review-rating-review",
        }
        candidate = HeteroCandidate(
            target_idx=src,
            neighbor_idx=dst,
            target_id=target_id,
            neighbor_id=neighbor_id,
            metapath=metapath,
            semantic_similarity=float(semantic_similarity),
            semantic_distance=float(semantic_distance),
            structural_score=float(structural_score),
            numeric_deviation=float(numeric_deviation),
            time_deviation=float(time_deviation),
            candidate_score=float(candidate_score),
            rating_diff=float(abs(numeric_features[src, 0] - numeric_features[dst, 0])) if numeric_features.shape[1] else 0.0,
            same_user=metapath == "review-user-review",
            same_device=metapath == "review-device-review",
            same_item=metapath == "review-item-review",
            same_item_or_business=same_item_or_business,
            time_gap_is_short=time_deviation > 0.7,
            burst_score=float(max(burst_by_idx[src], burst_by_idx[dst])),
            neighbor_risk_prior=0.0,
        )
        by_target[src].append(candidate)

    for target, candidates in by_target.items():
        by_target[target] = sorted(candidates, key=lambda item: item.candidate_score, reverse=True)[:top_k]
    return by_target


def candidates_to_edge_index(candidates_by_target: dict[int, list[HeteroCandidate]]) -> np.ndarray:
    pairs = [
        (candidate.target_idx, candidate.neighbor_idx)
        for candidates in candidates_by_target.values()
        for candidate in candidates
    ]
    return np.array(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), dtype=np.int64)


def _filter_edges_by_score(edge_index: np.ndarray, scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    selected: list[int] = []
    for src in np.unique(edge_index[0]):
        positions = np.flatnonzero(edge_index[0] == src)
        ranked = positions[np.argsort(scores[positions])[::-1]]
        selected.extend(ranked[:top_k].tolist())
    if not selected:
        return np.zeros((2, 0), dtype=np.int64)
    return edge_index[:, np.array(selected, dtype=np.int64)]


def _cosine_similarity_scores(edge_index: np.ndarray, features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if edge_index.size == 0:
        return np.array([], dtype=np.float32)
    src = features[edge_index[0]]
    dst = features[edge_index[1]]
    denom = np.linalg.norm(src, axis=1) * np.linalg.norm(dst, axis=1) + eps
    return np.sum(src * dst, axis=1) / denom


def _pair_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return float(np.dot(a, b) / denom)


def _metapath_proximity(metapath: str) -> float:
    return {
        "review-user-review": 0.95,
        "review-device-review": 0.90,
        "review-time-review": 0.75,
        "review-item-review": 0.65,
        "review-business-review": 0.65,
        "review-product-review": 0.65,
        "review-rating-review": 0.60,
        "review-month-review": 0.55,
        "review-week-review": 0.55,
    }.get(metapath, 0.50)


def _normalized_l1(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(min(np.sum(np.abs(a - b)) / max(scale, 1e-8), 1.0))


def _time_gap_score(left: float, right: float, max_gap: float) -> float:
    gap = abs(float(left) - float(right))
    return float(1.0 - min(gap / max(max_gap, 1.0), 1.0))


def _timestamps(nodes: pd.DataFrame, node_id_to_idx: dict[str, int], size: int) -> np.ndarray:
    values = np.zeros(size, dtype=np.float32)
    for row in nodes[[schema.NODE_ID, schema.TIMESTAMP]].itertuples(index=False):
        if row.node_id in node_id_to_idx:
            values[node_id_to_idx[row.node_id]] = float(row.timestamp)
    return values


def _max_pair_l1(numeric_features: np.ndarray, edge_index: np.ndarray) -> float:
    if edge_index.size == 0:
        return 1.0
    deltas = np.sum(np.abs(numeric_features[edge_index[0]] - numeric_features[edge_index[1]]), axis=1)
    return float(max(np.max(deltas), 1.0))


def _normalized_out_degree(edge_index: np.ndarray, size: int) -> np.ndarray:
    if edge_index.size == 0 or size == 0:
        return np.zeros(size, dtype=np.float32)
    degree = np.bincount(edge_index[0], minlength=size).astype(np.float32)
    max_degree = float(np.max(degree))
    if max_degree <= 0:
        return np.zeros(size, dtype=np.float32)
    return degree / max_degree


def _edge_type_lookup(edges: pd.DataFrame, node_id_to_idx: dict[str, int]) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    for src, dst, edge_type in edges[[schema.SRC, schema.DST, schema.EDGE_TYPE]].itertuples(index=False, name=None):
        if src in node_id_to_idx and dst in node_id_to_idx:
            lookup[(node_id_to_idx[src], node_id_to_idx[dst])] = str(edge_type)
    return lookup
