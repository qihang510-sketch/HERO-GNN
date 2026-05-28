from __future__ import annotations

import numpy as np

from src.data.schema import EVIDENCE_MECHANISMS
from src.graph.neighbor_retrieval import HeteroCandidate

MECHANISM_TO_ID = {name: index for index, name in enumerate(EVIDENCE_MECHANISMS)}


def cosine_dissimilarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(1.0 - np.dot(a, b) / denom)


def score_heterophilous_neighbors(features: np.ndarray, node_id: int, neighbors: np.ndarray) -> dict[int, float]:
    return {int(n): cosine_dissimilarity(features[node_id], features[int(n)]) for n in neighbors}


def pair_feature_vector(
    candidate: HeteroCandidate,
    target_text_emb: np.ndarray,
    neighbor_text_emb: np.ndarray,
    target_num_feat: np.ndarray,
    neighbor_num_feat: np.ndarray,
) -> np.ndarray:
    metapath_embedding = _metapath_embedding(candidate.metapath)
    scalars = np.array(
        [
            candidate.structural_score,
            candidate.semantic_similarity,
            candidate.numeric_deviation,
            candidate.time_deviation,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            target_text_emb,
            neighbor_text_emb,
            np.abs(target_text_emb - neighbor_text_emb),
            target_num_feat,
            neighbor_num_feat,
            np.abs(target_num_feat - neighbor_num_feat),
            metapath_embedding,
            scalars,
        ]
    ).astype(np.float32)


def risk_heterophily_score(
    candidate: HeteroCandidate,
    mechanism: str | None = None,
    risk_relevance_label: int | None = None,
    use_mechanism: bool = True,
) -> tuple[float, np.ndarray]:
    base_score = candidate.candidate_score
    if use_mechanism and mechanism and mechanism != "irrelevant_heterophily":
        base_score += 0.25
    if risk_relevance_label is not None:
        base_score = 0.6 * base_score + 0.4 * float(risk_relevance_label)
    score = float(np.clip(base_score, 0.0, 1.0))
    logits = np.full(len(EVIDENCE_MECHANISMS), -2.0, dtype=np.float32)
    mechanism_id = MECHANISM_TO_ID.get(mechanism or "irrelevant_heterophily", 0)
    logits[mechanism_id] = 2.0 + score
    return score, logits


def mechanism_id(mechanism: str) -> int:
    return MECHANISM_TO_ID.get(mechanism, MECHANISM_TO_ID["irrelevant_heterophily"])


def _metapath_embedding(metapath: str) -> np.ndarray:
    mapping = {
        "review-user-review": [1.0, 0.0, 0.0, 0.0],
        "review-item-review": [0.0, 1.0, 0.0, 0.0],
        "review-device-review": [0.0, 0.0, 1.0, 0.0],
        "review-time-review": [0.0, 0.0, 0.0, 1.0],
        "net_rur": [1.0, 0.0, 0.0, 0.0],
        "net_rsr": [0.0, 1.0, 0.0, 0.0],
        "net_rtr": [0.0, 0.0, 0.0, 1.0],
        "net_upu": [0.0, 1.0, 0.0, 0.0],
        "net_usu": [0.0, 0.0, 0.0, 1.0],
        "net_uvu": [0.0, 0.0, 1.0, 0.0],
    }
    return np.array(mapping.get(metapath, [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
