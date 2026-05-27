from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_evidence_chain(node_id: int, scored_neighbors: dict[int, float], max_length: int = 3) -> list[dict[str, float | int]]:
    ranked = sorted(scored_neighbors.items(), key=lambda item: item[1], reverse=True)
    return [
        {"source": int(node_id), "target": int(neighbor), "score": float(score)}
        for neighbor, score in ranked[:max_length]
    ]


def build_evidence_chains(
    target_id: str,
    labels: list[dict[str, Any]],
    all_labels: list[dict[str, Any]],
    top_k: int = 3,
    include_two_hop: bool = True,
) -> list[dict[str, Any]]:
    direct = [label for label in labels if label.get("risk_relevance", 0) == 1]
    chains = [_one_hop_chain(label) for label in direct]
    if include_two_hop:
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for label in all_labels:
            if label.get("risk_relevance", 0) == 1:
                by_target[label["target_id"]].append(label)
        for first in direct:
            for second in by_target.get(first["neighbor_id"], [])[:2]:
                if second["neighbor_id"] == target_id:
                    continue
                chains.append(_two_hop_chain(first, second))
    for chain in chains:
        _with_chain_quality(chain)
    return sorted(chains, key=lambda item: (item["chain_quality"], item["chain_score"]), reverse=True)[:top_k]


def _one_hop_chain(label: dict[str, Any]) -> dict[str, Any]:
    candidate = label.get("candidate", {})
    score = float(label.get("risk_score", label.get("confidence", 0.0)))
    signals = _chain_signals(label)
    return _with_chain_quality({
        "target_id": label["target_id"],
        "chain_nodes": [label["target_id"], label["neighbor_id"]],
        "chain_edges": [label["metapath"]],
        "mechanism": label["mechanism"],
        "risk_relevance": int(label.get("risk_relevance", 0)),
        "chain_score": score,
        "confidence": float(label.get("confidence", 0.0)),
        "rationale": label["rationale"],
        "neighbor_idx": candidate.get("neighbor_idx"),
        **signals,
    })


def _two_hop_chain(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_score = float(first.get("risk_score", first.get("confidence", 0.0)))
    second_score = float(second.get("risk_score", second.get("confidence", 0.0)))
    signals = _average_signals(first, second)
    return _with_chain_quality({
        "target_id": first["target_id"],
        "chain_nodes": [first["target_id"], first["neighbor_id"], second["neighbor_id"]],
        "chain_edges": [first["metapath"], second["metapath"]],
        "mechanism": first["mechanism"],
        "risk_relevance": int(first.get("risk_relevance", 0)),
        "chain_score": (first_score + second_score) / 2.0,
        "confidence": float(first.get("confidence", 0.0)),
        "rationale": f"{first['rationale']}; then {second['rationale']}",
        "neighbor_idx": first.get("candidate", {}).get("neighbor_idx"),
        **signals,
    })


def _with_chain_quality(chain: dict[str, Any]) -> dict[str, Any]:
    confidence = _bounded(chain.get("confidence", 0.0))
    risk_relevance = int(chain.get("risk_relevance", 0))
    chain_score = _bounded(chain.get("chain_score", 0.0))
    structural_score = _bounded(chain.get("structural_score", 0.0))
    numeric_deviation = _bounded(chain.get("numeric_deviation", 0.0))
    chain["confidence"] = confidence
    chain["chain_score"] = chain_score
    chain["structural_score"] = structural_score
    chain["numeric_deviation"] = numeric_deviation
    chain["time_deviation"] = _bounded(chain.get("time_deviation", 0.0))
    chain["semantic_dissimilarity"] = _bounded(chain.get("semantic_dissimilarity", 0.0))
    chain["chain_quality"] = _bounded(
        0.30 * confidence
        + 0.25 * float(risk_relevance)
        + 0.20 * chain_score
        + 0.15 * structural_score
        + 0.10 * numeric_deviation
    )
    return chain


def _chain_signals(label: dict[str, Any]) -> dict[str, float]:
    candidate = label.get("candidate", {}) or {}
    risk_card = label.get("risk_card", {}) or {}
    semantic_similarity = _first_numeric(candidate, risk_card, "semantic_similarity", default=1.0)
    return {
        "structural_score": _first_numeric(candidate, risk_card, "structural_score", default=0.0),
        "numeric_deviation": _first_numeric(candidate, risk_card, "numeric_deviation", default=0.0),
        "time_deviation": _first_numeric(candidate, risk_card, "time_deviation", default=0.0),
        "semantic_dissimilarity": _first_numeric(candidate, risk_card, "semantic_distance", default=1.0 - semantic_similarity),
    }


def _average_signals(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    left = _chain_signals(first)
    right = _chain_signals(second)
    return {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2.0 for key in left}


def _first_numeric(primary: dict[str, Any], secondary: dict[str, Any], key: str, default: float = 0.0) -> float:
    for container in (primary, secondary):
        if key in container and container[key] is not None:
            return _bounded(container[key])
    return _bounded(default)


def _bounded(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return float(min(max(numeric, 0.0), 1.0))
