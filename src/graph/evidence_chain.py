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
    return sorted(chains, key=lambda item: item["chain_score"], reverse=True)[:top_k]


def _one_hop_chain(label: dict[str, Any]) -> dict[str, Any]:
    candidate = label.get("candidate", {})
    score = float(label.get("risk_score", label.get("confidence", 0.0)))
    return {
        "target_id": label["target_id"],
        "chain_nodes": [label["target_id"], label["neighbor_id"]],
        "chain_edges": [label["metapath"]],
        "mechanism": label["mechanism"],
        "risk_relevance": int(label.get("risk_relevance", 0)),
        "chain_score": score,
        "confidence": float(label.get("confidence", 0.0)),
        "rationale": label["rationale"],
        "neighbor_idx": candidate.get("neighbor_idx"),
    }


def _two_hop_chain(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_score = float(first.get("risk_score", first.get("confidence", 0.0)))
    second_score = float(second.get("risk_score", second.get("confidence", 0.0)))
    return {
        "target_id": first["target_id"],
        "chain_nodes": [first["target_id"], first["neighbor_id"], second["neighbor_id"]],
        "chain_edges": [first["metapath"], second["metapath"]],
        "mechanism": first["mechanism"],
        "risk_relevance": int(first.get("risk_relevance", 0)),
        "chain_score": (first_score + second_score) / 2.0,
        "confidence": float(first.get("confidence", 0.0)),
        "rationale": f"{first['rationale']}; then {second['rationale']}",
        "neighbor_idx": first.get("candidate", {}).get("neighbor_idx"),
    }
