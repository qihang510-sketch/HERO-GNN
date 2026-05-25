from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import schema
from src.data.schema import GraphData
from src.features.numeric_encoder import NumericEncoder, numeric_feature_columns
from src.features.text_encoder import TfidfTextEncoder

NORMAL_TEXTS = (
    "The product arrived on time and worked as expected.",
    "Helpful service, fair price, and a normal buying experience.",
    "The quality matched the description and I would consider buying again.",
    "A straightforward purchase with no major issues.",
    "The item was useful and the review reflects my actual experience.",
)
HOMOPHILIC_FRAUD_TEXTS = (
    "Amazing deal five stars best product ever highly recommended.",
    "Perfect perfect perfect quality and fast shipping buy now.",
    "Excellent seller top quality best value five stars.",
    "Outstanding product everyone should order this immediately.",
    "Best purchase ever flawless quality highly recommended.",
)
HETEROPHILIC_COVER_TEXTS = (
    "The item was acceptable and matched the basic description.",
    "Delivery was fine and the product worked in ordinary use.",
    "This was a regular purchase with a reasonable overall experience.",
    "The service was normal and the item did what I needed.",
    "Nothing unusual happened and the order was completed.",
)


def build_synthetic_graph(
    num_nodes: int = 200,
    num_edges: int = 800,
    feature_dim: int = 16,
    seed: int = 42,
) -> GraphData:
    """Legacy in-memory graph builder kept for lightweight early tests."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_nodes, feature_dim)).astype(np.float32)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0.75).astype(np.int64)

    src = rng.integers(0, num_nodes, size=num_edges, endpoint=False)
    dst = rng.integers(0, num_nodes, size=num_edges, endpoint=False)
    keep = src != dst
    edge_index = np.stack([src[keep], dst[keep]], axis=0).astype(np.int64)

    evidence: dict[int, list[int]] = {}
    anomalous_nodes = np.flatnonzero(labels == 1)
    for node in anomalous_nodes:
        candidate_neighbors = edge_index[1, edge_index[0] == node]
        hetero = [int(n) for n in candidate_neighbors if labels[n] != labels[node]]
        if hetero:
            evidence[int(node)] = hetero[:3]

    return GraphData(
        features=features,
        edge_index=edge_index,
        labels=labels,
        evidence=evidence,
        metadata={
            "dataset": "synthetic",
            "seed": seed,
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[1]),
            "feature_dim": feature_dim,
        },
    )


def generate_synthetic_dataset(
    out_dir: str | Path = "data/synthetic",
    processed_dir: str | Path = "data/processed/synthetic",
    num_reviews: int = 3000,
    num_users: int = 600,
    num_items: int = 200,
    num_devices: int = 300,
    fraud_ratio: float = 0.15,
    hetero_fraud_ratio: float = 0.50,
    seed: int = 42,
    text_dim: int = 128,
) -> dict[str, Path]:
    if num_reviews < 10:
        raise ValueError("num_reviews must be at least 10")
    if not 0 < fraud_ratio < 1:
        raise ValueError("fraud_ratio must be between 0 and 1")
    if not 0 <= hetero_fraud_ratio <= 1:
        raise ValueError("hetero_fraud_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    processed_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    review_ids = _make_ids("r", num_reviews)
    user_ids = _make_ids("u", num_users)
    item_ids = _make_ids("i", num_items)
    device_ids = _make_ids("d", num_devices)

    num_fraud = max(2, int(round(num_reviews * fraud_ratio)))
    num_hetero = int(round(num_fraud * hetero_fraud_ratio))
    num_homo = num_fraud - num_hetero
    num_hetero = max(1, num_hetero)
    num_homo = max(1, num_homo)
    if num_homo + num_hetero > num_reviews:
        num_homo = max(1, num_reviews - num_hetero)

    shuffled_reviews = np.array(review_ids, dtype=object)
    rng.shuffle(shuffled_reviews)
    hetero_ids = set(shuffled_reviews[:num_hetero].tolist())
    homo_ids = set(shuffled_reviews[num_hetero : num_hetero + num_homo].tolist())
    normal_ids = set(review_ids) - hetero_ids - homo_ids

    base_timestamp = 1_700_000_000
    rows: list[dict] = []
    review_records: dict[str, dict] = {}

    fraud_user_pool = user_ids[: max(1, min(num_users, max(8, num_users // 12)))]
    fraud_device_pool = device_ids[: max(1, min(num_devices, max(8, num_devices // 10)))]
    fraud_item_pool = item_ids[: max(1, min(num_items, max(10, num_items // 8)))]

    split_by_id = _stratified_split_lookup(normal_ids, homo_ids, hetero_ids, seed=seed)
    for index, review_id in enumerate(review_ids):
        if review_id in hetero_ids:
            review_kind = "heterophilic_fraud"
            label = 1
            text = _choice(rng, HETEROPHILIC_COVER_TEXTS)
            user_id = _choice(rng, fraud_user_pool)
            item_id = _choice(rng, item_ids)
            device_id = _choice(rng, fraud_device_pool)
            timestamp = base_timestamp + int(rng.integers(0, 7 * 24 * 3600))
            rating = float(_choice(rng, [1, 2, 5]))
            burst_score = float(rng.normal(2.6, 0.35))
            account_age = float(rng.normal(12, 5))
            device_risk = float(rng.normal(2.2, 0.3))
            semantic_risk = float(rng.normal(0.15, 0.05))
        elif review_id in homo_ids:
            review_kind = "homophilic_fraud"
            label = 1
            text = _choice(rng, HOMOPHILIC_FRAUD_TEXTS)
            user_id = _choice(rng, fraud_user_pool)
            item_id = _choice(rng, fraud_item_pool)
            device_id = _choice(rng, fraud_device_pool)
            timestamp = base_timestamp + int(rng.integers(0, 14 * 24 * 3600))
            rating = float(_choice(rng, [5, 5, 1]))
            burst_score = float(rng.normal(1.8, 0.4))
            account_age = float(rng.normal(20, 8))
            device_risk = float(rng.normal(1.6, 0.4))
            semantic_risk = float(rng.normal(2.0, 0.25))
        else:
            review_kind = "normal"
            label = 0
            text = _choice(rng, NORMAL_TEXTS)
            user_id = _choice(rng, user_ids)
            item_id = _choice(rng, item_ids)
            device_id = _choice(rng, device_ids)
            timestamp = base_timestamp + int(rng.integers(0, 120 * 24 * 3600))
            rating = float(_choice(rng, [3, 4, 4, 5]))
            burst_score = float(rng.normal(0.0, 0.35))
            account_age = float(rng.normal(180, 50))
            device_risk = float(rng.normal(0.0, 0.25))
            semantic_risk = float(rng.normal(0.05, 0.04))

        row = {
            schema.NODE_ID: review_id,
            schema.NODE_TYPE: "review",
            schema.TEXT: text,
            schema.LABEL: label,
            schema.SPLIT: split_by_id[review_id],
            schema.TIMESTAMP: timestamp,
            schema.REVIEW_KIND: review_kind,
            "feat_0": rating,
            "feat_1": burst_score,
            "feat_2": account_age,
            "feat_3": device_risk,
            "feat_4": semantic_risk,
        }
        rows.append(row)
        review_records[review_id] = {
            **row,
            "user_id": user_id,
            "item_id": item_id,
            "device_id": device_id,
            "order": index,
        }

    rows.extend(_entity_rows(user_ids, "user", base_timestamp, rng, feature_count=5))
    rows.extend(_entity_rows(item_ids, "item", base_timestamp, rng, feature_count=5))
    rows.extend(_entity_rows(device_ids, "device", base_timestamp, rng, feature_count=5))

    nodes = pd.DataFrame(rows)
    feature_cols = numeric_feature_columns(nodes.columns.tolist())
    nodes = nodes[list(schema.BASE_NODE_COLUMNS) + feature_cols]

    edges = _build_edges(review_records, rng)
    evidence = _build_evidence(review_records, edges)

    raw_paths = _write_raw(out_dir, nodes, edges, evidence)
    processed_paths = _write_processed(processed_dir, nodes, edges, evidence, text_dim=text_dim)
    return {**raw_paths, **processed_paths}


def write_synthetic_graph(output_dir: str | Path, seed: int = 42, **kwargs: int) -> Path:
    """Compatibility wrapper.

    For the current project flow, prefer generate_synthetic_dataset().
    """
    graph = build_synthetic_graph(seed=seed, **kwargs)
    output_path = Path(output_dir) / "graph.npz"
    graph.save_npz(output_path)
    return output_path


def _make_ids(prefix: str, count: int) -> list[str]:
    width = max(3, len(str(count - 1)))
    return [f"{prefix}_{index:0{width}d}" for index in range(count)]


def _stratified_split_lookup(
    normal_ids: set[str],
    homo_ids: set[str],
    hetero_ids: set[str],
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed + 17)
    lookup: dict[str, str] = {}
    for group in (normal_ids, homo_ids, hetero_ids):
        ids = np.array(sorted(group), dtype=object)
        rng.shuffle(ids)
        train_end = int(len(ids) * 0.6)
        val_end = train_end + int(len(ids) * 0.2)
        for node_id in ids[:train_end]:
            lookup[str(node_id)] = "train"
        for node_id in ids[train_end:val_end]:
            lookup[str(node_id)] = "val"
        for node_id in ids[val_end:]:
            lookup[str(node_id)] = "test"
    return lookup


def _choice(rng: np.random.Generator, values):
    return values[int(rng.integers(0, len(values)))]


def _entity_rows(
    ids: list[str],
    node_type: str,
    base_timestamp: int,
    rng: np.random.Generator,
    feature_count: int,
) -> list[dict]:
    rows = []
    for node_id in ids:
        row = {
            schema.NODE_ID: node_id,
            schema.NODE_TYPE: node_type,
            schema.TEXT: "",
            schema.LABEL: -1,
            schema.SPLIT: "",
            schema.TIMESTAMP: base_timestamp,
            schema.REVIEW_KIND: "",
        }
        for feature_index in range(feature_count):
            row[f"feat_{feature_index}"] = float(rng.normal(0.0, 0.05))
        rows.append(row)
    return rows


def _build_edges(review_records: dict[str, dict], rng: np.random.Generator) -> pd.DataFrame:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for review_id, record in review_records.items():
        groups[("review-user-review", record["user_id"])].append(review_id)
        groups[("review-item-review", record["item_id"])].append(review_id)
        groups[("review-device-review", record["device_id"])].append(review_id)
        time_bucket = int(record[schema.TIMESTAMP] // 3600)
        groups[("review-time-review", str(time_bucket))].append(review_id)

    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for (edge_type, _group_id), ids in groups.items():
        if len(ids) < 2:
            continue
        pairs = list(combinations(sorted(ids), 2))
        if len(pairs) > 120:
            selected = rng.choice(len(pairs), size=120, replace=False)
            pairs = [pairs[int(index)] for index in selected]
        for left, right in pairs:
            timestamp = min(review_records[left][schema.TIMESTAMP], review_records[right][schema.TIMESTAMP])
            for src, dst in ((left, right), (right, left)):
                key = (src, dst, edge_type)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        schema.SRC: src,
                        schema.DST: dst,
                        schema.EDGE_TYPE: edge_type,
                        schema.TIMESTAMP: timestamp,
                    }
                )

    return pd.DataFrame(rows, columns=list(schema.BASE_EDGE_COLUMNS))


def _build_evidence(review_records: dict[str, dict], edges: pd.DataFrame) -> dict[str, dict]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        adjacency[row.src].append((row.dst, row.edge_type))

    evidence: dict[str, dict] = {}
    for review_id, record in review_records.items():
        if record[schema.REVIEW_KIND] != "heterophilic_fraud":
            continue
        candidates: list[tuple[str, int]] = []
        for neighbor, edge_type in adjacency.get(review_id, []):
            neighbor_record = review_records.get(neighbor)
            if neighbor_record is None:
                continue
            score = 0
            if neighbor_record[schema.REVIEW_KIND] == "homophilic_fraud":
                score += 3
            if neighbor_record["user_id"] == record["user_id"]:
                score += 2
            if neighbor_record["device_id"] == record["device_id"]:
                score += 2
            if abs(neighbor_record[schema.TIMESTAMP] - record[schema.TIMESTAMP]) <= 3600:
                score += 1
            if abs(neighbor_record["feat_0"] - record["feat_0"]) >= 3:
                score += 1
            if edge_type in {"review-user-review", "review-device-review", "review-time-review"}:
                score += 1
            if score > 0:
                candidates.append((neighbor, score))
        candidates = sorted(candidates, key=lambda item: (-item[1], item[0]))[:3]
        evidence[review_id] = {
            "evidence_neighbors": [neighbor for neighbor, _score in candidates],
            "evidence_mechanisms": _mechanisms_for(record, [neighbor for neighbor, _score in candidates], review_records),
        }
    return evidence


def _mechanisms_for(record: dict, neighbors: list[str], review_records: dict[str, dict]) -> list[str]:
    mechanisms = {"behavioral_contradiction"}
    for neighbor in neighbors:
        neighbor_record = review_records[neighbor]
        if neighbor_record["user_id"] == record["user_id"]:
            mechanisms.add("identity_sharing")
        if neighbor_record["device_id"] == record["device_id"]:
            mechanisms.add("device_sharing")
        if abs(neighbor_record[schema.TIMESTAMP] - record[schema.TIMESTAMP]) <= 3600:
            mechanisms.add("temporal_burst")
        if abs(neighbor_record["feat_0"] - record["feat_0"]) >= 3:
            mechanisms.add("rating_divergence")
    return sorted(mechanisms)


def _write_raw(out_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, evidence: dict[str, dict]) -> dict[str, Path]:
    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    evidence_path = out_dir / "evidence_gt.json"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return {"raw_nodes": nodes_path, "raw_edges": edges_path, "raw_evidence": evidence_path}


def _write_processed(
    processed_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    evidence: dict[str, dict],
    text_dim: int,
) -> dict[str, Path]:
    nodes_path = processed_dir / "nodes.csv"
    edges_path = processed_dir / "edges.csv"
    evidence_path = processed_dir / "evidence_gt.json"
    features_path = processed_dir / "features.npz"
    split_path = processed_dir / "split.json"

    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")

    text_encoder = TfidfTextEncoder(max_features=text_dim)
    text_features = text_encoder.fit_transform(nodes[schema.TEXT].fillna("").astype(str).tolist())

    numeric_columns = numeric_feature_columns(nodes.columns.tolist())
    numeric_values = nodes[numeric_columns].fillna(0.0).to_numpy(dtype=np.float32)
    numeric_features = NumericEncoder().fit_transform(numeric_values).astype(np.float32)
    features = np.concatenate([text_features, numeric_features], axis=1).astype(np.float32)
    np.savez_compressed(
        features_path,
        node_ids=nodes[schema.NODE_ID].to_numpy(dtype=object),
        features=features,
        text_features=text_features,
        numeric_features=numeric_features,
        numeric_columns=np.array(numeric_columns, dtype=object),
    )

    split = {
        name: nodes.loc[nodes[schema.SPLIT] == name, schema.NODE_ID].tolist()
        for name in ("train", "val", "test")
    }
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "processed_nodes": nodes_path,
        "processed_edges": edges_path,
        "processed_evidence": evidence_path,
        "features": features_path,
        "split": split_path,
    }
