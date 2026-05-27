from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.proxy_label import build_yelp_proxy_labels
from src.features.numeric_encoder import NumericEncoder, numeric_feature_columns
from src.features.text_encoder import TfidfTextEncoder

YELP_LABEL_FEATURES = [
    "stars",
    "review_text_length",
    "useful",
    "funny",
    "cool",
    "user_review_count",
    "business_review_count",
    "proxy_anomaly_score",
]


def preprocess_yelp_academic(
    raw_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_reviews: int = 100_000,
    max_neighbors_per_type: int = 30,
    text_dim: int = 128,
    proxy_label_mode: str = "simple",
    remove_label_features: bool = False,
) -> Path:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_path = raw_dir / "yelp_academic_dataset_review.json"
    user_path = raw_dir / "yelp_academic_dataset_user.json"
    business_path = raw_dir / "yelp_academic_dataset_business.json"
    _require_files([review_path, user_path, business_path])

    reviews = _read_jsonl(review_path, max_records=max_reviews)
    users = _read_jsonl(user_path)
    businesses = _read_jsonl(business_path)

    review_df = _review_frame(reviews)
    user_df = _user_frame(users)
    business_df = _business_frame(businesses)
    review_df = review_df.merge(user_df, on="user_id", how="left")
    review_df = review_df.merge(business_df, on="business_id", how="left")
    review_df = review_df.fillna(0)

    labels, scores = build_yelp_proxy_labels(review_df, mode=proxy_label_mode)
    review_df["label"] = labels
    review_df["proxy_anomaly_score"] = scores
    review_df["split"] = _stratified_splits(labels, seed=seed)
    review_df = _add_weak_time_features(review_df)

    nodes, removed_label_features = _nodes_from_reviews(review_df, remove_label_features=remove_label_features)
    edges = _build_edges(review_df, seed=seed, max_neighbors_per_type=max_neighbors_per_type)
    final_feature_dim = _write_processed(output_dir, nodes, edges, text_dim=text_dim)
    _write_preprocess_report(
        output_dir=output_dir,
        dataset="yelp_academic",
        proxy_label_mode=proxy_label_mode,
        remove_label_features=remove_label_features,
        nodes=nodes,
        edges=edges,
        removed_label_features=removed_label_features,
        final_feature_dim=final_feature_dim,
    )
    return output_dir


def _read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def _review_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(records):
        text = str(row.get("text", ""))
        timestamp = pd.to_datetime(row.get("date", "1970-01-01"), errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp("1970-01-01")
        rows.append(
            {
                "node_id": str(row.get("review_id", f"yr_{index:07d}")),
                "user_id": str(row.get("user_id", "")),
                "business_id": str(row.get("business_id", "")),
                "text": text,
                "stars": float(row.get("stars", 0.0)),
                "useful": float(row.get("useful", 0.0)),
                "funny": float(row.get("funny", 0.0)),
                "cool": float(row.get("cool", 0.0)),
                "review_text_length": float(len(text)),
                "timestamp": int(timestamp.timestamp()),
                "month": timestamp.strftime("%Y-%m"),
            }
        )
    return pd.DataFrame(rows)


def _user_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in records:
        rows.append(
            {
                "user_id": str(row.get("user_id", "")),
                "user_review_count": float(row.get("review_count", 0.0)),
                "user_average_stars": float(row.get("average_stars", 0.0)),
                "user_fans": float(row.get("fans", 0.0)),
                "user_useful": float(row.get("useful", 0.0)),
                "user_funny": float(row.get("funny", 0.0)),
                "user_cool": float(row.get("cool", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _business_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in records:
        rows.append(
            {
                "business_id": str(row.get("business_id", "")),
                "business_stars": float(row.get("stars", 0.0)),
                "business_review_count": float(row.get("review_count", 0.0)),
                "business_is_open": float(row.get("is_open", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _add_weak_time_features(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews.copy()
    timestamps = pd.to_datetime(reviews["timestamp"], unit="s", errors="coerce")
    month = timestamps.dt.month.fillna(1).astype(float)
    day_of_week = timestamps.dt.dayofweek.fillna(0).astype(float)
    reviews["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    reviews["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    reviews["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    reviews["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    return reviews


def _nodes_from_reviews(reviews: pd.DataFrame, remove_label_features: bool) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [column for column in _feature_columns(remove_label_features) if column in reviews.columns]
    removed_label_features = [column for column in YELP_LABEL_FEATURES if column in reviews.columns] if remove_label_features else []
    nodes = pd.DataFrame(
        {
            "node_id": reviews["node_id"],
            "node_type": "review",
            "text": reviews["text"],
            "label": reviews["label"].astype(int),
            "split": reviews["split"],
            "timestamp": reviews["timestamp"].astype(int),
        }
    )
    for idx, column in enumerate(feature_columns):
        nodes[f"feat_{idx}"] = reviews[column].astype(float)
    return nodes, removed_label_features


def _feature_columns(remove_label_features: bool) -> list[str]:
    if remove_label_features:
        return [
            "month_sin",
            "month_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "user_average_stars",
            "user_fans",
            "user_useful",
            "user_funny",
            "user_cool",
            "business_stars",
            "business_is_open",
        ]
    return [
        "stars",
        "useful",
        "funny",
        "cool",
        "review_text_length",
        "user_review_count",
        "user_average_stars",
        "user_fans",
        "user_useful",
        "user_funny",
        "user_cool",
        "business_stars",
        "business_review_count",
        "business_is_open",
        "proxy_anomaly_score",
    ]


def _build_edges(reviews: pd.DataFrame, seed: int, max_neighbors_per_type: int) -> pd.DataFrame:
    frames = [
        _edges_for_group(reviews, ["user_id"], "review-user-review", seed, max_neighbors_per_type),
        _edges_for_group(reviews, ["business_id"], "review-business-review", seed + 1, max_neighbors_per_type),
        _edges_for_group(reviews.assign(rating_bucket=reviews["stars"].round().astype(int)), ["business_id", "rating_bucket"], "review-rating-review", seed + 2, max_neighbors_per_type),
        _edges_for_group(reviews, ["business_id", "month"], "review-month-review", seed + 3, max_neighbors_per_type),
    ]
    edges = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["src", "dst", "edge_type", "timestamp"])
    return edges.drop_duplicates(["src", "dst", "edge_type"]).reset_index(drop=True)


def _edges_for_group(
    reviews: pd.DataFrame,
    group_cols: list[str],
    edge_type: str,
    seed: int,
    max_neighbors_per_type: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _group, group in reviews.groupby(group_cols):
        ids = group["node_id"].astype(str).to_numpy()
        times = dict(zip(group["node_id"].astype(str), group["timestamp"].astype(int)))
        if len(ids) < 2:
            continue
        for src in ids:
            candidates = ids[ids != src]
            if candidates.size > max_neighbors_per_type:
                candidates = rng.choice(candidates, size=max_neighbors_per_type, replace=False)
            for dst in candidates:
                rows.append({"src": src, "dst": str(dst), "edge_type": edge_type, "timestamp": int(min(times[src], times[str(dst)]))})
    return pd.DataFrame(rows, columns=["src", "dst", "edge_type", "timestamp"])


def _write_processed(output_dir: Path, nodes: pd.DataFrame, edges: pd.DataFrame, text_dim: int) -> int:
    nodes.to_csv(output_dir / "nodes.csv", index=False)
    edges.to_csv(output_dir / "edges.csv", index=False)

    text_features = TfidfTextEncoder(max_features=text_dim).fit_transform(nodes["text"].fillna("").astype(str).tolist())
    numeric_columns = numeric_feature_columns(nodes.columns.tolist())
    numeric_features = _numeric_feature_matrix(nodes, numeric_columns)
    features = np.concatenate([text_features, numeric_features], axis=1).astype(np.float32)
    np.savez_compressed(
        output_dir / "features.npz",
        node_ids=nodes["node_id"].to_numpy(dtype=object),
        features=features,
        text_features=text_features,
        numeric_features=numeric_features,
        numeric_columns=np.array(numeric_columns, dtype=object),
    )
    split = {name: nodes.loc[nodes["split"] == name, "node_id"].tolist() for name in ("train", "val", "test")}
    (output_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    return int(features.shape[1])


def _numeric_feature_matrix(nodes: pd.DataFrame, numeric_columns: list[str]) -> np.ndarray:
    if not numeric_columns:
        return np.zeros((len(nodes), 0), dtype=np.float32)
    return NumericEncoder().fit_transform(nodes[numeric_columns].fillna(0.0).to_numpy(dtype=np.float32)).astype(np.float32)


def _write_preprocess_report(
    output_dir: Path,
    dataset: str,
    proxy_label_mode: str,
    remove_label_features: bool,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    removed_label_features: list[str],
    final_feature_dim: int,
) -> None:
    labels = nodes["label"].astype(int)
    num_positive = int(np.sum(labels == 1))
    num_negative = int(np.sum(labels == 0))
    report = {
        "dataset": dataset,
        "proxy_label_mode": proxy_label_mode,
        "remove_label_features": bool(remove_label_features),
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "positive_rate": float(num_positive / len(nodes)) if len(nodes) else 0.0,
        "removed_label_features": removed_label_features,
        "final_feature_dim": int(final_feature_dim),
    }
    (output_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[PROXY] mode={proxy_label_mode}")
    print(f"[PROXY] anomaly_ratio={report['positive_rate']}")
    print(f"[PROXY] remove_label_features={str(remove_label_features).lower()}")
    print(f"[PROXY] num_positive={num_positive}")
    print(f"[PROXY] num_negative={num_negative}")
    print(f"[FEATURES] removed_label_features={removed_label_features}")
    print(f"[FEATURES] final_feature_dim={final_feature_dim}")


def _stratified_splits(labels: np.ndarray, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    splits = np.empty(len(labels), dtype=object)
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.6)
        val_end = train_end + int(len(indices) * 0.2)
        splits[indices[:train_end]] = "train"
        splits[indices[train_end:val_end]] = "val"
        splits[indices[val_end:]] = "test"
    return splits.tolist()


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Yelp raw file(s): {missing}")
