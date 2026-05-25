import json

import pandas as pd

from src.data.loader import load_processed_data
from src.data.yelp_preprocess import preprocess_yelp_academic


def test_yelp_preprocess_mock_raw_files(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    _write_jsonl(
        raw_dir / "yelp_academic_dataset_review.json",
        [
            {"review_id": "yr1", "user_id": "u1", "business_id": "b1", "stars": 5, "useful": 0, "funny": 0, "cool": 0, "text": "Great", "date": "2020-01-01 10:00:00"},
            {"review_id": "yr2", "user_id": "u1", "business_id": "b1", "stars": 1, "useful": 1, "funny": 0, "cool": 0, "text": "Bad", "date": "2020-01-01 11:00:00"},
            {"review_id": "yr3", "user_id": "u2", "business_id": "b1", "stars": 4, "useful": 2, "funny": 1, "cool": 1, "text": "Pretty good meal", "date": "2020-01-05 10:00:00"},
            {"review_id": "yr4", "user_id": "u3", "business_id": "b2", "stars": 2, "useful": 0, "funny": 0, "cool": 0, "text": "Not for me", "date": "2020-02-01 10:00:00"},
            {"review_id": "yr5", "user_id": "u3", "business_id": "b2", "stars": 5, "useful": 0, "funny": 0, "cool": 0, "text": "Amazing", "date": "2020-02-01 10:30:00"},
            {"review_id": "yr6", "user_id": "u4", "business_id": "b2", "stars": 3, "useful": 1, "funny": 0, "cool": 0, "text": "Average place", "date": "2020-02-03 10:00:00"},
        ],
    )
    _write_jsonl(
        raw_dir / "yelp_academic_dataset_user.json",
        [
            {"user_id": "u1", "review_count": 2, "average_stars": 3.0, "fans": 0, "useful": 1, "funny": 0, "cool": 0},
            {"user_id": "u2", "review_count": 10, "average_stars": 4.0, "fans": 1, "useful": 5, "funny": 1, "cool": 2},
            {"user_id": "u3", "review_count": 2, "average_stars": 3.5, "fans": 0, "useful": 0, "funny": 0, "cool": 0},
            {"user_id": "u4", "review_count": 8, "average_stars": 3.5, "fans": 0, "useful": 1, "funny": 0, "cool": 0},
        ],
    )
    _write_jsonl(
        raw_dir / "yelp_academic_dataset_business.json",
        [
            {"business_id": "b1", "stars": 4.0, "review_count": 100, "is_open": 1},
            {"business_id": "b2", "stars": 3.0, "review_count": 50, "is_open": 1},
        ],
    )

    output = preprocess_yelp_academic(raw_dir, out_dir, seed=0, max_reviews=10, text_dim=8)
    assert output == out_dir
    for name in ["nodes.csv", "edges.csv", "features.npz", "split.json"]:
        assert (out_dir / name).exists()

    nodes = pd.read_csv(out_dir / "nodes.csv")
    edges = pd.read_csv(out_dir / "edges.csv")
    graph = load_processed_data(out_dir)
    assert set(nodes["node_type"]) == {"review"}
    assert {"review-user-review", "review-business-review", "review-rating-review", "review-month-review"} & set(edges["edge_type"])
    assert graph.features.shape[0] == len(nodes)
    assert set(nodes["label"]).issubset({0, 1})


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
