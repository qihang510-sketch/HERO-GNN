import gzip
import json

import pandas as pd

from src.data.amazon_preprocess import preprocess_amazon_video
from src.data.loader import load_processed_data


def test_amazon_preprocess_mock_raw_file(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()
    records = [
        {"reviewerID": "u1", "asin": "p1", "reviewText": "Excellent video", "overall": 5, "summary": "Great", "unixReviewTime": 1_600_000_000, "helpful": [0, 4]},
        {"reviewerID": "u1", "asin": "p1", "reviewText": "Terrible", "overall": 1, "summary": "Bad", "unixReviewTime": 1_600_000_100, "helpful": [0, 3]},
        {"reviewerID": "u2", "asin": "p1", "reviewText": "It was fine", "overall": 4, "summary": "Fine", "unixReviewTime": 1_600_050_000, "helpful": [2, 3]},
        {"reviewerID": "u3", "asin": "p2", "reviewText": "Okay", "overall": 3, "summary": "Okay", "unixReviewTime": 1_600_100_000, "helpful": [1, 1]},
        {"reviewerID": "u3", "asin": "p2", "reviewText": "Loved it", "overall": 5, "summary": "Loved", "unixReviewTime": 1_600_100_050, "helpful": [0, 5]},
        {"reviewerID": "u4", "asin": "p2", "reviewText": "Average", "overall": 3, "summary": "Average", "unixReviewTime": 1_600_200_000, "helpful": [1, 2]},
    ]
    with gzip.open(raw_dir / "reviews_Amazon_Instant_Video_5.json.gz", "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    output = preprocess_amazon_video(raw_dir, out_dir, seed=0, text_dim=8)
    assert output == out_dir
    for name in ["nodes.csv", "edges.csv", "features.npz", "split.json"]:
        assert (out_dir / name).exists()

    nodes = pd.read_csv(out_dir / "nodes.csv")
    edges = pd.read_csv(out_dir / "edges.csv")
    graph = load_processed_data(out_dir)
    assert set(nodes["node_type"]) == {"review"}
    assert {"review-user-review", "review-product-review", "review-rating-review", "review-week-review"} & set(edges["edge_type"])
    assert graph.features.shape[0] == len(nodes)
    assert set(nodes["label"]).issubset({0, 1})
