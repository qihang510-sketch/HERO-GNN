from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_proxy_labels(scores: np.ndarray, anomaly_rate: float = 0.1) -> np.ndarray:
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    threshold = np.quantile(scores, 1.0 - anomaly_rate)
    return (scores >= threshold).astype(np.int64)


def build_yelp_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    stars = reviews["stars"].astype(float)
    timestamps = reviews["timestamp"].astype(float)
    extreme = stars.isin([1.0, 5.0]).astype(float)
    rating_extreme_score = extreme

    user_day = reviews.assign(day=(timestamps // 86_400).astype(int)).groupby(["user_id", "day"])["node_id"].transform("count")
    user_burst_score = _minmax(user_day.to_numpy(dtype=float))

    business_day_extreme = (
        reviews.assign(day=(timestamps // 86_400).astype(int), extreme=extreme)
        .groupby(["business_id", "day"])["extreme"]
        .transform("sum")
    )
    business_burst_score = _minmax(business_day_extreme.to_numpy(dtype=float))

    user_history = reviews["user_review_count"].astype(float).fillna(0.0)
    low_user_history_score = ((user_history <= 3) & (extreme == 1)).astype(float)

    text_length = reviews["review_text_length"].astype(float)
    short_text_extreme_rating_score = ((text_length <= 80) & (extreme == 1)).astype(float)

    score = (
        rating_extreme_score.to_numpy(dtype=float)
        + user_burst_score
        + business_burst_score
        + low_user_history_score.to_numpy(dtype=float)
        + short_text_extreme_rating_score.to_numpy(dtype=float)
    )
    return quantile_proxy_labels(score, anomaly_rate), score


def build_amazon_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    overall = reviews["overall"].astype(float)
    timestamps = reviews["unixReviewTime"].astype(float)
    extreme = overall.isin([1.0, 5.0]).astype(float)
    helpful_total = reviews["helpful_total"].astype(float)
    helpful_ratio = reviews["helpful_ratio"].astype(float)

    low_helpfulness_score = ((helpful_total >= 3) & (helpful_ratio <= 0.34)).astype(float)
    rating_extreme_score = extreme

    reviewer_week = reviews.assign(week=(timestamps // (7 * 86_400)).astype(int)).groupby(["reviewerID", "week"])["node_id"].transform("count")
    reviewer_burst_score = _minmax(reviewer_week.to_numpy(dtype=float))

    product_week_extreme = (
        reviews.assign(week=(timestamps // (7 * 86_400)).astype(int), extreme=extreme)
        .groupby(["asin", "week"])["extreme"]
        .transform("sum")
    )
    product_burst_score = _minmax(product_week_extreme.to_numpy(dtype=float))

    text_length = reviews["review_text_length"].astype(float)
    short_text_extreme_rating_score = ((text_length <= 80) & (extreme == 1)).astype(float)

    score = (
        low_helpfulness_score.to_numpy(dtype=float)
        + rating_extreme_score.to_numpy(dtype=float)
        + reviewer_burst_score
        + product_burst_score
        + short_text_extreme_rating_score.to_numpy(dtype=float)
    )
    return quantile_proxy_labels(score, anomaly_rate), score


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)
