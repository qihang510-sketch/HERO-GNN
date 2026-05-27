from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_proxy_labels(scores: np.ndarray, anomaly_rate: float = 0.1) -> np.ndarray:
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    threshold = np.quantile(scores, 1.0 - anomaly_rate)
    return (scores >= threshold).astype(np.int64)


def build_yelp_proxy_labels(
    reviews: pd.DataFrame,
    anomaly_rate: float = 0.12,
    mode: str = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "simple":
        return _build_yelp_simple_proxy_labels(reviews, anomaly_rate=anomaly_rate)
    if mode == "hard":
        return _build_yelp_hard_proxy_labels(reviews, anomaly_rate=anomaly_rate)
    raise ValueError("mode must be one of: simple, hard")


def build_amazon_proxy_labels(
    reviews: pd.DataFrame,
    anomaly_rate: float = 0.12,
    mode: str = "simple",
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "simple":
        return _build_amazon_simple_proxy_labels(reviews, anomaly_rate=anomaly_rate)
    if mode == "hard":
        return _build_amazon_hard_proxy_labels(reviews, anomaly_rate=anomaly_rate)
    raise ValueError("mode must be one of: simple, hard")


def _build_yelp_simple_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
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


def _build_yelp_hard_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    frame = reviews.copy()
    stars = frame["stars"].astype(float)
    timestamps = frame["timestamp"].astype(float)
    day = (timestamps // 86_400).astype(int)
    month = pd.to_datetime(frame["timestamp"], unit="s", errors="coerce").dt.strftime("%Y-%m")
    extreme = stars.isin([1.0, 5.0]).astype(float)
    frame = frame.assign(day=day, month=month, extreme=extreme)

    user_day = frame.groupby(["user_id", "day"])["stars"]
    user_day_count = user_day.transform("count").astype(float)
    user_day_range = (user_day.transform("max") - user_day.transform("min")).astype(float)
    user_contradiction = ((user_day_count >= 2) & (user_day_range >= 4)).astype(float)

    business_day = frame.groupby(["business_id", "day"])
    business_day_count = business_day["node_id"].transform("count").astype(float)
    business_day_extreme = business_day["extreme"].transform("sum").astype(float)
    business_day_neighbor_count = (business_day_count - 1.0).clip(lower=0.0)
    business_day_neighbor_extreme = (business_day_extreme - extreme).clip(lower=0.0)
    business_day_extreme_share = business_day_neighbor_extreme / business_day_neighbor_count.clip(lower=1.0)

    business_month = frame.groupby(["business_id", "month"])
    business_month_count = business_month["node_id"].transform("count").astype(float)
    business_month_extreme = business_month["extreme"].transform("sum").astype(float)
    business_month_neighbor_count = (business_month_count - 1.0).clip(lower=0.0)
    business_month_neighbor_extreme = (business_month_extreme - extreme).clip(lower=0.0)
    business_month_extreme_share = business_month_neighbor_extreme / business_month_neighbor_count.clip(lower=1.0)

    user_business = frame.groupby(["user_id", "business_id"])["stars"]
    user_business_count = user_business.transform("count").astype(float)
    user_business_range = (user_business.transform("max") - user_business.transform("min")).astype(float)
    user_business_inconsistency = ((user_business_count >= 2) & (user_business_range >= 3)).astype(float)

    business_rating = frame.groupby("business_id")["stars"]
    business_count = business_rating.transform("count").astype(float)
    business_std = business_rating.transform("std").fillna(0.0).astype(float)
    business_extreme_share = frame.groupby("business_id")["extreme"].transform("mean").astype(float)
    business_polarization = _minmax((business_std * business_extreme_share * np.log1p(business_count)).to_numpy(dtype=float))

    score = (
        2.0 * user_contradiction.to_numpy(dtype=float)
        + _minmax((business_day_neighbor_extreme * business_day_extreme_share).to_numpy(dtype=float))
        + _minmax((business_month_neighbor_extreme * business_month_extreme_share).to_numpy(dtype=float))
        + 1.5 * user_business_inconsistency.to_numpy(dtype=float)
        + business_polarization
    )
    return _top_fraction_proxy_labels(score, anomaly_rate), score


def _build_amazon_simple_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
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


def _build_amazon_hard_proxy_labels(reviews: pd.DataFrame, anomaly_rate: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    frame = reviews.copy()
    overall = frame["overall"].astype(float)
    timestamps = frame["unixReviewTime"].astype(float)
    week = (timestamps // (7 * 86_400)).astype(int)
    day = (timestamps // 86_400).astype(int)
    extreme = overall.isin([1.0, 5.0]).astype(float)
    frame = frame.assign(week=week, day=day, extreme=extreme)

    reviewer_week_count = frame.groupby(["reviewerID", "week"])["node_id"].transform("count").astype(float)
    reviewer_day_count = frame.groupby(["reviewerID", "day"])["node_id"].transform("count").astype(float)
    reviewer_burst = np.maximum(
        _minmax(reviewer_week_count.to_numpy(dtype=float)),
        _minmax(reviewer_day_count.to_numpy(dtype=float)),
    )

    product_week = frame.groupby(["asin", "week"])
    product_week_count = product_week["node_id"].transform("count").astype(float)
    product_week_extreme = product_week["extreme"].transform("sum").astype(float)
    product_week_neighbor_count = (product_week_count - 1.0).clip(lower=0.0)
    product_week_neighbor_extreme = (product_week_extreme - extreme).clip(lower=0.0)
    product_week_extreme_share = product_week_neighbor_extreme / product_week_neighbor_count.clip(lower=1.0)
    product_week_burst = _minmax((product_week_neighbor_count * product_week_extreme_share).to_numpy(dtype=float))

    product_day_count = frame.groupby(["asin", "day"])["node_id"].transform("count").astype(float)
    product_short_burst = _minmax((product_day_count - 1.0).clip(lower=0.0).to_numpy(dtype=float))

    product_rating = frame.groupby("asin")["overall"]
    product_count = product_rating.transform("count").astype(float)
    product_std = product_rating.transform("std").fillna(0.0).astype(float)
    product_extreme_share = frame.groupby("asin")["extreme"].transform("mean").astype(float)
    product_polarization = _minmax((product_std * product_extreme_share * np.log1p(product_count)).to_numpy(dtype=float))

    helpful_total = frame["helpful_total"].astype(float)
    helpful_ratio = frame["helpful_ratio"].astype(float)
    suspicious_helpful = ((helpful_total >= 3) & (helpful_ratio <= 0.34)).astype(float)
    neighborhood_burst = np.maximum(reviewer_burst, np.maximum(product_week_burst, product_short_burst))
    helpful_with_burst = suspicious_helpful.to_numpy(dtype=float) * (neighborhood_burst > 0).astype(float) * neighborhood_burst

    score = (
        1.5 * reviewer_burst
        + 1.5 * product_week_burst
        + product_short_burst
        + product_polarization
        + helpful_with_burst
    )
    return _top_fraction_proxy_labels(score, anomaly_rate), score


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def _top_fraction_proxy_labels(scores: np.ndarray, anomaly_rate: float) -> np.ndarray:
    if not 0 < anomaly_rate < 1:
        raise ValueError("anomaly_rate must be between 0 and 1")
    scores = np.asarray(scores, dtype=float)
    labels = np.zeros(scores.shape[0], dtype=np.int64)
    if scores.size == 0:
        return labels
    count = max(1, int(round(scores.size * anomaly_rate)))
    count = min(count, scores.size)
    ranked = np.argsort(scores, kind="mergesort")[::-1]
    labels[ranked[:count]] = 1
    return labels
