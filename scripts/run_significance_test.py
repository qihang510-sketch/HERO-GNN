from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["macro_f1", "auroc", "auprc"]
OUTPUT_COLUMNS = ["dataset", "metric", "hero_mean", "baseline_name", "baseline_mean", "delta", "p_value", "test_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired significance tests between HERO-GNN and the best baseline.")
    parser.add_argument("--result_dir", default="outputs/results", help="Directory containing metrics.json files.")
    parser.add_argument("--out_file", default="outputs/summary/significance_table.csv", help="Output CSV path.")
    parser.add_argument("--metrics", nargs="+", default=METRICS, help="Metrics to test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _load_metrics(Path(args.result_dir))
    rows = significance_rows(frame, args.metrics)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
    print(f"Wrote significance table to {out_file}")


def significance_rows(frame: pd.DataFrame, metrics: list[str]) -> list[dict]:
    if frame.empty:
        return []
    rows = []
    for dataset, dataset_frame in frame.groupby("dataset"):
        hero = dataset_frame[dataset_frame["model"] == "hero_gnn"]
        if hero.empty:
            continue
        baselines = dataset_frame[~dataset_frame["model"].str.startswith("hero_")].copy()
        if baselines.empty:
            continue
        for metric in metrics:
            if metric not in dataset_frame:
                continue
            baseline_name = _best_baseline_name(baselines, metric)
            if not baseline_name:
                continue
            baseline = baselines[baselines["model"] == baseline_name]
            paired = hero[["seed", metric]].merge(baseline[["seed", metric]], on="seed", suffixes=("_hero", "_baseline"))
            hero_mean = float(hero[metric].mean())
            baseline_mean = float(baseline[metric].mean())
            delta = hero_mean - baseline_mean
            p_value = _wilcoxon_signed_rank_pvalue((paired[f"{metric}_hero"] - paired[f"{metric}_baseline"]).to_numpy(dtype=float))
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "hero_mean": hero_mean,
                    "baseline_name": baseline_name,
                    "baseline_mean": baseline_mean,
                    "delta": delta,
                    "p_value": p_value,
                    "test_type": "wilcoxon_signed_rank_normal_approx",
                }
            )
    return rows


def _load_metrics(result_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(result_dir.glob("*/*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["dataset", "model", "seed", *METRICS])


def _best_baseline_name(frame: pd.DataFrame, metric: str) -> str:
    if metric not in frame or frame.empty:
        return ""
    means = frame.groupby("model")[metric].mean().sort_values(ascending=False)
    return str(means.index[0]) if not means.empty else ""


def _wilcoxon_signed_rank_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > 1e-12]
    n = int(diffs.size)
    if n == 0:
        return 1.0
    ranks = _ranks(np.abs(diffs))
    w_plus = float(np.sum(ranks[diffs > 0]))
    w_minus = float(np.sum(ranks[diffs < 0]))
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return 1.0
    z = (statistic - mean + 0.5) / math.sqrt(variance)
    return float(min(max(2.0 * _normal_cdf(z), 0.0), 1.0))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


if __name__ == "__main__":
    main()
