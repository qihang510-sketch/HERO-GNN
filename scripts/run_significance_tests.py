from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


COMPARISONS = {
    "text_rich_proxy": {
        "datasets": ["yelp_academic", "amazon_video"],
        "hero": "hero_gnn",
        "baselines": ["dgp", "mled", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx"],
    },
    "official_benchmark": {
        "datasets": ["fraud_yelp", "fraud_amazon"],
        "hero": "hero_official",
        "baselines": ["care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx"],
    },
    "transaction_benchmark": {
        "datasets": ["elliptic"],
        "hero": "hero_official",
        "baselines": ["bwgnn", "linkx", "hogrl", "rgtan"],
    },
}
METRICS = ["Macro-F1", "AUROC", "AUPRC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired significance tests for submission experiments.")
    parser.add_argument("--input_dir", default="outputs/submission_experiments")
    parser.add_argument("--output_dir", default="outputs/paper_tables_submission")
    parser.add_argument("--min_seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _read_metrics(Path(args.input_dir))
    rows = []
    for family, spec in COMPARISONS.items():
        for dataset in spec["datasets"]:
            for baseline in spec["baselines"]:
                for metric in METRICS:
                    rows.append(_compare(frame, family, dataset, spec["hero"], baseline, metric, args.min_seeds))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    _write_table(table, out_dir / "table_significance_tests")
    print(f"Wrote significance tests to {out_dir / 'table_significance_tests.csv'}")


def _read_metrics(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*/*/seed_*/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(payload)
    return pd.DataFrame(rows)


def _compare(frame: pd.DataFrame, family: str, dataset: str, hero: str, baseline: str, metric: str, min_seeds: int) -> dict[str, Any]:
    base = {
        "family": family,
        "dataset": dataset,
        "metric": metric,
        "hero_model": hero,
        "baseline_model": baseline,
    }
    if frame.empty:
        return {**base, "status": "missing_results", "paired_seed_count": 0, "p_ttest": "NA", "p_wilcoxon": "NA"}
    hero_rows = _metric_by_seed(frame, dataset, hero, metric)
    baseline_rows = _metric_by_seed(frame, dataset, baseline, metric)
    seeds = sorted(set(hero_rows) & set(baseline_rows))
    if len(seeds) < min_seeds:
        return {**base, "status": "insufficient_seeds", "paired_seed_count": len(seeds), "p_ttest": "NA", "p_wilcoxon": "NA"}
    hero_values = np.asarray([hero_rows[seed] for seed in seeds], dtype=np.float64)
    baseline_values = np.asarray([baseline_rows[seed] for seed in seeds], dtype=np.float64)
    diff = hero_values - baseline_values
    return {
        **base,
        "status": "ok",
        "paired_seed_count": len(seeds),
        "hero_mean": float(np.mean(hero_values)),
        "baseline_mean": float(np.mean(baseline_values)),
        "delta": float(np.mean(diff)),
        "p_ttest": _paired_ttest(diff),
        "p_wilcoxon": _wilcoxon(hero_values, baseline_values),
    }


def _metric_by_seed(frame: pd.DataFrame, dataset: str, model: str, metric: str) -> dict[int, float]:
    subset = frame[(frame["dataset"] == dataset) & (frame["model"] == model)] if {"dataset", "model"}.issubset(frame.columns) else pd.DataFrame()
    out: dict[int, float] = {}
    for _, row in subset.iterrows():
        value = row.get(metric, row.get(metric.lower(), row.get(metric.replace("-", "_").lower())))
        try:
            out[int(row["seed"])] = float(value)
        except (TypeError, ValueError, KeyError):
            continue
    return out


def _paired_ttest(diff: np.ndarray) -> float | str:
    try:
        from scipy.stats import ttest_1samp
    except ImportError:
        return "scipy_unavailable"
    result = ttest_1samp(diff, 0.0)
    return float(result.pvalue) if math.isfinite(float(result.pvalue)) else "NA"


def _wilcoxon(left: np.ndarray, right: np.ndarray) -> float | str:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return "scipy_unavailable"
    try:
        result = wilcoxon(left, right, zero_method="wilcox")
    except ValueError:
        return "NA"
    return float(result.pvalue) if math.isfinite(float(result.pvalue)) else "NA"


def _write_table(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_to_markdown(frame), encoding="utf-8")
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
