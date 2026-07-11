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

from src.training.submission import DATASET_MODEL_MATRIX  # noqa: E402


METRICS = ["Macro-F1", "AUROC", "AUPRC"]
METRIC_ALIASES = {
    "Macro-F1": ["Macro-F1", "macro_f1", "macro-f1", "Macro_F1"],
    "AUROC": ["AUROC", "auroc"],
    "AUPRC": ["AUPRC", "auprc"],
}
OUTPUT_COLUMNS = [
    "dataset",
    "metric",
    "hero_model",
    "baseline_model",
    "hero_mean",
    "baseline_mean",
    "delta",
    "paired_seed_count",
    "p_ttest",
    "p_wilcoxon",
    "significant_0.05",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired HERO-vs-baseline significance tests from per-seed metrics.")
    parser.add_argument("--input_dir", default="outputs/submission_experiments", help="Suite output directory or raw result directory.")
    parser.add_argument("--output_dir", default="outputs/paper_tables_submission", help="Directory for significance tables.")
    parser.add_argument("--min_paired_seeds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = read_metric_frame(Path(args.input_dir))
    rows = significance_rows(frame, metrics=METRICS, min_paired_seeds=args.min_paired_seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _write_table(table, out_dir / "table_significance")
    _write_table(table, out_dir / "table_significance_tests")
    print(f"Wrote significance tables to {out_dir}")


def read_metric_frame(input_dir: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    raw_dir = input_dir / "raw" if (input_dir / "raw").exists() else input_dir
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("metrics.json")):
        if any(part.startswith("_project_runs") for part in path.parts):
            continue
        payload = _read_json(path)
        dataset, model, seed, suite = _infer_identity(path.parent, raw_dir, payload)
        if not dataset or not model or seed is None:
            continue
        row = {
            "suite": suite or str(payload.get("suite", "main") or "main"),
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "status": str(payload.get("status", "") or ""),
            "metrics_file": str(path),
        }
        has_all_metrics = True
        for metric in METRICS:
            value = _metric_value(payload, metric)
            if value is None:
                has_all_metrics = False
            row[metric] = value
        if not row["status"]:
            row["status"] = "ok" if has_all_metrics else "missing"
        if row["status"] == "skipped":
            row["status"] = "missing"
        rows.append(row)
    return pd.DataFrame(rows, columns=["suite", "dataset", "model", "seed", "status", "metrics_file", *METRICS])


def significance_rows(
    frame: pd.DataFrame,
    metrics: list[str] | None = None,
    min_paired_seeds: int = 3,
) -> list[dict[str, Any]]:
    metrics = metrics or METRICS
    frame = _canonical_metric_frame(frame)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    datasets = _ordered_datasets(frame)
    for dataset in datasets:
        dataset_frame = frame[frame["dataset"].astype(str) == dataset]
        hero_model = _hero_model_for_dataset(dataset, dataset_frame)
        baselines = _baseline_models_for_dataset(dataset, dataset_frame)
        for baseline in baselines:
            for metric in metrics:
                rows.append(_compare(dataset_frame, dataset, hero_model, baseline, metric, min_paired_seeds))
    return rows


def _compare(
    frame: pd.DataFrame,
    dataset: str,
    hero_model: str,
    baseline_model: str,
    metric: str,
    min_paired_seeds: int,
) -> dict[str, Any]:
    subset = frame[
        (frame["suite"].astype(str).isin(["main", ""]))
        & (frame["status"].astype(str).isin(["ok", "exists"]))
    ].copy()
    base = {
        "dataset": dataset,
        "metric": metric,
        "hero_model": hero_model,
        "baseline_model": baseline_model,
        "hero_mean": pd.NA,
        "baseline_mean": pd.NA,
        "delta": pd.NA,
        "paired_seed_count": 0,
        "p_ttest": "NA",
        "p_wilcoxon": "NA",
        "significant_0.05": False,
        "status": "missing_results",
    }
    if metric not in subset:
        return base
    values = subset[["model", "seed", metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=[metric])
    hero = values[values["model"].astype(str) == hero_model][["seed", metric]].rename(columns={metric: "hero"})
    baseline = values[values["model"].astype(str) == baseline_model][["seed", metric]].rename(columns={metric: "baseline"})
    if hero.empty or baseline.empty:
        return base
    paired = hero.merge(baseline, on="seed", how="inner").sort_values("seed")
    paired_count = int(paired["seed"].nunique())
    hero_values = paired["hero"].to_numpy(dtype=float)
    baseline_values = paired["baseline"].to_numpy(dtype=float)
    hero_mean = float(np.mean(hero_values)) if hero_values.size else pd.NA
    baseline_mean = float(np.mean(baseline_values)) if baseline_values.size else pd.NA
    delta = float(hero_mean - baseline_mean) if hero_values.size else pd.NA
    base.update(
        {
            "hero_mean": hero_mean,
            "baseline_mean": baseline_mean,
            "delta": delta,
            "paired_seed_count": paired_count,
        }
    )
    if paired_count < min_paired_seeds:
        base["status"] = "insufficient_seeds"
        return base
    diff = hero_values - baseline_values
    p_ttest = _paired_ttest(diff)
    p_wilcoxon = _wilcoxon(hero_values, baseline_values)
    base.update(
        {
            "p_ttest": p_ttest,
            "p_wilcoxon": p_wilcoxon,
            "significant_0.05": _significant(p_ttest, p_wilcoxon),
            "status": "ok",
        }
    )
    return base


def _canonical_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["suite", "dataset", "model", "seed", "status", *METRICS])
    result = frame.copy()
    for column, default in [("suite", "main"), ("dataset", ""), ("model", ""), ("seed", pd.NA), ("status", "ok")]:
        if column not in result:
            result[column] = default
    for metric in METRICS:
        if metric not in result:
            result[metric] = pd.NA
        for alias in METRIC_ALIASES[metric]:
            if alias in result and alias != metric:
                result[metric] = result[metric].where(result[metric].notna(), result[alias])
    return result


def _ordered_datasets(frame: pd.DataFrame) -> list[str]:
    observed = [str(dataset) for dataset in frame["dataset"].dropna().unique()]
    ordered = [dataset for dataset in DATASET_MODEL_MATRIX if dataset in observed]
    extras = sorted(dataset for dataset in observed if dataset not in DATASET_MODEL_MATRIX)
    return [*ordered, *extras]


def _hero_model_for_dataset(dataset: str, frame: pd.DataFrame) -> str:
    preferred = "hero_gnn" if dataset in {"yelp_academic", "amazon_video"} else "hero_official"
    observed = set(str(model) for model in frame["model"].dropna().unique())
    if preferred in observed:
        return preferred
    for candidate in ["hero_gnn", "hero_official", "hero"]:
        if candidate in observed:
            return candidate
    hero_like = sorted(model for model in observed if _is_hero_model(model))
    return hero_like[0] if hero_like else preferred


def _baseline_models_for_dataset(dataset: str, frame: pd.DataFrame) -> list[str]:
    if dataset in DATASET_MODEL_MATRIX:
        return [model for model in DATASET_MODEL_MATRIX[dataset] if not _is_hero_model(model)]
    observed = sorted(str(model) for model in frame["model"].dropna().unique())
    return [model for model in observed if not _is_hero_model(model)]


def _is_hero_model(model: str) -> bool:
    text = str(model)
    return text in {"hero", "hero_gnn", "hero_official"} or text.startswith("hero_")


def _paired_ttest(diff: np.ndarray) -> float | str:
    diff = _finite(np.asarray(diff, dtype=float))
    if diff.size < 3:
        return "NA"
    try:
        from scipy.stats import ttest_1samp  # type: ignore
    except Exception:
        return "scipy_unavailable"
    result = ttest_1samp(diff, 0.0, nan_policy="omit")
    p_value = float(result.pvalue)
    return p_value if math.isfinite(p_value) else "NA"


def _wilcoxon(left: np.ndarray, right: np.ndarray) -> float | str:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 3:
        return "NA"
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        return _wilcoxon_normal_approx(left - right)
    try:
        result = wilcoxon(left, right, zero_method="wilcox", alternative="two-sided", mode="auto")
    except ValueError:
        return "NA"
    p_value = float(result.pvalue)
    return p_value if math.isfinite(p_value) else "NA"


def _wilcoxon_normal_approx(diff: np.ndarray) -> float | str:
    diff = _finite(np.asarray(diff, dtype=float))
    diff = diff[np.abs(diff) > 1e-12]
    n = int(diff.size)
    if n < 3:
        return "NA"
    ranks = _ranks(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return "NA"
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


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _significant(*p_values: float | str) -> bool:
    numeric_values = []
    for value in p_values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    return any(value < 0.05 for value in numeric_values)


def _metric_value(payload: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES[metric]:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _infer_identity(run_dir: Path, raw_dir: Path, payload: dict[str, Any]) -> tuple[str, str, int | None, str]:
    dataset = str(payload.get("dataset", "") or "")
    model = str(payload.get("model", payload.get("variant", "")) or "")
    seed = _safe_int(payload.get("seed"))
    suite = str(payload.get("suite", "") or "")
    if dataset and model and seed is not None:
        return dataset, model, seed, suite or "main"
    try:
        parts = run_dir.relative_to(raw_dir).parts
    except ValueError:
        parts = run_dir.parts
    if parts and parts[0] in {"robustness", "faithfulness", "cost"}:
        suite = suite or parts[0]
        parts = parts[1:]
    if len(parts) >= 3:
        dataset = dataset or parts[0]
        model = model or parts[1]
        seed = seed if seed is not None else _safe_int(str(parts[2]).replace("seed_", ""))
    return dataset, model, seed, suite or "main"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_table(table: pd.DataFrame, stem: Path) -> None:
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_to_markdown(table), encoding="utf-8")
    stem.with_suffix(".tex").write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def _to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "| status |\n|---|\n| missing |\n"
    return table.to_markdown(index=False)


if __name__ == "__main__":
    main()
