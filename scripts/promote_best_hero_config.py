from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import HERO_CONFIG_KEYS  # noqa: E402


METRIC_MAP = {
    "auprc": "AUPRC_mean",
    "ap": "AUPRC_mean",
    "macro_f1": "Macro-F1_mean",
    "macro-f1": "Macro-F1_mean",
    "f1": "Macro-F1_mean",
    "auroc": "AUROC_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely promote a HERO tuning config only if it beats baseline.")
    parser.add_argument("--tuning_dir", required=True)
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.005)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuning_dir = Path(args.tuning_dir)
    results_path = tuning_dir / "tuning_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing tuning results: {results_path}")
    frame = pd.read_csv(results_path)
    ranking_frame = _ranking_rows(frame)
    ranked = _rank(ranking_frame, args.metric)
    top_k = ranked.head(max(1, int(args.top_k))).copy()
    if top_k.empty:
        raise ValueError("No tuning rows available to promote.")
    baseline = _baseline_row(ranked)
    if baseline is None:
        raise ValueError("Missing baseline trial in tuning_results.csv. Re-run run_hero_tuning.py with baseline trial enabled.")
    nonbaseline = ranked[~ranked["is_baseline"].map(_is_true)].copy()
    if nonbaseline.empty:
        _write_no_promote_report(
            tuning_dir=tuning_dir,
            output_config=Path(args.output_config),
            baseline=baseline,
            candidate=None,
            min_delta=float(args.min_delta),
            reason="No non-baseline tuning trial was found.",
            top_k=top_k,
            metric=args.metric,
        )
        return
    best = nonbaseline.iloc[0].to_dict()
    score_col = _safety_score_column(ranked)
    baseline_score = _float_or_none(baseline.get(score_col))
    candidate_score = _float_or_none(best.get(score_col))
    delta = None if baseline_score is None or candidate_score is None else candidate_score - baseline_score
    top_payload = [_json_ready(_row_with_config(row)) for row in top_k.to_dict(orient="records")]
    (tuning_dir / "top_promoted_configs.json").write_text(json.dumps(top_payload, indent=2, sort_keys=True), encoding="utf-8")
    if delta is None or delta < float(args.min_delta):
        _write_no_promote_report(
            tuning_dir=tuning_dir,
            output_config=Path(args.output_config),
            baseline=baseline,
            candidate=best,
            min_delta=float(args.min_delta),
            reason="No tuned config outperformed baseline. Keep original HERO-GNN config.",
            top_k=top_k,
            metric=args.metric,
        )
        print("No tuned config outperformed baseline. Keep original HERO-GNN config.")
        return
    output_config = Path(args.output_config)
    if output_config.exists() and not args.overwrite:
        raise FileExistsError(f"{output_config} exists. Pass --overwrite to replace it.")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    promoted = _promoted_payload(best, _parse_config(best), source=str(results_path), score_col=score_col, baseline=baseline)
    output_config.write_text(yaml.safe_dump(promoted, sort_keys=False), encoding="utf-8")
    _write_promote_report(tuning_dir, output_config, top_k, args.metric, baseline, best, score_col, float(args.min_delta))
    print(f"Promoted HERO config: {output_config}")


def _ranking_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in frame:
        dataset_values = set(frame["dataset"].astype(str))
        if "joint" in dataset_values:
            return frame[frame["dataset"].astype(str) == "joint"].copy()
    return frame.copy()


def _rank(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "mean_AUPRC_across_datasets" in frame and frame["mean_AUPRC_across_datasets"].notna().any():
        for column in ["mean_AUPRC_across_datasets", "min_AUPRC_across_datasets", "Macro-F1_mean"]:
            if column not in frame:
                frame[column] = float("nan")
        ranked = frame.sort_values(
            ["mean_AUPRC_across_datasets", "min_AUPRC_across_datasets", "Macro-F1_mean"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    else:
        primary = METRIC_MAP.get(str(metric).lower(), "AUPRC_mean")
        for column in [primary, "Macro-F1_mean", "AUROC_mean"]:
            if column not in frame:
                frame[column] = float("nan")
        ranked = frame.sort_values([primary, "Macro-F1_mean", "AUROC_mean"], ascending=False, na_position="last").reset_index(drop=True)
    ranked["rank"] = list(range(1, len(ranked) + 1))
    return ranked


def _baseline_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    baseline_rows = frame[frame["is_baseline"].map(_is_true)] if "is_baseline" in frame else pd.DataFrame()
    if baseline_rows.empty:
        return None
    return baseline_rows.iloc[0].to_dict()


def _safety_score_column(frame: pd.DataFrame) -> str:
    if "dataset" in frame and "joint" in set(frame["dataset"].astype(str)) and "mean_AUPRC_across_datasets" in frame:
        return "mean_AUPRC_across_datasets"
    return "AUPRC_mean"


def _parse_config(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("config_json", "{}")
    if isinstance(raw, str):
        params = json.loads(raw)
    else:
        params = dict(raw or {})
    return params


def _promoted_payload(
    row: dict[str, Any],
    params: dict[str, Any],
    source: str,
    score_col: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    hero_config = {key: value for key, value in params.items() if key in HERO_CONFIG_KEYS}
    payload = {
        "name": "hero_gnn_tuned",
        "base_model": row.get("model", "hero_gnn"),
        "display_name": "HERO-GNN tuned",
        "implementation_source": "project",
        "tuning_source": source,
        "trial_id": str(row.get("trial_id", "")),
        "trial_name": str(row.get("trial_name", "")),
        "dataset": str(row.get("dataset", "")),
        "seeds": str(row.get("seeds", "")),
        "safety_check": {
            "score_column": score_col,
            "baseline_score": _float_or_none(baseline.get(score_col)),
            "promoted_score": _float_or_none(row.get(score_col)),
            "baseline_trial_id": str(baseline.get("trial_id", "baseline")),
        },
        "ranking_metrics": {
            "AUPRC_mean": _float_or_none(row.get("AUPRC_mean")),
            "Macro-F1_mean": _float_or_none(row.get("Macro-F1_mean")),
            "AUROC_mean": _float_or_none(row.get("AUROC_mean")),
            "mean_AUPRC_across_datasets": _float_or_none(row.get("mean_AUPRC_across_datasets")),
            "min_AUPRC_across_datasets": _float_or_none(row.get("min_AUPRC_across_datasets")),
        },
        "hero_config": hero_config,
    }
    for key in ["learning_rate", "hidden_dim", "top_k", "epochs"]:
        if key in params:
            payload[key] = params[key]
    return payload


def _row_with_config(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "config": _parse_config(row)}


def _write_no_promote_report(
    tuning_dir: Path,
    output_config: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any] | None,
    min_delta: float,
    reason: str,
    top_k: pd.DataFrame,
    metric: str,
) -> None:
    score_col = _safety_score_column(top_k)
    candidate = candidate or {}
    lines = [
        "# Promoted HERO config report",
        "",
        f"- output_config: {output_config}",
        f"- action: no_promote",
        f"- reason: {reason}",
        f"- min_delta: {min_delta}",
        f"- safety_score_column: {score_col}",
        f"- baseline_trial_id: {baseline.get('trial_id')}",
        f"- baseline_score: {_fmt(baseline.get(score_col))}",
        f"- best_nonbaseline_trial_id: {candidate.get('trial_id', '')}",
        f"- best_nonbaseline_score: {_fmt(candidate.get(score_col))}",
        "",
        "No tuned config outperformed baseline. Keep original HERO-GNN config.",
        "",
        "The original configs/models/hero_gnn.yaml is not modified by this script.",
    ]
    (tuning_dir / "promoted_config_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_promote_report(
    tuning_dir: Path,
    output_config: Path,
    top_k: pd.DataFrame,
    metric: str,
    baseline: dict[str, Any],
    best: dict[str, Any],
    score_col: str,
    min_delta: float,
) -> None:
    lines = [
        "# Promoted HERO config report",
        "",
        f"- output_config: {output_config}",
        f"- action: promote",
        f"- ranked_by: {_ranking_text(metric, top_k)}",
        f"- min_delta: {min_delta}",
        f"- safety_score_column: {score_col}",
        f"- baseline_trial_id: {baseline.get('trial_id')}",
        f"- baseline_score: {_fmt(baseline.get(score_col))}",
        f"- promoted_trial_id: {best.get('trial_id')}",
        f"- promoted_score: {_fmt(best.get(score_col))}",
        f"- score_delta: {_fmt((_float_or_none(best.get(score_col)) or 0.0) - (_float_or_none(baseline.get(score_col)) or 0.0))}",
        f"- top_k: {len(top_k)}",
        "",
        "The original configs/models/hero_gnn.yaml is not modified by this script.",
    ]
    (tuning_dir / "promoted_config_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ranking_text(metric: str, frame: pd.DataFrame) -> str:
    if "dataset" in frame and "joint" in set(frame["dataset"].astype(str)) and "mean_AUPRC_across_datasets" in frame:
        return "mean_AUPRC_across_datasets, min_AUPRC_across_datasets, Macro-F1_mean"
    return f"{METRIC_MAP.get(str(metric).lower(), 'AUPRC_mean')}, Macro-F1_mean, AUROC_mean"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None else f"{numeric:.6f}"


if __name__ == "__main__":
    main()
