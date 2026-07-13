from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_significance_tests import METRICS, significance_rows  # noqa: E402
from src.training.submission import DATASET_MODEL_MATRIX, TEXT_RICH_DATASETS, hero_display_name  # noqa: E402


METRIC_ALIASES = {
    "Macro-F1": ["Macro-F1", "macro_f1", "macro-f1", "Macro_F1"],
    "AUROC": ["AUROC", "auroc"],
    "AUPRC": ["AUPRC", "auprc"],
}
SUMMARY_COLUMNS = [
    "suite",
    "dataset",
    "model",
    "seed",
    "status",
    "skip_reason",
    "Macro-F1",
    "AUROC",
    "AUPRC",
    "run_dir",
    "metrics_file",
]
TEXT_RICH_SET = set(TEXT_RICH_DATASETS)
TRANSFER_DATASETS = {"fraud_yelp", "fraud_amazon", "elliptic"}
MEAN_STD_COLUMNS = [
    "dataset",
    "model",
    "seed_count",
    "Macro-F1_mean",
    "Macro-F1_std",
    "Macro-F1_mean_std",
    "AUROC_mean",
    "AUROC_std",
    "AUROC_mean_std",
    "AUPRC_mean",
    "AUPRC_std",
    "AUPRC_mean_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize unified HERO experiment suite outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_paired_seeds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_suite(Path(args.output_dir), min_paired_seeds=args.min_paired_seeds)
    print(f"Wrote experiment summaries to {Path(args.output_dir) / 'summary'}")


def summarize_suite(output_dir: str | Path, min_paired_seeds: int = 3) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    summary_dir = output_dir / "summary"
    tables_dir = output_dir / "tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_run_records(output_dir)
    missing = missing_runs(output_dir, all_runs)
    main_table = mean_std_table(all_runs, suite="main", datasets=TEXT_RICH_SET)
    ablation_table = mean_std_table(all_runs, suite="ablation", model_label="variant", datasets=TEXT_RICH_SET)
    transfer_table = mean_std_table(all_runs, suite={"main", "transfer"}, datasets=TRANSFER_DATASETS)
    sig_table = pd.DataFrame(significance_rows(all_runs, metrics=METRICS, min_paired_seeds=min_paired_seeds))

    outputs = {
        "all_raw_runs": all_runs,
        "table_main_mean_std": main_table,
        "table_ablation_mean_std": ablation_table,
        "table_transfer_mean_std": transfer_table,
        "table_significance": sig_table,
        "table_missing_runs": missing,
    }
    outputs.update(_advanced_tables(output_dir))
    for stem, frame in outputs.items():
        frame.to_csv(summary_dir / f"{stem}.csv", index=False)
        frame.to_csv(tables_dir / f"{stem}.csv", index=False)
    return outputs


def load_run_records(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw" if (output_dir / "raw").exists() else output_dir
    rows = []
    seen: set[Path] = set()
    markers = [*raw_dir.rglob("config.json"), *raw_dir.rglob("metrics.json"), *raw_dir.rglob("skip_reason.json")]
    for marker in sorted(markers):
        if _is_internal_project_path(marker):
            continue
        run_dir = marker.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        row = _read_run_dir(run_dir, raw_dir)
        if row is not None:
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[SUMMARY_COLUMNS + [col for col in frame.columns if col not in SUMMARY_COLUMNS]]


def _is_internal_project_path(path: Path) -> bool:
    return any(part.startswith("_project_runs") for part in path.parts)


def _read_run_dir(run_dir: Path, raw_dir: Path) -> dict[str, Any] | None:
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    skip = _read_json(run_dir / "skip_reason.json")
    runtime = _read_json(run_dir / "runtime.json")
    payload: dict[str, Any] = {}
    for source in [config, metrics, skip, runtime]:
        payload.update({key: value for key, value in source.items() if value is not None})
    dataset, model, seed, suite = _infer_identity(run_dir, raw_dir, payload)
    if _is_missing_scalar(dataset) or _is_missing_scalar(model) or seed is None:
        return None
    status = _safe_str(payload.get("status", ""))
    has_metrics = metrics and all(_metric_value(metrics, metric) is not None for metric in ["Macro-F1", "AUROC", "AUPRC"])
    if not status:
        status = "ok" if has_metrics else ("missing" if skip else "unknown")
    if status == "skipped":
        status = "missing"
    row = {
        "suite": suite or payload.get("suite", "main"),
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "status": status,
        "skip_reason": str(payload.get("skip_reason", payload.get("reason", ""))),
        "run_dir": str(run_dir),
        "metrics_file": str(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else "",
    }
    for metric in ["Macro-F1", "AUROC", "AUPRC"]:
        row[metric] = _metric_value(metrics, metric) if metrics else None
    return row


def _infer_identity(run_dir: Path, raw_dir: Path, payload: dict[str, Any]) -> tuple[str, str | None, int | None, str]:
    dataset = _safe_str(payload.get("dataset", ""))
    model = payload.get("model", payload.get("variant"))
    seed = payload.get("seed")
    suite = _safe_str(payload.get("suite", ""))
    if dataset and not _is_missing_scalar(model) and seed is not None:
        return dataset, _safe_str(model), _safe_int(seed), suite or "main"
    try:
        parts = run_dir.relative_to(raw_dir).parts
    except ValueError:
        parts = run_dir.parts
    if parts and parts[0] in {"robustness", "labeler_comparison", "faithfulness", "cost"}:
        suite = parts[0]
        parts = parts[1:]
    if len(parts) >= 3:
        dataset = dataset or parts[0]
        model = _safe_str(model) if not _is_missing_scalar(model) else str(parts[1])
        seed = _safe_int(seed if seed is not None else str(parts[2]).replace("seed_", ""))
    return dataset, _safe_str(model) if not _is_missing_scalar(model) else None, _safe_int(seed), suite or "main"


def _metric_value(payload: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES[metric]:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def mean_std_table(
    frame: pd.DataFrame,
    suite: str | set[str],
    model_label: str = "model",
    datasets: set[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_mean_std_table(model_label=model_label)
    if isinstance(suite, set):
        suite_mask = frame["suite"].astype(str).isin(suite)
    else:
        suite_mask = frame["suite"].astype(str) == suite
    if datasets is not None:
        suite_mask &= frame["dataset"].astype(str).isin(datasets)
    subset = frame[suite_mask & (frame["status"].astype(str).isin(["ok", "exists"]))]
    if subset.empty:
        return _append_not_applicable_rows(_empty_mean_std_table(model_label=model_label), frame[suite_mask], model_label)
    rows = []
    for (dataset, model), group in subset.groupby(["dataset", "model"], dropna=False):
        row = {
            "dataset": dataset,
            "model": _summary_model_label(model),
            "seed_count": int(group["seed"].nunique()),
        }
        if model_label != "model":
            row[model_label] = _summary_model_label(model)
        for metric in ["Macro-F1", "AUROC", "AUPRC"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else pd.NA
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size >= 2 else pd.NA
            row[f"{metric}_mean_std"] = _paper_mean_std(values)
        rows.append(row)
    table = pd.DataFrame(rows)
    table = _append_not_applicable_rows(table, frame[suite_mask], model_label)
    return _ordered_mean_std_columns(table, model_label=model_label)


def _empty_mean_std_table(model_label: str) -> pd.DataFrame:
    columns = ["dataset", "model", "seed_count"]
    if model_label != "model":
        columns.insert(1, model_label)
    for metric in ["Macro-F1", "AUROC", "AUPRC"]:
        columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"])
    return pd.DataFrame(columns=columns)


def _paper_mean_std(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "--"
    mean = float(np.mean(values) * 100.0)
    if values.size < 2:
        return f"{mean:.2f} \u00b1 NA"
    std = float(np.std(values, ddof=1) * 100.0)
    return f"{mean:.2f} \u00b1 {std:.2f}"


def _append_not_applicable_rows(table: pd.DataFrame, frame: pd.DataFrame, model_label: str) -> pd.DataFrame:
    if frame.empty or "skip_reason" not in frame:
        return _ordered_mean_std_columns(table, model_label=model_label)
    skip = frame[
        frame["skip_reason"].astype(str).str.contains("model_not_applicable_to_dataset", na=False)
    ]
    if skip.empty:
        return _ordered_mean_std_columns(table, model_label=model_label)
    rows = []
    existing = set()
    if not table.empty and {"dataset", "model"}.issubset(table.columns):
        existing = set(zip(table["dataset"].astype(str), table["model"].astype(str)))
    for (dataset, model), _group in skip.groupby(["dataset", "model"], dropna=False):
        label = _summary_model_label(model)
        key = (str(dataset), label)
        if key in existing:
            continue
        row = {"dataset": dataset, "model": model, "seed_count": 0}
        row["model"] = label
        if model_label != "model":
            row[model_label] = label
        for metric in ["Macro-F1", "AUROC", "AUPRC"]:
            row[f"{metric}_mean"] = pd.NA
            row[f"{metric}_std"] = pd.NA
            row[f"{metric}_mean_std"] = "--"
        rows.append(row)
    if rows:
        table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    return _ordered_mean_std_columns(table, model_label=model_label)


def _ordered_mean_std_columns(table: pd.DataFrame, model_label: str) -> pd.DataFrame:
    columns = list(MEAN_STD_COLUMNS)
    if model_label != "model":
        columns.insert(1, model_label)
    for column in columns:
        if column not in table:
            table[column] = pd.NA
    extras = [column for column in table.columns if column not in columns]
    return table[columns + extras]


def _summary_model_label(model: Any) -> str:
    text = _safe_str(model)
    if text in {"hero", "hero_full", "hero_gnn", "hero_official"}:
        return "HERO"
    return hero_display_name(text) if text.startswith("hero_") else text


def _advanced_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    summary_dir = output_dir / "summary"
    tables = {}
    candidates = {
        "table_llm_robustness": summary_dir / "table_llm_robustness.csv",
        "table_llm_labeler_comparison": summary_dir / "table_llm_labeler_comparison.csv",
        "table_faithfulness": summary_dir / "table_faithfulness.csv",
        "table_cost_scalability": summary_dir / "table_cost_scalability.csv",
        "table_cost": summary_dir / "table_cost.csv",
    }
    for stem, path in candidates.items():
        if not path.exists():
            continue
        try:
            tables[stem] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            tables[stem] = pd.DataFrame()
    if "table_cost" not in tables and "table_cost_scalability" in tables:
        tables["table_cost"] = tables["table_cost_scalability"].copy()
    return tables


def missing_runs(output_dir: str | Path, records: pd.DataFrame | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    records = load_run_records(output_dir) if records is None else records
    expected = expected_runs_from_config(output_dir)
    rows = []
    observed = set()
    if not records.empty:
        for row in records.itertuples(index=False):
            seed = _safe_int(getattr(row, "seed", None))
            if seed is None:
                continue
            observed.add((str(row.suite), str(row.dataset), _canonical_model_key(row.model), seed))
            if str(row.status) not in {"ok", "exists"}:
                rows.append(
                    {
                        "suite": row.suite,
                        "dataset": row.dataset,
                        "model": row.model,
                        "seed": seed,
                        "status": row.status,
                        "reason": getattr(row, "skip_reason", ""),
                    }
                )
    for item in expected:
        seed = _safe_int(item.get("seed"))
        if seed is None:
            continue
        key = (str(item["suite"]), str(item["dataset"]), _canonical_model_key(item["model"]), seed)
        if key not in observed:
            rows.append({**item, "seed": seed, "status": "missing", "reason": "raw_result_absent"})
    return pd.DataFrame(rows, columns=["suite", "dataset", "model", "seed", "status", "reason"])


def expected_runs_from_config(output_dir: str | Path) -> list[dict[str, Any]]:
    config = _read_json(Path(output_dir) / "config.json")
    rows = config.get("expected_runs", [])
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        try:
            normalized.append(
                {
                    "suite": str(row["suite"]),
                    "dataset": str(row["dataset"]),
                    "model": str(row["model"]),
                    "seed": int(row["seed"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_int(value: Any) -> int | None:
    try:
        if _is_missing_scalar(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if _is_missing_scalar(value):
        return ""
    return str(value)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _canonical_model_key(value: Any) -> str:
    text = _safe_str(value).lower().replace("-", "_")
    if text in {"hero", "hero_full", "hero_gnn", "hero_official"}:
        return "hero_full"
    return text


if __name__ == "__main__":
    main()
