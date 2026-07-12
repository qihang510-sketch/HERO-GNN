from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.advanced_experiment_utils import write_frame  # noqa: E402
from scripts.paper_artifact_utils import write_latex  # noqa: E402
from src.training.submission import SUBMISSION_DATASETS, resolve_processed_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect HERO annotation cost and scalability metadata from real logs/caches.")
    parser.add_argument("--input_dir", default=None, help="Experiment output root. Defaults to --output_dir.")
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--datasets", nargs="+", default=list(SUBMISSION_DATASETS))
    parser.add_argument("--models", nargs="+", default=["hero_gnn", "hero_official"])
    parser.add_argument("--data_root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_cost_scalability(args)


def collect_cost_scalability(args: argparse.Namespace) -> pd.DataFrame:
    input_dir = Path(args.input_dir or args.output_dir)
    output_dir = Path(args.output_dir)
    rows = []
    for dataset in args.datasets:
        processed_dir = resolve_processed_dir(dataset, args.data_root)
        metrics_rows = _read_metric_rows(input_dir, dataset, set(args.models))
        annotation_files = _annotation_files(processed_dir, input_dir, dataset)
        candidate_cards = _candidate_card_count(processed_dir, input_dir, dataset)
        annotated_cards = sum(_count_jsonl(path) for path in annotation_files)
        cache_size_mb = _cache_size_mb([processed_dir, input_dir], dataset)
        token_stats = _token_stats(input_dir, dataset)
        for model, group in _group_metrics(metrics_rows).items():
            seeds = sorted({int(row.get("seed")) for row in group if _has_int(row.get("seed"))})
            row = {
                "dataset": dataset,
                "candidate_cards": int(candidate_cards),
                "annotated_cards": int(annotated_cards),
                "annotation_coverage": float(annotated_cards / candidate_cards) if candidate_cards else pd.NA,
                "avg_tokens_per_card": token_stats.get("avg_tokens_per_card", pd.NA),
                "total_tokens": token_stats.get("total_tokens", pd.NA),
                "annotation_time_seconds": _mean(group, ["annotation_time_seconds", "time_mock_labeling_sec"]),
                "cache_size_mb": float(cache_size_mb),
                "train_time_seconds_per_seed": _mean(group, ["Training time", "time_training_sec"]),
                "peak_gpu_memory_mb": _mean(group, ["peak_gpu_memory_mb", "max_gpu_memory_mb"]),
                "device": _first_nonempty(group, "device"),
                "model": model,
                "seeds": " ".join(str(seed) for seed in seeds),
                "seed_count": len(seeds),
                "status": "ok" if group else "missing",
                "source": str(input_dir),
            }
            rows.append(row)
        if not metrics_rows:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_cards": int(candidate_cards),
                    "annotated_cards": int(annotated_cards),
                    "annotation_coverage": float(annotated_cards / candidate_cards) if candidate_cards else pd.NA,
                    "avg_tokens_per_card": token_stats.get("avg_tokens_per_card", pd.NA),
                    "total_tokens": token_stats.get("total_tokens", pd.NA),
                    "annotation_time_seconds": pd.NA,
                    "cache_size_mb": float(cache_size_mb),
                    "train_time_seconds_per_seed": pd.NA,
                    "peak_gpu_memory_mb": pd.NA,
                    "device": "",
                    "model": "hero",
                    "seeds": "",
                    "seed_count": 0,
                    "status": "missing",
                    "source": str(input_dir),
                }
            )
    table = pd.DataFrame(rows)
    table = _canonical_cost_table(table)
    write_frame(output_dir / "summary" / "table_cost_scalability.csv", table)
    write_frame(output_dir / "summary" / "table_cost.csv", table)
    write_latex(output_dir / "summary" / "table_cost_scalability.tex", table)
    write_frame(output_dir / "tables" / "table_cost_scalability.csv", table)
    write_frame(output_dir / "tables" / "table_cost.csv", table)
    write_latex(output_dir / "tables" / "table_cost_scalability.tex", table)
    try:
        from scripts.plot_cost_scalability import plot_cost_scalability

        plot_cost_scalability(output_dir)
    except Exception as exc:
        report = output_dir / "summary" / "cost_plot_status.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"cost plots unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")
    return table


def _canonical_cost_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "model",
        "candidate_cards",
        "annotated_cards",
        "annotation_coverage",
        "avg_tokens_per_card",
        "total_tokens",
        "annotation_time_seconds",
        "cache_size_mb",
        "train_time_seconds_per_seed",
        "peak_gpu_memory_mb",
        "device",
        "seed_count",
        "status",
    ]
    table = table.copy()
    for column in columns:
        if column not in table:
            table[column] = pd.NA
    extras = [column for column in table.columns if column not in columns]
    return table[columns + extras]


def _read_metric_rows(input_dir: Path, dataset: str, models: set[str]) -> list[dict[str, Any]]:
    roots = [input_dir / "raw", input_dir]
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("metrics.json")):
            if path in seen or any(part.startswith("_project_runs") for part in path.parts):
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if str(payload.get("dataset", "")) != dataset:
                continue
            model = str(payload.get("model", payload.get("suite_model", "")))
            if model not in models and not model.startswith("hero"):
                continue
            if str(payload.get("status", "ok")) not in {"ok", "exists", ""}:
                continue
            payload["_metrics_file"] = str(path)
            rows.append(payload)
    return rows


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model", row.get("suite_model", "hero")))
        groups.setdefault(model, []).append(row)
    return groups


def _annotation_files(processed_dir: Path, input_dir: Path, dataset: str) -> list[Path]:
    candidates: list[Path] = []
    if processed_dir.exists():
        candidates.extend(sorted(processed_dir.glob("*label*.jsonl")))
    for root in [input_dir / "annotations", input_dir / "raw"]:
        if root.exists():
            candidates.extend(sorted(root.rglob(f"*{dataset}*label*.jsonl")))
            candidates.extend(sorted((root / dataset).rglob("*label*.jsonl")) if (root / dataset).exists() else [])
    return _dedupe_paths(candidates)


def _candidate_card_count(processed_dir: Path, input_dir: Path, dataset: str) -> int:
    card_files: list[Path] = []
    if processed_dir.exists():
        card_files.extend(sorted(processed_dir.glob("*risk*card*.jsonl")))
    for root in [input_dir / "annotations", input_dir / "raw"]:
        if root.exists():
            card_files.extend(sorted(root.rglob(f"*{dataset}*risk*card*.jsonl")))
            card_files.extend(sorted((root / dataset).rglob("*risk*card*.jsonl")) if (root / dataset).exists() else [])
    counts = [_count_jsonl(path) for path in _dedupe_paths(card_files)]
    if counts:
        return max(counts)
    hetero = processed_dir / "hetero_candidates.pkl"
    if hetero.exists():
        try:
            with hetero.open("rb") as handle:
                payload = pickle.load(handle)
            candidates = payload.get("candidates_by_target", payload) if isinstance(payload, dict) else payload
            if isinstance(candidates, dict):
                return int(sum(len(values) for values in candidates.values()))
        except Exception:
            return 0
    return 0


def _token_stats(input_dir: Path, dataset: str) -> dict[str, Any]:
    totals = []
    avgs = []
    for path in sorted(input_dir.rglob(f"*{dataset}*report*.json")) + sorted(input_dir.rglob("*llm_label_build_report*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(payload.get("dataset", dataset)) != dataset:
            continue
        for key in ["total_tokens", "tokens_total"]:
            if key in payload and _has_float(payload[key]):
                totals.append(float(payload[key]))
        for key in ["avg_tokens_per_card", "average_tokens_per_card"]:
            if key in payload and _has_float(payload[key]):
                avgs.append(float(payload[key]))
    return {
        "total_tokens": float(np.sum(totals)) if totals else pd.NA,
        "avg_tokens_per_card": float(np.mean(avgs)) if avgs else pd.NA,
    }


def _cache_size_mb(roots: list[Path], dataset: str) -> float:
    suffixes = {".jsonl", ".json", ".pkl", ".pt", ".npy"}
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and dataset in str(path):
                total += path.stat().st_size
    return total / (1024.0 * 1024.0)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _mean(rows: list[dict[str, Any]], keys: list[str]) -> float | Any:
    values = []
    for row in rows:
        for key in keys:
            if key in row and _has_float(row[key]):
                values.append(float(row[key]))
                break
    return float(np.mean(values)) if values else pd.NA


def _first_nonempty(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key, "") or "")
        if value:
            return value
    return ""


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(path.resolve() for path in paths if path.exists()))


def _has_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _has_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
