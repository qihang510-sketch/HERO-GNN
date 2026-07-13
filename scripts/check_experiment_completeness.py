from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment_suite import DEFAULT_SEEDS, _expected_runs, _expand_suites  # noqa: E402
from scripts.summarize_experiment_suite import expected_runs_from_config, load_run_records, missing_runs  # noqa: E402
from src.training.submission import SUBMISSION_DATASETS, normalize_dataset_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check completeness of unified HERO experiment outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite", default=None, choices=["main", "transfer", "ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost", "all"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--generate_commands", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records = load_known_run_records(output_dir)
    expected = _expected_from_args_or_config(args, output_dir)
    if expected:
        _write_expected_config_shadow(output_dir, expected)
    table = completeness_table(output_dir, records=records, expected=expected)
    out = output_dir / "summary" / "missing_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote missing run table to {out}")
    _print_missing_commands(table, output_dir, write_file=args.generate_commands)


def completeness_table(
    output_dir: str | Path,
    records: pd.DataFrame | None = None,
    expected: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if expected is None:
        return missing_runs(output_dir, load_known_run_records(output_dir, records))
    records = load_known_run_records(output_dir, records)
    observed = {}
    if not records.empty:
        for row in records.itertuples(index=False):
            seed = _seed_value(getattr(row, "seed", None))
            if seed < 0:
                continue
            key = (str(row.suite), str(row.dataset), _canonical_model_key(getattr(row, "model", "")), seed)
            previous = observed.get(key)
            if previous is None or _status_rank(getattr(row, "status", "")) > _status_rank(getattr(previous, "status", "")):
                observed[key] = row
    rows = []
    for item in expected:
        seed = _seed_value(item.get("seed"))
        key = (str(item["suite"]), str(item["dataset"]), _canonical_model_key(item["model"]), seed)
        row = observed.get(key)
        if row is None:
            rows.append({**item, "seed": seed, "status": "missing", "reason": "raw_result_absent"})
        elif str(row.status) not in {"ok", "exists"}:
            rows.append({**item, "seed": seed, "status": str(row.status), "reason": _row_reason(row)})
    return pd.DataFrame(rows, columns=["suite", "dataset", "model", "seed", "status", "reason"])


def load_known_run_records(output_dir: str | Path, records: pd.DataFrame | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    frames: list[pd.DataFrame] = []
    if records is not None and not records.empty:
        frames.append(records)
    scanned = load_run_records(output_dir)
    if not scanned.empty:
        frames.append(scanned)
    for path in [output_dir / "summary" / "all_raw_runs.csv", output_dir / "tables" / "all_raw_runs.csv"]:
        if path.exists():
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if not frame.empty:
                frames.append(frame)
    for path in [output_dir / "run_manifest.json", output_dir / "summary" / "run_manifest.json"]:
        frame = _records_from_manifest(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["suite", "dataset", "model", "seed", "status", "skip_reason"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in ["suite", "dataset", "model", "seed", "status", "skip_reason", "reason"]:
        if column not in combined:
            combined[column] = ""
    combined["_canonical_model"] = combined["model"].map(_canonical_model_key)
    combined["_seed_key"] = combined["seed"].map(_seed_value)
    combined["_status_rank"] = combined["status"].map(_status_rank)
    combined = combined.sort_values("_status_rank", ascending=False)
    combined = combined.drop_duplicates(subset=["suite", "dataset", "_canonical_model", "_seed_key"], keep="first")
    return combined.drop(columns=["_canonical_model", "_seed_key", "_status_rank"], errors="ignore")


def _records_from_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()
    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "skip_reason" not in frame and "reason" in frame:
        frame["skip_reason"] = frame["reason"]
    return frame


def _expected_from_args_or_config(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    if args.suite or args.datasets or args.models is not None or args.seeds:
        suites = _expand_suites(args.suite or "main")
        suite_args = argparse.Namespace(
            datasets=[normalize_dataset_name(dataset) for dataset in args.datasets] if args.datasets else None,
            models=args.models,
            variants=None,
            seeds=args.seeds or list(DEFAULT_SEEDS),
        )
        return _expected_runs(suites=suites, args=suite_args)
    return expected_runs_from_config(output_dir)


def _write_expected_config_shadow(output_dir: Path, expected: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "completeness_expected_runs.csv"
    pd.DataFrame(expected).to_csv(config_path, index=False)


def _print_missing_commands(table: pd.DataFrame, output_dir: Path, write_file: bool) -> None:
    if table.empty:
        print("No missing runs detected.")
        return
    commands = []
    grouped = table.groupby(["suite", "dataset", "model"], dropna=False)
    for (suite, dataset, model), group in grouped:
        seed_values = sorted(_seed_value(seed) for seed in group["seed"].unique())
        seed_values = [seed for seed in seed_values if seed >= 0]
        seed_arg = f"--seeds {' '.join(str(seed) for seed in seed_values)} " if seed_values else ""
        command = (
            f"python scripts/run_experiment_suite.py --suite {suite} "
            f"--datasets {dataset} --models {model} {seed_arg}"
            f"--output_dir {output_dir} --skip_existing"
        )
        commands.append(command)
    print("Missing run commands:")
    for command in commands:
        print(command)
    if write_file:
        path = output_dir / "summary" / "missing_commands.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(commands) + "\n", encoding="utf-8")
        print(f"Wrote rerun commands to {path}")


def _seed_value(value: Any) -> int:
    try:
        if value is None:
            return -1
        if not isinstance(value, (list, dict, tuple, pd.Series, pd.DataFrame)) and bool(pd.isna(value)):
            return -1
        if value == "":
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1


def _canonical_model_key(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().replace("-", "_")
    if text in {"hero", "hero_full", "hero_gnn", "hero_official"}:
        return "hero_full"
    return text


def _status_rank(status: Any) -> int:
    text = "" if status is None else str(status)
    return {"ok": 4, "exists": 4, "skipped": 3, "unavailable": 2, "missing": 1, "failed": 0}.get(text, 1)


def _row_reason(row: Any) -> str:
    for attr in ["skip_reason", "reason"]:
        value = getattr(row, attr, "")
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value)
        if text:
            return text
    return ""


if __name__ == "__main__":
    main()
