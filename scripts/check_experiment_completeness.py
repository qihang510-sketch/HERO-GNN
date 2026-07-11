from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment_suite import _expected_runs, _expand_suites  # noqa: E402
from scripts.summarize_experiment_suite import expected_runs_from_config, load_run_records, missing_runs  # noqa: E402
from src.training.submission import SUBMISSION_DATASETS, normalize_dataset_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check completeness of unified HERO experiment outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite", default=None, choices=["main", "ablation", "robustness", "labeler_comparison", "faithfulness", "cost", "all"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--generate_commands", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    records = load_run_records(output_dir)
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
        return missing_runs(output_dir, records)
    records = load_run_records(output_dir) if records is None else records
    observed = {}
    if not records.empty:
        for row in records.itertuples(index=False):
            observed[(str(row.suite), str(row.dataset), str(row.model), int(row.seed))] = row
    rows = []
    for item in expected:
        key = (str(item["suite"]), str(item["dataset"]), str(item["model"]), int(item["seed"]))
        row = observed.get(key)
        if row is None:
            rows.append({**item, "status": "missing", "reason": "raw_result_absent"})
        elif str(row.status) not in {"ok", "exists"}:
            rows.append({**item, "status": str(row.status), "reason": str(getattr(row, "skip_reason", ""))})
    return pd.DataFrame(rows, columns=["suite", "dataset", "model", "seed", "status", "reason"])


def _expected_from_args_or_config(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    if args.suite or args.datasets or args.models is not None or args.seeds:
        suites = _expand_suites(args.suite or "main")
        datasets = [normalize_dataset_name(dataset) for dataset in (args.datasets or list(SUBMISSION_DATASETS))]
        seeds = args.seeds or [0, 1, 2, 3, 4]
        return _expected_runs(suites=suites, datasets=datasets, models=args.models, seeds=seeds)
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
        seeds = " ".join(str(int(seed)) for seed in sorted(group["seed"].astype(int).unique()))
        command = (
            f"python scripts/run_experiment_suite.py --suite {suite} "
            f"--datasets {dataset} --models {model} --seeds {seeds} "
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


if __name__ == "__main__":
    main()
