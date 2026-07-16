from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_main_experiment_cost import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    DEFAULT_SUITES,
    NA,
    build_compact_table,
    build_detail_table,
    _copy_to_final_artifacts,
    _dedupe_texts,
    _ensure_dirs,
    _requested_models,
    _write_csv,
    _write_latex,
    _write_markdown,
    _write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format existing main-cost raw runs into paper tables.")
    parser.add_argument("--input_dir", default="outputs/main_cost", help="Directory containing raw/main_cost_raw_runs.csv.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to --input_dir.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--suite_names", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--write_latex", action="store_true")
    parser.add_argument("--write_markdown", action="store_true")
    parser.add_argument("--copy_to_final_artifacts", action="store_true")
    parser.add_argument("--final_artifacts_dir", default="outputs/final_artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = format_main_cost_table(args)
    print(f"Detailed cost table: {paths['detail_csv']}")
    print(f"Compact cost table: {paths['compact_csv']}")


def format_main_cost_table(args: argparse.Namespace) -> dict[str, Path]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir or args.input_dir)
    dirs = _ensure_dirs(output_dir)
    raw_path = input_dir / "raw" / "main_cost_raw_runs.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, keep_default_na=False)
    else:
        raw = pd.DataFrame()
    records = raw.to_dict(orient="records")
    for record in records:
        if isinstance(record.get("source_files"), str):
            record["source_files"] = [part for part in record["source_files"].split(";") if part]
    models = _dedupe_texts(_requested_models(args.models).values())
    detail = build_detail_table(records, datasets=args.datasets, suite_names=args.suite_names, models=models)
    compact = build_compact_table(detail, datasets=args.datasets)
    paths: dict[str, Path] = {}
    paths["raw_csv"] = raw_path
    paths["detail_csv"] = _write_csv(dirs["tables_csv"] / "supp_table_main_experiment_cost.csv", detail)
    paths["compact_csv"] = _write_csv(dirs["tables_csv"] / "supp_table_main_experiment_cost_compact.csv", compact)
    if bool(args.write_latex):
        paths["detail_latex"] = _write_latex(
            dirs["tables_latex"] / "supp_table_main_experiment_cost.tex",
            detail,
            caption="Detailed runtime and annotation cost of main experiments.",
            label="tab:main_cost_detail",
            compact=False,
        )
        paths["compact_latex"] = _write_latex(
            dirs["tables_latex"] / "supp_table_main_experiment_cost_compact.tex",
            compact,
            caption="Computational cost of main experiments.",
            label="tab:main_cost_compact",
            compact=True,
        )
    if bool(args.write_markdown):
        paths["detail_markdown"] = _write_markdown(dirs["tables_markdown"] / "supp_table_main_experiment_cost.md", detail)
        paths["compact_markdown"] = _write_markdown(dirs["tables_markdown"] / "supp_table_main_experiment_cost_compact.md", compact)
    paths["report"] = _write_report(
        dirs["reports"] / "MAIN_EXPERIMENT_COST_REPORT.md",
        source_outputs=[input_dir],
        source_warnings=[] if raw_path.exists() else [f"missing_raw_runs: {raw_path}"],
        suite_names=args.suite_names,
        datasets=args.datasets,
        models=models,
        detail=detail,
        paths=paths,
    )
    if bool(args.copy_to_final_artifacts):
        paths["final_artifacts_dir"] = _copy_to_final_artifacts(paths, Path(args.final_artifacts_dir))
    if raw.empty and not raw_path.exists():
        (dirs["raw"] / "main_cost_raw_runs.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "runtime_unavailable", "reason": NA}]).to_csv(dirs["raw"] / "main_cost_raw_runs.csv", index=False)
    return paths


if __name__ == "__main__":
    main()
