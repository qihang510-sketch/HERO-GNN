from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import read_csv_or_empty, write_csv  # noqa: E402
from scripts.run_sensitivity_analysis import plot_sensitivity as _plot_sensitivity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HERO sensitivity figures from sensitivity figure-data CSVs.")
    parser.add_argument("--output_dir", default="outputs/submission_sensitivity")
    parser.add_argument("--curve_csv", default=None)
    parser.add_argument("--heatmap_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_sensitivity(args.output_dir, curve_csv=args.curve_csv, heatmap_csv=args.heatmap_csv)


def plot_sensitivity(output_dir: str | Path, curve_csv: str | Path | None = None, heatmap_csv: str | Path | None = None) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    curve_target = output_dir / "figure_data" / "sensitivity_curve_data.csv"
    heatmap_target = output_dir / "figure_data" / "sensitivity_heatmap_data.csv"
    curve = read_csv_or_empty(curve_csv or curve_target)
    heatmap = read_csv_or_empty(heatmap_csv or heatmap_target)
    if curve.empty:
        curve = pd.DataFrame([{"status": "unavailable", "reason": "sensitivity_curve_data.csv missing or empty"}])
    if heatmap.empty:
        heatmap = pd.DataFrame([{"status": "unavailable", "reason": "sensitivity_heatmap_data.csv missing or empty"}])
    write_csv(curve_target, curve)
    write_csv(heatmap_target, heatmap)
    _plot_sensitivity(output_dir)
    _write_report(output_dir, curve, heatmap)
    return {"curve": curve, "heatmap": heatmap}


def _write_report(output_dir: Path, curve: pd.DataFrame, heatmap: pd.DataFrame) -> None:
    report = output_dir / "reports" / "plot_sensitivity_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# plot_sensitivity\n\n"
        f"- curve_rows: {len(curve)}\n"
        f"- heatmap_rows: {len(heatmap)}\n"
        "- source: figure_data CSV files\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
