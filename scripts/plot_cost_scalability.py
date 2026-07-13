from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import import_matplotlib, read_csv_or_empty, save_figure, write_csv  # noqa: E402


PLOTS = [
    ("annotation_time_seconds", "Annotation time (s)", "fig_cost_annotation_time"),
    ("train_time_seconds_per_seed", "Training time / seed (s)", "fig_cost_training_time"),
    ("cache_size_mb", "Cache size (MB)", "fig_cost_cache_size"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cost and scalability figures from collected HERO cost CSV.")
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--table", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_cost_scalability(args.output_dir, table_path=args.table)


def plot_cost_scalability(output_dir: str | Path, table_path: str | Path | None = None) -> None:
    output_dir = Path(output_dir)
    table = read_csv_or_empty(table_path or output_dir / "summary" / "table_cost_scalability.csv")
    if table.empty:
        table = pd.DataFrame([{"status": "unavailable", "reason": "table_cost_scalability.csv missing or empty"}])
    write_csv(output_dir / "figure_data" / "cost_scalability_data.csv", table)
    for column, ylabel, stem in PLOTS:
        _plot_bar(table, column, ylabel, output_dir / "figures" / f"{stem}.pdf", output_dir / "figures" / f"{stem}.png")
    report = output_dir / "reports" / "plot_cost_scalability_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"# plot_cost_scalability\n\n- rows: {len(table)}\n- source: table_cost_scalability.csv\n", encoding="utf-8")


def _plot_bar(table: pd.DataFrame, column: str, ylabel: str, pdf_path: Path, png_path: Path) -> None:
    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    if table.empty or column not in table:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = table.copy()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=[column])
        if "status" in data:
            data = data[data["status"].astype(str).isin(["ok", "exists", "missing", "unavailable"])]
        if data.empty:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
            ax.set_axis_off()
        else:
            labels = [f"{row.dataset}\n{row.model}" for row in data.itertuples(index=False)]
            x = np.arange(len(data))
            ax.bar(x, data[column].to_numpy(dtype=float), color="#59A14F")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, pdf_path, png_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
