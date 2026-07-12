from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import ensure_artifact_dirs, import_matplotlib, read_csv_or_empty, save_figure, write_csv  # noqa: E402


DATASETS = ["yelp_academic", "amazon_video"]
HERO_MODELS = {"hero", "hero_gnn", "hero_official"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 5-seed AUPRC stability for HERO and strongest baselines.")
    parser.add_argument("--input_csv", default="outputs/submission_main_20260711_225137/summary/all_raw_runs.csv")
    parser.add_argument("--output_dir", default="outputs/final_artifacts")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_seed_stability(args.input_csv, args.output_dir, datasets=args.datasets)


def plot_seed_stability(input_csv: str | Path, output_dir: str | Path, datasets: list[str] | None = None) -> pd.DataFrame:
    dirs = ensure_artifact_dirs(output_dir)
    raw = read_csv_or_empty(input_csv)
    data = seed_stability_data(raw, datasets=datasets or DATASETS)
    write_csv(dirs["figure_data"] / "seed_stability_data.csv", data)
    _plot(data, dirs["figures_pdf"] / "fig_seed_stability_auprc.pdf", dirs["figures_png"] / "fig_seed_stability_auprc.png")
    report = dirs["reports"] / "seed_stability_report.md"
    missing = data[data["status"].astype(str) != "ok"] if "status" in data else pd.DataFrame()
    lines = ["# Seed Stability Report", "", f"- input_csv: `{input_csv}`", f"- rows: {len(data)}"]
    if not missing.empty:
        lines.append("- missing/unavailable rows are recorded in `figure_data/seed_stability_data.csv`.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def seed_stability_data(raw: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    if raw.empty or "AUPRC" not in raw:
        return pd.DataFrame([{"dataset": dataset, "model": "hero_gnn", "seed": pd.NA, "AUPRC": pd.NA, "status": "unavailable", "reason": "all_raw_runs.csv missing or lacks AUPRC"} for dataset in datasets])
    frame = raw.copy()
    status = frame["status"] if "status" in frame else pd.Series(["ok"] * len(frame), index=frame.index)
    frame = frame[status.astype(str).isin(["ok", "exists"])]
    frame["AUPRC"] = pd.to_numeric(frame["AUPRC"], errors="coerce")
    rows = []
    for dataset in datasets:
        subset = frame[frame["dataset"].astype(str) == dataset].dropna(subset=["AUPRC"])
        if subset.empty:
            rows.append({"dataset": dataset, "model": "hero_gnn", "seed": pd.NA, "AUPRC": pd.NA, "status": "unavailable", "reason": "dataset AUPRC rows missing"})
            continue
        models = _selected_models(subset)
        for model in models:
            model_rows = subset[subset["model"].astype(str) == model]
            observed = set(int(seed) for seed in pd.to_numeric(model_rows["seed"], errors="coerce").dropna().astype(int))
            for _, row in model_rows.sort_values("seed").iterrows():
                rows.append({"dataset": dataset, "model": model, "seed": int(row["seed"]), "AUPRC": float(row["AUPRC"]), "status": "ok", "reason": ""})
            missing = [seed for seed in [0, 1, 2, 3, 4] if seed not in observed]
            for seed in missing:
                rows.append({"dataset": dataset, "model": model, "seed": seed, "AUPRC": pd.NA, "status": "missing", "reason": "seed_absent"})
    return pd.DataFrame(rows)


def _selected_models(subset: pd.DataFrame) -> list[str]:
    observed = [str(model) for model in subset["model"].dropna().unique()]
    hero = "hero_gnn" if "hero_gnn" in observed else next((model for model in observed if model in HERO_MODELS or model.startswith("hero_")), "hero_gnn")
    baselines = subset[~subset["model"].astype(str).isin(HERO_MODELS) & ~subset["model"].astype(str).str.startswith("hero_")]
    ranks = baselines.groupby("model")["AUPRC"].mean().sort_values(ascending=False)
    selected = [hero, *[str(model) for model in ranks.index[:2]]]
    return list(dict.fromkeys(selected))


def _plot(data: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    plt = import_matplotlib()
    datasets = sorted(str(value) for value in data.get("dataset", pd.Series(dtype=str)).dropna().unique()) if not data.empty else ["unavailable"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3.0), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = data[(data["dataset"].astype(str) == dataset) & (data["status"].astype(str) == "ok")] if not data.empty else pd.DataFrame()
        if subset.empty:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        for model, group in subset.groupby("model", dropna=False):
            group = group.sort_values("seed")
            ax.plot(group["seed"], group["AUPRC"], marker="o", linewidth=1.8, label=str(model))
        ax.set_title(dataset)
        ax.set_xlabel("seed")
        ax.set_ylabel("AUPRC")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, pdf_path, png_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
