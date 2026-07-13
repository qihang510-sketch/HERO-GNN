from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.advanced_experiment_utils import (  # noqa: E402
    METRICS,
    NOISE_RATIOS,
    annotation_stats,
    ensure_annotation_file,
    metric_summary,
    read_jsonl_labels,
    train_hero_with_labels,
    write_frame,
    write_jsonl_labels,
)
from scripts.paper_artifact_utils import import_matplotlib, save_figure, write_latex  # noqa: E402
from src.data import schema  # noqa: E402
from src.training.submission import processed_ready, resolve_processed_dir  # noqa: E402
from src.utils.io import write_json  # noqa: E402


NOISE_TYPES = ("relevance_flip", "mechanism_shuffle", "confidence_gaussian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HERO robustness to perturbed LLM annotations.")
    parser.add_argument("--datasets", nargs="+", default=["yelp_academic", "amazon_video"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--noise_type", choices=NOISE_TYPES, default=None)
    parser.add_argument("--noise_types", nargs="+", choices=NOISE_TYPES, default=None)
    parser.add_argument("--noise_ratios", nargs="+", type=float, default=NOISE_RATIOS)
    parser.add_argument("--annotation_file", default=None)
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max_cards", type=int, default=2000)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_robustness(args)


def run_robustness(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    summary_dir = output_dir / "summary"
    figures_dir = output_dir / "figures"
    raw_rows: list[dict[str, Any]] = []
    noise_types = _selected_noise_types(args)
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        for seed in args.seeds:
            annotation_file, annotation_source, base_stats = ensure_annotation_file(
                dataset=dataset,
                data_root=args.data_root,
                output_dir=output_dir,
                seed=int(seed),
                explicit=args.annotation_file,
                max_cards=int(args.max_cards),
            )
            if annotation_file is None or not processed_ready(data_dir):
                for noise_type in noise_types:
                    for ratio in args.noise_ratios:
                        raw_rows.append(_missing_row(dataset, int(seed), noise_type, ratio, annotation_source, base_stats))
                continue
            base_labels = read_jsonl_labels(annotation_file)
            for noise_type in noise_types:
                for ratio in args.noise_ratios:
                    ratio = _bounded_ratio(ratio)
                    run_dir = _run_dir(output_dir, dataset, noise_type, ratio, int(seed))
                    metrics_path = run_dir / "metrics.json"
                    if args.skip_existing and metrics_path.exists():
                        row = json.loads(metrics_path.read_text(encoding="utf-8"))
                        raw_rows.append(row)
                        print(f"[exists] {metrics_path}")
                        continue
                    run_dir.mkdir(parents=True, exist_ok=True)
                    start = time.perf_counter()
                    labels = perturb_labels(base_labels, noise_type=noise_type, noise_ratio=ratio, seed=int(seed))
                    perturbed_file = output_dir / "annotations" / "robustness" / dataset / noise_type / f"noise_{_ratio_tag(ratio)}_seed_{seed}.jsonl"
                    write_jsonl_labels(perturbed_file, labels)
                    stats = annotation_stats(
                        labels,
                        candidate_cards=int(_safe_number(base_stats.get("candidate_cards"), len(base_labels))),
                        annotation_source=annotation_source,
                        annotation_time_seconds=float(_safe_number(base_stats.get("annotation_time_seconds"), 0.0)),
                    )
                    try:
                        payload = train_hero_with_labels(
                            dataset=dataset,
                            seed=int(seed),
                            label_file=perturbed_file,
                            output_root=output_dir / "raw" / "_project_runs_robustness",
                            data_root=args.data_root,
                            epochs=int(args.epochs),
                            lr=float(args.lr),
                            hidden_dim=int(args.hidden_dim),
                            device=args.device,
                            experiment_tag=f"robustness_{noise_type}_{_ratio_tag(ratio)}",
                            labeler=f"{annotation_source}:{noise_type}",
                        )
                        status = "ok"
                        reason = ""
                    except Exception as exc:
                        payload = {}
                        status = "missing"
                        reason = f"experiment_failed: {type(exc).__name__}: {exc}"
                    elapsed = time.perf_counter() - start
                    row = {
                        "suite": "robustness",
                        "dataset": dataset,
                        "model": "hero_gnn",
                        "seed": int(seed),
                        "noise_type": noise_type,
                        "noise_ratio": ratio,
                        "annotation_source": annotation_source,
                        "annotation_file": str(annotation_file),
                        "perturbed_annotation_file": str(perturbed_file),
                        "status": status,
                        "skip_reason": reason,
                        "runtime_seconds": float(elapsed),
                        **stats,
                        **payload,
                    }
                    _write_run_artifacts(run_dir, row, args)
                    _copy_prediction(row, run_dir)
                    raw_rows.append(row)
                    print(f"[{status}] robustness dataset={dataset} noise={noise_type} ratio={ratio} seed={seed} {reason}")
    raw = pd.DataFrame(raw_rows)
    summary = summarize_robustness(raw)
    plot_data = robustness_plot_data(raw)
    write_frame(summary_dir / "robustness_raw.csv", raw)
    write_frame(summary_dir / "robustness_summary.csv", summary)
    write_frame(summary_dir / "robustness_plot_data.csv", plot_data)
    write_frame(summary_dir / "table_llm_robustness.csv", summary)
    write_frame(output_dir / "tables" / "table_llm_robustness.csv", summary)
    write_latex(summary_dir / "table_llm_robustness.tex", summary)
    write_latex(output_dir / "tables" / "table_llm_robustness.tex", summary)
    write_frame(output_dir / "figure_data" / "robustness_curve_data.csv", plot_data)
    write_frame(figures_dir / "robustness_curve_data.csv", plot_data)
    plot_robustness_figures(output_dir)
    return raw_rows


def perturb_labels(labels: list[dict[str, Any]], noise_type: str, noise_ratio: float, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ratio = _bounded_ratio(noise_ratio)
    perturbed = [dict(label) for label in labels]
    if not perturbed or ratio <= 0.0:
        return perturbed
    if noise_type == "confidence_gaussian":
        for label in perturbed:
            confidence = float(label.get("confidence", 0.0)) + float(rng.normal(0.0, ratio))
            label["confidence"] = float(np.clip(confidence, 0.0, 1.0))
        return perturbed
    count = max(1, int(round(len(perturbed) * ratio)))
    selected = rng.choice(len(perturbed), size=min(count, len(perturbed)), replace=False)
    mechanisms = list(schema.EVIDENCE_MECHANISMS)
    for index in selected:
        label = perturbed[int(index)]
        if noise_type == "relevance_flip":
            relevance = int(_safe_number(label.get("risk_relevance"), 0.0))
            label["risk_relevance"] = 0 if relevance == 1 else 1
        elif noise_type == "mechanism_shuffle":
            current = str(label.get("mechanism", "irrelevant_heterophily"))
            choices = [name for name in mechanisms if name != current] or mechanisms
            label["mechanism"] = str(rng.choice(choices))
        else:
            raise ValueError(f"Unknown noise_type={noise_type}")
    return perturbed


def summarize_robustness(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    summary = metric_summary(raw, ["dataset", "noise_type", "noise_ratio", "annotation_source"])
    drops = _mean_metric_drops(raw)
    if summary.empty:
        return drops
    if drops.empty:
        return summary
    return summary.merge(drops, on=["dataset", "noise_type", "noise_ratio", "annotation_source"], how="left")


def robustness_plot_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows = _with_metric_drops(raw)
    keep = [
        "dataset",
        "seed",
        "noise_type",
        "noise_ratio",
        "annotation_source",
        "status",
        *METRICS,
        *[f"{metric}_drop" for metric in METRICS],
    ]
    for column in keep:
        if column not in rows:
            rows[column] = pd.NA
    return rows[keep]


def plot_robustness_figures(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    curve_path = output_dir / "figure_data" / "robustness_curve_data.csv"
    if not curve_path.exists():
        curve_path = output_dir / "figures" / "robustness_curve_data.csv"
    try:
        curve = pd.read_csv(curve_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        curve = pd.DataFrame()
    for metric, suffix in [("AUPRC", "auprc"), ("AUROC", "auroc")]:
        _plot_metric_curve(curve, metric, output_dir / "figures" / f"fig_llm_robustness_{suffix}.pdf", output_dir / "figures" / f"fig_llm_robustness_{suffix}.png")


def _plot_metric_curve(curve: pd.DataFrame, metric: str, pdf_path: Path, png_path: Path) -> None:
    plt = import_matplotlib()
    datasets = sorted(str(value) for value in curve.get("dataset", pd.Series(dtype=str)).dropna().unique()) if not curve.empty else []
    if not datasets:
        datasets = ["unavailable"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.0), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        if curve.empty or dataset == "unavailable" or metric not in curve:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        status = curve["status"] if "status" in curve else pd.Series(["ok"] * len(curve), index=curve.index)
        subset = curve[(curve["dataset"].astype(str) == dataset) & (status.astype(str).isin(["ok", "exists"]))]
        subset = subset.copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset["noise_ratio"] = pd.to_numeric(subset["noise_ratio"], errors="coerce")
        plotted = False
        for noise_type, group in subset.groupby("noise_type", dropna=False):
            series = group.groupby("noise_ratio", dropna=False)[metric].mean().dropna().sort_index()
            if series.empty:
                continue
            ax.plot(series.index.to_numpy(dtype=float), series.to_numpy(dtype=float), marker="o", linewidth=1.8, label=str(noise_type))
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_title(dataset)
        ax.set_xlabel("Noise ratio")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, pdf_path, png_path)
    plt.close(fig)


def _with_metric_drops(raw: pd.DataFrame) -> pd.DataFrame:
    rows = raw.copy()
    for metric in METRICS:
        rows[f"{metric}_drop"] = pd.NA
    if raw.empty:
        return rows
    for (dataset, seed, noise_type, source), group in rows.groupby(["dataset", "seed", "noise_type", "annotation_source"], dropna=False):
        noise_ratio = pd.to_numeric(group["noise_ratio"], errors="coerce")
        base = group[noise_ratio == 0.0]
        if base.empty:
            continue
        base_row = base.iloc[0]
        mask = (
            (rows["dataset"] == dataset)
            & (rows["seed"] == seed)
            & (rows["noise_type"] == noise_type)
            & (rows["annotation_source"] == source)
        )
        for metric in METRICS:
            base_value = _safe_float(base_row.get(metric))
            rows.loc[mask, f"{metric}_drop"] = base_value - pd.to_numeric(rows.loc[mask, metric], errors="coerce")
    return rows


def _mean_metric_drops(raw: pd.DataFrame) -> pd.DataFrame:
    rows = _with_metric_drops(raw)
    if rows.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for keys, group in rows.groupby(["dataset", "noise_type", "noise_ratio", "annotation_source"], dropna=False):
        row = {name: value for name, value in zip(["dataset", "noise_type", "noise_ratio", "annotation_source"], keys)}
        for metric in METRICS:
            values = pd.to_numeric(group[f"{metric}_drop"], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_drop_mean"] = float(np.mean(values)) if values.size else pd.NA
            row[f"{metric}_drop_std"] = float(np.std(values, ddof=1)) if values.size >= 2 else pd.NA
        out.append(row)
    return pd.DataFrame(out)


def _write_run_artifacts(run_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> None:
    write_json(run_dir / "metrics.json", _json_safe(row))
    write_json(
        run_dir / "config.json",
        {
            "suite": "robustness",
            "dataset": row["dataset"],
            "model": "hero_gnn",
            "seed": int(row["seed"]),
            "noise_type": row["noise_type"],
            "noise_ratio": float(row["noise_ratio"]),
            "annotation_source": row["annotation_source"],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "device": args.device,
        },
    )
    write_json(run_dir / "runtime.json", {"status": row["status"], "runtime_seconds": float(row.get("runtime_seconds", 0.0))})
    (run_dir / "log.txt").write_text(str(row.get("skip_reason", "")) + "\n", encoding="utf-8")


def _copy_prediction(row: dict[str, Any], run_dir: Path) -> None:
    source = row.get("predictions_file")
    if not _is_missing_scalar(source) and Path(str(source)).exists():
        shutil.copyfile(str(source), run_dir / "predictions.npy")


def _missing_row(dataset: str, seed: int, noise_type: str, ratio: float, annotation_source: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": "robustness",
        "dataset": dataset,
        "model": "hero_gnn",
        "seed": int(seed),
        "noise_type": noise_type,
        "noise_ratio": _bounded_ratio(ratio),
        "annotation_source": annotation_source,
        "status": "missing",
        "skip_reason": str(stats.get("skip_reason", "annotation_cache_and_fallback_unavailable")),
        **stats,
    }


def _selected_noise_types(args: argparse.Namespace) -> list[str]:
    if not _is_missing_scalar(getattr(args, "noise_type", None)):
        return [args.noise_type]
    noise_types = getattr(args, "noise_types", None)
    if not _is_missing_scalar(noise_types):
        if isinstance(noise_types, (list, tuple, set)):
            return list(dict.fromkeys(noise_types)) or list(NOISE_TYPES)
        return [str(noise_types)]
    return list(NOISE_TYPES)


def _run_dir(output_dir: Path, dataset: str, noise_type: str, ratio: float, seed: int) -> Path:
    return output_dir / "raw" / "robustness" / dataset / noise_type / f"noise_{_ratio_tag(ratio)}" / f"seed_{seed}"


def _ratio_tag(value: float) -> str:
    return str(int(round(_bounded_ratio(value) * 100)))


def _bounded_ratio(value: float) -> float:
    return float(min(max(float(_safe_number(value, 0.0)), 0.0), 1.0))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_number(value: Any, default: float) -> float:
    if _is_missing_scalar(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, np.ndarray, pd.Series, pd.DataFrame)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_missing_scalar(value):
            out[key] = None
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
