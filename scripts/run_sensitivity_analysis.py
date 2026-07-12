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

from scripts.advanced_experiment_utils import METRICS, metric_summary, write_frame  # noqa: E402
from scripts.paper_artifact_utils import import_matplotlib, save_figure, write_latex  # noqa: E402
from src.training.submission import _submission_metric_payload, processed_ready, resolve_processed_dir  # noqa: E402
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


DEFAULT_DATASETS = ["yelp_academic", "amazon_video"]
DEFAULT_SEEDS = [0, 1, 2]
K_VALUES = [3, 5, 10, 15, 20]
LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]
THRESHOLD_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HERO hyperparameter sensitivity experiments.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--k_values", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--lambda_rel_values", nargs="+", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--lambda_chain_values", nargs="+", type=float, default=LAMBDA_VALUES)
    parser.add_argument("--confidence_thresholds", nargs="+", type=float, default=THRESHOLD_VALUES)
    parser.add_argument("--output_dir", default="outputs/submission_sensitivity")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sensitivity(args)


def run_sensitivity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.quick_test:
        args.datasets = ["yelp_academic"]
        args.seeds = [0]
        args.k_values = [3]
        args.lambda_rel_values = [0.0]
        args.lambda_chain_values = [0.0]
        args.confidence_thresholds = [0.3]
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        specs = _sensitivity_specs(args)
        for seed in args.seeds:
            for spec in specs:
                rows.append(_run_case(args, output_dir, data_dir, dataset, int(seed), spec))
    raw = pd.DataFrame(rows)
    summary = summarize_sensitivity(raw)
    curve = sensitivity_curve_data(summary)
    heatmap = sensitivity_heatmap_data(summary)
    write_frame(output_dir / "summary" / "sensitivity_raw.csv", raw)
    write_frame(output_dir / "summary" / "table_sensitivity.csv", summary)
    write_latex(output_dir / "summary" / "table_sensitivity.tex", summary)
    write_frame(output_dir / "tables" / "table_sensitivity.csv", summary)
    write_latex(output_dir / "tables" / "table_sensitivity.tex", summary)
    write_frame(output_dir / "figure_data" / "sensitivity_curve_data.csv", curve)
    write_frame(output_dir / "figure_data" / "sensitivity_heatmap_data.csv", heatmap)
    plot_sensitivity(output_dir)
    return rows


def summarize_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    summary = metric_summary(raw, ["dataset", "parameter", "param_value", "lambda_rel", "lambda_chain", "confidence_threshold", "candidate_neighbor_k"])
    if summary.empty:
        return _unavailable_summary(raw)
    meta_cols = ["status", "skip_reason", "supported_config_key"]
    meta = []
    for keys, group in raw.groupby(["dataset", "parameter", "param_value", "lambda_rel", "lambda_chain", "confidence_threshold", "candidate_neighbor_k"], dropna=False):
        row = {name: value for name, value in zip(["dataset", "parameter", "param_value", "lambda_rel", "lambda_chain", "confidence_threshold", "candidate_neighbor_k"], keys)}
        for col in meta_cols:
            row[col] = group[col].dropna().iloc[0] if col in group and not group[col].dropna().empty else ""
        meta.append(row)
    return pd.DataFrame(meta).merge(summary, on=["dataset", "parameter", "param_value", "lambda_rel", "lambda_chain", "confidence_threshold", "candidate_neighbor_k"], how="left")


def sensitivity_curve_data(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    curve = summary[summary["parameter"].astype(str).isin(["candidate_neighbor_k", "lambda_rel", "confidence_threshold"])].copy()
    keep = ["dataset", "parameter", "param_value", "seed_count", "AUPRC_mean", "AUPRC_std", "AUPRC_mean_std", "status", "skip_reason"]
    for col in keep:
        if col not in curve:
            curve[col] = pd.NA
    return curve[keep]


def sensitivity_heatmap_data(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    heatmap = summary[summary["parameter"].astype(str) == "lambda_grid"].copy()
    keep = ["dataset", "lambda_rel", "lambda_chain", "seed_count", "AUPRC_mean", "AUPRC_std", "AUPRC_mean_std", "status", "skip_reason"]
    for col in keep:
        if col not in heatmap:
            heatmap[col] = pd.NA
    return heatmap[keep]


def plot_sensitivity(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    curve = _read_csv(output_dir / "figure_data" / "sensitivity_curve_data.csv")
    heatmap = _read_csv(output_dir / "figure_data" / "sensitivity_heatmap_data.csv")
    _plot_curve(curve, "candidate_neighbor_k", "Candidate neighbor K", output_dir / "figures" / "fig_sensitivity_k.pdf")
    _plot_curve(curve, "lambda_rel", "lambda_rel", output_dir / "figures" / "fig_sensitivity_lambda_rel.pdf")
    _plot_heatmap(heatmap, output_dir / "figures" / "fig_sensitivity_lambda_heatmap.pdf")


def _run_case(args: argparse.Namespace, output_dir: Path, data_dir: Path, dataset: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = output_dir / "raw" / dataset / str(spec["parameter"]) / str(spec["tag"]) / f"seed_{seed}"
    if args.skip_existing and (run_dir / "metrics.json").exists():
        try:
            return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    run_dir.mkdir(parents=True, exist_ok=True)
    if not processed_ready(data_dir):
        row = _status_row(dataset, seed, spec, "unavailable", "processed_data_missing")
        _write_artifacts(run_dir, row, args)
        return row
    if not spec.get("supported", True):
        row = _status_row(dataset, seed, spec, "unsupported", str(spec.get("reason", "unsupported_parameter")))
        _write_artifacts(run_dir, row, args)
        return row
    start = time.perf_counter()
    try:
        hero_config = _resolve_hero_config("hero_gnn", dict(spec["hero_config"]))
        metrics = train_single_experiment(
            dataset=dataset,
            model_name="hero_gnn",
            seed=seed,
            data_dir=data_dir,
            output_root=output_dir / "raw" / "_project_runs_sensitivity",
            epochs=int(args.epochs),
            lr=float(args.lr),
            hidden_dim=int(args.hidden_dim),
            top_k=int(spec.get("top_k", hero_config.get("neighbor_budget", 5))),
            device=args.device,
            hero_config=hero_config,
        )
        payload = _submission_metric_payload(metrics, dataset, "hero_gnn", seed, "project")
        row = {
            **_status_row(dataset, seed, spec, "ok", ""),
            **payload,
            "runtime_seconds": float(time.perf_counter() - start),
        }
        pred = metrics.get("predictions_file")
        if pred and Path(str(pred)).exists():
            shutil.copyfile(str(pred), run_dir / "predictions.npy")
    except Exception as exc:
        row = _status_row(dataset, seed, spec, "missing", f"experiment_failed: {type(exc).__name__}: {exc}")
    _write_artifacts(run_dir, row, args)
    print(f"[{row['status']}] sensitivity dataset={dataset} parameter={spec['parameter']} value={spec['param_value']} seed={seed}")
    return row


def _sensitivity_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for value in args.k_values:
        specs.append(_spec("candidate_neighbor_k", value, {"neighbor_budget": int(value)}, top_k=int(value), key="neighbor_budget"))
    for value in args.lambda_rel_values:
        specs.append(_spec("lambda_rel", value, {"mechanism_loss_weight": float(value)}, key="mechanism_loss_weight"))
    for value in args.confidence_thresholds:
        specs.append(_spec("confidence_threshold", value, {"risk_relevance_threshold": float(value)}, key="risk_relevance_threshold"))
    for rel in args.lambda_rel_values:
        for chain in args.lambda_chain_values:
            specs.append(
                _spec(
                    "lambda_grid",
                    f"rel_{rel}_chain_{chain}",
                    {"mechanism_loss_weight": float(rel), "chain_loss_weight": float(chain)},
                    key="mechanism_loss_weight,chain_loss_weight",
                    lambda_rel=float(rel),
                    lambda_chain=float(chain),
                )
            )
    return specs


def _spec(parameter: str, value: Any, hero_config: dict[str, Any], key: str, top_k: int | None = None, lambda_rel: float | None = None, lambda_chain: float | None = None) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "param_value": value,
        "tag": str(value).replace(".", "p"),
        "hero_config": hero_config,
        "supported": True,
        "supported_config_key": key,
        "top_k": top_k,
        "lambda_rel": lambda_rel if lambda_rel is not None else (float(value) if parameter == "lambda_rel" else pd.NA),
        "lambda_chain": lambda_chain if lambda_chain is not None else pd.NA,
        "confidence_threshold": float(value) if parameter == "confidence_threshold" else pd.NA,
        "candidate_neighbor_k": int(value) if parameter == "candidate_neighbor_k" else pd.NA,
    }


def _status_row(dataset: str, seed: int, spec: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "suite": "sensitivity",
        "dataset": dataset,
        "model": "hero_gnn",
        "seed": int(seed),
        "parameter": spec["parameter"],
        "param_value": spec["param_value"],
        "lambda_rel": spec.get("lambda_rel", pd.NA),
        "lambda_chain": spec.get("lambda_chain", pd.NA),
        "confidence_threshold": spec.get("confidence_threshold", pd.NA),
        "candidate_neighbor_k": spec.get("candidate_neighbor_k", pd.NA),
        "supported_config_key": spec.get("supported_config_key", ""),
        "status": status,
        "skip_reason": reason,
    }


def _write_artifacts(run_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> None:
    write_json(run_dir / "metrics.json", _json_safe(row))
    write_json(
        run_dir / "config.json",
        {
            "suite": "sensitivity",
            "dataset": row["dataset"],
            "model": "hero_gnn",
            "seed": int(row["seed"]),
            "parameter": row["parameter"],
            "param_value": row["param_value"],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "device": args.device,
        },
    )
    write_json(run_dir / "runtime.json", {"status": row["status"], "reason": row.get("skip_reason", "")})
    (run_dir / "log.txt").write_text(str(row.get("skip_reason", "")) + "\n", encoding="utf-8")


def _unavailable_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "parameter", "param_value", "lambda_rel", "lambda_chain", "confidence_threshold", "candidate_neighbor_k"]
    for keys, group in raw.groupby(group_cols, dropna=False):
        row = {name: value for name, value in zip(group_cols, keys)}
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        row["status"] = group["status"].dropna().iloc[0] if "status" in group and not group["status"].dropna().empty else "unavailable"
        row["skip_reason"] = group["skip_reason"].dropna().iloc[0] if "skip_reason" in group and not group["skip_reason"].dropna().empty else ""
        row["supported_config_key"] = group["supported_config_key"].dropna().iloc[0] if "supported_config_key" in group and not group["supported_config_key"].dropna().empty else ""
        for metric in METRICS:
            row[f"{metric}_mean"] = pd.NA
            row[f"{metric}_std"] = pd.NA
            row[f"{metric}_mean_std"] = "--"
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curve(curve: pd.DataFrame, parameter: str, xlabel: str, pdf_path: Path) -> None:
    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    subset = curve[curve["parameter"].astype(str) == parameter].copy() if not curve.empty and "parameter" in curve else pd.DataFrame()
    if subset.empty or "AUPRC_mean" not in subset:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        subset["AUPRC_mean"] = pd.to_numeric(subset["AUPRC_mean"], errors="coerce")
        subset["param_value"] = pd.to_numeric(subset["param_value"], errors="coerce")
        plotted = False
        for dataset, group in subset.groupby("dataset", dropna=False):
            series = group.groupby("param_value")["AUPRC_mean"].mean().dropna().sort_index()
            if series.empty:
                continue
            ax.plot(series.index.to_numpy(dtype=float), series.to_numpy(dtype=float), marker="o", label=str(dataset))
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel("AUPRC")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, pdf_path, pdf_path.with_suffix(".png"))
    plt.close(fig)


def _plot_heatmap(heatmap: pd.DataFrame, pdf_path: Path) -> None:
    plt = import_matplotlib()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    if heatmap.empty or "AUPRC_mean" not in heatmap:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = heatmap.copy()
        data["lambda_rel"] = pd.to_numeric(data["lambda_rel"], errors="coerce")
        data["lambda_chain"] = pd.to_numeric(data["lambda_chain"], errors="coerce")
        data["AUPRC_mean"] = pd.to_numeric(data["AUPRC_mean"], errors="coerce")
        pivot = data.pivot_table(index="lambda_chain", columns="lambda_rel", values="AUPRC_mean", aggfunc="mean")
        if pivot.empty:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
            ax.set_axis_off()
        else:
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(value) for value in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([str(value) for value in pivot.index])
            ax.set_xlabel("lambda_rel")
            ax.set_ylabel("lambda_chain")
            fig.colorbar(image, ax=ax, label="AUPRC")
    fig.tight_layout()
    save_figure(fig, pdf_path, pdf_path.with_suffix(".png"))
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif value is pd.NA:
            out[key] = None
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
