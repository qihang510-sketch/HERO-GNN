from __future__ import annotations

import argparse
import json
import os
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
    annotation_stats,
    has_full_llm_config,
    labels_from_cards,
    load_or_build_risk_cards,
    metric_summary,
    read_jsonl_labels,
    train_hero_with_labels,
    write_frame,
    write_jsonl_labels,
)
from src.training.submission import processed_ready, resolve_processed_dir  # noqa: E402
from src.utils.io import write_json  # noqa: E402


LABELERS = ("random_labeler", "rule_based_labeler", "proxy_labeler", "cached_llm_labeler", "full_llm_labeler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HERO annotation labelers with real training runs where labels are available.")
    parser.add_argument("--datasets", nargs="+", default=["yelp_academic", "amazon_video"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--labelers", nargs="+", default=list(LABELERS), choices=list(LABELERS))
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--risk_card_file", default=None)
    parser.add_argument("--cached_llm_label_file", default=None)
    parser.add_argument("--full_llm_label_file", default=None)
    parser.add_argument("--call_full_llm", action="store_true")
    parser.add_argument("--max_cards", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_labeler_comparison(args)


def run_labeler_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        cards, risk_card_path, risk_card_status = load_or_build_risk_cards(
            dataset=dataset,
            data_root=args.data_root,
            output_dir=output_dir,
            seed=int(args.seeds[0]) if args.seeds else 0,
            risk_card_file=args.risk_card_file,
            max_cards=int(args.max_cards),
        )
        label_specs = _label_specs(dataset, data_dir, output_dir, cards, risk_card_path, risk_card_status, args)
        for spec in label_specs:
            labeler = str(spec["labeler"])
            label_file = spec.get("label_file")
            status = str(spec.get("status", "ok"))
            reason = str(spec.get("reason", ""))
            labels = read_jsonl_labels(label_file) if label_file else []
            stats = annotation_stats(
                labels,
                candidate_cards=len(cards) if cards else len(labels),
                annotation_source=str(spec.get("annotation_source", labeler)),
                annotation_time_seconds=spec.get("annotation_time_seconds"),
            )
            for seed in args.seeds:
                run_dir = output_dir / "raw" / "labeler_comparison" / dataset / labeler / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    print(f"[exists] {metrics_path}")
                    continue
                if status != "ok" or label_file is None or not processed_ready(data_dir):
                    row = _unavailable_row(dataset, labeler, int(seed), status, reason or "labeler_unavailable", stats)
                    _write_run_artifacts(run_dir, row, args)
                    rows.append(row)
                    print(f"[{row['status']}] labeler={labeler} dataset={dataset} seed={seed}: {row['skip_reason']}")
                    continue
                start = time.perf_counter()
                try:
                    payload = train_hero_with_labels(
                        dataset=dataset,
                        seed=int(seed),
                        label_file=label_file,
                        output_root=output_dir / "raw" / "_project_runs_labeler_comparison",
                        data_root=args.data_root,
                        epochs=int(args.epochs),
                        lr=float(args.lr),
                        hidden_dim=int(args.hidden_dim),
                        device=args.device,
                        experiment_tag=f"labeler_{labeler}",
                        labeler=labeler,
                    )
                    run_status = "ok"
                    run_reason = ""
                except Exception as exc:
                    payload = {}
                    run_status = "missing"
                    run_reason = f"experiment_failed: {type(exc).__name__}: {exc}"
                row = {
                    "suite": "labeler_comparison",
                    "dataset": dataset,
                    "model": "hero_gnn",
                    "labeler": labeler,
                    "seed": int(seed),
                    "label_file": str(label_file),
                    "risk_card_file": str(risk_card_path or ""),
                    "status": run_status,
                    "skip_reason": run_reason,
                    "runtime_seconds": float(time.perf_counter() - start),
                    **stats,
                    **payload,
                }
                _write_run_artifacts(run_dir, row, args)
                _copy_prediction(row, run_dir)
                rows.append(row)
                print(f"[{run_status}] labeler={labeler} dataset={dataset} seed={seed} {run_reason}")
    raw = pd.DataFrame(rows)
    table = summarize_labeler_comparison(raw)
    plot = labeler_plot_data(raw)
    write_frame(output_dir / "summary" / "llm_labeler_comparison_raw.csv", raw)
    write_frame(output_dir / "summary" / "table_llm_labeler_comparison.csv", table)
    write_frame(output_dir / "tables" / "table_llm_labeler_comparison.csv", table)
    write_frame(output_dir / "figures" / "labeler_comparison_data.csv", plot)
    return rows


def summarize_labeler_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    summary = metric_summary(raw, ["dataset", "labeler", "annotation_source"])
    stat_cols = [
        "candidate_cards",
        "annotated_cards",
        "annotation_coverage",
        "risk_relevance_positive_rate",
        "mechanism_distribution",
        "average_confidence",
        "annotation_time_seconds",
        "status",
        "skip_reason",
    ]
    meta_rows = []
    for keys, group in raw.groupby(["dataset", "labeler", "annotation_source"], dropna=False):
        row = {name: value for name, value in zip(["dataset", "labeler", "annotation_source"], keys)}
        for col in stat_cols:
            if col not in group:
                row[col] = pd.NA
            elif col in {"mechanism_distribution", "status", "skip_reason"}:
                row[col] = group[col].dropna().iloc[0] if not group[col].dropna().empty else pd.NA
            else:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                row[col] = float(values.mean()) if not values.empty else pd.NA
        meta_rows.append(row)
    meta = pd.DataFrame(meta_rows)
    if summary.empty:
        return meta
    return meta.merge(summary, on=["dataset", "labeler", "annotation_source"], how="left")


def labeler_plot_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    keep = [
        "dataset",
        "labeler",
        "seed",
        "annotation_source",
        "status",
        "annotation_coverage",
        "risk_relevance_positive_rate",
        "average_confidence",
        *METRICS,
    ]
    for column in keep:
        if column not in raw:
            raw[column] = pd.NA
    return raw[keep]


def _label_specs(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    cards: list[dict[str, Any]],
    risk_card_path: Path | None,
    risk_card_status: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for labeler in args.labelers:
        if labeler in {"random_labeler", "rule_based_labeler", "proxy_labeler"}:
            if not cards:
                specs.append({"labeler": labeler, "status": "unavailable", "reason": risk_card_status, "annotation_source": labeler})
                continue
            labels, elapsed = labels_from_cards(cards, labeler=labeler, seed=0)
            out_file = output_dir / "annotations" / "labeler_comparison" / dataset / f"{labeler}.jsonl"
            write_jsonl_labels(out_file, labels)
            specs.append(
                {
                    "labeler": labeler,
                    "label_file": out_file,
                    "status": "ok",
                    "annotation_source": labeler,
                    "annotation_time_seconds": elapsed,
                }
            )
            continue
        if labeler == "cached_llm_labeler":
            path = _find_cached_llm_file(data_dir, args.cached_llm_label_file)
            specs.append(
                {
                    "labeler": labeler,
                    "label_file": path,
                    "status": "ok" if path is not None else "unavailable",
                    "reason": "" if path is not None else "no_real_llm_cache_found",
                    "annotation_source": "cached_llm" if path is not None else "cached_llm_missing",
                }
            )
            continue
        if labeler == "full_llm_labeler":
            specs.append(_full_llm_spec(dataset, output_dir, cards, risk_card_path, args))
    return specs


def _find_cached_llm_file(data_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for pattern in ["*qwen*.jsonl", "*Qwen*.jsonl", "*openai*.jsonl", "*gpt*.jsonl", "*claude*.jsonl", "*full_llm*.jsonl"]:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _full_llm_spec(dataset: str, output_dir: Path, cards: list[dict[str, Any]], risk_card_path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    if args.full_llm_label_file:
        path = Path(args.full_llm_label_file)
        return {
            "labeler": "full_llm_labeler",
            "label_file": path if path.exists() else None,
            "status": "ok" if path.exists() else "unavailable",
            "reason": "" if path.exists() else "full_llm_label_file_missing",
            "annotation_source": "full_llm_cache",
        }
    if not args.call_full_llm:
        return {
            "labeler": "full_llm_labeler",
            "status": "unavailable",
            "reason": "full_llm_requires_existing_label_file_or_explicit_call_full_llm",
            "annotation_source": "full_llm_unavailable",
        }
    if not has_full_llm_config():
        return {
            "labeler": "full_llm_labeler",
            "status": "unavailable",
            "reason": "OPENAI_API_KEY_or_LOCAL_QWEN_MODEL_PATH_not_configured",
            "annotation_source": "full_llm_unavailable",
        }
    if not cards or risk_card_path is None:
        return {"labeler": "full_llm_labeler", "status": "unavailable", "reason": "risk_cards_unavailable", "annotation_source": "full_llm_unavailable"}
    return _build_full_llm_labels(dataset, output_dir, risk_card_path, args)


def _build_full_llm_labels(dataset: str, output_dir: Path, risk_card_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from scripts.build_llm_labels import _make_labeler, build_labels

    backend = "openai" if os.environ.get("OPENAI_API_KEY") else "local_qwen"
    out_file = output_dir / "annotations" / "labeler_comparison" / dataset / f"full_llm_labeler_{backend}.jsonl"
    build_args = argparse.Namespace(
        dataset=dataset,
        data_dir=None,
        labeler=backend,
        risk_card_file=str(risk_card_path),
        max_cards=int(args.max_cards),
        out_file=str(out_file),
        seed=0,
        max_candidates_per_node=20,
        max_target_nodes=None,
        openai_model="gpt-5.4-mini",
        model_name_or_path=os.environ.get("LOCAL_QWEN_MODEL_PATH", os.environ.get("QWEN_MODEL_PATH", "")),
        local_model_path=None,
        max_new_tokens=256,
        report_dir=str(output_dir / "summary"),
    )
    start = time.perf_counter()
    labeler = _make_labeler(build_args)
    report = build_labels(build_args, labeler)
    report_path = output_dir / "summary" / f"llm_label_build_report_{dataset}_{backend}.json"
    write_json(report_path, report)
    return {
        "labeler": "full_llm_labeler",
        "label_file": out_file,
        "status": "ok",
        "reason": "",
        "annotation_source": f"full_llm_{backend}",
        "annotation_time_seconds": time.perf_counter() - start,
    }


def _unavailable_row(dataset: str, labeler: str, seed: int, status: str, reason: str, stats: dict[str, Any]) -> dict[str, Any]:
    normalized_status = "missing" if status == "ok" else status
    if normalized_status not in {"unavailable", "missing"}:
        normalized_status = "unavailable"
    return {
        "suite": "labeler_comparison",
        "dataset": dataset,
        "model": "hero_gnn",
        "labeler": labeler,
        "seed": int(seed),
        "status": normalized_status,
        "skip_reason": reason,
        **stats,
    }


def _write_run_artifacts(run_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", _json_safe(row))
    write_json(
        run_dir / "config.json",
        {
            "suite": "labeler_comparison",
            "dataset": row["dataset"],
            "model": "hero_gnn",
            "labeler": row["labeler"],
            "seed": int(row["seed"]),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "device": args.device,
        },
    )
    write_json(run_dir / "runtime.json", {"status": row["status"], "runtime_seconds": float(row.get("runtime_seconds", 0.0) or 0.0)})
    (run_dir / "log.txt").write_text(str(row.get("skip_reason", "")) + "\n", encoding="utf-8")


def _copy_prediction(row: dict[str, Any], run_dir: Path) -> None:
    source = row.get("predictions_file")
    if source and Path(str(source)).exists():
        shutil.copyfile(str(source), run_dir / "predictions.npy")


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
