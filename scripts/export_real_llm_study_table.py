from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.base_labeler import normalize_label  # noqa: E402

OUTPUT_COLUMNS = [
    "dataset",
    "labeler",
    "experiment_tag",
    "num_eval_target_nodes",
    "num_risk_cards",
    "llm_label_coverage_rate",
    "macro_f1",
    "auroc",
    "auprc",
    "evidence_recall_proxy",
    "evidence_necessity_gap",
    "parse_error_count",
    "risk_relevance_rate",
    "avg_confidence",
    "mechanism_distribution",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the high-coverage real LLM subset study table.")
    parser.add_argument("--datasets", nargs="*", default=["yelp_academic", "amazon_video"], help="Datasets to scan.")
    parser.add_argument("--results_root", default="outputs/results_llm_comparison", help="Tagged result root.")
    parser.add_argument("--summary_dir", default="outputs/summary_llm", help="LLM label build report directory.")
    parser.add_argument("--out_dir", default="outputs/summary_llm_highcov", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = export_rows(
        datasets=args.datasets,
        results_root=Path(args.results_root),
        summary_dir=Path(args.summary_dir),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "real_llm_highcov_table.csv"
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
    print(f"Wrote real LLM high-coverage table to {out_file}")


def export_rows(datasets: list[str], results_root: Path, summary_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_root = results_root / dataset
        if not dataset_root.exists():
            continue
        for tag_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            metrics_files = sorted((tag_dir / "hero_gnn").glob("seed_*/metrics.json"))
            if not metrics_files:
                continue
            metrics_rows = [_read_json(path) for path in metrics_files]
            metrics_rows = [row for row in metrics_rows if row]
            if not metrics_rows:
                continue
            if not any(row.get("eval_target_file") or int(row.get("num_eval_target_nodes", 0) or 0) > 0 for row in metrics_rows):
                continue
            first = metrics_rows[0]
            label_file = Path(str(first.get("llm_label_file", first.get("external_llm_label_file", ""))))
            label_stats = _label_stats(label_file) if str(label_file) else _empty_label_stats()
            labeler = str(first.get("llm_labeler", _infer_labeler(label_file)))
            report = _read_label_report(summary_dir, dataset, labeler, label_file)
            rows.append(
                {
                    "dataset": dataset,
                    "labeler": labeler,
                    "experiment_tag": str(first.get("experiment_tag", tag_dir.name)),
                    "num_eval_target_nodes": int(round(_mean(metrics_rows, "num_eval_target_nodes"))),
                    "num_risk_cards": int(label_stats["num_risk_cards"]),
                    "llm_label_coverage_rate": _mean(metrics_rows, "llm_label_coverage_rate"),
                    "macro_f1": _mean(metrics_rows, "macro_f1"),
                    "auroc": _mean(metrics_rows, "auroc"),
                    "auprc": _mean(metrics_rows, "auprc"),
                    "evidence_recall_proxy": _mean(metrics_rows, "evidence_recall_proxy"),
                    "evidence_necessity_gap": _mean(metrics_rows, "evidence_necessity_gap"),
                    "parse_error_count": int(report.get("parse_error_count", 0)),
                    "risk_relevance_rate": float(label_stats["risk_relevance_rate"]),
                    "avg_confidence": float(label_stats["avg_confidence"]),
                    "mechanism_distribution": json.dumps(label_stats["mechanism_distribution"], sort_keys=True),
                }
            )
    return rows


def _label_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_label_stats()
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            labels.append(normalize_label(json.loads(line)))
    if not labels:
        return _empty_label_stats()
    mechanisms: dict[str, int] = {}
    for label in labels:
        mechanism = str(label["mechanism"])
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    return {
        "num_risk_cards": len(labels),
        "risk_relevance_rate": float(np.mean([int(label["risk_relevance"]) for label in labels])),
        "avg_confidence": float(np.mean([float(label["confidence"]) for label in labels])),
        "mechanism_distribution": mechanisms,
    }


def _empty_label_stats() -> dict[str, Any]:
    return {
        "num_risk_cards": 0,
        "risk_relevance_rate": 0.0,
        "avg_confidence": 0.0,
        "mechanism_distribution": {},
    }


def _read_label_report(summary_dir: Path, dataset: str, labeler: str, label_file: Path) -> dict[str, Any]:
    candidates = [
        summary_dir / f"llm_label_build_report_{dataset}_{labeler}.json",
        Path("outputs/summary_llm") / f"llm_label_build_report_{dataset}_{labeler}.json",
    ]
    for path in candidates:
        payload = _read_json(path)
        if not payload:
            continue
        out_file = str(payload.get("out_file", ""))
        if not out_file or not str(label_file) or Path(out_file) == label_file or Path(out_file).name == label_file.name:
            return payload
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else 0.0


def _infer_labeler(label_file: Path) -> str:
    stem = label_file.stem.lower()
    if "qwen" in stem:
        return "local_qwen"
    if "openai" in stem:
        return "openai"
    if "mock" in stem:
        return "mock"
    return "external"


if __name__ == "__main__":
    main()
