from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.base_labeler import label_key, normalize_label  # noqa: E402
from src.utils.metrics import macro_f1, safe_auprc, safe_auroc  # noqa: E402


OUTPUT_COLUMNS = [
    "dataset",
    "labeler",
    "num_cards",
    "risk_relevance_rate",
    "avg_confidence",
    "mechanism_distribution",
    "agreement_with_mock",
    "parse_error_count",
    "macro_f1",
    "auroc",
    "auprc",
    "evidence_necessity_gap",
    "llm_label_coverage_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mock and real LLM mechanism label files.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--label_files", nargs="+", required=True, help="JSONL label files to compare.")
    parser.add_argument("--out_dir", default="outputs/summary_llm", help="Output summary directory.")
    parser.add_argument("--out_file", default=None, help="Optional explicit CSV path.")
    parser.add_argument("--results_root", default="outputs/results_llm_comparison", help="Root containing tagged HERO-GNN results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows = build_comparison_rows(
        args.dataset,
        [Path(path) for path in args.label_files],
        out_dir=out_dir,
        results_root=Path(args.results_root),
    )
    out_file = Path(args.out_file) if args.out_file else out_dir / "labeler_comparison_table.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
    print(f"Wrote labeler comparison to {out_file}")


def build_comparison_rows(
    dataset: str,
    label_files: list[Path],
    out_dir: Path | None = None,
    results_root: Path | None = None,
) -> list[dict[str, Any]]:
    out_dir = out_dir or Path("outputs/summary_llm")
    results_root = results_root or Path("outputs/results_llm_comparison")
    labels_by_name = {labeler_name(path): read_label_file(path) for path in label_files}
    if not labels_by_name:
        return []
    reference_name = "mock" if "mock" in labels_by_name else next(iter(labels_by_name))
    reference = labels_by_name[reference_name]
    rows = []
    for path in label_files:
        name = labeler_name(path)
        labels = labels_by_name[name]
        shared_keys = sorted(set(reference) & set(labels))
        y_true = np.asarray([reference[key]["risk_relevance"] for key in shared_keys], dtype=np.int64)
        y_pred = np.asarray([labels[key]["risk_relevance"] for key in shared_keys], dtype=np.int64)
        scores = np.asarray([_positive_score(labels[key]) for key in shared_keys], dtype=np.float32)
        agreement = float(np.mean(y_true == y_pred)) if shared_keys else 0.0
        hero_metrics = _find_hero_metrics(dataset, path, name, results_root)
        rows.append(
            {
                "dataset": dataset,
                "labeler": name,
                "num_cards": len(labels),
                "risk_relevance_rate": _risk_rate(labels),
                "avg_confidence": _avg_confidence(labels),
                "mechanism_distribution": json.dumps(_mechanism_distribution(labels), sort_keys=True),
                "agreement_with_mock": agreement,
                "parse_error_count": _parse_error_count(dataset, name, out_dir),
                "macro_f1": hero_metrics.get("macro_f1", macro_f1(y_true, y_pred) if shared_keys else 0.0),
                "auroc": hero_metrics.get("auroc", safe_auroc(y_true, scores) if shared_keys else 0.0),
                "auprc": hero_metrics.get("auprc", safe_auprc(y_true, scores) if shared_keys else 0.0),
                "evidence_necessity_gap": hero_metrics.get("evidence_necessity_gap", 0.0),
                "llm_label_coverage_rate": hero_metrics.get("llm_label_coverage_rate", 0.0),
            }
        )
    return rows


def read_label_file(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return labels
    for line in text.splitlines():
        if not line.strip():
            continue
        label = normalize_label(json.loads(line))
        labels[label_key(label)] = label
    return labels


def labeler_name(path: Path) -> str:
    stem = path.stem.lower()
    if "mock" in stem:
        return "mock"
    if "openai" in stem:
        return "openai"
    if "qwen" in stem:
        return "local_qwen"
    return stem


def _positive_score(label: dict[str, Any]) -> float:
    return float(label["confidence"]) if int(label["risk_relevance"]) == 1 else 0.0


def _risk_rate(labels: dict[str, dict[str, Any]]) -> float:
    return float(np.mean([label["risk_relevance"] for label in labels.values()])) if labels else 0.0


def _avg_confidence(labels: dict[str, dict[str, Any]]) -> float:
    return float(np.mean([label["confidence"] for label in labels.values()])) if labels else 0.0


def _mechanism_distribution(labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels.values():
        mechanism = str(label["mechanism"])
        counts[mechanism] = counts.get(mechanism, 0) + 1
    return counts


def _parse_error_count(dataset: str, labeler: str, out_dir: Path) -> int:
    candidates = [
        out_dir / f"llm_label_build_report_{dataset}_{labeler}.json",
        Path("outputs/summary_llm") / f"llm_label_build_report_{dataset}_{labeler}.json",
    ]
    if labeler == "local_qwen":
        candidates.append(out_dir / f"llm_label_build_report_{dataset}_local_qwen.json")
    for path in candidates:
        if not path.exists():
            continue
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("parse_error_count", 0))
        except (json.JSONDecodeError, ValueError):
            continue
    return 0


def _find_hero_metrics(dataset: str, label_file: Path, labeler: str, results_root: Path) -> dict[str, float]:
    tag_candidates = _tag_candidates(label_file, labeler)
    metrics_files: list[Path] = []
    for tag in tag_candidates:
        metrics_files.extend((results_root / dataset / tag / "hero_gnn").glob("seed_*/metrics.json"))
    rows = []
    for path in metrics_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("model") != "hero_gnn":
            continue
        rows.append(payload)
    if not rows:
        return {}
    return {
        "macro_f1": _mean_metric(rows, "macro_f1"),
        "auroc": _mean_metric(rows, "auroc"),
        "auprc": _mean_metric(rows, "auprc"),
        "evidence_necessity_gap": _mean_metric(rows, "evidence_necessity_gap"),
        "llm_label_coverage_rate": _mean_metric(rows, "llm_label_coverage_rate"),
    }


def _tag_candidates(label_file: Path, labeler: str) -> list[str]:
    stem = label_file.stem
    tags = [stem]
    if stem.startswith("llm_labels_"):
        tags.append(stem[len("llm_labels_") :])
    tags.append(labeler)
    return list(dict.fromkeys(tags))


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
