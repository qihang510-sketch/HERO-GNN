from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    "mechanism_distribution",
    "avg_confidence",
    "agreement_with_mock",
    "macro_f1",
    "auroc",
    "auprc",
    "evidence_necessity_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mock and real LLM label files on shared risk cards.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--label_files", nargs="+", required=True, help="JSONL label files to compare.")
    parser.add_argument("--out_file", default="outputs/summary/labeler_comparison_table.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_comparison_rows(args.dataset, [Path(path) for path in args.label_files])
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
    print(f"Wrote labeler comparison to {out_file}")


def build_comparison_rows(dataset: str, label_files: list[Path]) -> list[dict]:
    labels_by_name = {labeler_name(path): read_label_file(path) for path in label_files}
    if not labels_by_name:
        return []
    reference_name = "mock" if "mock" in labels_by_name else next(iter(labels_by_name))
    reference = labels_by_name[reference_name]
    rows = []
    for name, labels in labels_by_name.items():
        shared_keys = sorted(set(reference) & set(labels))
        y_true = np.asarray([reference[key]["risk_relevance"] for key in shared_keys], dtype=np.int64)
        y_pred = np.asarray([labels[key]["risk_relevance"] for key in shared_keys], dtype=np.int64)
        scores = np.asarray([_positive_score(labels[key]) for key in shared_keys], dtype=np.float32)
        agreement = float(np.mean(y_true == y_pred)) if shared_keys else 0.0
        rows.append(
            {
                "dataset": dataset,
                "labeler": name,
                "num_cards": len(labels),
                "risk_relevance_rate": _risk_rate(labels),
                "mechanism_distribution": json.dumps(_mechanism_distribution(labels), sort_keys=True),
                "avg_confidence": _avg_confidence(labels),
                "agreement_with_mock": agreement,
                "macro_f1": macro_f1(y_true, y_pred) if shared_keys else 0.0,
                "auroc": safe_auroc(y_true, scores) if shared_keys else 0.0,
                "auprc": safe_auprc(y_true, scores) if shared_keys else 0.0,
                "evidence_necessity_gap": _evidence_necessity_gap(labels, reference if shared_keys else None),
            }
        )
    return rows


def read_label_file(path: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
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


def _positive_score(label: dict) -> float:
    return float(label["confidence"]) if int(label["risk_relevance"]) == 1 else 0.0


def _risk_rate(labels: dict[str, dict]) -> float:
    return float(np.mean([label["risk_relevance"] for label in labels.values()])) if labels else 0.0


def _avg_confidence(labels: dict[str, dict]) -> float:
    return float(np.mean([label["confidence"] for label in labels.values()])) if labels else 0.0


def _mechanism_distribution(labels: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels.values():
        mechanism = str(label["mechanism"])
        counts[mechanism] = counts.get(mechanism, 0) + 1
    return counts


def _evidence_necessity_gap(labels: dict[str, dict], reference: dict[str, dict] | None = None) -> float:
    positive_scores = []
    negative_scores = []
    for key, label in labels.items():
        ref_label = reference.get(key) if reference is not None else label
        if ref_label is None:
            continue
        if int(ref_label["risk_relevance"]) == 1:
            positive_scores.append(_positive_score(label))
        else:
            negative_scores.append(_positive_score(label))
    pos = float(np.mean(positive_scores)) if positive_scores else 0.0
    neg = float(np.mean(negative_scores)) if negative_scores else 0.0
    return pos - neg


if __name__ == "__main__":
    main()
