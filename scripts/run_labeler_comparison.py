from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.base_labeler import label_key, normalize_label  # noqa: E402
from src.training.submission import _submission_metric_payload, processed_ready, resolve_processed_dir, write_skip  # noqa: E402
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.metrics import macro_f1, safe_auprc, safe_auroc  # noqa: E402
from src.utils.io import write_json  # noqa: E402


OUTPUT_COLUMNS = [
    "dataset",
    "labeler",
    "seed",
    "num_cards",
    "coverage",
    "risk_relevance_rate",
    "avg_confidence",
    "mechanism_distribution",
    "agreement_with_rule",
    "parse_error_count",
    "Macro-F1",
    "AUROC",
    "AUPRC",
    "Accuracy",
    "Precision",
    "Recall",
    "evidence_necessity_gap",
    "llm_label_coverage_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rule/mock and Qwen labelers with real HERO-GNN training runs.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--label_files", nargs="+", default=None, help="JSONL label files to compare.")
    parser.add_argument("--labelers", nargs="*", default=None, help="Submission alias: labelers to compare, e.g. rule qwen.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="Seeds to train for each labeler.")
    parser.add_argument("--use_existing_annotations", default="true", help="Real LLM calls are not made by this script.")
    parser.add_argument("--out_dir", default="outputs/summary_llm", help="Output summary directory.")
    parser.add_argument("--output_dir", default=None, help="Alias for --out_dir.")
    parser.add_argument("--out_file", default=None, help="Optional explicit CSV path.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results_root", default="outputs/results_llm_comparison", help="Legacy root containing tagged HERO-GNN results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or args.out_dir)
    data_dir = resolve_processed_dir(args.dataset, args.data_root)
    specs = _resolve_label_specs(data_dir, args.labelers, args.label_files)
    rows: list[dict[str, Any]] = []

    for spec in specs:
        labeler = str(spec["labeler"])
        label_file = spec.get("path")
        result_base = out_dir / args.dataset / labeler
        if not processed_ready(data_dir):
            for seed in args.seeds or [0]:
                write_skip(
                    result_base / f"seed_{seed}",
                    args.dataset,
                    labeler,
                    seed,
                    "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.",
                    {"data_dir": str(data_dir)},
                )
            continue
        if label_file is None or not Path(label_file).exists():
            for seed in args.seeds or [0]:
                write_skip(
                    result_base / f"seed_{seed}",
                    args.dataset,
                    labeler,
                    seed,
                    "missing_existing_annotations",
                    {"labeler": labeler},
                )
            continue

        label_path = Path(label_file)
        annotation_stats = _annotation_stats(label_path)
        hero_config = _resolve_hero_config("hero_gnn", {"use_llm_annotation": labeler == "qwen"})
        for seed in args.seeds or [0]:
            result_dir = result_base / f"seed_{seed}"
            metrics_path = result_dir / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                print(f"[exists] {metrics_path}")
                try:
                    rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    pass
                continue
            try:
                metrics = train_single_experiment(
                    dataset=args.dataset,
                    model_name="hero_gnn",
                    seed=seed,
                    data_dir=data_dir,
                    output_root=out_dir / "_project_runs",
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    llm_label_file=label_path,
                    experiment_tag=labeler,
                    llm_labeler=labeler,
                    disable_llm_fallback=True,
                    device=args.device,
                    hero_config=hero_config,
                )
            except Exception as exc:
                write_skip(
                    result_dir,
                    args.dataset,
                    labeler,
                    seed,
                    f"experiment_failed: {type(exc).__name__}: {exc}",
                    {"labeler": labeler, "llm_label_file": str(label_path)},
                )
                continue
            payload = _labeler_metric_payload(
                metrics=metrics,
                dataset=args.dataset,
                seed=seed,
                labeler=labeler,
                label_file=label_path,
                annotation_stats=annotation_stats,
                hero_config=hero_config,
            )
            write_json(metrics_path, payload)
            _write_config(result_dir, payload, args, hero_config)
            (result_dir / "run.log").write_text(
                f"Completed labeler comparison dataset={args.dataset} labeler={labeler} seed={seed}\n",
                encoding="utf-8",
            )
            rows.append(payload)
            print(f"[ok] {metrics_path}")

    out_file = Path(args.out_file) if args.out_file else out_dir / "labeler_comparison_table.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_comparison_csv_row(row) for row in rows], columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
    print(f"Wrote labeler comparison to {out_file}")


def _resolve_label_specs(processed_dir: Path, labelers: list[str] | None, label_files: list[str] | None) -> list[dict[str, Any]]:
    if label_files:
        return [{"labeler": labeler_name(Path(path)), "path": Path(path)} for path in label_files]
    requested = [_canonical_labeler(labeler) for labeler in (labelers or ["rule", "qwen"])]
    specs: list[dict[str, Any]] = []
    for labeler in requested:
        specs.append({"labeler": labeler, "path": _find_label_file(processed_dir, labeler)})
    return specs


def _find_label_file(processed_dir: Path, labeler: str) -> Path | None:
    if labeler in {"rule", "mock"}:
        candidates = [
            processed_dir / "llm_labels_rule.jsonl",
            processed_dir / "rule_labels.jsonl",
            processed_dir / "llm_labels.jsonl",
            processed_dir / "llm_labels_mock.jsonl",
            *sorted(processed_dir.glob("*mock*.jsonl")),
            *sorted(processed_dir.glob("*rule*.jsonl")),
        ]
    elif labeler == "qwen":
        candidates = [*sorted(processed_dir.glob("*qwen*.jsonl")), *sorted(processed_dir.glob("*Qwen*.jsonl"))]
    else:
        candidates = [*sorted(processed_dir.glob(f"*{labeler}*.jsonl"))]
    return next((path for path in candidates if path.exists()), None)


def _labeler_metric_payload(
    metrics: dict[str, Any],
    dataset: str,
    seed: int,
    labeler: str,
    label_file: Path,
    annotation_stats: dict[str, Any],
    hero_config: dict[str, Any],
) -> dict[str, Any]:
    payload = _submission_metric_payload(metrics, dataset, "hero_gnn", seed, "project")
    coverage = float(metrics.get("llm_label_coverage_rate", 0.0))
    if coverage == 0.0 and annotation_stats.get("num_cards", 0):
        coverage = 1.0
    payload.update(
        {
            "labeler": labeler,
            "coverage": coverage,
            "llm_label_file": str(label_file),
            "hero_config": hero_config,
            **annotation_stats,
        }
    )
    return payload


def _annotation_stats(path: Path) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    parse_error_count = 0
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if not line.strip():
            continue
        try:
            labels.append(normalize_label(json.loads(line)))
        except Exception:
            parse_error_count += 1
    risk_values = [int(label.get("risk_relevance", 0)) for label in labels]
    confidences = [float(label.get("confidence", 0.0)) for label in labels]
    return {
        "num_cards": int(len(labels)),
        "risk_relevance_rate": float(sum(risk_values) / len(risk_values)) if risk_values else 0.0,
        "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "mechanism_distribution": json.dumps(_mechanism_distribution({label_key(label): label for label in labels}), sort_keys=True),
        "agreement_with_rule": 1.0,
        "parse_error_count": int(parse_error_count),
    }


def _write_config(result_dir: Path, payload: dict[str, Any], args: argparse.Namespace, hero_config: dict[str, Any]) -> None:
    config = {
        "dataset": payload["dataset"],
        "model": "hero_gnn",
        "labeler": str(payload["labeler"]),
        "seed": int(payload["seed"]),
        "llm_label_file": str(payload["llm_label_file"]),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "disable_llm_fallback": True,
        "hero_config": hero_config,
        **hero_config,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _comparison_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset", ""),
        "labeler": row.get("labeler", ""),
        "seed": row.get("seed", ""),
        "num_cards": row.get("num_cards", 0),
        "coverage": row.get("coverage", row.get("llm_label_coverage_rate", 0.0)),
        "risk_relevance_rate": row.get("risk_relevance_rate", 0.0),
        "avg_confidence": row.get("avg_confidence", 0.0),
        "mechanism_distribution": row.get("mechanism_distribution", "{}"),
        "agreement_with_rule": row.get("agreement_with_rule", 1.0),
        "parse_error_count": row.get("parse_error_count", 0),
        "Macro-F1": row.get("Macro-F1", row.get("macro_f1", 0.0)),
        "AUROC": row.get("AUROC", row.get("auroc", 0.0)),
        "AUPRC": row.get("AUPRC", row.get("auprc", 0.0)),
        "Accuracy": row.get("Accuracy", 0.0),
        "Precision": row.get("Precision", 0.0),
        "Recall": row.get("Recall", 0.0),
        "evidence_necessity_gap": row.get("evidence_necessity_gap", 0.0),
        "llm_label_coverage_rate": row.get("llm_label_coverage_rate", row.get("coverage", 0.0)),
    }


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
    reference_name = "rule" if "rule" in labels_by_name else ("mock" if "mock" in labels_by_name else next(iter(labels_by_name)))
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
                "seed": 0,
                "num_cards": len(labels),
                "coverage": hero_metrics.get("llm_label_coverage_rate", 0.0),
                "risk_relevance_rate": _risk_rate(labels),
                "avg_confidence": _avg_confidence(labels),
                "mechanism_distribution": json.dumps(_mechanism_distribution(labels), sort_keys=True),
                "agreement_with_rule": agreement,
                "parse_error_count": _parse_error_count(dataset, name, out_dir),
                "Macro-F1": hero_metrics.get("macro_f1", macro_f1(y_true, y_pred) if shared_keys else 0.0),
                "AUROC": hero_metrics.get("auroc", safe_auroc(y_true, scores) if shared_keys else 0.0),
                "AUPRC": hero_metrics.get("auprc", safe_auprc(y_true, scores) if shared_keys else 0.0),
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
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
    if "qwen" in stem:
        return "qwen"
    if "rule" in stem:
        return "rule"
    if "mock" in stem or stem == "llm_labels":
        return "mock"
    if "openai" in stem:
        return "openai"
    return _canonical_labeler(stem)


def _canonical_labeler(labeler: str) -> str:
    text = str(labeler).strip().lower().replace("-", "_")
    if text in {"qwen", "local_qwen", "qwen2.5", "qwen2p5_7b"}:
        return "qwen"
    if text in {"rule", "rules"}:
        return "rule"
    if text in {"mock", "llm_labels"}:
        return "mock"
    return text


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
    tags = [stem, labeler]
    if stem.startswith("llm_labels_"):
        tags.append(stem[len("llm_labels_") :])
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
