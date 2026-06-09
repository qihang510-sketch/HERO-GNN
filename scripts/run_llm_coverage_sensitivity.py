from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.base_labeler import normalize_label  # noqa: E402
from src.training.submission import _submission_metric_payload, processed_ready, resolve_processed_dir, write_skip  # noqa: E402
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM annotation coverage sensitivity using existing Qwen annotations.")
    parser.add_argument("--dataset", default="yelp_academic")
    parser.add_argument("--coverages", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--use_existing_qwen_annotations", default="true")
    parser.add_argument("--qwen_label_file", default=None)
    parser.add_argument("--output_dir", default="outputs/submission_llm_coverage")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = resolve_processed_dir(args.dataset, args.data_root)
    qwen_file = Path(args.qwen_label_file) if args.qwen_label_file else _find_qwen_file(data_dir)
    source_lines = _read_nonempty_lines(qwen_file) if qwen_file and qwen_file.exists() else []

    for coverage in args.coverages:
        coverage_value = _bounded_coverage(coverage)
        tag = _coverage_tag(coverage_value)
        result_base = output_dir / args.dataset / tag
        if not processed_ready(data_dir):
            for seed in args.seeds:
                write_skip(
                    result_base / f"seed_{seed}",
                    args.dataset,
                    tag,
                    seed,
                    "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.",
                    {"data_dir": str(data_dir)},
                )
            continue
        if coverage_value > 0.0 and (qwen_file is None or not qwen_file.exists()):
            for seed in args.seeds:
                write_skip(
                    result_base / f"seed_{seed}",
                    args.dataset,
                    tag,
                    seed,
                    "missing_existing_qwen_annotations",
                    {"requested_coverage": coverage_value},
                )
            continue

        label_file = _subsample_labels(
            source_lines=source_lines,
            dst=output_dir / "annotations" / f"{args.dataset}_{tag}_qwen.jsonl",
            coverage=coverage_value,
            source=str(qwen_file) if qwen_file else "",
        )
        annotation_stats = _label_file_stats(label_file, source_count=len(source_lines), requested_coverage=coverage_value)
        hero_config = _resolve_hero_config("hero_gnn", {"use_llm_annotation": coverage_value > 0.0})
        for seed in args.seeds:
            result_dir = result_base / f"seed_{seed}"
            metrics_path = result_dir / "metrics.json"
            if metrics_path.exists() and not args.overwrite:
                print(f"[exists] {metrics_path}")
                continue
            try:
                metrics = train_single_experiment(
                    dataset=args.dataset,
                    model_name="hero_gnn",
                    seed=seed,
                    data_dir=data_dir,
                    output_root=output_dir / "_project_runs",
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    llm_label_file=label_file,
                    experiment_tag=tag,
                    llm_labeler="qwen",
                    disable_llm_fallback=True,
                    device=args.device,
                    hero_config=hero_config,
                )
            except Exception as exc:
                write_skip(
                    result_dir,
                    args.dataset,
                    tag,
                    seed,
                    f"experiment_failed: {type(exc).__name__}: {exc}",
                    {"coverage": coverage_value, "llm_label_file": str(label_file)},
                )
                continue
            payload = _coverage_metric_payload(
                metrics=metrics,
                dataset=args.dataset,
                seed=seed,
                tag=tag,
                coverage=coverage_value,
                label_file=label_file,
                annotation_stats=annotation_stats,
                hero_config=hero_config,
            )
            write_json(metrics_path, payload)
            _write_config(result_dir, payload, args, hero_config)
            (result_dir / "run.log").write_text(
                f"Completed LLM coverage sensitivity dataset={args.dataset} coverage={coverage_value} seed={seed}\n",
                encoding="utf-8",
            )
            print(f"[ok] {metrics_path}")
    print(f"LLM coverage sensitivity finished under {output_dir}")


def _find_qwen_file(processed_dir: Path) -> Path | None:
    for pattern in ["*qwen*.jsonl", "*Qwen*.jsonl"]:
        matches = sorted(processed_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_nonempty_lines(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _subsample_labels(source_lines: list[str], dst: Path, coverage: float, source: str) -> Path:
    keep = int(round(len(source_lines) * _bounded_coverage(coverage)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = source_lines[:keep]
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report = {
        "source": source,
        "output": str(dst),
        "coverage": float(_bounded_coverage(coverage)),
        "num_source": len(source_lines),
        "num_kept": keep,
    }
    dst.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dst


def _label_file_stats(path: Path, source_count: int, requested_coverage: float) -> dict[str, Any]:
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
    actual_coverage = len(labels) / max(int(source_count), 1) if source_count else float(_bounded_coverage(requested_coverage))
    return {
        "coverage": float(_bounded_coverage(requested_coverage)),
        "actual_annotation_coverage": float(actual_coverage),
        "num_annotations": int(len(labels)),
        "parse_error_count": int(parse_error_count),
        "risk_relevance_rate": float(sum(risk_values) / len(risk_values)) if risk_values else 0.0,
        "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
    }


def _coverage_metric_payload(
    metrics: dict[str, Any],
    dataset: str,
    seed: int,
    tag: str,
    coverage: float,
    label_file: Path,
    annotation_stats: dict[str, Any],
    hero_config: dict[str, Any],
) -> dict[str, Any]:
    payload = _submission_metric_payload(metrics, dataset, "hero_gnn", seed, "project")
    payload.update(
        {
            "coverage": float(coverage),
            "coverage_tag": tag,
            "labeler": "qwen",
            "llm_label_file": str(label_file),
            "hero_config": hero_config,
            **annotation_stats,
        }
    )
    return payload


def _write_config(result_dir: Path, payload: dict[str, Any], args: argparse.Namespace, hero_config: dict[str, Any]) -> None:
    config = {
        "dataset": payload["dataset"],
        "model": "hero_gnn",
        "seed": int(payload["seed"]),
        "coverage": float(payload["coverage"]),
        "coverage_tag": str(payload["coverage_tag"]),
        "labeler": "qwen",
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


def _coverage_tag(coverage: float) -> str:
    return f"coverage_{int(round(_bounded_coverage(coverage) * 100))}"


def _bounded_coverage(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


if __name__ == "__main__":
    main()
