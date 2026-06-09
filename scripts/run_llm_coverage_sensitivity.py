from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import run_submission_experiment, write_skip  # noqa: E402


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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    qwen_file = Path(args.qwen_label_file) if args.qwen_label_file else _find_qwen_file(Path(args.data_root) / "processed" / args.dataset)
    for coverage in args.coverages:
        tag = f"coverage_{int(round(float(coverage) * 100))}"
        label_file = None
        if float(coverage) > 0:
            if qwen_file is None or not qwen_file.exists():
                for seed in args.seeds:
                    write_skip(output_dir / tag / args.dataset / "hero_gnn" / f"seed_{seed}", args.dataset, "hero_gnn", seed, "missing_existing_qwen_annotations")
                continue
            label_file = _subsample_labels(qwen_file, output_dir / "annotations" / f"{args.dataset}_{tag}_qwen.jsonl", coverage=float(coverage))
        for seed in args.seeds:
            run_submission_experiment(
                dataset=args.dataset,
                model="hero_gnn",
                seed=seed,
                output_dir=output_dir / tag,
                data_root=args.data_root,
                epochs=args.epochs,
                overwrite=args.overwrite,
                device=args.device,
                llm_label_file=label_file,
            )
    print(f"LLM coverage sensitivity finished under {output_dir}")


def _find_qwen_file(processed_dir: Path) -> Path | None:
    for pattern in ["*qwen*.jsonl", "*Qwen*.jsonl"]:
        matches = sorted(processed_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _subsample_labels(src: Path, dst: Path, coverage: float) -> Path:
    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = int(round(len(lines) * max(0.0, min(float(coverage), 1.0))))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines[:keep]) + ("\n" if keep else ""), encoding="utf-8")
    report = {"source": str(src), "output": str(dst), "coverage": float(coverage), "num_source": len(lines), "num_kept": keep}
    (dst.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dst


if __name__ == "__main__":
    main()
