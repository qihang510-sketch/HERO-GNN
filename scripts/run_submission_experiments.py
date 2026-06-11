from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import (  # noqa: E402
    SUBMISSION_DATASETS,
    default_models_for_dataset,
    normalize_dataset_name,
    normalize_model_name,
    run_submission_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission-grade HERO-GNN experiments.")
    parser.add_argument("--datasets", nargs="+", default=list(SUBMISSION_DATASETS))
    parser.add_argument("--models", nargs="*", default=None, help="Optional model list. Defaults to the dataset-specific matrix.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", "--output_root", default="outputs/submission_experiments")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--llm_label_file", default=None, help="Optional annotation file for DGP/MLED/HERO text-rich runs.")
    parser.add_argument("--config", default=None, help="Optional tuned config for hero_gnn/hero_official only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results = []
    for raw_dataset in args.datasets:
        dataset = normalize_dataset_name(raw_dataset)
        models = [normalize_model_name(model) for model in args.models] if args.models else list(default_models_for_dataset(dataset))
        for seed in args.seeds:
            for model in models:
                result = run_submission_experiment(
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    output_dir=output_dir,
                    data_root=args.data_root,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    top_k=args.top_k,
                    overwrite=args.overwrite,
                    device=args.device,
                    llm_label_file=args.llm_label_file,
                    config=args.config,
                )
                results.append(result)
                print(f"[{result.status}] dataset={result.dataset} model={result.model} seed={result.seed} path={result.path} {result.reason}")
    ok = sum(1 for result in results if result.status in {"ok", "exists"})
    skipped = sum(1 for result in results if result.status == "skipped")
    print(f"Submission experiments finished: ok_or_exists={ok} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
    main()
