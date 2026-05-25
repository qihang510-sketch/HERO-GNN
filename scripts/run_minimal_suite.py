from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import MODEL_NAMES, train_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal HERO-GNN baseline suite.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), choices=list(MODEL_NAMES), help="Models to run.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--output_root", default="outputs", help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k neighbors for filtered GNN baselines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for model_name in args.models:
            metrics = train_single_experiment(
                dataset=args.dataset,
                model_name=model_name,
                seed=seed,
                data_dir=args.data_dir,
                output_root=args.output_root,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                top_k=args.top_k,
            )
            print(metrics)


if __name__ == "__main__":
    main()
