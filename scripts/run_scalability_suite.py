from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_aaai_main_suite import DEFAULT_METHODS, _load_dataset_config, _preprocess_dataset  # noqa: E402
from src.training.trainer import train_single_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale single-seed scalability experiments.")
    parser.add_argument("--datasets", nargs="+", default=["yelp_academic", "amazon_video"], choices=["yelp_academic", "amazon_video"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--max_reviews_yelp", type=int, default=50000)
    parser.add_argument("--max_reviews_amazon", type=int, default=30000)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_candidates_per_node", type=int, default=20)
    parser.add_argument("--homophilic_topk", type=int, default=5)
    parser.add_argument("--heterophilic_topk", type=int, default=5)
    parser.add_argument("--topk_chains", type=int, default=3)
    parser.add_argument("--skip_preprocess", action="store_true", help="Use existing processed data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = [args.seed]
    for dataset in args.datasets:
        config = _load_dataset_config(dataset)
        dataset_cfg = config.get("dataset", {})
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        if not args.skip_preprocess:
            _preprocess_dataset(dataset, args, dataset_cfg)
        for model_name in args.models:
            metrics = train_single_experiment(
                dataset=dataset,
                model_name=model_name,
                seed=args.seed,
                data_dir=dataset_cfg.get("processed_dir", f"data/processed/{dataset}"),
                output_root=args.output_root,
                epochs=int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 10)),
                lr=float(args.lr if args.lr is not None else training_cfg.get("lr", 0.001)),
                hidden_dim=int(args.hidden_dim if args.hidden_dim is not None else model_cfg.get("hidden_dim", 64)),
                top_k=args.top_k,
                max_candidates_per_node=args.max_candidates_per_node,
                homophilic_topk=args.homophilic_topk,
                heterophilic_topk=args.heterophilic_topk,
                topk_chains=args.topk_chains,
            )
            print(metrics)


if __name__ == "__main__":
    main()
