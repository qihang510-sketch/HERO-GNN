from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.amazon_preprocess import preprocess_amazon_video  # noqa: E402
from src.data.yelp_preprocess import preprocess_yelp_academic  # noqa: E402
from src.training.trainer import train_single_experiment  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_METHODS = [
    "mlp",
    "graphsage",
    "semsim_gnn",
    "rulehetero_gnn",
    "sec_gfd_lite",
    "dga_gnn_lite",
    "flag_lite",
    "hero_gnn",
    "hero_wo_chain",
    "hero_wo_hetero",
    "hero_wo_mechanism",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AAAI main hard-proxy experiments on text-rich datasets.")
    parser.add_argument("--datasets", nargs="+", default=["yelp_academic", "amazon_video"], choices=["yelp_academic", "amazon_video"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--max_reviews_yelp", type=int, default=30000)
    parser.add_argument("--max_reviews_amazon", type=int, default=15000)
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
    for dataset in args.datasets:
        config = _load_dataset_config(dataset)
        dataset_cfg = config.get("dataset", {})
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        if not args.skip_preprocess:
            _preprocess_dataset(dataset, args, dataset_cfg)
        for seed in args.seeds:
            for model_name in args.models:
                metrics = train_single_experiment(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
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


def _load_dataset_config(dataset: str) -> dict:
    path = Path(f"configs/{dataset}_hero.yaml")
    return load_config(path) if path.exists() else {}


def _preprocess_dataset(dataset: str, args: argparse.Namespace, dataset_cfg: dict) -> None:
    if dataset == "yelp_academic":
        preprocess_yelp_academic(
            raw_dir=dataset_cfg.get("raw_dir", "data/raw/yelp_academic"),
            output_dir=dataset_cfg.get("processed_dir", "data/processed/yelp_academic"),
            seed=args.seeds[0],
            max_reviews=args.max_reviews_yelp,
            proxy_label_mode="hard",
            remove_label_features=True,
        )
    elif dataset == "amazon_video":
        preprocess_amazon_video(
            raw_dir=dataset_cfg.get("raw_dir", "data/raw/amazon_video"),
            output_dir=dataset_cfg.get("processed_dir", "data/processed/amazon_video"),
            seed=args.seeds[0],
            max_reviews=args.max_reviews_amazon,
            proxy_label_mode="hard",
            remove_label_features=True,
        )
    else:
        raise ValueError(f"Unsupported main-suite dataset: {dataset}")


if __name__ == "__main__":
    main()
