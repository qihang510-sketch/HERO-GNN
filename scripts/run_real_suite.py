from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import MODEL_NAMES, train_single_experiment
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HERO-GNN experiments on preprocessed real datasets.")
    parser.add_argument("--dataset", required=True, choices=["yelp_academic", "amazon_video"], help="Dataset name.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES), choices=list(MODEL_NAMES), help="Models to run.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--output_root", default="outputs", help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k neighbors.")
    parser.add_argument("--config", default=None, help="Optional dataset config path.")
    parser.add_argument("--max_target_nodes", type=int, default=None, help="Maximum HERO target nodes to build candidates for.")
    parser.add_argument("--max_candidates_per_node", type=int, default=None, help="Maximum raw heterophilous candidates per target.")
    parser.add_argument("--homophilic_topk", type=int, default=None, help="Top-k homophilic neighbors.")
    parser.add_argument("--heterophilic_topk", type=int, default=None, help="Top-k heterophilous candidates used downstream.")
    parser.add_argument("--topk_chains", type=int, default=None, help="Top-k evidence chains per target.")
    parser.add_argument("--max_chain_length", type=int, default=None, help="Maximum evidence chain length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or f"configs/{args.dataset}_hero.yaml"
    config = load_config(config_path) if Path(config_path).exists() else {}
    dataset_cfg = config.get("dataset", {})
    neighbor_cfg = config.get("neighbor_retrieval", {})
    chain_cfg = config.get("evidence_chain", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("outputs", {})
    for seed in args.seeds:
        for model_name in args.models:
            metrics = train_single_experiment(
                dataset=args.dataset,
                model_name=model_name,
                seed=seed,
                data_dir=args.data_dir or dataset_cfg.get("processed_dir"),
                output_root=output_cfg.get("root", args.output_root),
                epochs=int(training_cfg.get("epochs", args.epochs)),
                lr=float(training_cfg.get("lr", args.lr)),
                hidden_dim=int(model_cfg.get("hidden_dim", args.hidden_dim)),
                top_k=args.top_k,
                max_target_nodes=_first_not_none(args.max_target_nodes, neighbor_cfg.get("max_target_nodes")),
                max_candidates_per_node=int(_first_not_none(args.max_candidates_per_node, neighbor_cfg.get("max_candidates_per_node", 20))),
                homophilic_topk=int(_first_not_none(args.homophilic_topk, neighbor_cfg.get("homophilic_topk", 5))),
                heterophilic_topk=int(_first_not_none(args.heterophilic_topk, neighbor_cfg.get("heterophilic_topk", 5))),
                topk_chains=int(_first_not_none(args.topk_chains, chain_cfg.get("topk_chains", 3))),
                max_chain_length=int(_first_not_none(args.max_chain_length, chain_cfg.get("max_chain_length", 2))),
            )
            print(metrics)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


if __name__ == "__main__":
    main()
