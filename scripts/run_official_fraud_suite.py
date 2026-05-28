from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dgl_fraud_preprocess import preprocess_dgl_fraud_dataset  # noqa: E402
from src.training.trainer import train_single_experiment  # noqa: E402
from src.utils.config import load_config  # noqa: E402

OFFICIAL_METHODS = [
    "mlp",
    "graphsage",
    "semsim_gnn",
    "rulehetero_gnn",
    "sec_gfd_lite",
    "dga_gnn_lite",
    "hero_official",
    "hero_official_wo_hetero",
    "hero_official_wo_relation",
    "hero_official_wo_feature_deviation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-label DGL fraud benchmark experiments.")
    parser.add_argument("--datasets", nargs="+", default=["fraud_yelp_official", "fraud_amazon_official"], choices=["fraud_yelp_official", "fraud_amazon_official"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=OFFICIAL_METHODS, choices=OFFICIAL_METHODS)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_candidates_per_node", type=int, default=None)
    parser.add_argument("--homophilic_topk", type=int, default=None)
    parser.add_argument("--heterophilic_topk", type=int, default=None)
    parser.add_argument("--topk_chains", type=int, default=None)
    parser.add_argument("--enable_official_chain", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip_preprocess", action="store_true", help="Do not attempt DGL preprocessing when processed files are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        config = _load_dataset_config(dataset)
        dataset_cfg = config.get("dataset", {})
        processed_dir = Path(dataset_cfg.get("processed_dir", f"data/processed/{dataset}"))
        if not args.skip_preprocess and not _processed_ready(processed_dir):
            preprocess_dgl_fraud_dataset(dataset=dataset, output_dir=processed_dir, seed=args.seeds[0], raw_dir=dataset_cfg.get("raw_dir"))
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        for seed in args.seeds:
            for model_name in args.models:
                metrics = train_single_experiment(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
                    data_dir=processed_dir,
                    output_root=args.output_root,
                    epochs=int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 10)),
                    lr=float(args.lr if args.lr is not None else training_cfg.get("lr", 0.001)),
                    hidden_dim=int(args.hidden_dim if args.hidden_dim is not None else model_cfg.get("hidden_dim", 64)),
                    top_k=args.top_k,
                    max_candidates_per_node=args.max_candidates_per_node if args.max_candidates_per_node is not None else 100,
                    homophilic_topk=args.homophilic_topk if args.homophilic_topk is not None else 20,
                    heterophilic_topk=args.heterophilic_topk if args.heterophilic_topk is not None else 20,
                    topk_chains=(args.topk_chains if args.topk_chains is not None else 3) if args.enable_official_chain else 0,
                    lambda_chain_pos=0.0 if not args.enable_official_chain else float(training_cfg.get("lambda_chain_pos", 0.03)),
                    lambda_chain_neg=0.0 if not args.enable_official_chain else float(training_cfg.get("lambda_chain_neg", 0.01)),
                    enable_official_chain=bool(args.enable_official_chain),
                    device=args.device,
                )
                print(metrics)


def _load_dataset_config(dataset: str) -> dict:
    path = Path(f"configs/{dataset}_hero.yaml")
    return load_config(path) if path.exists() else {}


def _processed_ready(path: Path) -> bool:
    return all((path / name).exists() for name in ["nodes.csv", "edges.csv", "features.npz", "split.json"])


if __name__ == "__main__":
    main()
