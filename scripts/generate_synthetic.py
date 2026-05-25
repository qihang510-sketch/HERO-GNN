from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic_builder import generate_synthetic_dataset
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic HERO-GNN graph dataset.")
    parser.add_argument("--config", default="configs/synthetic_hero.yaml", help="Path to a YAML config.")
    parser.add_argument("--out_dir", default=None, help="Directory for raw synthetic CSV/JSON files.")
    parser.add_argument("--processed_dir", default=None, help="Directory for processed synthetic files.")
    parser.add_argument("--num_reviews", type=int, default=None, help="Number of review nodes.")
    parser.add_argument("--num_users", type=int, default=None, help="Number of user nodes.")
    parser.add_argument("--num_items", type=int, default=None, help="Number of item nodes.")
    parser.add_argument("--num_devices", type=int, default=None, help="Number of device nodes.")
    parser.add_argument("--fraud_ratio", type=float, default=None, help="Fraction of review nodes labeled as fraud.")
    parser.add_argument(
        "--hetero_fraud_ratio",
        type=float,
        default=None,
        help="Fraction of fraud reviews that are heterophilic fraud.",
    )
    parser.add_argument("--text_dim", type=int, default=None, help="Fixed TF-IDF text feature dimension.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    dataset = config.get("dataset", {})
    paths = generate_synthetic_dataset(
        out_dir=args.out_dir or dataset.get("out_dir", "data/synthetic"),
        processed_dir=args.processed_dir or dataset.get("processed_dir", "data/processed/synthetic"),
        seed=seed,
        num_reviews=args.num_reviews or int(dataset.get("num_reviews", 3000)),
        num_users=args.num_users or int(dataset.get("num_users", 600)),
        num_items=args.num_items or int(dataset.get("num_items", 200)),
        num_devices=args.num_devices or int(dataset.get("num_devices", 300)),
        fraud_ratio=args.fraud_ratio if args.fraud_ratio is not None else float(dataset.get("fraud_ratio", 0.15)),
        hetero_fraud_ratio=(
            args.hetero_fraud_ratio
            if args.hetero_fraud_ratio is not None
            else float(dataset.get("hetero_fraud_ratio", 0.5))
        ),
        text_dim=args.text_dim or int(dataset.get("text_dim", 128)),
    )
    print(f"Wrote raw synthetic data to {paths['raw_nodes'].parent}")
    print(f"Wrote processed synthetic data to {paths['processed_nodes'].parent}")


if __name__ == "__main__":
    main()
