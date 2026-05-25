from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.amazon_preprocess import preprocess_amazon_video
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Amazon Video data.")
    parser.add_argument("--config", default="configs/amazon_video_hero.yaml", help="Path to a YAML config.")
    parser.add_argument("--raw_dir", default=None, help="Directory containing raw Amazon Video files.")
    parser.add_argument("--out_dir", default=None, help="Processed output directory.")
    parser.add_argument("--max_reviews", type=int, default=None, help="Maximum number of reviews to read.")
    parser.add_argument("--max_neighbors_per_type", type=int, default=30, help="Neighbor cap per review and relation.")
    parser.add_argument("--text_dim", type=int, default=128, help="Fixed TF-IDF text feature dimension.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset", {})
    seed = args.seed if args.seed is not None else int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)
    output = preprocess_amazon_video(
        args.raw_dir or dataset.get("raw_dir", "data/raw/amazon_video"),
        args.out_dir or dataset.get("processed_dir", "data/processed/amazon_video"),
        seed=seed,
        max_reviews=args.max_reviews,
        max_neighbors_per_type=args.max_neighbors_per_type,
        text_dim=args.text_dim,
    )
    print(f"Wrote processed Amazon Video data to {output}")


if __name__ == "__main__":
    main()
