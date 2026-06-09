from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.elliptic_loader import prepare_elliptic_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Elliptic Bitcoin transaction graph.")
    parser.add_argument("--raw_dir", default="data/raw/elliptic")
    parser.add_argument("--output_dir", default="data/processed/elliptic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed Elliptic files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = prepare_elliptic_dataset(args.raw_dir, args.output_dir, seed=args.seed, force=args.force)
    print(f"Prepared Elliptic dataset at {out}")


if __name__ == "__main__":
    main()
