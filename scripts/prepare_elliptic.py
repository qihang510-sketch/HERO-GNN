from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.elliptic_loader import (  # noqa: E402
    check_elliptic_processed,
    format_elliptic_check_report,
    prepare_elliptic_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Elliptic Bitcoin transaction graph.")
    parser.add_argument("--raw_dir", default="data/raw/elliptic")
    parser.add_argument("--output_dir", default="data/processed/elliptic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed Elliptic files.")
    parser.add_argument("--check_only", action="store_true", help="Only inspect the processed Elliptic split and class distribution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        payload = check_elliptic_processed(args.output_dir)
        print(format_elliptic_check_report(payload))
        return
    out = prepare_elliptic_dataset(args.raw_dir, args.output_dir, seed=args.seed, force=args.force)
    print(f"Prepared Elliptic dataset at {out}")
    payload = check_elliptic_processed(out)
    print(format_elliptic_check_report(payload))


if __name__ == "__main__":
    main()
