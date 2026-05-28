from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dgl_fraud_preprocess import (  # noqa: E402
    DGL_FRAUD_DATASETS,
    DGL_REQUIRED_MESSAGE,
    preprocess_dgl_fraud_dataset,
)
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess DGL fraud benchmark datasets.")
    parser.add_argument("--dataset", required=True, choices=sorted(DGL_FRAUD_DATASETS), help="Official DGL fraud dataset name.")
    parser.add_argument("--out_dir", required=True, help="Processed output directory.")
    parser.add_argument("--raw_dir", default=None, help="Optional DGL raw/cache directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by DGL mask generation.")
    parser.add_argument("--train_size", type=float, default=0.7, help="DGL train mask ratio when supported.")
    parser.add_argument("--val_size", type=float, default=0.1, help="DGL validation mask ratio when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    try:
        output = preprocess_dgl_fraud_dataset(
            dataset=args.dataset,
            output_dir=args.out_dir,
            raw_dir=args.raw_dir,
            seed=args.seed,
            train_size=args.train_size,
            val_size=args.val_size,
        )
    except ImportError as exc:
        if str(exc) == DGL_REQUIRED_MESSAGE:
            raise SystemExit(DGL_REQUIRED_MESSAGE) from exc
        raise
    print(f"Wrote processed DGL fraud data to {output}")


if __name__ == "__main__":
    main()
