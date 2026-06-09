from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import run_submission_experiment, write_skip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HERO-GNN sensitivity experiments.")
    parser.add_argument("--dataset", default="yelp_academic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output_dir", default="outputs/submission_sensitivity")
    parser.add_argument("--neighbor_budgets", nargs="+", type=int, default=[5, 10, 15, 20, 30])
    parser.add_argument("--routing_lambdas", nargs="+", type=float, default=[0, 0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--chain_lambdas", nargs="+", type=float, default=[0, 0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--mechanism_dims", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--coverages", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    for budget in args.neighbor_budgets:
        for seed in args.seeds:
            run_submission_experiment(
                dataset=args.dataset,
                model="hero_gnn",
                seed=seed,
                output_dir=root / "neighbor_budget" / f"k_{budget}",
                data_root=args.data_root,
                epochs=args.epochs,
                top_k=budget,
                overwrite=args.overwrite,
                device=args.device,
            )
    _write_unsupported_grid(root, args.dataset, "routing_loss", args.routing_lambdas, args.seeds, "routing lambda is not exposed as a separate trainer argument yet")
    _write_unsupported_grid(root, args.dataset, "chain_loss", args.chain_lambdas, args.seeds, "chain lambda grid requires dedicated HERO trainer wiring")
    _write_unsupported_grid(root, args.dataset, "mechanism_dim", args.mechanism_dims, args.seeds, "mechanism embedding dimension is not exposed as a separate trainer argument yet")
    print(f"Sensitivity run finished under {root}. Use run_llm_coverage_sensitivity.py for Qwen coverage.")


def _write_unsupported_grid(root: Path, dataset: str, family: str, values: list[float | int], seeds: list[int], reason: str) -> None:
    for value in values:
        for seed in seeds:
            result_dir = root / family / str(value).replace(".", "p") / dataset / "hero_gnn" / f"seed_{seed}"
            write_skip(result_dir, dataset, f"hero_gnn_{family}_{value}", seed, reason)


if __name__ == "__main__":
    main()
