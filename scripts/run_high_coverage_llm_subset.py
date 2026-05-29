from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import train_single_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the high-coverage real-LLM subset HERO-GNN study.")
    parser.add_argument("--dataset", required=True, choices=["yelp_academic", "amazon_video"], help="Text-rich dataset name.")
    parser.add_argument("--data_dir", default=None, help="Processed dataset directory.")
    parser.add_argument("--target_file", required=True, help="Subset target-id JSON file.")
    parser.add_argument("--llm_label_file", required=True, help="Dense LLM label JSONL file for the subset.")
    parser.add_argument("--experiment_tag", required=True, help="Output tag, e.g. qwen2p5_7b_highcov_500x10.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Training device.")
    parser.add_argument("--output_root", default="outputs", help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for the subset smoke/study run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_single_experiment(
        dataset=args.dataset,
        model_name="hero_gnn",
        seed=int(args.seed),
        data_dir=Path(args.data_dir or f"data/processed/{args.dataset}"),
        output_root=args.output_root,
        epochs=int(args.epochs),
        lr=float(args.lr),
        hidden_dim=int(args.hidden_dim),
        max_candidates_per_node=20,
        homophilic_topk=5,
        heterophilic_topk=10,
        topk_chains=3,
        llm_label_file=args.llm_label_file,
        experiment_tag=args.experiment_tag,
        llm_labeler=_infer_labeler(args.llm_label_file),
        eval_target_file=args.target_file,
        disable_llm_fallback=True,
        device=args.device,
    )
    print(metrics)


def _infer_labeler(label_file: str | Path) -> str:
    stem = Path(label_file).stem.lower()
    if "qwen" in stem:
        return "local_qwen"
    if "openai" in stem:
        return "openai"
    if "mock" in stem:
        return "mock"
    return "external"


if __name__ == "__main__":
    main()
