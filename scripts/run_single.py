from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import train_single_experiment
from src.training.trainer import MODEL_NAMES
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one HERO-GNN baseline experiment.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--dataset", default="synthetic", help="Dataset name.")
    parser.add_argument(
        "--model",
        default="mlp",
        choices=list(MODEL_NAMES),
        help="Model to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--output_root", default="outputs", help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k neighbors for filtered GNN baselines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("outputs", {})
    experiment_cfg = config.get("experiment", {})

    model_name = _model_name_from_config(model_cfg, args.model)
    metrics = train_single_experiment(
        dataset=dataset_cfg.get("name", args.dataset),
        model_name=model_name,
        seed=args.seed if args.seed is not None else int(experiment_cfg.get("seed", 0)),
        data_dir=args.data_dir or dataset_cfg.get("processed_dir"),
        output_root=output_cfg.get("root", args.output_root),
        epochs=int(training_cfg.get("epochs", args.epochs)),
        lr=float(training_cfg.get("lr", args.lr)),
        hidden_dim=int(model_cfg.get("hidden_dim", args.hidden_dim)),
        top_k=int(model_cfg.get("top_k", args.top_k)),
    )
    print(metrics)


def _model_name_from_config(model_cfg: dict, fallback: str) -> str:
    name = model_cfg.get("name", fallback)
    if name != "hero_gnn":
        return name
    if model_cfg.get("use_heterophily") is False:
        return "hero_wo_hetero"
    if model_cfg.get("use_mechanism") is False:
        return "hero_wo_mechanism"
    if model_cfg.get("use_chain") is False:
        return "hero_wo_chain"
    return name


if __name__ == "__main__":
    main()
