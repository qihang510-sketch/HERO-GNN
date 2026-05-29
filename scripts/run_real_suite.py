from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

from src.training.trainer import MODEL_NAMES, train_single_experiment
from src.utils.config import load_config

TEXT_RICH_DEFAULT_METHODS = [
    "mlp",
    "graphsage",
    "semsim_gnn",
    "rulehetero_gnn",
    "sec_gfd_lite",
    "dga_gnn_lite",
    "flag_lite",
    "hero_gnn",
    "hero_wo_chain",
    "hero_wo_hetero",
    "hero_wo_mechanism",
]
OFFICIAL_DEFAULT_METHODS = [
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
    parser = argparse.ArgumentParser(description="Run HERO-GNN experiments on preprocessed real datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["yelp_academic", "amazon_video", "fraud_yelp_official", "fraud_amazon_official"],
        help="Dataset name.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--models", nargs="*", default=None, choices=list(MODEL_NAMES), help="Models to run.")
    parser.add_argument("--methods", nargs="*", default=None, choices=list(MODEL_NAMES), help="Alias for --models.")
    parser.add_argument("--data_dir", default=None, help="Processed data directory.")
    parser.add_argument("--output_root", default="outputs", help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
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
    parser.add_argument("--min_chain_quality", type=float, default=None, help="Minimum evidence chain quality used by HERO.")
    parser.add_argument("--lambda_chain_pos", type=float, default=None, help="Positive-sample chain contribution loss weight.")
    parser.add_argument("--lambda_chain_neg", type=float, default=None, help="Negative-sample chain contribution loss weight.")
    parser.add_argument("--llm_label_file", default=None, help="Optional prebuilt LLM label JSONL file for HERO-style models.")
    parser.add_argument("--experiment_tag", default=None, help="Optional tag for LLM labeler comparison outputs.")
    parser.add_argument("--eval_target_file", default=None, help="Optional target-id JSON file for subset test evaluation.")
    parser.add_argument("--disable_llm_fallback", action="store_true", help="Use irrelevant_heterophily for missing external LLM labels.")
    parser.add_argument("--enable_official_chain", action="store_true", help="Enable evidence-chain knobs for HERO-official variants.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Training device.")
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
    data_dir = Path(args.data_dir or dataset_cfg.get("processed_dir", f"data/processed/{args.dataset}"))
    official_mode = _is_official_dataset(args.dataset, data_dir)
    selected_models = args.methods if args.methods is not None else args.models
    if not selected_models:
        selected_models = OFFICIAL_DEFAULT_METHODS if official_mode else TEXT_RICH_DEFAULT_METHODS
    experiment_tag = args.experiment_tag
    if args.llm_label_file and not experiment_tag:
        experiment_tag = _infer_experiment_tag(args.llm_label_file)
    llm_labeler = _infer_llm_labeler(args.llm_label_file) if args.llm_label_file else "mock"
    max_candidates = _first_not_none(
        args.max_candidates_per_node,
        100 if official_mode else neighbor_cfg.get("max_candidates_per_node"),
        20,
    )
    homo_topk = _first_not_none(args.homophilic_topk, 20 if official_mode else neighbor_cfg.get("homophilic_topk"), 5)
    hetero_topk = _first_not_none(args.heterophilic_topk, 20 if official_mode else neighbor_cfg.get("heterophilic_topk"), 5)
    if official_mode and not args.enable_official_chain:
        topk_chains = 0
        lambda_chain_pos = 0.0
        lambda_chain_neg = 0.0
    else:
        topk_chains = _first_not_none(args.topk_chains, chain_cfg.get("topk_chains"), 3)
        lambda_chain_pos = _first_not_none(args.lambda_chain_pos, training_cfg.get("lambda_chain_pos"), 0.03)
        lambda_chain_neg = _first_not_none(args.lambda_chain_neg, training_cfg.get("lambda_chain_neg"), 0.01)
    for seed in args.seeds:
        metrics_by_model = {}
        for model_name in selected_models:
            metrics = train_single_experiment(
                dataset=args.dataset,
                model_name=model_name,
                seed=seed,
                data_dir=data_dir,
                output_root=output_cfg.get("root", args.output_root),
                epochs=int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 50)),
                lr=float(training_cfg.get("lr", args.lr)),
                hidden_dim=int(model_cfg.get("hidden_dim", args.hidden_dim)),
                top_k=args.top_k,
                max_target_nodes=_first_not_none(args.max_target_nodes, neighbor_cfg.get("max_target_nodes")),
                max_candidates_per_node=int(max_candidates),
                homophilic_topk=int(homo_topk),
                heterophilic_topk=int(hetero_topk),
                topk_chains=int(topk_chains),
                max_chain_length=int(_first_not_none(args.max_chain_length, chain_cfg.get("max_chain_length", 2))),
                min_chain_quality=float(_first_not_none(args.min_chain_quality, chain_cfg.get("min_chain_quality", 0.45))),
                lambda_chain_pos=float(lambda_chain_pos),
                lambda_chain_neg=float(lambda_chain_neg),
                llm_label_file=args.llm_label_file,
                experiment_tag=experiment_tag if args.llm_label_file else None,
                llm_labeler=llm_labeler,
                eval_target_file=args.eval_target_file,
                disable_llm_fallback=bool(args.disable_llm_fallback),
                enable_official_chain=bool(args.enable_official_chain),
                device=args.device,
            )
            metrics_by_model[model_name] = metrics
            print(metrics)
        _warn_if_variants_identical(metrics_by_model)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _is_official_dataset(dataset: str, data_dir: Path) -> bool:
    report_path = data_dir / "preprocess_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text(encoding="utf-8")).get("label_source") == "official"
        except json.JSONDecodeError:
            return False
    return dataset in {"fraud_yelp_official", "fraud_amazon_official"}


def _infer_experiment_tag(label_file: str | Path) -> str:
    stem = Path(label_file).stem
    if stem.startswith("llm_labels_"):
        stem = stem[len("llm_labels_") :]
    return stem or "external_llm"


def _infer_llm_labeler(label_file: str | Path | None) -> str:
    if label_file is None:
        return "mock"
    stem = Path(label_file).stem.lower()
    if "qwen" in stem:
        return "local_qwen"
    if "openai" in stem:
        return "openai"
    if "mock" in stem:
        return "mock"
    return "external"


def _warn_if_variants_identical(metrics_by_model: dict) -> None:
    left = metrics_by_model.get("hero_wo_chain")
    right = metrics_by_model.get("hero_wo_hetero")
    if not left or not right:
        return
    keys = [
        "macro_f1",
        "auroc",
        "auprc",
        "pred_positive_rate",
        "best_threshold",
        "mean_pred_prob_pos",
        "mean_pred_prob_neg",
    ]
    if all(left.get(key) == right.get(key) for key in keys):
        print("[WARNING] hero_wo_chain and hero_wo_hetero have identical metrics. Check variant implementation.")
        branch_keys = [
            "branch_mask_target",
            "branch_mask_homo",
            "branch_mask_hetero",
            "branch_mask_mechanism",
            "branch_mask_chain",
        ]
        diagnostic_keys = [
            "hetero_repr_norm",
            "delta_zero_hetero",
            "final_repr_norm",
        ]
        masks_differ = any(left.get(key) != right.get(key) for key in branch_keys)
        diagnostics_differ = any(left.get(key) != right.get(key) for key in diagnostic_keys)
        if masks_differ:
            print("[ERROR] hero_wo_chain and hero_wo_hetero have identical predictions despite different branch masks. Hetero branch may not affect classifier.")
            print(
                "[DEBUG] "
                f"branch_masks_chain={[left.get(key) for key in branch_keys]} "
                f"branch_masks_hetero={[right.get(key) for key in branch_keys]} "
                f"diagnostics_differ={diagnostics_differ}"
            )


if __name__ == "__main__":
    main()
