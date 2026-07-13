from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import (  # noqa: E402
    TEXT_RICH_DATASETS,
    _submission_metric_payload,
    processed_ready,
    resolve_processed_dir,
    write_skip,
)
from src.training.trainer import HERO_CONFIG_KEYS, _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


DEFAULT_ABLATION_VARIANTS = [
    "hero_full",
    "hero_no_llm",
    "hero_no_mechanism",
    "hero_no_risk_weighting",
    "hero_no_relation_loss",
    "hero_no_chain_consistency",
    "hero_structure_only",
    "hero_semantic_only",
    "hero_no_dual_branch",
]

ABLATION_VARIANTS = {
    "hero_full": {
        "trainer_model": "hero_gnn",
        "name": "HERO",
        "hero_config": {},
    },
    "hero_gnn": {
        "trainer_model": "hero_gnn",
        "name": "HERO",
        "hero_config": {},
    },
    "hero_no_llm": {
        "trainer_model": "hero_gnn",
        "name": "w/o LLM Annotation",
        "hero_config": {"use_llm_annotation": False, "labeler_source": "rule_or_structure"},
    },
    "hero_no_mechanism": {
        "trainer_model": "hero_gnn",
        "name": "w/o Mechanism Annotation",
        "hero_config": {"use_mechanism_annotation": False, "use_llm_annotation": False},
    },
    "hero_no_risk_weighting": {
        "trainer_model": "hero_gnn",
        "name": "w/o Risk-aware Weighting",
        "hero_config": {"use_heterophily_filter": False, "heterophily_weight_mode": "uniform"},
    },
    "hero_no_relation_loss": {
        "trainer_model": "hero_gnn",
        "name": "w/o Relation Alignment Loss",
        "hero_config": {},
        "status": "unsupported",
        "fallback": "No explicit relation alignment loss key exists; run equals hero_full unless relation loss is implemented.",
    },
    "hero_no_chain_consistency": {
        "trainer_model": "hero_gnn",
        "name": "w/o Evidence-chain Consistency",
        "hero_config": {"chain_loss_weight": 0.0, "routing_loss_weight": 0.0},
    },
    "hero_structure_only": {
        "trainer_model": "hero_gnn",
        "name": "Structure Only",
        "hero_config": {
            "use_risk_relevant_heterophily": False,
            "use_mechanism_annotation": False,
            "use_evidence_chain": False,
            "use_llm_annotation": False,
            "labeler_source": "structure",
        },
    },
    "hero_semantic_only": {
        "trainer_model": "hero_gnn",
        "name": "Semantic/Mechanism Only",
        "hero_config": {
            "use_mechanism_annotation": True,
            "use_llm_annotation": True,
            "use_evidence_chain": True,
            "heterophily_weight_mode": "relevance_confidence",
        },
        "fallback": "Backbone target/homophilic encoders remain present in the current HEROGNN implementation.",
    },
    "hero_no_dual_branch": {
        "trainer_model": "hero_gnn",
        "name": "w/o Dual-Branch Encoder",
        "hero_config": {"use_dual_branch_encoder": False, "encoder_type": "single_branch"},
    },
    "wo_risk_relevant_heterophily": {
        "trainer_model": "hero_gnn",
        "name": "w/o Risk-relevant Heterophily",
        "hero_config": {"use_risk_relevant_heterophily": False},
    },
    "wo_mechanism_annotation": {
        "trainer_model": "hero_gnn",
        "name": "w/o Mechanism Annotation",
        "hero_config": {"use_mechanism_annotation": False, "use_llm_annotation": False},
    },
    "wo_evidence_chain": {
        "trainer_model": "hero_gnn",
        "name": "w/o Evidence Chain",
        "hero_config": {"use_evidence_chain": False},
    },
    "wo_llm_annotation": {
        "trainer_model": "hero_gnn",
        "name": "w/o LLM Annotation",
        "hero_config": {"use_llm_annotation": False, "labeler_source": "rule_or_structure"},
    },
    "wo_heterophily_filter": {
        "trainer_model": "hero_gnn",
        "name": "w/o Heterophily Filter",
        "hero_config": {"use_heterophily_filter": False, "heterophily_weight_mode": "uniform"},
    },
    "wo_dual_branch_encoder": {
        "trainer_model": "hero_gnn",
        "name": "w/o Dual-Branch Encoder",
        "hero_config": {"use_dual_branch_encoder": False, "encoder_type": "single_branch"},
    },
    "wo_gated_fusion": {
        "trainer_model": "hero_gnn",
        "name": "w/o Gated Fusion",
        "hero_config": {"use_gated_fusion": False, "fusion_type": "concat_linear"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HERO-GNN ablation experiments with real training where switches exist.")
    parser.add_argument("--datasets", nargs="+", default=list(TEXT_RICH_DATASETS), choices=list(TEXT_RICH_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default="outputs/submission_experiments_ablation")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_ABLATION_VARIANTS)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--config", default=None, help="Optional HERO-GNN tuned config used as the base for every ablation variant.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    base_hero_config, base_trainer_config, base_config_source = _load_base_config(args.config)
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        for seed in args.seeds:
            for variant in args.variants:
                variant = _normalize_variant_name(variant)
                spec = ABLATION_VARIANTS[variant]
                result_dir = output_dir / dataset / variant / f"seed_{seed}"
                if (result_dir / "metrics.json").exists() and not args.overwrite:
                    print(f"[exists] {result_dir}")
                    continue
                if not processed_ready(data_dir):
                    write_skip(result_dir, dataset, variant, seed, "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.")
                    print(f"[skipped] {dataset}/{variant}/seed_{seed}: missing data")
                    continue
                if spec.get("status") == "unsupported":
                    reason = str(spec.get("fallback", "unsupported_ablation_variant"))
                    write_skip(result_dir, dataset, variant, seed, reason)
                    print(f"[unsupported] {dataset}/{variant}/seed_{seed}: {reason}")
                    continue
                variant_overrides = dict(spec.get("hero_config", {}))
                merged_config = {**base_hero_config, **variant_overrides}
                resolved_config = _resolve_hero_config(str(spec["trainer_model"]), merged_config)
                warnings = _variant_warnings(variant, base_hero_config, resolved_config)
                trainer_params = _resolve_trainer_params(args, base_trainer_config, resolved_config)
                metrics = train_single_experiment(
                    dataset=dataset,
                    model_name=str(spec["trainer_model"]),
                    seed=seed,
                    data_dir=data_dir,
                    output_root=output_dir / "_project_runs",
                    epochs=trainer_params["epochs"],
                    lr=trainer_params["lr"],
                    hidden_dim=trainer_params["hidden_dim"],
                    top_k=trainer_params["top_k"],
                    device=args.device,
                    hero_config=resolved_config,
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                payload = _submission_metric_payload(metrics, dataset, variant, seed, "project")
                payload["variant"] = variant
                payload["ablation_name"] = str(spec["name"])
                payload["trainer_model"] = str(spec["trainer_model"])
                payload["hero_config"] = resolved_config
                payload["base_config"] = base_config_source
                if spec.get("status"):
                    payload["variant_status"] = str(spec.get("status"))
                if spec.get("fallback"):
                    payload["fallback"] = str(spec.get("fallback"))
                if warnings:
                    payload["warning"] = "; ".join(warnings)
                stale_skip = result_dir / "skip_reason.json"
                if stale_skip.exists():
                    stale_skip.unlink()
                write_json(result_dir / "metrics.json", payload)
                _write_ablation_config(
                    result_dir=result_dir,
                    dataset=dataset,
                    variant=variant,
                    seed=seed,
                    spec=spec,
                    resolved_config=resolved_config,
                    base_config_source=base_config_source,
                    base_hero_config=base_hero_config,
                    variant_overrides=variant_overrides,
                    trainer_params=trainer_params,
                    warnings=warnings,
                )
                prediction_file = metrics.get("predictions_file")
                if prediction_file and Path(str(prediction_file)).exists():
                    shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
                (result_dir / "run.log").write_text(f"Completed ablation {dataset}/{variant}/seed_{seed}\n", encoding="utf-8")
                print(f"[ok] {result_dir / 'metrics.json'}")


def _load_base_config(config_path: str | None) -> tuple[dict, dict, str]:
    if config_path is None:
        return {}, {}, ""
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(payload.get("hero_config"), dict):
        hero_source = payload.get("hero_config")
    elif isinstance(payload.get("hero"), dict):
        hero_source = payload.get("hero")
    else:
        hero_source = payload
    hero_config = {key: value for key, value in dict(hero_source).items() if key in HERO_CONFIG_KEYS}
    trainer_config: dict = {}
    for source in (payload, hero_source):
        if not isinstance(source, dict):
            continue
        for source_key, target_key in [
            ("learning_rate", "lr"),
            ("lr", "lr"),
            ("epochs", "epochs"),
            ("hidden_dim", "hidden_dim"),
            ("top_k", "top_k"),
        ]:
            if source_key in source:
                trainer_config[target_key] = source[source_key]
    if "neighbor_budget" in hero_config and "top_k" not in trainer_config:
        trainer_config["top_k"] = hero_config["neighbor_budget"]
    return hero_config, trainer_config, str(path)


def _normalize_variant_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "_")
    aliases = {
        "hero": "hero_full",
        "full": "hero_full",
        "full_hero": "hero_full",
        "wo_llm_annotation": "hero_no_llm",
        "wo_mechanism_annotation": "hero_no_mechanism",
        "wo_risk_aware_weighting": "hero_no_risk_weighting",
        "wo_risk_weighting": "hero_no_risk_weighting",
        "wo_relation_alignment_loss": "hero_no_relation_loss",
        "wo_chain_consistency": "hero_no_chain_consistency",
        "wo_evidence_chain_consistency": "hero_no_chain_consistency",
        "wo_evidence_chain": "hero_no_chain_consistency",
        "wo_dual_branch_encoder": "hero_no_dual_branch",
        "wo_dual_branch": "hero_no_dual_branch",
        "no_dual_branch": "hero_no_dual_branch",
        "wo_hetero": "wo_risk_relevant_heterophily",
        "wo_mechanism": "hero_no_mechanism",
        "wo_chain": "hero_no_chain_consistency",
    }
    canonical = aliases.get(text, text)
    if canonical not in ABLATION_VARIANTS:
        raise ValueError(f"unknown_ablation_variant:{name}")
    return canonical


def _resolve_trainer_params(args: argparse.Namespace, base_trainer_config: dict, resolved_config: dict) -> dict:
    top_k_default = int(resolved_config.get("neighbor_budget", 10))
    return {
        "epochs": int(args.epochs if args.epochs is not None else base_trainer_config.get("epochs", 50)),
        "lr": float(args.lr if args.lr is not None else base_trainer_config.get("lr", 0.001)),
        "hidden_dim": int(args.hidden_dim if args.hidden_dim is not None else base_trainer_config.get("hidden_dim", 64)),
        "top_k": int(args.top_k if args.top_k is not None else base_trainer_config.get("top_k", top_k_default)),
    }


def _variant_warnings(variant: str, base_hero_config: dict, resolved_config: dict) -> list[str]:
    warnings = []
    if variant == "wo_gated_fusion" and base_hero_config:
        base_gated = bool(base_hero_config.get("use_gated_fusion", True))
        base_fusion_type = str(base_hero_config.get("fusion_type", "gated"))
        if not base_gated or base_fusion_type != "gated":
            warnings.append("base config already disables gated fusion")
    if variant == "wo_heterophily_filter" and base_hero_config and not bool(base_hero_config.get("use_heterophily_filter", True)):
        warnings.append("base config already disables heterophily filter")
    if variant == "wo_dual_branch_encoder" and base_hero_config and not bool(base_hero_config.get("use_dual_branch_encoder", True)):
        warnings.append("base config already disables dual-branch encoder")
    return warnings


def _write_ablation_config(
    result_dir: Path,
    dataset: str,
    variant: str,
    seed: int,
    spec: dict,
    resolved_config: dict,
    base_config_source: str,
    base_hero_config: dict,
    variant_overrides: dict,
    trainer_params: dict,
    warnings: list[str],
) -> None:
    payload = {
        "dataset": dataset,
        "base_config": base_config_source,
        "variant": variant,
        "ablation_name": str(spec["name"]),
        "model_name": str(spec["trainer_model"]),
        "seed": int(seed),
        "epochs": int(trainer_params["epochs"]),
        "lr": float(trainer_params["lr"]),
        "hidden_dim": int(trainer_params["hidden_dim"]),
        "top_k": int(trainer_params["top_k"]),
        "base_hero_config": base_hero_config,
        "variant_overrides": variant_overrides,
        "hero_config": resolved_config,
        "warning": "; ".join(warnings),
        **resolved_config,
    }
    (result_dir / "config_resolved.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
