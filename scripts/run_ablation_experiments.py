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
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


ABLATION_VARIANTS = {
    "hero_gnn": {
        "trainer_model": "hero_gnn",
        "name": "HERO-GNN",
        "hero_config": {},
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
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        for seed in args.seeds:
            for variant, spec in ABLATION_VARIANTS.items():
                result_dir = output_dir / dataset / variant / f"seed_{seed}"
                if (result_dir / "metrics.json").exists() and not args.overwrite:
                    print(f"[exists] {result_dir}")
                    continue
                if not processed_ready(data_dir):
                    write_skip(result_dir, dataset, variant, seed, "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.")
                    print(f"[skipped] {dataset}/{variant}/seed_{seed}: missing data")
                    continue
                resolved_config = _resolve_hero_config(str(spec["trainer_model"]), dict(spec.get("hero_config", {})))
                metrics = train_single_experiment(
                    dataset=dataset,
                    model_name=str(spec["trainer_model"]),
                    seed=seed,
                    data_dir=data_dir,
                    output_root=output_dir / "_project_runs",
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    device=args.device,
                    hero_config=resolved_config,
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                payload = _submission_metric_payload(metrics, dataset, variant, seed, "project")
                payload["variant"] = variant
                payload["ablation_name"] = str(spec["name"])
                payload["trainer_model"] = str(spec["trainer_model"])
                payload["hero_config"] = resolved_config
                stale_skip = result_dir / "skip_reason.json"
                if stale_skip.exists():
                    stale_skip.unlink()
                write_json(result_dir / "metrics.json", payload)
                _write_ablation_config(result_dir, dataset, variant, seed, spec, args, resolved_config)
                prediction_file = metrics.get("predictions_file")
                if prediction_file and Path(str(prediction_file)).exists():
                    shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
                (result_dir / "run.log").write_text(f"Completed ablation {dataset}/{variant}/seed_{seed}\n", encoding="utf-8")
                print(f"[ok] {result_dir / 'metrics.json'}")


def _write_ablation_config(result_dir: Path, dataset: str, variant: str, seed: int, spec: dict, args: argparse.Namespace, resolved_config: dict) -> None:
    payload = {
        "dataset": dataset,
        "variant": variant,
        "ablation_name": str(spec["name"]),
        "model_name": str(spec["trainer_model"]),
        "seed": int(seed),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "hero_config": resolved_config,
        **resolved_config,
    }
    (result_dir / "config_resolved.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
