from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import (  # noqa: E402
    TEXT_RICH_DATASETS,
    _submission_metric_payload,
    processed_ready,
    resolve_processed_dir,
    write_skip,
)
from src.training.trainer import train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


ABLATION_VARIANTS = {
    "hero_gnn": {"trainer_model": "hero_gnn", "name": "HERO-GNN"},
    "wo_risk_relevant_heterophily": {"trainer_model": "hero_wo_hetero", "name": "w/o Risk-relevant Heterophily"},
    "wo_mechanism_annotation": {"trainer_model": "hero_wo_mechanism", "name": "w/o Mechanism Annotation"},
    "wo_evidence_chain": {"trainer_model": "hero_wo_chain", "name": "w/o Evidence Chain"},
    "wo_llm_annotation": {"skip_reason": "separate no-LLM ablation switch is not implemented yet"},
    "wo_heterophily_filter": {"skip_reason": "separate heterophily-filter ablation switch is not implemented yet"},
    "wo_dual_branch_encoder": {"skip_reason": "separate dual-branch encoder ablation switch is not implemented yet"},
    "wo_gated_fusion": {"skip_reason": "separate gated-fusion ablation switch is not implemented yet"},
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
                if "skip_reason" in spec:
                    write_skip(result_dir, dataset, variant, seed, str(spec["skip_reason"]))
                    print(f"[skipped] {dataset}/{variant}/seed_{seed}: {spec['skip_reason']}")
                    continue
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
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                payload = _submission_metric_payload(metrics, dataset, variant, seed, "project")
                payload["ablation_name"] = str(spec["name"])
                payload["trainer_model"] = str(spec["trainer_model"])
                write_json(result_dir / "metrics.json", payload)
                prediction_file = metrics.get("predictions_file")
                if prediction_file and Path(str(prediction_file)).exists():
                    shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
                (result_dir / "run.log").write_text(f"Completed ablation {dataset}/{variant}/seed_{seed}\n", encoding="utf-8")
                print(f"[ok] {result_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
