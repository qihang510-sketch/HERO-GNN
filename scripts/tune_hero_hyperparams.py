from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.submission import hero_base_model_for_dataset, processed_ready, resolve_processed_dir, write_skip  # noqa: E402
from src.training.trainer import HERO_CONFIG_KEYS, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


DEFAULT_DATASETS = ["yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
VAL_METRICS = ["val_AUPRC", "val_AUROC", "val_Macro-F1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune full HERO with validation metrics only; never select by test metrics.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_trials", type=int, default=12)
    parser.add_argument("--search_stage", choices=["coarse", "fine", "all"], default="coarse")
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_hero(args)


def tune_hero(args: argparse.Namespace) -> pd.DataFrame:
    if args.quick_test:
        args.datasets = ["yelp_academic"]
        args.seeds = [0]
        args.max_trials = min(int(args.max_trials), 1)
        args.epochs = min(int(args.epochs), 1)
    output_dir = Path(args.output_dir or f"outputs/tuning_hero_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    candidates = _candidate_configs(limit=int(args.max_trials), stage=str(args.search_stage))
    if args.dry_run:
        for dataset in args.datasets:
            for trial_id, candidate in enumerate(candidates):
                print(f"[dry-run] tune dataset={dataset} trial={trial_id} seeds={' '.join(map(str, args.seeds))} config={candidate}")
        return pd.DataFrame()
    for child in ["raw", "summary", "best_configs", "logs"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        trainer_model = hero_base_model_for_dataset(dataset, "hero_full")
        for trial_id, candidate in enumerate(candidates):
            for seed in args.seeds:
                rows.append(_run_trial(args, output_dir, data_dir, dataset, trainer_model, int(seed), trial_id, candidate))
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "summary" / "tuning_results.csv", index=False)
    best = _best_configs(results, candidates)
    best.to_csv(output_dir / "summary" / "best_hero_configs.csv", index=False)
    _write_best_config_yamls(output_dir, best)
    return results


def _candidate_configs(limit: int, stage: str = "coarse") -> list[dict[str, Any]]:
    if stage == "all":
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in _candidate_configs(limit=max(limit, 1), stage="coarse") + _candidate_configs(limit=max(limit, 1), stage="fine"):
            key = _config_hash(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
            if len(merged) >= limit:
                break
        return merged
    grid = {
        "learning_rate": [0.001, 0.003, 0.005],
        "weight_decay": [0.0, 1e-5, 5e-5, 1e-4],
        "hidden_dim": [64, 128, 256],
        "dropout": [0.2, 0.4, 0.5],
        "num_layers": [2, 3],
        "neighbor_k": [5, 10, 15, 20],
        "lambda_rel": [0.0, 0.1, 0.3, 0.5, 1.0],
        "lambda_chain": [0.0, 0.1, 0.3, 0.5],
        "confidence_threshold": [0.0, 0.3, 0.5, 0.7],
        "risk_weight_temperature": [0.5, 1.0, 2.0],
        "optimizer": ["adamw", "adam"],
        "scheduler": ["reduce_on_plateau", "cosine", "none"],
        "early_stopping_patience": [20, 30, 50],
        "use_class_weight": [True, False],
    }
    # Deterministic compact search: vary one axis at a time around a conservative center.
    base = {
        "learning_rate": 0.001,
        "weight_decay": 5e-5,
        "hidden_dim": 128,
        "dropout": 0.4,
        "num_layers": 2,
        "neighbor_k": 10,
        "lambda_rel": 0.3,
        "lambda_chain": 0.1,
        "confidence_threshold": 0.3,
        "risk_weight_temperature": 1.0,
        "optimizer": "adamw",
        "scheduler": "reduce_on_plateau",
        "early_stopping_patience": 30,
        "use_class_weight": True,
    }
    if stage == "fine":
        grid.update(
            {
                "learning_rate": [0.0007, 0.001, 0.0015, 0.003],
                "weight_decay": [1e-5, 5e-5, 1e-4],
                "dropout": [0.3, 0.4, 0.5],
                "neighbor_k": [8, 10, 12, 15],
                "lambda_rel": [0.1, 0.3, 0.5],
                "lambda_chain": [0.05, 0.1, 0.2, 0.3],
                "confidence_threshold": [0.0, 0.2, 0.3, 0.5],
            }
        )
    candidates = [dict(base)]
    if limit <= 1:
        return candidates[:1]
    for key, values in grid.items():
        for value in values:
            if value == base[key]:
                continue
            candidate = dict(base)
            candidate[key] = value
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    return candidates[:limit]


def _run_trial(
    args: argparse.Namespace,
    output_dir: Path,
    data_dir: Path,
    dataset: str,
    trainer_model: str,
    seed: int,
    trial_id: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    run_dir = output_dir / "raw" / dataset / f"trial_{trial_id:03d}" / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    row = {
        "dataset": dataset,
        "model": "hero_full",
        "trainer_model": trainer_model,
        "trial_id": trial_id,
        "seed": seed,
        "status": "ok",
        "config_hash": _config_hash(candidate),
        **candidate,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {"selection_rule": "validation_AUPRC_then_AUROC_then_Macro-F1", "candidate": candidate})
    if not processed_ready(data_dir):
        reason = "missing_data"
        write_skip(run_dir, dataset, "hero_full", seed, reason)
        row.update({"status": "unavailable", "reason": reason})
    else:
        try:
            hero_config = _hero_config(candidate)
            metrics = train_single_experiment(
                dataset=dataset,
                model_name=trainer_model,
                seed=seed,
                data_dir=data_dir,
                output_root=output_dir / "raw" / "_project_runs_tuning",
                epochs=int(args.epochs),
                lr=float(candidate["learning_rate"]),
                hidden_dim=int(candidate["hidden_dim"]),
                top_k=int(candidate["neighbor_k"]),
                device=args.device,
                hero_config=hero_config,
            )
            for metric in VAL_METRICS:
                row[metric] = metrics.get(metric)
            row["best_epoch"] = metrics.get("best_epoch")
            row["best_val_AUPRC"] = metrics.get("best_val_AUPRC", row.get("val_AUPRC"))
            row["best_val_AUROC"] = metrics.get("best_val_AUROC", row.get("val_AUROC"))
            row["best_val_Macro-F1"] = metrics.get("best_val_Macro-F1", row.get("val_Macro-F1"))
            if all(pd.isna(row.get(metric)) for metric in VAL_METRICS):
                row["status"] = "missing_validation_metric"
                row["reason"] = "trainer did not return validation metrics; no test metric was used for selection"
        except Exception as exc:
            row.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            (run_dir / "error.log").write_text(str(row["reason"]) + "\n", encoding="utf-8")
    row["duration_sec"] = time.perf_counter() - started
    write_json(metrics_path, row)
    write_json(run_dir / "runtime.json", {"duration_sec": row["duration_sec"], "status": row["status"]})
    (run_dir / "log.txt").write_text(str(row.get("reason", "")) + "\n", encoding="utf-8")
    return row


def _hero_config(candidate: dict[str, Any]) -> dict[str, Any]:
    requested = {
        "neighbor_budget": int(candidate["neighbor_k"]),
        "dropout": float(candidate["dropout"]),
        "weight_decay": float(candidate["weight_decay"]),
        "mechanism_loss_weight": float(candidate["lambda_rel"]),
        "chain_loss_weight": float(candidate["lambda_chain"]),
        "routing_loss_weight": float(candidate["lambda_chain"]),
        "confidence_threshold": float(candidate["confidence_threshold"]),
        "risk_weight_temperature": float(candidate["risk_weight_temperature"]),
        "optimizer": str(candidate.get("optimizer", "adamw")),
        "scheduler": str(candidate.get("scheduler", "reduce_on_plateau")),
        "early_stopping_patience": int(candidate.get("early_stopping_patience", 30)),
        "use_class_weight": bool(candidate.get("use_class_weight", True)),
    }
    return {key: value for key, value in requested.items() if key in HERO_CONFIG_KEYS}


def _best_configs(results: pd.DataFrame, candidates: list[dict[str, Any]]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in results.groupby("dataset"):
        ok = group[group["status"].astype(str) == "ok"].copy()
        if ok.empty:
            rows.append({"dataset": dataset, "model": "hero_full", "status": "unavailable", "reason": "no validation results"})
            continue
        agg = ok.groupby("trial_id", as_index=False)[VAL_METRICS].mean(numeric_only=True)
        agg = agg.sort_values(VAL_METRICS, ascending=[False, False, False], na_position="last")
        best_trial = int(agg.iloc[0]["trial_id"])
        candidate = dict(candidates[best_trial])
        best_rows = ok[ok["trial_id"].astype(int) == best_trial]
        best_epoch_values = pd.to_numeric(best_rows.get("best_epoch", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "dataset": dataset,
                "model": "hero_full",
                "status": "ok",
                "selected_by": "validation_AUPRC",
                "selection_metric": "val_AUPRC",
                "trial_id": best_trial,
                "config_hash": _config_hash(candidate),
                "best_epoch": int(round(float(best_epoch_values.mean()))) if not best_epoch_values.empty else pd.NA,
                **{f"{metric}_mean": agg.iloc[0][metric] for metric in VAL_METRICS},
                **candidate,
            }
        )
    return pd.DataFrame(rows)


def _write_best_config_yamls(output_dir: Path, best: pd.DataFrame) -> None:
    for _, row in best.iterrows():
        dataset = str(row.get("dataset", ""))
        payload = {
            "status": row.get("status", "unavailable"),
            "selection": {
                "selected_by": "validation_AUPRC",
                "rule": "validation_AUPRC_then_AUROC_then_Macro-F1",
                "note": "Test metrics are not used for hyperparameter selection.",
                "trial_id": None if pd.isna(row.get("trial_id")) else int(row.get("trial_id")),
                "best_epoch": None if pd.isna(row.get("best_epoch")) else int(row.get("best_epoch")),
                "config_hash": str(row.get("config_hash", "")),
                "val_metrics": {
                    "AUPRC": None if pd.isna(row.get("val_AUPRC_mean")) else float(row.get("val_AUPRC_mean")),
                    "AUROC": None if pd.isna(row.get("val_AUROC_mean")) else float(row.get("val_AUROC_mean")),
                    "Macro-F1": None if pd.isna(row.get("val_Macro-F1_mean")) else float(row.get("val_Macro-F1_mean")),
                },
            },
            "trainer": {
                "lr": None if pd.isna(row.get("learning_rate")) else float(row.get("learning_rate")),
                "hidden_dim": None if pd.isna(row.get("hidden_dim")) else int(row.get("hidden_dim")),
                "top_k": None if pd.isna(row.get("neighbor_k")) else int(row.get("neighbor_k")),
            },
            "unsupported_parameters": {
                "num_layers": None if pd.isna(row.get("num_layers")) else int(row.get("num_layers")),
                "reason": "Current HEROGNN/HEROOfficial constructors do not expose num_layers; value is recorded but not applied.",
            },
            "hero": _hero_config(row.to_dict()) if row.get("status") == "ok" else {},
        }
        (output_dir / "best_configs" / f"hero_full_{dataset}.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_hash(candidate: dict[str, Any]) -> str:
    text = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
