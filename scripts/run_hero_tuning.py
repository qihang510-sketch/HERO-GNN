from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import normalize_dataset_name, normalize_model_name, resolve_processed_dir  # noqa: E402
from src.training.trainer import HERO_CONFIG_KEYS, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


DEFAULT_SEARCH_SPACE = {
    "hero_gnn": {
        "fusion_type": ["gated", "concat_linear", "mean", "fixed_sum"],
        "use_gated_fusion": [True, False],
        "risk_relevance_threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
        "neighbor_budget": [5, 10, 15, 20, 30],
        "hetero_branch_weight": [0.2, 0.4, 0.6, 0.8, 1.0],
        "heterophily_weight_mode": ["relevance_confidence", "softmax", "uniform"],
        "mechanism_loss_weight": [0.0, 0.1, 0.3, 0.5],
        "chain_loss_weight": [0.0, 0.1, 0.3, 0.5],
        "routing_loss_weight": [0.0, 0.1, 0.3],
        "learning_rate": [0.001, 0.0005, 0.0001],
        "dropout": [0.2, 0.3, 0.5],
        "weight_decay": [0.0, 0.00001, 0.0001],
        "class_weight_strategy": ["none", "inverse_frequency", "effective_number"],
    },
    "hero_official": {
        "official_hetero_score": [
            "feature_deviation_only",
            "relation_aware",
            "feature_relation_degree",
            "feature_relation_degree_rarity",
        ],
        "risk_relevance_threshold": [0.2, 0.3, 0.4, 0.5],
        "hetero_branch_weight": [0.2, 0.4, 0.6, 0.8],
        "fusion_type": ["concat_linear", "mean", "gated"],
        "class_weight_strategy": ["none", "inverse_frequency", "effective_number"],
        "learning_rate": [0.001, 0.0005, 0.0001],
        "dropout": [0.2, 0.3, 0.5],
    },
}

FOCUSED_HERO_GNN_TRIALS = [
    {"fusion_type": "concat_linear", "use_gated_fusion": False},
    {"fusion_type": "mean", "use_gated_fusion": False},
    {"fusion_type": "gated", "use_gated_fusion": True},
    {"risk_relevance_threshold": 0.2},
    {"risk_relevance_threshold": 0.3},
    {"heterophily_weight_mode": "softmax"},
    {"heterophily_weight_mode": "uniform"},
    {"heterophily_weight_mode": "relevance_confidence"},
    {"hetero_branch_weight": 0.2},
    {"hetero_branch_weight": 0.4},
    {"hetero_branch_weight": 0.6},
    {"hetero_branch_weight": 0.8},
]

METRIC_MAP = {
    "auprc": "AUPRC",
    "ap": "AUPRC",
    "macro_f1": "Macro-F1",
    "macro-f1": "Macro-F1",
    "f1": "Macro-F1",
    "auroc": "AUROC",
}


@dataclass
class TrialConfig:
    trial_id: int
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search tuning for HERO-GNN/HERO-official.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max_trials", type=int, default=40)
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--search_space", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=None, help="Run one fixed config instead of random search.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = normalize_model_name(args.model)
    if model not in {"hero_gnn", "hero_official"}:
        raise ValueError("--model must be hero_gnn or hero_official")
    datasets = [normalize_dataset_name(name) for name in args.datasets]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = _load_search_space(args.search_space, model)
    fixed_config_source = args.config or ""
    trials = _fixed_config_trials(args.config) if args.config else _random_trials(search_space, model, args.max_trials)
    rows = []
    for trial in trials:
        trial_metrics = []
        for dataset in datasets:
            data_dir = resolve_processed_dir(dataset, args.data_root)
            if not data_dir.exists():
                raise FileNotFoundError(f"Missing processed dataset: {data_dir}")
            for seed in args.seeds:
                metrics = _run_one(
                    args=args,
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    data_dir=data_dir,
                    trial=trial,
                    config_source=fixed_config_source,
                )
                trial_metrics.append(metrics)
        rows.append(_aggregate_trial(trial, model, datasets, args.seeds, trial_metrics, fixed_config_source))
    frame = pd.DataFrame(rows)
    frame = _rank_results(frame, args.metric)
    frame.to_csv(output_dir / "tuning_results.csv", index=False)
    (output_dir / "tuning_results.md").write_text(_to_markdown(frame), encoding="utf-8")
    top_rows = [_json_ready(row) for row in frame.head(max(1, int(args.top_k))).to_dict(orient="records")]
    write_json(output_dir / "top_configs.json", top_rows)
    _write_report(output_dir, frame, args, metric_col=_metric_column(args.metric), fixed_config_source=fixed_config_source)
    print(f"HERO tuning finished: trials={len(frame)} output_dir={output_dir}")


def _load_search_space(path: str | None, model: str) -> dict[str, list[Any]]:
    if path:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    else:
        default_path = Path("configs") / "tuning" / ("hero_official.yaml" if model == "hero_official" else "hero_gnn_text_rich.yaml")
        payload = yaml.safe_load(default_path.read_text(encoding="utf-8")) if default_path.exists() else {}
    if "search_space" in payload:
        payload = payload["search_space"]
    if model in payload and isinstance(payload[model], dict):
        payload = payload[model]
    return {**DEFAULT_SEARCH_SPACE[model], **dict(payload or {})}


def _fixed_config_trials(path: str | None) -> list[TrialConfig]:
    if path is None:
        return []
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    params = dict(payload.get("hero_config", payload))
    for key in ["learning_rate", "hidden_dim", "top_k", "epochs"]:
        if key in payload:
            params[key] = payload[key]
    params["config_source"] = str(path)
    return [TrialConfig(trial_id=0, params=params)]


def _random_trials(search_space: dict[str, list[Any]], model: str, max_trials: int) -> list[TrialConfig]:
    rng = random.Random(0)
    trials: list[dict[str, Any]] = []
    focused = FOCUSED_HERO_GNN_TRIALS if model == "hero_gnn" else []
    for overrides in focused[: max(0, int(max_trials))]:
        base = _sample(search_space, rng)
        base.update(overrides)
        _repair_hero_switches(base)
        trials.append(base)
    while len(trials) < int(max_trials):
        params = _sample(search_space, rng)
        _repair_hero_switches(params)
        trials.append(params)
    return [TrialConfig(trial_id=index, params=params) for index, params in enumerate(trials)]


def _sample(search_space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(list(values)) for key, values in search_space.items() if isinstance(values, list) and values}


def _repair_hero_switches(params: dict[str, Any]) -> None:
    fusion_type = str(params.get("fusion_type", "gated"))
    params["use_gated_fusion"] = bool(fusion_type == "gated" and params.get("use_gated_fusion", True))
    if not params["use_gated_fusion"] and fusion_type == "gated":
        params["fusion_type"] = "concat_linear"
    if str(params.get("heterophily_weight_mode", "")) == "uniform":
        params["use_heterophily_filter"] = False
    params.setdefault("use_risk_relevant_heterophily", True)
    params.setdefault("use_mechanism_annotation", True)
    params.setdefault("use_evidence_chain", True)
    params.setdefault("use_llm_annotation", True)
    params.setdefault("use_dual_branch_encoder", True)
    params.setdefault("encoder_type", "dual_branch")
    params.setdefault("labeler_source", "llm_or_mock")


def _run_one(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    seed: int,
    data_dir: Path,
    trial: TrialConfig,
    config_source: str,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    trial_dir = output_dir / dataset / f"trial_{trial.trial_id}"
    seed_dir = trial_dir / f"seed_{seed}"
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists() and args.resume and not args.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    hero_config = {key: value for key, value in trial.params.items() if key in HERO_CONFIG_KEYS}
    lr = float(trial.params.get("learning_rate", 0.001))
    neighbor_budget = int(trial.params.get("neighbor_budget", args.top_k))
    trainer_dataset = {"fraud_yelp": "fraud_yelp_official", "fraud_amazon": "fraud_amazon_official"}.get(dataset, dataset)
    internal_root = trial_dir / "_trainer"
    metrics = train_single_experiment(
        dataset=trainer_dataset,
        model_name=model,
        seed=seed,
        data_dir=data_dir,
        output_root=internal_root,
        epochs=int(args.epochs),
        lr=lr,
        hidden_dim=int(args.hidden_dim),
        top_k=neighbor_budget,
        heterophilic_topk=neighbor_budget,
        max_candidates_per_node=max(20, neighbor_budget),
        min_chain_quality=float(trial.params.get("risk_relevance_threshold", 0.45)),
        lambda_chain_pos=float(trial.params.get("chain_loss_weight", 0.03)),
        lambda_chain_neg=float(trial.params.get("routing_loss_weight", 0.01)),
        device=args.device,
        hero_config=hero_config,
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)
    config_payload = {
        "model": model,
        "dataset": dataset,
        "trial_id": int(trial.trial_id),
        "seed": int(seed),
        "config_source": config_source,
        "hero_config": hero_config,
        "trainer": {
            "epochs": int(args.epochs),
            "learning_rate": lr,
            "hidden_dim": int(args.hidden_dim),
            "top_k": neighbor_budget,
        },
    }
    (trial_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    _write_trial_log(trial_dir, dataset, model, trial, seed, metrics)
    _copy_trainer_log(internal_root, trainer_dataset, model, seed, seed_dir)
    return metrics


def _copy_trainer_log(internal_root: Path, trainer_dataset: str, model: str, seed: int, seed_dir: Path) -> None:
    source = internal_root / "logs" / trainer_dataset / model / f"seed_{seed}.log"
    if source.exists():
        shutil.copyfile(source, seed_dir / "run.log")


def _write_trial_log(trial_dir: Path, dataset: str, model: str, trial: TrialConfig, seed: int, metrics: dict[str, Any]) -> None:
    lines = [
        f"dataset={dataset}",
        f"model={model}",
        f"trial={trial.trial_id}",
        f"seed={seed}",
        f"AUPRC={_metric_value(metrics, 'AUPRC')}",
        f"Macro-F1={_metric_value(metrics, 'Macro-F1')}",
        f"AUROC={_metric_value(metrics, 'AUROC')}",
    ]
    (trial_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_trial(
    trial: TrialConfig,
    model: str,
    datasets: list[str],
    seeds: list[int],
    metrics_rows: list[dict[str, Any]],
    config_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_id": int(trial.trial_id),
        "model": model,
        "datasets": " ".join(datasets),
        "seeds": " ".join(str(seed) for seed in seeds),
        "seed_count": int(len(seeds)),
        "seed0_only": bool(set(seeds) == {0}),
        "config_source": config_source,
        "config_json": json.dumps(trial.params, sort_keys=True),
    }
    for metric in ["AUPRC", "Macro-F1", "AUROC", "Accuracy", "Precision", "Recall"]:
        values = [_metric_value(metrics, metric) for metrics in metrics_rows]
        values = [value for value in values if value is not None]
        row[f"{metric}_mean"] = float(mean(values)) if values else float("nan")
        row[f"{metric}_std"] = float(stdev(values)) if len(values) > 1 else 0.0
    return row


def _rank_results(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    primary = _metric_column(metric)
    for column in [primary, "Macro-F1_mean", "AUROC_mean"]:
        if column not in frame:
            frame[column] = float("nan")
    return frame.sort_values([primary, "Macro-F1_mean", "AUROC_mean"], ascending=False, na_position="last").reset_index(drop=True)


def _metric_column(metric: str) -> str:
    return f"{METRIC_MAP.get(str(metric).lower(), 'AUPRC')}_mean"


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    candidates = [metric, metric.lower(), metric.replace("-", "_"), metric.replace("-", "_").lower()]
    if metric == "Macro-F1":
        candidates += ["macro_f1", "f1_macro", "macro-f1"]
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def _write_report(output_dir: Path, frame: pd.DataFrame, args: argparse.Namespace, metric_col: str, fixed_config_source: str) -> None:
    top = frame.iloc[0].to_dict() if not frame.empty else {}
    lines = [
        "# HERO tuning report",
        "",
        f"- model: {normalize_model_name(args.model)}",
        f"- datasets: {' '.join(args.datasets)}",
        f"- seeds: {' '.join(str(seed) for seed in args.seeds)}",
        f"- trials: {len(frame)}",
        f"- ranking: {metric_col}, then Macro-F1_mean, then AUROC_mean",
        "- validation strategy: trainer selects checkpoints by validation AUPRC; no separate val_AUPRC export was found.",
    ]
    if set(args.seeds) == {0}:
        lines.append("- stability note: seed0 only.")
    if fixed_config_source:
        lines.append(f"- fixed config: {fixed_config_source}")
    if top:
        lines += ["", "## Best trial", "", f"- trial_id: {top.get('trial_id')}", f"- {metric_col}: {top.get(metric_col)}"]
    (output_dir / "tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| status | warning |\n|---|---|\n| empty | no tuning metrics found |\n"
    display = frame.copy()
    for column in display.columns:
        if column.endswith("_std") or column.endswith("_mean"):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
