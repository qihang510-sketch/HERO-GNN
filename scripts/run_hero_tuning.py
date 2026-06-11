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

SAFE_HERO_GNN_VARIANTS = [
    ("only_disable_gated_fusion", {"use_gated_fusion": False, "fusion_type": "concat_linear"}),
    ("only_low_threshold", {"risk_relevance_threshold": 0.2}),
    ("only_softmax_heterophily", {"heterophily_weight_mode": "softmax"}),
    ("only_uniform_heterophily", {"heterophily_weight_mode": "uniform"}),
    ("lower_hetero_weight", {"hetero_branch_weight": 0.4}),
    ("lower_chain_loss", {"chain_loss_weight": 0.0, "routing_loss_weight": 0.0}),
    ("no_auxiliary_loss", {"mechanism_loss_weight": 0.0, "chain_loss_weight": 0.0, "routing_loss_weight": 0.0}),
    (
        "disable_gated_plus_low_threshold",
        {"use_gated_fusion": False, "fusion_type": "concat_linear", "risk_relevance_threshold": 0.2},
    ),
    (
        "disable_gated_plus_softmax",
        {"use_gated_fusion": False, "fusion_type": "concat_linear", "heterophily_weight_mode": "softmax"},
    ),
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
    trial_id: str
    trial_name: str
    params: dict[str, Any]
    is_baseline: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe random-search tuning for HERO-GNN/HERO-official.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max_trials", type=int, default=40)
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--search_space", default=None)
    parser.add_argument("--search_mode", choices=["safe", "broad", "mixed"], default="safe")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=None, help="Run a fixed config after the baseline trial.")
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
    baseline_params, baseline_source = _baseline_params(model)
    search_space = _load_search_space(args.search_space, model)
    trials = _build_trials(
        model=model,
        search_space=search_space,
        baseline_params=baseline_params,
        max_trials=max(1, int(args.max_trials)),
        search_mode=args.search_mode,
        config_path=args.config,
    )
    rows = []
    for trial in trials:
        metrics_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for dataset in datasets:
            data_dir = resolve_processed_dir(dataset, args.data_root)
            if not data_dir.exists():
                raise FileNotFoundError(f"Missing processed dataset: {data_dir}")
            dataset_metrics = []
            for seed in args.seeds:
                metrics = _run_one(
                    args=args,
                    dataset=dataset,
                    model=model,
                    seed=seed,
                    data_dir=data_dir,
                    trial=trial,
                    config_source=baseline_source if trial.is_baseline else str(args.config or ""),
                )
                dataset_metrics.append(metrics)
            metrics_by_dataset[dataset] = dataset_metrics
        rows.extend(_aggregate_trial_rows(trial, model, datasets, args.seeds, metrics_by_dataset, baseline_source))
    frame = pd.DataFrame(rows)
    ranking = _rank_results(_ranking_rows(frame, datasets), args.metric, joint=len(datasets) > 1)
    ranking = ranking.copy()
    ranking["rank"] = list(range(1, len(ranking) + 1))
    rank_map = {str(row["trial_id"]): int(idx + 1) for idx, row in ranking.iterrows()}
    frame["rank"] = frame["trial_id"].map(lambda value: rank_map.get(str(value), 999999))
    frame = frame.sort_values(["rank", "trial_id", "dataset"], kind="stable").reset_index(drop=True)
    frame.to_csv(output_dir / "tuning_results.csv", index=False)
    (output_dir / "tuning_results.md").write_text(_to_markdown(frame), encoding="utf-8")
    top_rows = _top_configs_with_baseline(ranking, args.top_k)
    write_json(output_dir / "top_configs.json", [_json_ready(row) for row in top_rows])
    _write_report(output_dir, frame, ranking, args, baseline_source)
    print(f"HERO tuning finished: trials={len(trials)} rows={len(frame)} output_dir={output_dir}")


def _baseline_params(model: str) -> tuple[dict[str, Any], str]:
    path = Path("configs") / "models" / ("hero_official.yaml" if model == "hero_official" else "hero_gnn.yaml")
    if not path.exists():
        return {}, "trainer_defaults"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = dict(payload.get("hero_config", payload))
    for key in ["learning_rate", "lr", "hidden_dim", "top_k", "epochs"]:
        if key in payload:
            params[key] = payload[key]
    return params, str(path)


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


def _build_trials(
    model: str,
    search_space: dict[str, list[Any]],
    baseline_params: dict[str, Any],
    max_trials: int,
    search_mode: str,
    config_path: str | None,
) -> list[TrialConfig]:
    baseline = TrialConfig("baseline", "baseline", dict(baseline_params), is_baseline=True)
    if config_path:
        fixed = _fixed_config_trial(config_path)
        return [baseline, fixed] if max_trials > 1 else [baseline]
    if model == "hero_gnn" and search_mode == "safe":
        return [baseline, *_safe_trials(baseline_params)[: max_trials - 1]]
    if model == "hero_gnn" and search_mode == "mixed":
        safe_trials = _safe_trials(baseline_params)
        remaining = max_trials - 1 - len(safe_trials)
        random_trials = _random_trials(search_space, start_index=1 + len(safe_trials), count=max(0, remaining))
        return [baseline, *safe_trials[: max_trials - 1], *random_trials]
    return [baseline, *_random_trials(search_space, start_index=1, count=max_trials - 1)]


def _fixed_config_trial(path: str) -> TrialConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    params = dict(payload.get("hero_config", payload))
    for key in ["learning_rate", "lr", "hidden_dim", "top_k", "epochs"]:
        if key in payload:
            params[key] = payload[key]
    params["config_source"] = str(path)
    _repair_hero_switches(params)
    return TrialConfig("config", "fixed_config", params, is_baseline=False)


def _safe_trials(baseline_params: dict[str, Any]) -> list[TrialConfig]:
    trials = []
    for name, overrides in SAFE_HERO_GNN_VARIANTS:
        params = dict(baseline_params)
        params.update(overrides)
        _repair_hero_switches(params)
        trials.append(TrialConfig(name, name, params, is_baseline=False))
    return trials


def _random_trials(search_space: dict[str, list[Any]], start_index: int, count: int) -> list[TrialConfig]:
    rng = random.Random(0)
    trials = []
    for offset in range(max(0, int(count))):
        trial_id = str(start_index + offset)
        params = _sample(search_space, rng)
        _repair_hero_switches(params)
        trials.append(TrialConfig(trial_id, f"random_{trial_id}", params, is_baseline=False))
    return trials


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
    params.setdefault("use_heterophily_filter", True)
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
    lr = float(trial.params.get("learning_rate", trial.params.get("lr", 0.001)))
    neighbor_budget = int(trial.params.get("neighbor_budget", args.top_k))
    trainer_dataset = {"fraud_yelp": "fraud_yelp_official", "fraud_amazon": "fraud_amazon_official"}.get(dataset, dataset)
    internal_root = trial_dir / "_trainer"
    metrics = train_single_experiment(
        dataset=trainer_dataset,
        model_name=model,
        seed=seed,
        data_dir=data_dir,
        output_root=internal_root,
        epochs=int(trial.params.get("epochs", args.epochs)),
        lr=lr,
        hidden_dim=int(trial.params.get("hidden_dim", args.hidden_dim)),
        top_k=neighbor_budget,
        heterophilic_topk=neighbor_budget,
        max_candidates_per_node=max(20, neighbor_budget),
        min_chain_quality=float(trial.params.get("risk_relevance_threshold", 0.45)),
        lambda_chain_pos=float(trial.params.get("chain_loss_weight", 0.03)),
        lambda_chain_neg=float(trial.params.get("routing_loss_weight", 0.01)),
        device=args.device,
        hero_config=hero_config,
    )
    metrics = {
        **metrics,
        "tuning_trial_id": trial.trial_id,
        "tuning_trial_name": trial.trial_name,
        "is_baseline": bool(trial.is_baseline),
    }
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, metrics)
    config_payload = {
        "model": model,
        "dataset": dataset,
        "trial_id": trial.trial_id,
        "trial_name": trial.trial_name,
        "is_baseline": bool(trial.is_baseline),
        "seed": int(seed),
        "config_source": config_source,
        "hero_config": hero_config,
        "trainer": {
            "epochs": int(trial.params.get("epochs", args.epochs)),
            "learning_rate": lr,
            "hidden_dim": int(trial.params.get("hidden_dim", args.hidden_dim)),
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
        f"trial_name={trial.trial_name}",
        f"is_baseline={trial.is_baseline}",
        f"seed={seed}",
        f"AUPRC={_metric_value(metrics, 'AUPRC')}",
        f"Macro-F1={_metric_value(metrics, 'Macro-F1')}",
        f"AUROC={_metric_value(metrics, 'AUROC')}",
    ]
    (trial_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_trial_rows(
    trial: TrialConfig,
    model: str,
    datasets: list[str],
    seeds: list[int],
    metrics_by_dataset: dict[str, list[dict[str, Any]]],
    baseline_source: str,
) -> list[dict[str, Any]]:
    dataset_summaries = {dataset: _metric_summary(metrics_by_dataset.get(dataset, [])) for dataset in datasets}
    dataset_auprcs = [
        summary.get("AUPRC_mean")
        for summary in dataset_summaries.values()
        if summary.get("AUPRC_mean") is not None and not pd.isna(summary.get("AUPRC_mean"))
    ]
    joint_mean = float(mean(dataset_auprcs)) if dataset_auprcs else float("nan")
    joint_min = float(min(dataset_auprcs)) if dataset_auprcs else float("nan")
    rows = [
        _base_row(trial, model, dataset, seeds, summary, dataset_summaries, joint_mean, joint_min, baseline_source)
        for dataset, summary in dataset_summaries.items()
    ]
    if len(datasets) > 1:
        all_metrics = [metrics for dataset in datasets for metrics in metrics_by_dataset.get(dataset, [])]
        joint_summary = _metric_summary(all_metrics)
        rows.append(_base_row(trial, model, "joint", seeds, joint_summary, dataset_summaries, joint_mean, joint_min, baseline_source))
    return rows


def _base_row(
    trial: TrialConfig,
    model: str,
    dataset: str,
    seeds: list[int],
    summary: dict[str, Any],
    dataset_summaries: dict[str, dict[str, Any]],
    joint_mean: float,
    joint_min: float,
    baseline_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_id": trial.trial_id,
        "trial_name": trial.trial_name,
        "is_baseline": bool(trial.is_baseline),
        "model": model,
        "dataset": dataset,
        "seed_count": int(len(seeds)),
        "seeds": " ".join(str(seed) for seed in seeds),
        "seed0_only": bool(set(seeds) == {0}),
        "Macro-F1_mean": summary.get("Macro-F1_mean", float("nan")),
        "Macro-F1_std": summary.get("Macro-F1_std", 0.0),
        "AUROC_mean": summary.get("AUROC_mean", float("nan")),
        "AUROC_std": summary.get("AUROC_std", 0.0),
        "AUPRC_mean": summary.get("AUPRC_mean", float("nan")),
        "AUPRC_std": summary.get("AUPRC_std", 0.0),
        "mean_AUPRC_across_datasets": joint_mean,
        "min_AUPRC_across_datasets": joint_min,
        "config_source": baseline_source if trial.is_baseline else str(trial.params.get("config_source", "")),
        "config_json": json.dumps(trial.params, sort_keys=True),
        "status": "ok" if summary.get("count", 0) > 0 else "missing",
        "warning": "" if summary.get("count", 0) > 0 else "no metrics found",
    }
    for name, item in dataset_summaries.items():
        row[f"{name}_AUPRC_mean"] = item.get("AUPRC_mean", float("nan"))
        row[f"{name}_Macro-F1_mean"] = item.get("Macro-F1_mean", float("nan"))
        row[f"{name}_AUROC_mean"] = item.get("AUROC_mean", float("nan"))
    return row


def _metric_summary(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(metrics_rows)}
    for metric in ["AUPRC", "Macro-F1", "AUROC", "Accuracy", "Precision", "Recall"]:
        values = [_metric_value(metrics, metric) for metrics in metrics_rows]
        values = [value for value in values if value is not None]
        summary[f"{metric}_mean"] = float(mean(values)) if values else float("nan")
        summary[f"{metric}_std"] = float(stdev(values)) if len(values) > 1 else 0.0
    return summary


def _ranking_rows(frame: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    if len(datasets) > 1 and "joint" in set(frame["dataset"].astype(str)):
        return frame[frame["dataset"].astype(str) == "joint"].copy()
    return frame[frame["dataset"].astype(str) == str(datasets[0])].copy()


def _rank_results(frame: pd.DataFrame, metric: str, joint: bool) -> pd.DataFrame:
    if joint:
        for column in ["mean_AUPRC_across_datasets", "min_AUPRC_across_datasets", "Macro-F1_mean"]:
            if column not in frame:
                frame[column] = float("nan")
        return frame.sort_values(
            ["mean_AUPRC_across_datasets", "min_AUPRC_across_datasets", "Macro-F1_mean"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    primary = _metric_column(metric)
    for column in [primary, "Macro-F1_mean", "AUROC_mean"]:
        if column not in frame:
            frame[column] = float("nan")
    return frame.sort_values([primary, "Macro-F1_mean", "AUROC_mean"], ascending=False, na_position="last").reset_index(drop=True)


def _metric_column(metric: str) -> str:
    return f"{METRIC_MAP.get(str(metric).lower(), 'AUPRC')}_mean"


def _top_configs_with_baseline(ranking: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    top = ranking.head(max(1, int(top_k))).to_dict(orient="records")
    if not any(_is_true(row.get("is_baseline")) for row in top):
        baseline_rows = ranking[ranking["is_baseline"].map(_is_true)]
        if not baseline_rows.empty:
            top.append(baseline_rows.iloc[0].to_dict())
    return top


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    candidates = [metric, metric.lower(), metric.replace("-", "_"), metric.replace("-", "_").lower()]
    if metric == "Macro-F1":
        candidates += ["macro_f1", "f1_macro", "macro-f1"]
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def _write_report(output_dir: Path, frame: pd.DataFrame, ranking: pd.DataFrame, args: argparse.Namespace, baseline_source: str) -> None:
    baseline_rows = ranking[ranking["is_baseline"].map(_is_true)]
    baseline = baseline_rows.iloc[0].to_dict() if not baseline_rows.empty else {}
    nonbaseline = ranking[~ranking["is_baseline"].map(_is_true)].copy()
    best_nonbaseline = nonbaseline.iloc[0].to_dict() if not nonbaseline.empty else {}
    score_col = "mean_AUPRC_across_datasets" if len(args.datasets) > 1 else "AUPRC_mean"
    baseline_score = _float_or_none(baseline.get(score_col))
    candidate_score = _float_or_none(best_nonbaseline.get(score_col))
    improved = candidate_score is not None and baseline_score is not None and candidate_score > baseline_score
    lines = [
        "# HERO tuning report",
        "",
        f"- model: {normalize_model_name(args.model)}",
        f"- datasets: {' '.join(args.datasets)}",
        f"- search_mode: {args.search_mode}",
        f"- baseline_source: {baseline_source}",
        f"- seeds: {' '.join(str(seed) for seed in args.seeds)}",
        f"- ranking: {'mean_AUPRC_across_datasets, min_AUPRC_across_datasets, Macro-F1_mean' if len(args.datasets) > 1 else _metric_column(args.metric) + ', Macro-F1_mean, AUROC_mean'}",
        "- validation strategy: trainer selects checkpoints by validation AUPRC; no separate val_AUPRC export was found.",
    ]
    if set(args.seeds) == {0}:
        lines.append("- stability note: seed0 only.")
    lines += ["", "## Baseline Performance", ""]
    if baseline:
        lines += [
            f"- trial_id: {baseline.get('trial_id')}",
            f"- Macro-F1_mean: {baseline.get('Macro-F1_mean')}",
            f"- AUROC_mean: {baseline.get('AUROC_mean')}",
            f"- AUPRC_mean: {baseline.get('AUPRC_mean')}",
            f"- mean_AUPRC_across_datasets: {baseline.get('mean_AUPRC_across_datasets')}",
            f"- min_AUPRC_across_datasets: {baseline.get('min_AUPRC_across_datasets')}",
        ]
    else:
        lines.append("- missing baseline row")
    lines += ["", "## Top 10 Non-Baseline Configs", ""]
    if nonbaseline.empty:
        lines.append("| rank | trial_id | AUPRC_mean | mean_AUPRC_across_datasets | min_AUPRC_across_datasets | Macro-F1_mean |")
        lines.append("|---|---|---|---|---|---|")
    else:
        lines.append("| rank | trial_id | AUPRC_mean | mean_AUPRC_across_datasets | min_AUPRC_across_datasets | Macro-F1_mean |")
        lines.append("|---|---|---|---|---|---|")
        for rank, (_, row) in enumerate(nonbaseline.head(10).iterrows(), start=1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        str(row.get("trial_id")),
                        _fmt(row.get("AUPRC_mean")),
                        _fmt(row.get("mean_AUPRC_across_datasets")),
                        _fmt(row.get("min_AUPRC_across_datasets")),
                        _fmt(row.get("Macro-F1_mean")),
                    ]
                )
                + " |"
            )
    lines += ["", "## Promotion Check", ""]
    if improved:
        lines.append(f"- best_nonbaseline_score: {_fmt(candidate_score)}")
        lines.append(f"- baseline_score: {_fmt(baseline_score)}")
        lines.append("- best config exceeds baseline before applying promote min_delta.")
        lines.append("- promotion recommendation: run promote_best_hero_config.py; it will still enforce --min_delta.")
    else:
        lines.append(f"- best_nonbaseline_score: {_fmt(candidate_score)}")
        lines.append(f"- baseline_score: {_fmt(baseline_score)}")
        lines.append("- best config does not exceed baseline.")
        lines.append("- promotion recommendation: keep original HERO-GNN config.")
    (output_dir / "tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| status | warning |\n|---|---|\n| empty | no tuning metrics found |\n"
    display = frame.copy()
    for column in display.columns:
        if column.endswith("_std") or column.endswith("_mean") or column in {"mean_AUPRC_across_datasets", "min_AUPRC_across_datasets"}:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    numeric = _float_or_none(value)
    return "" if numeric is None else f"{numeric:.6f}"


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
