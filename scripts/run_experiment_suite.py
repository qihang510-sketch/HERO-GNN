from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ablation_experiments import (  # noqa: E402
    ABLATION_VARIANTS,
    DEFAULT_ABLATION_VARIANTS,
    _load_base_config,
    _normalize_variant_name,
    _resolve_trainer_params,
    _variant_warnings,
)
from src.training.submission import (  # noqa: E402
    DATASET_MODEL_MATRIX,
    SUBMISSION_DATASETS,
    TEXT_RICH_DATASETS,
    _submission_metric_payload,
    default_models_for_dataset,
    hero_base_model_for_dataset,
    hero_display_name,
    normalize_dataset_name,
    normalize_model_name,
    processed_ready,
    resolve_processed_dir,
    run_submission_experiment,
    write_skip,
)
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


SUITES = ("main", "transfer", "ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost", "final", "all")
RUNNABLE_SUITES = ("main", "transfer", "ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost", "final")
PLACEHOLDER_SUITES: tuple[str, ...] = ()
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
TEXT_RICH_MAIN_DATASETS = ["yelp_academic", "amazon_video"]
TRANSFER_DATASETS = ["fraud_yelp", "fraud_amazon", "elliptic"]
SUBMISSION_BASELINES = ["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero"]
DEFAULT_ABLATION_SUITE_VARIANTS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified HERO experiment suites.")
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--continue_on_error", dest="continue_on_error", action="store_true", default=True)
    parser.add_argument("--no_continue_on_error", dest="continue_on_error", action="store_false")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--save_evidence", action="store_true")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--config", default=None, help="Optional tuned HERO config.")
    parser.add_argument("--tuned_config_dir", default=None, help="Directory produced by tune_hero_hyperparams.py.")
    parser.add_argument("--hero_gnn_config", default=None, help="Frozen config for text-rich HERO runs.")
    parser.add_argument("--hero_official_config", default=None, help="Frozen config for official/transaction HERO runs.")
    parser.add_argument("--llm_label_file", default=None)
    parser.add_argument("--input_dir", default=None, help="Existing suite root for faithfulness/cost collection. Defaults to --output_dir.")
    parser.add_argument("--annotation_file", default=None, help="Optional annotation JSONL for robustness.")
    parser.add_argument("--risk_card_file", default=None, help="Optional risk-card JSONL for labeler comparison.")
    parser.add_argument("--cached_llm_label_file", default=None)
    parser.add_argument("--full_llm_label_file", default=None)
    parser.add_argument("--call_full_llm", action="store_true")
    parser.add_argument("--max_cards", type=int, default=2000)
    parser.add_argument("--noise_types", nargs="+", choices=["relevance_flip", "mechanism_shuffle", "confidence_gaussian"], default=None)
    parser.add_argument("--noise_ratios", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
        "--labelers",
        nargs="+",
        choices=["random_labeler", "rule_based_labeler", "proxy_labeler", "cached_llm_labeler", "full_llm_labeler"],
        default=["random_labeler", "rule_based_labeler", "proxy_labeler", "cached_llm_labeler", "full_llm_labeler"],
    )
    parser.add_argument("--faithfulness_topks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument(
        "--faithfulness_settings",
        nargs="+",
        choices=["original_graph", "remove_topk_evidence", "remove_random_edges", "remove_irrelevant_edges", "keep_only_topk_evidence"],
        default=["original_graph", "remove_topk_evidence", "remove_random_edges", "remove_irrelevant_edges", "keep_only_topk_evidence"],
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--main_dir", default=None, help="Existing main/transfer suite root used by final artifact generation.")
    parser.add_argument("--ablation_dir", default=None)
    parser.add_argument("--robustness_dir", default=None)
    parser.add_argument("--labeler_dir", default=None)
    parser.add_argument("--faithfulness_dir", default=None)
    parser.add_argument("--sensitivity_dir", default=None)
    parser.add_argument("--cost_dir", default=None)
    parser.add_argument("--final_output_dir", default="outputs/final_artifacts")
    parser.add_argument("--sensitivity_k_values", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    parser.add_argument("--sensitivity_lambda_rel_values", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--sensitivity_lambda_chain_values", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--sensitivity_confidence_thresholds", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.9])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_quick_test(args)
    suites = _expand_suites(args.suite)
    output_dir = Path(args.output_dir or f"outputs/experiment_suite_{args.suite}")
    expected_runs = _expected_runs(suites=suites, args=args)
    _print_plan(output_dir, expected_runs, args.dry_run)
    if args.dry_run:
        return

    _prepare_output_dirs(output_dir)
    _write_suite_config(output_dir, args, suites, expected_runs)
    results: list[dict[str, Any]] = []
    for suite in suites:
        datasets = _datasets_for_suite(suite, args)
        if suite == "main":
            results.extend(_run_main_suite(args, output_dir, datasets))
        elif suite == "transfer":
            results.extend(_run_transfer_suite(args, output_dir, datasets))
        elif suite == "ablation":
            results.extend(_run_ablation_suite(args, output_dir, datasets))
        elif suite == "robustness":
            results.extend(_run_robustness_suite(args, output_dir, datasets))
        elif suite == "labeler_comparison":
            results.extend(_run_labeler_comparison_suite(args, output_dir, datasets))
        elif suite == "faithfulness":
            results.extend(_run_faithfulness_suite(args, output_dir, datasets))
        elif suite == "sensitivity":
            results.extend(_run_sensitivity_suite(args, output_dir, datasets))
        elif suite == "cost":
            results.extend(_run_cost_suite(args, output_dir, datasets))
        elif suite == "final":
            results.extend(_run_final_suite(args, output_dir))
        else:
            results.extend(_write_placeholder_suite(args, output_dir, suite, datasets))

    _run_post_summaries(args, output_dir, suites, results)
    manifest = {
        "output_dir": str(output_dir),
        "created_at": _timestamp(),
        "suite": args.suite,
        "expanded_suites": suites,
        "num_expected_runs": len(expected_runs),
        "num_recorded_runs": len(results),
        "max_parallel": int(args.max_parallel),
        "continue_on_error": bool(args.continue_on_error),
        "runs": results,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "summary" / "run_manifest.json", manifest)
    _write_failed_runs(output_dir, results)
    ok = sum(1 for row in results if row.get("status") in {"ok", "exists"})
    non_ok = sum(1 for row in results if row.get("status") not in {"ok", "exists"})
    print(f"Experiment suite finished: ok_or_exists={ok} non_ok={non_ok} output_dir={output_dir}")


def _run_post_summaries(args: argparse.Namespace, output_dir: Path, suites: list[str], results: list[dict[str, Any]]) -> None:
    if not any(suite in {"main", "transfer", "ablation"} for suite in suites):
        return
    try:
        from scripts.build_performance_diagnosis import build_performance_diagnosis
        from scripts.summarize_experiment_suite import summarize_suite

        summarize_suite(output_dir)
        build_performance_diagnosis(output_dir, tuned_config_dir=args.tuned_config_dir)
    except Exception as exc:
        report = output_dir / "reports" / "summary_error.log"
        report.parent.mkdir(parents=True, exist_ok=True)
        reason = f"{type(exc).__name__}: {exc}"
        report.write_text(reason + "\n", encoding="utf-8")
        results.append(
            {
                "suite": "summary",
                "dataset": "",
                "model": "summarize_experiment_suite",
                "seed": -1,
                "status": "failed",
                "reason": reason,
                "run_dir": str(output_dir / "summary"),
                "error_log": str(report),
            }
        )
        if not args.continue_on_error:
            raise


def _apply_quick_test(args: argparse.Namespace) -> None:
    if not args.quick_test:
        return
    args.datasets = ["yelp_academic"]
    args.seeds = [0]
    if args.suite in {"main", "transfer", "all"} and args.models is None:
        args.models = ["mlp", "gcn", "hero"]
    if args.suite in {"ablation", "all"} and args.variants is None:
        args.variants = ["hero_full", "hero_no_llm"]
    if args.suite in {"robustness", "all"}:
        args.noise_ratios = [0.1]
    if args.suite in {"faithfulness", "all"}:
        args.faithfulness_topks = [1]
    if args.suite in {"sensitivity", "all"}:
        args.sensitivity_k_values = [5, 10]
        args.sensitivity_lambda_rel_values = [0.0]
        args.sensitivity_lambda_chain_values = [0.0]
        args.sensitivity_confidence_thresholds = [0.3]
    args.epochs = min(int(args.epochs), 1)


def _expand_suites(suite: str) -> list[str]:
    if suite == "all":
        return list(RUNNABLE_SUITES)
    return [suite]


def _prepare_output_dirs(output_dir: Path) -> None:
    for child in ["raw", "logs", "summary", "tables", "figures", "figure_data", "configs", "reports"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def _write_suite_config(
    output_dir: Path,
    args: argparse.Namespace,
    suites: list[str],
    expected_runs: list[dict[str, Any]],
) -> None:
    payload = {
        "suite": args.suite,
        "expanded_suites": suites,
        "datasets": args.datasets,
        "models": args.models,
        "variants": args.variants,
        "seeds": [int(seed) for seed in args.seeds],
        "device": args.device,
        "data_root": args.data_root,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "top_k": int(args.top_k),
        "config": str(args.config or ""),
        "tuned_config_dir": str(args.tuned_config_dir or ""),
        "hero_gnn_config": str(args.hero_gnn_config or ""),
        "hero_official_config": str(args.hero_official_config or ""),
        "llm_label_file": str(args.llm_label_file or ""),
        "quick_test": bool(args.quick_test),
        "max_parallel": int(args.max_parallel),
        "continue_on_error": bool(args.continue_on_error),
        "save_predictions": bool(args.save_predictions),
        "save_embeddings": bool(args.save_embeddings),
        "save_evidence": bool(args.save_evidence),
        "created_at": _timestamp(),
        "expected_runs": expected_runs,
    }
    write_json(output_dir / "config.json", payload)
    write_json(output_dir / "configs" / "suite_config.json", payload)


def _expected_runs(
    suites: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in suites:
        for dataset in _datasets_for_suite(suite, args):
            for model in _models_for_suite(suite, dataset, args):
                seeds = [None] if suite in {"cost", "final"} else args.seeds
                for seed in seeds:
                    rows.append({"suite": suite, "dataset": dataset, "model": model, "seed": "" if seed is None else int(seed)})
    return rows


def _datasets_for_suite(suite: str, args: argparse.Namespace) -> list[str]:
    if suite == "final":
        return ["final_artifacts"]
    if args.datasets:
        return [normalize_dataset_name(dataset) for dataset in args.datasets]
    if suite == "main":
        return list(TEXT_RICH_MAIN_DATASETS)
    if suite == "transfer":
        return list(TRANSFER_DATASETS)
    if suite in {"ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity"}:
        return list(TEXT_RICH_MAIN_DATASETS)
    if suite == "cost":
        return list(SUBMISSION_DATASETS)
    return []


def _models_for_suite(suite: str, dataset: str, args: argparse.Namespace) -> list[str]:
    dataset = normalize_dataset_name(dataset)
    if suite in {"main", "transfer"}:
        selected_models = SUBMISSION_BASELINES if args.models is None else args.models
        selected = [_main_model_for_dataset(dataset, model) for model in selected_models]
        return _dedupe(selected)
    if suite == "ablation":
        variants = args.variants if args.variants is not None else args.models
        if variants is None or "hero" in {str(model).lower() for model in variants}:
            return list(DEFAULT_ABLATION_SUITE_VARIANTS)
        return _dedupe(_normalize_ablation_variant(model) for model in variants)
    if suite == "labeler_comparison":
        return ["hero_gnn"]
    if suite == "sensitivity":
        return ["hero_gnn"]
    if suite == "cost":
        return _dedupe(args.models or ["hero_full"])
    if suite == "final":
        return ["final_artifacts"]
    return [_main_model_for_dataset(dataset, "hero")]


def _main_model_for_dataset(dataset: str, model: str) -> str:
    text = str(model).strip()
    lowered = text.lower().replace("-", "_")
    if lowered in {"hero", "hero_full", "full_hero", "full"}:
        return "hero_full"
    if lowered in {
        "hero_no_llm",
        "hero_no_mechanism",
        "hero_no_risk_weighting",
        "hero_no_relation_loss",
        "hero_no_chain_consistency",
        "hero_structure_only",
        "hero_semantic_only",
        "hero_no_dual_branch",
    }:
        return lowered
    return normalize_model_name(text)


def _normalize_ablation_variant(model: str) -> str:
    return _normalize_variant_name(model)


def _dedupe(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _run_main_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        models = _models_for_suite("main", dataset, args)
        for seed in args.seeds:
            for model in models:
                rows.append(_run_main_case(args, output_dir, "main", dataset, model, int(seed)))
    return rows


def _run_transfer_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        models = _models_for_suite("transfer", dataset, args)
        for seed in args.seeds:
            for model in models:
                if dataset not in TRANSFER_DATASETS:
                    result_dir = _run_dir(output_dir / "raw", dataset, model, int(seed))
                    reason = "transfer_suite_requires_transfer_dataset"
                    write_skip(result_dir, dataset, model, int(seed), reason)
                    rows.append(
                        _ensure_run_artifacts(
                            result_dir=result_dir,
                            output_dir=output_dir,
                            suite="transfer",
                            dataset=dataset,
                            model=model,
                            seed=int(seed),
                            status="skipped",
                            reason=reason,
                            started_at=_timestamp(),
                            ended_at=_timestamp(),
                            log_text=f"{reason}\n",
                            args=args,
                        )
                    )
                else:
                    rows.append(_run_main_case(args, output_dir, "transfer", dataset, model, int(seed)))
    return rows


def _run_main_case(args: argparse.Namespace, output_dir: Path, suite: str, dataset: str, model: str, seed: int) -> dict[str, Any]:
    raw_dir = output_dir / "raw"
    result_dir = _run_dir(raw_dir, dataset, model, seed)
    active_config, tuned_config_used = _config_for_run(args, dataset, model)
    if args.skip_existing and (result_dir / "metrics.json").exists():
        return _ensure_run_artifacts(
            result_dir=result_dir,
            output_dir=output_dir,
            suite=suite,
            dataset=dataset,
            model=model,
            seed=seed,
            status="exists",
            reason="skip_existing",
            started_at=_timestamp(),
            ended_at=_timestamp(),
            log_text="Skipped existing run.\n",
            args=args,
            active_config=active_config,
            tuned_config_used=tuned_config_used,
        )
    started = time.perf_counter()
    started_at = _timestamp()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = run_submission_experiment(
                dataset=dataset,
                model=model,
                seed=seed,
                output_dir=raw_dir,
                data_root=args.data_root,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                top_k=args.top_k,
                overwrite=False,
                device=args.device,
                llm_label_file=args.llm_label_file,
                config=active_config,
            )
        ended_at = _timestamp()
        result_dir = result.path.parent
        duration = time.perf_counter() - started
        log_text = buffer.getvalue()
        row = _ensure_run_artifacts(
            result_dir=result_dir,
            output_dir=output_dir,
            suite=suite,
            dataset=result.dataset,
            model=result.model,
            seed=result.seed,
            status=result.status,
            reason=result.reason,
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=duration,
            log_text=log_text,
            args=args,
            active_config=active_config,
            tuned_config_used=tuned_config_used,
        )
    except Exception as exc:
        ended_at = _timestamp()
        duration = time.perf_counter() - started
        row = _record_failed_run(
            result_dir=result_dir,
            output_dir=output_dir,
            suite=suite,
            dataset=dataset,
            model=model,
            seed=seed,
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=duration,
            log_text=buffer.getvalue(),
            args=args,
        )
        if not args.continue_on_error:
            raise
    print(f"[{row['status']}] {suite} dataset={row['dataset']} model={row['model']} seed={row['seed']} {row.get('reason', '')}")
    return row


def _config_for_run(args: argparse.Namespace, dataset: str, model: str) -> tuple[str | None, bool]:
    model = normalize_model_name(model)
    if model.startswith("hero"):
        tuned = _tuned_config_for_dataset(args.tuned_config_dir, dataset, model)
        if tuned is not None:
            return str(tuned), True
        if dataset in TEXT_RICH_DATASETS and args.hero_gnn_config:
            return str(args.hero_gnn_config), False
        if dataset not in TEXT_RICH_DATASETS and args.hero_official_config:
            return str(args.hero_official_config), False
    return (str(args.config), False) if args.config else (None, False)


def _tuned_config_for_dataset(tuned_config_dir: str | None, dataset: str, model: str) -> Path | None:
    if not tuned_config_dir:
        return None
    root = Path(tuned_config_dir)
    dataset = normalize_dataset_name(dataset)
    candidates = [
        root / "best_configs" / f"hero_full_{dataset}.yaml",
        root / "best_configs" / f"{model}_{dataset}.yaml",
        root / f"hero_full_{dataset}.yaml",
        root / f"{model}_{dataset}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _run_ablation_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        tuned_config = _tuned_config_for_dataset(args.tuned_config_dir, dataset, "hero_full")
        base_config_path = str(tuned_config) if tuned_config is not None else args.config
        base_hero_config, base_trainer_config, base_config_source = _load_base_config(base_config_path)
        tuned_config_used = tuned_config is not None
        variants = _models_for_suite("ablation", dataset, args)
        for seed in args.seeds:
            for variant in variants:
                rows.append(
                    _run_ablation_case(
                        args=args,
                        output_dir=output_dir,
                        dataset=dataset,
                        variant=variant,
                        seed=int(seed),
                        base_hero_config=base_hero_config,
                        base_trainer_config=base_trainer_config,
                        base_config_source=base_config_source,
                        tuned_config_used=tuned_config_used,
                    )
                )
    return rows


def _run_ablation_case(
    args: argparse.Namespace,
    output_dir: Path,
    dataset: str,
    variant: str,
    seed: int,
    base_hero_config: dict[str, Any],
    base_trainer_config: dict[str, Any],
    base_config_source: str,
    tuned_config_used: bool = False,
) -> dict[str, Any]:
    raw_dir = output_dir / "raw"
    result_dir = _run_dir(raw_dir, dataset, variant, seed)
    if args.skip_existing and (result_dir / "metrics.json").exists():
        return _ensure_run_artifacts(
            result_dir=result_dir,
            output_dir=output_dir,
            suite="ablation",
            dataset=dataset,
            model=variant,
            seed=seed,
            status="exists",
            reason="skip_existing",
            started_at=_timestamp(),
            ended_at=_timestamp(),
            log_text="Skipped existing ablation run.\n",
            args=args,
            active_config=base_config_source,
            tuned_config_used=tuned_config_used,
        )
    started = time.perf_counter()
    started_at = _timestamp()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            if dataset not in TEXT_RICH_DATASETS:
                write_skip(result_dir, dataset, variant, seed, "ablation_suite_supports_text_rich_datasets_only")
                status = "skipped"
                reason = "ablation_suite_supports_text_rich_datasets_only"
            elif variant not in ABLATION_VARIANTS:
                write_skip(result_dir, dataset, variant, seed, f"unknown_ablation_variant:{variant}")
                status = "skipped"
                reason = f"unknown_ablation_variant:{variant}"
            else:
                spec = ABLATION_VARIANTS[variant]
                if spec.get("status") == "unsupported":
                    reason = str(spec.get("fallback", "unsupported_ablation_variant"))
                    write_skip(result_dir, dataset, variant, seed, reason)
                    status = "skipped"
                else:
                    data_dir = resolve_processed_dir(dataset, args.data_root)
                    if not processed_ready(data_dir):
                        write_skip(result_dir, dataset, variant, seed, "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.")
                        status = "skipped"
                        reason = "missing_data"
                    else:
                        variant_overrides = dict(spec.get("hero_config", {}))
                        merged_config = {**base_hero_config, **variant_overrides}
                        resolved_config = _resolve_hero_config(str(spec["trainer_model"]), merged_config)
                        trainer_params = _resolve_trainer_params(args, base_trainer_config, resolved_config)
                        warnings = _variant_warnings(variant, base_hero_config, resolved_config)
                        metrics = train_single_experiment(
                            dataset=dataset,
                            model_name=str(spec["trainer_model"]),
                            seed=seed,
                            data_dir=data_dir,
                            output_root=raw_dir / "_project_runs_ablation",
                            epochs=trainer_params["epochs"],
                            lr=trainer_params["lr"],
                            hidden_dim=trainer_params["hidden_dim"],
                            top_k=trainer_params["top_k"],
                            device=args.device,
                            hero_config=resolved_config,
                        )
                        payload = _submission_metric_payload(metrics, dataset, variant, seed, "project")
                        payload.update(
                            {
                                "suite": "ablation",
                                "variant": variant,
                                "ablation_name": str(spec["name"]),
                                "trainer_model": str(spec["trainer_model"]),
                                "hero_config": resolved_config,
                                "base_config": base_config_source,
                            }
                        )
                        if spec.get("fallback"):
                            payload["fallback"] = str(spec.get("fallback"))
                        if warnings:
                            payload["warning"] = "; ".join(warnings)
                        result_dir.mkdir(parents=True, exist_ok=True)
                        write_json(result_dir / "metrics.json", payload)
                        prediction_file = metrics.get("predictions_file")
                        if prediction_file and Path(str(prediction_file)).exists():
                            shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
                        status = "ok"
                        reason = ""
    except Exception as exc:
        ended_at = _timestamp()
        duration = time.perf_counter() - started
        row = _record_failed_run(
            result_dir=result_dir,
            output_dir=output_dir,
            suite="ablation",
            dataset=dataset,
            model=variant,
            seed=seed,
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=duration,
            log_text=buffer.getvalue(),
            args=args,
        )
        if not args.continue_on_error:
            raise
        print(f"[{row['status']}] ablation dataset={dataset} variant={variant} seed={seed} {row.get('reason', '')}")
        return row
    ended_at = _timestamp()
    duration = time.perf_counter() - started
    row = _ensure_run_artifacts(
        result_dir=result_dir,
        output_dir=output_dir,
        suite="ablation",
        dataset=dataset,
        model=variant,
        seed=seed,
        status=status,
        reason=reason,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration,
        log_text=buffer.getvalue(),
        args=args,
        active_config=base_config_source,
        tuned_config_used=tuned_config_used,
    )
    print(f"[{row['status']}] ablation dataset={dataset} variant={variant} seed={seed} {row.get('reason', '')}")
    return row


def _run_robustness_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    try:
        from scripts.run_llm_annotation_robustness import run_robustness

        suite_args = argparse.Namespace(**vars(args))
        suite_args.datasets = datasets
        suite_args.output_dir = str(output_dir)
        suite_args.noise_type = None
        rows = run_robustness(suite_args)
        return _manifest_rows(rows, "robustness")
    except Exception as exc:
        return [_advanced_failed_row(args, output_dir, "robustness", exc)]


def _run_labeler_comparison_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    try:
        from scripts.run_llm_labeler_comparison import run_labeler_comparison

        suite_args = argparse.Namespace(**vars(args))
        suite_args.datasets = datasets
        suite_args.output_dir = str(output_dir)
        rows = run_labeler_comparison(suite_args)
        return _manifest_rows(rows, "labeler_comparison")
    except Exception as exc:
        return [_advanced_failed_row(args, output_dir, "labeler_comparison", exc)]


def _run_faithfulness_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    try:
        from scripts.run_evidence_faithfulness import run_faithfulness

        suite_args = argparse.Namespace(**vars(args))
        suite_args.datasets = datasets
        suite_args.output_dir = str(output_dir)
        suite_args.input_dir = args.input_dir or str(output_dir)
        suite_args.topks = args.faithfulness_topks
        suite_args.settings = args.faithfulness_settings
        rows = run_faithfulness(suite_args)
        return _manifest_rows(rows, "faithfulness")
    except Exception as exc:
        return [_advanced_failed_row(args, output_dir, "faithfulness", exc)]


def _run_cost_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    try:
        from scripts.collect_cost_scalability import collect_cost_scalability

        suite_args = argparse.Namespace(**vars(args))
        suite_args.datasets = datasets
        suite_args.output_dir = str(output_dir)
        suite_args.input_dir = args.input_dir or str(output_dir)
        suite_args.models = args.models or ["hero_gnn", "hero_official"]
        table = collect_cost_scalability(suite_args)
        return _manifest_rows(table.to_dict(orient="records"), "cost")
    except Exception as exc:
        return [_advanced_failed_row(args, output_dir, "cost", exc)]


def _run_final_suite(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main_dir = Path(args.main_dir or args.input_dir or output_dir)
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        from scripts.build_final_tables_and_figures import build_final_artifacts
        from scripts.run_significance_tests import main as significance_main
        from scripts.summarize_experiment_suite import summarize_suite

        if (main_dir / "raw").exists():
            summarize_suite(main_dir)
        old_argv = sys.argv[:]
        try:
            sys.argv = ["run_significance_tests.py", "--main_dir", str(main_dir)]
            significance_main()
        finally:
            sys.argv = old_argv
        final_args = argparse.Namespace(
            main_dir=str(main_dir),
            ablation_dir=args.ablation_dir,
            robustness_dir=args.robustness_dir or str(output_dir),
            labeler_dir=args.labeler_dir or str(output_dir),
            faithfulness_dir=args.faithfulness_dir or str(output_dir),
            sensitivity_dir=args.sensitivity_dir or str(output_dir),
            cost_dir=args.cost_dir or str(output_dir),
            output_dir=args.final_output_dir,
            data_root=args.data_root,
            quick_test=bool(args.quick_test),
            allow_missing=True,
        )
        artifacts = build_final_artifacts(final_args)
        status = "ok"
        reason = ""
        run_dir = Path(args.final_output_dir)
        (output_dir / "reports" / "final_artifacts_path.txt").write_text(str(run_dir), encoding="utf-8")
        rows.append(
            {
                "suite": "final",
                "dataset": "",
                "model": "final_artifacts",
                "seed": -1,
                "status": status,
                "reason": reason,
                "run_dir": str(run_dir),
                "artifact_count": len(artifacts),
                "duration_sec": time.perf_counter() - started,
                "started_at": started_at,
                "ended_at": _timestamp(),
            }
        )
    except Exception as exc:
        run_dir = output_dir / "raw" / "final" / "final_artifacts" / "seed_0"
        row = _record_failed_run(
            result_dir=run_dir,
            output_dir=output_dir,
            suite="final",
            dataset="",
            model="final_artifacts",
            seed=0,
            reason=f"{type(exc).__name__}: {exc}",
            started_at=started_at,
            ended_at=_timestamp(),
            duration_sec=time.perf_counter() - started,
            log_text="",
            args=args,
        )
        rows.append(row)
        if not args.continue_on_error:
            raise
    return rows


def _run_sensitivity_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    try:
        from scripts.run_sensitivity_analysis import run_sensitivity

        suite_args = argparse.Namespace(**vars(args))
        suite_args.datasets = datasets
        suite_args.output_dir = str(output_dir)
        suite_args.k_values = args.sensitivity_k_values
        suite_args.lambda_rel_values = args.sensitivity_lambda_rel_values
        suite_args.lambda_chain_values = args.sensitivity_lambda_chain_values
        suite_args.confidence_thresholds = args.sensitivity_confidence_thresholds
        suite_args.quick_test = False
        rows = run_sensitivity(suite_args)
        return _manifest_rows(rows, "sensitivity")
    except Exception as exc:
        return [_advanced_failed_row(args, output_dir, "sensitivity", exc)]


def _advanced_failed_row(args: argparse.Namespace, output_dir: Path, suite: str, exc: Exception) -> dict[str, Any]:
    row = _record_failed_run(
        result_dir=output_dir / "raw" / suite / "suite_level" / "seed_0",
        output_dir=output_dir,
        suite=suite,
        dataset="",
        model=suite,
        seed=0,
        reason=f"{type(exc).__name__}: {exc}",
        started_at=_timestamp(),
        ended_at=_timestamp(),
        duration_sec=0.0,
        log_text="",
        args=args,
    )
    if not args.continue_on_error:
        raise exc
    return row


def _manifest_rows(rows: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    manifest = []
    for row in rows:
        manifest.append(
            {
                "suite": suite,
                "dataset": str(row.get("dataset", "")),
                "model": str(row.get("model", row.get("labeler", "hero_gnn"))),
                "seed": int(row.get("seed", -1)) if str(row.get("seed", "")).lstrip("-").isdigit() else -1,
                "status": str(row.get("status", "ok")),
                "reason": str(row.get("skip_reason", row.get("reason", ""))),
                "run_dir": str(row.get("run_dir", "")),
            }
        )
    return manifest


def _write_placeholder_suite(args: argparse.Namespace, output_dir: Path, suite: str, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        model = _main_model_for_dataset(dataset, "hero")
        for seed in args.seeds:
            result_dir = _run_dir(output_dir / "raw", dataset, model, int(seed), suite=suite)
            reason = f"{suite}_suite_not_implemented_in_runner; no fake metrics written"
            write_skip(result_dir, dataset, model, int(seed), reason)
            row = _ensure_run_artifacts(
                result_dir=result_dir,
                output_dir=output_dir,
                suite=suite,
                dataset=dataset,
                model=model,
                seed=int(seed),
                status="missing",
                reason=reason,
                started_at=_timestamp(),
                ended_at=_timestamp(),
                log_text=f"{reason}\n",
                args=args,
            )
            rows.append(row)
            print(f"[missing] {suite} dataset={dataset} model={model} seed={seed}: {reason}")
    return rows


def _run_dir(raw_dir: Path, dataset: str, model: str, seed: int, suite: str | None = None) -> Path:
    if suite in PLACEHOLDER_SUITES:
        return raw_dir / suite / dataset / model / f"seed_{seed}"
    return raw_dir / dataset / model / f"seed_{seed}"


def _ensure_run_artifacts(
    result_dir: Path,
    output_dir: Path,
    suite: str,
    dataset: str,
    model: str,
    seed: int,
    status: str,
    reason: str,
    started_at: str,
    ended_at: str,
    log_text: str,
    args: argparse.Namespace,
    duration_sec: float = 0.0,
    active_config: str | None = None,
    tuned_config_used: bool = False,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    normalized_status = status or "missing"
    runtime = {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "status": normalized_status,
        "reason": reason,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_sec": float(duration_sec),
    }
    existing_metrics = _read_json_if_exists(result_dir / "metrics.json")
    if existing_metrics:
        for metric_key, runtime_key in [
            ("time_training_sec", "train_time_seconds"),
            ("Training time", "train_time_seconds"),
            ("peak_gpu_memory_mb", "peak_gpu_memory_mb"),
            ("device", "device"),
        ]:
            if metric_key in existing_metrics and runtime_key not in runtime:
                runtime[runtime_key] = existing_metrics[metric_key]
    config = {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "display_model": hero_display_name(model) if str(model).startswith("hero") else model,
        "hero_trainer_model": hero_base_model_for_dataset(dataset, model) if str(model).startswith("hero") else "",
        "seed": int(seed),
        "device": args.device,
        "data_root": args.data_root,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "top_k": int(args.top_k),
        "config": str(active_config or args.config or ""),
        "base_config": str(args.config or ""),
        "tuned_config_dir": str(args.tuned_config_dir or ""),
        "tuned_config_used": bool(tuned_config_used),
        "hero_gnn_config": str(args.hero_gnn_config or ""),
        "hero_official_config": str(args.hero_official_config or ""),
        "llm_label_file": str(args.llm_label_file or ""),
    }
    write_json(result_dir / "runtime.json", runtime)
    write_json(result_dir / "config.json", config)
    if not (result_dir / "metrics.json").exists():
        payload = {
            "suite": suite,
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "status": normalized_status,
            "skip_reason": reason or normalized_status,
        }
        skip_path = result_dir / "skip_reason.json"
        if skip_path.exists():
            try:
                payload.update(json.loads(skip_path.read_text(encoding="utf-8")))
                payload["status"] = normalized_status
            except json.JSONDecodeError:
                pass
        write_json(result_dir / "metrics.json", payload)
    else:
        _tag_metrics(result_dir / "metrics.json", suite)
    final_log = log_text
    run_log = result_dir / "run.log"
    if run_log.exists():
        final_log += ("\n" if final_log and not final_log.endswith("\n") else "") + run_log.read_text(encoding="utf-8", errors="ignore")
    (result_dir / "log.txt").write_text(final_log, encoding="utf-8")
    log_copy = output_dir / "logs" / f"{suite}_{dataset}_{model}_seed_{seed}.txt"
    log_copy.parent.mkdir(parents=True, exist_ok=True)
    log_copy.write_text(final_log, encoding="utf-8")
    return {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "status": normalized_status,
        "reason": reason,
        "run_dir": str(result_dir),
        "error_log": str(result_dir / "error.log") if (result_dir / "error.log").exists() else "",
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_failed_run(
    result_dir: Path,
    output_dir: Path,
    suite: str,
    dataset: str,
    model: str,
    seed: int,
    reason: str,
    started_at: str,
    ended_at: str,
    duration_sec: float,
    log_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    error_log = result_dir / "error.log"
    error_log.write_text(reason + "\n", encoding="utf-8")
    return _ensure_run_artifacts(
        result_dir=result_dir,
        output_dir=output_dir,
        suite=suite,
        dataset=dataset,
        model=model,
        seed=seed,
        status="failed",
        reason=reason,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration_sec,
        log_text=(log_text or "") + ("\n" if log_text else "") + reason + "\n",
        args=args,
    )


def _write_failed_runs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = ["suite", "dataset", "model", "seed", "status", "reason", "run_dir", "error_log"]
    non_ok = [row for row in rows if row.get("status") not in {"ok", "exists"}]
    for path in [output_dir / "failed_runs.csv", output_dir / "summary" / "failed_runs.csv"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in non_ok:
                writer.writerow({field: row.get(field, "") for field in fields})


def _tag_metrics(path: Path, suite: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if payload.get("suite") == suite:
        return
    payload["suite"] = suite
    write_json(path, payload)


def _print_plan(output_dir: Path, expected_runs: list[dict[str, Any]], dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}output_dir={output_dir}")
    print(f"{prefix}planned_runs={len(expected_runs)}")
    for row in expected_runs[:20]:
        print(f"{prefix}{row['suite']} dataset={row['dataset']} model={row['model']} seed={row['seed']}")
    if len(expected_runs) > 20:
        print(f"{prefix}... {len(expected_runs) - 20} more runs")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
