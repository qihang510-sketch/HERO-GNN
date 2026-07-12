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
    _load_base_config,
    _resolve_trainer_params,
    _variant_warnings,
)
from src.training.submission import (  # noqa: E402
    DATASET_MODEL_MATRIX,
    SUBMISSION_DATASETS,
    TEXT_RICH_DATASETS,
    _submission_metric_payload,
    default_models_for_dataset,
    normalize_dataset_name,
    normalize_model_name,
    processed_ready,
    resolve_processed_dir,
    run_submission_experiment,
    write_skip,
)
from src.training.trainer import _resolve_hero_config, train_single_experiment  # noqa: E402
from src.utils.io import write_json  # noqa: E402


SUITES = ("main", "ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost", "all")
RUNNABLE_SUITES = ("main", "ablation", "robustness", "labeler_comparison", "faithfulness", "sensitivity", "cost")
PLACEHOLDER_SUITES: tuple[str, ...] = ()
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified HERO experiment suites.")
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--datasets", nargs="+", default=list(SUBMISSION_DATASETS))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--config", default=None, help="Optional tuned HERO config.")
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
    parser.add_argument("--sensitivity_k_values", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    parser.add_argument("--sensitivity_lambda_rel_values", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--sensitivity_lambda_chain_values", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--sensitivity_confidence_thresholds", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.9])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = _expand_suites(args.suite)
    output_dir = Path(args.output_dir or f"outputs/experiment_suite_{args.suite}")
    datasets = [normalize_dataset_name(dataset) for dataset in args.datasets]
    expected_runs = _expected_runs(suites=suites, datasets=datasets, models=args.models, seeds=args.seeds)
    _print_plan(output_dir, expected_runs, args.dry_run)
    if args.dry_run:
        return

    _prepare_output_dirs(output_dir)
    _write_suite_config(output_dir, args, suites, datasets, expected_runs)
    results: list[dict[str, Any]] = []
    for suite in suites:
        if suite == "main":
            results.extend(_run_main_suite(args, output_dir, datasets))
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
        else:
            results.extend(_write_placeholder_suite(args, output_dir, suite, datasets))

    write_json(
        output_dir / "summary" / "run_manifest.json",
        {
            "output_dir": str(output_dir),
            "created_at": _timestamp(),
            "num_expected_runs": len(expected_runs),
            "num_recorded_runs": len(results),
            "runs": results,
        },
    )
    ok = sum(1 for row in results if row.get("status") in {"ok", "exists"})
    missing = sum(1 for row in results if row.get("status") not in {"ok", "exists"})
    print(f"Experiment suite finished: ok_or_exists={ok} missing_or_skipped={missing} output_dir={output_dir}")


def _expand_suites(suite: str) -> list[str]:
    if suite == "all":
        return list(RUNNABLE_SUITES)
    return [suite]


def _prepare_output_dirs(output_dir: Path) -> None:
    for child in ["raw", "summary", "tables", "figures", "logs"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def _write_suite_config(
    output_dir: Path,
    args: argparse.Namespace,
    suites: list[str],
    datasets: list[str],
    expected_runs: list[dict[str, Any]],
) -> None:
    payload = {
        "suite": args.suite,
        "expanded_suites": suites,
        "datasets": datasets,
        "models": args.models,
        "seeds": [int(seed) for seed in args.seeds],
        "device": args.device,
        "data_root": args.data_root,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "top_k": int(args.top_k),
        "config": str(args.config or ""),
        "llm_label_file": str(args.llm_label_file or ""),
        "created_at": _timestamp(),
        "expected_runs": expected_runs,
    }
    write_json(output_dir / "config.json", payload)


def _expected_runs(
    suites: list[str],
    datasets: list[str],
    models: list[str] | None,
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in suites:
        for dataset in datasets:
            for model in _models_for_suite(suite, dataset, models):
                for seed in seeds:
                    rows.append({"suite": suite, "dataset": dataset, "model": model, "seed": int(seed)})
    return rows


def _models_for_suite(suite: str, dataset: str, models: list[str] | None) -> list[str]:
    dataset = normalize_dataset_name(dataset)
    if suite == "main":
        selected = list(default_models_for_dataset(dataset)) if models is None else [_main_model_for_dataset(dataset, model) for model in models]
        return _dedupe(selected)
    if suite == "ablation":
        if models is None or "hero" in {str(model).lower() for model in models}:
            return list(ABLATION_VARIANTS)
        return _dedupe(_normalize_ablation_variant(model) for model in models)
    if suite == "labeler_comparison":
        return ["hero_gnn"]
    if suite == "sensitivity":
        return ["hero_gnn"]
    return [_main_model_for_dataset(dataset, "hero")]


def _main_model_for_dataset(dataset: str, model: str) -> str:
    text = str(model).strip()
    lowered = text.lower().replace("-", "_")
    if lowered in {"hero", "hero_gnn", "hero_official"}:
        return "hero_gnn" if dataset in TEXT_RICH_DATASETS else "hero_official"
    return normalize_model_name(text)


def _normalize_ablation_variant(model: str) -> str:
    text = str(model).strip().lower().replace("-", "_")
    aliases = {
        "hero": "hero_gnn",
        "full": "hero_gnn",
        "full_hero": "hero_gnn",
        "wo_chain": "wo_evidence_chain",
        "wo_hetero": "wo_risk_relevant_heterophily",
        "wo_mechanism": "wo_mechanism_annotation",
    }
    return aliases.get(text, text)


def _dedupe(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _run_main_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        models = _models_for_suite("main", dataset, args.models)
        for seed in args.seeds:
            for model in models:
                rows.append(_run_main_case(args, output_dir, dataset, model, int(seed)))
    return rows


def _run_main_case(args: argparse.Namespace, output_dir: Path, dataset: str, model: str, seed: int) -> dict[str, Any]:
    raw_dir = output_dir / "raw"
    result_dir = _run_dir(raw_dir, dataset, model, seed)
    if args.skip_existing and (result_dir / "metrics.json").exists():
        return _ensure_run_artifacts(
            result_dir=result_dir,
            output_dir=output_dir,
            suite="main",
            dataset=dataset,
            model=model,
            seed=seed,
            status="exists",
            reason="skip_existing",
            started_at=_timestamp(),
            ended_at=_timestamp(),
            log_text="Skipped existing run.\n",
            args=args,
        )
    started = time.perf_counter()
    started_at = _timestamp()
    buffer = io.StringIO()
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
            config=args.config,
        )
    ended_at = _timestamp()
    result_dir = result.path.parent
    duration = time.perf_counter() - started
    log_text = buffer.getvalue()
    row = _ensure_run_artifacts(
        result_dir=result_dir,
        output_dir=output_dir,
        suite="main",
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
    )
    print(f"[{row['status']}] main dataset={row['dataset']} model={row['model']} seed={row['seed']} {row.get('reason', '')}")
    return row


def _run_ablation_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    rows = []
    base_hero_config, base_trainer_config, base_config_source = _load_base_config(args.config)
    for dataset in datasets:
        variants = _models_for_suite("ablation", dataset, args.models)
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
        )
    started = time.perf_counter()
    started_at = _timestamp()
    buffer = io.StringIO()
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
            data_dir = resolve_processed_dir(dataset, args.data_root)
            if not processed_ready(data_dir):
                write_skip(result_dir, dataset, variant, seed, "Missing dataset files. This is expected on local VSCode. Please run on AutoDL or provide data path.")
                status = "skipped"
                reason = "missing_data"
            else:
                spec = ABLATION_VARIANTS[variant]
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
                if warnings:
                    payload["warning"] = "; ".join(warnings)
                result_dir.mkdir(parents=True, exist_ok=True)
                write_json(result_dir / "metrics.json", payload)
                prediction_file = metrics.get("predictions_file")
                if prediction_file and Path(str(prediction_file)).exists():
                    shutil.copyfile(str(prediction_file), result_dir / "predictions.npy")
                status = "ok"
                reason = ""
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
    )
    print(f"[{row['status']}] ablation dataset={dataset} variant={variant} seed={seed} {row.get('reason', '')}")
    return row


def _run_robustness_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    from scripts.run_llm_annotation_robustness import run_robustness

    suite_args = argparse.Namespace(**vars(args))
    suite_args.datasets = datasets
    suite_args.output_dir = str(output_dir)
    suite_args.noise_type = None
    rows = run_robustness(suite_args)
    return _manifest_rows(rows, "robustness")


def _run_labeler_comparison_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    from scripts.run_llm_labeler_comparison import run_labeler_comparison

    suite_args = argparse.Namespace(**vars(args))
    suite_args.datasets = datasets
    suite_args.output_dir = str(output_dir)
    rows = run_labeler_comparison(suite_args)
    return _manifest_rows(rows, "labeler_comparison")


def _run_faithfulness_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    from scripts.run_evidence_faithfulness import run_faithfulness

    suite_args = argparse.Namespace(**vars(args))
    suite_args.datasets = datasets
    suite_args.output_dir = str(output_dir)
    suite_args.input_dir = args.input_dir or str(output_dir)
    suite_args.topks = args.faithfulness_topks
    suite_args.settings = args.faithfulness_settings
    rows = run_faithfulness(suite_args)
    return _manifest_rows(rows, "faithfulness")


def _run_cost_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
    from scripts.collect_cost_scalability import collect_cost_scalability

    suite_args = argparse.Namespace(**vars(args))
    suite_args.datasets = datasets
    suite_args.output_dir = str(output_dir)
    suite_args.input_dir = args.input_dir or str(output_dir)
    suite_args.models = args.models or ["hero_gnn", "hero_official"]
    table = collect_cost_scalability(suite_args)
    return _manifest_rows(table.to_dict(orient="records"), "cost")


def _run_sensitivity_suite(args: argparse.Namespace, output_dir: Path, datasets: list[str]) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    normalized_status = "missing" if status == "skipped" else status
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
    config = {
        "suite": suite,
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "device": args.device,
        "data_root": args.data_root,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "top_k": int(args.top_k),
        "config": str(args.config or ""),
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
    }


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
