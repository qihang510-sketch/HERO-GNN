from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import (  # noqa: E402
    copy_if_exists,
    ensure_artifact_dirs,
    read_csv_or_empty,
    write_csv,
    write_latex,
    write_unavailable_csv,
)
from scripts.build_risk_card_case_table import build_risk_card_case_table  # noqa: E402
from scripts.run_significance_tests import METRICS, significance_rows  # noqa: E402
from scripts.summarize_experiment_suite import TEXT_RICH_SET, TRANSFER_DATASETS, mean_std_table  # noqa: E402
from src.training.submission import SUBMISSION_DATASETS, resolve_processed_dir  # noqa: E402


CSV_TARGETS = {
    "table2_main_text_rich": "table_main_mean_std.csv",
    "table3_transfer": "table_transfer_mean_std.csv",
    "table4_ablation": "table_ablation_mean_std.csv",
    "supp_table_significance": "table_significance.csv",
    "supp_table_llm_robustness": "table_llm_robustness.csv",
    "supp_table_labeler_comparison": "table_llm_labeler_comparison.csv",
    "supp_table_faithfulness": "table_faithfulness.csv",
    "supp_table_sensitivity": "table_sensitivity.csv",
    "supp_table_cost_scalability": "table_cost_scalability.csv",
    "supp_table_missing_or_skipped_runs": "table_missing_runs.csv",
}

LATEX_TARGETS = [
    "table1_dataset_statistics",
    "table2_main_text_rich",
    "table3_transfer",
    "table4_ablation",
    "table5_evidence_cases",
    "supp_table_significance",
    "supp_table_llm_robustness",
    "supp_table_labeler_comparison",
    "supp_table_faithfulness",
    "supp_table_sensitivity",
    "supp_table_cost_scalability",
]

RISK_CARD_DATASETS = ["yelp_academic", "amazon_video", "fraud_yelp", "fraud_amazon", "elliptic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final paper tables and figures from real HERO experiment outputs.")
    parser.add_argument("--main_dir", default="outputs/submission_main_20260711_225137")
    parser.add_argument("--transfer_dir", default=None)
    parser.add_argument("--significance_dir", default=None)
    parser.add_argument("--ablation_dir", default=None)
    parser.add_argument("--robustness_dir", default=None)
    parser.add_argument("--labeler_dir", default=None)
    parser.add_argument("--faithfulness_dir", default=None)
    parser.add_argument("--sensitivity_dir", default=None)
    parser.add_argument("--cost_dir", default=None)
    parser.add_argument("--output_dir", default="outputs/final_artifacts")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_final_artifacts(args)


def build_final_artifacts(args: argparse.Namespace) -> dict[str, Path]:
    allow_missing = bool(args.allow_missing or args.quick_test)
    dirs = ensure_artifact_dirs(args.output_dir)
    report: list[str] = []
    sources = _source_dirs(args)

    main_dir = Path(args.main_dir)
    transfer_dir = Path(args.transfer_dir) if getattr(args, "transfer_dir", None) else main_dir
    all_raw = _concat_raw([main_dir, transfer_dir])
    main_raw = _read_all_raw(main_dir)
    transfer_raw = _read_all_raw(transfer_dir)
    table1 = dataset_statistics(all_raw, data_root=args.data_root, datasets=list(SUBMISSION_DATASETS))
    write_csv(dirs["tables_csv"] / "table1_dataset_statistics.csv", table1)
    write_latex(dirs["tables_latex"] / "table1_dataset_statistics.tex", table1)

    table2 = _table_from_source("main", main_dir, "table_main_mean_std.csv", allow_missing, main_raw, datasets=TEXT_RICH_SET)
    write_csv(dirs["tables_csv"] / "table2_main_text_rich.csv", table2)
    table3 = _table_from_source("transfer", transfer_dir, "table_transfer_mean_std.csv", allow_missing, transfer_raw, datasets=TRANSFER_DATASETS)
    write_csv(dirs["tables_csv"] / "table3_transfer.csv", table3)
    table4 = _optional_table(Path(args.ablation_dir) if args.ablation_dir else main_dir, "table_ablation_mean_std.csv", "ablation results missing", allow_missing)
    write_csv(dirs["tables_csv"] / "table4_ablation.csv", table4)
    table5 = evidence_case_table(sources, allow_missing)
    write_csv(dirs["tables_csv"] / "table5_evidence_cases.csv", table5)

    significance_dir = Path(args.significance_dir) if getattr(args, "significance_dir", None) else main_dir
    significance = _significance_table(significance_dir, all_raw, allow_missing)
    write_csv(dirs["tables_csv"] / "supp_table_significance.csv", significance)
    robustness = _optional_table(Path(args.robustness_dir) if args.robustness_dir else None, "table_llm_robustness.csv", "robustness results missing", allow_missing)
    labeler = _optional_table(Path(args.labeler_dir) if args.labeler_dir else None, "table_llm_labeler_comparison.csv", "labeler comparison results missing", allow_missing)
    faithfulness = _optional_table(Path(args.faithfulness_dir) if args.faithfulness_dir else None, "table_faithfulness.csv", "faithfulness results missing", allow_missing)
    sensitivity = _optional_table(Path(args.sensitivity_dir) if getattr(args, "sensitivity_dir", None) else None, "table_sensitivity.csv", "sensitivity results missing", allow_missing)
    cost = _optional_table(Path(args.cost_dir) if args.cost_dir else main_dir, "table_cost_scalability.csv", "cost/scalability results missing", allow_missing)
    missing = _missing_table(main_dir, allow_missing)
    for name, frame in [
        ("supp_table_llm_robustness", robustness),
        ("supp_table_labeler_comparison", labeler),
        ("supp_table_faithfulness", faithfulness),
        ("supp_table_sensitivity", sensitivity),
        ("supp_table_cost_scalability", cost),
        ("supp_table_missing_or_skipped_runs", missing),
    ]:
        write_csv(dirs["tables_csv"] / f"{name}.csv", frame)

    csv_frames = {
        "table2_main_text_rich": table2,
        "table3_transfer": table3,
        "table4_ablation": table4,
        "table5_evidence_cases": table5,
        "table1_dataset_statistics": table1,
        "supp_table_significance": significance,
        "supp_table_llm_robustness": robustness,
        "supp_table_labeler_comparison": labeler,
        "supp_table_faithfulness": faithfulness,
        "supp_table_sensitivity": sensitivity,
        "supp_table_cost_scalability": cost,
    }
    for name in LATEX_TARGETS:
        write_latex(dirs["tables_latex"] / f"{name}.tex", csv_frames.get(name, pd.DataFrame()), highlight_metrics=name in {"table2_main_text_rich", "table3_transfer", "table4_ablation"})

    copied = copy_figures_and_data(sources, dirs)
    risk_card_status = _build_risk_card_cases(args, dirs, sources, allow_missing)
    seed_status = _build_seed_stability(main_dir, dirs["root"])
    report.extend(_report_lines(args, table1, csv_frames, missing, copied))
    report.extend(["", "## Risk Card Cases", *risk_card_status])
    report.extend(["", "## Seed Stability", f"- {seed_status}"])
    report_path = dirs["reports"] / "final_artifacts_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return dirs


def _build_risk_card_cases(args: argparse.Namespace, dirs: dict[str, Path], sources: list[Path], allow_missing: bool) -> list[str]:
    output_dir = dirs["root"] / "risk_card_cases"
    source_outputs = [str(path) for path in sources if path and path.exists()]
    if not source_outputs:
        source_outputs = [str(dirs["root"])]
    risk_args = argparse.Namespace(
        data_root=getattr(args, "data_root", "data"),
        output_dir=str(output_dir),
        datasets=RISK_CARD_DATASETS,
        source_outputs=source_outputs,
        top_cases_per_dataset=1,
        include_field_trace=True,
        write_latex=True,
        write_markdown=True,
        strict=False,
        copy_to_final_artifacts=True,
        final_artifacts_dir=str(dirs["root"]),
    )
    try:
        paths = build_risk_card_case_table(risk_args)
    except Exception as exc:
        if not allow_missing:
            raise
        report_path = dirs["reports"] / "RISK_CARD_CASE_REPORT.md"
        message = (
            "# Risk Card Case Report\n\n"
            "status: unavailable\n\n"
            f"reason: risk-card case generation failed with {type(exc).__name__}: {exc}\n\n"
            "All risk-card cases are extracted from real data or cached model outputs. Missing fields are marked as N/A rather than imputed.\n"
        )
        report_path.write_text(message, encoding="utf-8")
        for subdir, filename in [
            ("tables_csv", "supp_table_risk_card_cases_compact.csv"),
            ("tables_csv", "supp_table_risk_card_field_trace.csv"),
        ]:
            write_csv(dirs[subdir] / filename, pd.DataFrame([{"dataset": dataset, "status": "unavailable", "reason": str(exc)} for dataset in RISK_CARD_DATASETS]))
        for filename in ["supp_table_risk_card_cases_compact.tex", "supp_table_risk_card_field_trace.tex"]:
            (dirs["tables_latex"] / filename).write_text(
                "\\begin{tabular}{lll}\n\\toprule\ndataset & status & reason \\\\\n\\midrule\n"
                + "\n".join(f"{dataset} & unavailable & generation failed \\\\" for dataset in RISK_CARD_DATASETS)
                + "\n\\bottomrule\n\\end{tabular}\n",
                encoding="utf-8",
            )
        return [f"- unavailable: {type(exc).__name__}: {exc}"]
    return [
        f"- compact CSV: `{paths.get('compact_csv', '')}`",
        f"- field trace CSV: `{paths.get('field_trace_csv', '')}`",
        f"- copied to final artifacts: `{dirs['root']}`",
    ]


def _build_seed_stability(main_dir: Path, output_dir: Path) -> str:
    try:
        from scripts.plot_seed_stability import plot_seed_stability

        plot_seed_stability(main_dir / "summary" / "all_raw_runs.csv", output_dir, datasets=["yelp_academic", "amazon_video"])
        return "generated from summary/all_raw_runs.csv"
    except Exception as exc:
        report = output_dir / "reports" / "seed_stability_error.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        message = f"seed stability unavailable: {type(exc).__name__}: {exc}"
        report.write_text(message + "\n", encoding="utf-8")
        return message


def dataset_statistics(all_raw: pd.DataFrame, data_root: str | Path = "data", datasets: list[str] | None = None) -> pd.DataFrame:
    if datasets is None:
        if not all_raw.empty and "dataset" in all_raw:
            datasets = sorted(all_raw["dataset"].dropna().astype(str).unique())
        else:
            datasets = list(SUBMISSION_DATASETS)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        row = _dataset_stats_from_raw(all_raw, dataset)
        if row is None:
            row = _dataset_stats_from_processed(dataset, data_root)
        rows.append(row)
    return pd.DataFrame(rows)


def evidence_case_table(sources: list[Path], allow_missing: bool) -> pd.DataFrame:
    patterns = ["table_evidence_cases.csv", "table_evidence_chain_case.csv", "*evidence*case*.csv"]
    for root in sources:
        if root is None or not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return read_csv_or_empty(matches[0])
    if not allow_missing:
        raise FileNotFoundError("No evidence-chain case table found")
    return pd.DataFrame([{"status": "unavailable", "reason": "evidence-chain case table not found"}])


def copy_figures_and_data(sources: list[Path], dirs: dict[str, Path]) -> list[str]:
    copied: list[str] = []
    for root in sources:
        if root is None or not root.exists():
            continue
        for folder_name in ["figure_data", "figures"]:
            folder = root / folder_name
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() == ".csv":
                    target = dirs["figure_data"] / path.name
                elif path.suffix.lower() == ".pdf":
                    target = dirs["figures_pdf"] / path.name
                elif path.suffix.lower() == ".png":
                    target = dirs["figures_png"] / path.name
                else:
                    continue
                if copy_if_exists(path, target):
                    copied.append(str(target))
    return copied


def _table_from_source(label: str, root: Path, filename: str, allow_missing: bool, all_raw: pd.DataFrame, datasets: set[str]) -> pd.DataFrame:
    table = read_csv_or_empty(root / "summary" / filename)
    if table.empty and not all_raw.empty:
        suite = {"main", "transfer"} if label == "transfer" else "main"
        table = mean_std_table(all_raw, suite=suite, datasets=datasets)
    if table.empty:
        if not allow_missing:
            raise FileNotFoundError(f"{label} table missing: {root / 'summary' / filename}")
        return pd.DataFrame([{"status": "missing", "reason": f"{filename} not found", "source_dir": str(root)}])
    if "dataset" in table:
        table = table[table["dataset"].astype(str).isin(datasets)].copy()
    return table


def _optional_table(root: Path | None, filename: str, reason: str, allow_missing: bool) -> pd.DataFrame:
    if root is not None:
        table = read_csv_or_empty(root / "summary" / filename)
        if not table.empty:
            return table
    if not allow_missing:
        raise FileNotFoundError(f"{filename} missing under {root}")
    return pd.DataFrame([{"status": "missing", "reason": reason, "source_dir": str(root or '')}])


def _significance_table(main_dir: Path, all_raw: pd.DataFrame, allow_missing: bool) -> pd.DataFrame:
    table = read_csv_or_empty(main_dir / "summary" / "table_significance.csv")
    if not table.empty:
        return table
    if not all_raw.empty:
        return pd.DataFrame(significance_rows(all_raw, metrics=METRICS))
    if not allow_missing:
        raise FileNotFoundError("table_significance.csv missing and all_raw_runs.csv unavailable")
    return pd.DataFrame([{"status": "missing", "reason": "significance requires all_raw_runs.csv"}])


def _missing_table(main_dir: Path, allow_missing: bool) -> pd.DataFrame:
    for name in ["table_missing_runs.csv", "missing_runs.csv", "failed_runs.csv"]:
        table = read_csv_or_empty(main_dir / "summary" / name)
        if not table.empty:
            return table
    if not allow_missing:
        raise FileNotFoundError("missing/skipped run table not found")
    return pd.DataFrame([{"status": "missing", "reason": "missing/skipped run table not found", "source_dir": str(main_dir)}])


def _read_all_raw(main_dir: Path) -> pd.DataFrame:
    return read_csv_or_empty(main_dir / "summary" / "all_raw_runs.csv")


def _concat_raw(roots: list[Path]) -> pd.DataFrame:
    frames = [_read_all_raw(root) for root in roots]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _source_dirs(args: argparse.Namespace) -> list[Path]:
    values = [
        args.main_dir,
        getattr(args, "transfer_dir", None),
        getattr(args, "significance_dir", None),
        args.ablation_dir,
        args.robustness_dir,
        args.labeler_dir,
        args.faithfulness_dir,
        getattr(args, "sensitivity_dir", None),
        args.cost_dir,
    ]
    return [Path(value) for value in values if value]


def _dataset_stats_from_raw(all_raw: pd.DataFrame, dataset: str) -> dict[str, Any] | None:
    if all_raw.empty or "dataset" not in all_raw:
        return None
    subset = all_raw[all_raw["dataset"].astype(str) == dataset]
    if subset.empty:
        return None
    row: dict[str, Any] = {"dataset": dataset, "source": "all_raw_runs.csv", "status": "partial"}
    mapping = {
        "num_nodes": ["num_nodes", "nodes", "n_nodes"],
        "num_edges": ["num_edges", "edges", "n_edges"],
        "positive_rate": ["positive_rate", "test_positive_rate", "fraud_rate"],
        "train_nodes": ["num_train", "train_size", "train_nodes"],
        "val_nodes": ["num_val", "val_size", "val_nodes"],
        "test_nodes": ["num_test", "test_size", "test_nodes"],
    }
    found_any = False
    for target, candidates in mapping.items():
        for column in candidates:
            if column in subset:
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                if not values.empty:
                    row[target] = float(values.iloc[0])
                    found_any = True
                    break
        if target not in row:
            row[target] = pd.NA
    return row if found_any else None


def _dataset_stats_from_processed(dataset: str, data_root: str | Path) -> dict[str, Any]:
    base = {"dataset": dataset, "source": "processed_data", "status": "unavailable"}
    try:
        from src.data.loader import load_processed_data

        graph = load_processed_data(resolve_processed_dir(dataset, data_root))
        labels = np.asarray(graph.labels)
        valid = labels >= 0
        positives = labels[valid] == 1
        return {
            **base,
            "num_nodes": int(graph.features.shape[0]),
            "num_edges": int(graph.edge_index.shape[1]) if getattr(graph, "edge_index", np.empty((2, 0))).ndim == 2 else pd.NA,
            "positive_rate": float(positives.mean()) if positives.size else pd.NA,
            "train_nodes": int(len(graph.split.get("train", []))),
            "val_nodes": int(len(graph.split.get("val", []))),
            "test_nodes": int(len(graph.split.get("test", []))),
            "status": "ok",
        }
    except Exception as exc:
        return {**base, "reason": f"processed stats unavailable: {type(exc).__name__}: {exc}"}


def _report_lines(args: argparse.Namespace, table1: pd.DataFrame, frames: dict[str, pd.DataFrame], missing: pd.DataFrame, copied: list[str]) -> list[str]:
    lines = [
        "# Final Artifact Build Report",
        "",
        f"- main_dir: `{args.main_dir}`",
        f"- transfer_dir: `{getattr(args, 'transfer_dir', '') or args.main_dir}`",
        f"- output_dir: `{args.output_dir}`",
        f"- allow_missing: `{bool(args.allow_missing or args.quick_test)}`",
        "",
        "## Tables",
    ]
    lines.append(f"- table1_dataset_statistics.csv rows: {len(table1)}")
    for name, frame in frames.items():
        status = "ok"
        if frame.empty:
            status = "empty"
        elif "status" in frame and frame["status"].astype(str).isin(["missing", "unavailable"]).all():
            status = "missing_or_unavailable"
        lines.append(f"- {name}: {status}, rows={len(frame)}")
    lines.extend(["", "## Missing/Skipped", f"- rows: {len(missing)}", "", "## Copied Figures/Data"])
    if copied:
        lines.extend(f"- `{path}`" for path in copied)
    else:
        lines.append("- none")
    return lines


if __name__ == "__main__":
    main()
