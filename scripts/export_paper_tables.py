from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAIN_COLUMNS = ["dataset", "method", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]
ABLATION_COLUMNS = ["dataset", "variant", "macro_f1_mean", "auroc_mean", "auprc_mean", "evidence_recall_proxy", "evidence_necessity", "evidence_necessity_gap"]
NEIGHBOR_COLUMNS = ["dataset", "strategy", "auprc", "auroc", "macro_f1", "avg_selected_neighbors", "evidence_recall_proxy"]
SIGNIFICANCE_COLUMNS = ["dataset", "metric", "hero_mean", "baseline_name", "baseline_mean", "delta", "p_value", "test_type"]
RUNTIME_COLUMNS = ["dataset", "method", "time_total_mean", "time_total_std", "time_retrieval_mean", "time_training_mean"]
LABELER_COLUMNS = [
    "dataset",
    "labeler",
    "num_cards",
    "risk_relevance_rate",
    "mechanism_distribution",
    "avg_confidence",
    "agreement_with_mock",
    "macro_f1",
    "auroc",
    "auprc",
    "evidence_necessity_gap",
]
OFFICIAL_DATASETS = {"fraud_yelp_official", "fraud_amazon_official"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final paper-ready CSV tables.")
    parser.add_argument("--summary_dir", default="outputs/summary", help="Directory containing summary CSV files.")
    parser.add_argument("--out_dir", default="outputs/paper_tables", help="Paper table output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_paper_tables(Path(args.summary_dir), Path(args.out_dir))


def export_paper_tables(summary_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_empty(summary_dir / "main_table.csv", out_dir / "table_main.csv", MAIN_COLUMNS)
    _copy_or_empty(summary_dir / "ablation_table.csv", out_dir / "table_ablation.csv", ABLATION_COLUMNS)
    _copy_or_empty(summary_dir / "neighbor_strategy_table.csv", out_dir / "table_neighbor_strategy.csv", NEIGHBOR_COLUMNS)
    _export_official_fraud(summary_dir / "main_table.csv", out_dir / "table_official_fraud.csv")
    _export_scalability(summary_dir, out_dir / "table_scalability.csv")
    _copy_or_empty(summary_dir / "significance_table.csv", out_dir / "table_significance.csv", SIGNIFICANCE_COLUMNS)
    _copy_or_empty(summary_dir / "runtime_table.csv", out_dir / "table_runtime.csv", RUNTIME_COLUMNS)
    _copy_or_empty(summary_dir / "labeler_comparison_table.csv", out_dir / "table_labeler_comparison.csv", LABELER_COLUMNS)
    print(f"Wrote paper tables to {out_dir}")


def _copy_or_empty(src: Path, dst: Path, columns: list[str]) -> None:
    if src.exists():
        frame = pd.read_csv(src)
    else:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(dst, index=False)


def _export_official_fraud(main_path: Path, dst: Path) -> None:
    if main_path.exists():
        frame = pd.read_csv(main_path)
        if "dataset" in frame:
            frame = frame[frame["dataset"].isin(OFFICIAL_DATASETS)].copy()
    else:
        frame = pd.DataFrame(columns=MAIN_COLUMNS)
    frame.to_csv(dst, index=False)


def _export_scalability(summary_dir: Path, dst: Path) -> None:
    explicit = summary_dir / "scalability_table.csv"
    if explicit.exists():
        pd.read_csv(explicit).to_csv(dst, index=False)
        return
    runtime = summary_dir / "runtime_table.csv"
    if runtime.exists():
        pd.read_csv(runtime).to_csv(dst, index=False)
        return
    pd.DataFrame(columns=RUNTIME_COLUMNS).to_csv(dst, index=False)


if __name__ == "__main__":
    main()
