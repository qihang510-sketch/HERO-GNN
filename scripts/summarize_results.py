from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HERO-GNN result artifacts.")
    parser.add_argument("--result_dir", "--results-dir", default="outputs/results", help="Directory containing results.")
    parser.add_argument("--out_dir", default="outputs/summary", help="Directory for summary CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in sorted(result_dir.glob("*/*/seed_*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(payload)

    table_path = out_dir / "main_table.csv"
    if not rows:
        _empty_main().to_csv(table_path, index=False)
        _empty_ablation().to_csv(out_dir / "ablation_table.csv", index=False)
        _empty_neighbor().to_csv(out_dir / "neighbor_strategy_table.csv", index=False)
        _empty_diagnostic().to_csv(out_dir / "diagnostic_table.csv", index=False)
        print(f"Wrote empty summary to {table_path}")
        return

    frame = pd.DataFrame(rows).rename(columns={"model": "method"})
    summary = _summary(frame)
    summary.to_csv(table_path, index=False)
    _ablation_table(frame).to_csv(out_dir / "ablation_table.csv", index=False)
    _neighbor_strategy_table(frame).to_csv(out_dir / "neighbor_strategy_table.csv", index=False)
    _diagnostic_table(frame).to_csv(out_dir / "diagnostic_table.csv", index=False)
    print(f"Wrote summary to {table_path}")


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
        )
        .fillna(0.0)
    )


def _ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    methods = ["hero_gnn", "hero_wo_hetero", "hero_wo_mechanism", "hero_wo_chain"]
    subset = frame[frame["method"].isin(methods)].copy()
    if subset.empty:
        return _empty_ablation()
    cols = ["dataset", "method", "macro_f1", "auroc", "auprc", "evidence_recall_proxy", "evidence_necessity_score"]
    for col in cols:
        if col not in subset:
            subset[col] = 0.0
    table = (
        subset[cols]
        .groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            evidence_recall_proxy=("evidence_recall_proxy", "mean"),
            evidence_necessity=("evidence_necessity_score", "mean"),
        )
        .fillna(0.0)
    )
    return table.rename(columns={"method": "variant"})


def _neighbor_strategy_table(frame: pd.DataFrame) -> pd.DataFrame:
    methods = ["graphsage", "semsim_gnn", "rulehetero_gnn", "hero_gnn"]
    subset = frame[frame["method"].isin(methods)].copy()
    if subset.empty:
        return _empty_neighbor()
    for col in ["auprc", "auroc", "macro_f1", "avg_selected_neighbors", "avg_num_chains", "evidence_recall_proxy"]:
        if col not in subset:
            subset[col] = 0.0
    subset["avg_selected_neighbors"] = subset.apply(_selected_neighbor_value, axis=1)
    table = (
        subset.groupby(["dataset", "method"], as_index=False)
        .agg(
            auprc=("auprc", "mean"),
            auroc=("auroc", "mean"),
            macro_f1=("macro_f1", "mean"),
            avg_selected_neighbors=("avg_selected_neighbors", "mean"),
            evidence_recall_proxy=("evidence_recall_proxy", "mean"),
        )
        .fillna(0.0)
    )
    return table.rename(columns={"method": "strategy"})


def _diagnostic_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if "method" in table:
        table = table.rename(columns={"method": "model"})
    if "evidence_necessity_score" in table and "evidence_necessity" not in table:
        table["evidence_necessity"] = table["evidence_necessity_score"]
    if "avg_evidence_necessity_all" not in table and "evidence_necessity" in table:
        table["avg_evidence_necessity_all"] = table["evidence_necessity"]

    columns = _diagnostic_columns()
    for col in columns:
        if col not in table:
            table[col] = 0.0
    return table[columns].fillna(0.0)


def _selected_neighbor_value(row: pd.Series) -> float:
    if float(row.get("avg_selected_neighbors", 0.0)) > 0:
        return float(row["avg_selected_neighbors"])
    if row["method"] == "hero_gnn":
        return float(row.get("avg_num_chains", 0.0))
    if row["method"] in {"semsim_gnn", "rulehetero_gnn"}:
        return 10.0
    return 0.0


def _empty_main() -> pd.DataFrame:
    return pd.DataFrame(columns=["dataset", "method", "macro_f1_mean", "macro_f1_std", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std"])


def _empty_ablation() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "variant",
            "macro_f1_mean",
            "auroc_mean",
            "auprc_mean",
            "evidence_recall_proxy",
            "evidence_necessity",
        ]
    )


def _empty_neighbor() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "strategy",
            "auprc",
            "auroc",
            "macro_f1",
            "avg_selected_neighbors",
            "evidence_recall_proxy",
        ]
    )


def _empty_diagnostic() -> pd.DataFrame:
    return pd.DataFrame(columns=_diagnostic_columns())


def _diagnostic_columns() -> list[str]:
    return [
        "dataset",
        "model",
        "seed",
        "positive_rate_train",
        "positive_rate_val",
        "positive_rate_test",
        "pred_positive_rate",
        "best_threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "macro_f1",
        "auroc",
        "auprc",
        "avg_selected_neighbors",
        "evidence_recall_proxy",
        "evidence_necessity",
        "avg_evidence_necessity_all",
        "avg_evidence_necessity_pos",
        "avg_evidence_necessity_neg",
        "avg_num_chains",
    ]


if __name__ == "__main__":
    main()
