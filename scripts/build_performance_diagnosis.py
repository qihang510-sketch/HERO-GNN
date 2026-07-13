from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_RICH = ["yelp_academic", "amazon_video"]
TRANSFER = ["fraud_yelp", "fraud_amazon", "elliptic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HERO performance diagnosis from real summary CSVs.")
    parser.add_argument("--output_dir", default="outputs/experiment_suite_main")
    parser.add_argument("--tuned_config_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_performance_diagnosis(args.output_dir, tuned_config_dir=args.tuned_config_dir)


def build_performance_diagnosis(output_dir: str | Path, tuned_config_dir: str | Path | None = None) -> Path:
    output_dir = Path(output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(summary_dir / "all_raw_runs.csv")
    tuned = _read_tuned_configs(tuned_config_dir)
    lines = [
        "# Performance Diagnosis",
        "",
        "This report is generated from raw/summary outputs only. Do not fabricate results or edit metrics by hand.",
        "",
        "## Tuned HERO Configs",
    ]
    if tuned.empty:
        lines.append("- tuned config: missing or not provided")
    else:
        for _, row in tuned.iterrows():
            dataset = str(row.get("dataset", ""))
            status = str(row.get("status", ""))
            config_hash = str(row.get("config_hash", ""))
            best_epoch = row.get("best_epoch", "NA")
            val_auprc = row.get("val_AUPRC_mean", row.get("val_AUPRC", "NA"))
            lines.append(f"- {dataset}: status={status}, selected_by=validation_AUPRC, best_epoch={best_epoch}, val_AUPRC={val_auprc}, config_hash={config_hash}")
    lines.extend(["", "## HERO vs Strongest Baseline"])
    comparisons = _hero_baseline_comparisons(raw)
    if comparisons.empty:
        lines.append("- missing: summary/all_raw_runs.csv does not contain comparable HERO and baseline AUPRC rows")
    else:
        for _, row in comparisons.iterrows():
            trend = "improved" if float(row["delta_auprc"]) > 0 else "not_improved"
            lines.append(
                f"- {row['dataset']}: HERO AUPRC={row['hero_auprc']:.6f}, strongest baseline={row['baseline_model']} "
                f"AUPRC={row['baseline_auprc']:.6f}, delta={row['delta_auprc']:.6f}, status={trend}"
            )
    weak = comparisons[comparisons["delta_auprc"] <= 0] if not comparisons.empty else pd.DataFrame()
    lines.extend(["", "## Weak Datasets"])
    if weak.empty:
        lines.append("- none detected from available comparable AUPRC rows")
    else:
        for _, row in weak.iterrows():
            lines.append(f"- {row['dataset']}: HERO trails {row['baseline_model']} by {abs(float(row['delta_auprc'])):.6f} AUPRC")
    lines.extend(
        [
            "",
            "## Possible Causes To Check",
            "- tuned config missing or not loaded for HERO.",
            "- text/rating/time/relation features unavailable in processed data.",
            "- LLM cache unavailable, causing proxy/rule fallback.",
            "- risk_weight_temperature or risk_relevance_threshold too aggressive for heterophilic edges.",
            "- class imbalance remains severe; validate class_weight_strategy on validation only.",
            "",
            "## Integrity Reminder",
            "- Do not tune on test metrics.",
            "- Do not hand-edit CSV metrics.",
            "- Re-run with validation-selected configs when results are weak.",
        ]
    )
    path = summary_dir / "performance_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _hero_baseline_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "dataset" not in raw or "model" not in raw or "AUPRC" not in raw:
        return pd.DataFrame()
    frame = raw.copy()
    status = frame["status"] if "status" in frame else pd.Series(["ok"] * len(frame), index=frame.index)
    frame = frame[status.astype(str).isin(["ok", "exists", ""])]
    frame["AUPRC"] = pd.to_numeric(frame["AUPRC"], errors="coerce")
    frame = frame.dropna(subset=["AUPRC"])
    rows: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset", dropna=False):
        model_means = group.groupby("model")["AUPRC"].mean().sort_values(ascending=False)
        hero_models = [model for model in model_means.index if str(model).startswith("hero") or str(model).lower() == "hero"]
        baseline_models = [model for model in model_means.index if model not in hero_models]
        if not hero_models or not baseline_models:
            continue
        hero_model = "hero_full" if "hero_full" in hero_models else hero_models[0]
        baseline_model = baseline_models[0]
        rows.append(
            {
                "dataset": str(dataset),
                "hero_model": str(hero_model),
                "hero_auprc": float(model_means[hero_model]),
                "baseline_model": str(baseline_model),
                "baseline_auprc": float(model_means[baseline_model]),
                "delta_auprc": float(model_means[hero_model] - model_means[baseline_model]),
            }
        )
    return pd.DataFrame(rows)


def _read_tuned_configs(root: str | Path | None) -> pd.DataFrame:
    if not root:
        return pd.DataFrame()
    return _read_csv(Path(root) / "summary" / "best_hero_configs.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


if __name__ == "__main__":
    main()
