from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.submission import DATASET_MODEL_MATRIX, FORBIDDEN_SUBMISSION_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether submission experiments are paper-ready.")
    parser.add_argument("--results_dir", default="outputs/submission_experiments")
    parser.add_argument("--tables_dir", default="outputs/paper_tables_submission")
    parser.add_argument("--figures_dir", default="outputs/paper_figures_submission")
    parser.add_argument("--output", default="outputs/submission_readiness_report.md")
    parser.add_argument("--min_seeds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    checks = []
    metrics = _read_metrics(results_dir)
    all_results = _read_table(tables_dir / "table_all_results.csv")
    text_table = _read_table(tables_dir / "table_text_rich_main.csv")
    official_table = _read_table(tables_dir / "table_official_benchmark.csv")
    transaction_table = _read_table(tables_dir / "table_transaction_benchmark.csv")
    coverage_table = _read_table(tables_dir / "table_llm_coverage_sensitivity.csv")

    checks.append(_check_forbidden_names(results_dir, metrics))
    checks.append(_check_seed_counts(all_results, ["yelp_academic", "amazon_video"], args.min_seeds, "Text-rich proxy has 5 seeds"))
    checks.append(_check_seed_counts(all_results, ["fraud_yelp", "fraud_amazon"], args.min_seeds, "Official benchmark has 5 seeds"))
    checks.append(_check_seed_counts(all_results, ["elliptic"], args.min_seeds, "Elliptic has 5 seeds"))
    checks.append(_check_rank(text_table, "hero_gnn", "AUPRC_mean", "HERO-GNN text-rich AUPRC average rank first"))
    checks.append(_check_beats(text_table, "hero_gnn", ["dgp", "mled"], ["AUPRC_mean", "Macro-F1_mean"], "HERO-GNN beats DGP/MLED on AUPRC and Macro-F1"))
    checks.append(_check_beats(text_table, "hero_gnn", ["care_gnn", "graphconsis", "pc_gnn", "bwgnn"], ["AUPRC_mean", "Macro-F1_mean"], "HERO-GNN beats classic fraud baselines"))
    checks.append(_check_rank(official_table, "hero_official", "AUPRC_mean", "HERO-official official benchmark average rank top two", max_rank=2))
    checks.append(_check_beats(transaction_table, "hero_official", ["graphsage", "bwgnn", "linkx"], ["AUPRC_mean"], "HERO-official beats generic/anomaly baselines on Elliptic"))
    checks.append(_check_coverage(coverage_table))
    checks.append(_check_file(tables_dir / "table_significance_tests.csv", "Significance tests generated"))
    checks.append(_check_figures(figures_dir))
    checks.append(_check_na(all_results))
    checks.append(_check_std_source(all_results, args.min_seeds))
    checks.append(_check_split_consistency(metrics))

    status = _overall_status(checks)
    report = _render_report(status, checks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote readiness report to {out}: {status}")


def _read_metrics(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_dir.glob("*/*/seed_*/metrics.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _check_forbidden_names(results_dir: Path, metrics: list[dict[str, Any]]) -> dict[str, str]:
    names = {str(row.get("model", "")).lower() for row in metrics}
    paths = [str(path) for path in results_dir.glob("**/*") if "lite" in str(path).lower()]
    bad = sorted(name for name in names if name in FORBIDDEN_SUBMISSION_NAMES or "lite" in name)
    if bad or paths:
        return _fail("No lite or FLAG-lite names in submission outputs", f"forbidden={bad}; paths={paths[:5]}")
    return _pass("No lite or FLAG-lite names in submission outputs")


def _check_seed_counts(table: pd.DataFrame, datasets: list[str], min_seeds: int, label: str) -> dict[str, str]:
    if table.empty:
        return _fail(label, "table_all_results.csv is missing or empty")
    subset = table[table["dataset"].isin(datasets)]
    bad = subset[pd.to_numeric(subset["seed_count"], errors="coerce").fillna(0) < min_seeds]
    if not bad.empty:
        return _fail(label, "; ".join(f"{row.dataset}/{row.model}:{row.seed_count}" for row in bad.itertuples()))
    return _pass(label)


def _check_rank(table: pd.DataFrame, model: str, metric: str, label: str, max_rank: int = 1) -> dict[str, str]:
    if table.empty or metric not in table:
        return _fail(label, "required table or metric missing")
    failures = []
    for dataset, group in table.groupby("dataset"):
        group = group.copy()
        group[metric] = pd.to_numeric(group[metric], errors="coerce")
        group["rank"] = group[metric].rank(ascending=False, method="min", na_option="bottom")
        row = group[group["model"] == model]
        if row.empty or float(row.iloc[0]["rank"]) > max_rank:
            failures.append(str(dataset))
    return _fail(label, ",".join(failures)) if failures else _pass(label)


def _check_beats(table: pd.DataFrame, hero: str, baselines: list[str], metrics: list[str], label: str) -> dict[str, str]:
    if table.empty:
        return _fail(label, "required table missing")
    failures = []
    for dataset, group in table.groupby("dataset"):
        hero_row = group[group["model"] == hero]
        if hero_row.empty:
            failures.append(f"{dataset}:missing_{hero}")
            continue
        for metric in metrics:
            if metric not in group:
                failures.append(f"{dataset}:missing_{metric}")
                continue
            hero_value = pd.to_numeric(hero_row.iloc[0][metric], errors="coerce")
            for baseline in baselines:
                base_row = group[group["model"] == baseline]
                if base_row.empty:
                    failures.append(f"{dataset}:missing_{baseline}")
                    continue
                base_value = pd.to_numeric(base_row.iloc[0][metric], errors="coerce")
                if pd.isna(hero_value) or pd.isna(base_value) or hero_value <= base_value:
                    failures.append(f"{dataset}:{metric}:{hero}<={baseline}")
    return _fail(label, "; ".join(failures)) if failures else _pass(label)


def _check_coverage(table: pd.DataFrame) -> dict[str, str]:
    label = "Qwen high-coverage outperforms rule/mock"
    if table.empty or "labeler" not in table:
        return _fail(label, "coverage table missing")
    qwen = table[table["labeler"].astype(str).str.contains("qwen", case=False, na=False)]
    rule = table[table["labeler"].astype(str).str.contains("rule|mock", case=False, na=False)]
    if qwen.empty or rule.empty:
        return _fail(label, "missing qwen or rule/mock rows")
    metric = "AUPRC_mean" if "AUPRC_mean" in table else "AUPRC"
    if metric not in table:
        return _fail(label, "missing AUPRC metric")
    if pd.to_numeric(qwen[metric], errors="coerce").max() <= pd.to_numeric(rule[metric], errors="coerce").max():
        return _fail(label, "qwen is not higher than rule/mock")
    return _pass(label)


def _check_file(path: Path, label: str) -> dict[str, str]:
    return _pass(label) if path.exists() else _fail(label, f"missing {path}")


def _check_figures(figures_dir: Path) -> dict[str, str]:
    expected = [f"fig_{name}.pdf" for name in [
        "motivating_observation",
        "framework_hero_gnn",
        "main_results_bar",
        "ablation_drop_bar",
        "llm_coverage_curve",
        "transaction_results_dotplot",
        "evidence_chain_case",
        "seed_stability_pointplot",
    ]]
    missing = [name for name in expected if not (figures_dir / name).exists()]
    return _fail("Figures 1-8 generated", f"missing={missing}") if missing else _pass("Figures 1-8 generated")


def _check_na(table: pd.DataFrame) -> dict[str, str]:
    if table.empty:
        return _fail("NA results are explicitly visible", "all_results table missing")
    has_na = table.astype(str).apply(lambda col: col.str.contains("NA|missing|skipped", case=False, na=False)).any().any()
    return _warn("NA results are explicitly visible", "NA/skipped values present") if has_na else _pass("NA results are explicitly visible")


def _check_std_source(table: pd.DataFrame, min_seeds: int) -> dict[str, str]:
    if table.empty:
        return _fail("All std values come from real seeds", "all_results table missing")
    bad = table[(pd.to_numeric(table["seed_count"], errors="coerce").fillna(0) < min_seeds) & table.filter(like="_std").notna().any(axis=1)]
    if not bad.empty:
        return _fail("All std values come from real seeds", "some std columns exist with insufficient seeds")
    return _pass("All std values come from real seeds")


def _check_split_consistency(metrics: list[dict[str, Any]]) -> dict[str, str]:
    split_ids = {(row.get("dataset"), row.get("model"), row.get("split_id", "default")) for row in metrics}
    bad = {}
    for dataset, model, split in split_ids:
        bad.setdefault((dataset, model), set()).add(split)
    inconsistent = [f"{dataset}/{model}" for (dataset, model), splits in bad.items() if len(splits) > 1]
    return _fail("All data splits are consistent", ",".join(inconsistent)) if inconsistent else _pass("All data splits are consistent")


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARNING" for check in checks):
        return "WARNING"
    return "PASS"


def _render_report(status: str, checks: list[dict[str, str]]) -> str:
    lines = [f"# Submission Readiness Report", "", f"Overall status: **{status}**", ""]
    for idx, check in enumerate(checks, start=1):
        detail = f" - {check['detail']}" if check.get("detail") else ""
        lines.append(f"{idx}. **{check['status']}** {check['label']}{detail}")
    if status == "FAIL":
        lines.extend(["", "Do not present this as submission-ready. Re-run missing experiments or tune failed model modules first."])
    return "\n".join(lines) + "\n"


def _pass(label: str, detail: str = "") -> dict[str, str]:
    return {"status": "PASS", "label": label, "detail": detail}


def _warn(label: str, detail: str = "") -> dict[str, str]:
    return {"status": "WARNING", "label": label, "detail": detail}


def _fail(label: str, detail: str = "") -> dict[str, str]:
    return {"status": "FAIL", "label": label, "detail": detail}


if __name__ == "__main__":
    main()
