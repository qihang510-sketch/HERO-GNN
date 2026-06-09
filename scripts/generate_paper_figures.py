from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission paper figures from real tables.")
    parser.add_argument("--tables_dir", default="outputs/paper_tables_submission")
    parser.add_argument("--output_dir", default="outputs/paper_figures_submission")
    parser.add_argument("--paper_figures_dir", default="paper/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    _figure1(output_dir)
    _figure2(output_dir)
    _figure3(tables_dir, output_dir, skipped)
    _figure4(tables_dir, output_dir, skipped)
    _figure5(tables_dir, output_dir, skipped)
    _figure6(tables_dir, output_dir, skipped)
    _figure7(output_dir, skipped)
    _figure8(tables_dir, output_dir, skipped)
    for path in output_dir.glob("*.pdf"):
        shutil.copyfile(path, paper_dir / path.name)
    for path in output_dir.glob("*.png"):
        shutil.copyfile(path, paper_dir / path.name)
    (output_dir / "skipped_figures_report.md").write_text("\n".join(f"- {item}" for item in skipped) + "\n", encoding="utf-8")
    print(f"Wrote figures to {output_dir}; copied available figures to {paper_dir}")


def _figure1(output_dir: Path) -> None:
    labels = ["risk-relevant", "irrelevant/noisy"]
    values = [906, 3254]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.42})
    axes[0].set_title("Risk-card relevance")
    case_labels = ["C1\nbehavioral\ncontradiction", "C2\ncounterparty\nrisk", "C3\nirrelevant\nheterophily"]
    axes[1].bar(case_labels, [1, 1, 1], color=["#4c78a8", "#72b7b2", "#bab0ac"])
    axes[1].set_ylim(0, 1.25)
    axes[1].set_yticks([])
    axes[1].set_title("Representative cases")
    _save(fig, output_dir, "fig_motivating_observation")


def _figure2(output_dir: Path) -> None:
    steps = [
        "Input graph",
        "Candidate retrieval",
        "Risk card builder",
        "LLM annotation",
        "Risk-relevant filter",
        "Dual-branch encoder",
        "Fraud prediction",
        "Top evidence chains",
    ]
    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.axis("off")
    for idx, step in enumerate(steps):
        x = idx / max(len(steps) - 1, 1)
        ax.text(x, 0.55, step, ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#4c78a8"})
        if idx < len(steps) - 1:
            ax.annotate("", xy=((idx + 0.72) / (len(steps) - 1), 0.55), xytext=((idx + 0.28) / (len(steps) - 1), 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("HERO-GNN framework", fontsize=11)
    _save(fig, output_dir, "fig_framework_hero_gnn")


def _figure3(tables_dir: Path, output_dir: Path, skipped: list[str]) -> None:
    table = _read_table(tables_dir / "table_text_rich_main.csv")
    methods = ["graphsage", "care_gnn", "graphconsis", "dgp", "mled", "hero_gnn"]
    if table.empty or not _has_methods(table, methods):
        skipped.append("fig_main_results_bar: missing text-rich main results for required methods.")
        return
    if pd.to_numeric(table[table["model"].isin(methods)]["AUPRC_mean"], errors="coerce").dropna().empty:
        skipped.append("fig_main_results_bar: required methods exist but metrics are NA.")
        return
    _grouped_bar(table, methods, "AUPRC_mean", output_dir, "fig_main_results_bar", "Text-rich AUPRC")


def _figure4(tables_dir: Path, output_dir: Path, skipped: list[str]) -> None:
    table = _read_table(tables_dir / "table_ablation.csv")
    metric_col = _metric_column(table, "AUPRC")
    if table.empty or metric_col is None:
        skipped.append("fig_ablation_drop_bar: missing ablation table.")
        return
    model_col = "variant" if "variant" in table else "model"
    label_col = "ablation" if "ablation" in table else model_col
    hero = table[table[model_col].astype(str).isin(["hero_gnn", "HERO-GNN"])]
    if hero.empty:
        skipped.append("fig_ablation_drop_bar: missing HERO-GNN reference.")
        return
    rows = []
    for dataset, group in table.groupby("dataset"):
        hero_row = group[group[model_col].astype(str).isin(["hero_gnn", "HERO-GNN"])]
        if hero_row.empty:
            continue
        ref = pd.to_numeric(hero_row.iloc[0].get(metric_col), errors="coerce")
        if pd.isna(ref):
            continue
        for _, row in group.iterrows():
            variant = str(row.get(model_col, ""))
            if variant in {"hero_gnn", "HERO-GNN"}:
                continue
            value = pd.to_numeric(row.get(metric_col), errors="coerce")
            if pd.isna(value):
                continue
            rows.append({"label": str(row.get(label_col, variant)), "drop": float(ref) - float(value)})
    drop_table = pd.DataFrame(rows)
    if drop_table.empty:
        skipped.append("fig_ablation_drop_bar: ablation metrics are all NA.")
        return
    drop_table = drop_table.groupby("label", as_index=False)["drop"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(drop_table["label"], drop_table["drop"])
    ax.set_ylabel("AUPRC drop")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_dir, "fig_ablation_drop_bar")


def _figure5(tables_dir: Path, output_dir: Path, skipped: list[str]) -> None:
    table = _read_table(tables_dir / "table_llm_coverage_sensitivity.csv")
    coverage_col = "coverage" if "coverage" in table else ("llm_label_coverage_rate" if "llm_label_coverage_rate" in table else "")
    if table.empty or not coverage_col:
        skipped.append("fig_llm_coverage_curve: missing LLM coverage table.")
        return
    x = pd.to_numeric(table[coverage_col], errors="coerce")
    if x.dropna().empty:
        skipped.append("fig_llm_coverage_curve: coverage values are NA.")
        return
    table = table.assign(_coverage_x=x).sort_values("_coverage_x")
    x = table["_coverage_x"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    plotted = False
    for metric in ["Macro-F1", "AUROC", "AUPRC"]:
        col = _metric_column(table, metric)
        if col is not None:
            y = pd.to_numeric(table[col], errors="coerce")
            if not y.dropna().empty:
                ax.plot(x, y, marker="o", label=metric)
                plotted = True
    if not plotted:
        plt.close(fig)
        skipped.append("fig_llm_coverage_curve: no numeric metric columns found.")
        return
    ax.set_xlabel("Qwen coverage")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    _save(fig, output_dir, "fig_llm_coverage_curve")


def _figure6(tables_dir: Path, output_dir: Path, skipped: list[str]) -> None:
    table = _read_table(tables_dir / "table_transaction_benchmark.csv")
    methods = ["graphsage", "bwgnn", "linkx", "hogrl", "rgtan", "hero_official"]
    if table.empty or not _has_methods(table, methods):
        skipped.append("fig_transaction_results_dotplot: missing Elliptic transaction results.")
        return
    metric_col = _metric_column(table, "AUPRC")
    if metric_col is None:
        skipped.append("fig_transaction_results_dotplot: transaction table lacks AUPRC metric.")
        return
    subset = table[table["model"].isin(methods)].copy()
    subset[metric_col] = pd.to_numeric(subset[metric_col], errors="coerce")
    if subset[metric_col].dropna().empty:
        skipped.append("fig_transaction_results_dotplot: Elliptic methods are all skipped or metrics are NA.")
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.scatter(subset[metric_col], subset["model"])
    ax.set_xlabel("AUPRC")
    _save(fig, output_dir, "fig_transaction_results_dotplot")


def _figure7(output_dir: Path, skipped: list[str]) -> None:
    case = _load_evidence_case(output_dir)
    if case is None:
        table_rows = _load_textual_evidence_case_rows()
        if table_rows:
            _write_evidence_case_table(output_dir, table_rows)
            skipped.append("fig_evidence_chain_case: no real routing_weight was found; wrote table_evidence_chain_case instead.")
        else:
            skipped.append("fig_evidence_chain_case: no real evidence-chain case with rationale was found.")
        return
    (output_dir / "evidence_chain_case_source.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
    chains = case["chains"][:3]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axis("off")
    ax.text(0.05, 0.5, f"target\n{case.get('target_id', case.get('target_idx', ''))}", ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#4c78a8"})
    for idx, chain in enumerate(chains):
        y = 0.82 - idx * 0.28
        nodes = chain.get("chain_nodes", [])
        neighbor = str(chain.get("neighbor_id", nodes[-1] if nodes else chain.get("neighbor_idx", "neighbor")))
        ax.annotate("", xy=(0.3, y), xytext=(0.12, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.0})
        ax.text(0.45, y, f"{neighbor}\n{chain.get('mechanism', '')}", ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#72b7b2"})
        ax.text(0.78, y, f"w={float(chain['routing_weight']):.3f}\n{str(chain['rationale'])[:48]}", ha="center", va="center", fontsize=7)
    _save(fig, output_dir, "fig_evidence_chain_case")


def _figure8(tables_dir: Path, output_dir: Path, skipped: list[str]) -> None:
    table = _read_table(tables_dir / "table_text_rich_main.csv")
    methods = ["dgp", "mled", "care_gnn", "graphconsis", "hero_gnn"]
    if table.empty or not _has_methods(table, methods):
        skipped.append("fig_seed_stability_pointplot: missing 5-seed main results.")
        return
    subset = table[table["model"].isin(methods)].copy()
    if pd.to_numeric(subset["AUPRC_mean"], errors="coerce").dropna().empty:
        skipped.append("fig_seed_stability_pointplot: methods exist but metrics are NA.")
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(subset.shape[0])
    means = pd.to_numeric(subset["AUPRC_mean"], errors="coerce")
    stds = pd.to_numeric(subset["AUPRC_std"], errors="coerce").fillna(0.0)
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4)
    ax.set_xticks(x, subset["model"], rotation=25, ha="right")
    ax.set_ylabel("AUPRC mean +/- std")
    _save(fig, output_dir, "fig_seed_stability_pointplot")


def _grouped_bar(table: pd.DataFrame, methods: list[str], metric: str, output_dir: Path, stem: str, title: str) -> None:
    datasets = list(dict.fromkeys(table["dataset"].astype(str).tolist()))
    x = np.arange(len(datasets))
    width = 0.12
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for idx, method in enumerate(methods):
        values = []
        for dataset in datasets:
            row = table[(table["dataset"] == dataset) & (table["model"] == method)]
            values.append(float(pd.to_numeric(row.iloc[0][metric], errors="coerce")) if not row.empty else np.nan)
        ax.bar(x + (idx - len(methods) / 2) * width, values, width=width, label=method)
    if table[table["model"].isin(methods)][metric].pipe(pd.to_numeric, errors="coerce").dropna().empty:
        plt.close(fig)
        return
    ax.set_xticks(x, datasets)
    ax.set_ylabel(metric.replace("_mean", ""))
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save(fig, output_dir, stem)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_evidence_case(output_dir: Path) -> dict | None:
    candidates = [
        output_dir / "evidence_chain_case_source.json",
        Path("data/processed/yelp_academic/evidence_chains.jsonl"),
        *Path("outputs").glob("**/*case*.jsonl"),
        *Path("outputs").glob("**/*case*.json"),
    ]
    for path in candidates:
        if not path.exists() or path.is_dir():
            continue
        for payload in _json_records(path):
            chains = payload.get("chains") if isinstance(payload, dict) else None
            if chains is None and isinstance(payload, dict):
                chains = payload.get("top_chains")
            if not chains:
                continue
            normalized = []
            for chain in chains:
                if not isinstance(chain, dict):
                    continue
                if "routing_weight" not in chain or "rationale" not in chain:
                    continue
                normalized.append(chain)
            if normalized:
                return {**payload, "chains": normalized}
    return None


def _load_textual_evidence_case_rows() -> list[dict[str, object]]:
    candidates = [
        Path("data/processed/yelp_academic/evidence_chains.jsonl"),
        Path("data/processed/yelp_academic/llm_labels.jsonl"),
        *Path("outputs").glob("**/examples.jsonl"),
    ]
    rows: list[dict[str, object]] = []
    for path in candidates:
        if not path.exists() or path.is_dir():
            continue
        for payload in _json_records(path):
            if not isinstance(payload, dict):
                continue
            rows.extend(_textual_rows_from_payload(payload))
            if len(rows) >= 5:
                return rows[:5]
    return rows[:5]


def _textual_rows_from_payload(payload: dict) -> list[dict[str, object]]:
    chains = payload.get("chains") or payload.get("top_chains")
    if isinstance(chains, list):
        target_id = payload.get("target_id", payload.get("target_idx", ""))
        rows = []
        for chain in chains:
            if not isinstance(chain, dict) or "rationale" not in chain:
                continue
            nodes = chain.get("chain_nodes", [])
            row_target = chain.get("target_id", target_id or (nodes[0] if nodes else ""))
            rows.append(
                {
                    "target_id": row_target,
                    "neighbor_id": chain.get("neighbor_id", chain.get("neighbor_idx", nodes[-1] if nodes else "")),
                    "mechanism": chain.get("mechanism", ""),
                    "risk_relevance": chain.get("risk_relevance", ""),
                    "confidence": chain.get("confidence", ""),
                    "rationale": chain.get("rationale", ""),
                    "kept_or_filtered": chain.get("kept_or_filtered", "kept"),
                }
            )
        return rows
    if {"target_id", "neighbor_id", "rationale"}.issubset(payload):
        return [
            {
                "target_id": payload.get("target_id", ""),
                "neighbor_id": payload.get("neighbor_id", ""),
                "mechanism": payload.get("mechanism", ""),
                "risk_relevance": payload.get("risk_relevance", ""),
                "confidence": payload.get("confidence", ""),
                "rationale": payload.get("rationale", ""),
                "kept_or_filtered": payload.get("kept_or_filtered", "candidate"),
            }
        ]
    return []


def _write_evidence_case_table(output_dir: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    stem = output_dir / "table_evidence_chain_case"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(_to_markdown(frame), encoding="utf-8")
    stem.with_suffix(".tex").write_text(frame.to_latex(index=False, escape=True), encoding="utf-8")


def _json_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
    except Exception:
        pass
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _has_methods(table: pd.DataFrame, methods: list[str]) -> bool:
    if "model" not in table:
        return False
    present = set(table["model"].dropna().astype(str))
    return set(methods).issubset(present)


def _metric_column(table: pd.DataFrame, metric: str) -> str | None:
    if table.empty:
        return None
    candidates = [
        f"{metric}_mean",
        metric,
        metric.lower(),
        f"{metric.lower()}_mean",
        metric.replace("-", "_"),
        metric.replace("-", "_").lower(),
        f"{metric.replace('-', '_').lower()}_mean",
    ]
    for candidate in candidates:
        if candidate in table.columns:
            return candidate
    return None


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    if not columns:
        return "\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
