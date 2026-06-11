from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.trainer import HERO_CONFIG_KEYS  # noqa: E402


METRIC_MAP = {
    "auprc": "AUPRC_mean",
    "ap": "AUPRC_mean",
    "macro_f1": "Macro-F1_mean",
    "macro-f1": "Macro-F1_mean",
    "f1": "Macro-F1_mean",
    "auroc": "AUROC_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote the best HERO tuning config to configs/models/*.yaml.")
    parser.add_argument("--tuning_dir", required=True)
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuning_dir = Path(args.tuning_dir)
    results_path = tuning_dir / "tuning_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing tuning results: {results_path}")
    frame = pd.read_csv(results_path)
    ranked = _rank(frame, args.metric)
    top_k = ranked.head(max(1, int(args.top_k))).copy()
    if top_k.empty:
        raise ValueError("No tuning rows available to promote.")
    best = top_k.iloc[0].to_dict()
    best_params = _parse_config(best)
    output_config = Path(args.output_config)
    if output_config.exists() and not args.overwrite:
        raise FileExistsError(f"{output_config} exists. Pass --overwrite to replace it.")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    promoted = _promoted_payload(best, best_params, source=str(results_path))
    output_config.write_text(yaml.safe_dump(promoted, sort_keys=False), encoding="utf-8")
    top_payload = [_json_ready(_row_with_config(row)) for row in top_k.to_dict(orient="records")]
    (tuning_dir / "top_promoted_configs.json").write_text(json.dumps(top_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(tuning_dir, output_config, top_k, args.metric)
    print(f"Promoted HERO config: {output_config}")


def _rank(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    primary = METRIC_MAP.get(str(metric).lower(), "AUPRC_mean")
    for column in [primary, "Macro-F1_mean", "AUROC_mean"]:
        if column not in frame:
            frame[column] = float("nan")
    return frame.sort_values([primary, "Macro-F1_mean", "AUROC_mean"], ascending=False, na_position="last").reset_index(drop=True)


def _parse_config(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("config_json", "{}")
    if isinstance(raw, str):
        params = json.loads(raw)
    else:
        params = dict(raw or {})
    return params


def _promoted_payload(row: dict[str, Any], params: dict[str, Any], source: str) -> dict[str, Any]:
    hero_config = {key: value for key, value in params.items() if key in HERO_CONFIG_KEYS}
    payload = {
        "name": "hero_gnn_tuned",
        "base_model": row.get("model", "hero_gnn"),
        "display_name": "HERO-GNN tuned",
        "implementation_source": "project",
        "tuning_source": source,
        "trial_id": int(row.get("trial_id", 0)),
        "datasets": str(row.get("datasets", "")),
        "seeds": str(row.get("seeds", "")),
        "ranking_metrics": {
            "AUPRC_mean": _float_or_none(row.get("AUPRC_mean")),
            "Macro-F1_mean": _float_or_none(row.get("Macro-F1_mean")),
            "AUROC_mean": _float_or_none(row.get("AUROC_mean")),
        },
        "hero_config": hero_config,
    }
    for key in ["learning_rate", "hidden_dim", "top_k", "epochs"]:
        if key in params:
            payload[key] = params[key]
    return payload


def _row_with_config(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "config": _parse_config(row)}


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


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_report(tuning_dir: Path, output_config: Path, top_k: pd.DataFrame, metric: str) -> None:
    best = top_k.iloc[0].to_dict()
    primary = METRIC_MAP.get(str(metric).lower(), "AUPRC_mean")
    lines = [
        "# Promoted HERO config report",
        "",
        f"- output_config: {output_config}",
        f"- ranked_by: {primary}, then Macro-F1_mean, then AUROC_mean",
        f"- promoted_trial_id: {best.get('trial_id')}",
        f"- promoted_{primary}: {best.get(primary)}",
        f"- top_k: {len(top_k)}",
        "",
        "The original configs/models/hero_gnn.yaml is not modified by this script.",
    ]
    (tuning_dir / "promoted_config_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
