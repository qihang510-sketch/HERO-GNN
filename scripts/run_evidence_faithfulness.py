from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.advanced_experiment_utils import METRICS, metric_summary, write_frame  # noqa: E402
from scripts.paper_artifact_utils import import_matplotlib, save_figure, write_latex  # noqa: E402
from src.data import schema  # noqa: E402
from src.data.loader import load_processed_data  # noqa: E402
from src.graph.neighbor_retrieval import filter_topk_semantic_edges  # noqa: E402
from src.training.evaluator import binary_classification_metrics  # noqa: E402
from src.training.submission import resolve_processed_dir  # noqa: E402
from src.training.trainer import (  # noqa: E402
    _chain_feature_matrix,
    _hetero_feature_matrix,
    _neighbor_mean_features,
    _split_hero_features_torch,
    _valid_label_indices,
)
from src.utils.io import write_json  # noqa: E402


SETTINGS = ("original_graph", "remove_topk_evidence", "remove_random_edges", "remove_irrelevant_edges", "keep_only_topk_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HERO evidence-chain faithfulness from saved checkpoints.")
    parser.add_argument("--datasets", nargs="+", default=["yelp_academic", "amazon_video"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--input_dir", default=None, help="Existing suite output root. Defaults to --output_dir.")
    parser.add_argument("--output_dir", default="outputs/submission_unified")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_faithfulness(args)


def run_faithfulness(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    input_dir = Path(args.input_dir or args.output_dir)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = resolve_processed_dir(dataset, args.data_root)
        for seed in args.seeds:
            checkpoint_path = find_checkpoint(input_dir, dataset, int(seed), args.checkpoint_dir)
            if checkpoint_path is None:
                rows.extend(_unavailable_rows(dataset, int(seed), args.topks, args.settings, "checkpoint_missing"))
                continue
            try:
                evaluator = CheckpointFaithfulnessEvaluator(
                    dataset=dataset,
                    seed=int(seed),
                    data_dir=data_dir,
                    checkpoint_path=checkpoint_path,
                    device=args.device,
                )
                rows.extend(evaluator.evaluate(topks=args.topks, settings=args.settings))
            except Exception as exc:
                rows.extend(_unavailable_rows(dataset, int(seed), args.topks, args.settings, f"checkpoint_eval_failed: {type(exc).__name__}: {exc}"))
    raw = pd.DataFrame(rows)
    summary = summarize_faithfulness(raw)
    table = table_faithfulness(summary)
    write_frame(output_dir / "summary" / "faithfulness_raw.csv", raw)
    write_frame(output_dir / "summary" / "faithfulness_summary.csv", summary)
    write_frame(output_dir / "summary" / "table_faithfulness.csv", table)
    write_latex(output_dir / "summary" / "table_faithfulness.tex", table)
    write_frame(output_dir / "tables" / "table_faithfulness.csv", table)
    write_latex(output_dir / "tables" / "table_faithfulness.tex", table)
    plot_data = faithfulness_plot_data(summary)
    write_frame(output_dir / "figure_data" / "faithfulness_bar_data.csv", plot_data)
    write_frame(output_dir / "figures" / "faithfulness_bar_data.csv", plot_data)
    plot_faithfulness(output_dir)
    _write_raw_artifacts(output_dir, rows)
    return rows


class CheckpointFaithfulnessEvaluator:
    def __init__(self, dataset: str, seed: int, data_dir: Path, checkpoint_path: Path, device: str) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.device = _resolve_device(device)
        self.graph = load_processed_data(data_dir)
        self.checkpoint = _load_checkpoint(checkpoint_path, self.device)
        self.metrics = dict(self.checkpoint.get("metrics", {}))
        self.artifacts = dict(self.checkpoint.get("hero_artifacts", {}))
        if "model_state_dict" not in self.checkpoint:
            raise ValueError("checkpoint lacks model_state_dict; cannot do checkpoint faithfulness evaluation")
        if not self.artifacts:
            raise ValueError("checkpoint lacks hero_artifacts; cannot reconstruct evidence-chain inputs")
        self.feature_dims = dict(self.artifacts.get("feature_dims", {}))
        if not self.feature_dims:
            raise ValueError("checkpoint hero_artifacts lacks feature_dims")
        self.test_idx = _valid_label_indices(self.graph.split.get("test", np.array([], dtype=np.int64)), self.graph.labels)
        if self.test_idx.size == 0:
            raise ValueError("test split has no labeled nodes")
        self.model = self._load_model()
        self.original_scores = self._scores(self._chains_by_idx(), setting="original_graph", topk=None)
        threshold = float(self.metrics.get("best_threshold", 0.5))
        self.original_metrics = _metrics(self.graph.labels[self.test_idx], self.original_scores, threshold)

    def evaluate(self, topks: list[int], settings: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for topk in topks:
            for setting in settings:
                chains = self._perturbed_chains(setting=setting, topk=int(topk))
                scores = self._scores(chains, setting=setting, topk=int(topk))
                current = _metrics(self.graph.labels[self.test_idx], scores, float(self.metrics.get("best_threshold", 0.5)))
                row = {
                    "suite": "faithfulness",
                    "dataset": self.dataset,
                    "model": "hero_gnn",
                    "seed": self.seed,
                    "setting": setting,
                    "top_k": int(topk),
                    "status": "ok",
                    "checkpoint": str(self.checkpoint_path),
                    "perturbation_scope": "checkpoint_chain_inputs",
                    "original_prediction_probability_mean": float(np.mean(self.original_scores)),
                    "perturbed_prediction_probability_mean": float(np.mean(scores)),
                    "prediction_probability_drop": float(np.mean(self.original_scores) - np.mean(scores)),
                }
                for metric in METRICS:
                    row[metric] = current[metric]
                    row[f"{metric}_original"] = self.original_metrics[metric]
                    row[f"{metric}_drop"] = self.original_metrics[metric] - current[metric]
                row["comprehensiveness"] = row["prediction_probability_drop"] if setting.startswith("remove_") else pd.NA
                row["sufficiency"] = _sufficiency(row["AUPRC"], self.original_metrics["AUPRC"]) if setting == "keep_only_topk_evidence" else pd.NA
                rows.append(row)
        return rows

    def _load_model(self):
        try:
            import torch
            from src.models.hero_gnn import HEROGNN
        except Exception as exc:
            raise ValueError("torch and HEROGNN are required for checkpoint faithfulness") from exc
        state = self.checkpoint["model_state_dict"]
        hidden_dim = int(state["target_encoder.0.weight"].shape[0])
        model = HEROGNN(
            input_dim=int(self.feature_dims["target_dim"]),
            hidden_dim=hidden_dim,
            output_dim=1,
            num_mechanisms=len(schema.EVIDENCE_MECHANISMS),
            use_heterophily=bool(self.artifacts.get("use_hetero", self.metrics.get("use_hetero", True))),
            use_mechanism=bool(self.artifacts.get("use_mechanism", self.metrics.get("use_mechanism", True))),
            use_chain=bool(self.artifacts.get("use_chain", self.metrics.get("use_chain", True))),
            use_dual_branch_encoder=bool(self.metrics.get("use_dual_branch_encoder", True)),
            use_gated_fusion=bool(self.metrics.get("use_gated_fusion", True)),
            fusion_type=str(self.metrics.get("fusion_type", "gated")),
            hetero_input_dim=int(self.feature_dims["hetero_dim"]),
            mechanism_input_dim=int(self.feature_dims["mechanism_dim"]),
            chain_input_dim=int(self.feature_dims["chain_dim"]),
            dropout=0.0,
        ).to(self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def _scores(self, chains_by_idx: dict[int, list[dict[str, Any]]], setting: str, topk: int | None) -> np.ndarray:
        import torch

        features = self._features(chains_by_idx, setting=setting, topk=topk)
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        target_x, homo_x, hetero_x, mechanism_x, chain_x = _split_hero_features_torch(x, self.feature_dims)
        test_tensor = torch.tensor(self.test_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(target_x, homo_x, hetero_x, mechanism_x, chain_x)
            return torch.sigmoid(logits[test_tensor]).detach().cpu().numpy().astype(np.float32)

    def _features(self, chains_by_idx: dict[int, list[dict[str, Any]]], setting: str, topk: int | None) -> np.ndarray:
        homo_edges = filter_topk_semantic_edges(self.graph.edge_index, self.graph.text_features, top_k=5)
        homo = _neighbor_mean_features(self.graph.features, homo_edges)
        labels_by_target = self._labels_by_target()
        hetero, mechanism = _hetero_feature_matrix(
            graph=self.graph,
            labels_by_target=labels_by_target,
            use_mechanism=bool(self.artifacts.get("use_mechanism", True)),
            use_heterophily_filter=True,
            weight_mode=str(self.metrics.get("heterophily_weight_mode", "relevance_confidence")),
            min_risk_score=float(self.metrics.get("risk_relevance_threshold", 0.0)),
        )
        chain = _chain_feature_matrix(
            self.graph,
            chains_by_idx,
            use_mechanism=bool(self.artifacts.get("use_mechanism", True)),
            use_heterophily_filter=True,
            weight_mode=str(self.metrics.get("heterophily_weight_mode", "relevance_confidence")),
        )
        if setting == "keep_only_topk_evidence":
            hetero = np.zeros_like(hetero, dtype=np.float32)
            mechanism = np.zeros_like(mechanism, dtype=np.float32)
        return np.concatenate([self.graph.features, homo, hetero, mechanism, chain], axis=1).astype(np.float32)

    def _chains_by_idx(self) -> dict[int, list[dict[str, Any]]]:
        raw = self.artifacts.get("chains_by_idx", {})
        return {int(key): list(value) for key, value in raw.items()}

    def _labels_by_target(self) -> dict[int, list[dict[str, Any]]]:
        raw = self.artifacts.get("labels_by_target", {})
        return {int(key): list(value) for key, value in raw.items()}

    def _perturbed_chains(self, setting: str, topk: int) -> dict[int, list[dict[str, Any]]]:
        original = {
            target: sorted([dict(chain) for chain in chains], key=lambda item: _chain_rank(item), reverse=True)
            for target, chains in self._chains_by_idx().items()
        }
        if setting == "original_graph":
            return original
        if setting == "remove_topk_evidence":
            return {target: chains[topk:] for target, chains in original.items()}
        if setting == "keep_only_topk_evidence":
            return {target: chains[:topk] for target, chains in original.items()}
        if setting == "remove_irrelevant_edges":
            return {target: sorted(chains, key=lambda item: _chain_rank(item), reverse=True)[:-topk] if topk < len(chains) else [] for target, chains in original.items()}
        if setting == "remove_random_edges":
            rng = np.random.default_rng(self.seed + topk)
            out: dict[int, list[dict[str, Any]]] = {}
            for target, chains in original.items():
                if len(chains) <= topk:
                    out[target] = []
                    continue
                remove = set(int(idx) for idx in rng.choice(len(chains), size=topk, replace=False))
                out[target] = [chain for idx, chain in enumerate(chains) if idx not in remove]
            return out
        raise ValueError(f"Unknown faithfulness setting: {setting}")


def find_checkpoint(input_dir: Path, dataset: str, seed: int, explicit_dir: str | None = None) -> Path | None:
    model = "hero_gnn" if dataset in {"yelp_academic", "amazon_video"} else "hero_official"
    candidates = []
    if explicit_dir:
        root = Path(explicit_dir)
        candidates.extend([root / dataset / model / f"seed_{seed}" / "best.pt", root / dataset / "hero_gnn" / f"seed_{seed}" / "best.pt"])
    candidates.extend(
        [
            input_dir / "raw" / "_project_runs" / "checkpoints" / dataset / model / f"seed_{seed}" / "best.pt",
            input_dir / "raw" / "_project_runs" / "checkpoints" / dataset / "hero_gnn" / f"seed_{seed}" / "best.pt",
            input_dir / "checkpoints" / dataset / model / f"seed_{seed}" / "best.pt",
            input_dir / "checkpoints" / dataset / "hero_gnn" / f"seed_{seed}" / "best.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(input_dir.rglob(f"checkpoints*/{dataset}/*/seed_{seed}/best.pt"))
    return matches[0] if matches else None


def summarize_faithfulness(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if "status" in raw and raw["status"].astype(str).isin(["ok", "exists"]).sum() == 0:
        return _unavailable_faithfulness_summary(raw)
    summary = metric_summary(raw, ["dataset", "setting", "top_k"])
    drop_cols = ["prediction_probability_drop", *[f"{metric}_drop" for metric in METRICS], "comprehensiveness", "sufficiency"]
    rows = []
    for keys, group in raw[raw["status"].astype(str) == "ok"].groupby(["dataset", "setting", "top_k"], dropna=False):
        row = {name: value for name, value in zip(["dataset", "setting", "top_k"], keys)}
        for col in drop_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna() if col in group else pd.Series(dtype=float)
            row[f"{col}_mean"] = float(values.mean()) if not values.empty else pd.NA
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) >= 2 else pd.NA
        rows.append(row)
    drops = pd.DataFrame(rows)
    if summary.empty:
        return drops
    return summary.merge(drops, on=["dataset", "setting", "top_k"], how="left")


def table_faithfulness(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.copy()


def faithfulness_plot_data(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keep = [
        "dataset",
        "setting",
        "top_k",
        "prediction_probability_drop_mean",
        "AUROC_drop_mean",
        "AUPRC_drop_mean",
        "Macro-F1_drop_mean",
        "comprehensiveness_mean",
        "sufficiency_mean",
        "status",
        "skip_reason",
    ]
    for col in keep:
        if col not in summary:
            summary[col] = pd.NA
    return summary[keep]


def plot_faithfulness(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    path = output_dir / "figure_data" / "faithfulness_bar_data.csv"
    if not path.exists():
        path = output_dir / "figures" / "faithfulness_bar_data.csv"
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        frame = pd.DataFrame()
    plt = import_matplotlib()
    if frame.empty or "AUPRC_drop_mean" not in frame:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
        save_figure(fig, output_dir / "figures" / "fig_evidence_faithfulness.pdf", output_dir / "figures" / "fig_evidence_faithfulness.png")
        plt.close(fig)
        return
    status = frame["status"] if "status" in frame else pd.Series(["ok"] * len(frame), index=frame.index)
    ok = frame[status.astype(str).isin(["ok", "exists", ""])]
    ok = ok.copy()
    ok["AUPRC_drop_mean"] = pd.to_numeric(ok["AUPRC_drop_mean"], errors="coerce")
    ok["top_k"] = pd.to_numeric(ok["top_k"], errors="coerce")
    ok = ok.dropna(subset=["AUPRC_drop_mean", "top_k"])
    if ok.empty:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
        save_figure(fig, output_dir / "figures" / "fig_evidence_faithfulness.pdf", output_dir / "figures" / "fig_evidence_faithfulness.png")
        plt.close(fig)
        return
    settings = [str(value) for value in ok["setting"].dropna().unique()]
    topks = sorted(int(value) for value in ok["top_k"].dropna().unique())
    width = 0.8 / max(len(settings), 1)
    x = np.arange(len(topks), dtype=float)
    fig, ax = plt.subplots(figsize=(max(5.5, len(topks) * 1.2), 3.2))
    for offset, setting in enumerate(settings):
        values = []
        for topk in topks:
            subset = ok[(ok["setting"].astype(str) == setting) & (ok["top_k"].astype(int) == topk)]
            values.append(float(subset["AUPRC_drop_mean"].mean()) if not subset.empty else np.nan)
        ax.bar(x + (offset - (len(settings) - 1) / 2) * width, values, width=width, label=setting)
    ax.set_xticks(x)
    ax.set_xticklabels([str(topk) for topk in topks])
    ax.set_xlabel("top-k evidence edges")
    ax.set_ylabel("AUPRC drop")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "fig_evidence_faithfulness.pdf", output_dir / "figures" / "fig_evidence_faithfulness.png")
    plt.close(fig)


def _unavailable_faithfulness_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(["dataset", "setting", "top_k"], dropna=False):
        row = {name: value for name, value in zip(["dataset", "setting", "top_k"], keys)}
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        row["status"] = group["status"].dropna().iloc[0] if "status" in group and not group["status"].dropna().empty else "unavailable"
        row["skip_reason"] = group["skip_reason"].dropna().iloc[0] if "skip_reason" in group and not group["skip_reason"].dropna().empty else "checkpoint_or_evidence_missing"
        for metric in METRICS:
            row[f"{metric}_mean"] = pd.NA
            row[f"{metric}_std"] = pd.NA
            row[f"{metric}_mean_std"] = "--"
            row[f"{metric}_drop_mean"] = pd.NA
            row[f"{metric}_drop_std"] = pd.NA
        for col in ["prediction_probability_drop", "comprehensiveness", "sufficiency"]:
            row[f"{col}_mean"] = pd.NA
            row[f"{col}_std"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def _write_raw_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        run_dir = output_dir / "raw" / "faithfulness" / str(row["dataset"]) / str(row["setting"]) / f"topk_{row['top_k']}" / f"seed_{row['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "metrics.json", _json_safe(row))
        write_json(run_dir / "config.json", {"suite": "faithfulness", "dataset": row["dataset"], "model": "hero_gnn", "seed": int(row["seed"]), "setting": row["setting"], "top_k": int(row["top_k"])})
        write_json(run_dir / "runtime.json", {"status": row["status"], "reason": row.get("skip_reason", "")})
        (run_dir / "log.txt").write_text(str(row.get("skip_reason", "")) + "\n", encoding="utf-8")


def _unavailable_rows(dataset: str, seed: int, topks: list[int], settings: list[str], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "suite": "faithfulness",
            "dataset": dataset,
            "model": "hero_gnn",
            "seed": int(seed),
            "setting": setting,
            "top_k": int(topk),
            "status": "unavailable",
            "skip_reason": reason,
        }
        for topk in topks
        for setting in settings
    ]


def _load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    try:
        import torch

        try:
            return torch.load(path, map_location=device)
        except Exception:
            return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unsupported checkpoint payload: {path}")


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    payload = binary_classification_metrics(labels, scores, k=100, threshold=threshold)
    return {
        "Macro-F1": float(payload.get("macro_f1", 0.0)),
        "AUROC": float(payload.get("auroc", 0.0)),
        "AUPRC": float(payload.get("auprc", 0.0)),
    }


def _chain_rank(chain: dict[str, Any]) -> float:
    return float(chain.get("chain_quality", chain.get("chain_score", chain.get("confidence", 0.0))))


def _sufficiency(current_auprc: float, original_auprc: float) -> float:
    if not math.isfinite(original_auprc) or original_auprc <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, current_auprc / original_auprc)))


def _resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "auto" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif value is pd.NA:
            out[key] = None
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
