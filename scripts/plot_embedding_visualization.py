from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_artifact_utils import import_matplotlib, write_csv  # noqa: E402
from src.training.submission import resolve_processed_dir  # noqa: E402


DEFAULT_DATASETS = ["yelp_academic", "amazon_video"]
DEFAULT_MODELS = ["gcn", "graphsage", "hero"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optional t-SNE/UMAP embedding visualizations if saved embeddings exist.")
    parser.add_argument("--input_dirs", nargs="+", default=["outputs"])
    parser.add_argument("--output_dir", default="outputs/final_artifacts")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--max_points", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_embedding_visualization(args)


def plot_embedding_visualization(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    (output_dir / "figure_data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_rows = []
        for model in args.models:
            embedding_path = find_embedding(args.input_dirs, dataset, model)
            if embedding_path is None:
                dataset_rows.append({"dataset": dataset, "model": model, "status": "unavailable", "reason": "embedding_file_not_found"})
                continue
            try:
                dataset_rows.extend(_coordinates(embedding_path, dataset, model, args))
            except Exception as exc:
                dataset_rows.append({"dataset": dataset, "model": model, "status": "unavailable", "reason": f"embedding_projection_failed: {type(exc).__name__}: {exc}"})
        rows.extend(dataset_rows)
        _plot_dataset(pd.DataFrame(dataset_rows), dataset, output_dir / "figures" / f"fig_tsne_{dataset}.pdf", output_dir / "figures" / f"fig_tsne_{dataset}.png")
    table = pd.DataFrame(rows)
    write_csv(output_dir / "figure_data" / "tsne_coordinates.csv", table)
    report = output_dir / "figure_data" / "tsne_status.csv"
    write_csv(report, table[["dataset", "model", "status", "reason"]].drop_duplicates() if not table.empty and {"dataset", "model", "status", "reason"}.issubset(table.columns) else table)
    return table


def find_embedding(input_dirs: list[str], dataset: str, model: str) -> Path | None:
    model_aliases = {model}
    if model == "hero":
        model_aliases.update({"hero_gnn", "hero_official"})
    patterns = ["*embedding*.npy", "*embeddings*.npy", "*node_repr*.npy", "*node_embedding*.npz"]
    for root_text in input_dirs:
        root = Path(root_text)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                text = str(path).lower()
                if dataset.lower() in text and any(alias.lower() in text for alias in model_aliases):
                    return path
    return None


def _coordinates(path: Path, dataset: str, model: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    embedding = _load_embedding(path)
    if embedding.ndim != 2:
        raise ValueError("embedding array must be 2D")
    labels = _load_labels(dataset, args.data_root, embedding.shape[0])
    if embedding.shape[0] > int(args.max_points):
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(embedding.shape[0], size=int(args.max_points), replace=False))
        embedding = embedding[idx]
        labels = labels[idx]
    coords = _project(embedding, args.method)
    rows = []
    for idx, (xy, label) in enumerate(zip(coords, labels)):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "node_index": int(idx),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "label": int(label) if np.isfinite(label) else -1,
                "embedding_file": str(path),
                "status": "ok",
                "reason": "",
            }
        )
    return rows


def _load_embedding(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        key = "embeddings" if "embeddings" in payload else payload.files[0]
        return np.asarray(payload[key], dtype=np.float32)
    return np.asarray(payload, dtype=np.float32)


def _project(embedding: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap  # type: ignore

            return umap.UMAP(n_components=2, random_state=0).fit_transform(embedding)
        except Exception:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, embedding.shape[0] // 4))
        if embedding.shape[0] <= perplexity + 1:
            method = "pca"
        else:
            return TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(embedding)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=0).fit_transform(embedding)


def _load_labels(dataset: str, data_root: str | Path, n: int) -> np.ndarray:
    try:
        from src.data.loader import load_processed_data

        labels = np.asarray(load_processed_data(resolve_processed_dir(dataset, data_root)).labels)
        if labels.size >= n:
            return labels[:n]
    except Exception:
        pass
    return np.full(n, -1)


def _plot_dataset(frame: pd.DataFrame, dataset: str, pdf_path: Path, png_path: Path) -> None:
    plt = import_matplotlib()
    ok = frame[frame["status"].astype(str) == "ok"] if not frame.empty and "status" in frame else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    if ok.empty:
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        for model, group in ok.groupby("model", dropna=False):
            ax.scatter(group["x"], group["y"], s=8, alpha=0.65, label=str(model))
        ax.set_title(dataset)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, markerscale=2, fontsize=8)
    fig.tight_layout()
    from scripts.paper_artifact_utils import save_figure

    save_figure(fig, pdf_path, png_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
