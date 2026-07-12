import pandas as pd

from scripts.plot_cost_scalability import plot_cost_scalability
from scripts.plot_seed_stability import plot_seed_stability


def test_cost_plot_generation_from_fake_csv(tmp_path):
    output_dir = tmp_path / "cost"
    summary = output_dir / "summary"
    summary.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "yelp_academic",
                "model": "hero_gnn",
                "annotation_time_seconds": 1.0,
                "train_time_seconds_per_seed": 2.0,
                "cache_size_mb": 3.0,
                "status": "ok",
            }
        ]
    ).to_csv(summary / "table_cost_scalability.csv", index=False)

    plot_cost_scalability(output_dir)

    assert (output_dir / "figures" / "fig_cost_annotation_time.pdf").exists()
    assert (output_dir / "figures" / "fig_cost_annotation_time.png").exists()


def test_seed_stability_plot_from_fake_all_raw(tmp_path):
    raw_csv = tmp_path / "all_raw_runs.csv"
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "model": "hero_gnn", "seed": 0, "status": "ok", "AUPRC": 0.8},
            {"dataset": "yelp_academic", "model": "gcn", "seed": 0, "status": "ok", "AUPRC": 0.6},
            {"dataset": "yelp_academic", "model": "mlp", "seed": 0, "status": "ok", "AUPRC": 0.5},
        ]
    ).to_csv(raw_csv, index=False)

    plot_seed_stability(raw_csv, tmp_path / "final", datasets=["yelp_academic"])

    assert (tmp_path / "final" / "figures_pdf" / "fig_seed_stability_auprc.pdf").exists()
    assert (tmp_path / "final" / "figures_png" / "fig_seed_stability_auprc.png").exists()
