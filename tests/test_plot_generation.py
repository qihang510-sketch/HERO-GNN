import pandas as pd

from scripts.plot_cost_scalability import plot_cost_scalability
from scripts.plot_evidence_faithfulness import plot_evidence_faithfulness
from scripts.plot_labeler_comparison import plot_labeler_comparison
from scripts.plot_llm_robustness import plot_llm_robustness
from scripts.plot_seed_stability import plot_seed_stability
from scripts.plot_sensitivity import plot_sensitivity


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
    assert (tmp_path / "final" / "figures" / "fig_seed_stability_auprc.pdf").exists()


def test_review_plot_wrappers_from_fake_csv(tmp_path):
    robustness = tmp_path / "robustness"
    (robustness / "figure_data").mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "noise_type": "relevance_flip", "noise_ratio": 0.0, "AUPRC": 0.7, "AUROC": 0.8, "status": "ok"},
            {"dataset": "yelp_academic", "noise_type": "relevance_flip", "noise_ratio": 0.1, "AUPRC": 0.68, "AUROC": 0.79, "status": "ok"},
        ]
    ).to_csv(robustness / "figure_data" / "robustness_curve_data.csv", index=False)
    plot_llm_robustness(robustness)
    assert (robustness / "figures" / "fig_llm_robustness_auprc.pdf").exists()
    assert (robustness / "figures" / "fig_llm_robustness_auprc.png").exists()

    labeler = tmp_path / "labeler"
    (labeler / "figure_data").mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "labeler": "random_labeler", "AUPRC": 0.4, "status": "ok"},
            {"dataset": "yelp_academic", "labeler": "proxy_labeler", "AUPRC": 0.6, "status": "ok"},
        ]
    ).to_csv(labeler / "figure_data" / "labeler_comparison_data.csv", index=False)
    plot_labeler_comparison(labeler)
    assert (labeler / "figures" / "fig_labeler_comparison_auprc.pdf").exists()
    assert (labeler / "figures" / "fig_labeler_comparison_auprc.png").exists()

    faithfulness = tmp_path / "faithfulness"
    (faithfulness / "figure_data").mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "setting": "remove_topk_evidence", "top_k": 1, "AUPRC_drop_mean": 0.05, "status": "ok"},
            {"dataset": "yelp_academic", "setting": "remove_random_edges", "top_k": 1, "AUPRC_drop_mean": 0.01, "status": "ok"},
        ]
    ).to_csv(faithfulness / "figure_data" / "faithfulness_bar_data.csv", index=False)
    plot_evidence_faithfulness(faithfulness)
    assert (faithfulness / "figures" / "fig_evidence_faithfulness.pdf").exists()
    assert (faithfulness / "figures" / "fig_evidence_faithfulness.png").exists()


def test_sensitivity_plot_wrapper_from_fake_csv(tmp_path):
    output_dir = tmp_path / "sensitivity"
    figure_data = output_dir / "figure_data"
    figure_data.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "parameter": "candidate_neighbor_k", "param_value": 5, "AUPRC_mean": 0.6, "status": "ok"},
            {"dataset": "yelp_academic", "parameter": "candidate_neighbor_k", "param_value": 10, "AUPRC_mean": 0.65, "status": "ok"},
            {"dataset": "yelp_academic", "parameter": "lambda_rel", "param_value": 0.1, "AUPRC_mean": 0.62, "status": "ok"},
        ]
    ).to_csv(figure_data / "sensitivity_curve_data.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "lambda_rel": 0.1, "lambda_chain": 0.1, "AUPRC_mean": 0.62, "status": "ok"},
            {"dataset": "yelp_academic", "lambda_rel": 0.3, "lambda_chain": 0.1, "AUPRC_mean": 0.64, "status": "ok"},
        ]
    ).to_csv(figure_data / "sensitivity_heatmap_data.csv", index=False)

    plot_sensitivity(output_dir)

    assert (output_dir / "figures" / "fig_sensitivity_k.pdf").exists()
    assert (output_dir / "figures" / "fig_sensitivity_k.png").exists()
    assert (output_dir / "figures" / "fig_sensitivity_lambda_heatmap.pdf").exists()
    assert (output_dir / "figures" / "fig_sensitivity_lambda_heatmap.png").exists()
