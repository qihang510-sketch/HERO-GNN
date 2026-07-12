import pandas as pd

from scripts.run_sensitivity_analysis import plot_sensitivity, summarize_sensitivity


def test_sensitivity_missing_data_summarizes_unavailable(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "dataset": "yelp_academic",
                "seed": 0,
                "parameter": "candidate_neighbor_k",
                "param_value": 3,
                "lambda_rel": pd.NA,
                "lambda_chain": pd.NA,
                "confidence_threshold": pd.NA,
                "candidate_neighbor_k": 3,
                "status": "unavailable",
                "skip_reason": "processed_data_missing",
                "supported_config_key": "neighbor_budget",
            }
        ]
    )

    summary = summarize_sensitivity(raw)

    assert summary.iloc[0]["status"] == "unavailable"
    assert summary.iloc[0]["AUPRC_mean_std"] == "--"


def test_sensitivity_plots_from_fake_csv(tmp_path):
    output_dir = tmp_path / "sensitivity"
    figure_data = output_dir / "figure_data"
    figure_data.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "parameter": "candidate_neighbor_k", "param_value": 3, "AUPRC_mean": 0.6, "status": "ok"},
            {"dataset": "yelp_academic", "parameter": "candidate_neighbor_k", "param_value": 5, "AUPRC_mean": 0.7, "status": "ok"},
            {"dataset": "yelp_academic", "parameter": "lambda_rel", "param_value": 0.1, "AUPRC_mean": 0.65, "status": "ok"},
        ]
    ).to_csv(figure_data / "sensitivity_curve_data.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "lambda_rel": 0.1, "lambda_chain": 0.1, "AUPRC_mean": 0.62, "status": "ok"},
            {"dataset": "yelp_academic", "lambda_rel": 0.3, "lambda_chain": 0.1, "AUPRC_mean": 0.66, "status": "ok"},
        ]
    ).to_csv(figure_data / "sensitivity_heatmap_data.csv", index=False)

    plot_sensitivity(output_dir)

    assert (output_dir / "figures" / "fig_sensitivity_k.pdf").exists()
    assert (output_dir / "figures" / "fig_sensitivity_lambda_rel.pdf").exists()
    assert (output_dir / "figures" / "fig_sensitivity_lambda_heatmap.pdf").exists()
