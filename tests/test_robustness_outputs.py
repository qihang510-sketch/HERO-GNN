import pandas as pd

from scripts.run_llm_annotation_robustness import plot_robustness_figures


def test_robustness_plots_from_fake_curve_csv(tmp_path):
    output_dir = tmp_path / "robustness"
    figure_data = output_dir / "figure_data"
    figure_data.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "seed": 0, "noise_type": "relevance_flip", "noise_ratio": 0.0, "annotation_source": "proxy", "status": "ok", "AUROC": 0.8, "AUPRC": 0.7},
            {"dataset": "yelp_academic", "seed": 0, "noise_type": "relevance_flip", "noise_ratio": 0.1, "annotation_source": "proxy", "status": "ok", "AUROC": 0.78, "AUPRC": 0.68},
            {"dataset": "yelp_academic", "seed": 0, "noise_type": "confidence_gaussian", "noise_ratio": 0.1, "annotation_source": "proxy", "status": "ok", "AUROC": 0.79, "AUPRC": 0.69},
        ]
    ).to_csv(figure_data / "robustness_curve_data.csv", index=False)

    plot_robustness_figures(output_dir)

    assert (output_dir / "figures" / "fig_llm_robustness_auprc.pdf").exists()
    assert (output_dir / "figures" / "fig_llm_robustness_auprc.png").exists()
    assert (output_dir / "figures" / "fig_llm_robustness_auroc.pdf").exists()
    assert (output_dir / "figures" / "fig_llm_robustness_auroc.png").exists()
