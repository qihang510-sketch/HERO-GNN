import pandas as pd

from scripts.run_evidence_faithfulness import plot_faithfulness, summarize_faithfulness


def test_faithfulness_unavailable_rows_are_summarized():
    raw = pd.DataFrame(
        [
            {
                "dataset": "yelp_academic",
                "seed": 0,
                "setting": "remove_topk_evidence",
                "top_k": 1,
                "status": "unavailable",
                "skip_reason": "checkpoint_missing",
            }
        ]
    )

    summary = summarize_faithfulness(raw)

    assert summary.iloc[0]["status"] == "unavailable"
    assert summary.iloc[0]["skip_reason"] == "checkpoint_missing"
    assert summary.iloc[0]["AUPRC_mean_std"] == "--"


def test_faithfulness_plot_from_fake_csv(tmp_path):
    output_dir = tmp_path / "faithfulness"
    figure_data = output_dir / "figure_data"
    figure_data.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "yelp_academic", "setting": "remove_topk_evidence", "top_k": 1, "AUPRC_drop_mean": 0.05, "status": "ok"},
            {"dataset": "yelp_academic", "setting": "remove_random_edges", "top_k": 1, "AUPRC_drop_mean": 0.01, "status": "ok"},
        ]
    ).to_csv(figure_data / "faithfulness_bar_data.csv", index=False)

    plot_faithfulness(output_dir)

    assert (output_dir / "figures" / "fig_evidence_faithfulness.pdf").exists()
    assert (output_dir / "figures" / "fig_evidence_faithfulness.png").exists()
