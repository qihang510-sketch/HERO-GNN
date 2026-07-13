from __future__ import annotations

import pandas as pd

from scripts.build_evidence_case_table import build_evidence_case_table
from scripts.plot_llm_robustness import plot_llm_robustness


def test_evidence_case_table_unavailable_when_no_real_files(tmp_path):
    table = build_evidence_case_table(tmp_path / "missing", output_dir=tmp_path / "out")
    assert table["status"].iloc[0] == "unavailable"
    assert (tmp_path / "out" / "summary" / "table_evidence_cases.csv").exists()
    assert (tmp_path / "out" / "summary" / "table_evidence_cases.tex").exists()


def test_evidence_case_table_from_real_chain_csv(tmp_path):
    chain_dir = tmp_path / "run" / "raw" / "yelp_academic" / "hero_full" / "seed_0"
    chain_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "target_id": 1,
                "neighbor_id": 2,
                "mechanism_label": "burst_review",
                "risk_relevance_score": 0.9,
                "confidence": 0.8,
                "evidence_text": "cached evidence text",
            }
        ]
    ).to_csv(chain_dir / "evidence_chains.csv", index=False)

    table = build_evidence_case_table(tmp_path / "run", output_dir=tmp_path / "out")

    assert table["dataset"].iloc[0] == "yelp_academic"
    assert table["target"].iloc[0] == 1
    assert table["neighbor"].iloc[0] == 2
    assert table["status"].iloc[0] == "ok"


def test_missing_plot_data_writes_unavailable_summary(tmp_path):
    frame = plot_llm_robustness(tmp_path / "robustness_missing")
    assert frame["status"].iloc[0] == "unavailable"
    assert (tmp_path / "robustness_missing" / "figure_data" / "robustness_curve_data.csv").exists()
    assert (tmp_path / "robustness_missing" / "reports" / "plot_llm_robustness_report.md").exists()
