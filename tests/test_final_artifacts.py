import argparse
import json

import pandas as pd

from scripts.build_final_tables_and_figures import build_final_artifacts


def test_final_artifacts_build_from_fake_main_outputs(tmp_path):
    main_dir = tmp_path / "main"
    summary = main_dir / "summary"
    summary.mkdir(parents=True)
    raw = pd.DataFrame(
        [
            {"suite": "main", "dataset": "yelp_academic", "model": "hero_gnn", "seed": 0, "status": "ok", "Macro-F1": 0.8, "AUROC": 0.9, "AUPRC": 0.7},
            {"suite": "main", "dataset": "yelp_academic", "model": "mlp", "seed": 0, "status": "ok", "Macro-F1": 0.7, "AUROC": 0.8, "AUPRC": 0.6},
            {"suite": "main", "dataset": "fraud_yelp", "model": "hero_official", "seed": 0, "status": "ok", "Macro-F1": 0.75, "AUROC": 0.85, "AUPRC": 0.65},
        ]
    )
    raw.to_csv(summary / "all_raw_runs.csv", index=False)
    pd.DataFrame([{"suite": "main", "dataset": "elliptic", "model": "mled", "seed": 0, "status": "missing", "reason": "model_not_applicable_to_dataset"}]).to_csv(summary / "table_missing_runs.csv", index=False)
    sensitivity_dir = tmp_path / "sensitivity"
    (sensitivity_dir / "summary").mkdir(parents=True)
    pd.DataFrame([{"dataset": "yelp_academic", "parameter": "candidate_neighbor_k", "param_value": 5, "AUPRC_mean_std": "70.00 ± NA", "status": "ok"}]).to_csv(
        sensitivity_dir / "summary" / "table_sensitivity.csv",
        index=False,
    )

    out = tmp_path / "final"
    build_final_artifacts(
        argparse.Namespace(
            main_dir=str(main_dir),
            ablation_dir=None,
            robustness_dir=None,
            labeler_dir=None,
            faithfulness_dir=None,
            sensitivity_dir=str(sensitivity_dir),
            cost_dir=None,
            output_dir=str(out),
            data_root=str(tmp_path / "data"),
            quick_test=True,
            allow_missing=True,
        )
    )

    table2 = pd.read_csv(out / "tables_csv" / "table2_main_text_rich.csv")
    table3 = pd.read_csv(out / "tables_csv" / "table3_transfer.csv")
    assert set(table2["dataset"]) == {"yelp_academic"}
    assert set(table3["dataset"]) == {"fraud_yelp"}
    assert (out / "tables_latex" / "table2_main_text_rich.tex").exists()
    assert (out / "tables_latex" / "table1_dataset_statistics.tex").exists()
    assert (out / "tables_csv" / "supp_table_sensitivity.csv").exists()
    assert (out / "tables_latex" / "supp_table_sensitivity.tex").exists()
    assert (out / "reports" / "final_artifacts_report.md").exists()
