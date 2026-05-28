import pandas as pd

from scripts.summarize_results import _diagnostic_table


def test_diagnostic_table_includes_lite_baseline_fields():
    table = _diagnostic_table(
        pd.DataFrame(
            [
                {
                    "dataset": "synthetic",
                    "method": "flag_lite",
                    "seed": 0,
                    "model_family": "flag_lite",
                    "uses_semantic_enhancement": True,
                    "macro_f1": 0.5,
                    "auroc": 0.6,
                    "auprc": 0.7,
                }
            ]
        )
    )

    for column in [
        "model_family",
        "uses_heterophily_filter",
        "uses_semantic_enhancement",
        "num_attribute_groups",
        "num_neighbor_groups",
    ]:
        assert column in table.columns
    assert table.loc[0, "model_family"] == "flag_lite"
    assert bool(table.loc[0, "uses_semantic_enhancement"]) is True
    assert table.loc[0, "num_attribute_groups"] == 0.0
