# Risk Card Construction Case Study

This module builds paper-ready case-study tables showing how HERO constructs risk cards on five datasets:

- `yelp_academic`
- `amazon_video`
- `fraud_yelp`
- `fraud_amazon`
- `elliptic`

It is designed for the main paper or supplementary material. The paper-ready table is the concise AAAI-friendly version, the compact table keeps a fuller one-row-per-dataset summary, and the field trace table records where each risk-card field came from, how it was computed, and what risk meaning it supports.

## Inputs

The builder searches real project artifacts in this order:

- raw review or transaction data when available
- processed graph data under `data/processed`
- Qwen/local LLM annotation caches such as `qwen_annotations.jsonl`, `cached_llm_annotations.jsonl`, and `llm_labels.jsonl`
- HERO risk-card or risk-weight caches
- evidence chains such as `evidence_chains.jsonl` or `evidence_chains.csv`

No case is manually invented. If a field or dataset is unavailable, the output records `N/A`, `unavailable`, or `not available in this dataset`.

## Outputs

The standalone output directory is usually `outputs/risk_card_cases`:

- `raw/selected_cases.jsonl`
- `raw/risk_card_field_traces.jsonl`
- `tables_csv/table_risk_card_cases_compact.csv`
- `tables_csv/table_risk_card_cases_paper_ready.csv`
- `tables_csv/table_risk_card_field_trace.csv`
- `tables_latex/table_risk_card_cases_compact.tex`
- `tables_latex/table_risk_card_cases_paper_ready.tex`
- `tables_latex/table_risk_card_field_trace.tex`
- `tables_markdown/table_risk_card_cases_compact.md`
- `tables_markdown/table_risk_card_cases_paper_ready.md`
- `tables_markdown/table_risk_card_field_trace.md`
- `reports/RISK_CARD_CASE_REPORT.md`

When copied into final artifacts, the supplement files are:

- `outputs/final_artifacts/tables_csv/supp_table_risk_card_cases_compact.csv`
- `outputs/final_artifacts/tables_csv/supp_table_risk_card_cases_paper_ready.csv`
- `outputs/final_artifacts/tables_csv/supp_table_risk_card_field_trace.csv`
- `outputs/final_artifacts/tables_latex/supp_table_risk_card_cases_compact.tex`
- `outputs/final_artifacts/tables_latex/supp_table_risk_card_cases_paper_ready.tex`
- `outputs/final_artifacts/tables_latex/supp_table_risk_card_field_trace.tex`
- `outputs/final_artifacts/reports/RISK_CARD_CASE_REPORT.md`

## Run Standalone

```bash
python scripts/build_risk_card_case_table.py \
  --data_root data \
  --output_dir outputs/risk_card_cases \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic \
  --source_outputs outputs/review_rerun_20260713_210101 \
  --top_cases_per_dataset 1 \
  --include_field_trace \
  --write_latex \
  --write_markdown \
  --copy_to_final_artifacts
```

## Run By Stage

```bash
bash scripts/run_stage.sh --stage risk_card_cases --device cuda
```

This writes logs to:

```text
<output_root>/logs/risk_card_cases.log
```

and records failures in:

```text
<output_root>/failed_runs.csv
```

## Final Artifacts

```bash
bash scripts/run_stage.sh --stage final_artifacts --device cuda
```

The final-artifacts builder generates the risk-card case tables and copies the supplement files into `tables_csv`, `tables_latex`, and `reports`.

## Validate

```bash
python scripts/validate_risk_card_cases.py \
  --case_dir outputs/risk_card_cases \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic
```

For final artifacts:

```bash
python scripts/check_final_artifacts.py --output_dir outputs/final_artifacts
```

## How To Read The Tables

The paper-ready table has one row per dataset and keeps only `dataset`, a truncated `target_neighbor_pair`, relation/path, short evidence, 2-3 derived cues, mechanism, score/confidence, and decision. It avoids long provenance fields and is intended for direct paper placement.

The compact table has one row per dataset. It shows the selected target, neighbor, relation or path, key raw fields, derived cues, mechanism candidate, risk relevance, confidence, and provenance summary.

The field trace table has one row per field. It records `source_column_or_file`, `raw_value_target`, `raw_value_neighbor`, `computation_rule`, `computed_value`, `threshold_or_normalization`, `risk_card_slot`, and `risk_interpretation`.

## Missing Fields

Missing fields are never imputed. They are written as:

- `N/A`
- `unavailable`
- `not available in this dataset`

If an entire dataset lacks a usable processed graph or cached model output, the compact table still contains a row for that dataset and the report marks it as unavailable.

All risk-card cases are extracted from real data or cached model outputs. Missing fields are marked as N/A rather than imputed.
