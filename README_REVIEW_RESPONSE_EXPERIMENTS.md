# Review-Response Experiment Pipeline

This pipeline is for reproducible reviewer-response experiments. It never fabricates metrics: every CSV/figure is built from raw runs, existing summaries, checkpoints, annotation caches, or real script output. Missing data is reported as `missing`, `unavailable`, or `insufficient_seeds`.

## Reuse Existing Main Results

The completed AutoDL main run can be reused directly:

```bash
python scripts/summarize_experiment_suite.py --output_dir outputs/submission_main_20260711_225137
python scripts/run_significance_tests.py --main_dir outputs/submission_main_20260711_225137
python scripts/build_final_tables_and_figures.py \
  --main_dir outputs/submission_main_20260711_225137 \
  --output_dir outputs/final_artifacts \
  --allow_missing
python scripts/plot_seed_stability.py \
  --input_csv outputs/submission_main_20260711_225137/summary/all_raw_runs.csv \
  --output_dir outputs/final_artifacts
```

This regenerates the main table, transfer table, significance table, final CSV/LaTeX tables, and seed-stability figure from existing outputs.

## Robustness

```bash
python scripts/run_llm_annotation_robustness.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --noise_ratios 0.0 0.1 0.2 0.3 0.4 \
  --output_dir outputs/review_robustness \
  --device cuda \
  --skip_existing
```

Outputs:

- `summary/table_llm_robustness.csv`
- `summary/table_llm_robustness.tex`
- `figure_data/robustness_curve_data.csv`
- `figures/fig_llm_robustness_auprc.pdf/png`
- `figures/fig_llm_robustness_auroc.pdf/png`

If no real LLM cache is found, the script uses existing `llm_labels.jsonl` as a proxy/rule cache or generates rule-based fallback labels where processed text-rich data exists. The `annotation_source` column records the source.

## Labeler Comparison

```bash
python scripts/run_llm_labeler_comparison.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/review_labeler \
  --device cuda \
  --skip_existing
```

Compares `random_labeler`, `rule_based_labeler`, `proxy_labeler`, `cached_llm_labeler`, and `full_llm_labeler`. Full LLM labeling is `unavailable` unless an existing label file is passed or `--call_full_llm` is used with a configured API/local model.

## Faithfulness

```bash
python scripts/run_evidence_faithfulness.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --topks 1 3 5 \
  --input_dir outputs/submission_main_20260711_225137 \
  --output_dir outputs/review_faithfulness \
  --device cuda
```

Requires HERO checkpoints with `hero_artifacts`. If checkpoints or evidence-chain artifacts are missing, rows are marked `unavailable`.

## Sensitivity

```bash
python scripts/run_sensitivity_analysis.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 \
  --output_dir outputs/review_sensitivity \
  --device cuda \
  --skip_existing
```

Supported mappings:

- candidate neighbor K -> `neighbor_budget`
- `lambda_rel` -> `mechanism_loss_weight`
- `lambda_chain` -> `chain_loss_weight`
- confidence threshold -> `risk_relevance_threshold`

Unsupported or missing processed-data cases are reported in the CSV instead of being imputed.

## Cost And Scalability

```bash
python scripts/collect_cost_scalability.py \
  --input_dir outputs/submission_main_20260711_225137 \
  --output_dir outputs/review_cost \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic
python scripts/plot_cost_scalability.py --output_dir outputs/review_cost
```

LLM annotation is treated as offline cost; training only reads cached annotations. Token counts are `NA` when unavailable.

## One-Command Quick Test

```bash
bash scripts/run_review_response_experiments.sh \
  --quick_test \
  --device cpu \
  --main_dir outputs/submission_main_20260711_225137 \
  --run_all
```

PowerShell:

```powershell
.\scripts\run_review_response_experiments.ps1 --quick_test --device cpu --main_dir outputs/submission_main_20260711_225137 --run_all
```

Quick test uses `yelp_academic`, seed `0`, a small robustness ratio, and top-k `1` for faithfulness.

## Full Run

```bash
bash scripts/run_review_response_experiments.sh \
  --device cuda \
  --skip_existing \
  --main_dir outputs/submission_main_20260711_225137 \
  --run_all
```

The final consolidated artifacts are written to:

```text
outputs/final_artifacts/
  tables_csv/
  tables_latex/
  figures_pdf/
  figures_png/
  figure_data/
  reports/
```

## Missing Or Unavailable

Check:

- `outputs/final_artifacts/reports/final_artifacts_report.md`
- `outputs/final_artifacts/tables_csv/supp_table_missing_or_skipped_runs.csv`
- each experiment directory under `summary/`

`missing` means expected raw results were not found. `unavailable` means the experiment needs files not present in the current workspace, such as checkpoints, evidence artifacts, embeddings, or real LLM caches.

## Paper Placement

Suggested main text:

- table2 main text-rich performance
- table3 transfer/generalization
- table4 ablation
- seed stability figure if space allows

Suggested supplement:

- significance tests
- LLM robustness
- labeler comparison
- faithfulness
- sensitivity
- cost/scalability
- missing/skipped run audit
