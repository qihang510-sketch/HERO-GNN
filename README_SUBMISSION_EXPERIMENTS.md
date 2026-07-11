# HERO Submission Experiments

This document describes the unified experiment pipeline for submission runs. All tables must be generated from real per-seed `metrics.json` files or explicit missing/skipped records. Do not hand-edit metrics, fill missing seeds by hand, or copy values into CSVs.

## Output Layout

Use one output root per submission sweep:

```text
outputs/submission_unified/
  config.json
  raw/
  summary/
  tables/
  figures/
  logs/
```

Each run directory under `raw/` is expected to contain:

- `config.json`
- `metrics.json`
- `runtime.json`
- `log.txt`
- `predictions.npy` when the underlying trainer produced predictions

Missing or unsupported runs are written as `status=missing` with a reason. They are not used as real results.

## Main Experiments

Run the full 5-seed main benchmark:

```bash
python scripts/run_experiment_suite.py \
  --suite main \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --device cuda \
  --skip_existing
```

If `--models` is omitted, the runner uses the dataset-specific matrix in `src.training.submission`:

- `yelp_academic`, `amazon_video`: `mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx dgp mled hero_gnn`
- `fraud_yelp`, `fraud_amazon`: `mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx hero_official`
- `elliptic`: `mlp gcn gat graphsage bwgnn linkx hogrl rgtan hero_official`

Dry-run the plan without training:

```bash
python scripts/run_experiment_suite.py \
  --suite main \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --dry_run
```

## Ablation

Run the 5-seed HERO ablation suite:

```bash
python scripts/run_experiment_suite.py \
  --suite ablation \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --device cuda \
  --skip_existing
```

Supported variants are read from `scripts.run_ablation_experiments.ABLATION_VARIANTS`: `hero_gnn`, `wo_risk_relevant_heterophily`, `wo_mechanism_annotation`, `wo_evidence_chain`, `wo_llm_annotation`, `wo_heterophily_filter`, `wo_dual_branch_encoder`, and `wo_gated_fusion`.

## Robustness

The robustness suite perturbs an existing annotation cache or relation annotation file and retrains HERO for each dataset, seed, noise type, and noise ratio. If no real LLM cache exists, it builds rule-based fallback annotations from risk cards and records `annotation_source` in every row:

```bash
python scripts/run_experiment_suite.py \
  --suite robustness \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --skip_existing
```

Standalone entry:

```bash
python scripts/run_llm_annotation_robustness.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --noise_types relevance_flip mechanism_shuffle confidence_gaussian \
  --noise_ratios 0.0 0.1 0.2 0.3 0.4 \
  --output_dir outputs/submission_unified \
  --device cuda \
  --skip_existing
```

Outputs include `summary/robustness_raw.csv`, `summary/robustness_summary.csv`, `summary/robustness_plot_data.csv`, `summary/table_llm_robustness.csv`, and `figures/robustness_curve_data.csv`.

## Labeler Comparison

Compare random, rule-based, proxy, cached LLM, and full LLM labelers:

```bash
python scripts/run_experiment_suite.py \
  --suite labeler_comparison \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --device cuda \
  --skip_existing
```

Standalone entry:

```bash
python scripts/run_llm_labeler_comparison.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --device cuda \
  --skip_existing
```

`cached_llm_labeler` runs only when a real cache such as Qwen/OpenAI labels is found. `full_llm_labeler` requires `--full_llm_label_file`, or explicit `--call_full_llm` with a configured backend. Outputs include `summary/table_llm_labeler_comparison.csv` and `figures/labeler_comparison_data.csv`.

## Faithfulness

Faithfulness evaluates saved HERO checkpoints by replaying the checkpoint with evidence-chain input perturbations. If a checkpoint or `hero_artifacts` are missing, rows are marked `unavailable`:

```bash
python scripts/run_experiment_suite.py \
  --suite faithfulness \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --skip_existing
```

Standalone entry:

```bash
python scripts/run_evidence_faithfulness.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --topks 1 3 5 \
  --input_dir outputs/submission_unified \
  --output_dir outputs/submission_unified \
  --device cuda
```

Outputs include `summary/faithfulness_raw.csv`, `summary/faithfulness_summary.csv`, `summary/table_faithfulness.csv`, and `figures/faithfulness_bar_data.csv`.

## Cost And Scalability

Cost/scalability metadata should be collected from real logs, annotation cache metadata, and runtime files:

```bash
python scripts/run_experiment_suite.py \
  --suite cost \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_unified \
  --skip_existing
```

Required fields include candidate cards, annotation coverage, total tokens when available, annotation time, cache size, training time, and GPU/CPU information.

Standalone entry:

```bash
python scripts/collect_cost_scalability.py \
  --input_dir outputs/submission_unified \
  --output_dir outputs/submission_unified \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic
```

Outputs include `summary/table_cost_scalability.csv`, `summary/table_cost.csv`, and mirrored copies under `tables/`.

## Summarize Results

Generate raw run tables, mean ± std tables, missing-run tables, and HERO-vs-baseline significance tests:

```bash
python scripts/summarize_experiment_suite.py \
  --output_dir outputs/submission_unified
```

Main outputs:

- `summary/all_raw_runs.csv`
- `summary/table_main_mean_std.csv`
- `summary/table_ablation_mean_std.csv`
- `summary/table_transfer_mean_std.csv`
- `summary/table_significance.csv`
- `summary/table_missing_runs.csv`

The same CSVs are mirrored under `tables/` for paper use.

## Significance Tests

Run significance tests directly when needed:

```bash
python scripts/run_significance_tests.py \
  --input_dir outputs/submission_unified \
  --output_dir outputs/submission_unified/tables
```

For every dataset and metric, the script compares HERO with each baseline using matched seeds only. If fewer than three paired seeds exist, the row is marked `insufficient_seeds`.

## Missing Checks

Check completeness and generate rerun commands:

```bash
python scripts/check_experiment_completeness.py \
  --output_dir outputs/submission_unified \
  --generate_commands
```

The checker writes:

- `summary/missing_runs.csv`
- `summary/missing_commands.txt` when `--generate_commands` is used

## Reproduce Paper Tables

Recommended sequence:

```bash
python scripts/run_experiment_suite.py --suite main --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic --seeds 0 1 2 3 4 --output_dir outputs/submission_unified --device cuda --skip_existing
python scripts/run_experiment_suite.py --suite ablation --datasets yelp_academic amazon_video --seeds 0 1 2 3 4 --output_dir outputs/submission_unified --device cuda --skip_existing
python scripts/summarize_experiment_suite.py --output_dir outputs/submission_unified
python scripts/check_experiment_completeness.py --output_dir outputs/submission_unified --generate_commands
```

The paper tables are valid only where the underlying per-seed raw results exist. Missing runs must remain missing until they are rerun.
