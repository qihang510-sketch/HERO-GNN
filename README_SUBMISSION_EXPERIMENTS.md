# HERO-GNN Submission Experiments

This document describes the submission-grade experiment workflow. Local VSCode/Codex is for code, configs, smoke tests, and readiness checks. Large real datasets and 5-seed experiments should run on AutoDL.

## Goals

The submission workflow evaluates HERO-GNN on text-rich proxy review graphs, official fraud benchmarks, and a transaction graph, with reproduced strong baselines, multi-seed summaries, ablations, LLM annotation studies, significance tests, figures, and readiness checks.

No script may fabricate results. Missing experiments must produce `skip_reason.json` or `NA` table cells.

## Data Paths

Expected processed data:

- `data/processed/yelp_academic/`
- `data/processed/amazon_video/`
- `data/processed/fraud_yelp_official/` or `data/processed/fraud_yelp/`
- `data/processed/fraud_amazon_official/` or `data/processed/fraud_amazon/`
- `data/processed/elliptic/`

Expected raw data for preparation:

- Yelp Academic: `data/raw/yelp_academic/yelp_academic_dataset_review.json`, `yelp_academic_dataset_business.json`, `yelp_academic_dataset_user.json`
- Amazon Video: `data/raw/amazon_video/reviews.json.gz`, `meta.json.gz`, or compatible `Video_Games.json.gz`, `meta_Video_Games.json.gz`
- FraudYelp/FraudAmazon: prefer processed; otherwise use DGL dataset preparation on AutoDL
- Elliptic: `data/raw/elliptic/elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`

Elliptic maps `illicit`/`1` to `1`, `licit`/`2` to `0`, and `unknown` to `-1`. Unknown nodes remain in the graph as context but are excluded from train/val/test loss and metrics.

## Baselines

Formal submission names only:

- `mlp`, `gcn`, `gat`, `graphsage`
- `care_gnn`, `graphconsis`, `pc_gnn`
- `bwgnn`, `linkx`
- `dgp`, `mled`
- `hogrl`, `rgtan`
- `hero_gnn`, `hero_official`

Implementation sources:

- `mlp`, `gcn`, `gat`, `graphsage`: `benchmark`
- `care_gnn`, `graphconsis`, `pc_gnn`, `bwgnn`, `linkx`, `dgp`, `mled`, `hogrl`, `rgtan`: `reproduced`
- `hero_gnn`, `hero_official`: `project`

Lite names are forbidden in submission outputs. Existing historical `*_lite` code is not deleted, but submission runners, tables, and readiness checks exclude `CARE-GNN-lite`, `GraphConsis-lite`, `PC-GNN-lite`, `BWGNN-lite`, `DGP-lite`, `MLED-lite`, `HOGRL-lite`, `RGTAN-lite`, `FLAG-lite`, and any lowercase/underscore equivalent.

## Dataset-Model Matrix

Text-rich datasets `yelp_academic` and `amazon_video`:

`mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx dgp mled hero_gnn`

Official fraud datasets `fraud_yelp` and `fraud_amazon`:

`mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx hero_official`

Transaction dataset `elliptic`:

`mlp gcn gat graphsage bwgnn linkx hogrl rgtan hero_official`

## Main Experiments

```bash
python scripts/run_submission_experiments.py \
  --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_experiments
```

Grouped runs:

```bash
python scripts/run_submission_experiments.py \
  --datasets yelp_academic amazon_video \
  --models mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx dgp mled hero_gnn \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_experiments
```

```bash
python scripts/run_submission_experiments.py \
  --datasets fraud_yelp fraud_amazon \
  --models mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx hero_official \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_experiments
```

```bash
python scripts/run_submission_experiments.py \
  --datasets elliptic \
  --models mlp gcn gat graphsage bwgnn linkx hogrl rgtan hero_official \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_experiments
```

Each seed writes:

- `metrics.json`
- `predictions.npy`
- `config_resolved.yaml`
- `run.log`

Skipped runs write `skip_reason.json`.

## Ablations

```bash
python scripts/run_ablation_experiments.py \
  --datasets yelp_academic amazon_video \
  --seeds 0 1 2 3 4 \
  --output_dir outputs/submission_experiments_ablation
```

Currently implemented real switches reuse existing HERO variants: full HERO-GNN, w/o risk-relevant heterophily, w/o mechanism annotation, and w/o evidence chain. Other requested switches are marked as skipped until separate trainer switches exist.

## Sensitivity

```bash
python scripts/run_sensitivity_experiments.py \
  --dataset yelp_academic \
  --seeds 0 1 2 \
  --output_dir outputs/submission_sensitivity
```

Neighbor-budget sensitivity is runnable. Other grids are written as skipped unless the trainer exposes those knobs.

## LLM Labeler And Coverage

Default labeler comparison uses existing annotations only:

```bash
python scripts/run_labeler_comparison.py \
  --dataset yelp_academic \
  --labelers rule qwen \
  --seeds 0 1 2 3 4 \
  --use_existing_annotations true \
  --output_dir outputs/submission_llm_labeler
```

Coverage sensitivity:

```bash
python scripts/run_llm_coverage_sensitivity.py \
  --dataset yelp_academic \
  --coverages 0 0.25 0.5 0.75 1.0 \
  --seeds 0 1 2 \
  --use_existing_qwen_annotations true \
  --output_dir outputs/submission_llm_coverage
```

The scripts do not call Qwen unless a future explicit `--call_llm true` path is implemented. Missing Qwen annotations are skipped, not replaced with mock labels.

## Summary, Significance, Figures

```bash
python scripts/summarize_submission_experiments.py \
  --input_dir outputs/submission_experiments \
  --output_dir outputs/paper_tables_submission
```

```bash
python scripts/run_significance_tests.py \
  --input_dir outputs/submission_experiments \
  --output_dir outputs/paper_tables_submission
```

```bash
python scripts/generate_paper_figures.py \
  --tables_dir outputs/paper_tables_submission \
  --output_dir outputs/paper_figures_submission
```

Figures are generated from real tables. If data are missing, the script writes `skipped_figures_report.md`.

## Readiness

```bash
python scripts/check_submission_readiness.py \
  --results_dir outputs/submission_experiments \
  --tables_dir outputs/paper_tables_submission \
  --figures_dir outputs/paper_figures_submission \
  --output outputs/submission_readiness_report.md
```

PASS means all core checks pass. WARNING means optional or competitive-but-not-SOTA issues remain. FAIL means the project is not submission-ready.

Core failure examples:

- HERO-GNN does not beat DGP/MLED.
- Qwen high-coverage does not beat rule/mock.
- Missing main 5-seed results.
- Lite baseline names appear in submission outputs.
- Std values are not from real multi-seed runs.

## AutoDL Stages

Run stages in order:

```bash
bash scripts/autodl_stage0_check_env.sh
bash scripts/autodl_stage1_prepare_data.sh
bash scripts/autodl_stage2_smoke_test.sh
bash scripts/autodl_stage3_run_main_5seeds.sh
bash scripts/autodl_stage4_run_ablation.sh
bash scripts/autodl_stage5_run_llm_experiments.sh
bash scripts/autodl_stage6_summarize_and_figures.sh
bash scripts/autodl_stage7_readiness_check.sh
```

Large runs must be done on AutoDL. Local runs are only for smoke tests, static checks, and toy/small processed data.

## No Fake Results

Do not edit metrics to improve significance. Do not fill missing std by hand. Do not hide failed baselines. If HERO-GNN is not first, the readiness report must say so and the model should be tuned or re-run.
