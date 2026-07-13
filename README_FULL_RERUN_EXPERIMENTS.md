# HERO Full Rerun Experiments

This guide describes the reproducible rerun pipeline for reviewer-response experiments. The pipeline never fabricates metrics: missing data, unsupported models, and unavailable checkpoints are recorded as `missing`, `skipped`, or `unavailable`.

## Quick Test

Use this before a full AutoDL run:

```bash
bash scripts/run_full_rerun_pipeline.sh --step quick_test --device cuda --quick_test
```

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_rerun_pipeline.ps1 -Step quick_test -Device cuda -QuickTest
```

Quick test uses `yelp_academic`, seed `0`, models `mlp gcn hero`, ablations `hero_full hero_no_llm`, robustness noise ratio `0.1`, faithfulness top-k `1`, and sensitivity K values `5,10`.

## Tune HERO

Tune only with validation metrics:

```bash
bash scripts/run_full_rerun_pipeline.sh --step tune --device cuda --skip_existing
```

The tuner writes `summary/tuning_results.csv`, `summary/best_hero_configs.csv`, and `best_configs/hero_full_<dataset>.yaml`. Selection uses validation AUPRC, then validation AUROC, then validation Macro-F1. Test metrics are not used for selection.

The tuned configs record `selected_by=validation_AUPRC`, `best_epoch`, validation metrics, and `config_hash`. Current HERO constructors do not expose `num_layers`; the tuner records it as unsupported in the YAML instead of pretending it changed the model.

## Tuned HERO Rerun

Rerun only HERO with validation-selected configs while reusing complete baseline outputs:

```bash
bash scripts/run_rerun_with_tuned_hero.sh --device cuda --skip_existing --reuse_baselines_dir outputs/submission_main_20260711_225137 --models hero
```

Rerun all baselines as well:

```bash
bash scripts/run_rerun_with_tuned_hero.sh --device cuda --skip_existing --rerun_baselines
```

Restrict to selected datasets or seeds:

```bash
bash scripts/run_rerun_with_tuned_hero.sh --device cuda --datasets yelp_academic amazon_video --seeds 0 1 2 --models hero --skip_existing
```

The script runs tuning, main, transfer, ablation, summarization, significance tests, performance diagnosis, and final artifacts. It exports `PYTHONUNBUFFERED=1` and limits common BLAS thread pools to reduce AutoDL stalls.

## Run Individual Suites

Main text-rich results:

```bash
python scripts/run_experiment_suite.py --suite main --device cuda --seeds 0 1 2 3 4 --skip_existing --tuned_config_dir outputs/full_rerun_YYYYMMDD_HHMMSS/tuning_hero
```

Transfer / weak-text results:

```bash
python scripts/run_experiment_suite.py --suite transfer --device cuda --seeds 0 1 2 3 4 --skip_existing --tuned_config_dir outputs/full_rerun_YYYYMMDD_HHMMSS/tuning_hero
```

Ablation:

```bash
python scripts/run_experiment_suite.py --suite ablation --device cuda --seeds 0 1 2 3 4 --skip_existing
```

LLM annotation robustness:

```bash
python scripts/run_experiment_suite.py --suite robustness --device cuda --seeds 0 1 2 3 4 --skip_existing
```

LLM labeler comparison:

```bash
python scripts/run_experiment_suite.py --suite labeler_comparison --device cuda --seeds 0 1 2 3 4 --skip_existing
```

Evidence-chain faithfulness:

```bash
python scripts/run_experiment_suite.py --suite faithfulness --device cuda --input_dir outputs/full_rerun_YYYYMMDD_HHMMSS --skip_existing
```

Sensitivity:

```bash
python scripts/run_experiment_suite.py --suite sensitivity --device cuda --seeds 0 1 2 --skip_existing
```

Final tables and figures:

```bash
python scripts/run_experiment_suite.py --suite final --main_dir outputs/full_rerun_YYYYMMDD_HHMMSS --ablation_dir outputs/full_rerun_YYYYMMDD_HHMMSS --robustness_dir outputs/full_rerun_YYYYMMDD_HHMMSS --labeler_dir outputs/full_rerun_YYYYMMDD_HHMMSS --faithfulness_dir outputs/full_rerun_YYYYMMDD_HHMMSS --cost_dir outputs/full_rerun_YYYYMMDD_HHMMSS
```

## Full Run

AutoDL full rerun:

```bash
bash scripts/run_full_rerun_pipeline.sh --step all --device cuda --skip_existing --max_parallel 1
```

The faster tuned-HERO-first route is:

```bash
bash scripts/run_rerun_with_tuned_hero.sh --device cuda --skip_existing --reuse_baselines_dir outputs/submission_main_20260711_225137 --models hero
```

Resume after interruption by repeating the same command with `--output_dir <same_dir> --skip_existing`, or run only one step:

```bash
bash scripts/run_full_rerun_pipeline.sh --step robustness --output_dir outputs/full_rerun_YYYYMMDD_HHMMSS --device cuda --skip_existing
```

## Missing And Failed Runs

Each suite root contains:

```text
raw/
logs/
summary/
tables/
figures/
figure_data/
configs/
reports/
failed_runs.csv
run_manifest.json
```

Inspect failures:

```bash
cat outputs/full_rerun_YYYYMMDD_HHMMSS/failed_runs.csv
```

Generate rerun commands for missing runs:

```bash
python scripts/check_experiment_completeness.py --output_dir outputs/full_rerun_YYYYMMDD_HHMMSS --generate_commands
```

## Performance Diagnosis And Leakage Checks

After summarization, inspect:

```bash
cat outputs/full_rerun_YYYYMMDD_HHMMSS/summary/performance_diagnosis.md
```

Run the no-leakage and tuned-config tests:

```bash
python -m pytest tests/test_no_test_leakage.py tests/test_best_checkpoint_selection.py tests/test_tuned_config_loading.py -q
```

Expected evidence of no test leakage:

- `best_hero_configs.csv` has `selected_by=validation_AUPRC`.
- `best_configs/hero_full_<dataset>.yaml` contains validation metrics only in `selection.val_metrics`.
- `metrics.json` records final test metrics from the best validation checkpoint through `best_epoch` and `selected_by`.

## Download From AutoDL

The final paper-ready directory is:

```text
outputs/final_artifacts/
```

Download `tables_csv/`, `tables_latex/`, `figures_pdf/`, `figures_png/`, `figure_data/`, and `reports/`. Do not edit metric CSVs manually; rebuild them from raw results with the final suite.
