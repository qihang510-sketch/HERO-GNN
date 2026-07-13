# HERO Experiment Rerun Guide

This guide describes the final staged rerun workflow for reviewer-response experiments. The scripts never fabricate metrics, never hand-edit CSV values, and never select configs from test metrics. Missing or unavailable artifacts remain marked as `missing`, `skipped`, `unavailable`, or `insufficient_seeds`.

## Output Layout

Each full rerun writes to a fresh root:

```text
outputs/review_rerun_<timestamp>/
  tuning/
  main/
  transfer/
  ablation/
  robustness/
  labeler_comparison/
  faithfulness/
  sensitivity/
  cost/
  final_artifacts/
    tables_csv/
    tables_latex/
    figures_pdf/
    figures_png/
    figure_data/
    reports/
```

`quick_test/` and `significance/` may also be created as helper stage directories.

## Full Run

AutoDL recommended full command:

```bash
bash scripts/run_all_experiments_for_review.sh \
  --device cuda \
  --seeds 0 1 2 3 4 \
  --output_root outputs/review_rerun_$(date +%Y%m%d_%H%M%S) \
  --skip_existing \
  --continue_on_error
```

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_experiments_for_review.ps1 `
  -Device cuda `
  -Seeds 0,1,2,3,4 `
  -OutputRoot outputs/review_rerun_YYYYMMDD_HHMMSS `
  -SkipExisting `
  -ContinueOnError
```

## Stage Commands

All stages can be run independently:

```bash
bash scripts/run_stage.sh --stage quick_test
bash scripts/run_stage.sh --stage tune_hero
bash scripts/run_stage.sh --stage main_text_rich
bash scripts/run_stage.sh --stage transfer
bash scripts/run_stage.sh --stage ablation
bash scripts/run_stage.sh --stage significance
bash scripts/run_stage.sh --stage robustness
bash scripts/run_stage.sh --stage labeler_comparison
bash scripts/run_stage.sh --stage faithfulness
bash scripts/run_stage.sh --stage sensitivity
bash scripts/run_stage.sh --stage cost
bash scripts/run_stage.sh --stage seed_stability
bash scripts/run_stage.sh --stage evidence_cases
bash scripts/run_stage.sh --stage final_artifacts
```

Use the same `--output_root` when resuming an interrupted run.

## Common Commands

Quick test:

```bash
bash scripts/run_stage.sh --stage quick_test --device cuda
```

Only tune HERO:

```bash
bash scripts/run_stage.sh --stage tune_hero --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --skip_existing
```

Only run text-rich main experiments:

```bash
bash scripts/run_stage.sh --stage main_text_rich --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --tuned_config_dir outputs/review_rerun_YYYYMMDD_HHMMSS/tuning
```

Only run transfer experiments:

```bash
bash scripts/run_stage.sh --stage transfer --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --tuned_config_dir outputs/review_rerun_YYYYMMDD_HHMMSS/tuning
```

Only run ablation:

```bash
bash scripts/run_stage.sh --stage ablation --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --tuned_config_dir outputs/review_rerun_YYYYMMDD_HHMMSS/tuning
```

Only run robustness:

```bash
bash scripts/run_stage.sh --stage robustness --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS
```

Only run labeler comparison:

```bash
bash scripts/run_stage.sh --stage labeler_comparison --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS
```

Only run faithfulness:

```bash
bash scripts/run_stage.sh --stage faithfulness --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS
```

Only run sensitivity:

```bash
bash scripts/run_stage.sh --stage sensitivity --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS
```

Only generate final tables and figures:

```bash
bash scripts/run_stage.sh --stage final_artifacts --output_root outputs/review_rerun_YYYYMMDD_HHMMSS
```

If you prefer to pass the `best_configs` directory directly, this is also supported:

```bash
bash scripts/run_stage.sh --stage main_text_rich --device cuda --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --tuned_config_dir outputs/review_rerun_YYYYMMDD_HHMMSS/tuning/best_configs
```

## Recovery And Missing Runs

Resume without rerunning completed runs:

```bash
bash scripts/run_all_experiments_for_review.sh --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --device cuda --skip_existing --continue_on_error
```

Only attempt missing runs for one stage:

```bash
bash scripts/run_stage.sh --stage main_text_rich --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --device cuda --only_missing
```

Rerun a stage after failures:

```bash
bash scripts/run_stage.sh --stage robustness --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --device cuda --rerun_failed --continue_on_error
```

Generate commands for missing runs:

```bash
bash scripts/run_stage.sh --stage main_text_rich --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --generate_missing_commands
```

Dry run:

```bash
bash scripts/run_all_experiments_for_review.sh --device cuda --dry_run --output_root outputs/review_rerun_dry_run
```

## Final Artifact Check

Run the final checker:

```bash
python scripts/check_final_artifacts.py --output_root outputs/review_rerun_YYYYMMDD_HHMMSS --final_dir outputs/review_rerun_YYYYMMDD_HHMMSS/final_artifacts --allow_missing
```

The checker writes:

```text
outputs/review_rerun_YYYYMMDD_HHMMSS/final_artifacts/reports/FINAL_EXPERIMENT_REPORT.md
outputs/review_rerun_YYYYMMDD_HHMMSS/final_artifacts/reports/final_artifact_checks.csv
```

Use `--strict` if you want the checker to exit nonzero when required artifacts are missing.

## AutoDL Recommended Order

1. `quick_test`
2. `tune_hero`
3. `main_text_rich`
4. `transfer`
5. `ablation`
6. `significance`
7. `robustness`
8. `labeler_comparison`
9. `faithfulness`
10. `sensitivity`
11. `cost`
12. `seed_stability`
13. `evidence_cases`
14. `final_artifacts`

The all-in-one script follows this order automatically.

## Integrity Notes

- Do not edit metric CSVs manually.
- Do not use test metrics to select HERO configs.
- `tuning/best_configs/hero_full_<dataset>.yaml` records validation-selected configs.
- `final_artifacts/reports/FINAL_EXPERIMENT_REPORT.md` reports missing/unavailable items instead of inventing them.
