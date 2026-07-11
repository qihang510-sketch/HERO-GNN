# Final Submission Experiment Commands

This repository now has two top-level submission runners:

- Linux / AutoDL: `scripts/run_submission_experiments.sh`
- Windows PowerShell: `scripts/run_submission_experiments.ps1`

Both runners create a fresh timestamped output directory by default:

```text
outputs/submission_full_YYYYMMDD_HHMMSS/
```

They do not overwrite an existing directory unless you explicitly pass `--skip_existing`, `--rerun_missing`, or `--summarize_only`.

## Local Quick Test

Linux / AutoDL shell:

```bash
bash scripts/run_submission_experiments.sh --quick_test --device cpu
```

Windows PowerShell:

```powershell
.\scripts\run_submission_experiments.ps1 -QuickTest -Device cpu
```

Quick test only runs:

- dataset: `yelp_academic`
- models: `mlp`, `gcn`, `hero`
- seed: `0`

It is a pipeline smoke test, not a complete paper run.

## AutoDL Full Run

Linux / AutoDL shell:

```bash
bash scripts/run_submission_experiments.sh --device cuda --max_parallel 1 --skip_existing
```

Windows PowerShell on a local GPU machine:

```powershell
.\scripts\run_submission_experiments.ps1 -Device cuda -MaxParallel 1 -SkipExisting
```

The full run executes:

- main experiments on `yelp_academic`, `amazon_video`, `fraud_yelp`, `fraud_amazon`, `elliptic`
- all dataset-supported baselines plus HERO
- HERO ablations on text-rich datasets
- LLM annotation robustness
- LLM labeler comparison
- evidence-chain faithfulness
- cost/scalability collection
- completeness checks, summaries, significance tests, and final table refresh

## Resume After Interruption

Reuse the same output directory with `--skip_existing`:

```bash
bash scripts/run_submission_experiments.sh \
  --output_dir outputs/submission_full_YYYYMMDD_HHMMSS \
  --device cuda \
  --skip_existing
```

PowerShell:

```powershell
.\scripts\run_submission_experiments.ps1 `
  -OutputDir outputs/submission_full_YYYYMMDD_HHMMSS `
  -Device cuda `
  -SkipExisting
```

Existing `metrics.json` files are skipped. Missing or failed runs are attempted again.

## Rerun Missing Only

Use `--rerun_missing`, which implies `--skip_existing`:

```bash
bash scripts/run_submission_experiments.sh \
  --output_dir outputs/submission_full_YYYYMMDD_HHMMSS \
  --device cuda \
  --rerun_missing
```

PowerShell:

```powershell
.\scripts\run_submission_experiments.ps1 `
  -OutputDir outputs/submission_full_YYYYMMDD_HHMMSS `
  -Device cuda `
  -RerunMissing
```

The runner also writes `summary/missing_commands.txt` after completeness checking.

## Summarize Only

Use this after manually adding or copying raw per-seed results:

```bash
bash scripts/run_submission_experiments.sh \
  --output_dir outputs/submission_full_YYYYMMDD_HHMMSS \
  --summarize_only
```

PowerShell:

```powershell
.\scripts\run_submission_experiments.ps1 `
  -OutputDir outputs/submission_full_YYYYMMDD_HHMMSS `
  -SummarizeOnly
```

## Final CSV Files

Expected final files:

```text
outputs/submission_full_*/summary/all_raw_runs.csv
outputs/submission_full_*/summary/table_main_mean_std.csv
outputs/submission_full_*/summary/table_transfer_mean_std.csv
outputs/submission_full_*/summary/table_ablation_mean_std.csv
outputs/submission_full_*/summary/table_significance.csv
outputs/submission_full_*/summary/table_llm_robustness.csv
outputs/submission_full_*/summary/table_llm_labeler_comparison.csv
outputs/submission_full_*/summary/table_faithfulness.csv
outputs/submission_full_*/summary/table_cost_scalability.csv
outputs/submission_full_*/summary/table_missing_runs.csv
outputs/submission_full_*/summary/failed_runs.csv
```

To inspect quickly:

```bash
ls outputs/submission_full_YYYYMMDD_HHMMSS/summary
```

PowerShell:

```powershell
Get-ChildItem outputs\submission_full_YYYYMMDD_HHMMSS\summary
```

Failed commands are recorded in `summary/failed_runs.csv`. Each failed run or failed stage has an `error.log`.

## CSV To LaTeX

Convert all `table_*.csv` files to LaTeX:

```bash
python -c "from pathlib import Path; import pandas as pd; src=Path('outputs/submission_full_YYYYMMDD_HHMMSS/summary'); dst=Path('paper/tables'); dst.mkdir(parents=True, exist_ok=True); [ (dst/(p.stem+'.tex')).write_text(pd.read_csv(p).to_latex(index=False), encoding='utf-8') for p in src.glob('table_*.csv') ]"
```

PowerShell:

```powershell
python -c "from pathlib import Path; import pandas as pd; src=Path('outputs/submission_full_YYYYMMDD_HHMMSS/summary'); dst=Path('paper/tables'); dst.mkdir(parents=True, exist_ok=True); [ (dst/(p.stem+'.tex')).write_text(pd.read_csv(p).to_latex(index=False), encoding='utf-8') for p in src.glob('table_*.csv') ]"
```

Replace `YYYYMMDD_HHMMSS` with the actual run directory suffix.
