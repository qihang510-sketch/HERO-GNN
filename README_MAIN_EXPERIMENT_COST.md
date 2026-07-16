# Main Experiment Cost Module

This module summarizes computational cost for HERO main experiments only. It is intended for supplementary material and reviewer response tables.

It only includes formal `main` / `main_text_rich` runs by default. Transfer, ablation, robustness, faithfulness, sensitivity, and other supplementary experiments are not part of this table. `main_quick` and `main_text_rich_quick` are excluded unless `--include_quick_test` is passed.

Cost values are extracted from logged runtime files and cached artifacts. Missing fields are reported as N/A rather than estimated.

## Inputs

The collector searches existing output roots for:

- per-run `runtime.json`, `metrics.json`, `config.json`, and `skip_reason.json`
- `run_manifest.json`
- `summary/all_raw_runs.csv`
- `logs/*.log` and `logs/*.txt` time fields
- HERO annotation caches such as `llm_labels.jsonl`, `qwen_annotations.jsonl`, `cached_llm_annotations.jsonl`, and `mechanism_annotations.jsonl`
- optional processed graph metadata for node, edge, feature, and split counts

It does not estimate runtime from file modification times.

## Outputs

Standalone outputs are written under `outputs/main_cost` by default:

- `raw/main_cost_raw_runs.csv`
- `raw/main_cost_runtime_sources.jsonl`
- `tables_csv/supp_table_main_experiment_cost.csv`
- `tables_csv/supp_table_main_experiment_cost_compact.csv`
- `tables_latex/supp_table_main_experiment_cost.tex`
- `tables_latex/supp_table_main_experiment_cost_compact.tex`
- `tables_markdown/supp_table_main_experiment_cost.md`
- `tables_markdown/supp_table_main_experiment_cost_compact.md`
- `reports/MAIN_EXPERIMENT_COST_REPORT.md`

With `--copy_to_final_artifacts`, these are copied to:

- `outputs/final_artifacts/tables_csv/supp_table_main_experiment_cost.csv`
- `outputs/final_artifacts/tables_csv/supp_table_main_experiment_cost_compact.csv`
- `outputs/final_artifacts/tables_latex/supp_table_main_experiment_cost.tex`
- `outputs/final_artifacts/tables_latex/supp_table_main_experiment_cost_compact.tex`
- `outputs/final_artifacts/reports/MAIN_EXPERIMENT_COST_REPORT.md`

## Run Standalone

```bash
python scripts/collect_main_experiment_cost.py \
  --source_outputs outputs/review_rerun_20260713_210101 outputs/submission_main_20260711_225137 \
  --output_dir outputs/main_cost \
  --datasets yelp_academic amazon_video \
  --suite_names main main_text_rich \
  --models mlp gcn gat graphsage care_gnn graphconsis pc_gnn bwgnn linkx dgp mled hero hero_full hero_gnn \
  --write_latex \
  --write_markdown \
  --copy_to_final_artifacts \
  --final_artifacts_dir outputs/final_artifacts
```

## Run Through Stage

```bash
bash scripts/run_stage.sh --stage main_cost --device cuda --skip_existing
```

The stage writes logs to:

```text
<output_root>/logs/main_cost.log
```

Failures are appended to:

```text
<output_root>/failed_runs.csv
```

## Validate

```bash
python scripts/validate_main_cost_table.py \
  --cost_dir outputs/main_cost \
  --datasets yelp_academic amazon_video
```

Validation rejects forecast, planning-only, not-for-paper, and manually fabricated example markers. `N/A`, `unavailable`, and `runtime_unavailable` are allowed missing-value markers.

## Table Interpretation

The detailed table keeps one row per suite, dataset, and model. It reports run counts, successful seed counts, runtime means/std, memory, HERO annotation/cache metadata, graph statistics, provenance, missing fields, and status.

The compact table is paper friendly. It keeps:

- `Dataset`
- `Model`
- `#Seeds`
- `Train Time / Seed`
- `Total Runtime`
- `Peak GPU Mem.`
- `LLM Annotation Cost`
- `Cache Size`
- `Cost Source`
- `Status`

For baseline models, `LLM Annotation Cost` is `N/A`. For HERO, annotation source is only reported as `cached_local_qwen` when the cache file or labeler metadata actually indicates Qwen/local Qwen; otherwise it remains `cached_annotation` or `unavailable`.

## Missing Runtime

If runtime fields are absent, the table keeps the row and marks the missing fields as `N/A`. If no usable runtime source exists for a requested model or dataset, `status=runtime_unavailable`.
