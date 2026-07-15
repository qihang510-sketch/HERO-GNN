#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

STAGE="quick_test"
DEVICE="cuda"
OUTPUT_ROOT=""
TUNED_CONFIG_DIR=""
DATA_ROOT="data"
MAX_PARALLEL=1
SKIP_EXISTING=0
RERUN_FAILED=0
ONLY_MISSING=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
GENERATE_MISSING_COMMANDS=0
ALLOW_MISSING=1
QUICK_TEST=0
SEEDS=("0" "1" "2" "3" "4")
DATASETS=()
MODELS=()
VARIANTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output_root|--output_dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --tuned_config_dir) TUNED_CONFIG_DIR="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --rerun_failed) RERUN_FAILED=1; shift ;;
    --only_missing) ONLY_MISSING=1; shift ;;
    --continue_on_error) CONTINUE_ON_ERROR=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    --generate_missing_commands) GENERATE_MISSING_COMMANDS=1; shift ;;
    --quick_test) QUICK_TEST=1; shift ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
    --variants) shift; VARIANTS=(); while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS+=("$1"); shift; done ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="outputs/review_rerun_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$TUNED_CONFIG_DIR" ]]; then
  TUNED_CONFIG_DIR="$OUTPUT_ROOT/tuning"
fi
if [[ "$ONLY_MISSING" -eq 1 ]]; then
  SKIP_EXISTING=1
fi
if [[ "$RERUN_FAILED" -eq 1 ]]; then
  SKIP_EXISTING=0
fi

COMMON=(--device "$DEVICE" --data_root "$DATA_ROOT" --max_parallel "$MAX_PARALLEL" --seeds "${SEEDS[@]}")
if [[ ${#DATASETS[@]} -gt 0 ]]; then COMMON+=(--datasets "${DATASETS[@]}"); fi
if [[ ${#MODELS[@]} -gt 0 ]]; then COMMON+=(--models "${MODELS[@]}"); fi
if [[ ${#VARIANTS[@]} -gt 0 ]]; then COMMON+=(--variants "${VARIANTS[@]}"); fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then COMMON+=(--skip_existing); fi
if [[ "$DRY_RUN" -eq 1 ]]; then COMMON+=(--dry_run); fi
if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then COMMON+=(--continue_on_error); fi

stage_dir() {
  case "$1" in
    quick_test) echo "$OUTPUT_ROOT/quick_test" ;;
    tune_hero) echo "$OUTPUT_ROOT/tuning" ;;
    main_text_rich) echo "$OUTPUT_ROOT/main" ;;
    transfer) echo "$OUTPUT_ROOT/transfer" ;;
    ablation) echo "$OUTPUT_ROOT/ablation" ;;
    significance) echo "$OUTPUT_ROOT/significance" ;;
    robustness) echo "$OUTPUT_ROOT/robustness" ;;
    labeler_comparison) echo "$OUTPUT_ROOT/labeler_comparison" ;;
    faithfulness) echo "$OUTPUT_ROOT/faithfulness" ;;
    sensitivity) echo "$OUTPUT_ROOT/sensitivity" ;;
    cost) echo "$OUTPUT_ROOT/cost" ;;
    seed_stability|evidence_cases|final_artifacts) echo "$OUTPUT_ROOT/final_artifacts" ;;
    risk_card_cases) echo "$OUTPUT_ROOT/risk_card_cases" ;;
    *) echo "$OUTPUT_ROOT/$1" ;;
  esac
}

suite_for_stage() {
  case "$1" in
    main_text_rich) echo "main" ;;
    transfer) echo "transfer" ;;
    ablation) echo "ablation" ;;
    robustness) echo "robustness" ;;
    labeler_comparison) echo "labeler_comparison" ;;
    faithfulness) echo "faithfulness" ;;
    sensitivity) echo "sensitivity" ;;
    cost) echo "cost" ;;
    *) echo "" ;;
  esac
}

run_cmd() {
  echo "[run] $*"
  "$@"
}

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

record_failed_run() {
  local stage="$1"
  local code="$2"
  local command="$3"
  local reason="$4"
  local failed="$OUTPUT_ROOT/failed_runs.csv"
  mkdir -p "$OUTPUT_ROOT"
  if [[ ! -f "$failed" ]]; then
    echo "stage,exit_code,command,reason" > "$failed"
  fi
  {
    csv_escape "$stage"
    printf ',%s,' "$code"
    csv_escape "$command"
    printf ','
    csv_escape "$reason"
    printf '\n'
  } >> "$failed"
}

check_missing_commands() {
  local suite="$1"
  local dir="$2"
  if [[ -z "$suite" ]]; then
    echo "Stage $STAGE does not have run_experiment_suite completeness checks."
    return
  fi
  run_cmd python scripts/check_experiment_completeness.py --output_dir "$dir" --suite "$suite" --generate_commands
}

run_tune_hero() {
  local out
  out="$(stage_dir tune_hero)"
  local args=(--device "$DEVICE" --data_root "$DATA_ROOT" --output_dir "$out" --seeds "${SEEDS[@]}")
  if [[ ${#DATASETS[@]} -gt 0 ]]; then args+=(--datasets "${DATASETS[@]}"); fi
  if [[ "$SKIP_EXISTING" -eq 1 ]]; then args+=(--skip_existing); fi
  if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
  if [[ "$QUICK_TEST" -eq 1 ]]; then args+=(--quick_test); fi
  run_cmd python scripts/tune_hero_hyperparams.py "${args[@]}"
}

run_suite_stage() {
  local stage="$1"
  local suite="$2"
  local out
  out="$(stage_dir "$stage")"
  if [[ "$GENERATE_MISSING_COMMANDS" -eq 1 ]]; then
    check_missing_commands "$suite" "$out"
    return
  fi
  local args=(--suite "$suite" --output_dir "$out" "${COMMON[@]}" --tuned_config_dir "$TUNED_CONFIG_DIR" --input_dir "$OUTPUT_ROOT" --main_dir "$OUTPUT_ROOT/main")
  if [[ "$stage" == "ablation" ]]; then args+=(--ablation_dir "$OUTPUT_ROOT/ablation"); fi
  run_cmd python scripts/run_experiment_suite.py "${args[@]}"
}

run_significance() {
  local out
  out="$(stage_dir significance)"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] summarize main/transfer and build combined significance in $out"
    return
  fi
  mkdir -p "$out/summary"
  [[ -d "$OUTPUT_ROOT/main" ]] && run_cmd python scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_ROOT/main"
  [[ -d "$OUTPUT_ROOT/transfer" ]] && run_cmd python scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_ROOT/transfer"
  run_cmd python -c "from pathlib import Path; import pandas as pd, sys; out=Path(sys.argv[1]); frames=[]; [frames.append(pd.read_csv(p)) for p in map(Path, sys.argv[2:]) if p.exists()]; out.mkdir(parents=True, exist_ok=True); (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(out/'all_raw_runs.csv', index=False)" "$out/summary" "$OUTPUT_ROOT/main/summary/all_raw_runs.csv" "$OUTPUT_ROOT/transfer/summary/all_raw_runs.csv"
  run_cmd python scripts/run_significance_tests.py --main_dir "$out"
}

run_seed_stability() {
  local out
  out="$(stage_dir seed_stability)"
  run_cmd python scripts/plot_seed_stability.py --input_csv "$OUTPUT_ROOT/main/summary/all_raw_runs.csv" --output_dir "$out"
}

run_evidence_cases() {
  local out
  out="$(stage_dir evidence_cases)"
  local input="$OUTPUT_ROOT/faithfulness"
  if [[ ! -d "$input" ]]; then input="$OUTPUT_ROOT/main"; fi
  run_cmd python scripts/build_evidence_case_table.py --input_dir "$input" --output_dir "$out"
  run_risk_card_cases
}

run_risk_card_cases() {
  local out
  out="$(stage_dir risk_card_cases)"
  local final_out
  final_out="$(stage_dir final_artifacts)"
  local log_dir="$OUTPUT_ROOT/logs"
  local log_file="$log_dir/risk_card_cases.log"
  local datasets=(yelp_academic amazon_video fraud_yelp fraud_amazon elliptic)
  local build_cmd=(
    python scripts/build_risk_card_case_table.py
    --data_root "$DATA_ROOT"
    --output_dir "$out"
    --datasets "${datasets[@]}"
    --source_outputs "$OUTPUT_ROOT"
    --top_cases_per_dataset 1
    --include_field_trace
    --write_latex
    --write_markdown
    --copy_to_final_artifacts
    --final_artifacts_dir "$final_out"
  )
  local validate_cmd=(
    python scripts/validate_risk_card_cases.py
    --case_dir "$out"
    --datasets "${datasets[@]}"
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] ${build_cmd[*]}"
    echo "[dry-run] ${validate_cmd[*]}"
    return
  fi
  if [[ "$SKIP_EXISTING" -eq 1 \
    && -f "$out/tables_csv/table_risk_card_cases_compact.csv" \
    && -f "$out/tables_csv/table_risk_card_field_trace.csv" \
    && -f "$final_out/tables_csv/supp_table_risk_card_cases_compact.csv" \
    && -f "$final_out/tables_csv/supp_table_risk_card_field_trace.csv" ]]; then
    mkdir -p "$log_dir"
    echo "[skip] risk_card_cases existing outputs found" | tee "$log_file"
    return
  fi
  mkdir -p "$log_dir" "$out" "$final_out"
  echo "[run] ${build_cmd[*]}" > "$log_file"
  if "${build_cmd[@]}" >> "$log_file" 2>&1; then
    :
  else
    local code=$?
    record_failed_run "risk_card_cases" "$code" "${build_cmd[*]}" "build_risk_card_case_table.py failed; see $log_file"
    echo "[failed] risk_card_cases build exit_code=$code (see $log_file)" >&2
    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then return 0; fi
    return "$code"
  fi
  echo "[run] ${validate_cmd[*]}" >> "$log_file"
  if "${validate_cmd[@]}" >> "$log_file" 2>&1; then
    :
  else
    local code=$?
    record_failed_run "risk_card_cases" "$code" "${validate_cmd[*]}" "validate_risk_card_cases.py failed; see $log_file"
    echo "[failed] risk_card_cases validate exit_code=$code (see $log_file)" >&2
    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then return 0; fi
    return "$code"
  fi
  echo "[ok] risk_card_cases (log: $log_file)"
}

run_final_artifacts() {
  local out
  out="$(stage_dir final_artifacts)"
  local args=(
    --main_dir "$OUTPUT_ROOT/main"
    --transfer_dir "$OUTPUT_ROOT/transfer"
    --significance_dir "$OUTPUT_ROOT/significance"
    --ablation_dir "$OUTPUT_ROOT/ablation"
    --robustness_dir "$OUTPUT_ROOT/robustness"
    --labeler_dir "$OUTPUT_ROOT/labeler_comparison"
    --faithfulness_dir "$OUTPUT_ROOT/faithfulness"
    --sensitivity_dir "$OUTPUT_ROOT/sensitivity"
    --cost_dir "$OUTPUT_ROOT/cost"
    --output_dir "$out"
    --data_root "$DATA_ROOT"
    --allow_missing
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] python scripts/build_final_tables_and_figures.py ${args[*]}"
  else
    run_cmd python scripts/build_final_tables_and_figures.py "${args[@]}"
    run_cmd python scripts/check_final_artifacts.py --output_root "$OUTPUT_ROOT" --final_dir "$out" --allow_missing
  fi
}

run_stage() {
  case "$1" in
    quick_test)
      local quick_dir
      quick_dir="$(stage_dir quick_test)"
      local args=(--suite all --quick_test --output_dir "$quick_dir" --final_output_dir "$quick_dir/final_artifacts" --device "$DEVICE" --data_root "$DATA_ROOT" --max_parallel "$MAX_PARALLEL")
      if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
      if [[ "$SKIP_EXISTING" -eq 1 ]]; then args+=(--skip_existing); fi
      run_cmd python scripts/run_experiment_suite.py "${args[@]}"
      ;;
    tune_hero) run_tune_hero ;;
    main_text_rich) run_suite_stage main_text_rich main ;;
    transfer) run_suite_stage transfer transfer ;;
    ablation) run_suite_stage ablation ablation ;;
    significance) run_significance ;;
    robustness) run_suite_stage robustness robustness ;;
    labeler_comparison) run_suite_stage labeler_comparison labeler_comparison ;;
    faithfulness) run_suite_stage faithfulness faithfulness ;;
    sensitivity) run_suite_stage sensitivity sensitivity ;;
    cost) run_suite_stage cost cost ;;
    seed_stability) run_seed_stability ;;
    risk_card_cases) run_risk_card_cases ;;
    evidence_cases) run_evidence_cases ;;
    final_artifacts) run_final_artifacts ;;
    *) echo "Unknown stage: $1" >&2; exit 2 ;;
  esac
}

run_stage "$STAGE"
echo "Stage finished: $STAGE"
echo "Output root: $OUTPUT_ROOT"
