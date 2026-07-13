#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

STEP="all"
DEVICE="cuda"
MAIN_DIR="outputs/submission_main_20260711_225137"
OUTPUT_ROOT=""
TUNED_CONFIG_DIR=""
QUICK_TEST=0
SKIP_EXISTING=0
DRY_RUN=0
MAX_PARALLEL=1
SEEDS=("0" "1" "2" "3" "4")
DATASETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --main_dir) MAIN_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --tuned_config_dir) TUNED_CONFIG_DIR="$2"; shift 2 ;;
    --quick_test) QUICK_TEST=1; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="outputs/full_rerun_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$TUNED_CONFIG_DIR" ]]; then
  TUNED_CONFIG_DIR="$OUTPUT_ROOT/tuning_hero"
fi

COMMON=(--device "$DEVICE" --output_dir "$OUTPUT_ROOT" --max_parallel "$MAX_PARALLEL")
COMMON+=(--seeds "${SEEDS[@]}")
if [[ ${#DATASETS[@]} -gt 0 ]]; then COMMON+=(--datasets "${DATASETS[@]}"); fi
if [[ "$QUICK_TEST" -eq 1 ]]; then COMMON+=(--quick_test); fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then COMMON+=(--skip_existing); fi
if [[ "$DRY_RUN" -eq 1 ]]; then COMMON+=(--dry_run); fi

run_tune() {
  local args=(--device "$DEVICE" --output_dir "$TUNED_CONFIG_DIR" --seeds "${SEEDS[@]}")
  if [[ ${#DATASETS[@]} -gt 0 ]]; then args+=(--datasets "${DATASETS[@]}"); fi
  if [[ "$QUICK_TEST" -eq 1 ]]; then args+=(--quick_test); fi
  if [[ "$SKIP_EXISTING" -eq 1 ]]; then args+=(--skip_existing); fi
  if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
  python scripts/tune_hero_hyperparams.py "${args[@]}"
}

run_suite() {
  local suite="$1"
  python scripts/run_experiment_suite.py --suite "$suite" "${COMMON[@]}" --tuned_config_dir "$TUNED_CONFIG_DIR" --main_dir "$MAIN_DIR" --input_dir "$OUTPUT_ROOT"
}

run_significance() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] python scripts/summarize_experiment_suite.py --output_dir $OUTPUT_ROOT"
    echo "[dry-run] python scripts/run_significance_tests.py --main_dir $OUTPUT_ROOT"
    return
  fi
  python scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_ROOT"
  python scripts/run_significance_tests.py --main_dir "$OUTPUT_ROOT"
}

run_final() {
  python scripts/run_experiment_suite.py --suite final "${COMMON[@]}" --main_dir "$OUTPUT_ROOT" --ablation_dir "$OUTPUT_ROOT" --robustness_dir "$OUTPUT_ROOT" --labeler_dir "$OUTPUT_ROOT" --faithfulness_dir "$OUTPUT_ROOT" --sensitivity_dir "$OUTPUT_ROOT" --cost_dir "$OUTPUT_ROOT"
}

run_step() {
  case "$1" in
    quick_test)
      local args=(--suite all --quick_test --output_dir "$OUTPUT_ROOT/quick_test" --device "$DEVICE" --max_parallel "$MAX_PARALLEL")
      if [[ "$SKIP_EXISTING" -eq 1 ]]; then args+=(--skip_existing); fi
      if [[ "$DRY_RUN" -eq 1 ]]; then args+=(--dry_run); fi
      python scripts/run_experiment_suite.py "${args[@]}"
      ;;
    tune|tune_hero_hyperparams) run_tune ;;
    main) run_suite main ;;
    transfer) run_suite transfer ;;
    ablation) run_suite ablation ;;
    significance) run_significance ;;
    robustness) run_suite robustness ;;
    labeler_comparison) run_suite labeler_comparison ;;
    faithfulness) run_suite faithfulness ;;
    sensitivity) run_suite sensitivity ;;
    cost) run_suite cost ;;
    final) run_final ;;
    *) echo "Unknown step: $1" >&2; exit 2 ;;
  esac
}

if [[ "$STEP" == "all" ]]; then
  run_step quick_test
  run_step tune
  run_step main
  run_step transfer
  run_step ablation
  run_step significance
  run_step robustness
  run_step labeler_comparison
  run_step faithfulness
  run_step sensitivity
  run_step cost
  run_step final
else
  run_step "$STEP"
fi

echo "Pipeline finished. Output root: $OUTPUT_ROOT"
