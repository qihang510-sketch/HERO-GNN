#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

DEVICE="cuda"
OUTPUT_ROOT=""
TUNED_CONFIG_DIR=""
REUSE_BASELINES_DIR=""
RERUN_BASELINES=0
QUICK_TEST=0
SKIP_EXISTING=0
DRY_RUN=0
MODELS=()
DATASETS=()
SEEDS=("0" "1" "2" "3" "4")
MAX_PARALLEL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --output_dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --tuned_config_dir) TUNED_CONFIG_DIR="$2"; shift 2 ;;
    --reuse_baselines_dir) REUSE_BASELINES_DIR="$2"; shift 2 ;;
    --rerun_baselines) RERUN_BASELINES=1; shift ;;
    --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --quick_test) QUICK_TEST=1; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="outputs/rerun_tuned_hero_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$TUNED_CONFIG_DIR" ]]; then
  TUNED_CONFIG_DIR="$OUTPUT_ROOT/tuning_hero"
fi

COMMON=(--device "$DEVICE" --output_dir "$OUTPUT_ROOT" --tuned_config_dir "$TUNED_CONFIG_DIR" --max_parallel "$MAX_PARALLEL" --seeds "${SEEDS[@]}")
TUNE_ARGS=(--device "$DEVICE" --output_dir "$TUNED_CONFIG_DIR" --seeds "${SEEDS[@]}")
if [[ ${#DATASETS[@]} -gt 0 ]]; then
  COMMON+=(--datasets "${DATASETS[@]}")
  TUNE_ARGS+=(--datasets "${DATASETS[@]}")
fi
if [[ "$QUICK_TEST" -eq 1 ]]; then
  COMMON+=(--quick_test)
  TUNE_ARGS+=(--quick_test)
fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  COMMON+=(--skip_existing)
  TUNE_ARGS+=(--skip_existing)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  COMMON+=(--dry_run)
  TUNE_ARGS+=(--dry_run)
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  if [[ "$RERUN_BASELINES" -eq 1 ]]; then
    MODEL_ARGS=()
  else
    MODEL_ARGS=(--models hero)
  fi
else
  MODEL_ARGS=(--models "${MODELS[@]}")
fi

if [[ -n "$REUSE_BASELINES_DIR" && "$RERUN_BASELINES" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/summary"
  if [[ -d "$REUSE_BASELINES_DIR/raw" ]]; then cp -an "$REUSE_BASELINES_DIR/raw/." "$OUTPUT_ROOT/raw/" || true; fi
  if [[ -d "$REUSE_BASELINES_DIR/logs" ]]; then cp -an "$REUSE_BASELINES_DIR/logs/." "$OUTPUT_ROOT/logs/" || true; fi
  if [[ -d "$REUSE_BASELINES_DIR/summary" ]]; then cp -an "$REUSE_BASELINES_DIR/summary/." "$OUTPUT_ROOT/summary/" || true; fi
fi

echo "[step 1] tune HERO on validation"
python scripts/tune_hero_hyperparams.py "${TUNE_ARGS[@]}"

echo "[step 2] rerun main"
python scripts/run_experiment_suite.py --suite main "${COMMON[@]}" "${MODEL_ARGS[@]}"

echo "[step 3] rerun transfer"
python scripts/run_experiment_suite.py --suite transfer "${COMMON[@]}" "${MODEL_ARGS[@]}"

echo "[step 4] rerun ablation from tuned full-HERO configs"
python scripts/run_experiment_suite.py --suite ablation "${COMMON[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] python scripts/summarize_experiment_suite.py --output_dir $OUTPUT_ROOT"
  echo "[dry-run] python scripts/run_significance_tests.py --main_dir $OUTPUT_ROOT"
else
  echo "[step 5] summarize"
  python scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_ROOT"
  echo "[step 6] significance"
  python scripts/run_significance_tests.py --main_dir "$OUTPUT_ROOT"
  echo "[step 7] performance diagnosis"
  python scripts/build_performance_diagnosis.py --output_dir "$OUTPUT_ROOT" --tuned_config_dir "$TUNED_CONFIG_DIR"
fi

echo "[step 8] final artifacts"
python scripts/run_experiment_suite.py --suite final "${COMMON[@]}" --main_dir "$OUTPUT_ROOT" --ablation_dir "$OUTPUT_ROOT" --cost_dir "$OUTPUT_ROOT"

echo "Done: $OUTPUT_ROOT"
