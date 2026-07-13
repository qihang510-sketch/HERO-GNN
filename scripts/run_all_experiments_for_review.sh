#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

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
SEEDS=("0" "1" "2" "3" "4")
DATASETS=()
MODELS=()
VARIANTS=()
STAGES=(
  quick_test
  tune_hero
  main_text_rich
  transfer
  ablation
  significance
  robustness
  labeler_comparison
  faithfulness
  sensitivity
  cost
  seed_stability
  evidence_cases
  final_artifacts
)

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --models) shift; MODELS=(); while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
    --variants) shift; VARIANTS=(); while [[ $# -gt 0 && "$1" != --* ]]; do VARIANTS+=("$1"); shift; done ;;
    --stages) shift; STAGES=(); while [[ $# -gt 0 && "$1" != --* ]]; do STAGES+=("$1"); shift; done ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="outputs/review_rerun_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "$TUNED_CONFIG_DIR" ]]; then
  TUNED_CONFIG_DIR="$OUTPUT_ROOT/tuning"
fi

COMMON=(--device "$DEVICE" --output_root "$OUTPUT_ROOT" --tuned_config_dir "$TUNED_CONFIG_DIR" --data_root "$DATA_ROOT" --max_parallel "$MAX_PARALLEL" --seeds "${SEEDS[@]}")
if [[ ${#DATASETS[@]} -gt 0 ]]; then COMMON+=(--datasets "${DATASETS[@]}"); fi
if [[ ${#MODELS[@]} -gt 0 ]]; then COMMON+=(--models "${MODELS[@]}"); fi
if [[ ${#VARIANTS[@]} -gt 0 ]]; then COMMON+=(--variants "${VARIANTS[@]}"); fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then COMMON+=(--skip_existing); fi
if [[ "$RERUN_FAILED" -eq 1 ]]; then COMMON+=(--rerun_failed); fi
if [[ "$ONLY_MISSING" -eq 1 ]]; then COMMON+=(--only_missing); fi
if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then COMMON+=(--continue_on_error); fi
if [[ "$DRY_RUN" -eq 1 ]]; then COMMON+=(--dry_run); fi
if [[ "$GENERATE_MISSING_COMMANDS" -eq 1 ]]; then COMMON+=(--generate_missing_commands); fi

echo "Review rerun output root: $OUTPUT_ROOT"
for stage in "${STAGES[@]}"; do
  echo "========== stage: $stage =========="
  bash scripts/run_stage.sh --stage "$stage" "${COMMON[@]}"
done

echo "All requested review experiment stages finished."
echo "Final artifacts: $OUTPUT_ROOT/final_artifacts"
