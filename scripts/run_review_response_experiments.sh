#!/usr/bin/env bash
set -u

DEVICE="cuda"
MAIN_DIR="outputs/submission_main_20260711_225137"
SKIP_EXISTING=0
QUICK_TEST=0
RUN_ROBUSTNESS=0
RUN_LABELER=0
RUN_FAITHFULNESS=0
RUN_SENSITIVITY=0
RUN_ALL=0
BUILD_FINAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick_test) QUICK_TEST=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --main_dir) MAIN_DIR="$2"; shift 2 ;;
    --run_robustness) RUN_ROBUSTNESS=1; shift ;;
    --run_labeler) RUN_LABELER=1; shift ;;
    --run_faithfulness) RUN_FAITHFULNESS=1; shift ;;
    --run_sensitivity) RUN_SENSITIVITY=1; shift ;;
    --run_all) RUN_ALL=1; shift ;;
    --build_final) BUILD_FINAL=1; shift ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

if [[ "$RUN_ALL" -eq 1 ]]; then
  RUN_ROBUSTNESS=1
  RUN_LABELER=1
  RUN_FAITHFULNESS=1
  RUN_SENSITIVITY=1
  BUILD_FINAL=1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="outputs/review_response_${STAMP}"
if [[ "$QUICK_TEST" -eq 1 ]]; then
  ROOT="outputs/review_response_quick_${STAMP}"
fi
mkdir -p "$ROOT/logs" "$ROOT/summary"
FAILED="$ROOT/summary/failed_runs.csv"
echo "stage,exit_code,command" > "$FAILED"

run_step() {
  local stage="$1"
  shift
  echo "[$stage] $*"
  "$@" > "$ROOT/logs/${stage}.log" 2>&1
  local code=$?
  if [[ "$code" -ne 0 ]]; then
    echo "$stage,$code,\"$*\"" >> "$FAILED"
    echo "[failed] $stage exit_code=$code (see $ROOT/logs/${stage}.log)"
  else
    echo "[ok] $stage"
  fi
  return 0
}

DATASETS=("yelp_academic" "amazon_video")
SEEDS=("0" "1" "2" "3" "4")
NOISE_RATIOS=("0.0" "0.1" "0.2" "0.3" "0.4")
TOPKS=("1" "3" "5")
if [[ "$QUICK_TEST" -eq 1 ]]; then
  DATASETS=("yelp_academic")
  SEEDS=("0")
  NOISE_RATIOS=("0.0" "0.1")
  TOPKS=("1")
fi

SKIP_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  SKIP_ARGS=(--skip_existing)
fi

if [[ -d "$MAIN_DIR" ]]; then
  run_step summarize_main python scripts/summarize_experiment_suite.py --output_dir "$MAIN_DIR"
  run_step significance python scripts/run_significance_tests.py --main_dir "$MAIN_DIR"
else
  echo "main_dir_missing,0,\"$MAIN_DIR\"" >> "$FAILED"
  echo "[missing] main_dir=$MAIN_DIR"
fi

ROBUSTNESS_DIR="$ROOT/robustness"
LABELER_DIR="$ROOT/labeler_comparison"
FAITHFULNESS_DIR="$ROOT/faithfulness"
SENSITIVITY_DIR="$ROOT/sensitivity"
COST_DIR="$ROOT/cost"

if [[ "$RUN_ROBUSTNESS" -eq 1 ]]; then
  run_step robustness python scripts/run_llm_annotation_robustness.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --noise_ratios "${NOISE_RATIOS[@]}" --output_dir "$ROBUSTNESS_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
fi
if [[ "$RUN_LABELER" -eq 1 ]]; then
  run_step labeler python scripts/run_llm_labeler_comparison.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --output_dir "$LABELER_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
fi
if [[ "$RUN_FAITHFULNESS" -eq 1 ]]; then
  run_step faithfulness python scripts/run_evidence_faithfulness.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --topks "${TOPKS[@]}" --input_dir "$MAIN_DIR" --output_dir "$FAITHFULNESS_DIR" --device "$DEVICE"
fi
if [[ "$RUN_SENSITIVITY" -eq 1 ]]; then
  SENS_ARGS=()
  if [[ "$QUICK_TEST" -eq 1 ]]; then
    SENS_ARGS=(--quick_test)
  fi
  run_step sensitivity python scripts/run_sensitivity_analysis.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --output_dir "$SENSITIVITY_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}" "${SENS_ARGS[@]}"
fi

run_step cost python scripts/collect_cost_scalability.py --input_dir "$MAIN_DIR" --output_dir "$COST_DIR" --datasets "${DATASETS[@]}"

if [[ "$BUILD_FINAL" -eq 1 ]]; then
  run_step final_artifacts python scripts/build_final_tables_and_figures.py --main_dir "$MAIN_DIR" --robustness_dir "$ROBUSTNESS_DIR" --labeler_dir "$LABELER_DIR" --faithfulness_dir "$FAITHFULNESS_DIR" --cost_dir "$COST_DIR" --output_dir outputs/final_artifacts --allow_missing
  run_step seed_stability python scripts/plot_seed_stability.py --input_csv "$MAIN_DIR/summary/all_raw_runs.csv" --output_dir outputs/final_artifacts
fi

echo "Review-response experiment driver finished."
echo "Run root: $ROOT"
echo "Final artifacts: outputs/final_artifacts"
