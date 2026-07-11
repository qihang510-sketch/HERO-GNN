#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
MAX_PARALLEL=1
QUICK_TEST=0
SKIP_EXISTING=0
SUMMARIZE_ONLY=0
RERUN_MISSING=0
OUTPUT_DIR=""
DATASETS=(yelp_academic amazon_video fraud_yelp fraud_amazon elliptic)
SEEDS=(0 1 2 3 4)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_submission_experiments.sh [options]

Options:
  --device cuda|cpu|auto
  --max_parallel N
  --quick_test
  --skip_existing
  --summarize_only
  --rerun_missing
  --output_dir PATH
  --datasets DATASET [DATASET ...]
  --seeds SEED [SEED ...]
  --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --max_parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --quick_test) QUICK_TEST=1; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --summarize_only) SUMMARIZE_ONLY=1; shift ;;
    --rerun_missing) RERUN_MISSING=1; SKIP_EXISTING=1; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --datasets)
      DATASETS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done
      ;;
    --seeds)
      SEEDS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$SUMMARIZE_ONLY" -eq 1 && -z "$OUTPUT_DIR" ]]; then
  echo "--summarize_only requires --output_dir" >&2
  exit 2
fi
if [[ "$RERUN_MISSING" -eq 1 && -z "$OUTPUT_DIR" ]]; then
  echo "--rerun_missing requires --output_dir" >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="outputs/submission_full_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -e "$OUTPUT_DIR" && "$SKIP_EXISTING" -eq 0 && "$SUMMARIZE_ONLY" -eq 0 && "$RERUN_MISSING" -eq 0 ]]; then
  echo "Refusing to reuse existing output dir without --skip_existing: $OUTPUT_DIR" >&2
  exit 2
fi

if [[ "$QUICK_TEST" -eq 1 ]]; then
  DATASETS=(yelp_academic)
  SEEDS=(0)
fi

mkdir -p "$OUTPUT_DIR"/{raw,summary,tables,figures,logs}
FAILED_CSV="$OUTPUT_DIR/summary/failed_runs.csv"
if [[ ! -f "$FAILED_CSV" ]]; then
  echo "suite,dataset,model,seed,exit_code,command,log_file,error_log" > "$FAILED_CSV"
fi

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

safe_name() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_.=-]+/_/g'
}

join_comma() {
  local IFS=,
  echo "$*"
}

text_datasets() {
  local out=()
  local dataset
  for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "yelp_academic" || "$dataset" == "amazon_video" ]]; then
      out+=("$dataset")
    fi
  done
  echo "${out[@]}"
}

append_failed() {
  local suite="$1" dataset="$2" model="$3" seed="$4" exit_code="$5" command="$6" log_file="$7" error_log="$8"
  {
    csv_escape "$suite"; printf ","
    csv_escape "$dataset"; printf ","
    csv_escape "$model"; printf ","
    csv_escape "$seed"; printf ","
    csv_escape "$exit_code"; printf ","
    csv_escape "$command"; printf ","
    csv_escape "$log_file"; printf ","
    csv_escape "$error_log"; printf "\n"
  } >> "$FAILED_CSV"
}

run_step() {
  local name="$1" suite="$2" dataset="$3" model="$4" seed="$5"
  shift 5
  local tag log_file fail_dir error_log exit_code command_string
  tag="$(safe_name "$name")"
  log_file="$OUTPUT_DIR/logs/${tag}.log"
  fail_dir="$OUTPUT_DIR/raw/failures/${tag}"
  error_log="$fail_dir/error.log"
  command_string="$*"
  echo "[RUN] $name"
  echo "[CMD] $command_string"
  "$@" > "$log_file" 2>&1
  exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    mkdir -p "$fail_dir"
    cp "$log_file" "$error_log"
    append_failed "$suite" "$dataset" "$model" "$seed" "$exit_code" "$command_string" "$log_file" "$error_log"
    echo "[FAILED] $name exit=$exit_code log=$error_log"
  else
    echo "[OK] $name"
  fi
}

write_expected_config() {
  local datasets_csv seeds_csv quick
  datasets_csv="$(join_comma "${DATASETS[@]}")"
  seeds_csv="$(join_comma "${SEEDS[@]}")"
  quick="$QUICK_TEST"
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$quick" "$datasets_csv" "$seeds_csv" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
quick = sys.argv[2] == "1"
datasets = [item for item in sys.argv[3].split(",") if item]
seeds = [int(item) for item in sys.argv[4].split(",") if item != ""]
main_matrix = {
    "yelp_academic": ["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero_gnn"],
    "amazon_video": ["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "dgp", "mled", "hero_gnn"],
    "fraud_yelp": ["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "hero_official"],
    "fraud_amazon": ["mlp", "gcn", "gat", "graphsage", "care_gnn", "graphconsis", "pc_gnn", "bwgnn", "linkx", "hero_official"],
    "elliptic": ["mlp", "gcn", "gat", "graphsage", "bwgnn", "linkx", "hogrl", "rgtan", "hero_official"],
}
ablations = [
    "hero_gnn",
    "wo_risk_relevant_heterophily",
    "wo_mechanism_annotation",
    "wo_evidence_chain",
    "wo_llm_annotation",
    "wo_heterophily_filter",
    "wo_dual_branch_encoder",
    "wo_gated_fusion",
]
expected = []
if quick:
    for model in ["mlp", "gcn", "hero_gnn"]:
        expected.append({"suite": "main", "dataset": "yelp_academic", "model": model, "seed": 0})
else:
    for dataset in datasets:
        for model in main_matrix.get(dataset, []):
            for seed in seeds:
                expected.append({"suite": "main", "dataset": dataset, "model": model, "seed": seed})
    text = [dataset for dataset in datasets if dataset in {"yelp_academic", "amazon_video"}]
    for dataset in text:
        for model in ablations:
            for seed in seeds:
                expected.append({"suite": "ablation", "dataset": dataset, "model": model, "seed": seed})
        for suite in ["robustness", "labeler_comparison", "faithfulness"]:
            for seed in seeds:
                expected.append({"suite": suite, "dataset": dataset, "model": "hero_gnn", "seed": seed})
payload = {
    "suite": "submission_full",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "datasets": datasets,
    "seeds": seeds,
    "quick_test": quick,
    "expected_runs": expected,
}
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

materialize_failures() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
failed_csv = root / "summary" / "failed_runs.csv"
failed_csv.parent.mkdir(parents=True, exist_ok=True)
existing = failed_csv.exists()
with failed_csv.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["suite", "dataset", "model", "seed", "exit_code", "command", "log_file", "error_log"])
    if not existing:
        writer.writeheader()
    for path in sorted((root / "raw").rglob("metrics.json")):
        if any(part.startswith("_project_runs") for part in path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        status = str(payload.get("status", "ok"))
        reason = str(payload.get("skip_reason", payload.get("reason", "")))
        failed = status in {"failed", "missing", "skipped"} or "experiment_failed" in reason
        if not failed:
            continue
        run_dir = path.parent
        error_log = run_dir / "error.log"
        if not error_log.exists():
            parts = [f"status={status}", f"reason={reason}", f"metrics={path}"]
            for name in ["log.txt", "run.log"]:
                log = run_dir / name
                if log.exists():
                    parts.append(log.read_text(encoding="utf-8", errors="ignore"))
            error_log.write_text("\n".join(parts) + "\n", encoding="utf-8")
        writer.writerow({
            "suite": payload.get("suite", ""),
            "dataset": payload.get("dataset", ""),
            "model": payload.get("model", payload.get("variant", "")),
            "seed": payload.get("seed", ""),
            "exit_code": "",
            "command": "",
            "log_file": str(run_dir / "log.txt") if (run_dir / "log.txt").exists() else "",
            "error_log": str(error_log),
        })
PY
}

ensure_required_tables() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = root / "summary"
summary.mkdir(parents=True, exist_ok=True)
required = [
    "all_raw_runs.csv",
    "table_main_mean_std.csv",
    "table_transfer_mean_std.csv",
    "table_ablation_mean_std.csv",
    "table_significance.csv",
    "table_llm_robustness.csv",
    "table_llm_labeler_comparison.csv",
    "table_faithfulness.csv",
    "table_cost_scalability.csv",
    "table_missing_runs.csv",
]
for name in required:
    path = summary / name
    if not path.exists():
        path.write_text("status,reason\nmissing,source_table_not_generated\n", encoding="utf-8")
PY
}

postprocess() {
  write_expected_config
  run_step "check_completeness" "postprocess" "all" "all" "all" \
    "$PYTHON_BIN" scripts/check_experiment_completeness.py --output_dir "$OUTPUT_DIR" --generate_commands
  run_step "summarize_suite" "postprocess" "all" "all" "all" \
    "$PYTHON_BIN" scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_DIR"
  run_step "significance_tests" "postprocess" "all" "all" "all" \
    "$PYTHON_BIN" scripts/run_significance_tests.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR/summary"
  run_step "cost_scalability" "postprocess" "all" "hero" "all" \
    "$PYTHON_BIN" scripts/collect_cost_scalability.py --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" --datasets "${DATASETS[@]}"
  run_step "summarize_suite_final" "postprocess" "all" "all" "all" \
    "$PYTHON_BIN" scripts/summarize_experiment_suite.py --output_dir "$OUTPUT_DIR"
  materialize_failures
  ensure_required_tables
}

echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max parallel requested: $MAX_PARALLEL"

if [[ "$SUMMARIZE_ONLY" -eq 1 ]]; then
  postprocess
  echo "Summary-only finished: $OUTPUT_DIR/summary"
  exit 0
fi

write_expected_config
SKIP_ARGS=()
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  SKIP_ARGS=(--skip_existing)
fi

if [[ "$QUICK_TEST" -eq 1 ]]; then
  run_step "main_quick" "main" "yelp_academic" "mlp_gcn_hero" "0" \
    "$PYTHON_BIN" scripts/run_experiment_suite.py --suite main --datasets yelp_academic --models mlp gcn hero --seeds 0 --output_dir "$OUTPUT_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
  postprocess
  echo "Quick test finished: $OUTPUT_DIR"
  exit 0
fi

TEXT_DATASETS=($(text_datasets))

run_step "main_all" "main" "all" "all" "all" \
  "$PYTHON_BIN" scripts/run_experiment_suite.py --suite main --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"

if [[ "${#TEXT_DATASETS[@]}" -gt 0 ]]; then
  run_step "ablation_all" "ablation" "text_rich" "hero_ablations" "all" \
    "$PYTHON_BIN" scripts/run_experiment_suite.py --suite ablation --datasets "${TEXT_DATASETS[@]}" --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
  run_step "llm_robustness" "robustness" "text_rich" "hero_gnn" "all" \
    "$PYTHON_BIN" scripts/run_experiment_suite.py --suite robustness --datasets "${TEXT_DATASETS[@]}" --seeds "${SEEDS[@]}" --noise_types relevance_flip mechanism_shuffle confidence_gaussian --noise_ratios 0.1 0.2 0.3 0.4 --output_dir "$OUTPUT_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
  run_step "labeler_comparison" "labeler_comparison" "text_rich" "hero_gnn" "all" \
    "$PYTHON_BIN" scripts/run_experiment_suite.py --suite labeler_comparison --datasets "${TEXT_DATASETS[@]}" --seeds "${SEEDS[@]}" --output_dir "$OUTPUT_DIR" --device "$DEVICE" "${SKIP_ARGS[@]}"
  run_step "faithfulness" "faithfulness" "text_rich" "hero_gnn" "all" \
    "$PYTHON_BIN" scripts/run_experiment_suite.py --suite faithfulness --datasets "${TEXT_DATASETS[@]}" --seeds "${SEEDS[@]}" --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" --device "$DEVICE"
fi

run_step "cost_collection" "cost" "all" "hero" "all" \
  "$PYTHON_BIN" scripts/run_experiment_suite.py --suite cost --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" --input_dir "$OUTPUT_DIR" --output_dir "$OUTPUT_DIR" --device "$DEVICE"

postprocess
echo "Submission experiment run finished: $OUTPUT_DIR"
echo "Final tables: $OUTPUT_DIR/summary"
