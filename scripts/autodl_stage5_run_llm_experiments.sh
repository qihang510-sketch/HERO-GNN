#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage5_run_llm_experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[time] $(date -Iseconds)"
echo "[pwd] $(pwd)"
python - <<'PY'
import importlib, platform
print("python", platform.python_version())
for name in ["torch", "dgl", "torch_geometric"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "unavailable", type(exc).__name__, exc)
PY
python scripts/run_labeler_comparison.py --dataset yelp_academic --labelers rule qwen --seeds 0 1 2 3 4 --use_existing_annotations true --output_dir outputs/submission_llm_labeler
python scripts/run_llm_coverage_sensitivity.py --dataset yelp_academic --coverages 0 0.25 0.5 0.75 1.0 --seeds 0 1 2 --use_existing_qwen_annotations true --output_dir outputs/submission_llm_coverage
echo "[stage5] done"
