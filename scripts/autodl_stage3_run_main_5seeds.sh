#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage3_run_main_5seeds_$(date +%Y%m%d_%H%M%S).log"
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
python scripts/run_submission_experiments.py --datasets yelp_academic amazon_video fraud_yelp fraud_amazon elliptic --seeds 0 1 2 3 4 --output_dir outputs/submission_experiments
echo "[stage3] done"
