#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage2_smoke_test_$(date +%Y%m%d_%H%M%S).log"
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
python scripts/run_submission_experiments.py --datasets yelp_academic --models mlp graphsage hero_gnn --seeds 0 --output_dir outputs/smoke_test
python scripts/run_submission_experiments.py --datasets fraud_yelp --models mlp graphsage hero_official --seeds 0 --output_dir outputs/smoke_test
python scripts/run_submission_experiments.py --datasets elliptic --models mlp graphsage hero_official --seeds 0 --output_dir outputs/smoke_test
echo "[stage2] done"
