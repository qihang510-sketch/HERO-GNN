#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage0_check_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[time] $(date -Iseconds)"
echo "[pwd] $(pwd)"
python - <<'PY'
import importlib, platform
print("python", platform.python_version())
for name in ["torch", "dgl", "torch_geometric", "numpy", "pandas"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "unknown"))
        if name == "torch":
            print("cuda_available", mod.cuda.is_available(), "cuda", getattr(mod.version, "cuda", None))
    except Exception as exc:
        print(name, "unavailable", type(exc).__name__, exc)
PY
python scripts/check_submission_readiness.py --output outputs/submission_readiness_report.md || true
echo "[stage0] done"
