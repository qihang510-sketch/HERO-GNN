#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage7_readiness_check_$(date +%Y%m%d_%H%M%S).log"
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
python scripts/run_significance_tests.py --input_dir outputs/submission_experiments --output_dir outputs/paper_tables_submission
python scripts/check_submission_readiness.py --results_dir outputs/submission_experiments --tables_dir outputs/paper_tables_submission --figures_dir outputs/paper_figures_submission --output outputs/submission_readiness_report.md
echo "[stage7] done"
