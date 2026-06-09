#!/usr/bin/env bash
set -e
mkdir -p outputs/logs_submission
LOG_FILE="outputs/logs_submission/stage1_prepare_data_$(date +%Y%m%d_%H%M%S).log"
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
python scripts/prepare_elliptic.py
if [ ! -f data/processed/yelp_academic/features.npz ]; then
  python scripts/preprocess_yelp_academic.py || echo "[warning] Yelp preprocess skipped or failed."
else
  echo "[stage1] keeping existing data/processed/yelp_academic"
fi
if [ ! -f data/processed/amazon_video/features.npz ]; then
  python scripts/preprocess_amazon_video.py || echo "[warning] Amazon preprocess skipped or failed."
else
  echo "[stage1] keeping existing data/processed/amazon_video"
fi
if [ ! -f data/processed/fraud_yelp_official/features.npz ]; then
  python scripts/preprocess_dgl_fraud.py --dataset fraud_yelp_official --out_dir data/processed/fraud_yelp_official || echo "[warning] FraudYelp DGL preprocess skipped or failed."
else
  echo "[stage1] keeping existing data/processed/fraud_yelp_official"
fi
if [ ! -f data/processed/fraud_amazon_official/features.npz ]; then
  python scripts/preprocess_dgl_fraud.py --dataset fraud_amazon_official --out_dir data/processed/fraud_amazon_official || echo "[warning] FraudAmazon DGL preprocess skipped or failed."
else
  echo "[stage1] keeping existing data/processed/fraud_amazon_official"
fi
echo "[stage1] done"
