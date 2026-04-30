#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="${1:-PSM.csv}"
PRETRAINED_MODEL="${2:-result/checkpoints/CATCH/PSM.pt}"
GPU_ID="${3:-0}"
SAVE_PATH="${4:-label/CATCH_inference_only}"

python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_label_multi_config.json" \
  --data-name-list "${DATA_NAME}" \
  --model-name "catch.CATCH" \
  --model-hyper-params "{\"pretrained_model\": \"${PRETRAINED_MODEL}\"}" \
  --gpus "${GPU_ID}" \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "${SAVE_PATH}"
