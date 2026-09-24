#!/usr/bin/env bash
set -euo pipefail
export HF_TOKEN="$(tr -d '\n\r' < /home/iris/workspace/key/hf.key)"
exec /home/iris/.local/bin/micromamba run -n localllm-chat hf download google/gemma-4-12B-it \
  --include '*.json' \
  --include '*.safetensors' \
  --include '*.model' \
  --include '*.txt' \
  --include '*.py' \
  --max-workers 4 \
  --token "$HF_TOKEN"
