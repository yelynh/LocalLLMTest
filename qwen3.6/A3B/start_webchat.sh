#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/qwen3.6/A3B"
ENV_NAME="${LOCAL_LLM_ENV_NAME:-localllm-chat}"
PORT="${1:-8732}"
GPU="${QWEN36_GPU:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/home/iris/workspace/key/hf.key}"
LOG_DIR="$ROOT/logs"
APP_PID_FILE="$ROOT/webchat.pid"
GPU_PID_FILE="$ROOT/gpu_watch.pid"

mkdir -p "$LOG_DIR"
rm -f "$APP_PID_FILE" "$GPU_PID_FILE"

HF_TOKEN_VALUE="$(cat "$HF_TOKEN_FILE")"

nohup env HF_TOKEN="$HF_TOKEN_VALUE" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  /home/iris/.local/bin/micromamba run -n "$ENV_NAME" bash -lc \
  "cd '$ROOT' && exec uvicorn webchat_app:app --host 127.0.0.1 --port '$PORT'" \
  >> "$LOG_DIR/webchat.log" 2>&1 </dev/null &
echo $! > "$APP_PID_FILE"

nohup bash -lc "while true; do date '+[%F %T]' >> '$LOG_DIR/gpu_usage.log'; /usr/bin/nvidia-smi -i '$GPU' --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader >> '$LOG_DIR/gpu_usage.log'; sleep 5; done" >/dev/null 2>&1 &
echo $! > "$GPU_PID_FILE"

echo "Qwen3.6 webchat starting on http://127.0.0.1:$PORT (GPU $GPU)"
