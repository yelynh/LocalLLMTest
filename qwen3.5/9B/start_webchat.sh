#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/qwen3.5/9B"
ENV_NAME="${LOCAL_LLM_ENV_NAME:-localllm-chat}"
PORT="${1:-8732}"
GPU="${QWEN35_GPU:-1}"
ENV_PREFIX="/home/iris/.local/share/mamba/envs/$ENV_NAME"
LOG_DIR="$ROOT/logs"
APP_PID_FILE="$ROOT/webchat.pid"
GPU_PID_FILE="$ROOT/gpu_watch.pid"

mkdir -p "$LOG_DIR"
rm -f "$APP_PID_FILE" "$GPU_PID_FILE"

cd "$ROOT"

setsid -f env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  "$ENV_PREFIX/bin/python" -m uvicorn webchat_app:app --app-dir "$ROOT" --host 127.0.0.1 --port "$PORT" \
  >> "$LOG_DIR/webchat.log" 2>&1 </dev/null

sleep 1
pgrep -f "webchat_app:app --app-dir $ROOT .*--port $PORT" | head -n 1 > "$APP_PID_FILE" || true

setsid -f bash -lc "while true; do date '+[%F %T]' >> '$LOG_DIR/gpu_usage.log'; /usr/bin/nvidia-smi -i '$GPU' --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader >> '$LOG_DIR/gpu_usage.log'; sleep 5; done" >/dev/null 2>&1
pgrep -f "$LOG_DIR/gpu_usage.log.*nvidia-smi -i '$GPU'" | head -n 1 > "$GPU_PID_FILE" || true

echo "Qwen3.5-9B webchat starting on http://127.0.0.1:$PORT (GPU $GPU)"
