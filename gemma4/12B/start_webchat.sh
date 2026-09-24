#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/gemma4/12B"
ENV_NAME="${LOCAL_LLM_ENV_NAME:-localllm-chat}"
PORT="${1:-8731}"
GPU="${GEMMA4_12B_GPU:-0,1}"
LOG_DIR="$ROOT/logs"
APP_PID_FILE="$ROOT/webchat_${PORT}.pid"
LEGACY_PID_FILE="$ROOT/webchat.pid"
LOCAL_NVIDIA_LIB_DIR="${LOCAL_NVIDIA_LIB_DIR:-/home/iris/workspace/MedicalAOS/runtime_assets/nvidia-535.309/root/usr/lib/x86_64-linux-gnu}"
ENV_BIN="/home/iris/.local/share/mamba/envs/$ENV_NAME/bin"

mkdir -p "$LOG_DIR"
rm -f "$APP_PID_FILE"
if [[ "$PORT" == "8731" ]]; then
  rm -f "$LEGACY_PID_FILE"
fi

if [[ -d "$LOCAL_NVIDIA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$LOCAL_NVIDIA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

setsid bash -lc "
  echo \$\$ > '$APP_PID_FILE'
  if [[ '$PORT' == '8731' ]]; then echo \$\$ > '$LEGACY_PID_FILE'; fi
  cd '$ROOT'
  exec env CUDA_VISIBLE_DEVICES='$GPU' PYTHONUNBUFFERED=1 LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}' \
    GEMMA4_12B_DEVICE_MAP='${GEMMA4_12B_DEVICE_MAP:-auto}' \
    GEMMA4_12B_LOAD_MODE='${GEMMA4_12B_LOAD_MODE:-fp16}' \
    '$ENV_BIN/uvicorn' --app-dir '$ROOT' webchat_app:app --host 127.0.0.1 --port '$PORT'
" >> "$LOG_DIR/webchat.log" 2>&1 < /dev/null &

echo "Gemma4 12B webchat starting on http://127.0.0.1:$PORT (GPU $GPU, load_mode ${GEMMA4_12B_LOAD_MODE:-fp16})"
