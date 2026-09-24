#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM"
GEMMA_ROOT="$ROOT/gemma4/12B"
QWEN_ROOT="$ROOT/qwen3.6/A3B"
CF_ROOT="/home/iris/workspace/MedicalAOS/frontend"
CF_CONFIG="$CF_ROOT/cloudflared-medicalaos.yml"
CF_LOG="$CF_ROOT/logs/cloudflared_named.log"
CF_PID="$CF_ROOT/cloudflared_named.pid"

GEMMA_PORT="${GEMMA_PORT:-8731}"
QWEN_PORT="${QWEN_PORT:-8732}"
START_TUNNEL="${START_TUNNEL:-1}"

mkdir -p "$CF_ROOT/logs"

echo "[1/3] Starting Gemma4 12B webchat on port $GEMMA_PORT"
"$GEMMA_ROOT/start_webchat.sh" "$GEMMA_PORT"

echo "[2/3] Starting Qwen3.6 webchat on port $QWEN_PORT"
"$QWEN_ROOT/start_webchat.sh" "$QWEN_PORT"

if [[ "$START_TUNNEL" == "1" ]]; then
  echo "[3/3] Starting Cloudflare tunnel"
  pkill -f "cloudflared tunnel --config $CF_CONFIG run medicalaos-frontend" 2>/dev/null || true
  nohup /bin/bash -lc "exec cloudflared tunnel --config '$CF_CONFIG' run medicalaos-frontend" \
    > "$CF_LOG" 2>&1 < /dev/null &
  echo $! > "$CF_PID"
  sleep 3
fi

echo
echo "Gemma4 12B   : http://127.0.0.1:$GEMMA_PORT"
echo "Qwen local   : http://127.0.0.1:$QWEN_PORT"
echo "Tunnel pid   : $(cat "$CF_PID" 2>/dev/null || echo not-started)"
echo "Gemma log    : $GEMMA_ROOT/logs/webchat.log"
echo "Qwen log     : $QWEN_ROOT/logs/webchat.log"
echo "Tunnel log   : $CF_LOG"
