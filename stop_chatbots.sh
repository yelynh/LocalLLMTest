#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM"
GEMMA_ROOT="$ROOT/gemma4/12B"
QWEN_ROOT="$ROOT/qwen3.6/A3B"
CF_ROOT="/home/iris/workspace/MedicalAOS/frontend"
CF_CONFIG="$CF_ROOT/cloudflared-medicalaos.yml"
CF_PID="$CF_ROOT/cloudflared_named.pid"

echo "[1/3] Stopping Gemma4 12B webchat"
"$GEMMA_ROOT/stop_webchat.sh" || true

echo "[2/3] Stopping Qwen3.6 webchat"
"$QWEN_ROOT/stop_webchat.sh" || true

echo "[3/3] Stopping Cloudflare tunnel"
if [[ -f "$CF_PID" ]] && kill -0 "$(cat "$CF_PID")" 2>/dev/null; then
  kill "$(cat "$CF_PID")" 2>/dev/null || true
fi
pkill -f "cloudflared tunnel --config $CF_CONFIG run medicalaos-frontend" 2>/dev/null || true
rm -f "$CF_PID"

echo
echo "All chatbot services stopped."
