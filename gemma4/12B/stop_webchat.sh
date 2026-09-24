#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/gemma4/12B"
PORT="${1:-8731}"
PID_FILE="$ROOT/webchat_${PORT}.pid"
LEGACY_PID_FILE="$ROOT/webchat.pid"

if [[ -f "$PID_FILE" ]]; then
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  rm -f "$PID_FILE"
fi
if [[ "$PORT" == "8731" && -f "$LEGACY_PID_FILE" ]]; then
  kill "$(cat "$LEGACY_PID_FILE")" 2>/dev/null || true
  rm -f "$LEGACY_PID_FILE"
fi

pkill -f "uvicorn --app-dir $ROOT webchat_app:app --host 127.0.0.1 --port $PORT" 2>/dev/null || true
pkill -f "$ROOT.*webchat_app:app.*--port $PORT" 2>/dev/null || true
echo "Gemma4 12B webchat stopped on port $PORT"
