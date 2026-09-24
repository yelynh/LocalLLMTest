#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/ollama"
BIN="$ROOT/root/bin/ollama"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
MODELS="${OLLAMA_MODELS:-$ROOT/models}"
LOG_DIR="$ROOT/logs"
PID_FILE="$ROOT/ollama.pid"

mkdir -p "$LOG_DIR" "$MODELS"
rm -f "$PID_FILE"

setsid -f env \
  OLLAMA_HOST="$HOST" \
  OLLAMA_MODELS="$MODELS" \
  OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cpu}" \
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}" \
  OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}" \
  "$BIN" serve >> "$LOG_DIR/ollama.log" 2>&1 </dev/null

sleep 1
pgrep -f "(/home/iris/.local/bin/ollama|$BIN) serve" | head -n 1 > "$PID_FILE" || true

echo "Ollama serving on http://$HOST"
