#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/ollama"
BIN="$ROOT/root/bin/ollama"
PID_FILE="$ROOT/ollama.pid"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

pkill -f "$BIN serve" 2>/dev/null || true
pkill -f "/home/iris/.local/bin/ollama serve" 2>/dev/null || true

echo "Ollama stopped"
