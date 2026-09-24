#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/iris/workspace/localLLM/qwen3.6/A3B"
for f in "$ROOT/webchat.pid" "$ROOT/gpu_watch.pid"; do
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    kill "$(cat "$f")"
  fi
  rm -f "$f"
done
pkill -f "uvicorn webchat_app:app --host 127.0.0.1 --port 8732" 2>/dev/null || true
echo "Qwen3.6 webchat stopped"
