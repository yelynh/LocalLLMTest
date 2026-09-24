#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/iris/workspace/localLLM/gemma4/E2B"
for f in "$ROOT/webchat.pid" "$ROOT/gpu_watch.pid"; do
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    kill "$(cat "$f")"
  fi
  rm -f "$f"
done
pkill -f "uvicorn webchat_app:app --host 127.0.0.1 --port 8731" 2>/dev/null || true
echo "Gemma4 webchat stopped"
