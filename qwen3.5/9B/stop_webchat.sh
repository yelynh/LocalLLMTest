#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/qwen3.5/9B"

for pid_file in "$ROOT/webchat.pid" "$ROOT/gpu_watch.pid"; do
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
done

echo "Qwen3.5-9B webchat stopped"
