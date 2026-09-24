#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/iris/workspace/localLLM/gemma4/12B"
PORT_A="${1:-8731}"
PORT_B="${2:-8735}"

GEMMA4_12B_GPU=0 GEMMA4_12B_DEVICE_MAP=single GEMMA4_12B_LOAD_MODE=int8 "$ROOT/start_webchat.sh" "$PORT_A"
GEMMA4_12B_GPU=1 GEMMA4_12B_DEVICE_MAP=single GEMMA4_12B_LOAD_MODE=int8 "$ROOT/start_webchat.sh" "$PORT_B"
