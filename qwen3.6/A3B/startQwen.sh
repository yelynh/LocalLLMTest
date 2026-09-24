#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/jhkim3/miniforge3}"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
ENV_NAME="${QWEN36_ENV_NAME:-qwen36}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[오류] conda 초기화 스크립트를 찾지 못했습니다: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${ENV_NAME}"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./startQwen.sh <mode> [args...]

Modes:
  local-chat
  local-infer "<prompt>"
  server
  server-chat
  server-infer "<prompt>"

Environment overrides:
  CONDA_BASE=/home/jhkim3/miniforge3
  QWEN36_ENV_NAME=qwen36
  QWEN36_SERVER_URL=http://localhost:8010/v1
  QWEN36_MODEL=/path/to/model-or-hf-id
EOF
}

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  help|-h|--help)
    usage
    ;;
  local-chat)
    exec python chat.py --mode local --quant bnb "$@"
    ;;
  local-infer)
    if [[ $# -lt 1 ]]; then
      echo "[오류] local-infer 모드는 프롬프트가 필요합니다." >&2
      usage
      exit 1
    fi
    exec python inference.py --mode local --quant bnb --prompt "$1" "${@:2}"
    ;;
  server)
    exec python serve.py "$@"
    ;;
  server-chat)
    exec python chat.py --mode server "$@"
    ;;
  server-infer)
    if [[ $# -lt 1 ]]; then
      echo "[오류] server-infer 모드는 프롬프트가 필요합니다." >&2
      usage
      exit 1
    fi
    exec python inference.py --mode server --prompt "$1" "${@:2}"
    ;;
  *)
    echo "[오류] 알 수 없는 모드: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
