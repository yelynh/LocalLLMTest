#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/jhkim3/miniforge3}"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
ENV_NAME="${GEMMA4_ENV_NAME:-gemma4}"

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
  ./startGemma.sh <mode> [args...]

Modes:
  local-chat
      transformers + bitsandbytes 4bit 인터랙티브 챗

  local-infer "<prompt>"
      transformers + bitsandbytes 4bit 단발 추론

  server
      vLLM 서버 실행
      예: ./startGemma.sh server --quant bnb --text-only --ctx 1024 --gpu-util 0.86

  server-chat
      떠 있는 vLLM 서버에 chat.py로 접속

  server-infer "<prompt>"
      떠 있는 vLLM 서버에 inference.py로 단발 요청

Environment overrides:
  CONDA_BASE=/home/jhkim3/miniforge3
  GEMMA4_ENV_NAME=gemma4
  GEMMA4_SERVER_URL=http://localhost:8000/v1
  GEMMA4_MODEL=/path/to/model-or-hf-id

Examples:
  ./startGemma.sh local-chat
  ./startGemma.sh local-infer "안녕? 한 줄로 자기소개해줘."
  ./startGemma.sh server --quant bnb --text-only --ctx 1024 --gpu-util 0.86
  ./startGemma.sh server-chat
  ./startGemma.sh server-infer "테스트 프롬프트"
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
