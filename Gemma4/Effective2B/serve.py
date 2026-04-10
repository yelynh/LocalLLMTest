"""
Gemma 4 E2B — vLLM OpenAI-compatible 서버 런처
Usage:
    python serve.py                   # 기본 (bfloat16, 32K ctx) — VRAM 12GB+
    python serve.py --quant bnb       # bitsandbytes 4-bit — VRAM 8GB (RTX 4060)
    python serve.py --quant bnb --text-only --ctx 1024 --gpu-util 0.95
    python serve.py --quant awq       # AWQ 4-bit (사전양자화 모델 필요)
    python serve.py --ctx 16384       # 컨텍스트 줄이기
    python serve.py --tools           # 함수 호출 활성화
"""

import argparse
import subprocess
import sys

from runtime import MODEL_ID, resolve_model_name_or_path

DEFAULT_PORT = 8000
DEFAULT_CTX = 32768


def build_cmd(args: argparse.Namespace) -> list[str]:
    model_name_or_path = resolve_model_name_or_path()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name_or_path,
        "--served-model-name", MODEL_ID,
        "--max-model-len", str(args.ctx),
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_util),
        "--dtype", "bfloat16",
        "--reasoning-parser", "gemma4",
    ]
    if args.quant == "bnb":
        # vLLM 0.19 기준 bitsandbytes 4-bit 로더 사용
        cmd += ["--quantization", "bitsandbytes"]
    elif args.quant:
        cmd += ["--quantization", args.quant]
    if args.tools:
        cmd += ["--tool-call-parser", "gemma4", "--enable-auto-tool-choice"]
    if args.tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel)]
    if args.text_only:
        cmd += ["--language-model-only"]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B vLLM 서버")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX, help="max-model-len (컨텍스트 길이)")
    parser.add_argument("--quant", choices=["bnb", "awq", "gptq"], default=None,
                        help="양자화: bnb=bitsandbytes 4bit(RTX 4060 권장), awq/gptq=사전양자화 모델 필요")
    parser.add_argument("--tools", action="store_true", help="함수 호출(Tool Calling) 활성화")
    parser.add_argument("--text-only", action="store_true",
                        help="텍스트 전용 모드 (멀티모달 encoder 비활성화, 8GB VRAM 테스트용 권장)")
    parser.add_argument("--gpu-util", type=float, default=0.85, dest="gpu_util", help="GPU 메모리 사용률 (0~1)")
    parser.add_argument("--tensor-parallel", type=int, default=1, dest="tensor_parallel", help="GPU 수 (멀티GPU)")
    args = parser.parse_args()

    # RTX 4060 8GB 경고
    if args.quant is None:
        print("[경고] VRAM 8GB 환경에서는 --quant bnb 옵션을 권장합니다.")

    model_name_or_path = resolve_model_name_or_path()
    cmd = build_cmd(args)
    print(f"[serve] 모델: {MODEL_ID}")
    print(f"[serve] 로드 소스: {model_name_or_path}")
    print(f"[serve] 주소: http://localhost:{args.port}/v1")
    print(f"[serve] 컨텍스트: {args.ctx} tokens")
    print(f"[serve] 양자화: {args.quant or 'none (bfloat16)'}")
    if args.text_only:
        print("[serve] 멀티모달 encoder: 비활성화 (text-only)")
    if args.tools:
        print("[serve] Tool Calling: 활성화")
    print(f"[serve] 실행: {' '.join(cmd)}\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
