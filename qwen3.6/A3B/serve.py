"""
Qwen3.6-35B-A3B — vLLM OpenAI-compatible 서버 런처

Usage:
    python serve.py
    python serve.py --quant bnb --ctx 32768 --gpu-util 0.92
    python serve.py --tensor-parallel 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from runtime import MODEL_ID, resolve_model_name_or_path

DEFAULT_PORT = 8010
DEFAULT_CTX = 32768


def build_cmd(args: argparse.Namespace) -> list[str]:
    model_name_or_path = resolve_model_name_or_path()
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name_or_path,
        "--served-model-name",
        MODEL_ID,
        "--max-model-len",
        str(args.ctx),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_util),
        "--dtype",
        "bfloat16",
    ]
    if args.quant == "bnb":
        cmd += ["--quantization", "bitsandbytes"]
    elif args.quant:
        cmd += ["--quantization", args.quant]
    if args.tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B vLLM 서버")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    parser.add_argument("--quant", choices=["bnb", "awq", "gptq"], default="bnb")
    parser.add_argument("--gpu-util", type=float, default=0.92, dest="gpu_util")
    parser.add_argument("--tensor-parallel", type=int, default=2, dest="tensor_parallel")
    args = parser.parse_args()

    model_name_or_path = resolve_model_name_or_path()
    cmd = build_cmd(args)
    print(f"[serve] 모델: {MODEL_ID}")
    print(f"[serve] 로드 소스: {model_name_or_path}")
    print(f"[serve] 주소: http://{args.host}:{args.port}/v1")
    print(f"[serve] 컨텍스트: {args.ctx} tokens")
    print(f"[serve] 양자화: {args.quant or 'none'}")
    print(f"[serve] TP: {args.tensor_parallel}")
    print(f"[serve] 실행: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
