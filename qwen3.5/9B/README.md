# Qwen3.5-9B MedicalAOS Runtime

Single RTX 4090 runtime for MedicalAOS Clinical Harness critic / SCC routing / KG-grounded review.

## Start

```bash
/home/iris/workspace/localLLM/qwen3.5/9B/start_webchat.sh 8732
```

## Endpoints

- `GET /health`
- `POST /api/chat`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Runtime Note

`vLLM 0.20.1` currently requires a CUDA 13 runtime path and fails during generation on this driver stack because flash-attn reports `CUDA driver version is insufficient for CUDA runtime version`.
This server intentionally uses Transformers + BF16 + SDPA as the stable fallback.
