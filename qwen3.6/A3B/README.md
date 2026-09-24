# Qwen3.6-35B-A3B Agent Wrapper

이 디렉토리는 `Qwen3.6-35B-A3B`를
`agent backend`처럼 바로 붙일 수 있도록 정리한 로컬 실행 뼈대다.

핵심 방향:

- 튜닝 없음
- PEFT 학습 없음
- `bitsandbytes 4bit` 기반 추론/서빙 우선
- OpenAI-compatible endpoint 우선
- agent workflow에서 쓰기 쉬운 `agent_wrapper.py` 제공

## Files

- `runtime.py`
- `serve.py`
- `inference.py`
- `chat.py`
- `agent_wrapper.py`
- `startQwen.sh`
- `requirements.txt`
- `environment.yml`

## Recommended Mode

현재 머신이 `RTX 4090 x2` 이므로
당장은 `bnb 4bit + tensor parallel 2` 조합이 가장 현실적이다.

예시:

```bash
cd /home/iris/workspace/localLLM/qwen3.6/A3B
./startQwen.sh server --quant bnb --tensor-parallel 2 --ctx 32768
```

## Why No PEFT Now

지금은

- 튜닝 데이터가 없음
- 연구 초점이 model fine-tuning보다 orchestration 쪽임
- 35B 계열은 추론/서빙 파이프라인 안정화가 먼저임

이유로, QLoRA/LoRA 파인튜닝은 일부러 넣지 않았다.

필요해지면 그때 `adapter-only` 학습 스크립트를 별도로 추가하는 편이 낫다.
