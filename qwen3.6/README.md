# Qwen3.6 Local Workspace

이 디렉토리는 `MedicalAOS` 및 관련 로컬 LLM 실험을 위해 준비한
`Qwen3.6` 공식 레포 보관 위치다.

## Included

- 공식 GitHub 레포:
  - `QwenLM/Qwen3.6`
- 로컬 clone 경로:
  - `/home/iris/workspace/localLLM/qwen3.6/repo`

## Current State

현재는 두 층으로 정리되어 있다.

- 공식 소스 레포:
  - `/home/iris/workspace/localLLM/qwen3.6/repo`
- 로컬 agent wrapper:
  - `/home/iris/workspace/localLLM/qwen3.6/A3B`

## Useful References

공식 레포 README 기준 핵심 모델 예시:

- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-122B-A10B`

## Next Step Candidates

현재는 `Qwen3.6-35B-A3B` 기준으로
`bitsandbytes 4bit` + `OpenAI-compatible server` + `agent_wrapper.py`
형태까지 맞춰둔 상태다.

## Quick Check

```bash
git -C /home/iris/workspace/localLLM/qwen3.6/repo remote -v
git -C /home/iris/workspace/localLLM/qwen3.6/repo branch -a -vv
git -C /home/iris/workspace/localLLM/qwen3.6/repo log --oneline -n 5
```
