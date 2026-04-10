# Gemma 4 Effective 2B — 작업 진행 상황

## 환경 정보
| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (VRAM 8GB) |
| CUDA | 12.9 (Driver) / 12.0 (Toolkit) |
| OS | Linux (WSL2) |
| conda 경로 | `/home/jhkim3/miniforge3/envs/gemma4` |

---

## 완료된 작업

### 1. conda 환경 생성
- 환경명: `gemma4`
- Python: 3.11
- 경로: `/home/jhkim3/miniforge3/envs/gemma4`

### 2. 패키지 설치
| 패키지 | 버전 | 용도 |
|---|---|---|
| torch | 2.10.0+cu128 | 딥러닝 백엔드 |
| vllm | 0.19.0 | OpenAI-compatible 서버 서빙 |
| transformers | 4.57.6 | 모델 로드 (로컬 모드) |
| bitsandbytes | 0.49.2 | 4-bit 양자화 (8GB VRAM 대응) |
| accelerate | 1.13.0 | 멀티GPU / 디바이스 맵 |
| librosa | 0.11.0 | 오디오 입력 처리 |
| Pillow | - | 이미지 입력 처리 |
| openai | 2.30.0 | vLLM 서버 클라이언트 |

### 3. 코드 파일 작성
| 파일 | 설명 |
|---|---|
| `serve.py` | vLLM OpenAI-compatible 서버 런처 |
| `inference.py` | 추론 클라이언트 (server / local 모드) |
| `requirements.txt` | 패키지 목록 |

### 4. SKILL.md 등록
- 경로: `~/.claude/skills/gemma4-effective2b/SKILL.md`
- Claude Code에서 `/gemma4-effective2b` 슬래시 커맨드로 호출 가능

---

## 미완료 / 남은 작업

- [ ] HuggingFace에서 `google/gemma-4-E2B-it` 모델 사전 다운로드
- [ ] serve.py 실제 기동 테스트 (bitsandbytes 4-bit 모드)
- [ ] inference.py 엔드투엔드 테스트 (텍스트 / 이미지 / 오디오)
- [ ] VS Code Python 인터프리터를 `gemma4` 환경으로 변경
- [ ] skill_agent.py와 연동 (gemma4-effective2b 스킬 주입 테스트)

---

## 사용법 요약

```bash
# 환경 활성화
conda activate gemma4

# 서버 실행 (RTX 4060 8GB → bnb 4-bit 필수)
python serve.py --quant bnb --ctx 16384

# 함수 호출 포함
python serve.py --quant bnb --ctx 16384 --tools

# 추론 (서버 모드)
python inference.py --prompt "안녕"

# 추론 (인터랙티브)
python inference.py

# 추론 (transformers 직접 로드, 서버 불필요)
python inference.py --mode local --prompt "안녕"
```

---

## 모델 정보
| 항목 | 내용 |
|---|---|
| 모델 ID | `google/gemma-4-E2B-it` |
| 출시일 | 2026-04-02 |
| 파라미터 | 2.3B effective (5.1B with embeddings) |
| 컨텍스트 | 128K tokens |
| 입력 모달리티 | 텍스트, 이미지, 오디오 |
| 라이선스 | Apache 2.0 |
| VRAM (bfloat16) | ~10GB → 8GB 초과, 양자화 필요 |
| VRAM (bnb 4-bit) | ~3GB → RTX 4060 정상 동작 |
