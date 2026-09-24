from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("QWEN35_MODEL_ID", "Qwen/Qwen3.5-9B")
DEFAULT_MODEL_PATH = (
    "/home/iris/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are MedicalAOS Clinical Harness critic. In this system, modalities mean "
    "EHR, CXR, lab, vitals, diagnosis, CEKG, and AI Patient KG evidence. "
    "CEKG means Clinical Evidence Knowledge Graph, not electrocardiography/ECG. "
    "AI Patient KG means patient-specific clinical knowledge graph. "
    "Answer with clinically grounded uncertainty and do not expose hidden reasoning."
)
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN35_MAX_NEW_TOKENS", "192"))
MAX_NEW_TOKENS_CAP = int(os.getenv("QWEN35_MAX_NEW_TOKENS_CAP", "512"))


class ChatRequest(BaseModel):
    message: str
    system: str | None = None
    thinking: bool = False
    max_new_tokens: int | None = None
    temperature: float = 0.2
    top_p: float = 0.9


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    thinking: bool = False


app = FastAPI(title="Qwen3.5 9B MedicalAOS Chat")
logger = logging.getLogger("qwen35_9b_webchat")
_state: dict[str, Any] = {}
_lock = threading.Lock()


def _resolve_model_path() -> str:
    return os.getenv("QWEN35_MODEL_PATH") or DEFAULT_MODEL_PATH


def _text_from_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content

    chunks: list[str] = []
    for item in content:
        if item.get("type") == "text":
            chunks.append(str(item.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _build_prompt(messages: list[dict[str, str]], thinking: bool) -> tuple[dict[str, torch.Tensor], int]:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return inputs, inputs["input_ids"].shape[-1]


def _clean_reply(reply: str) -> str:
    reply = reply.strip()
    for marker in ("Thinking Process:", "<think>", "</think>"):
        reply = reply.replace(marker, "")
    return reply.strip()


def load_model() -> None:
    model_path = _resolve_model_path()
    logger.info("loading model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=os.path.exists(model_path),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=os.path.exists(model_path),
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    _state["model_path"] = model_path
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    logger.info("model_ready: %s", model_path)


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_model()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "model_path": _state.get("model_path"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda": torch.version.cuda,
        "gpu_peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        if torch.cuda.is_available()
        else None,
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "MedicalAOS-local",
                "root": _state.get("model_path"),
            }
        ],
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    with _lock:
        started_at = time.perf_counter()
        messages = [
            {"role": "system", "content": req.system or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ]
        inputs, input_len = _build_prompt(messages, req.thinking)
        model = _state["model"]
        tokenizer = _state["tokenizer"]
        max_new_tokens = min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
            top_p=req.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = _clean_reply(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))
        logger.info(
            "chat_complete prompt_chars=%s reply_chars=%s duration_sec=%.2f peak_mem_gb=%.2f",
            len(req.message),
            len(reply),
            time.perf_counter() - started_at,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        return {"reply": reply, "model": MODEL_ID}


@app.post("/v1/chat/completions")
def openai_chat(req: OpenAIChatRequest):
    with _lock:
        started_at = time.perf_counter()
        normalized: list[dict[str, str]] = []
        has_system = False
        for message in req.messages:
            content = _text_from_content(message.content)
            if not content:
                continue
            if message.role == "system":
                has_system = True
            normalized.append({"role": message.role, "content": content})
        if not has_system:
            normalized.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        inputs, input_len = _build_prompt(normalized, req.thinking)
        model = _state["model"]
        tokenizer = _state["tokenizer"]
        max_new_tokens = min(req.max_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
            top_p=req.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = _clean_reply(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))
        logger.info(
            "openai_chat_complete messages=%s reply_chars=%s duration_sec=%.2f peak_mem_gb=%.2f",
            len(normalized),
            len(reply),
            time.perf_counter() - started_at,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": int(outputs.shape[-1] - input_len),
                "total_tokens": int(outputs.shape[-1]),
            },
        }
