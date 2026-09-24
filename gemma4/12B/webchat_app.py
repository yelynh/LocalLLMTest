from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


MODEL_ID = "google/gemma-4-12B-it"
DEFAULT_MODEL_PATH = (
    "/home/iris/.cache/huggingface/hub/models--google--gemma-4-12B-it/"
    "snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205"
)
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GEMMA4_12B_MAX_NEW_TOKENS", "2048"))
MAX_NEW_TOKENS_CAP = int(os.getenv("GEMMA4_12B_MAX_NEW_TOKENS_CAP", "2048"))
DEVICE_MAP = os.getenv("GEMMA4_12B_DEVICE_MAP", "auto")
LOAD_MODE = os.getenv("GEMMA4_12B_LOAD_MODE", "fp16").lower()
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "GEMMA4_12B_SYSTEM_PROMPT",
    (
        "You are MedicalAOS clinical harness runtime. Answer with calibrated clinical "
        "uncertainty, preserve source attribution, and avoid unsupported claims."
    ),
)


def _torch_dtype_for_load_mode(load_mode: str) -> torch.dtype:
    if load_mode == "bf16":
        return torch.bfloat16
    if load_mode in {"fp16", "float16", "half"}:
        return torch.float16
    if load_mode in {"int8", "8bit"}:
        return torch.bfloat16
    if load_mode in {"4bit", "nf4"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported GEMMA4_12B_LOAD_MODE={load_mode!r}; use fp16, bf16, int8, or 4bit")


def _summarize_device_map(model: Any) -> dict[str, Any]:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return {"available": False}
    counts: dict[str, int] = {}
    for device in device_map.values():
        key = str(device)
        counts[key] = counts.get(key, 0) + 1
    return {"available": True, "device_counts": counts}


def _checkpoint_hash(model_path: str) -> str:
    candidate = Path(model_path).name
    return candidate if len(candidate) == 40 else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None
    thinking: bool = False
    max_new_tokens: int | None = None


app = FastAPI(title="Gemma4 12B Chat")
logger = logging.getLogger("gemma4_12b_webchat")
_state: dict[str, Any] = {}
_lock = threading.Lock()


def _resolve_model_path() -> str:
    return os.getenv("GEMMA4_12B_MODEL_PATH") or os.getenv("GEMMA4_MODEL_PATH") or DEFAULT_MODEL_PATH


def _format_prompt(message: str, thinking: bool, system_prompt: str | None = None) -> str:
    resolved_system_prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
    return (
        "<bos><|turn>system\n"
        f"{'<|think|>\n' if thinking else ''}"
        f"{resolved_system_prompt}\n"
        "<turn|><|turn>user\n"
        f"{message.strip()}"
        "<turn|><|turn>assistant\n"
    )


def load_model() -> None:
    model_path = _resolve_model_path()
    torch_dtype = _torch_dtype_for_load_mode(LOAD_MODE)
    logger.info("loading model: %s load_mode=%s dtype=%s", model_path, LOAD_MODE, torch_dtype)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model_kwargs = {
        "local_files_only": True,
        "device_map": {"": 0} if DEVICE_MAP == "single" else DEVICE_MAP,
        "dtype": torch_dtype,
    }
    if LOAD_MODE in {"4bit", "nf4"}:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif LOAD_MODE in {"int8", "8bit"}:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(os.getenv("GEMMA4_12B_INT8_THRESHOLD", "6.0")),
        )
    model = AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)
    _state["model_path"] = model_path
    _state["load_mode"] = LOAD_MODE
    _state["torch_dtype"] = str(torch_dtype).replace("torch.", "")
    _state["device_map_summary"] = _summarize_device_map(model)
    _state["processor"] = processor
    _state["model"] = model
    logger.info("model_ready: %s load_mode=%s device_map=%s", model_path, LOAD_MODE, _state["device_map_summary"])


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
        "checkpoint_hash": _checkpoint_hash(str(_state.get("model_path") or "")),
        "process_id": os.getpid(),
        "load_mode": _state.get("load_mode"),
        "torch_dtype": _state.get("torch_dtype"),
        "device_map_summary": _state.get("device_map_summary"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    with _lock:
        started_at_utc = _utc_now()
        started_at = time.perf_counter()
        processor = _state["processor"]
        model = _state["model"]
        prompt = _format_prompt(req.message, req.thinking, req.system_prompt)
        model_device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = processor(text=prompt, return_tensors="pt").to(model_device)
        input_len = inputs["input_ids"].shape[-1]
        cuda_device = model_device if getattr(model_device, "type", "") == "cuda" else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
            do_sample=False,
            eos_token_id=[
                processor.tokenizer.eos_token_id,
                processor.tokenizer.convert_tokens_to_ids("<turn|>"),
            ],
        )
        raw_reply = processor.decode(outputs[0][input_len:], skip_special_tokens=False).strip()
        output_tokens = int(outputs.shape[-1] - input_len)
        try:
            parsed = processor.parse_response(raw_reply)
            reply = (parsed.get("content") or parsed.get("thinking") or raw_reply).strip()
        except Exception:
            reply = (
                raw_reply.replace("<|channel>thought\n", "")
                .replace("<channel|>", "")
                .replace("<turn|>", "")
                .strip()
            )
        logger.info(
            "chat_complete system_prompt_chars=%s message_chars=%s reply_chars=%s max_new_tokens=%s duration_sec=%.2f",
            len(req.system_prompt or DEFAULT_SYSTEM_PROMPT),
            len(req.message),
            len(reply),
            min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
            time.perf_counter() - started_at,
        )
        duration_sec = time.perf_counter() - started_at
        provenance = {
            "model_repository": MODEL_ID,
            "model_path": _state.get("model_path"),
            "checkpoint_hash": _checkpoint_hash(str(_state.get("model_path") or "")),
            "process_id": os.getpid(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "model_device": str(model_device),
            "load_mode": _state.get("load_mode"),
            "torch_dtype": _state.get("torch_dtype"),
            "device_map_summary": _state.get("device_map_summary"),
            "started_at_utc": started_at_utc,
            "ended_at_utc": _utc_now(),
            "latency_seconds": duration_sec,
            "input_tokens": int(input_len),
            "output_tokens": output_tokens,
            "peak_allocated_gpu_memory_mib": (
                int(torch.cuda.max_memory_allocated(cuda_device) / (1024 * 1024))
                if cuda_device is not None
                else 0
            ),
            "peak_reserved_gpu_memory_mib": (
                int(torch.cuda.max_memory_reserved(cuda_device) / (1024 * 1024))
                if cuda_device is not None
                else 0
            ),
            "fresh_backend_inference": True,
        }
        return {"reply": reply, "provenance": provenance}
