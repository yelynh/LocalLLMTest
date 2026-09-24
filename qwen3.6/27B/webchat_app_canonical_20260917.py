"""Canonical provenance server for the Qwen3.6 capacity-probe backend.

Same envelope contract as the gemma4 12B canonical server: /health exposes the
server/model/template identity and /api/chat returns reply + raw_reply +
provenance (checkpoint hash, process id, prompt sha256, template sha256,
load_mode, device map, tokens, peak memory, fresh_backend_inference). Single
GPU via CUDA_VISIBLE_DEVICES, lock-serialized calls, greedy decoding.

Model/load mode are env-selected so the same file serves the cached
Qwen3.6-35B-A3B official checkpoint under bitsandbytes nf4 or an AWQ variant.
"""
import hashlib
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_PATH = str(
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B"
    / "snapshots/7da1103448ba36029c34ce1a9a741dfe93ee0c50")
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 240
MAX_NEW_TOKENS_CAP = 1024
LOAD_MODE = os.getenv("QWEN36_LOAD_MODE", "nf4")
DEVICE_MAP = os.getenv("QWEN36_DEVICE_MAP", "single")

SERVER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_hash(model_path: str) -> str:
    candidate = Path(model_path).name
    return candidate if len(candidate) == 40 else ""


def _summarize_device_map(model) -> dict:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return {"available": False}
    counts: dict[str, int] = {}
    for device in device_map.values():
        counts[str(device)] = counts.get(str(device), 0) + 1
    return {"available": True, "device_counts": counts}


class ChatRequest(BaseModel):
    message: str
    system_prompt: str | None = None
    thinking: bool = False
    max_new_tokens: int | None = None


app = FastAPI(title="Qwen3.6 Canonical Chat")
logger = logging.getLogger("qwen36_webchat")
_state: dict[str, Any] = {}
_lock = threading.Lock()


def _resolve_model_path() -> str:
    return os.getenv("QWEN36_MODEL_PATH") or DEFAULT_MODEL_PATH


def _format_prompt(message: str, thinking: bool, system_prompt: str | None = None) -> str:
    messages = [
        {"role": "system", "content": (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()},
        {"role": "user", "content": message.strip()},
    ]
    return _state["tokenizer"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def load_model() -> None:
    model_path = _resolve_model_path()
    logger.info("loading model: %s load_mode=%s", model_path, LOAD_MODE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    template_path = Path(model_path) / "chat_template.jinja"
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text()
        _state["template_sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()
    else:
        _state["template_sha256"] = hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")).hexdigest()
    model_kwargs = {"local_files_only": True,
                    "device_map": {"": 0} if DEVICE_MAP == "single" else DEVICE_MAP,
                    "dtype": torch.bfloat16}
    if LOAD_MODE == "fp8":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype="auto",
            device_map="auto",
            max_memory={i: "20GiB" for i in range(torch.cuda.device_count())})
    elif LOAD_MODE in {"4bit", "nf4"}:
        quantized_cache = Path(__file__).parent / "qwen35b_a3b_nf4"
        if not (quantized_cache / "model.safetensors.index.json").exists():
            logger.info("quantizing to CPU nf4 cache: %s", quantized_cache)
            cpu_model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, device_map={"": "cpu"},
                dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
            cpu_model.save_pretrained(quantized_cache)
            tokenizer.save_pretrained(quantized_cache)
            del cpu_model
        logger.info("loading nf4 cache across visible GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            str(quantized_cache), local_files_only=True, dtype=torch.bfloat16,
            device_map="auto",
            max_memory={i: "15GiB" for i in range(torch.cuda.device_count())})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    _state["model_path"] = model_path
    _state["load_mode"] = LOAD_MODE
    _state["quantized_cache"] = str(Path(__file__).parent / "qwen35b_a3b_nf4") if LOAD_MODE in {"4bit", "nf4"} else None
    _state["torch_dtype"] = str(torch.bfloat16).replace("torch.", "")
    _state["device_map_summary"] = _summarize_device_map(model)
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    logger.info("model_ready: %s load_mode=%s device_map=%s",
                model_path, LOAD_MODE, _state["device_map_summary"])


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_model()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": os.getenv("QWEN36_MODEL_ID") or Path(str(_state.get("model_path") or "")).parents[1].name,
        "prompt_template_sha256": _state.get("template_sha256"),
        "server_source_sha256": SERVER_SOURCE_SHA256,
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
        tokenizer = _state["tokenizer"]
        model = _state["model"]
        prompt = _format_prompt(req.message, req.thinking, req.system_prompt)
        model_device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model_device)
        input_len = inputs["input_ids"].shape[-1]
        cuda_device = model_device if getattr(model_device, "type", "") == "cuda" else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, im_end],
        )
        raw_reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False).strip()
        reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        output_tokens = int(outputs.shape[-1] - input_len)
        provenance = {
            "prompt_template_sha256": _state.get("template_sha256"),
            "server_source_sha256": SERVER_SOURCE_SHA256,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model_repository": os.getenv("QWEN36_MODEL_ID") or MODEL_REPO_FALLBACK,
            "model_path": _state.get("model_path"),
            "checkpoint_hash": _checkpoint_hash(str(_state.get("model_path") or "")),
            "process_id": os.getpid(),
            "model_device": str(model_device),
            "load_mode": _state.get("load_mode"),
            "torch_dtype": _state.get("torch_dtype"),
            "device_map_summary": _state.get("device_map_summary"),
            "quantized_cache": _state.get("quantized_cache"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "input_tokens": int(input_len),
            "output_tokens": output_tokens,
            "peak_allocated_gpu_memory_mib": (
                int(torch.cuda.max_memory_allocated(cuda_device) / (1024 * 1024))
                if cuda_device is not None else None),
            "peak_reserved_gpu_memory_mib": (
                int(torch.cuda.max_memory_reserved(cuda_device) / (1024 * 1024))
                if cuda_device is not None else None),
            "latency_seconds": round(time.perf_counter() - started_at, 6),
            "started_at_utc": started_at_utc,
            "ended_at_utc": _utc_now(),
            "fresh_backend_inference": True,
        }
        logger.info("chat_complete prompt_chars=%s thinking=%s reply_chars=%s duration_sec=%.2f",
                    len(prompt), req.thinking, len(reply), time.perf_counter() - started_at)
        return {"reply": reply, "raw_reply": raw_reply, "provenance": provenance}


MODEL_REPO_FALLBACK = "Qwen/Qwen3.6-35B-A3B"
