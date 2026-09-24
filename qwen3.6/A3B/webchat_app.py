from __future__ import annotations

import os
import threading
import time
from typing import Any
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from runtime import MODEL_ID, resolve_model_name_or_path, resolve_server_url, uses_local_files_only


class ChatRequest(BaseModel):
    message: str
    thinking: bool = False
    max_new_tokens: int | None = None


app = FastAPI(title="Qwen3.6 Chat")
_state: dict[str, Any] = {}
_lock = threading.Lock()
logger = logging.getLogger("qwen36_webchat")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN36_MAX_NEW_TOKENS", "192"))
MAX_NEW_TOKENS_CAP = int(os.getenv("QWEN36_MAX_NEW_TOKENS_CAP", "512"))


HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3.6 Chat</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background:#10131d; color:#edf1ff; }
    .wrap { max-width: 920px; margin: 0 auto; padding: 24px; }
    .card { background:#161c2a; border:1px solid #2d3550; border-radius: 18px; padding: 18px; box-shadow:0 20px 50px rgba(0,0,0,.22); }
    h1 { margin:0 0 8px; font-size: 1.8rem; }
    p { color:#a7b0ca; }
    textarea { width:100%; min-height:140px; border-radius:14px; border:1px solid #3d4769; background:#0d1220; color:#eef4ff; padding:14px; font:inherit; resize:vertical; }
    button { margin-top:12px; padding:12px 16px; border:0; border-radius:999px; background:#4f7cff; color:white; font-weight:700; cursor:pointer; }
    pre { white-space:pre-wrap; background:#0d1220; border:1px solid #313b5f; border-radius:14px; padding:14px; min-height:180px; overflow-wrap:anywhere; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    .pill { display:inline-flex; padding:6px 10px; border-radius:999px; background:#202b45; color:#bfd0ff; font-size:.8rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <h1>Qwen3.6-35B-A3B</h1>
        <span class="pill">GPU 1</span>
        <span class="pill">BnB 4bit</span>
      </div>
      <p>간단 로컬 LLM 챗봇 테스트 페이지입니다.</p>
      <textarea id="prompt" placeholder="프롬프트를 입력하세요"></textarea>
      <div class="row">
        <label><input type="checkbox" id="thinking"> thinking</label>
        <button id="send">보내기</button>
      </div>
      <h3>Response</h3>
      <pre id="out">ready</pre>
    </div>
  </div>
  <script>
    const out = document.getElementById("out");
    document.getElementById("send").onclick = async () => {
      const message = document.getElementById("prompt").value.trim();
      const thinking = document.getElementById("thinking").checked;
      if (!message) return;
      out.textContent = "loading...";
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message, thinking})
      });
      const data = await res.json();
      out.textContent = data.reply || data.error || "empty";
    };
  </script>
</body>
</html>
"""


def load_model() -> None:
    if os.getenv("QWEN36_USE_VLLM_PROXY", "0") == "1":
        _state["proxy"] = True
        _state["server_url"] = resolve_server_url()
        logger.info("proxy_ready: %s", _state["server_url"])
        return

    model_name_or_path = resolve_model_name_or_path()
    local_files_only = uses_local_files_only(model_name_or_path)
    logger.info("loading model: %s (local_only=%s)", model_name_or_path, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    if "AWQ" in str(model_name_or_path).upper():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        try:
            from bitsandbytes.nn import Params4bit

            original_new = Params4bit.__new__

            def compat_new(cls, data=None, requires_grad=False, quant_state=None, blocksize=None,
                           compress_statistics=True, quant_type="fp4", quant_storage=torch.uint8,
                           module=None, bnb_quantized=False, **_ignored):
                return original_new(
                    cls,
                    data=data,
                    requires_grad=requires_grad,
                    quant_state=quant_state,
                    blocksize=blocksize,
                    compress_statistics=compress_statistics,
                    quant_type=quant_type,
                    quant_storage=quant_storage,
                    module=module,
                    bnb_quantized=bnb_quantized,
                )

            Params4bit.__new__ = staticmethod(compat_new)
        except Exception as exc:
            logger.warning("bitsandbytes Params4bit compatibility patch skipped: %s", exc)

        offload_dir = Path("./offload")
        offload_dir.mkdir(exist_ok=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        if os.getenv("QWEN36_SINGLE_GPU", "1") == "1":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
                max_memory={0: "22GiB", "cpu": "120GiB"},
                offload_folder=str(offload_dir),
            )
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    logger.info("model_ready: %s", model_name_or_path)


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_model()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "")}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/chat")
def chat(req: ChatRequest):
    with _lock:
        started_at = time.perf_counter()
        if _state.get("proxy"):
            from openai import OpenAI

            client = OpenAI(base_url=_state["server_url"], api_key="dummy")
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": req.message}],
                max_tokens=min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
                temperature=0.2,
                top_p=0.9,
            )
            reply = response.choices[0].message.content or ""
            logger.info(
                "proxy_chat_complete prompt_chars=%s thinking=%s reply_chars=%s duration_sec=%.2f",
                len(req.message),
                req.thinking,
                len(reply),
                time.perf_counter() - started_at,
            )
            return {"reply": reply.strip()}

        tokenizer = _state["tokenizer"]
        model = _state["model"]
        messages = [{"role": "user", "content": req.message}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=req.thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        logger.info(
            "chat_complete prompt_chars=%s thinking=%s reply_chars=%s duration_sec=%.2f",
            len(req.message),
            req.thinking,
            len(reply),
            time.perf_counter() - started_at,
        )
        return {"reply": reply.strip()}
