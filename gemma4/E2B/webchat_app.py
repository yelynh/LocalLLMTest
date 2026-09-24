from __future__ import annotations

import os
import threading
import time
from typing import Any
import logging

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from runtime import MODEL_ID, resolve_model_name_or_path, uses_local_files_only


class ChatRequest(BaseModel):
    message: str
    thinking: bool = False
    max_new_tokens: int | None = None


app = FastAPI(title="Gemma4 E2B Chat")
_state: dict[str, Any] = {}
_lock = threading.Lock()
logger = logging.getLogger("gemma4_webchat")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GEMMA4_MAX_NEW_TOKENS", "128"))
MAX_NEW_TOKENS_CAP = int(os.getenv("GEMMA4_MAX_NEW_TOKENS_CAP", "256"))


HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemma4 E2B Chat</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background:#0f1720; color:#e9f0f4; }
    .wrap { max-width: 920px; margin: 0 auto; padding: 24px; }
    .card { background:#13212d; border:1px solid #264253; border-radius: 18px; padding: 18px; box-shadow:0 20px 50px rgba(0,0,0,.22); }
    h1 { margin:0 0 8px; font-size: 1.8rem; }
    p { color:#9bb0bf; }
    textarea { width:100%; min-height:140px; border-radius:14px; border:1px solid #355163; background:#0d1822; color:#eef6fb; padding:14px; font:inherit; resize:vertical; }
    button { margin-top:12px; padding:12px 16px; border:0; border-radius:999px; background:#10b981; color:white; font-weight:700; cursor:pointer; }
    pre { white-space:pre-wrap; background:#0d1822; border:1px solid #294658; border-radius:14px; padding:14px; min-height:180px; overflow-wrap:anywhere; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    .pill { display:inline-flex; padding:6px 10px; border-radius:999px; background:#173142; color:#a6d4f1; font-size:.8rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row">
        <h1>Gemma4 E2B</h1>
        <span class="pill">GPU 0</span>
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
    model_name_or_path = resolve_model_name_or_path()
    local_files_only = uses_local_files_only(model_name_or_path)
    logger.info("loading model: %s (local_only=%s)", model_name_or_path, local_files_only)
    processor = AutoProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    _state["processor"] = processor
    _state["model"] = model


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
        processor = _state["processor"]
        model = _state["model"]
        messages = [{"role": "user", "content": [{"type": "text", "text": req.message}]}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.thinking,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP),
            temperature=0.2,
            top_p=0.9,
            top_k=64,
            do_sample=False,
        )
        reply = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        logger.info(
            "chat_complete prompt_chars=%s thinking=%s reply_chars=%s duration_sec=%.2f",
            len(req.message),
            req.thinking,
            len(reply),
            time.perf_counter() - started_at,
        )
        return {"reply": reply.strip()}
