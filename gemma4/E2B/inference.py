"""
Gemma 4 E2B — 추론 클라이언트
vLLM 서버(serve.py)가 떠 있거나, transformers 직접 로드 두 가지 모드 지원.

Usage:
    # vLLM 서버 모드 (serve.py 먼저 실행 필요)
    python inference.py --mode server --prompt "안녕"

    # transformers 직접 로드 모드
    python inference.py --mode local --prompt "안녕"

    # 이미지 포함
    python inference.py --mode server --prompt "이 이미지 설명해줘" --image path/to/img.jpg

    # 인터랙티브
    python inference.py --mode server
"""

import argparse
import base64
from pathlib import Path

from runtime import (
    can_reach_server,
    format_generated_response,
    MODEL_ID,
    resolve_model_name_or_path,
    resolve_server_url,
    uses_local_files_only,
)


# ── vLLM 서버 모드 ──────────────────────────────────────────────

def chat_server(prompt: str, image_path: str | None = None, thinking: bool = False) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=resolve_server_url(), api_key="dummy")

    content: list = []
    if image_path:
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode()
        ext = Path(image_path).suffix.lstrip(".")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    kwargs = {}
    if thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        **kwargs,
    )
    message = response.choices[0].message
    parts: list[str] = []
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        parts.append(f"[thinking]\n{reasoning}")
    if message.content:
        parts.append(message.content)
    return "\n\n".join(parts)


# ── transformers 직접 로드 모드 ─────────────────────────────────

def chat_local(
    prompt: str,
    image_path: str | None = None,
    thinking: bool = False,
    quant: str | None = None,
) -> str:
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "local 모드에 필요한 패키지가 현재 Python 환경에 없습니다. "
            "`conda activate gemma4` 후 다시 실행하거나 "
            "`./startGemma.sh local-infer \"프롬프트\"`를 사용하세요."
        ) from e

    if image_path:
        raise RuntimeError("local 모드의 이미지 입력은 아직 미지원입니다. server 모드를 사용하세요.")

    model_name_or_path = resolve_model_name_or_path()
    local_files_only = uses_local_files_only(model_name_or_path)

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model_kwargs = {
        "local_files_only": local_files_only,
    }
    if quant == "bnb":
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("bnb 4bit local 모드는 CUDA GPU가 필요합니다.")

        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
    )
    new_tokens = outputs[0][input_len:]
    response_text = processor.decode(new_tokens, skip_special_tokens=False)
    return format_generated_response(processor, response_text)


# ── 인터랙티브 루프 ─────────────────────────────────────────────

def interactive(mode: str, thinking: bool, quant: str | None):
    if mode == "server":
        fn = chat_server
    else:
        fn = lambda prompt, image_path=None, thinking=False: chat_local(
            prompt,
            image_path=image_path,
            thinking=thinking,
            quant=quant,
        )

    quant_name = quant or "bf16"
    print(f"[Gemma 4 E2B] mode={mode} | local-quant={quant_name} | 종료: exit\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit", "q"):
            break
        if not user:
            continue
        print("Gemma:", fn(user, thinking=thinking))
        print()


def resolve_runtime_mode(requested_mode: str, quant: str | None) -> tuple[str, str | None]:
    if requested_mode != "auto":
        return requested_mode, quant

    if can_reach_server(resolve_server_url()):
        print("[auto] vLLM 서버 감지됨 -> server 모드 사용")
        return "server", quant

    resolved_quant = quant or "bnb"
    print(f"[auto] vLLM 서버 미탐지 -> local 모드 사용 (quant={resolved_quant})")
    return "local", resolved_quant


# ── main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B 추론 클라이언트")
    parser.add_argument("--mode", choices=["auto", "server", "local"], default="auto")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="이미지 파일 경로")
    parser.add_argument("--thinking", action="store_true", help="Chain-of-Thought 활성화")
    parser.add_argument("--quant", choices=["bnb"], default=None,
                        help="local 모드용 양자화 옵션 (RTX 4060 8GB는 bnb 권장)")
    args = parser.parse_args()

    resolved_mode, resolved_quant = resolve_runtime_mode(args.mode, args.quant)

    try:
        if resolved_mode == "server":
            fn = chat_server
        else:
            fn = lambda prompt, image_path=None, thinking=False: chat_local(
                prompt,
                image_path=image_path,
                thinking=thinking,
                quant=resolved_quant,
            )

        if args.prompt:
            print(fn(args.prompt, image_path=args.image, thinking=args.thinking))
        else:
            interactive(resolved_mode, args.thinking, resolved_quant)
    except RuntimeError as e:
        print(f"[오류] {e}")


if __name__ == "__main__":
    main()
