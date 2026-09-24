"""
Qwen3.6-35B-A3B — 추론 클라이언트
vLLM 서버 모드와 transformers 직접 로드(BnB 4bit) 모드 지원.
"""

from __future__ import annotations

import argparse

from runtime import MODEL_ID, can_reach_server, resolve_model_name_or_path, resolve_server_url, uses_local_files_only


def chat_server(prompt: str, thinking: bool = False) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=resolve_server_url(), api_key="dummy")
    extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if thinking else None
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


def chat_local(prompt: str, thinking: bool = False, quant: str | None = "bnb") -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "local 모드에 필요한 패키지가 현재 Python 환경에 없습니다. "
            "`conda activate qwen36` 후 다시 실행하거나 "
            "`./startQwen.sh local-infer \"프롬프트\"`를 사용하세요."
        ) from e

    model_name_or_path = resolve_model_name_or_path()
    local_files_only = uses_local_files_only(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    model_kwargs = {"local_files_only": local_files_only}
    if quant == "bnb":
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("bnb 4bit local 모드는 CUDA GPU가 필요합니다.")

        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    messages = [{"role": "user", "content": prompt}]
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
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    )
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def interactive(mode: str, thinking: bool, quant: str | None):
    fn = chat_server if mode == "server" else (lambda prompt, thinking=False: chat_local(prompt, thinking=thinking, quant=quant))
    print(f"[Qwen3.6-35B-A3B] mode={mode} | quant={quant or 'bf16'} | 종료: exit\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit", "q"):
            break
        if not user:
            continue
        print("Qwen:", fn(user, thinking=thinking))
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B 추론 클라이언트")
    parser.add_argument("--mode", choices=["auto", "server", "local"], default="auto")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--quant", choices=["bnb"], default="bnb")
    args = parser.parse_args()

    resolved_mode, resolved_quant = resolve_runtime_mode(args.mode, args.quant)
    fn = chat_server if resolved_mode == "server" else (
        lambda prompt, thinking=False: chat_local(prompt, thinking=thinking, quant=resolved_quant)
    )
    if args.prompt:
        print(fn(args.prompt, thinking=args.thinking))
    else:
        interactive(resolved_mode, args.thinking, resolved_quant)


if __name__ == "__main__":
    main()
