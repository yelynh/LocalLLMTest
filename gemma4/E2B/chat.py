"""
Gemma 4 E2B — 대화형 CLI 챗봇
멀티턴 대화 히스토리 유지. vLLM 서버 또는 transformers 직접 로드.

Usage:
    # 서버 모드 (serve.py 먼저 실행)
    python chat.py
    python chat.py --thinking

    # transformers 직접 로드 (서버 불필요, 느림)
    python chat.py --mode local

명령어:
    /clear      대화 히스토리 초기화
    /thinking   Chain-of-Thought 토글
    /history    대화 히스토리 출력
    /exit       종료
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

BANNER = """
╔══════════════════════════════════════════╗
║        Gemma 4 Effective 2B Chat         ║
║  /clear  /thinking  /history  /exit      ║
╚══════════════════════════════════════════╝
"""


# ── vLLM 서버 모드 ──────────────────────────────────────────────

class ServerChat:
    def __init__(self, thinking: bool = False):
        from openai import OpenAI
        self.client = OpenAI(base_url=resolve_server_url(), api_key="dummy")
        self.history: list[dict] = []
        self.thinking = thinking

    def send(self, text: str, image_path: str | None = None) -> str:
        content: list = []
        if image_path:
            img_bytes = Path(image_path).read_bytes()
            b64 = base64.b64encode(img_bytes).decode()
            ext = Path(image_path).suffix.lstrip(".")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            })
        content.append({"type": "text", "text": text})

        self.history.append({"role": "user", "content": content})

        kwargs = {}
        if self.thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=self.history,
            max_tokens=2048,
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
        reply = "\n\n".join(parts)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def clear(self):
        self.history.clear()


# ── transformers 직접 로드 모드 ─────────────────────────────────

class LocalChat:
    def __init__(self, thinking: bool = False, quant: str | None = None):
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "local 모드에 필요한 패키지가 현재 Python 환경에 없습니다. "
                "`conda activate gemma4` 후 다시 실행하거나 "
                "`./startGemma.sh local-chat`을 사용하세요."
            ) from e

        self.quant = quant
        self.model_name_or_path = resolve_model_name_or_path()
        local_files_only = uses_local_files_only(self.model_name_or_path)

        print(f"[로딩] {self.model_name_or_path} (quant={quant or 'bf16'}) ...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        self.history: list[dict] = []
        self.thinking = thinking

    @staticmethod
    def _text_message(role: str, text: str) -> dict:
        return {"role": role, "content": [{"type": "text", "text": text}]}

    def send(self, text: str, image_path: str | None = None) -> str:
        if image_path:
            raise RuntimeError("local 모드의 이미지 입력은 아직 미지원입니다. server 모드를 사용하세요.")

        user_message = self._text_message("user", text)
        self.history.append(user_message)

        try:
            prompt = self.processor.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            if not isinstance(prompt, str):
                raise TypeError(
                    f"chat template returned {type(prompt).__name__}, expected str"
                )

            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
            )
            # 입력 토큰 제외하고 디코딩
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            response_text = self.processor.decode(new_tokens, skip_special_tokens=False)
            reply = format_generated_response(self.processor, response_text)
        except Exception:
            if self.history and self.history[-1] is user_message:
                self.history.pop()
            raise

        self.history.append(self._text_message("assistant", reply))
        return reply

    def clear(self):
        self.history.clear()


# ── 인터랙티브 루프 ─────────────────────────────────────────────

def run(chat):
    print(BANNER)
    mode_name = "server (vLLM)" if isinstance(chat, ServerChat) else "local (transformers)"
    print(f"모드: {mode_name} | thinking: {'ON' if chat.thinking else 'OFF'}\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if not user:
            continue

        # 슬래시 명령어
        if user == "/exit" or user == "/quit":
            print("[종료]")
            break
        elif user == "/clear":
            chat.clear()
            print("[히스토리 초기화]\n")
            continue
        elif user == "/thinking":
            chat.thinking = not chat.thinking
            print(f"[thinking: {'ON' if chat.thinking else 'OFF'}]\n")
            continue
        elif user == "/history":
            if not chat.history:
                print("[히스토리 없음]\n")
            else:
                for i, msg in enumerate(chat.history):
                    role = msg["role"].upper()
                    content = msg["content"]
                    if isinstance(content, list):
                        text = next((c["text"] for c in content if c.get("type") == "text"), "")
                    else:
                        text = content
                    print(f"[{i}] {role}: {text[:120]}{'...' if len(text) > 120 else ''}")
                print()
            continue

        # 이미지 첨부 파싱: "path/to/img.jpg: 이 이미지 설명해줘"
        image_path = None
        if ":" in user:
            maybe_path, _, rest = user.partition(":")
            maybe_path = maybe_path.strip()
            if Path(maybe_path).is_file() and maybe_path.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".gif")
            ):
                image_path = maybe_path
                user = rest.strip()

        try:
            print("Gemma: ", end="", flush=True)
            reply = chat.send(user, image_path=image_path)
            print(reply)
            print()
        except Exception as e:
            print(f"[오류] {e}\n")


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
    parser = argparse.ArgumentParser(description="Gemma 4 E2B 대화형 챗봇")
    parser.add_argument("--mode", choices=["auto", "server", "local"], default="auto",
                        help="auto=서버 우선 후 local fallback, server=vLLM 강제, local=transformers 직접 로드")
    parser.add_argument("--thinking", action="store_true", help="Chain-of-Thought 활성화")
    parser.add_argument("--quant", choices=["bnb"], default=None,
                        help="local 모드용 양자화 옵션 (RTX 4060 8GB는 bnb 권장)")
    args = parser.parse_args()

    resolved_mode, resolved_quant = resolve_runtime_mode(args.mode, args.quant)

    try:
        if resolved_mode == "server":
            chat = ServerChat(thinking=args.thinking)
        else:
            chat = LocalChat(thinking=args.thinking, quant=resolved_quant)

        run(chat)
    except RuntimeError as e:
        print(f"[오류] {e}")


if __name__ == "__main__":
    main()
