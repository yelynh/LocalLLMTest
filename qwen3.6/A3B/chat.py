from __future__ import annotations

import argparse

from inference import chat_local, chat_server
from runtime import can_reach_server, resolve_server_url


class ServerChat:
    def __init__(self, thinking: bool = False):
        from openai import OpenAI
        from runtime import MODEL_ID

        self.client = OpenAI(base_url=resolve_server_url(), api_key="dummy")
        self.model = MODEL_ID
        self.history: list[dict] = []
        self.thinking = thinking

    def send(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if self.thinking else None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            extra_body=extra_body,
        )
        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def clear(self):
        self.history.clear()


class LocalChat:
    def __init__(self, thinking: bool = False, quant: str | None = "bnb"):
        self.history: list[dict] = []
        self.thinking = thinking
        self.quant = quant

    def send(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        transcript = []
        for msg in self.history:
            transcript.append(f"{msg['role']}: {msg['content']}")
        prompt = "\n".join(transcript) + "\nassistant:"
        reply = chat_local(prompt, thinking=self.thinking, quant=self.quant)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def clear(self):
        self.history.clear()


def run(chat):
    print("[Qwen3.6 Chat] /clear /thinking /history /exit\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            print("[종료]")
            break
        if user == "/clear":
            chat.clear()
            print("[히스토리 초기화]\n")
            continue
        if user == "/thinking":
            chat.thinking = not chat.thinking
            print(f"[thinking: {'ON' if chat.thinking else 'OFF'}]\n")
            continue
        if user == "/history":
            for idx, msg in enumerate(chat.history):
                print(f"[{idx}] {msg['role'].upper()}: {msg['content'][:160]}")
            print()
            continue
        print("Qwen:", chat.send(user))
        print()


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6 대화형 CLI 챗봇")
    parser.add_argument("--mode", choices=["auto", "server", "local"], default="auto")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--quant", choices=["bnb"], default="bnb")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "server" if can_reach_server(resolve_server_url()) else "local"
    chat = ServerChat(thinking=args.thinking) if mode == "server" else LocalChat(thinking=args.thinking, quant=args.quant)
    run(chat)


if __name__ == "__main__":
    main()
