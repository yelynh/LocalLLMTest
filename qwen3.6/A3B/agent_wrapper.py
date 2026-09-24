from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from runtime import MODEL_ID, resolve_server_url


@dataclass
class AgentReply:
    text: str
    raw: Any


class Qwen36Agent:
    """
    Agent integration helper for Qwen3.6-35B-A3B.

    OpenAI-compatible endpoint를 우선 사용하고,
    tool use는 JSON envelope 기반으로 단순화한다.
    """

    def __init__(self, base_url: str | None = None, model: str = MODEL_ID):
        from openai import OpenAI

        self.base_url = (base_url or resolve_server_url()).rstrip("/")
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key="dummy")

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        thinking: bool = False,
    ) -> AgentReply:
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if thinking else None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        message = response.choices[0].message
        return AgentReply(text=message.content or "", raw=response)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Dict[str, Any],
        *,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        schema_text = json.dumps(schema_hint, ensure_ascii=False, indent=2)
        reply = self.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an agent backend. "
                        "Return exactly one valid JSON object and no extra prose.\n"
                        f"JSON schema hint:\n{schema_text}"
                    ),
                },
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return json.loads(reply.text)

    def choose_action(
        self,
        task: str,
        context: Dict[str, Any],
        actions: List[str],
    ) -> Dict[str, Any]:
        schema = {
            "action": "one of the allowed actions",
            "reason": "brief rationale",
            "confidence": "float between 0 and 1",
        }
        return self.complete_json(
            system_prompt="Choose the best next action for the agent workflow.",
            user_prompt=(
                f"Task:\n{task}\n\n"
                f"Context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                f"Allowed actions:\n{json.dumps(actions, ensure_ascii=False)}"
            ),
            schema_hint=schema,
        )
