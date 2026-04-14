from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_SERVER_URL = "http://localhost:8000/v1"

_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_CACHE_DIR = _HF_CACHE_DIR / "models--google--gemma-4-E2B-it"


def resolve_server_url() -> str:
    return os.getenv("GEMMA4_SERVER_URL", DEFAULT_SERVER_URL)


def resolve_cached_snapshot() -> Path | None:
    ref_path = _MODEL_CACHE_DIR / "refs" / "main"
    if not ref_path.exists():
        return None

    snapshot_name = ref_path.read_text(encoding="utf-8").strip()
    if not snapshot_name:
        return None

    snapshot_path = _MODEL_CACHE_DIR / "snapshots" / snapshot_name
    return snapshot_path if snapshot_path.exists() else None


def resolve_model_name_or_path() -> str:
    override = os.getenv("GEMMA4_MODEL") or os.getenv("GEMMA4_MODEL_PATH")
    if override:
        return override

    snapshot_path = resolve_cached_snapshot()
    if snapshot_path is not None:
        return str(snapshot_path)

    return MODEL_ID


def uses_local_files_only(model_name_or_path: str) -> bool:
    return Path(model_name_or_path).exists()


def format_generated_response(processor, response_text: str) -> str:
    text = response_text.strip()
    try:
        parsed = processor.parse_response(text)
    except Exception:
        return text

    if not isinstance(parsed, dict):
        return text

    parts: list[str] = []
    thinking = parsed.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        parts.append(f"[thinking]\n{thinking.strip()}")

    content = parsed.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())

    return "\n\n".join(parts) if parts else text


def can_reach_server(server_url: str, timeout: float = 1.0) -> bool:
    base = server_url.rstrip("/")
    candidates = []
    if base.endswith("/v1"):
        candidates.append(f"{base[:-3]}/health")
    candidates.append(f"{base}/models")

    for url in candidates:
        try:
            with urlopen(url, timeout=timeout) as response:
                if 200 <= response.status < 500:
                    return True
        except URLError:
            continue
        except Exception:
            continue
    return False
