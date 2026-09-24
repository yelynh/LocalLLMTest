from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_SERVER_URL = "http://localhost:8010/v1"

_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
_MODEL_CACHE_DIRS = [
    _HF_CACHE_DIR / "models--Qwen--Qwen3.6-35B-A3B",
    _HF_CACHE_DIR / "models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit",
]


def resolve_server_url() -> str:
    return os.getenv("QWEN36_SERVER_URL", DEFAULT_SERVER_URL)


def resolve_cached_snapshot() -> Path | None:
    for cache_dir in _MODEL_CACHE_DIRS:
        ref_path = cache_dir / "refs" / "main"
        if not ref_path.exists():
            continue

        snapshot_name = ref_path.read_text(encoding="utf-8").strip()
        if not snapshot_name:
            continue

        snapshot_path = cache_dir / "snapshots" / snapshot_name
        if snapshot_path.exists():
            return snapshot_path
    return None


def resolve_model_name_or_path() -> str:
    override = os.getenv("QWEN36_MODEL") or os.getenv("QWEN36_MODEL_PATH")
    if override:
        return override

    snapshot_path = resolve_cached_snapshot()
    if snapshot_path is not None:
        return str(snapshot_path)

    return MODEL_ID


def uses_local_files_only(model_name_or_path: str) -> bool:
    return Path(model_name_or_path).exists()


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
