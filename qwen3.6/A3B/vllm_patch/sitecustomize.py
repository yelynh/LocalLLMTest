"""Runtime-only vLLM patch for Qwen3.6 compressed-tensors MoE.

This file is loaded automatically by Python when its directory is on
PYTHONPATH.  It does not patch vLLM globally; it only activates when
QWEN36_DISABLE_MOE_MARLIN=1 is set for this process.
"""

from __future__ import annotations

import os


def _disable_moe_marlin() -> None:
    def _unsupported(*_args, **_kwargs) -> bool:
        return False

    try:
        from vllm.model_executor.layers.quantization.utils import marlin_utils

        marlin_utils.check_moe_marlin_supports_layer = _unsupported
    except Exception:
        pass

    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe,
        )

        compressed_tensors_moe.check_moe_marlin_supports_layer = _unsupported
    except Exception:
        pass


if os.getenv("QWEN36_DISABLE_MOE_MARLIN", "0") == "1":
    _disable_moe_marlin()
