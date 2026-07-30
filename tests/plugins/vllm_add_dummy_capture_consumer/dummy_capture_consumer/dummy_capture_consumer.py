# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal out-of-tree capture consumer used to test entry-point loading.

Registered under the ``vllm.capture_consumers`` group as
``dummy_capture_consumer`` (see ``setup.py``). Mirrors the shape of a
real third-party plugin: a ``CaptureConsumer`` subclass with a global
spec, constructed by the registry from ``(vllm_config, params)``.
"""

from typing import Any

import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec


class DummyCaptureConsumer(CaptureConsumer):
    """Records per-key row counts; captures for every request."""

    location = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        self._hooks: dict[str, list[int]] = params.get("hooks", {"post_block": [0]})
        self._positions = params.get("positions", "last_prompt")
        self.rows_by_key: dict[CaptureKey, int] = {}

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks=self._hooks, positions=self._positions)

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        rows = tensor.shape[0] if tensor.ndim > 0 else 0
        self.rows_by_key[key] = self.rows_by_key.get(key, 0) + rows
