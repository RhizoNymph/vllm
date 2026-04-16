# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal example capture consumer plugin.

Records the sum of every captured tensor. See the companion
``pyproject.toml`` for entry-point registration and
``docs/capture_consumers/plugin_authoring.md`` for a full tutorial.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec


class SumConsumer(CaptureConsumer):
    """Example consumer that records the sum of every captured tensor."""

    location: ClassVar[Literal["worker", "driver"]] = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        self.sums: dict[CaptureKey, float] = {}
        self._layers: list[int] = params.get("layers", [0])

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={"post_mlp": self._layers},
            positions="last_prompt",
        )

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.sums[key] = float(tensor.sum().item())
