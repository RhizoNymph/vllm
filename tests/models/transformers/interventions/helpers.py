# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the Transformers-backend intervention tests."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers import steering as steering_mod
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)

HOOKS = [
    SteeringHookPoint.PRE_ATTN,
    SteeringHookPoint.POST_ATTN,
    SteeringHookPoint.MLP_IN,
    SteeringHookPoint.MLP_OUT,
    SteeringHookPoint.POST_BLOCK,
]


def activate(layer: nn.Module, hook: SteeringHookPoint, vec: torch.Tensor, n: int):
    """Steer every one of the first `n` tokens by `vec` at `hook`, the way the
    runner does: table row 1 holds the vector and the index points at it."""
    table = getattr(layer, HOOK_POINT_TABLE_ATTR[hook])
    table[1].copy_(vec.to(table.dtype))
    getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hook]).fill_(True)
    layer.steering_index[:n].fill_(1)


def intercept_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, str, torch.Tensor]]:
    """Turn the capture gate on and record what every hook point observes.

    The capture custom op is CUDA-only, so the two Python taps that
    `apply_layer_steering` / `apply_block_steering` call are replaced."""
    seen: list[tuple[int, str, torch.Tensor]] = []
    monkeypatch.setattr(cap_mod, "get_active_capture_manager", lambda: object())
    monkeypatch.setattr(
        steering_mod,
        "maybe_capture_residual",
        lambda t, layer_idx, hook: seen.append((layer_idx, hook, t.clone())),
    )
    monkeypatch.setattr(
        steering_mod,
        "maybe_capture_residual_add",
        lambda a, b, layer_idx, hook: seen.append((layer_idx, hook, (a + b).clone())),
    )
    return seen
