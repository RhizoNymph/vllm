# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``apply_block_steering`` (the ``post_block`` hook).

``post_block`` must observe the **block output** -- ``residual + mlp_branch``,
i.e. HF's ``hidden_states[L + 1]`` -- not the bare post-attention residual.
vLLM defers each branch-add into the next layer's fused add+norm, so at the
end of a decoder layer ``residual`` does not yet include this layer's MLP
output. The old ``post_mlp`` hook captured bare ``residual``, which is
byte-identical to ``post_attn`` -- a footgun. ``apply_block_steering`` fixes
the captured value while leaving steering propagation unchanged (the delta
still rides the residual stream into the next layer's fused add).
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers import steering as steering_mod
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    apply_block_steering,
    apply_layer_steering,
)


class _Layer(nn.Module):
    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx


def _intercept_capture(monkeypatch, manager):
    """Force the capture gate to ``manager`` and record every capture call.

    Returns a list of ``(tensor, layer_idx, hook_name)`` tuples. ``manager``
    is what ``get_active_capture_manager`` returns -- pass a sentinel object
    to enable the capture path or ``None`` to exercise the gated-off path.
    """
    captured: list[tuple[torch.Tensor, int, str]] = []

    monkeypatch.setattr(cap_mod, "get_active_capture_manager", lambda: manager)

    def _record(tensor, layer_idx, hook_name):
        captured.append((tensor.clone(), layer_idx, hook_name))

    monkeypatch.setattr(steering_mod, "maybe_capture_residual", _record)
    return captured


def test_capture_sees_block_output(monkeypatch):
    """The captured tensor is ``residual + hidden_states``, not bare residual."""
    captured = _intercept_capture(monkeypatch, manager=object())
    layer = _Layer(layer_idx=3)
    hidden = torch.randn(4, 8)
    residual = torch.randn(4, 8)

    out_hidden, out_residual = apply_block_steering(layer, hidden, residual)

    # No steering table registered -> tensors returned unchanged.
    torch.testing.assert_close(out_hidden, hidden)
    torch.testing.assert_close(out_residual, residual)

    assert len(captured) == 1
    tensor, layer_idx, hook_name = captured[0]
    assert layer_idx == 3
    assert hook_name == SteeringHookPoint.POST_BLOCK.value == "post_block"
    torch.testing.assert_close(tensor, residual + hidden)


def test_capture_gated_off_when_no_manager(monkeypatch):
    """With no active capture manager, the residual+hidden sum is never run."""
    captured = _intercept_capture(monkeypatch, manager=None)
    layer = _Layer()

    out_hidden, out_residual = apply_block_steering(
        layer, torch.randn(2, 4), torch.randn(2, 4)
    )

    assert captured == []
    # Still a clean passthrough of both tensors.
    assert out_hidden.shape == (2, 4)
    assert out_residual.shape == (2, 4)


def test_block_output_differs_from_post_attn(monkeypatch):
    """Regression guard: ``post_block`` must NOT equal ``post_attn``.

    The rename-only code captured bare ``residual`` for ``post_block``,
    byte-identical to ``post_attn``. The two captured tensors must now differ
    by exactly the MLP branch (``hidden_states``).
    """
    captured = _intercept_capture(monkeypatch, manager=object())
    layer = _Layer()
    hidden = torch.randn(4, 8)
    residual = torch.randn(4, 8)

    # post_attn observes the bare residual (apply_layer_steering passes it
    # straight to maybe_capture_residual).
    apply_layer_steering(layer, residual, SteeringHookPoint.POST_ATTN)
    post_attn_capture = captured[-1][0]

    # post_block observes the block output.
    apply_block_steering(layer, hidden, residual)
    post_block_capture = captured[-1][0]

    assert not torch.allclose(post_block_capture, post_attn_capture)
    torch.testing.assert_close(post_block_capture - post_attn_capture, hidden)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_steering_rides_residual_not_hidden(monkeypatch):
    """When a table is active, steering is added to ``residual``; ``hidden``
    is returned untouched (identical propagation to the old behavior)."""
    _intercept_capture(monkeypatch, manager=None)  # capture irrelevant here
    device = torch.device("cuda")
    layer = _Layer().to(device)
    hidden = torch.randn(3, 16, dtype=torch.float16, device=device)
    residual = torch.randn(3, 16, dtype=torch.float16, device=device)

    table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK]
    flag_attr = HOOK_POINT_ANY_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK]
    table = torch.randn(5, 16, dtype=torch.float16, device=device)
    layer.register_buffer(table_attr, table)
    layer.register_buffer(flag_attr, torch.ones(1, dtype=torch.bool, device=device))
    layer.steering_index = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    out_hidden, out_residual = apply_block_steering(layer, hidden, residual)

    torch.testing.assert_close(out_hidden, hidden)
    torch.testing.assert_close(out_residual, residual + table[layer.steering_index])
