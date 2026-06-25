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
    register_steering_buffers,
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


def _emit_recorder(monkeypatch):
    """Replace the shared op-emitting helper with a recorder.

    Returns the list of ``(x, hook_point)`` it was called with. Lets the
    CPU tests assert *which* tensor/hook each entry point steers without
    needing the CUDA-only ``apply_steering`` op.
    """
    seen: list[tuple[torch.Tensor, SteeringHookPoint]] = []

    def _record(module, x, hook_point):
        seen.append((x, hook_point))
        return x

    monkeypatch.setattr(steering_mod, "_emit_steering_op", _record)
    return seen


def test_block_steering_routes_through_shared_emit(monkeypatch):
    """Regression: ``apply_block_steering`` must steer ``residual`` via the
    shared ``_emit_steering_op`` helper (POST_BLOCK), not a stale hand-rolled
    op call. Before the fix it emitted a 4-arg ``apply_steering`` against the
    12-arg op, crashing torch.compile tracing of the default hook."""
    _intercept_capture(monkeypatch, manager=None)
    seen = _emit_recorder(monkeypatch)
    layer = _Layer()
    # Only the table needs to exist for the hasattr gate to pass.
    layer.register_buffer(
        HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK], torch.zeros(5, 8)
    )
    hidden = torch.randn(3, 8)
    residual = torch.randn(3, 8)

    out_hidden, out_residual = apply_block_steering(layer, hidden, residual)

    assert len(seen) == 1
    steered_x, hook = seen[0]
    assert hook == SteeringHookPoint.POST_BLOCK
    torch.testing.assert_close(steered_x, residual)  # steers residual, not hidden
    torch.testing.assert_close(out_hidden, hidden)


def test_apply_steering_op_schema_has_full_arity():
    """The registered op carries the full 12-arg dynamic signature; both
    entry points build a call of this arity via ``_emit_steering_op``. Guards
    the op side of the same regression."""
    import vllm.model_executor.layers.steering  # noqa: F401  registers the op

    schema = torch.ops.vllm.apply_steering.default._schema
    assert len(schema.arguments) == 12, (
        f"apply_steering op has {len(schema.arguments)} args; "
        "callers in _emit_steering_op must match exactly"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_steering_rides_residual_not_hidden(monkeypatch):
    """When a table is active, steering is added to ``residual``; ``hidden``
    is returned untouched (identical propagation to the old behavior)."""
    _intercept_capture(monkeypatch, manager=None)  # capture irrelevant here
    device = torch.device("cuda")
    layer = _Layer().to(device)
    # Register the full per-hook + shared buffer set (scales default to 1.0,
    # monitor/tier inactive) so the real 12-arg op runs.
    register_steering_buffers(
        layer,
        16,
        max_steering_tokens=8,
        max_steering_configs=2,
        dtype=torch.float16,
    )
    layer.to(device)
    hidden = torch.randn(3, 16, dtype=torch.float16, device=device)
    residual = torch.randn(3, 16, dtype=torch.float16, device=device)

    table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK]
    flag_attr = HOOK_POINT_ANY_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK]
    table = torch.randn(5, 16, dtype=torch.float16, device=device)
    setattr(layer, table_attr, table)
    setattr(layer, flag_attr, torch.ones(1, dtype=torch.bool, device=device))
    layer.steering_index = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    out_hidden, out_residual = apply_block_steering(layer, hidden, residual)

    torch.testing.assert_close(out_hidden, hidden)
    torch.testing.assert_close(out_residual, residual + table[layer.steering_index])
