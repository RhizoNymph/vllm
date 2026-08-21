# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command-R's parallel block must reduce each branch before POST_ATTN.

Upstream runs `o_proj`/`down_proj` with ``reduce_results=False`` and fuses the
two all-reduces into a single reduction over ``attn + mlp``. That is cheaper,
but it leaves ``hidden_states_attention`` a partial sum under TP>1, so the
POST_ATTN intervention site -- which is a capture and activation-patching site
as well as a steering site -- would observe a shard-local tensor rather than the
post-attention residual.

These tests pin the resulting contract, which is invisible at TP=1 and so is not
covered by the single-GPU steering suites:

  * under TP>1 each branch is reduced separately, and
  * POST_ATTN observes ``residual + reduced(attn)``, not a partial sum.

The layer is built with ``object.__new__`` and stubbed collaborators so this
stays a CPU test (the real ``__init__`` constructs attention and MLP layers).
"""

import torch

from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.model_executor.models import commandr


class _Norm:
    """Stand-in for the fused add-norm: returns (normed, new_residual)."""

    def __call__(self, hidden_states, residual):
        return hidden_states, residual


def _make_layer(monkeypatch, tp_size, seen):
    layer = object.__new__(commandr.CohereDecoderLayer)
    layer.tp_size = tp_size
    layer.layer_idx = 0
    layer.input_layernorm = _Norm()
    layer.self_attn = lambda positions, hidden_states: torch.full_like(
        hidden_states, 2.0
    )
    layer.mlp = lambda hidden_states: torch.full_like(hidden_states, 5.0)

    def _fake_all_reduce(t):
        seen.append(t.clone())
        # A real all-reduce over 2 ranks holding equal shards doubles the value;
        # mimic that so a fused-vs-split mistake changes the arithmetic.
        return t * 2

    monkeypatch.setattr(commandr, "tensor_model_parallel_all_reduce", _fake_all_reduce)
    return layer


def _record_hook_inputs(monkeypatch):
    """Capture the tensor handed to each steering hook point."""
    seen: dict[SteeringHookPoint, torch.Tensor] = {}

    def _apply(module, hidden_states, hook_point):
        seen[hook_point] = hidden_states.clone()
        return hidden_states

    monkeypatch.setattr(commandr, "apply_layer_steering", _apply)
    return seen


def test_tp1_does_not_all_reduce(monkeypatch):
    """At TP=1 the branches are already full; no reduction may be issued."""
    reduced: list[torch.Tensor] = []
    _record_hook_inputs(monkeypatch)
    layer = _make_layer(monkeypatch, tp_size=1, seen=reduced)

    hidden = torch.ones(3, 4)
    out, _ = layer.forward(torch.zeros(3, dtype=torch.long), hidden, None)

    assert reduced == []
    # residual(1) + attn(2) + mlp(5)
    assert torch.allclose(out, torch.full((3, 4), 8.0))


def test_tp2_reduces_each_branch_separately(monkeypatch):
    """Under TP>1 attention and MLP are reduced independently, not as a sum."""
    reduced: list[torch.Tensor] = []
    _record_hook_inputs(monkeypatch)
    layer = _make_layer(monkeypatch, tp_size=2, seen=reduced)

    hidden = torch.ones(3, 4)
    out, _ = layer.forward(torch.zeros(3, dtype=torch.long), hidden, None)

    assert len(reduced) == 2, (
        "expected one all-reduce per branch; a single fused reduce over "
        "attn+mlp would make POST_ATTN observe a partial sum"
    )
    assert torch.allclose(reduced[0], torch.full((3, 4), 2.0))  # attention branch
    assert torch.allclose(reduced[1], torch.full((3, 4), 5.0))  # mlp branch
    # residual(1) + 2*attn(4) + 2*mlp(10)
    assert torch.allclose(out, torch.full((3, 4), 15.0))


def test_post_attn_hook_sees_reduced_attention(monkeypatch):
    """POST_ATTN must observe residual + reduced(attn), never a partial sum."""
    reduced: list[torch.Tensor] = []
    hooks = _record_hook_inputs(monkeypatch)
    layer = _make_layer(monkeypatch, tp_size=2, seen=reduced)

    hidden = torch.ones(3, 4)
    layer.forward(torch.zeros(3, dtype=torch.long), hidden, None)

    post_attn = hooks[SteeringHookPoint.POST_ATTN]
    # residual(1) + reduced attention(2*2) == 5; the unreduced value would be 3.
    assert torch.allclose(post_attn, torch.full((3, 4), 5.0)), (
        "POST_ATTN saw a pre-reduction tensor"
    )

    post_block = hooks[SteeringHookPoint.POST_BLOCK]
    assert torch.allclose(post_block, torch.full((3, 4), 15.0))
