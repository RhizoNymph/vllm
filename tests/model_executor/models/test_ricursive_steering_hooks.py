# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ricursive's sandwich-norm decoder layer emits all five intervention hooks.

The layer defers the attention branch add into ``pre_feedforward_layernorm``,
so ``post_attn`` must be read *after* that fused add (it is the residual that
includes this layer's attention output), and ``post_block`` is that residual
plus the post-FFN-normed MLP branch.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.steering_hook_utils import (
    ALL_HOOKS,
    Const,
    Scale,
    bare,
    calls,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.models import ricursive


def _layer():
    return bare(
        ricursive.RicursiveDecoderLayer,
        layer_idx=0,
        input_layernorm=Scale(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        pre_feedforward_layernorm=Scale(1.0),
        mlp=Const(5.0),
        post_feedforward_layernorm=Scale(3.0),
    )


def test_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, ricursive)
    layer = _layer()
    h = torch.ones(2, 4)
    out, residual = layer(torch.zeros(2, dtype=torch.long), h, None, None)

    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    # pre_attn: carried residual entering the layer.
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    # post_attn: residual (1) + post-attn-normed attention branch (3 * 2).
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 7.0))
    # mlp_in: the normed MLP input.
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 7.0))
    # mlp_out: the post-FFN-normed branch (5 * 3) that joins the residual.
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 15.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 22.0))
    # The layer output itself is unchanged by pass-through hooks.
    assert torch.equal(out + residual, torch.full((2, 4), 22.0))


def test_with_incoming_residual(monkeypatch):
    rec = record_hooks(monkeypatch, ricursive)
    layer = _layer()
    h = torch.ones(1, 4)
    r = torch.full((1, 4), 10.0)
    layer(torch.zeros(1, dtype=torch.long), h, r, None)
    # The fused input norm folds the previous layer's branch into the residual.
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 17.0))


def _built_layer(monkeypatch, vllm_config):
    """Run the real ``__init__`` with stubbed sublayer constructors."""
    monkeypatch.setattr(ricursive, "RicursiveAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(ricursive, "RicursiveMLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(ricursive, "RMSNorm", lambda *a, **k: Scale(1.0))
    config = SimpleNamespace(
        hidden_size=4, intermediate_size=8, ffn_clamp_threshold=None, rms_norm_eps=1e-6
    )
    return ricursive.RicursiveDecoderLayer(
        config, prefix="model.layers.5", vllm_config=vllm_config
    )


def test_init_registers_every_hook(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config())
    assert layer.layer_idx == 5
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, hook):
    """A residual-stream vector shifts the block output by exactly itself."""
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, 4)
    base_out, base_res = layer(pos, h, None, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None, None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, res = layer(pos, h.clone(), None, None)
    assert torch.equal(out + res, base_out + base_res)


def test_index_shared_across_layers():
    assert calls(
        (ricursive.RicursiveModel, "__init__"), "share_steering_index_across_layers"
    )
