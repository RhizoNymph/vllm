# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZAYA1's CCA-attention + MoE layer with residual scaling emits all hooks.

The layer carries the full fp32 residual stream (no deferred add), so
``pre_attn`` / ``post_attn`` / ``post_block`` see that stream; ``mlp_in`` is
the normed MoE input and ``mlp_out`` the scaled MoE branch as it joins the
scaled residual.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.model_executor.models.steering_hook_utils import (
    ALL_HOOKS,
    bare,
    calls,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.models import zaya

H = 4


class Norm(nn.Module):
    """RMSNorm stand-in exposing ``weight`` (the layer reads its dtype)."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Attn(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, positions):
        return torch.full_like(hidden_states, self.value)


class MoE(nn.Module):
    """Doubles its input so ``mlp_in`` steering is visible downstream."""

    def forward(self, hidden_states, router_states):
        return hidden_states * 2.0, router_states


def _scaling(hs, hb, rs, rb):
    m = zaya.ZayaResidualScaling(H)
    with torch.no_grad():
        m.hidden_states_scale.fill_(hs)
        m.hidden_states_bias.fill_(hb)
        m.residual_scale.fill_(rs)
        m.residual_bias.fill_(rb)
    return m


def _populate(layer):
    layer.input_layernorm = Norm(1.0)
    layer.self_attn = Attn(3.0)
    layer.post_attention_residual_scale = _scaling(2.0, 0.0, 1.0, 0.0)
    layer.post_attention_layernorm = Norm(0.5)
    layer.mlp = MoE()
    layer.post_mlp_residual_scale = _scaling(3.0, 1.0, 1.0, 1.0)
    return layer


def _layer():
    return _populate(bare(zaya.ZayaDecoderLayer, layer_idx=0))


def test_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, zaya)
    layer = _layer()
    r = torch.ones(2, H)
    out, router_states = layer(r, torch.zeros(2, dtype=torch.long), None)

    assert rec.order == ALL_HOOKS
    assert router_states is None
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, H), 1.0))
    # attn branch 3 * 2 + residual 1
    assert torch.equal(rec.seen("post_attn"), torch.full((2, H), 7.0))
    # 7 * 0.5
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, H), 3.5))
    # (3.5 * 2 + 1) * 3
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, H), 24.0))
    # 24 + (7 + 1) * 1
    assert torch.equal(rec.seen("post_block"), torch.full((2, H), 32.0))
    assert torch.equal(out, torch.full((2, H), 32.0))


def test_split_scaling_matches_fused_forward():
    m = _scaling(1.5, 0.25, 0.75, -0.5)
    h = torch.randn(3, H, dtype=torch.bfloat16)
    r = torch.randn(3, H, dtype=torch.float32)
    assert torch.equal(m(h, r), m.scale_branch(h) + m.scale_residual(r))


def _built_layer(monkeypatch, vllm_config, idx=3):
    monkeypatch.setattr(zaya, "RMSNorm", lambda *a, **k: Norm(1.0))
    monkeypatch.setattr(zaya, "ZayaAttention", lambda *a, **k: Attn(3.0))
    monkeypatch.setattr(zaya, "ZayaSparseMoeBlock", lambda *a, **k: MoE())
    config = SimpleNamespace(hidden_size=H, rms_norm_eps=1e-6)
    layer = zaya.ZayaDecoderLayer(
        config, idx, None, f"model.layers.{idx}", vllm_config=vllm_config
    )
    return _populate(layer)


def test_init_registers_every_hook(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config(), idx=6)
    assert layer.layer_idx == 6
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize(
    ("hook", "gain"),
    [
        # mlp_out and post_block join the output directly.
        ("mlp_out", 1.0),
        ("post_block", 1.0),
        # post_attn rides the residual (scale 1) and the MoE path:
        # v * 0.5 (norm) * 2 (MoE) * 3 (branch scale) + v * 1.
        ("post_attn", 4.0),
    ],
)
def test_residual_steering_reaches_block_output(monkeypatch, hook, gain):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    r = torch.ones(2, H)
    base, _ = layer(r, pos, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, _ = layer(r.clone(), pos, None)
    assert torch.allclose(out - base, gain * vec.expand(2, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    out, _ = layer(r.clone(), pos, None)
    assert torch.equal(out, base)


@pytest.mark.parametrize("hook", ["pre_attn", "mlp_in"])
def test_inner_hooks_change_output(monkeypatch, hook):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    r = torch.ones(2, H)
    base, _ = layer(r, pos, None)

    set_global_vector(layer, hook, torch.tensor([1.0, -2.0, 0.5, 4.0]), 2)
    out, _ = layer(r.clone(), pos, None)
    assert not torch.allclose(out, base)

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    out, _ = layer(r.clone(), pos, None)
    assert torch.equal(out, base)


def test_index_shared_across_layers():
    assert calls((zaya.ZayaModel, "__init__"), "share_steering_index_across_layers")
