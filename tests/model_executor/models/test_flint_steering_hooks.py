# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flint's decoder layer emits all five intervention hooks.

Flint has one implementation (no per-platform split). Its low-latency decode
path folds the input residual-add RMSNorm into the fused qkv kernel and the
post-attention residual-add RMSNorm + whole MoE block into one fused decode
MoE kernel, which leaves no tensor for pre_attn / post_attn / mlp_in /
mlp_out. Those fused paths are switched off when interventions are
configured, and kept when nothing is configured.
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
from vllm.models.flint import model as flint


class _Attn(Const):
    """Attention stub; the fused low-latency entry point must not run."""

    def __init__(self, value: float, flint_decode: bool = True):
        super().__init__(value)
        self.flint_decode = flint_decode
        self.ll_calls = 0

    def forward_ll(self, positions, hidden_states, residual, input_layernorm):
        self.ll_calls += 1
        return torch.full_like(hidden_states, 100.0), hidden_states


class _MoE(flint.FlintSparseMoeBlock):
    """MoE stub: plain forward is a constant; the fused tail is recorded."""

    def __init__(self, value: float):
        torch.nn.Module.__init__(self)
        self.value = value
        self.ll_calls = 0

    def forward(self, hidden_states):
        return torch.full_like(hidden_states, self.value)

    def forward_ll(self, attn_out, residual, norm):
        self.ll_calls += 1
        return torch.full_like(attn_out, 100.0), attn_out + residual


def _layer(mlp=None, *, fused=False, flint_decode=False):
    return bare(
        flint.FlintDecoderLayer,
        layer_idx=0,
        fused_low_latency=fused,
        input_layernorm=Scale(1.0),
        self_attn=_Attn(3.0, flint_decode=flint_decode),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0) if mlp is None else mlp,
    )


@pytest.mark.parametrize("moe", [False, True])
def test_hook_order_and_values(monkeypatch, moe):
    rec = record_hooks(monkeypatch, flint)
    layer = _layer(_MoE(5.0) if moe else None)
    h = torch.ones(2, 4)
    out, residual = layer(torch.zeros(2, dtype=torch.long), h, None)

    assert rec.order == ALL_HOOKS
    # pre_attn: carried residual entering the layer.
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    # post_attn: residual (1) + attention branch (3), read after the fused add.
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 4.0))
    # mlp_in: the normed MLP/MoE input (4 * 2).
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 8.0))
    # mlp_out: the whole MLP/MoE branch.
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 9.0))
    assert torch.equal(out + residual, torch.full((2, 4), 9.0))


def test_with_incoming_residual(monkeypatch):
    rec = record_hooks(monkeypatch, flint)
    layer = _layer()
    h = torch.ones(1, 4)
    r = torch.full((1, 4), 10.0)
    layer(torch.zeros(1, dtype=torch.long), h, r)
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 14.0))


def test_fused_low_latency_paths_skipped_when_off(monkeypatch):
    rec = record_hooks(monkeypatch, flint)
    moe = _MoE(5.0)
    layer = _layer(moe, fused=False, flint_decode=True)
    layer(torch.zeros(2, dtype=torch.long), torch.ones(2, 4), None)
    assert layer.self_attn.ll_calls == 0
    assert moe.ll_calls == 0
    assert rec.order == ALL_HOOKS


def test_fused_low_latency_paths_kept_when_on():
    moe = _MoE(5.0)
    layer = _layer(moe, fused=True, flint_decode=True)
    layer(torch.zeros(2, dtype=torch.long), torch.ones(2, 4), torch.ones(2, 4))
    assert layer.self_attn.ll_calls == 1
    assert moe.ll_calls == 1


def _built_layer(monkeypatch, vllm_config, idx: int = 5, moe: bool = False):
    """Run the real ``__init__`` with stubbed sublayer constructors."""
    monkeypatch.setattr(
        flint, "FlintAttention", lambda *a, **k: _Attn(3.0, flint_decode=True)
    )
    monkeypatch.setattr(flint, "FlintMLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(flint, "FlintSparseMoeBlock", lambda *a, **k: _MoE(5.0))
    monkeypatch.setattr(flint, "RMSNorm", lambda *a, **k: Scale(1.0))
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        rms_norm_eps=1e-6,
        mlp_only_layers=[] if moe else [idx],
    )
    return flint.FlintDecoderLayer(
        config, None, None, prefix=f"model.layers.{idx}", vllm_config=vllm_config
    )


@pytest.mark.parametrize("idx", [0, 5, 31])
def test_init_registers_every_hook(monkeypatch, idx):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config(), idx=idx)
    assert layer.layer_idx == idx
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook
    assert layer.fused_low_latency is False


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")
    assert layer.fused_low_latency is True


def test_init_without_vllm_config_keeps_fused_path(monkeypatch):
    layer = _built_layer(monkeypatch, None)
    assert not hasattr(layer, "steering_index")
    assert layer.fused_low_latency is True


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, hook, moe):
    """A residual-stream vector shifts the block output by exactly itself."""
    layer = _built_layer(monkeypatch, fake_vllm_config(), moe=moe)
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, 4)
    base_out, base_res = layer(pos, h, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))
    # The fused decode kernels never ran while interventions are configured.
    assert layer.self_attn.ll_calls == 0

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.equal(out + res, base_out + base_res)


def test_mlp_out_steering_shifts_block_output(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    base_out, base_res = layer(pos, torch.ones(2, 4), None)
    vec = torch.tensor([0.5, 0.0, -1.0, 2.0])
    set_global_vector(layer, "mlp_out", vec, num_tokens=2)
    out, res = layer(pos, torch.ones(2, 4), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))


def test_index_shared_across_layers():
    assert calls((flint.FlintModel, "__init__"), "share_steering_index_across_layers")


def test_model_passes_vllm_config_to_layers():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(flint.FlintModel.__init__)))
    layer_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "FlintDecoderLayer"
    ]
    assert layer_calls
    assert all(any(k.arg == "vllm_config" for k in c.keywords) for c in layer_calls)
