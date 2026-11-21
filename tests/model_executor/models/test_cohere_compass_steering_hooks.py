# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere Compass's parallel attention/MLP block emits all five hooks.

Like CommandR, one input norm feeds both attention and the MLP and the layer
returns the full stream ``residual + attn + mlp`` (no deferred add). Both
branches all-reduce inside their RowParallelLinear (``reduce_results=True``),
so every hook tensor is complete under TP. ``mlp_in`` is steered on the MLP's
copy of the shared normed input only; ``post_attn`` is ``residual + attn`` and
``post_block`` is ``post_attn + mlp_out``.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.model_executor.models.steering_hook_utils import (
    ALL_HOOKS,
    Const,
    bare,
    calls,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR
from vllm.model_executor.layers.steering import SteeringHookPoint as HP
from vllm.model_executor.models import cohere_compass

H = 4


class Norm(nn.Module):
    """CommandR LayerNorm stand-in: scales and passes the residual through."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, x, residual=None):
        return x * self.factor, residual


class Double(nn.Module):
    def forward(self, x):
        return x * 2.0


def _layer(mlp=None):
    return bare(
        cohere_compass.CohereCompassDecoderLayer,
        layer_idx=0,
        input_layernorm=Norm(0.5),
        self_attn=Const(3.0),
        mlp=mlp if mlp is not None else Const(5.0),
    )


def test_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, cohere_compass)
    h = torch.full((2, H), 2.0)
    out, _ = _layer()(torch.zeros(2, dtype=torch.long), h, None)

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, H), 2.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((2, H), 5.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, H), 1.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, H), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, H), 10.0))
    assert torch.equal(out, torch.full((2, H), 10.0))


def test_pass_through_matches_upstream_sum(monkeypatch):
    record_hooks(monkeypatch, cohere_compass)
    layer = _layer(mlp=Double())
    h = torch.randn(3, H, dtype=torch.bfloat16)
    out, residual = layer(torch.zeros(3, dtype=torch.long), h, None)
    normed = h * 0.5
    assert torch.equal(out, h + torch.full_like(h, 3.0) + normed * 2.0)
    assert torch.equal(residual, h)


def _built_layer(monkeypatch, vllm_config, idx=2):
    monkeypatch.setattr(
        cohere_compass, "CohereCompassAttention", lambda *a, **k: Const(3.0)
    )
    monkeypatch.setattr(cohere_compass, "CohereMLP", lambda *a, **k: Double())
    monkeypatch.setattr(cohere_compass, "LayerNorm", lambda *a, **k: Norm(0.5))
    config = SimpleNamespace(hidden_size=H, layer_norm_eps=1e-5)
    return cohere_compass.CohereCompassDecoderLayer(
        config, prefix=f"language_model.model.layers.{idx}", vllm_config=vllm_config
    )


def test_init_registers_every_hook(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(), idx=5)
    assert layer.layer_idx == 5
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[HP(hook)]), hook


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize(
    ("hook", "gain"),
    [
        ("pre_attn", 1.0),
        ("post_attn", 1.0),
        ("mlp_out", 1.0),
        ("post_block", 1.0),
        # Only the MLP sees mlp_in, which it doubles.
        ("mlp_in", 2.0),
    ],
)
def test_steering_reaches_block_output(monkeypatch, hook, gain):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, H)
    base, _ = layer(pos, h, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, _ = layer(pos, h.clone(), None)
    assert torch.allclose(out - base, gain * vec.expand(2, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    out, _ = layer(pos, h.clone(), None)
    assert torch.equal(out, base)


def test_branches_reduce_before_hooks():
    """No deferred/fused all-reduce: the MLP reduces itself under TP."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(cohere_compass.CohereCompassDecoderLayer))
    )
    mlp_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CohereMLP"
    ]
    assert len(mlp_calls) == 1
    kw = {k.arg: k.value for k in mlp_calls[0].keywords}
    assert ast.literal_eval(kw["reduce_results"]) is True
    attn_src = inspect.getsource(cohere_compass.CohereCompassAttention.__init__)
    assert "reduce_results" not in attn_src


def test_index_shared_across_layers():
    assert calls(
        (cohere_compass.CohereCompassTextModel, "__init__"),
        "share_steering_index_across_layers",
    )


def test_model_passes_vllm_config_to_layers():
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(cohere_compass.CohereCompassTextModel))
    )
    layer_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "CohereCompassDecoderLayer"
    ]
    assert layer_calls
    assert all(any(k.arg == "vllm_config" for k in c.keywords) for c in layer_calls)
