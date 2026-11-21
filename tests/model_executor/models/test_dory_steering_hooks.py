# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dory's single-sublayer nGPT layers emit the hooks that exist on them.

Each physical Dory layer is attention OR mlp, joined to the unit-norm stream
by a slerp; there is no separate residual tensor. Attention layers expose
``pre_attn`` (incoming stream) and ``post_attn`` (slerped stream); mlp layers
expose ``mlp_in`` / ``mlp_out`` around the MLP and ``post_block`` on the
slerped stream. Recurrent layers fire their hooks once per loop.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.steering_hook_utils import (
    Const,
    Scale,
    bare,
    calls,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    STANDARD_STEERING_HOOKS,
    SteeringHookPoint,
)
from vllm.model_executor.models import dory

H = 4


def _config(layer_types):
    return SimpleNamespace(
        hidden_size=H,
        layer_types=layer_types,
        ngpt_alpha_init=0.05,
        initializer_range=0.02,
    )


def _bare_layer(is_attention, mixer_value, layer_idx=0):
    alpha = torch.full((H,), 0.5)
    attrs = dict(
        layer_idx=layer_idx,
        is_attention=is_attention,
        _alpha_scale=1.0,
        mixer=Const(mixer_value),
    )
    attrs["attn_alpha" if is_attention else "mlp_alpha"] = alpha
    return bare(dory.DoryDecoderLayer, **attrs)


def _stream(n=2):
    return dory.justnorm(torch.arange(1.0, 1.0 + n * H).view(n, H))


def test_attention_layer_hooks(monkeypatch):
    rec = record_hooks(monkeypatch, dory)
    layer = _bare_layer(True, 3.0)
    h = _stream()
    out = layer(torch.zeros(2, dtype=torch.long), h)

    assert rec.order == ["pre_attn", "post_attn"]
    assert torch.equal(rec.seen("pre_attn"), h)
    expected = dory.slerp_alpha(h, torch.full_like(h, 3.0), layer.attn_alpha, 1.0)
    assert torch.equal(rec.seen("post_attn"), expected)
    assert torch.equal(out, expected)


def test_mlp_layer_hooks(monkeypatch):
    rec = record_hooks(monkeypatch, dory)
    layer = _bare_layer(False, 5.0)
    h = _stream()
    out = layer(torch.zeros(2, dtype=torch.long), h)

    assert rec.order == ["mlp_in", "mlp_out", "post_block"]
    assert torch.equal(rec.seen("mlp_in"), h)
    assert torch.equal(rec.seen("mlp_out"), torch.full_like(h, 5.0))
    expected = dory.slerp_alpha(h, torch.full_like(h, 5.0), layer.mlp_alpha, 1.0)
    assert torch.equal(rec.seen("post_block"), expected)
    assert torch.equal(out, expected)


def test_recurrent_layers_fire_once_per_loop(monkeypatch):
    rec = record_hooks(monkeypatch, dory)
    config = SimpleNamespace(n_input_layers=1, n_recurrent_layers=2)
    layers = torch.nn.ModuleList(
        [
            _bare_layer(True, 1.0, layer_idx=0),
            _bare_layer(True, 2.0, layer_idx=1),
            _bare_layer(False, 3.0, layer_idx=2),
            _bare_layer(False, 4.0, layer_idx=3),
        ]
    )
    backbone = bare(dory.DoryBackbone, config=config, num_loops=3, layers=layers)
    backbone.forward(None, torch.zeros(2, dtype=torch.long), inputs_embeds=_stream())

    one_loop = ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    assert rec.order == (
        ["pre_attn", "post_attn"] + one_loop * 3 + ["mlp_in", "mlp_out", "post_block"]
    )


def _built_layer(monkeypatch, layer_types, idx, vllm_config):
    monkeypatch.setattr(dory, "DoryAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(dory, "DoryMLP", lambda *a, **k: Const(5.0))
    layer = dory.DoryDecoderLayer(
        _config(layer_types),
        idx,
        None,
        None,
        f"backbone.layers.{idx}",
        vllm_config=vllm_config,
    )
    alpha = layer.attn_alpha if layer.is_attention else layer.mlp_alpha
    with torch.no_grad():
        alpha.fill_(5.0)
    return layer


@pytest.mark.parametrize(
    ("idx", "present"),
    [
        (1, {SteeringHookPoint.PRE_ATTN, SteeringHookPoint.POST_ATTN}),
        (
            2,
            {
                SteeringHookPoint.MLP_IN,
                SteeringHookPoint.MLP_OUT,
                SteeringHookPoint.POST_BLOCK,
            },
        ),
    ],
)
def test_init_registers_only_existing_hooks(monkeypatch, idx, present):
    layer = _built_layer(
        monkeypatch, ["attention", "attention", "mlp"], idx, fake_vllm_config()
    )
    assert layer.layer_idx == idx
    for hp in STANDARD_STEERING_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[hp]) == (hp in present), hp


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(
        monkeypatch, ["attention", "mlp"], 1, fake_vllm_config(steering=False)
    )
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize(
    ("layer_type", "hook"),
    [("attention", "post_attn"), ("mlp", "post_block")],
)
def test_output_hook_shifts_stream(monkeypatch, layer_type, hook):
    """A vector at the layer's output hook shifts the output by itself."""
    layer = _built_layer(monkeypatch, [layer_type], 0, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = _stream()
    base = layer(pos, h)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    assert torch.allclose(layer(pos, h.clone()) - base, vec.expand(2, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    assert torch.equal(layer(pos, h.clone()), base)


@pytest.mark.parametrize(
    ("layer_type", "hook"),
    [("attention", "pre_attn"), ("mlp", "mlp_in"), ("mlp", "mlp_out")],
)
def test_inner_hooks_change_output(monkeypatch, layer_type, hook):
    layer = _built_layer(monkeypatch, [layer_type], 0, fake_vllm_config())
    if layer_type == "mlp":
        # An input-dependent MLP so an mlp_in steer reaches the output.
        layer.mixer = Scale(2.0)
    pos = torch.zeros(2, dtype=torch.long)
    h = _stream()
    base = layer(pos, h)

    set_global_vector(layer, hook, torch.tensor([1.0, -2.0, 0.5, 4.0]), 2)
    assert not torch.allclose(layer(pos, h.clone()), base)

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    assert torch.equal(layer(pos, h.clone()), base)


def test_index_shared_across_layers():
    assert calls((dory.DoryBackbone, "__init__"), "share_steering_index_across_layers")


def test_backbone_receives_vllm_config():
    """The backbone must get vllm_config, or no layer registers buffers."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(dory.DoryForCausalLM)))
    backbone_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "DoryBackbone"
    ]
    assert backbone_calls
    assert all(
        any(kw.arg == "vllm_config" for kw in c.keywords) for c in backbone_calls
    )
