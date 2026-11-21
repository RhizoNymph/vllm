# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LFM3's hybrid conv/attention decoder layer emits all five hooks.

The operator (attention or short conv) is treated as the "attention" sublayer.
With sandwich norm the operator branch is post-normed and then folded into the
residual by ``ffn_norm``'s fused add, so ``post_attn`` is read after that call;
``mlp_out`` is the post-FFN-normed branch and ``post_block`` is
``residual + mlp_out``.
"""

import ast
import inspect
import textwrap
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
from vllm.model_executor.models import lfm3


class ConvStub(torch.nn.Module):
    """ShortConv writes its result into a caller-provided output buffer."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, output):
        output.fill_(self.value)


def _layer(sandwich: bool, attention: bool = True):
    attrs = dict(
        layer_idx=0,
        is_attention_layer=attention,
        operator_norm=Scale(1.0),
        ffn_norm=Scale(1.0),
        feed_forward=Const(5.0),
        use_sandwich_norm=sandwich,
    )
    if attention:
        attrs["self_attn"] = Const(3.0)
    else:
        attrs["short_conv"] = ConvStub(3.0)
    if sandwich:
        attrs["post_operator_norm"] = Scale(2.0)
        attrs["post_ffn_norm"] = Scale(3.0)
    return bare(lfm3.Lfm3DecoderLayer, **attrs)


@pytest.mark.parametrize("attention", [True, False])
def test_sandwich_hook_order_and_values(monkeypatch, attention):
    rec = record_hooks(monkeypatch, lfm3)
    layer = _layer(sandwich=True, attention=attention)
    out, residual = layer(torch.zeros(2, dtype=torch.long), torch.ones(2, 4), None)

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    # residual (1) + post-operator-normed operator branch (3 * 2).
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 7.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 7.0))
    # post-FFN-normed MLP branch (5 * 3).
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 15.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 22.0))
    assert torch.equal(out + residual, torch.full((2, 4), 22.0))


def test_plain_norm_hook_values(monkeypatch):
    rec = record_hooks(monkeypatch, lfm3)
    layer = _layer(sandwich=False)
    r = torch.full((1, 4), 10.0)
    layer(torch.zeros(1, dtype=torch.long), torch.ones(1, 4), r)

    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 14.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((1, 4), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((1, 4), 19.0))


def _built_layer(monkeypatch, vllm_config, idx=3, layer_type="full_attention"):
    monkeypatch.setattr(lfm3, "Lfm3Attention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(lfm3, "ShortConv", lambda *a, **k: ConvStub(3.0))
    monkeypatch.setattr(lfm3, "Lfm3MLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(lfm3, "RMSNorm", lambda *a, **k: Scale(1.0))
    layer_types = ["conv"] * 8
    layer_types[idx] = layer_type
    config = SimpleNamespace(
        hidden_size=4, norm_eps=1e-6, layer_types=layer_types, use_sandwich_norm=True
    )
    return lfm3.Lfm3DecoderLayer(
        config, prefix=f"model.layers.{idx}", vllm_config=vllm_config
    )


@pytest.mark.parametrize("layer_type", ["full_attention", "conv"])
def test_init_registers_every_hook(monkeypatch, layer_type):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config(), idx=5, layer_type=layer_type)
    assert layer.layer_idx == 5
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "mlp_out", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, hook):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, 4)
    base_out, base_res = layer(pos, h, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.equal(out + res, base_out + base_res)


def test_index_shared_across_layers():
    assert calls((lfm3.Lfm3Model, "__init__"), "share_steering_index_across_layers")


def test_model_passes_vllm_config_to_layers():
    tree = ast.parse(textwrap.dedent(inspect.getsource(lfm3.Lfm3Model)))
    layer_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "Lfm3DecoderLayer"
    ]
    assert layer_calls
    assert all(any(kw.arg == "vllm_config" for kw in c.keywords) for c in layer_calls)


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("lfm3_vl", "Lfm3VlForConditionalGeneration"),
        ("lfm3_audio", "Lfm3AudioForConditionalGeneration"),
    ],
)
def test_wrappers_build_lfm3_for_causal_lm(module, cls):
    """The multimodal wrappers inherit coverage by building Lfm3ForCausalLM."""
    import importlib

    mod = importlib.import_module(f"vllm.model_executor.models.{module}")
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(mod, cls))))
    archs = [
        kw.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "init_vllm_registered_model"
        for kw in n.keywords
        if kw.arg == "architectures"
    ]
    assert len(archs) == 1
    assert ast.literal_eval(archs[0]) == ["Lfm3ForCausalLM"]


def test_omni_inherits_vl_language_model():
    from vllm.model_executor.models import lfm3_omni, lfm3_vl

    assert issubclass(
        lfm3_omni.Lfm3OmniForConditionalGeneration,
        lfm3_vl.Lfm3VlForConditionalGeneration,
    )
    assert "language_model" not in inspect.getsource(
        lfm3_omni.Lfm3OmniForConditionalGeneration.__init__
    )
