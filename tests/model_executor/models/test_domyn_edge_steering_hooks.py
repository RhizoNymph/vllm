# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DomynEdge's sandwich-norm decoder layer emits all five intervention hooks.

The layer runs under the (hooked) ``Qwen3Model``, which forwards the steering
buffer sizes only to decoder layers whose ``__init__`` accepts them and shares
the steering index across layers. The attention branch joins the residual in
``pre_feedforward_norm``'s fused add, so ``post_attn`` is read after it.
"""

import ast
import inspect
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
from vllm.model_executor.layers.steering import (
    get_steering_buffer_config,
    get_steering_buffer_dtype,
)
from vllm.model_executor.models import domyn_edge, qwen3


def _layer():
    return bare(
        domyn_edge.DomynEdgeDecoderLayer,
        layer_idx=0,
        pre_attention_norm=Scale(1.0),
        attention=Const(3.0),
        post_attention_norm=Scale(2.0),
        pre_feedforward_norm=Scale(1.0),
        feedforward=Const(5.0),
        post_feedforward_norm=Scale(3.0),
    )


def test_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, domyn_edge)
    out, residual = _layer()(torch.zeros(2, dtype=torch.long), torch.ones(2, 4), None)

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 7.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 7.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 15.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 22.0))
    assert torch.equal(out + residual, torch.full((2, 4), 22.0))


def test_with_incoming_residual(monkeypatch):
    rec = record_hooks(monkeypatch, domyn_edge)
    r = torch.full((1, 4), 10.0)
    _layer()(torch.zeros(1, dtype=torch.long), torch.ones(1, 4), r)
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 17.0))


def _built_layer(monkeypatch, vllm_config, idx=2):
    monkeypatch.setattr(domyn_edge, "DomynEdgeAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(domyn_edge, "Qwen3MLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(domyn_edge, "RMSNorm", lambda *a, **k: Scale(1.0))
    n = idx + 1
    config = SimpleNamespace(
        hidden_size=4,
        position_layers=["rope"] * n,
        rope_parameters={"rope_type": "default"},
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        per_layer_config=[SimpleNamespace(head_dim=4)] * n,
        window_sizes=[(0, 0)] * n,
        intermediate_size=8,
        hidden_act="silu",
    )
    max_tokens, max_configs = get_steering_buffer_config(vllm_config)
    # Build exactly as Qwen3Model.__init__ does.
    kwargs = qwen3._build_decoder_layer_kwargs(
        domyn_edge.DomynEdgeDecoderLayer,
        config=config,
        cache_config=None,
        quant_config=None,
        prefix=f"model.layers.{idx}",
        max_steering_tokens=max_tokens,
        max_steering_configs=max_configs,
        steering_dtype=get_steering_buffer_dtype(vllm_config),
    )
    return domyn_edge.DomynEdgeDecoderLayer(**kwargs)


def test_qwen3_model_forwards_steering_sizes():
    kwargs = qwen3._build_decoder_layer_kwargs(
        domyn_edge.DomynEdgeDecoderLayer,
        prefix="x",
        max_steering_tokens=8,
        max_steering_configs=2,
        steering_dtype=torch.float32,
    )
    assert {"max_steering_tokens", "max_steering_configs", "steering_dtype"} <= set(
        kwargs
    )


def test_init_registers_every_hook(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config(), idx=4)
    assert layer.layer_idx == 4
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
    """DomynEdgeModel reuses Qwen3Model.__init__, which shares the index."""
    assert domyn_edge.DomynEdgeModel.decoder_layer_type is (
        domyn_edge.DomynEdgeDecoderLayer
    )
    tree = ast.parse(inspect.getsource(domyn_edge.DomynEdgeModel))
    cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "__init__" for n in cls_node.body
    )
    assert calls((qwen3.Qwen3Model, "__init__"), "share_steering_index_across_layers")
