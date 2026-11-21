# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2Horizon's decoder layer emits all five intervention hooks.

The layer is a pre-norm block whose attention branch add is deferred into
``post_attention_layernorm(hidden, residual)``, so ``post_attn`` is read after
that fused add. The MLP is either a dense ``K2HorizonMLP`` (``down_proj``
reduces) or a ``K2HorizonSparseMoeBlock`` (shared experts are combined and
reduced inside ``FusedMoEFactory``; the sequence-parallel path all-gathers back
to the full token count inside the block), so ``mlp_out`` is always complete.
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
from vllm.model_executor.models import k2_horizon


def _layer():
    return bare(
        k2_horizon.K2HorizonDecoderLayer,
        layer_idx=0,
        input_layernorm=Scale(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0),
    )


def test_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, k2_horizon)
    layer = _layer()
    h = torch.ones(2, 4)
    out, residual = layer(torch.zeros(2, dtype=torch.long), h, None)

    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    # pre_attn: the carried residual entering the layer (not the normed input).
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    # post_attn: residual (1) + attention branch (3), after the fused add-norm.
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 4.0))
    # mlp_in: the normed FFN input (4 * 2).
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 8.0))
    # mlp_out: the MLP branch that joins the residual.
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 9.0))
    assert torch.equal(out + residual, torch.full((2, 4), 9.0))


def test_with_incoming_residual(monkeypatch):
    rec = record_hooks(monkeypatch, k2_horizon)
    layer = _layer()
    h = torch.ones(1, 4)
    r = torch.full((1, 4), 10.0)
    layer(torch.zeros(1, dtype=torch.long), h, r)
    # The fused input norm folds the previous layer's branch into the residual.
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 14.0))
    assert torch.equal(rec.seen("post_block"), torch.full((1, 4), 19.0))


def _hf_config(**overrides):
    base = dict(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_experts=0,
        decoder_sparse_step=1,
        mova_num_experts=0,
        mova_num_experts_per_tok=1,
        moe_gate_bias=False,
        router_score_func="sigmoid",
        router_scaling_factor=1.0,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_parameters=None,
        rope_head_dim=4,
        query_key_norm=False,
        rms_norm_eps=1e-6,
        attention_bias=False,
        head_dim=4,
        attention_gate_func=None,
        hidden_act="silu",
        layernorm_num_groups=1,
        mlp_only_layers=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _built_layer(monkeypatch, vllm_config, *, prefix="model.layers.5", **cfg):
    """Run the real ``__init__`` with stubbed sublayer constructors."""
    built: dict[str, str] = {}

    def _stub(name, value):
        def make(*a, **k):
            built.setdefault("kinds", "")
            built["kinds"] += name + ","
            return Const(value)

        return make

    monkeypatch.setattr(k2_horizon, "K2HorizonAttention", _stub("attn", 3.0))
    monkeypatch.setattr(k2_horizon, "K2HorizonMoVAAttention", _stub("mova", 3.0))
    monkeypatch.setattr(k2_horizon, "K2HorizonMLP", _stub("mlp", 5.0))
    monkeypatch.setattr(k2_horizon, "K2HorizonSparseMoeBlock", _stub("moe", 5.0))
    monkeypatch.setattr(k2_horizon, "K2HorizonRMSNorm", lambda *a, **k: Scale(1.0))
    vllm_config.model_config.hf_text_config = _hf_config(**cfg)
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    layer = k2_horizon.K2HorizonDecoderLayer(vllm_config=vllm_config, prefix=prefix)
    layer._built_kinds = built.get("kinds", "")
    return layer


def test_init_registers_every_hook(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config())
    assert layer.layer_idx == 5
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


def test_init_registers_on_moe_mova_layer(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(
        monkeypatch,
        fake_vllm_config(),
        prefix="model.layers.3",
        num_experts=4,
        mova_num_experts=2,
    )
    assert layer._built_kinds == "mova,moe,"
    assert layer.layer_idx == 3
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
    base_out, base_res = layer(pos, h, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.equal(out + res, base_out + base_res)


def test_mlp_out_steering_shifts_branch(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, 4)
    base_out, base_res = layer(pos, h, None)
    vec = torch.tensor([0.5, 0.0, -1.0, 2.0])
    set_global_vector(layer, "mlp_out", vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose(out - base_out, vec.expand(2, 4))
    assert torch.equal(res, base_res)


def test_sequence_parallel_moe_returns_every_token(monkeypatch):
    """The SP MoE path hands the hook all tokens, not a rank's chunk."""
    block = bare(
        k2_horizon.K2HorizonSparseMoeBlock,
        is_sequence_parallel=True,
        experts=lambda hidden_states, router_logits: hidden_states * 2,
    )
    monkeypatch.setattr(k2_horizon, "sequence_parallel_chunk", lambda x: x[:2])
    monkeypatch.setattr(
        k2_horizon,
        "tensor_model_parallel_all_gather",
        lambda x, dim: torch.cat([x, x, x], dim=dim),
    )
    x = torch.arange(20, dtype=torch.float32).view(5, 4)
    assert block(x).shape == (5, 4)


def test_index_shared_across_layers():
    assert calls(
        (k2_horizon.K2HorizonModel, "__init__"), "share_steering_index_across_layers"
    )
