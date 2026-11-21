# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AliceAI's Block-AttnRes decoder layer emits all five intervention hooks.

AliceAI has no single carried residual: each sublayer reads a softmax mixture
over ``residual_bank`` (the completed block sources) plus the running
``partial`` (the sum of this block's sublayer outputs so far). The hooks steer
the *stream tip*, the source every later mixture and the next committed bank
row are built from:

* ``pre_attn``: mid-block, ``partial`` entering the layer; on a block-start
  layer (``partial`` is ``None``) the most recently committed bank row,
  written back into the bank so later mixtures read the steered source;
* ``post_attn``: ``partial`` after this layer's attention/KDA output is added;
* ``mlp_in``: ``post_attention_layernorm`` output (the normed MoE input);
* ``mlp_out``: the MoE output (shared + routed, reduced);
* ``post_block``: ``partial + mlp_out``, the returned ``partial``.

``attn_res`` is a Triton kernel, so tests swap in a torch reference of the
same math.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

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
from vllm.model_executor.models import alice_ai

H = 4
T = 2


def _reference_attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
):
    assert delta is None and output_norm_weight is None and block_write_idx == -1
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (prefix.shape[-1],), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


@pytest.fixture(autouse=True)
def _torch_attn_res(monkeypatch):
    monkeypatch.setattr(alice_ai, "attn_res", _reference_attn_res)


def _proj():
    lin = nn.Linear(H, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[0.3, -0.2, 0.1, 0.4]]))
    return lin


def _layer(layer_idx: int, *, block_size: int = 4, committed: bool = True):
    attrs = dict(
        layer_idx=layer_idx,
        layer_type="linear_attention",
        attn_res_block_size=block_size,
        attn_res_rms_norm_eps=1e-6,
        input_layernorm=Scale(1.0),
        linear_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0),
        mlp_res_proj=_proj(),
        mlp_res_norm_weight=nn.Parameter(torch.ones(H)),
        steer_committed_source=committed,
    )
    if layer_idx > 0:
        attrs["attn_res_proj"] = _proj()
        attrs["attn_res_norm_weight"] = nn.Parameter(torch.ones(H))
    return bare(alice_ai.AliceAIDecoderBlock, **attrs)


def _bank(num_sources: int = 3) -> torch.Tensor:
    return (
        torch.arange(T * num_sources * H, dtype=torch.float32).view(T, num_sources, H)
        / 7.0
    )


def _pos():
    return torch.zeros(T, dtype=torch.long)


def test_mid_block_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, alice_ai)
    layer = _layer(1)
    bank = _bank()
    partial = torch.full((T, H), 1.5)
    n, out = layer.forward_attn_res(_pos(), bank, 1, partial.clone())

    assert n == 1
    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    assert torch.equal(rec.seen("pre_attn"), partial)
    assert torch.equal(rec.seen("post_attn"), partial + 3.0)
    mixed = _reference_attn_res(
        partial + 3.0,
        None,
        bank,
        layer.mlp_res_norm_weight,
        layer.mlp_res_proj.weight.squeeze(0),
        None,
        1,
        -1,
        1e-6,
        0.0,
    )
    assert torch.allclose(rec.seen("mlp_in"), mixed * 2.0)
    assert torch.equal(rec.seen("mlp_out"), torch.full((T, H), 5.0))
    assert torch.equal(rec.seen("post_block"), partial + 8.0)
    assert torch.equal(out, partial + 8.0)


def test_block_start_pre_attn_sees_committed_source(monkeypatch):
    rec = record_hooks(monkeypatch, alice_ai)
    layer = _layer(4, block_size=4)
    bank = _bank()
    partial = torch.full((T, H), 2.5)
    n, out = layer.forward_attn_res(_pos(), bank, 1, partial.clone())

    assert n == 2
    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    # The block's partial was committed to bank row 1, which is the stream tip.
    assert torch.equal(rec.seen("pre_attn"), partial)
    # A fresh block's partial starts at this layer's attention output.
    assert torch.equal(rec.seen("post_attn"), torch.full((T, H), 3.0))
    assert torch.equal(rec.seen("post_block"), torch.full((T, H), 8.0))
    assert torch.equal(out, torch.full((T, H), 8.0))


def test_first_layer_pre_attn_sees_embeddings(monkeypatch):
    rec = record_hooks(monkeypatch, alice_ai)
    layer = _layer(0)
    bank = _bank()
    layer.forward_attn_res(_pos(), bank, 1, None)
    assert rec.order == ["pre_attn", "post_attn", "mlp_in", "mlp_out", "post_block"]
    assert torch.equal(rec.seen("pre_attn"), bank[:, 0])


def test_unconfigured_block_start_leaves_bank_untouched(monkeypatch):
    """Without interventions the committed-source hook is skipped entirely."""
    rec = record_hooks(monkeypatch, alice_ai)
    layer = _layer(4, committed=False)
    bank = _bank()
    before = bank.clone()
    partial = torch.full((T, H), 2.5)
    layer.forward_attn_res(_pos(), bank, 1, partial.clone())
    assert "pre_attn" not in rec.order
    expected = before.clone()
    expected[:, 1] = partial
    assert torch.equal(bank, expected)


def test_class_default_is_unconfigured():
    assert alice_ai.AliceAIDecoderBlock.steer_committed_source is False


def _hf_config():
    return SimpleNamespace(
        hidden_size=H,
        num_experts=2,
        rms_norm_eps=1e-6,
        block_attn_res_block_size=4,
        layer_types=["linear_attention"] * 8,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=H,
        linear_value_head_dim=H,
        linear_conv_kernel_dim=4,
    )


def _built_layer(monkeypatch, vllm_config, *, prefix="model.layers.5"):
    """Run the real ``__init__`` with stubbed sublayer constructors."""

    def kda(*a, **k):
        m = Const(3.0)
        m.o_norm = SimpleNamespace(eps=None)
        return m

    monkeypatch.setattr(alice_ai, "KimiGatedDeltaNetAttention", kda)
    monkeypatch.setattr(alice_ai, "Qwen3NextAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(alice_ai, "AliceAISparseMoeBlock", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(alice_ai, "GemmaRMSNorm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(alice_ai, "ReplicatedLinear", lambda *a, **k: _proj())
    vllm_config.model_config.hf_text_config = _hf_config()
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    vllm_config.parallel_config = SimpleNamespace(use_sequence_parallel_moe=False)
    return alice_ai.AliceAIDecoderBlock(
        vllm_config=vllm_config, layer_type="linear_attention", prefix=prefix
    )


def test_init_registers_every_hook(monkeypatch):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, fake_vllm_config())
    assert layer.layer_idx == 5
    assert layer.steer_committed_source is True
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")
    assert layer.steer_committed_source is False


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_mid_block_stream_steering_shifts_partial(monkeypatch, hook):
    """A stream vector shifts the returned block partial by exactly itself."""
    layer = _built_layer(monkeypatch, fake_vllm_config(), prefix="model.layers.5")
    bank = _bank()
    partial = torch.full((T, H), 1.5)
    _, base = layer.forward_attn_res(_pos(), bank.clone(), 2, partial.clone())

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=T)
    _, out = layer.forward_attn_res(_pos(), bank.clone(), 2, partial.clone())
    assert torch.allclose(out - base, vec.expand(T, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=T)
    _, out = layer.forward_attn_res(_pos(), bank.clone(), 2, partial.clone())
    assert torch.equal(out, base)


def test_block_start_steering_edits_committed_bank_row(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config(), prefix="model.layers.4")
    bank = _bank()
    partial = torch.full((T, H), 2.5)
    base_bank = bank.clone()
    layer.forward_attn_res(_pos(), base_bank, 1, partial.clone())

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, "pre_attn", vec, num_tokens=T)
    steered_bank = bank.clone()
    layer.forward_attn_res(_pos(), steered_bank, 1, partial.clone())
    assert torch.allclose(steered_bank[:, 1] - base_bank[:, 1], vec.expand(T, H))
    assert torch.equal(steered_bank[:, 0], base_bank[:, 0])

    set_global_vector(layer, "pre_attn", torch.zeros(H), num_tokens=T)
    zero_bank = bank.clone()
    layer.forward_attn_res(_pos(), zero_bank, 1, partial.clone())
    assert torch.equal(zero_bank, base_bank)


def test_mlp_out_steering_shifts_branch(monkeypatch):
    layer = _built_layer(monkeypatch, fake_vllm_config())
    bank = _bank()
    partial = torch.full((T, H), 1.5)
    _, base = layer.forward_attn_res(_pos(), bank.clone(), 2, partial.clone())
    vec = torch.tensor([0.5, 0.0, -1.0, 2.0])
    set_global_vector(layer, "mlp_out", vec, num_tokens=T)
    _, out = layer.forward_attn_res(_pos(), bank.clone(), 2, partial.clone())
    assert torch.allclose(out - base, vec.expand(T, H))


def test_index_shared_across_layers():
    assert calls(
        (alice_ai.AliceAIModel, "__init__"), "share_steering_index_across_layers"
    )
