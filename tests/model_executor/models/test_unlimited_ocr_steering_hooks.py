# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unlimited-OCR-2's DASH decoder layers take part in interventions.

``UnlimitedOCR2DecoderLayer`` reuses the hooked ``DeepseekV2DecoderLayer``
forward but replaces its ``__init__`` (skipping the parent's buffer
registration), and ``UnlimitedOCR2Model`` replaces ``DeepseekV2Model``'s
``__init__`` (skipping the index sharing); both are restored here.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.model_executor.models.steering_hook_utils import (
    Const,
    Scale,
    calls,
    fake_vllm_config,
    record_hooks,
    registers_buffers,
    set_global_vector,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    STANDARD_STEERING_HOOKS,
)
from vllm.model_executor.models import deepseek_v2, unlimited_ocr

H = 4


def _built_layer(monkeypatch, steering=True, idx=3):
    monkeypatch.setattr(unlimited_ocr, "DASHAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(unlimited_ocr, "DeepseekV2MLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(unlimited_ocr, "RMSNorm", lambda *a, **k: Scale(1.0))
    vllm_config = fake_vllm_config(steering=steering)
    vllm_config.model_config.hf_config = SimpleNamespace(
        hidden_size=H,
        n_routed_experts=None,
        first_k_dense_replace=0,
        num_attention_heads=1,
        intermediate_size=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )
    vllm_config.parallel_config = SimpleNamespace(use_sequence_parallel_moe=False)
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    return unlimited_ocr.UnlimitedOCR2DecoderLayer(vllm_config, f"model.layers.{idx}")


def test_init_registers_buffers_statically():
    assert registers_buffers((unlimited_ocr.UnlimitedOCR2DecoderLayer, "__init__"))


def test_init_registers_every_hook(monkeypatch):
    layer = _built_layer(monkeypatch, idx=7)
    assert layer.layer_idx == 7
    for hp in STANDARD_STEERING_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[hp]), hp


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, steering=False)
    assert not hasattr(layer, "steering_index")


def test_inherited_forward_fires_deepseek_hooks(monkeypatch):
    rec = record_hooks(monkeypatch, deepseek_v2)
    layer = _built_layer(monkeypatch)
    layer(torch.zeros(2, dtype=torch.long), torch.ones(2, H), None)
    # The DeepseekV2 decoder forward emits these sites.
    assert rec.order == ["pre_attn", "post_attn", "post_block"]
    assert torch.equal(rec.seen("post_attn"), torch.full((2, H), 4.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, H), 9.0))


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, hook):
    layer = _built_layer(monkeypatch)
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.ones(2, H)
    base_out, base_res = layer(pos, h, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.equal(out + res, base_out + base_res)


def test_index_shared_across_layers():
    assert calls(
        (unlimited_ocr.UnlimitedOCR2Model, "__init__"),
        "share_steering_index_across_layers",
    )


def test_text_lm_uses_unlimited_ocr2_model():
    assert (
        unlimited_ocr.UnlimitedOCR2TextForCausalLM.model_cls
        is unlimited_ocr.UnlimitedOCR2Model
    )
