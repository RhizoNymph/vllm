# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 Flash decoder layers emit the HY V4 intervention hooks.

``HYV4FlashDecoderLayer`` replaces ``HYV4DecoderLayer.__init__`` (so it must
register its own buffers) and inherits the hooked forward; ``HYV4FlashModel``
inherits the backbone ``__init__`` (index sharing) but has its own Engram
forward, which must still steer the final iHC channels before the head.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.model_executor.models.steering_hook_utils import (
    Const,
    Scale,
    bare,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.models.hy_v4 import interventions as hy_iv
from vllm.models.hy_v4_flash.nvidia import attention as flash_attn
from vllm.models.hy_v4_flash.nvidia import model as flash
from vllm.models.hy_v4_flash.nvidia import vl_model as flash_vl

N, H, T = 2, 4, 3
IHC_HOOKS = [
    "mhc_streams_pre_attn",
    "pre_attn",
    "post_attn",
    "mhc_streams_pre_mlp",
    "mlp_in",
    "mlp_out",
]


class _HC(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def prepare_input(self, h):
        return h.unsqueeze(1).repeat(1, N, 1) if h.dim() == 2 else h

    def pre(self, h):
        return h.sum(1), torch.full(h.shape[:2], 0.5), h

    def post(self, x, residual, post_gates):
        return residual + post_gates.unsqueeze(-1) * x.unsqueeze(1)


class _Attn(Const):
    def __init__(self, *args, **kwargs):
        super().__init__(3.0)
        self.mla_attn = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
        self.rotary_emb = None


def _hf_config(*, num_layers=3, engram=None):
    return SimpleNamespace(
        hidden_size=H,
        enable_ihc=True,
        hc_mult=N,
        num_hidden_layers=num_layers,
        rms_norm_eps=1e-6,
        mlp_layer_types=["dense"] * num_layers,
        engram_layer_ids=engram,
    )


def _built_layer(monkeypatch, *, layer_idx=1, steering=True):
    monkeypatch.setattr(flash_attn, "HYV4FlashAttention", _Attn)
    monkeypatch.setattr(flash, "RMSNorm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(flash, "HYV4FlashFeedForward", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(flash, "HYV4FlashMoE", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(flash, "HYV4HCLayer", _HC)
    cfg = fake_vllm_config(steering=steering)
    cfg.cache_config = None
    cfg.quant_config = None
    return flash.HYV4FlashDecoderLayer(
        _hf_config(), cfg, prefix=f"model.layers.{layer_idx}"
    )


def _table(layer, hook):
    return getattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)])


def _pos():
    return torch.zeros(T, dtype=torch.long)


def test_init_registers_ihc_hook_set(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=1)
    assert layer.layer_idx == 1
    for hook in ("pre_attn", "post_attn", "mlp_in", "mlp_out"):
        assert _table(layer, hook).shape[-1] == H, hook
    for hook in ("mhc_streams_pre_attn", "mhc_streams_pre_mlp"):
        assert _table(layer, hook).shape[-1] == N * H, hook
    assert not hasattr(
        layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_FINAL]
    )


def test_init_last_layer_owns_final_table(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=2)
    assert _table(layer, "mhc_streams_final").shape[-1] == N * H


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, steering=False)
    assert not hasattr(layer, "steering_index")


def test_inherited_forward_hook_order(monkeypatch):
    layer = _built_layer(monkeypatch)
    rec = record_hooks(monkeypatch, hy_iv)

    def _streams(module, streams, hook_point):
        rec.calls.append((hook_point.value, streams.detach().clone()))
        return streams

    monkeypatch.setattr(hy_iv, "apply_layer_steering_streams", _streams)
    out, _ = layer(_pos(), torch.ones(T, N, H), None)
    assert rec.order == IHC_HOOKS
    assert torch.equal(out, torch.full((T, N, H), 5.0))


@pytest.mark.parametrize("hook", ["mhc_streams_pre_attn", "mhc_streams_pre_mlp"])
def test_channel_vector_shifts_layer_output(monkeypatch, hook):
    layer = _built_layer(monkeypatch)
    h = torch.ones(T, N, H)
    base = layer(_pos(), h.clone(), None)[0]
    vec = torch.arange(N * H, dtype=torch.float32)
    set_global_vector(layer, hook, vec, num_tokens=T)
    assert torch.allclose(
        layer(_pos(), h.clone(), None)[0] - base, vec.view(N, H).expand(T, N, H)
    )
    set_global_vector(layer, hook, torch.zeros(N * H), num_tokens=T)
    assert torch.equal(layer(_pos(), h.clone(), None)[0], base)


class _Head(nn.Module):
    def forward(self, x):
        return x.mean(1)


class _Engram(nn.Module):
    def forward(self, hidden, hashes):
        return hidden + 1.0


def test_engram_forward_steers_final_channels(monkeypatch):
    first = _built_layer(monkeypatch, layer_idx=1)
    last = _built_layer(monkeypatch, layer_idx=2)
    first.engram_layer = _Engram()
    hash_state = SimpleNamespace(
        offsets=torch.zeros(1, 1, dtype=torch.int32),
        prepare_hashes_with_output=lambda *a, **k: None,
    )
    model = bare(
        flash.HYV4FlashModel,
        config=_hf_config(engram=[1]),
        layers=nn.ModuleList([first, last]),
        engram_hash=hash_state,
        hc_head=_Head(),
        norm=Scale(1.0),
    )
    model.embed_input_ids = lambda ids: torch.ones(ids.numel(), H)
    ids = torch.zeros(T, dtype=torch.long)
    base = model(ids, _pos())

    vec = torch.arange(N * H, dtype=torch.float32)
    set_global_vector(last, "mhc_streams_final", vec, num_tokens=T)
    got = model(ids, _pos())
    assert torch.allclose(got - base, vec.view(N, H).mean(0).expand(T, H))

    set_global_vector(last, "mhc_streams_final", torch.zeros(N * H), num_tokens=T)
    assert torch.equal(model(ids, _pos()), base)


def test_flash_model_keeps_backbone_init_and_no_sequence_parallel():
    # Index sharing lives in HYV4Model.__init__, which the Flash model reuses.
    assert "__init__" in vars(flash.HYV4FlashModel)
    assert flash.HYV4FlashModel.supports_ihc_sequence_parallel is False
    assert issubclass(flash.HYV4FlashModel, flash.HYV4Model)


def test_vl_wrapper_builds_the_flash_language_model():
    assert flash_vl.HYV4FlashVLForConditionalGeneration.language_model_architecture == (
        "HYV4FlashForCausalLM"
    )
