# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 decoder layers (NVIDIA and ROCm) emit the intervention hooks.

iHC layers carry ``hc_mult`` residual channels and take the DeepSeek-V4 mHC
hook set (single-stream ``pre_attn``/``post_attn``/``mlp_in``/``mlp_out``
plus ``mhc_streams_pre_attn``/``mhc_streams_pre_mlp`` and the model-level
``mhc_streams_final``); single-residual layers take the standard five hooks.
The MTP and DFly draft blocks reuse the decoder class and stay unhooked.

The iHC pre/post ops are replaced by small torch stand-ins with the same
signatures and additive residual semantics (the Triton/HPC kernels need a
GPU).
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
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
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.models.hy_v4 import interventions as hy_iv
from vllm.models.hy_v4.amd import model as hy_amd
from vllm.models.hy_v4.nvidia import dfly as hy_dfly
from vllm.models.hy_v4.nvidia import model as hy_nv
from vllm.models.hy_v4.nvidia import mtp as hy_mtp
from vllm.models.hy_v4.nvidia import vl_model as hy_vl

N, H, T = 2, 4, 3
PLATFORMS = [hy_nv, hy_amd]
IHC_HOOKS = [
    "mhc_streams_pre_attn",
    "pre_attn",
    "post_attn",
    "mhc_streams_pre_mlp",
    "mlp_in",
    "mlp_out",
]


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


class _HC(nn.Module):
    """iHC boundary stand-in: sum-reduce with 0.5 post gates, additive post."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hc_pre = SimpleNamespace(hpc_op=None)
        self.hc_post = SimpleNamespace(hpc_op=None)

    def prepare_input(self, h):
        return h.unsqueeze(1).repeat(1, N, 1) if h.dim() == 2 else h

    def pre(self, h):
        return h.sum(1), torch.full(h.shape[:2], 0.5), h

    def post(self, x, residual, post_gates):
        return residual + post_gates.unsqueeze(-1) * x.unsqueeze(1)


class _Attn(Const):
    def __init__(self, value: float = 3.0):
        super().__init__(value)
        self.rotary_emb = None


def _record(monkeypatch):
    rec = record_hooks(monkeypatch, hy_iv)

    def _streams(module, streams, hook_point):
        rec.calls.append((hook_point.value, streams.detach().clone()))
        return streams

    monkeypatch.setattr(hy_iv, "apply_layer_steering_streams", _streams)
    return rec


def _bare_layer(mod, *, ihc=True, cls=None, layer_idx=1):
    return bare(
        cls or mod.HYV4DecoderLayer,
        layer_idx=layer_idx,
        hidden_size=H,
        hc_mult=N,
        enable_ihc=ihc,
        defer_reduction=False,
        input_layernorm=Scale(1.0),
        self_attn=_Attn(3.0),
        post_attention_layernorm=Scale(2.0 if not ihc else 1.0),
        mlp=Const(5.0),
        hc_attn_layer=_HC(),
        hc_mlp_layer=_HC(),
    )


def _pos():
    return torch.zeros(T, dtype=torch.long)


# ---------------------------------------------------------------------------
# Hook order and values (stubs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_plain_layer_hook_order_and_values(monkeypatch, mod):
    rec = _record(monkeypatch)
    layer = _bare_layer(mod, ihc=False)
    out, residual = layer(_pos(), torch.ones(T, H), torch.full((T, H), 2.0))

    assert rec.order == ALL_HOOKS
    # pre_attn: the carried stream (previous branch 1 + residual 2).
    assert torch.equal(rec.seen("pre_attn"), torch.full((T, H), 3.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((T, H), 6.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((T, H), 12.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((T, H), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((T, H), 11.0))
    assert torch.equal(out + residual, torch.full((T, H), 11.0))


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_ihc_layer_hook_order_and_values(monkeypatch, mod):
    rec = _record(monkeypatch)
    layer = _bare_layer(mod)
    out, residual = layer(_pos(), torch.ones(T, N, H), None)

    assert rec.order == IHC_HOOKS
    assert torch.equal(rec.seen("mhc_streams_pre_attn"), torch.ones(T, N, H))
    # Normed single-stream attention input (the pre op's reduction).
    assert torch.equal(rec.seen("pre_attn"), torch.full((T, H), 2.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((T, H), 3.0))
    # Channels entering the FFN: 1 + 0.5 * attention 3.
    assert torch.equal(rec.seen("mhc_streams_pre_mlp"), torch.full((T, N, H), 2.5))
    assert torch.equal(rec.seen("mlp_in"), torch.full((T, H), 5.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((T, H), 5.0))
    assert residual is None
    assert torch.equal(out, torch.full((T, N, H), 5.0))


def test_amd_fused_post_pre_path_keeps_hooks(monkeypatch):
    """ROCm fuses attention post + FFN pre + RMSNorm into one Triton kernel."""
    rec = _record(monkeypatch)
    layer = _bare_layer(hy_amd)

    def _fused(x, residual, post, *args):
        new = residual + post.unsqueeze(-1) * x.unsqueeze(1)
        return new.sum(1), post, new

    monkeypatch.setattr(hy_amd, "triton_ihc_supported", lambda x: True)
    monkeypatch.setattr(hy_amd, "triton_ihc_post_pre_rms_norm", _fused)
    layer.hc_mlp_layer.hc_pre = SimpleNamespace(
        hpc_op=None, hc_fn=SimpleNamespace(weight=None), hc_scale=None,
        hc_base=None, magnitude=2.0, hc_eps=1e-6, layernorm_epsilon=1e-6,
    )  # fmt: skip
    layer.post_attention_layernorm.weight = None
    out, _ = layer(_pos(), torch.ones(T, N, H), None)
    assert rec.order == IHC_HOOKS
    assert torch.equal(rec.seen("mhc_streams_pre_mlp"), torch.full((T, N, H), 2.5))
    assert torch.equal(out, torch.full((T, N, H), 5.0))


@pytest.mark.parametrize(
    "cls",
    [hy_mtp.HYV4MTPDecoderLayer, hy_dfly.HYV4DFlyDecoderLayer],
    ids=["mtp", "dfly"],
)
@pytest.mark.parametrize("ihc", [True, False])
def test_draft_blocks_emit_no_hooks(monkeypatch, cls, ihc):
    rec = _record(monkeypatch)
    layer = _bare_layer(hy_nv, ihc=ihc, cls=cls, layer_idx=0)
    h = torch.ones(T, N, H) if ihc else torch.ones(T, H)
    layer(_pos(), h, None)
    assert rec.order == []


def test_mtp_builds_the_unhooked_block():
    assert calls(
        (hy_mtp.HYV4MultiTokenPredictorLayer, "__init__"), "HYV4MTPDecoderLayer"
    )
    assert not calls(
        (hy_mtp.HYV4MultiTokenPredictorLayer, "__init__"), "HYV4DecoderLayer"
    )


# ---------------------------------------------------------------------------
# Model-level final channels hook
# ---------------------------------------------------------------------------


def _pp(monkeypatch, mod):
    monkeypatch.setattr(
        mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    if hasattr(mod, "set_token_rope"):
        monkeypatch.setattr(mod, "set_token_rope", lambda *a, **k: None)


class _Head(nn.Module):
    def forward(self, x):
        return x.mean(1)


def _bare_model(mod, layers):
    return bare(
        mod.HYV4Model,
        layers=nn.ModuleList(layers),
        start_layer=0,
        end_layer=len(layers),
        enable_ihc=True,
        ihc_sequence_parallel=False,
        hc_head=_Head(),
        norm=Scale(1.0),
        dfly_enabled=False,
        pp_buffers={},
    )


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_model_steers_final_channels_before_head(monkeypatch, mod):
    rec = _record(monkeypatch)
    _pp(monkeypatch, mod)
    model = _bare_model(mod, [_bare_layer(mod, layer_idx=0), _bare_layer(mod)])
    out = model(None, _pos(), None, inputs_embeds=torch.ones(T, H))
    assert rec.order == [*IHC_HOOKS, *IHC_HOOKS, "mhc_streams_final"]
    final = rec.seen("mhc_streams_final")
    assert final.shape == (T, N, H)
    assert torch.equal(out, final.mean(1))


# ---------------------------------------------------------------------------
# Real __init__ registration
# ---------------------------------------------------------------------------


def _vllm_config(*, steering=True):
    cfg = fake_vllm_config(steering=steering)
    cfg.cache_config = None
    cfg.quant_config = None
    return cfg


def _hf_config(*, ihc=True, num_layers=3):
    return SimpleNamespace(
        hidden_size=H,
        enable_ihc=ihc,
        hc_mult=N,
        num_hidden_layers=num_layers,
        max_position_embeddings=16,
        num_attention_heads=1,
        qk_nope_head_dim=H,
        qk_rope_head_dim=0,
        v_head_dim=H,
        q_lora_rank=None,
        kv_lora_rank=H,
        rms_norm_eps=1e-6,
        mlp_layer_types=["dense"] * num_layers,
        intermediate_size=8,
        hidden_act="silu",
    )


def _stub_ctors(monkeypatch, mod, cls):
    monkeypatch.setattr(mod, "HYV4MLAAttention", lambda *a, **k: _Attn(3.0))
    monkeypatch.setattr(mod, "RMSNorm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(mod, "HYV4HCLayer", _HC)
    monkeypatch.setattr(mod, "HYV4FeedForward", lambda *a, **k: Const(5.0))
    if "feed_forward_cls" in vars(mod.HYV4DecoderLayer):
        monkeypatch.setattr(
            cls, "feed_forward_cls", staticmethod(lambda *a, **k: Const(5.0))
        )


def _built_layer(monkeypatch, mod, *, layer_idx=1, ihc=True, steering=True, cls=None):
    cls = cls or mod.HYV4DecoderLayer
    _stub_ctors(monkeypatch, mod, cls)
    return cls(
        config=_hf_config(ihc=ihc),
        vllm_config=_vllm_config(steering=steering),
        prefix=f"model.layers.{layer_idx}",
    )


def _has(layer, hook):
    return hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)])


def _table(layer, hook):
    return getattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)])


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_init_registers_ihc_hook_set(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, layer_idx=1)
    assert layer.layer_idx == 1
    for hook in ("pre_attn", "post_attn", "mlp_in", "mlp_out"):
        assert _table(layer, hook).shape[-1] == H, hook
    for hook in ("mhc_streams_pre_attn", "mhc_streams_pre_mlp"):
        assert _table(layer, hook).shape[-1] == N * H, hook
    assert not _has(layer, "post_block")
    assert not _has(layer, "mhc_streams_final")


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_init_last_layer_owns_final_table(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, layer_idx=2)
    assert _table(layer, "mhc_streams_final").shape[-1] == N * H


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_init_registers_standard_hooks_without_ihc(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, layer_idx=2, ihc=False)
    for hook in ALL_HOOKS:
        assert _table(layer, hook).shape[-1] == H, hook


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
@pytest.mark.parametrize("ihc", [True, False])
def test_init_without_steering_registers_nothing(monkeypatch, mod, ihc):
    layer = _built_layer(monkeypatch, mod, ihc=ihc, steering=False)
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize(
    "cls",
    [hy_mtp.HYV4MTPDecoderLayer, hy_dfly.HYV4DFlyDecoderLayer],
    ids=["mtp", "dfly"],
)
def test_draft_blocks_register_nothing(monkeypatch, cls):
    layer = _built_layer(monkeypatch, hy_nv, layer_idx=0, ihc=False, cls=cls)
    assert not hasattr(layer, "steering_index")


# ---------------------------------------------------------------------------
# Real steering op on CPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
@pytest.mark.parametrize("hook", ["mhc_streams_pre_attn", "mhc_streams_pre_mlp"])
def test_channel_vector_shifts_layer_output(monkeypatch, mod, hook):
    layer = _built_layer(monkeypatch, mod, layer_idx=1)
    h = torch.ones(T, N, H)
    base = layer(_pos(), h.clone(), None)[0]

    vec = torch.arange(N * H, dtype=torch.float32) - 3.0
    set_global_vector(layer, hook, vec, num_tokens=T)
    got = layer(_pos(), h.clone(), None)[0]
    assert torch.allclose(got - base, vec.view(N, H).expand(T, N, H))

    set_global_vector(layer, hook, torch.zeros(N * H), num_tokens=T)
    assert torch.equal(layer(_pos(), h.clone(), None)[0], base)


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_final_channel_vector_shifts_model_output(monkeypatch, mod):
    _pp(monkeypatch, mod)
    first = _built_layer(monkeypatch, mod, layer_idx=1)
    last = _built_layer(monkeypatch, mod, layer_idx=2)
    model = _bare_model(mod, [first, last])
    emb = torch.ones(T, H)
    base = model(None, _pos(), None, inputs_embeds=emb.clone())

    vec = torch.arange(N * H, dtype=torch.float32)
    set_global_vector(last, "mhc_streams_final", vec, num_tokens=T)
    got = model(None, _pos(), None, inputs_embeds=emb.clone())
    assert torch.allclose(got - base, vec.view(N, H).mean(0).expand(T, H))

    set_global_vector(last, "mhc_streams_final", torch.zeros(N * H), num_tokens=T)
    assert torch.equal(model(None, _pos(), None, inputs_embeds=emb.clone()), base)


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_plain_residual_vector_shifts_block_output(monkeypatch, mod, hook):
    layer = _built_layer(monkeypatch, mod, layer_idx=1, ihc=False)
    h, r = torch.ones(T, H), torch.full((T, H), 2.0)
    base = sum(layer(_pos(), h.clone(), r.clone()))

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=T)
    got = sum(layer(_pos(), h.clone(), r.clone()))
    assert torch.allclose(got - base, vec.expand(T, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=T)
    assert torch.equal(sum(layer(_pos(), h.clone(), r.clone())), base)


# ---------------------------------------------------------------------------
# Model wiring and fused paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mod", PLATFORMS, ids=["nvidia", "amd"])
def test_index_shared_across_layers(mod):
    assert calls((mod.HYV4Model, "__init__"), "share_steering_index_across_layers")


def _sp_config(*, steering):
    cfg = _vllm_config(steering=steering)
    cfg.parallel_config = SimpleNamespace(
        tensor_parallel_size=2,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        all2all_backend="allgather_reducescatter",
    )
    cfg.lora_config = None
    cfg.speculative_config = None
    return cfg


def test_ihc_sequence_parallel_off_under_interventions(monkeypatch):
    monkeypatch.setattr(hy_nv.envs, "VLLM_BATCH_INVARIANT", False)
    assert hy_nv._use_ihc_sequence_parallel(
        _sp_config(steering=False), enable_ihc=True, supported=True
    )
    assert not hy_nv._use_ihc_sequence_parallel(
        _sp_config(steering=True), enable_ihc=True, supported=True
    )
    assert calls((hy_nv.HYV4Model, "__init__"), "_use_ihc_sequence_parallel")


def test_vl_wrapper_builds_the_hooked_language_model():
    assert hy_vl.HYV4VLForConditionalGeneration.language_model_architecture == (
        "HYV4ForCausalLM"
    )
    src = inspect.getsource(hy_vl.HYV4VLForConditionalGeneration.__init__)
    assert "init_vllm_registered_model" in src
    assert "self.language_model_architecture" in src
