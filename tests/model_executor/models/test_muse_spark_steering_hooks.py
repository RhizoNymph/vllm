# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's decoder layers (NVIDIA and ROCm) emit all five intervention hooks.

Muse Spark blends an FP32 residual stream through a gated connection
(``alpha * stream + beta * post_norm(branch)``) and defers each layer's FFN
blend into the next layer's fused boundary kernel. With an intervention
configured, target layers take the hooked path: hooks on the stream, the
normed FFN input, the post-FFN-normed branch, and the blended layer output,
which the layer returns as a plain tensor. Without one, the deferred fused
path is untouched.
"""

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
from vllm.models.muse_spark.amd import model as amd
from vllm.models.muse_spark.common import interventions as common
from vllm.models.muse_spark.common.layers import (
    MuseSparkNormPreservingResidualConnection,
    MuseSparkResidualConnection,
    MuseSparkResidualStream,
    norm_then_gated_residual,
)
from vllm.models.muse_spark.nvidia import model as nvidia
from vllm.transformers_utils.configs.muse_spark import MuseSparkDSparkConfig

PLATFORMS = [nvidia, amd]
ids = ["nvidia", "amd"]


class _Norm(nn.Module):
    """MuseSparkRMSNorm stub: scales in FP32, honours ``out_dtype``."""

    has_weight = False
    weight = None
    applies_weight_before_norm = False
    variance_size_override = None

    def __init__(self, factor: float = 1.0, *args, **kwargs):
        super().__init__()
        self.factor = factor

    def forward(self, x, residual=None, out_dtype=None):
        return (x.float() * self.factor).to(out_dtype or x.dtype)

    def supports_fused_kernel(self, x):
        return False


class _Attn(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.pre_qkv_proj_norm = _Norm(1.0)
        self.o_proj = SimpleNamespace(reduce_results=True)

    def can_fuse_sp_gather(self, x):
        return False

    def forward(self, positions, hidden_states, **kwargs):
        return torch.full_like(hidden_states, self.value)


class _NvMoE(nvidia.MuseSparkMoE):
    """NVIDIA MoE stub recording the router input it receives."""

    def __init__(self, value: float, wants_router: bool = False):
        nn.Module.__init__(self)
        self.value = value
        self.wants_router = wants_router
        self.use_mega_moe = False
        self.router_seen = None

    def wants_fp32_router_input(self, num_tokens):
        return self.wants_router

    def forward(self, hidden_states, router_input=None, **kwargs):
        self.router_seen = None if router_input is None else router_input.clone()
        return torch.full_like(hidden_states, self.value)


class _AdditiveConnection(MuseSparkResidualConnection):
    """Stands in for the gated connection class: a plain sum."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _connection(gated: bool):
    if not gated:
        return MuseSparkResidualConnection()
    conn = MuseSparkNormPreservingResidualConnection(dim=4)
    conn.gate.data = torch.tensor([0.1, -0.2, 0.3, 0.0])
    return conn


def _layer(platform, *, intervenable=True, ffn=None, gated=False):
    attrs = dict(
        layer_idx=0,
        hidden_size=4,
        intervenable=intervenable,
        self_attn=_Attn(3.0),
        post_attn_norm=_Norm(2.0),
        pre_ffn_norm=_Norm(1.0),
        feed_forward=Const(5.0) if ffn is None else ffn,
        post_ffn_norm=_Norm(3.0),
        add_attn=_connection(gated),
        add_ffn=_connection(gated),
        residual_dtype=torch.float32,
        default_dtype=torch.bfloat16,
    )
    if platform is nvidia:
        attrs.update(
            is_sequence_parallel=False,
            is_sequence_parallel_input=False,
            sp_min_tokens=1,
            all_gather_dual_norm=None,
            all_reduce_dual_norm=None,
        )
    return bare(platform.MuseSparkDecoderLayer, **attrs)


def _run(layer, x, n=2):
    return layer(torch.zeros(n, dtype=torch.long), x)


def _block_output(out):
    if isinstance(out, MuseSparkResidualStream):
        return norm_then_gated_residual(out)
    return out


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_hook_order_and_values(monkeypatch, platform):
    rec = record_hooks(monkeypatch, platform)
    layer = _layer(platform)
    out = _run(layer, torch.ones(2, 4))

    assert rec.order == ALL_HOOKS
    # pre_attn: the FP32 stream entering attention.
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    assert rec.seen("pre_attn").dtype == torch.float32
    # post_attn: stream (1) + post-attn-normed attention branch (3 * 2).
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 7.0))
    assert rec.seen("post_attn").dtype == torch.float32
    # mlp_in: the normed FFN input.
    assert torch.equal(rec.seen("mlp_in").float(), torch.full((2, 4), 7.0))
    # mlp_out: the post-FFN-normed branch (5 * 3) in full precision.
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 15.0))
    assert rec.seen("mlp_out").dtype == torch.float32
    # post_block: the blended layer output, returned as a plain tensor.
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 22.0))
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, torch.full((2, 4), 22.0))


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_deferred_stream_input(monkeypatch, platform):
    """A previous layer's deferred boundary is blended before pre_attn."""
    rec = record_hooks(monkeypatch, platform)
    layer = _layer(platform)
    prev = MuseSparkResidualStream(
        stream=torch.full((1, 4), 10.0),
        branch=torch.ones(1, 4, dtype=torch.bfloat16),
        post_norm=_Norm(1.0),
        connection=MuseSparkResidualConnection(),
    )
    _run(layer, prev, n=1)
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 17.0))


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_hooked_path_matches_fused_path(monkeypatch, platform, gated):
    """The hooked path's output equals the deferred path's blended output."""
    rec = record_hooks(monkeypatch, platform)
    x = torch.randn(3, 4)
    hooked = _run(_layer(platform, gated=gated), x.clone(), n=3)
    fused = _run(_layer(platform, intervenable=False, gated=gated), x.clone(), n=3)
    assert isinstance(fused, MuseSparkResidualStream)
    assert torch.allclose(hooked, _block_output(fused), atol=1e-6)
    # Only the hooked layer emitted hooks.
    assert rec.order == ALL_HOOKS


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_unconfigured_layer_keeps_deferred_path(monkeypatch, platform):
    rec = record_hooks(monkeypatch, platform)
    out = _run(_layer(platform, intervenable=False), torch.ones(2, 4))
    assert isinstance(out, MuseSparkResidualStream)
    assert rec.order == []


# ---------------------------------------------------------------------------
# Real steering op
# ---------------------------------------------------------------------------


def _steerable(platform, gated=False, ffn=None):
    from vllm.models.common.interventions import register_layer_steering_buffers

    layer = _layer(platform, gated=gated, ffn=ffn)
    register_layer_steering_buffers(layer, fake_vllm_config(), 4)
    return layer


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "mlp_out", "post_block"])
@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_residual_steering_reaches_block_output(platform, hook):
    """With an additive connection a stream vector shifts the output by itself."""
    layer = _steerable(platform)
    x = torch.ones(2, 4)
    base = _run(layer, x.clone())

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out = _run(layer, x.clone())
    assert torch.allclose(out - base, vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out = _run(layer, x.clone())
    assert torch.equal(out, base)


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_gated_blend_scales_upstream_edits(platform):
    """Through the gated blend, a post_attn edit arrives scaled by the FFN alpha;
    a post_block edit arrives exactly."""
    layer = _steerable(platform, gated=True)
    x = torch.ones(2, 4)
    base = _run(layer, x.clone())
    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    alpha, _ = layer.add_ffn.coefficients(
        layer.add_ffn.gate.float() / layer.add_ffn.temperature
    )

    set_global_vector(layer, "post_attn", vec, num_tokens=2)
    out = _run(layer, x.clone())
    assert torch.allclose(out - base, (alpha * vec).expand(2, 4), atol=1e-6)
    set_global_vector(layer, "post_attn", torch.zeros(4), num_tokens=2)

    set_global_vector(layer, "post_block", vec, num_tokens=2)
    out = _run(layer, x.clone())
    assert torch.allclose(out - base, vec.expand(2, 4), atol=1e-6)


def test_nvidia_mlp_in_edit_reaches_fp32_router_input():
    """The FP32 router input from the fused boundary follows mlp_in edits."""
    moe = _NvMoE(5.0, wants_router=True)
    layer = _steerable(nvidia, ffn=moe)
    x = torch.ones(2, 4)
    _run(layer, x.clone())
    assert torch.equal(moe.router_seen, torch.full((2, 4), 7.0))

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, "mlp_in", vec, num_tokens=2)
    _run(layer, x.clone())
    assert moe.router_seen.dtype == torch.float32
    assert torch.allclose(moe.router_seen, (7.0 + vec).expand(2, 4))

    set_global_vector(layer, "mlp_in", torch.zeros(4), num_tokens=2)
    _run(layer, x.clone())
    assert torch.equal(moe.router_seen, torch.full((2, 4), 7.0))


def test_nvidia_sequence_shard_is_gathered_for_hooks(monkeypatch):
    """Under sequence parallelism a hook sees every token, then is re-sharded."""
    rec = record_hooks(monkeypatch, nvidia)
    other_rank = torch.full((2, 4), 100.0)
    monkeypatch.setattr(nvidia, "sp_all_gather", lambda x: torch.cat([x, other_rank]))
    monkeypatch.setattr(nvidia, "sp_shard", lambda x: x[:2])
    layer = _layer(nvidia)
    shard = torch.ones(2, 4)
    out = layer._hook(shard, nvidia.SteeringHookPoint.POST_ATTN, True, 3)
    # The gather is trimmed to the unpadded token count.
    assert rec.seen("post_attn").shape == (3, 4)
    assert torch.equal(out, shard)

    out = layer._hook(shard, nvidia.SteeringHookPoint.POST_ATTN, False, 3)
    assert rec.seen("post_attn", 1).shape == (2, 4)


# ---------------------------------------------------------------------------
# Real __init__
# ---------------------------------------------------------------------------


def _text_config(**extra):
    return SimpleNamespace(
        hidden_size=4,
        max_position_embeddings=64,
        mlp_layer_types=["dense"] * 32,
        rms_norm_eps=1e-6,
        post_norm_eps=1e-6,
        intermediate_size=8,
        residual_gate_temperature=1.0,
        **extra,
    )


def _draft_config():
    cfg = object.__new__(MuseSparkDSparkConfig)
    cfg.__dict__.update(
        hidden_size=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rms_post_norm_eps=None,
        norm_affine=True,
        post_norm_affine=False,
        norm_zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        output_norm_gain_center_type="zero",
        use_ffn_learnable_scales=True,
        intermediate_size_mlp=8,
        residual_dtype=torch.float32,
        use_residual_gate=True,
        moe=SimpleNamespace(is_enabled=False),
    )
    return cfg


def _vllm_config(text_config, steering=True):
    cfg = fake_vllm_config(steering=steering)
    cfg.model_config.hf_text_config = text_config
    cfg.model_config.max_model_len = 64
    cfg.cache_config = None
    cfg.quant_config = None
    return cfg


def _patch_ctors(monkeypatch, platform):
    monkeypatch.setattr(platform, "MuseSparkAttention", lambda *a, **k: _Attn(3.0))
    monkeypatch.setattr(platform, "LlamaMLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(platform, "MuseSparkRMSNorm", _Norm)
    monkeypatch.setattr(platform, "MuseSparkRMSNormWithInputScale", _Norm)
    monkeypatch.setattr(
        platform, "MuseSparkNormPreservingResidualConnection", _AdditiveConnection
    )
    if platform is nvidia:
        monkeypatch.setattr(nvidia, "is_sequence_parallel_enabled", lambda c: False)
        monkeypatch.setattr(nvidia, "get_trtllm_token_threshold", lambda c: 0)
    else:
        monkeypatch.setattr(amd, "token_sharded_stream_supported", lambda c: False)


def _built_layer(monkeypatch, platform, vllm_config, idx=5):
    _patch_ctors(monkeypatch, platform)
    return platform.MuseSparkDecoderLayer(vllm_config, prefix=f"model.layers.{idx}")


@pytest.mark.parametrize("idx", [0, 5, 31])
@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_init_registers_every_hook(monkeypatch, platform, idx):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, platform, _vllm_config(_text_config()), idx)
    assert layer.layer_idx == idx
    assert layer.intervenable is True
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_init_without_steering_registers_nothing(monkeypatch, platform):
    vllm_config = _vllm_config(_text_config(), steering=False)
    layer = _built_layer(monkeypatch, platform, vllm_config)
    assert not hasattr(layer, "steering_index")
    assert layer.intervenable is False


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_draft_layers_never_register(monkeypatch, platform):
    """DSpark drafter layers reuse this class; they must stay unhooked."""
    layer = _built_layer(monkeypatch, platform, _vllm_config(_draft_config()), idx=40)
    assert not hasattr(layer, "steering_index")
    assert layer.intervenable is False


@pytest.mark.parametrize("platform", PLATFORMS, ids=ids)
def test_built_layer_steers(monkeypatch, platform):
    layer = _built_layer(monkeypatch, platform, _vllm_config(_text_config()))
    x = torch.ones(2, 4)
    base = _run(layer, x.clone())
    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, "pre_attn", vec, num_tokens=2)
    assert torch.allclose(_run(layer, x.clone()) - base, vec.expand(2, 4))


# ---------------------------------------------------------------------------
# Model wiring and switched-off paths
# ---------------------------------------------------------------------------


def test_index_shared_across_layers():
    from vllm.model_executor.models.llama import LlamaModel

    for platform in PLATFORMS:
        assert issubclass(platform.MuseSparkModel, LlamaModel)
    assert calls((LlamaModel, "__init__"), "share_steering_index_across_layers")


def test_wire_helper():
    layer = bare(nn.Module, layer_idx=3, hidden_size=4)
    assert common.wire_decoder_layer_interventions(
        layer, fake_vllm_config(), is_draft=False
    )
    assert hasattr(layer, "steering_index")
    draft = bare(nn.Module, layer_idx=3, hidden_size=4)
    assert not common.wire_decoder_layer_interventions(
        draft, fake_vllm_config(), is_draft=True
    )
    assert not hasattr(draft, "steering_index")


def test_refuse_hook_bypassing_backend():
    with pytest.raises(common.MuseSparkInterventionsUnsupportedError):
        common.refuse_hook_bypassing_backend(
            fake_vllm_config(), enabled=True, backend="x"
        )
    common.refuse_hook_bypassing_backend(fake_vllm_config(), enabled=False, backend="x")
    common.refuse_hook_bypassing_backend(
        fake_vllm_config(steering=False), enabled=True, backend="x"
    )


@pytest.mark.parametrize(
    ("platform", "refusal"),
    [(nvidia, "refuse_specialized_decode"), (amd, "refuse_monokernel")],
    ids=ids,
)
def test_whole_layer_kernels_refused_under_interventions(platform, refusal):
    """NVIDIA's latency_specialized decode and ROCm's monokernel bypass the
    Python decoder forward; both are refused when an intervention is set."""
    assert calls((platform.MuseSparkForCausalLM, "__init__"), refusal)


def test_nvidia_specialized_decode_refused(monkeypatch):
    cfg = fake_vllm_config()
    cfg.additional_config = {"decode_backend": "latency_specialized"}
    with pytest.raises(common.MuseSparkInterventionsUnsupportedError):
        nvidia.refuse_specialized_decode(cfg)
    cfg.additional_config = {}
    nvidia.refuse_specialized_decode(cfg)
    off = fake_vllm_config(steering=False)
    off.additional_config = {"decode_backend": "latency_specialized"}
    nvidia.refuse_specialized_decode(off)


def test_amd_monokernel_refused():
    cfg = fake_vllm_config()
    cfg.kernel_config = SimpleNamespace(monokernel=True)
    with pytest.raises(common.MuseSparkInterventionsUnsupportedError):
        amd.refuse_monokernel(cfg)
    cfg.kernel_config = SimpleNamespace(monokernel=False)
    amd.refuse_monokernel(cfg)
    off = fake_vllm_config(steering=False)
    off.kernel_config = SimpleNamespace(monokernel=True)
    amd.refuse_monokernel(off)


def test_nvidia_column_gather_not_configured_under_interventions():
    """Deferred column-sharded unpooler outputs would hand mlp_out a shard."""
    model = bare(nvidia.MuseSparkModel, interventions=True)
    # Returns before touching any layer (the bare model has none).
    model._configure_column_gather()


def test_nvidia_model_sets_interventions_flag():
    assert calls((nvidia.MuseSparkModel, "__init__"), "interventions_configured")


@pytest.mark.parametrize("supported", [False, True])
def test_amd_token_shard_off_under_interventions(monkeypatch, supported):
    monkeypatch.setattr(amd, "token_sharded_stream_supported", lambda c: supported)
    monkeypatch.setattr(amd.envs, "VLLM_ROCM_MUSE_SP_MIN_TOKENS", 64, raising=False)
    assert amd.token_shard_min_tokens(fake_vllm_config()) == 0
    expected = 64 if supported else 0
    assert amd.token_shard_min_tokens(fake_vllm_config(steering=False)) == expected
