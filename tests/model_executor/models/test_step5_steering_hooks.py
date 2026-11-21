# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5's decoder layers emit all five intervention hooks on both platforms.

NVIDIA (``vllm/models/step5/nvidia/model.py``) defers each branch add into the
next fused add-norm, like Qwen3, with an optional FP32 residual stream. Its
sequence-parallel MegaMoE path (EP + TP>1 + ``deep_gemm_mega_moe``) shards the
residual stream across TP ranks between layers, so with any intervention
configured sequence parallelism is switched off (and the then TP-sharded
MegaMoE shared expert reduces its own output).

ROCm (``vllm/models/step5/amd/model.py``) carries the full FP32 stream and adds
each branch in the layer.

The MTP drafter reuses ``Step5DecoderLayer`` on both platforms; it is built
with ``interventions=False`` so it registers no buffers.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest import mock

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
from vllm.models.step5.nvidia import model as nv_model
from vllm.models.step5.nvidia import mtp as nv_mtp

# ``vllm.platforms.rocm`` probes the GPU arch at import; give it one on CPU.
with mock.patch.object(
    torch.cuda,
    "get_device_properties",
    lambda *a, **k: SimpleNamespace(gcnArchName="gfx942"),
):
    from vllm.models.step5.amd import model as amd_model
    from vllm.models.step5.amd import mtp as amd_mtp


def _hook_tables():
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    return [HOOK_POINT_TABLE_ATTR[SteeringHookPoint(h)] for h in ALL_HOOKS]


def _decoder_layer_kwargs(cls) -> dict[str, ast.expr]:
    """Keyword args of the ``Step5DecoderLayer(...)`` call in ``cls.__init__``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Step5DecoderLayer"
        ):
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError(f"{cls.__name__} does not build a Step5DecoderLayer")


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------


class _FusedNorm(nn.Module):
    """Stand-in for ``Step5RMSNorm``: the FP32-residual fused add-norm."""

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, x, residual=None, *, out_dtype=None):
        if residual is None:
            return (x * self.factor).to(out_dtype)
        s = x.float() + residual.float()
        return (s * self.factor).to(out_dtype), s


def _nv_stub_layer(*, fp32_residual: bool = False, norm=Scale):
    return bare(
        nv_model.Step5DecoderLayer,
        layer_idx=0,
        fp32_residual_connection=fp32_residual,
        use_sequence_parallel=False,
        use_moe=False,
        input_layernorm=norm(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=norm(2.0),
        mlp=Const(5.0),
    )


def test_nvidia_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, nv_model)
    layer = _nv_stub_layer()
    out, residual = layer(torch.zeros(2, dtype=torch.long), torch.ones(2, 4))

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    # residual (1) + attention branch (3).
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 4.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 8.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 9.0))
    assert torch.equal(out + residual, torch.full((2, 4), 9.0))


def test_nvidia_with_incoming_residual(monkeypatch):
    rec = record_hooks(monkeypatch, nv_model)
    layer = _nv_stub_layer()
    layer(torch.zeros(1, dtype=torch.long), torch.ones(1, 4), torch.full((1, 4), 10.0))
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, 4), 14.0))


def test_nvidia_fp32_residual_path(monkeypatch):
    """The FP32 residual (fused add-norm) branch hooks the FP32 stream."""
    monkeypatch.setattr(nv_model, "Step5RMSNorm", _FusedNorm)
    rec = record_hooks(monkeypatch, nv_model)
    layer = _nv_stub_layer(fp32_residual=True, norm=_FusedNorm)
    h = torch.ones(2, 4, dtype=torch.float32)
    r = torch.full((2, 4), 10.0, dtype=torch.float32)
    out, residual = layer(torch.zeros(2, dtype=torch.long), h, r)

    assert rec.order == ALL_HOOKS
    assert rec.seen("pre_attn").dtype == torch.float32
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 14.0))
    assert rec.seen("mlp_in").dtype == torch.bfloat16
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 19.0))


def _nv_vllm_config(steering=True, *, ep=False, tp=1, backend="auto"):
    vc = fake_vllm_config(steering=steering)
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        fp32_residual_connection=False,
        sliding_window=None,
        att_impl_type="GQA",
        attention_other_setting=None,
        partial_rotary_factors=[],
        max_position_embeddings=64,
        num_attention_heads=2,
        num_attention_groups=1,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        zero_centered=True,
        num_hidden_layers=3,
        vocab_size=16,
        moe_num_experts=4,
        tie_word_embeddings=False,
    )
    vc.model_config.hf_config = config
    vc.model_config.hf_text_config = config
    vc.model_config.max_model_len = 64
    vc.cache_config = None
    vc.quant_config = None
    vc.parallel_config = SimpleNamespace(
        enable_expert_parallel=ep, tensor_parallel_size=tp
    )
    vc.kernel_config = SimpleNamespace(moe_backend=backend)
    return vc


def _nv_built_layer(monkeypatch, vllm_config, prefix="model.layers.2", **kw):
    """Run the real ``Step5DecoderLayer.__init__`` with stubbed sublayers."""
    monkeypatch.setattr(nv_model, "Step5Attention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(nv_model, "Step5MLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(nv_model, "make_step5_rms_norm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(nv_model, "get_step5_norm_dtype", lambda c: None)
    monkeypatch.setattr(nv_model, "get_step5_sparse_config", lambda c: None)
    monkeypatch.setattr(nv_model, "checkpoint_has_step5_sparse_config", lambda c: True)
    monkeypatch.setattr(nv_model, "_get_step5_layer_types", lambda c: [])
    monkeypatch.setattr(nv_model, "_get_step5_moe_layer_indices", lambda c: set())
    monkeypatch.setattr(nv_model, "get_tp_group", lambda: None)
    monkeypatch.setattr(nv_model, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(nv_model, "get_tensor_model_parallel_world_size", lambda: 1)
    return nv_model.Step5DecoderLayer(vllm_config, prefix=prefix, **kw)


def test_nvidia_init_registers_every_hook(monkeypatch):
    layer = _nv_built_layer(monkeypatch, _nv_vllm_config())
    assert layer.layer_idx == 2
    for attr in _hook_tables():
        assert hasattr(layer, attr), attr


def test_nvidia_init_without_steering_registers_nothing(monkeypatch):
    layer = _nv_built_layer(monkeypatch, _nv_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


def test_nvidia_drafter_layer_registers_nothing(monkeypatch):
    layer = _nv_built_layer(
        monkeypatch,
        _nv_vllm_config(),
        prefix="model.layers.3.mtp_block",
        interventions=False,
    )
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "mlp_out", "post_block"])
def test_nvidia_residual_steering_reaches_block_output(monkeypatch, hook):
    layer = _nv_built_layer(monkeypatch, _nv_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.randn(2, 4)
    base_out, base_res = layer(pos, h.clone(), None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, res = layer(pos, h.clone(), None)
    assert torch.equal(out + res, base_out + base_res)


def test_nvidia_sequence_parallel_off_under_interventions():
    sp = dict(ep=True, tp=2, backend="deep_gemm_mega_moe")
    assert nv_model._step5_use_sequence_parallel(_nv_vllm_config(False, **sp))
    assert not nv_model._step5_use_sequence_parallel(_nv_vllm_config(True, **sp))


def _nv_built_moe(monkeypatch, vllm_config):
    import vllm.model_executor.layers.fused_moe as fused_moe_pkg
    from vllm.models.step5.nvidia import mega_moe

    shared_kwargs: dict = {}

    def _mlp(*a, **k):
        shared_kwargs.update(k)
        return nn.Identity()

    monkeypatch.setattr(nv_model, "Step5MLP", _mlp)
    monkeypatch.setattr(nv_model, "GateLinear", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(nv_model, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        nv_model,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2)),
    )
    monkeypatch.setattr(nv_model, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(mega_moe, "Step5MegaMoE", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(fused_moe_pkg, "FusedMoEFactory", lambda *a, **k: nn.Identity())
    cfg = vllm_config.model_config.hf_config
    cfg.use_moe_router_bias = True
    cfg.need_fp32_gate = True
    cfg.swiglu_limits = []
    cfg.share_expert_dim = 8
    cfg.moe_top_k = 2
    cfg.moe_intermediate_size = 8
    cfg.norm_expert_weight = True
    cfg.moe_router_scaling_factor = 1.0
    vllm_config.parallel_config.enable_eplb = False
    vllm_config.parallel_config.eplb_config = SimpleNamespace(num_redundant_experts=0)
    moe = nv_model.Step5MoE(vllm_config, prefix="model.layers.1.moe")
    return moe, shared_kwargs


def test_nvidia_mega_moe_shared_expert_reduces_under_interventions(monkeypatch):
    sp = dict(ep=True, tp=2, backend="deep_gemm_mega_moe")
    moe, shared = _nv_built_moe(monkeypatch, _nv_vllm_config(True, **sp))
    assert not moe.use_sequence_parallel
    assert not moe.fuse_all_reduce
    assert shared["sequence_parallel"] is False
    # Every TP rank holds every token, so the TP-sharded shared expert must
    # all-reduce its own output for mlp_out / post_block to be complete.
    assert shared["reduce_results"] is True


def test_nvidia_mega_moe_unchanged_without_interventions(monkeypatch):
    sp = dict(ep=True, tp=2, backend="deep_gemm_mega_moe")
    moe, shared = _nv_built_moe(monkeypatch, _nv_vllm_config(False, **sp))
    assert moe.use_sequence_parallel
    assert not moe.fuse_all_reduce
    assert shared["sequence_parallel"] is True
    assert shared["reduce_results"] is False


def test_nvidia_index_shared_across_layers(monkeypatch):
    assert calls(
        (nv_model.Step5Model, "__init__"), "share_steering_index_across_layers"
    )

    class _Layer(nn.Module):
        def __init__(self, vllm_config, prefix=""):
            super().__init__()
            self.register_buffer("steering_index", torch.zeros(4, dtype=torch.long))

    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(nv_model, "get_pp_group", lambda: pp)
    monkeypatch.setattr(nv_model, "VocabParallelEmbedding", lambda *a: nn.Identity())
    monkeypatch.setattr(nv_model, "Step5DecoderLayer", _Layer)
    monkeypatch.setattr(nv_model, "make_step5_rms_norm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(nv_model, "get_step5_norm_dtype", lambda c: None)
    monkeypatch.setattr(
        nv_model,
        "make_layers",
        lambda n, fn, prefix: (
            0,
            n,
            nn.ModuleList([fn(prefix=f"{prefix}.{i}") for i in range(n)]),
        ),
    )
    model = nv_model.Step5Model(_nv_vllm_config(), prefix="model")
    first = model.layers[0].steering_index
    assert all(layer.steering_index is first for layer in model.layers)


def test_nvidia_mtp_drafter_skips_interventions():
    kwargs = _decoder_layer_kwargs(nv_mtp.Step5MultiTokenPredictorLayer)
    assert isinstance(kwargs.get("interventions"), ast.Constant)
    assert kwargs["interventions"].value is False


# ---------------------------------------------------------------------------
# ROCm
# ---------------------------------------------------------------------------


def _amd_stub_layer():
    return bare(
        amd_model.Step5DecoderLayer,
        layer_idx=0,
        use_moe=False,
        act_dtype=torch.float32,
        input_layernorm=Scale(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0),
    )


def test_amd_hook_order_and_values(monkeypatch):
    rec = record_hooks(monkeypatch, amd_model)
    layer = _amd_stub_layer()
    out = layer(torch.zeros(2, dtype=torch.long), torch.ones(2, 4))

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 4.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 8.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 9.0))
    assert torch.equal(out, torch.full((2, 4), 9.0))


def _amd_built_layer(monkeypatch, vllm_config, prefix="model.layers.4", **kw):
    monkeypatch.setattr(amd_model, "Step5Attention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(amd_model, "Step5MLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(amd_model, "GemmaRMSNorm", lambda *a, **k: Scale(1.0))
    monkeypatch.setattr(amd_model, "_is_moe_layer", lambda c, i: False)
    vllm_config.model_config.hf_config = SimpleNamespace(
        hidden_size=4, intermediate_size=8, rms_norm_eps=1e-6
    )
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    return amd_model.Step5DecoderLayer(vllm_config, prefix=prefix, **kw)


def test_amd_init_registers_every_hook(monkeypatch):
    layer = _amd_built_layer(monkeypatch, fake_vllm_config())
    assert layer.layer_idx == 4
    for attr in _hook_tables():
        assert hasattr(layer, attr), attr


def test_amd_init_without_steering_registers_nothing(monkeypatch):
    layer = _amd_built_layer(monkeypatch, fake_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


def test_amd_drafter_layer_registers_nothing(monkeypatch):
    layer = _amd_built_layer(
        monkeypatch,
        fake_vllm_config(),
        prefix="model.layers.5.mtp_block",
        interventions=False,
    )
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "mlp_out", "post_block"])
def test_amd_residual_steering_reaches_block_output(monkeypatch, hook):
    layer = _amd_built_layer(monkeypatch, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.randn(2, 4)
    base = layer(pos, h.clone())

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out = layer(pos, h.clone())
    assert torch.allclose(out - base, vec.expand(2, 4))

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    assert torch.equal(layer(pos, h.clone()), base)


def test_amd_index_shared_across_layers():
    assert calls(
        (amd_model.Step5Model, "__init__"), "share_steering_index_across_layers"
    )


def test_amd_mtp_drafter_skips_interventions():
    kwargs = _decoder_layer_kwargs(amd_mtp.Step5MultiTokenPredictorLayer)
    assert isinstance(kwargs.get("interventions"), ast.Constant)
    assert kwargs["interventions"].value is False
