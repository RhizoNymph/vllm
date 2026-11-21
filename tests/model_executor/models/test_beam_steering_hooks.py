# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Beam's sandwich-norm, depth-scaled decoder layer emits all five hooks.

Beam carries the full residual stream between layers (no deferred add) and
scales it at every sublayer join: ``h = x_scale * post_norm(branch) +
r_scale * h``. Its fused boundary kernels also normalize the *next*
sublayer's input in the same launch, and on NVIDIA an MNNVL all-reduce is
folded into that boundary. None of those leave a hookable tensor, so with any
intervention configured the layer takes an unfused path:

* pre_attn: the residual stream entering the layer;
* post_attn: the stream after the depth-scaled attention join;
* mlp_in: the pre-feedforward-normed MLP/MoE input;
* mlp_out: the depth-scaled, post-FFN-normed branch (``x_scale * norm(mlp)``);
* post_block: ``r_scale * post_attn + mlp_out``, the stream leaving the layer.

Both platform files are tested (the Triton kernels are swapped for a torch
reference so the layers run on CPU).
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
from vllm.models.beam.amd import model as amd_model
from vllm.models.beam.common import interventions as beam_interventions
from vllm.models.beam.nvidia import model as nvidia_model

PLATFORMS = [
    pytest.param(nvidia_model, id="nvidia"),
    pytest.param(amd_model, id="amd"),
]


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def _ref_boundary(x, residual, post_w, post_eps, next_w, next_eps, scales, **_):
    x_scale, r_scale = scales
    h = _ref_rms_norm(x, post_w, post_eps) * x_scale + residual * r_scale
    y = _ref_rms_norm(h, next_w, next_eps) if next_w is not None else None
    return h, y


@pytest.fixture
def cpu_kernels(monkeypatch):
    for mod in (nvidia_model, amd_model):
        monkeypatch.setattr(mod, "beam_rms_norm", _ref_rms_norm)
        monkeypatch.setattr(mod, "beam_norm_residual_norm", _ref_boundary)
    if hasattr(nvidia_model, "beam_allreduce_norm_residual_norm"):
        monkeypatch.setattr(
            nvidia_model, "beam_allreduce_norm_residual_norm", _ref_boundary
        )


def _norm(weight: float, n: int = 4) -> SimpleNamespace:
    return SimpleNamespace(weight=torch.full((n,), weight))


def _stub_layer(mod, steerable: bool = True):
    """Weights/consts chosen so every hook value is a small exact number.

    RMS-norm of a constant row is the norm weight, so with eps=0:
    attn branch = 2 (post-attn weight) * 0.5 = 1, post_attn = 1 + 2 * 1 = 3,
    mlp_in = 1, mlp branch = 3 (post-FFN weight) * 2 = 6, post_block = 3 + 6.
    """
    layer = bare(
        mod.BeamDecoderLayer,
        layer_idx=0,
        layer_id=0,
        steerable=steerable,
        rms_norm_eps=0.0,
        self_attn=Const(7.0),
        mlp=Const(5.0),
        attn_scales=(0.5, 2.0),
        mlp_scales=(2.0, 1.0),
    )
    # Plain attributes (not Modules), so set them past nn.Module.__setattr__.
    for name, w in (
        ("input_layernorm", 1.0),
        ("post_attention_layernorm", 2.0),
        ("pre_feedforward_layernorm", 1.0),
        ("post_feedforward_layernorm", 3.0),
    ):
        object.__setattr__(layer, name, _norm(w))
    if not steerable:
        layer.attention_boundary = _ref_boundary
        layer.mlp_boundary = _ref_boundary
        layer.finalize_boundary = None
    return layer


@pytest.mark.parametrize("mod", PLATFORMS)
def test_hook_order_and_values(monkeypatch, cpu_kernels, mod):
    rec = record_hooks(monkeypatch, mod, beam_interventions)
    layer = _stub_layer(mod)
    h = torch.ones(2, 4)
    out, normalized = layer(torch.zeros(2, dtype=torch.long), h, None, None)

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, 4), 1.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((2, 4), 3.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, 4), 1.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, 4), 6.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, 4), 9.0))
    assert torch.equal(out, torch.full((2, 4), 9.0))
    # The next layer re-normalizes the (possibly steered) stream itself.
    assert normalized is None


@pytest.mark.parametrize("mod", PLATFORMS)
def test_steerable_path_matches_fused_path(monkeypatch, cpu_kernels, mod):
    """With pass-through hooks the unfused path computes the fused result."""
    record_hooks(monkeypatch, mod, beam_interventions)
    pos = torch.zeros(3, dtype=torch.long)
    h = torch.randn(3, 4)
    steer_out, _ = _stub_layer(mod, steerable=True)(pos, h, None, None)
    fused_out, _ = _stub_layer(mod, steerable=False)(pos, h, None, None)
    assert torch.allclose(steer_out, fused_out)


@pytest.mark.parametrize("mod", PLATFORMS)
def test_fused_path_untouched_without_interventions(monkeypatch, cpu_kernels, mod):
    rec = record_hooks(monkeypatch, mod, beam_interventions)
    layer = _stub_layer(mod, steerable=False)
    next_w = torch.full((4,), 2.0)
    _, normalized = layer(
        torch.zeros(2, dtype=torch.long), torch.ones(2, 4), None, next_w
    )
    assert rec.order == []
    # The fused boundary still hands the next layer its normalized input.
    assert normalized is not None


class _FakeWorkspace:
    max_tokens = 8


class _StubMLP(Const):
    """A real class (not a lambda) so the layer's ``isinstance`` checks work."""

    def __init__(self, *args, **kwargs):
        super().__init__(5.0)


def _built_layer(monkeypatch, mod, vllm_config, *, prefix="model.layers.3", **kw):
    """Run the real ``BeamDecoderLayer.__init__`` with stubbed sublayers."""
    monkeypatch.setattr(mod, "BeamAttention", lambda *a, **k: Const(7.0))
    monkeypatch.setattr(mod, "BeamMLP", _StubMLP)
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        mlp_layer_types=["dense"] * 8,
    )
    vllm_config.model_config.hf_config = config
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    return mod.BeamDecoderLayer(vllm_config=vllm_config, prefix=prefix, **kw)


@pytest.mark.parametrize("mod", PLATFORMS)
def test_init_registers_every_hook(monkeypatch, mod):
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    layer = _built_layer(monkeypatch, mod, fake_vllm_config())
    assert layer.layer_idx == 3
    assert layer.steerable
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


@pytest.mark.parametrize("mod", PLATFORMS)
def test_init_without_steering_registers_nothing(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, fake_vllm_config(steering=False))
    assert layer.layer_idx == 3
    assert not layer.steerable
    assert not hasattr(layer, "steering_index")


def test_nvidia_fused_allreduce_boundaries_kept_without_interventions(monkeypatch):
    layer = _built_layer(
        monkeypatch,
        nvidia_model,
        fake_vllm_config(steering=False),
        allreduce_workspace=_FakeWorkspace(),
    )
    assert layer.attention_boundary is not nvidia_model.beam_norm_residual_norm
    assert layer.mlp_boundary is not nvidia_model.beam_norm_residual_norm


@pytest.mark.parametrize("mod", PLATFORMS)
@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, cpu_kernels, mod, hook):
    """A residual-stream vector rides the depth-scaled stream to the output.

    The stream is rescaled at each join, so a vector added before a join
    reaches the block output times the remaining ``r_scale`` factors.
    """
    layer = _built_layer(monkeypatch, mod, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.randn(2, 4)
    base, _ = layer(pos, h, None, None)

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    factor = {
        "pre_attn": layer.attn_scales[1] * layer.mlp_scales[1],
        "post_attn": layer.mlp_scales[1],
        "post_block": 1.0,
    }[hook]
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, _ = layer(pos, h.clone(), None, None)
    assert torch.allclose(out - base, factor * vec.expand(2, 4), atol=1e-5)

    set_global_vector(layer, hook, torch.zeros(4), num_tokens=2)
    out, _ = layer(pos, h.clone(), None, None)
    assert torch.equal(out, base)


@pytest.mark.parametrize("mod", PLATFORMS)
def test_mlp_out_steering_reaches_block_output(monkeypatch, cpu_kernels, mod):
    layer = _built_layer(monkeypatch, mod, fake_vllm_config())
    pos = torch.zeros(2, dtype=torch.long)
    h = torch.randn(2, 4)
    base, _ = layer(pos, h, None, None)
    vec = torch.tensor([0.25, 1.0, -1.0, 2.0])
    set_global_vector(layer, "mlp_out", vec, num_tokens=2)
    out, _ = layer(pos, h.clone(), None, None)
    assert torch.allclose(out - base, vec.expand(2, 4), atol=1e-5)


# ---------------------------------------------------------------------------
# BeamModel: index sharing and the NVIDIA all-reduce workspace switch
# ---------------------------------------------------------------------------


def _build_model(monkeypatch, mod, vllm_config):
    created: list[object] = []
    seen_workspaces: list[object] = []

    class _Layer(nn.Module):
        def __init__(self, vllm_config, prefix="", allreduce_workspace=None):
            super().__init__()
            seen_workspaces.append(allreduce_workspace)
            self.register_buffer("steering_index", torch.zeros(4, dtype=torch.long))

    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(mod, "get_pp_group", lambda: pp)
    monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mod, "VocabParallelEmbedding", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(mod, "BeamDecoderLayer", _Layer)
    monkeypatch.setattr(
        mod,
        "make_layers",
        lambda n, fn, prefix: (
            0,
            n,
            nn.ModuleList([fn(prefix=f"{prefix}.{i}") for i in range(n)]),
        ),
    )
    if hasattr(mod, "BeamAllReduceWorkspace"):
        monkeypatch.setattr(mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=0))

        def _create(*a, **k):
            ws = _FakeWorkspace()
            created.append(ws)
            return ws

        monkeypatch.setattr(mod.BeamAllReduceWorkspace, "create", _create)

    config = SimpleNamespace(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=3,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
    )
    vllm_config.model_config.hf_config = config
    vllm_config.cache_config = None
    vllm_config.quant_config = None
    vllm_config.scheduler_config.max_num_seqs = 4
    model = mod.BeamModel(vllm_config=vllm_config, prefix="model")
    return model, created, seen_workspaces


@pytest.mark.parametrize("mod", PLATFORMS)
def test_index_shared_across_layers(monkeypatch, mod):
    assert calls((mod.BeamModel, "__init__"), "share_steering_index_across_layers")
    model, _, _ = _build_model(monkeypatch, mod, fake_vllm_config())
    first = model.layers[0].steering_index
    assert all(layer.steering_index is first for layer in model.layers)


def test_nvidia_allreduce_fusion_disabled_under_interventions(monkeypatch):
    model, created, seen = _build_model(monkeypatch, nvidia_model, fake_vllm_config())
    assert created == []
    assert model.allreduce_workspace is None
    assert seen == [None, None, None]


def test_nvidia_allreduce_fusion_kept_without_interventions(monkeypatch):
    model, created, seen = _build_model(
        monkeypatch, nvidia_model, fake_vllm_config(steering=False)
    )
    assert len(created) == 1
    assert model.allreduce_workspace is created[0]
    assert seen == created * 3
