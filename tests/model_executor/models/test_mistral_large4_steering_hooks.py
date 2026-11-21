# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4's decoder layer emits all five intervention hooks.

The model loop runs ``layer.input_norm`` (which folds the previous layer's
MLP branch into the residual) and then ``layer.forward``; ``pre_attn`` is the
residual that ``input_norm`` returns, ``post_attn`` the residual after the
fused add in ``post_attention_layernorm``, and ``post_block`` that residual
plus the MLP branch. Both the nvidia and the amd model files mirror this
layer, so every test runs on both.

The ``mistral_large4`` package ``__init__`` refuses to import off-GPU, so the
platform modules are loaded under a bare package stub when it does.
"""

import importlib
import pathlib
import sys
import types
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
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)

_PKG = "vllm.models.mistral.mistral_large4"


def _load(platform: str) -> types.ModuleType:
    try:
        importlib.import_module(_PKG)
    except NotImplementedError:
        import vllm.models.mistral as mistral

        stub = types.ModuleType(_PKG)
        stub.__path__ = [str(pathlib.Path(mistral.__file__).parent / "mistral_large4")]
        sys.modules[_PKG] = stub
    return importlib.import_module(f"{_PKG}.{platform}.model")


nv = _load("nvidia")
amd = _load("amd")
PLATFORMS = pytest.mark.parametrize("mod", [nv, amd], ids=["nvidia", "amd"])

HIDDEN = 4
NUM_LAYERS = 3


def _layer(mod):
    return bare(
        mod.MistralLarge4DecoderLayer,
        layer_idx=0,
        defer_allreduce=False,
        input_layernorm=Scale(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0),
    )


def _run(layer, h, residual=None):
    """One model-loop step: ``input_norm`` then ``forward``."""
    pos = torch.zeros(h.shape[0], dtype=torch.long)
    hidden, residual = layer.input_norm(h, residual)
    return layer(pos, hidden, residual, pos.float(), None)


@PLATFORMS
def test_hook_order_and_values(monkeypatch, mod):
    rec = record_hooks(monkeypatch, mod)
    out, residual = _run(_layer(mod), torch.ones(2, HIDDEN))

    assert rec.order == ALL_HOOKS
    # pre_attn: the residual stream entering the layer.
    assert torch.equal(rec.seen("pre_attn"), torch.full((2, HIDDEN), 1.0))
    # post_attn: residual (1) + attention branch (3), after the fused add.
    assert torch.equal(rec.seen("post_attn"), torch.full((2, HIDDEN), 4.0))
    # mlp_in: the normed MLP input (2 * 4).
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, HIDDEN), 8.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, HIDDEN), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((2, HIDDEN), 9.0))
    assert torch.equal(out + residual, torch.full((2, HIDDEN), 9.0))


@PLATFORMS
def test_with_incoming_residual(monkeypatch, mod):
    rec = record_hooks(monkeypatch, mod)
    _run(_layer(mod), torch.ones(1, HIDDEN), torch.full((1, HIDDEN), 10.0))
    # input_norm folds the previous layer's branch (1) into the residual (10).
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, HIDDEN), 11.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((1, HIDDEN), 14.0))


def _model_vllm_config(*, steering=True, fuse_allreduce_rms=False, **interventions):
    cfg = fake_vllm_config(steering=steering, dtype=torch.float32)
    for name, value in interventions.items():
        setattr(cfg, name, value)
    cfg.compilation_config = SimpleNamespace(
        pass_config=SimpleNamespace(fuse_allreduce_rms=fuse_allreduce_rms)
    )
    cfg.parallel_config = SimpleNamespace(
        enable_expert_parallel=False, use_sequence_parallel_moe=False
    )
    cfg.quant_config = None
    return cfg


def _built_layer(monkeypatch, mod, vllm_config, idx=1):
    """Run the real ``__init__`` with stubbed sublayer constructors."""
    monkeypatch.setattr(mod, "MistralMLAAttention", lambda *a, **k: Const(3.0))
    monkeypatch.setattr(mod, "MistralMoE", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(mod, "MistralMLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(mod, "RMSNorm", lambda *a, **k: Scale(1.0))
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=NUM_LAYERS,
    )
    return mod.MistralLarge4DecoderLayer(
        vllm_config,
        config,
        SimpleNamespace(is_moe=True, layer_idx=idx),
        prefix=f"model.layers.{idx}",
        topk_indices_buffer=None,
        index_group_builder=None,
        zero_kv_inputs=(torch.empty(0), torch.empty(0)),
        quant_config=None,
    )


@PLATFORMS
def test_init_registers_every_hook(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, _model_vllm_config(), idx=2)
    assert layer.layer_idx == 2
    for hook in ALL_HOOKS:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook


@PLATFORMS
def test_init_without_steering_registers_nothing(monkeypatch, mod):
    layer = _built_layer(monkeypatch, mod, _model_vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


@PLATFORMS
def test_dspark_draft_layer_registers_nothing(monkeypatch, mod):
    """DSpark builds this layer class at indices past its own layer count."""
    layer = _built_layer(monkeypatch, mod, _model_vllm_config(), idx=NUM_LAYERS)
    assert not hasattr(layer, "steering_index")


@PLATFORMS
@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_residual_steering_reaches_block_output(monkeypatch, mod, hook):
    """A residual-stream vector shifts the block output by exactly itself."""
    layer = _built_layer(monkeypatch, mod, _model_vllm_config())
    h = torch.ones(2, HIDDEN)
    base_out, base_res = _run(layer, h.clone())

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=2)
    out, res = _run(layer, h.clone())
    assert torch.allclose((out + res) - (base_out + base_res), vec.expand(2, HIDDEN))

    set_global_vector(layer, hook, torch.zeros(HIDDEN), num_tokens=2)
    out, res = _run(layer, h.clone())
    assert torch.equal(out + res, base_out + base_res)


@PLATFORMS
def test_index_shared_across_layers(mod):
    assert calls(
        (mod.MistralLarge4Model, "__init__"), "share_steering_index_across_layers"
    )


def _defer_config(**kwargs):
    cfg = _model_vllm_config(fuse_allreduce_rms=True, **kwargs)
    cfg.model_config.dtype = torch.bfloat16
    return cfg


@PLATFORMS
def test_deferred_allreduce_kept_without_interventions(mod):
    assert mod.defers_allreduce(_defer_config(steering=False))


@PLATFORMS
@pytest.mark.parametrize(
    "configured",
    [
        {"steering": True},
        {"steering": False, "patch_config": object()},
        {"steering": False, "capture_consumers_config": object()},
    ],
)
def test_deferred_allreduce_off_under_interventions(mod, configured):
    """A deferred all-reduce leaves mlp_out / post_block TP-partial."""
    assert not mod.defers_allreduce(_defer_config(**configured))


def _decode_config(*, steering, backend):
    cfg = _model_vllm_config(steering=steering)
    cfg.kernel_config = SimpleNamespace(monokernel=False)
    cfg.additional_config = {"decode_backend": backend}
    return cfg


def test_amd_specialized_decode_refused_under_interventions(monkeypatch):
    monkeypatch.setattr(amd, "specialized_decode_config", lambda vc: "decode")
    with pytest.raises(ValueError, match="intervention"):
        amd._specialized_decode_config(
            _decode_config(steering=True, backend="latency_specialized")
        )


def test_amd_specialized_decode_auto_under_interventions(monkeypatch):
    monkeypatch.setattr(amd, "specialized_decode_config", lambda vc: "decode")
    cfg = _decode_config(steering=True, backend="auto")
    assert amd._specialized_decode_config(cfg) is None


def test_amd_specialized_decode_kept_without_interventions(monkeypatch):
    monkeypatch.setattr(amd, "specialized_decode_config", lambda vc: "decode")
    cfg = _decode_config(steering=False, backend="latency_specialized")
    assert amd._specialized_decode_config(cfg) == "decode"


def test_amd_model_uses_guarded_specialized_decode():
    assert calls((amd.MistralLarge4Model, "__init__"), "_specialized_decode_config")
