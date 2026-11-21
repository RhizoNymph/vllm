# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's mHC decoder layer emits the DeepSeek-V4 mHC hook set.

V4.1 carries the same ``hc_mult``-stream residual as DeepSeek-V4 (with a
*delayed* pre-mix: each sublayer collapses with the previous sublayer's
pre-mix), so it reuses V4's wiring: per sublayer the multi-stream residual
(``mhc_streams_pre_attn`` / ``mhc_streams_pre_mlp``) and the normed
single-stream input (``pre_attn`` / ``mlp_in``), the single-stream branch
outputs (``post_attn`` / ``mlp_out``), and the model-level
``mhc_streams_final`` on the last layer of the last PP rank.

The mHC kernels are TileLang / CuTe DSL, so the layer tests swap them for
small torch stand-ins with the same signatures. The nvidia module imports on
CPU; the amd module does not (``vllm.platforms.rocm`` probes the GPU at
import), so its wiring is pinned by static checks on its source.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.model_executor.models.steering_hook_utils import (
    calls,
    fake_vllm_config,
    record_hooks,
    set_global_vector,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.models.deepseek_v4.common import interventions as v4_interventions
from vllm.models.deepseek_v41.common import interventions as v41_interventions
from vllm.models.deepseek_v41.nvidia import model as nv

HIDDEN = 4
HC = 2
NUM_LAYERS = 4
MHC_ORDER = [
    "mhc_streams_pre_attn",
    "pre_attn",
    "post_attn",
    "mhc_streams_pre_mlp",
    "mlp_in",
    "mlp_out",
]
AMD_MODEL = (
    pathlib.Path(nv.__file__).resolve().parent.parent / "amd" / "model.py"
).resolve()


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    """``attn(positions, x, None)`` -> ``2 x + 1``."""

    def forward(self, positions, x, _):
        return 2 * x + 1


class _Ffn(nn.Module):
    """``ffn(x, input_ids, mega_gate_metadata)`` -> ``3 x``."""

    def forward(self, x, input_ids=None, mega_gate_metadata=None):
        return 3 * x


class _Norm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = eps


def _fake_pre(residual, *args, pre_mix=None, x=None, **kwargs):
    """Stand-in mHC pre: identity mixes; the "normed" input is half the sum."""
    n, hc, _ = residual.shape
    post_mix = torch.ones(n, hc, 1)
    res_mix = torch.eye(hc).expand(n, hc, hc).contiguous()
    return post_mix, res_mix, residual.sum(1) * 0.5, torch.ones(n, hc)


def _fake_shifted_post_pre(x, residual, post_mix, res_mix, *args, **kwargs):
    """Stand-in mHC post (add the branch to every stream), then pre."""
    new_residual = residual + x.unsqueeze(1)
    post, res, layer_input, pre = _fake_pre(new_residual)
    return new_residual, post, res, layer_input, pre, x.new_empty(0)


def _fake_hc_expand(x, hc_mult):
    return x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()


def _patch_kernels(monkeypatch):
    monkeypatch.setattr(nv, "mhc_pre_delayed_tilelang", _fake_pre)
    monkeypatch.setattr(nv, "mhc_shifted_post_pre", _fake_shifted_post_pre)
    monkeypatch.setattr(nv, "hc_expand", _fake_hc_expand)


def _hf_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        hc_mult=HC,
        hc_sinkhorn_iters=1,
        hc_eps=1e-6,
        num_hidden_layers=NUM_LAYERS,
    )


def _vllm_config(*, steering=True, **interventions):
    cfg = fake_vllm_config(steering=steering)
    for name, value in interventions.items():
        setattr(cfg, name, value)
    cfg.model_config.hf_config = _hf_config()
    cfg.kernel_config = SimpleNamespace(enable_jit_warmup=False, moe_backend="auto")
    cfg.parallel_config = SimpleNamespace(
        enable_expert_parallel=False,
        tensor_parallel_size=1,
        data_parallel_size=1,
        use_ubatching=False,
        prefill_context_parallel_size=1,
    )
    cfg.speculative_config = None
    cfg.additional_config = {}
    return cfg


def _built_layer(monkeypatch, vllm_config, idx=1):
    """Run the real nvidia ``__init__`` with stubbed sublayer constructors."""
    _patch_kernels(monkeypatch)
    monkeypatch.setattr(nv, "_select_dsv4_attn_cls", lambda vc: lambda *a, **k: _Attn())
    monkeypatch.setattr(nv, "DeepseekV4MoE", lambda *a, **k: _Ffn())
    monkeypatch.setattr(nv, "RMSNorm", _Norm)
    monkeypatch.setattr(nv, "use_ampere_mhc", lambda: False)
    monkeypatch.setattr(nv, "_shard_fused_wqa_wkv", lambda vc: False)
    monkeypatch.setattr(nv, "mhc_boundary_weight", lambda *a: None)
    layer = nv.DeepseekV4DecoderLayer(vllm_config, prefix=f"model.layers.{idx}")
    layer.hc_attn_fn_broadcast = torch.empty(0)
    return layer


def _run(layer, h, residual=None):
    pos = torch.zeros(h.shape[0], dtype=torch.long)
    pre_mix = None if residual is None else torch.ones(h.shape[0], HC)
    post = None if residual is None else torch.ones(h.shape[0], HC, 1)
    res = None if residual is None else torch.eye(HC).expand(h.shape[0], HC, HC)
    x, residual, *_ = layer(h, pos, None, pre_mix, post, res, residual)
    return x, residual


def _record_streams(monkeypatch, rec):
    def _streams(module, streams, hook_point):
        rec.calls.append((hook_point.value, streams.detach().clone().flatten(1)))
        return streams

    monkeypatch.setattr(v4_interventions, "apply_layer_steering_streams", _streams)
    monkeypatch.setattr(v41_interventions, "apply_layer_steering_streams", _streams)


# ---------------------------------------------------------------------------
# Layer hooks (nvidia)
# ---------------------------------------------------------------------------


def test_hook_order_and_values_first_layer(monkeypatch):
    layer = _built_layer(monkeypatch, _vllm_config(steering=False), idx=0)
    rec = record_hooks(monkeypatch, nv, v4_interventions)
    _record_streams(monkeypatch, rec)
    x, residual = _run(layer, torch.ones(2, HIDDEN))

    assert rec.order == MHC_ORDER
    # The embedding broadcast to hc copies, then the normed collapsed input.
    assert torch.equal(rec.seen("mhc_streams_pre_attn"), torch.ones(2, HC * HIDDEN))
    assert torch.equal(rec.seen("pre_attn"), torch.ones(2, HIDDEN))
    # The attention branch output (complete: no deferred all-reduce here).
    assert torch.equal(rec.seen("post_attn"), torch.full((2, HIDDEN), 3.0))
    # The streams after the attention post (1 + 3), and the FFN's normed input.
    assert torch.equal(
        rec.seen("mhc_streams_pre_mlp"), torch.full((2, HC * HIDDEN), 4.0)
    )
    assert torch.equal(rec.seen("mlp_in"), torch.full((2, HIDDEN), 4.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((2, HIDDEN), 12.0))
    assert torch.equal(x, torch.full((2, HIDDEN), 12.0))
    assert torch.equal(residual, torch.full((2, HC, HIDDEN), 4.0))


def test_hook_order_with_carried_streams(monkeypatch):
    layer = _built_layer(monkeypatch, _vllm_config(steering=False), idx=2)
    rec = record_hooks(monkeypatch, nv, v4_interventions)
    _record_streams(monkeypatch, rec)
    residual = torch.full((1, HC, HIDDEN), 10.0)
    _run(layer, torch.ones(1, HIDDEN), residual)

    assert rec.order == MHC_ORDER
    # The previous layer's FFN output (1) is posted into the carried streams.
    assert torch.equal(
        rec.seen("mhc_streams_pre_attn"), torch.full((1, HC * HIDDEN), 11.0)
    )
    assert torch.equal(rec.seen("pre_attn"), torch.full((1, HIDDEN), 11.0))


def test_init_registers_mhc_hooks(monkeypatch):
    layer = _built_layer(monkeypatch, _vllm_config(), idx=1)
    assert layer.layer_idx == 1
    for hook in MHC_ORDER:
        assert hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]), hook
    stream_table = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_PRE_ATTN]
    assert getattr(layer, stream_table).shape[-1] == HC * HIDDEN
    # The model-level final hook lives on the last layer only.
    final = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_FINAL]
    assert not hasattr(layer, final)


def test_last_layer_registers_final_hook(monkeypatch):
    layer = _built_layer(monkeypatch, _vllm_config(), idx=NUM_LAYERS - 1)
    final = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_FINAL]
    assert getattr(layer, final).shape[-1] == HC * HIDDEN


def test_init_without_steering_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, _vllm_config(steering=False))
    assert not hasattr(layer, "steering_index")


def test_dspark_draft_layer_registers_nothing(monkeypatch):
    """DSpark builds this layer class at global indices past the target's."""
    layer = _built_layer(monkeypatch, _vllm_config(), idx=NUM_LAYERS)
    assert not hasattr(layer, "steering_index")


@pytest.mark.parametrize(
    ("hook", "width", "residual_shift", "x_shift"),
    [
        # A stream vector rides the carried residual (the attention input was
        # already collapsed from the pre-steering streams).
        ("mhc_streams_pre_attn", HC * HIDDEN, 1.0, None),
        ("mhc_streams_pre_mlp", HC * HIDDEN, 1.0, None),
        # The attention input feeds 2x+1, which posts into every stream.
        ("pre_attn", HIDDEN, 2.0, None),
        ("post_attn", HIDDEN, 1.0, None),
        ("mlp_in", HIDDEN, None, 3.0),
        ("mlp_out", HIDDEN, None, 1.0),
    ],
)
def test_real_steering_shifts_layer_outputs(
    monkeypatch, hook, width, residual_shift, x_shift
):
    layer = _built_layer(monkeypatch, _vllm_config(), idx=1)
    h = torch.ones(2, HIDDEN)
    base_x, base_res = _run(layer, h.clone())

    vec = torch.arange(1.0, width + 1.0)
    set_global_vector(layer, hook, vec, num_tokens=2)
    x, res = _run(layer, h.clone())
    if residual_shift is not None:
        delta = vec.view(-1, HIDDEN) if width == HC * HIDDEN else vec
        assert torch.allclose(res - base_res, residual_shift * delta.expand_as(res))
    if x_shift is not None:
        assert torch.allclose(x - base_x, x_shift * vec.expand(2, HIDDEN))

    set_global_vector(layer, hook, torch.zeros(width), num_tokens=2)
    x, res = _run(layer, h.clone())
    assert torch.equal(x, base_x)
    assert torch.equal(res, base_res)


# ---------------------------------------------------------------------------
# Model-level wiring (nvidia)
# ---------------------------------------------------------------------------


def test_index_shared_across_layers():
    assert calls((nv.DeepseekV4Model, "__init__"), "share_steering_index_across_layers")


def test_model_forward_steers_final_streams():
    assert calls((nv.DeepseekV4Model, "forward"), "steer_final_streams")


def test_final_streams_hook_on_last_layer(monkeypatch):
    last = _built_layer(monkeypatch, _vllm_config(), idx=NUM_LAYERS - 1)
    layers = [nn.Identity()] * (NUM_LAYERS - 1) + [last]
    streams = torch.ones(2, HC, HIDDEN)
    vec = torch.arange(1.0, HC * HIDDEN + 1.0)
    set_global_vector(last, "mhc_streams_final", vec, num_tokens=2)
    out = v41_interventions.steer_final_streams(layers, 0, NUM_LAYERS, streams)
    assert torch.allclose(out - streams, vec.view(HC, HIDDEN).expand_as(streams))


def test_final_streams_noop_without_local_layers():
    streams = torch.ones(2, HC, HIDDEN)
    assert v41_interventions.steer_final_streams([], 3, 3, streams) is streams


def test_final_streams_unregistered_is_identity(monkeypatch):
    last = _built_layer(monkeypatch, _vllm_config(steering=False), idx=NUM_LAYERS - 1)
    streams = torch.ones(2, HC, HIDDEN)
    out = v41_interventions.steer_final_streams([last], 0, 1, streams)
    assert out is streams


# ---------------------------------------------------------------------------
# Fused / specialized paths off under interventions (nvidia)
# ---------------------------------------------------------------------------

_CONFIGURED = [
    {"steering": True},
    {"steering": False, "patch_config": object()},
    {"steering": False, "capture_consumers_config": object()},
]


def _with(**kwargs):
    steering = kwargs.pop("steering", False)
    return _vllm_config(steering=steering, **kwargs)


@pytest.mark.parametrize("configured", _CONFIGURED)
def test_mhc_overlap_off_under_interventions(monkeypatch, configured):
    """The side-stream coefficients (and the all-reduce fused with them)."""
    monkeypatch.setattr(nv, "supports_mhc_overlap", lambda vc: True)
    monkeypatch.setattr(nv, "supports_ampere_mhc_overlap", lambda vc: True)
    assert nv._mhc_overlap_support(_with(**configured)) == (False, False)


def test_mhc_overlap_kept_without_interventions(monkeypatch):
    monkeypatch.setattr(nv, "supports_mhc_overlap", lambda vc: True)
    monkeypatch.setattr(nv, "supports_ampere_mhc_overlap", lambda vc: False)
    assert nv._mhc_overlap_support(_vllm_config(steering=False)) == (True, False)


def _sp_config(*, steering, moe_backend="auto"):
    cfg = _vllm_config(steering=steering)
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.tensor_parallel_size = 2
    cfg.parallel_config.data_parallel_size = 2
    cfg.kernel_config.moe_backend = moe_backend
    return cfg


def test_sequence_parallel_kept_without_interventions():
    assert nv._use_sequence_parallel(_sp_config(steering=False))


def test_sequence_parallel_off_under_interventions():
    """Token-sharded hook tensors would not line up with steering_index."""
    assert not nv._use_sequence_parallel(_sp_config(steering=True))


def test_mega_moe_sequence_parallel_refused_under_interventions():
    backend = sorted(nv.MEGA_MOE_BACKENDS)[0]
    assert nv._use_sequence_parallel(_sp_config(steering=False, moe_backend=backend))
    with pytest.raises(NotImplementedError, match="intervention"):
        nv._use_sequence_parallel(_sp_config(steering=True, moe_backend=backend))


def _replay_model(vllm_config):
    model = object.__new__(nv.DeepseekV4Model)
    nn.Module.__init__(model)
    model.start_layer, model.end_layer = 0, NUM_LAYERS
    model.use_sequence_parallel = False
    model.config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS, sliding_window=8, engram_layer_ids=()
    )
    return model


def test_decoder_replay_off_under_interventions():
    """The replay layers run only each request's trailing window of rows."""
    model = _replay_model(_vllm_config())
    assert not model._decoder_replay_supported(_vllm_config(), 1)


def test_decoder_replay_kept_without_interventions():
    cfg = _vllm_config(steering=False)
    assert _replay_model(cfg)._decoder_replay_supported(cfg, 1)


def test_specialized_decode_refused_under_interventions(monkeypatch):
    monkeypatch.setattr(nv, "specialized_decode_config", lambda vc: "decode")
    cfg = _vllm_config()
    cfg.additional_config = {"decode_backend": "latency_specialized"}
    with pytest.raises(ValueError, match="intervention"):
        nv._specialized_decode_config(cfg)


def test_specialized_decode_auto_under_interventions(monkeypatch):
    monkeypatch.setattr(nv, "specialized_decode_config", lambda vc: "decode")
    assert nv._specialized_decode_config(_vllm_config()) is None


def test_specialized_decode_kept_without_interventions(monkeypatch):
    monkeypatch.setattr(nv, "specialized_decode_config", lambda vc: "decode")
    cfg = _vllm_config(steering=False)
    cfg.additional_config = {"decode_backend": "latency_specialized"}
    assert nv._specialized_decode_config(cfg) == "decode"


def test_causal_lm_uses_guarded_specialized_decode():
    assert calls(
        (nv.DeepseekV41LLMForCausalLM, "__init__"), "_specialized_decode_config"
    )


# ---------------------------------------------------------------------------
# amd: static checks (the module cannot be imported on a CPU host)
# ---------------------------------------------------------------------------


def _amd_method(cls_name: str, method: str) -> ast.FunctionDef:
    tree = ast.parse(AMD_MODEL.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )


def _amd_calls(cls_name: str, method: str) -> list[str]:
    out = []
    for node in ast.walk(_amd_method(cls_name, method)):
        if isinstance(node, ast.Call):
            f = node.func
            out.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return out


def _amd_hook_points(cls_name: str, method: str) -> list[str]:
    """``SteeringHookPoint.X`` / sublayer names passed to the hook helpers."""
    out = []
    for node in ast.walk(_amd_method(cls_name, method)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", ""))
        if name == "apply_layer_steering":
            out.append(node.args[2].attr)
        elif name == "steer_and_capture_mhc":
            out.append(node.args[-1].value)
    return out


def test_amd_layer_registers_buffers():
    assert "register_v41_steering_buffers" in _amd_calls(
        "DeepseekV4DecoderLayer", "__init__"
    )


def test_amd_layer_hooks_every_site():
    assert sorted(_amd_hook_points("DeepseekV4DecoderLayer", "forward")) == sorted(
        ["attn", "POST_ATTN", "ffn", "MLP_OUT"]
    )


def test_amd_model_shares_index_and_steers_final():
    assert "share_steering_index_across_layers" in _amd_calls(
        "DeepseekV4Model", "__init__"
    )
    assert "steer_final_streams" in _amd_calls("DeepseekV4Model", "forward")


def test_amd_fused_paths_gated_on_interventions():
    """Overlap stream and decoder replay (model), mono decode layer (layer)."""
    assert "interventions_configured" in _amd_calls("DeepseekV4Model", "__init__")
    assert "interventions_configured" in _amd_calls(
        "DeepseekV4Model", "_decoder_replay_supported"
    )
    assert "interventions_configured" in _amd_calls(
        "DeepseekV4DecoderLayer", "__init__"
    )
