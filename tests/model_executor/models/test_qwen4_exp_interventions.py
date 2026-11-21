# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering, patching and capture wiring on the Qwen4Exp decoder.

Qwen4Exp carries ``hc_count`` hyper-connection (HC) streams instead of one
residual. ``vllm/models/qwen4_exp/common/interventions.py`` defines the hook
semantics:

* ``pre_attn`` / ``post_attn`` / ``post_block`` take the stream mean, and an
  edit applies the same delta to every stream;
* ``mlp_in`` / ``mlp_out`` are the single-stream MoE branch;
* ``mhc_streams_*`` are the full multi-stream state.

These tests drive the real NVIDIA and ROCm ``Qwen4ExpDecoderLayer.forward``
and ``Qwen4ExpModel.forward``. The layers are built with ``object.__new__``,
and the HC modules, attention and MoE are deterministic CPU stand-ins that
keep the HC protocol: ``mix`` / ``combine_and_mix`` return
``(streams, block_input, injection_logits)``, and the combine is
``streams + block * 2 * sigmoid(logits / hc_count)``. Capture goes through a
recording capture manager, so the real capture custom ops run.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers.patch import (
    PATCH_ALPHA_ATTR,
    PATCH_ANY_ACTIVE_ATTR,
    PATCH_INDEX_ATTR,
    PATCH_TABLE_ATTR,
    register_patch_buffers,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    register_steering_buffers,
    share_steering_index_across_layers,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.qwen4_exp.amd import model as amd_model
from vllm.models.qwen4_exp.common import interventions as hc_iv
from vllm.models.qwen4_exp.common.hyperconnection import (
    GatedResidual as ReferenceGatedResidual,
)
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia import model as nvidia_model
from vllm.sequence import IntermediateTensors
from vllm.v1.capture.types import (
    build_hook_schema,
    build_model_hook_schema,
    default_hook_schema,
)

HP = SteeringHookPoint
T, HC, H = 3, 4, 8
NUM_LAYERS = 2
LAYER_IDX = 0
ROW = 3  # first per-request row above the reserved rows

PLATFORMS = pytest.mark.parametrize(
    "model_mod", [nvidia_model, amd_model], ids=["nvidia", "amd"]
)

ALL_LAYER_HOOKS = (
    "pre_attn",
    "post_attn",
    "post_block",
    "mlp_in",
    "mlp_out",
    "mhc_streams_pre_attn",
    "mhc_streams_pre_mlp",
)


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


def _combine(streams, block, logits):
    """Reference HC combine (unit weight when ``logits`` is None)."""
    per = streams.unflatten(-1, (HC, H))
    if logits is None:
        weight = torch.ones(per.shape[:-1], dtype=per.dtype)
    else:
        weight = 2.0 * torch.sigmoid(logits / HC)
    out = per + block.unsqueeze(-2) * weight.unsqueeze(-1)
    return out.flatten(-2).to(streams.dtype)


class _StubHC:
    """Deterministic HC module with the NVIDIA/ROCm delayed-combine protocol."""

    def __init__(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.gate = torch.rand(HC, generator=gen) + 0.5
        self.inject = torch.randn(HC * H, HC, generator=gen) * 0.3

    def mix(self, streams):
        per = streams.unflatten(-1, (HC, H))
        block_input = (per * self.gate[:, None].to(per.dtype)).mean(dim=-2)
        logits = streams @ self.inject.to(streams.dtype)
        return streams, block_input, logits

    def combine(self, streams, block, logits):
        return _combine(streams, block, logits)

    def combine_and_mix(self, streams, block, logits):
        return self.mix(self.combine(streams, block, logits))


class _Recorder:
    """A sublayer stand-in that records its input and applies an affine map."""

    def __init__(self, scale: float, shift: float) -> None:
        self.scale, self.shift = scale, shift
        self.inputs: list[torch.Tensor] = []

    def __call__(self, *args, **kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        self.inputs.append(x.clone())
        return x * self.scale + self.shift


class _CaptureRecorder:
    """Capture manager stand-in: records every ``on_hook`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, torch.Tensor]] = []

    def on_hook(self, layer_idx, hook_name, hidden_states):
        self.calls.append((layer_idx, hook_name, hidden_states.clone()))

    def by_hook(self) -> dict[str, tuple[int, torch.Tensor]]:
        out = {}
        for layer_idx, name, tensor in self.calls:
            assert name not in out, f"{name} fired twice"
            out[name] = (layer_idx, tensor)
        return out


@pytest.fixture
def capture():
    rec = _CaptureRecorder()
    cap_mod.set_active_capture_manager(rec)
    yield rec
    cap_mod.set_active_capture_manager(None)


def _make_layer(
    model_mod,
    *,
    layer_idx: int = LAYER_IDX,
    hooked: bool = True,
    steering: bool = True,
    dtype: torch.dtype = torch.float32,
    ple=None,
):
    layer = object.__new__(model_mod.Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        hc_count=HC, hidden_size=H, num_hidden_layers=NUM_LAYERS
    )
    layer.layer_type = "linear_attention"
    layer.layer_idx = layer_idx
    layer.use_hc_sequence_parallel = False
    layer.use_sequence_parallel = False
    layer.ple = ple
    layer.attn_hyper_connection = _StubHC(seed=1)
    layer.mlp_hyper_connection = _StubHC(seed=2)
    layer.linear_attn = _Recorder(scale=2.0, shift=0.25)
    layer.mlp = _Recorder(scale=-1.5, shift=0.5)
    geometry = hc_iv.HCStreamGeometry(hc_count=HC, hidden_size=H)
    layer.hc_interventions = geometry if hooked else None
    if hooked and steering:
        register_steering_buffers(
            layer,
            H,
            max_steering_tokens=T,
            max_steering_configs=1,
            dtype=dtype,
            hook_widths=hc_iv.qwen4_exp_hook_widths(
                geometry, is_last_layer=layer_idx == NUM_LAYERS - 1
            ),
        )
    return layer


def _set_vector(layer, hook: SteeringHookPoint, vec: torch.Tensor) -> None:
    table = getattr(layer, HOOK_POINT_TABLE_ATTR[hook])
    table[ROW] = vec.to(table.dtype)
    layer.steering_index[:T] = ROW
    getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hook]).fill_(True)


def _inputs(dtype=torch.float32, seed: int = 0, pending: bool = False):
    gen = torch.Generator().manual_seed(seed)
    streams = torch.randn(T, HC * H, generator=gen).to(dtype)
    if not pending:
        return streams, None, None
    block = torch.randn(T, H, generator=gen).to(dtype)
    logits = torch.randn(T, HC, generator=gen).to(dtype)
    return streams, block, logits


def _run(layer, streams, block=None, logits=None, ple_inputs=False):
    kwargs = dict(input_ids=None, query_start_loc=None, ngram_context=None)
    if ple_inputs:
        kwargs = dict(
            input_ids=torch.zeros(T, dtype=torch.long),
            query_start_loc=torch.tensor([0, T]),
            ngram_context=torch.zeros(1, 2, dtype=torch.long),
        )
    return layer(streams, block, logits, torch.arange(T), **kwargs)


def _expected_sites(layer, streams, block, logits):
    """Recompute every hook tensor from the stand-ins, with no hooks applied."""
    attn_hc, mlp_hc = layer.attn_hyper_connection, layer.mlp_hyper_connection
    if block is not None:
        streams = attn_hc.combine(streams, block, logits)
    if layer.ple is not None:
        streams = layer.ple(streams, None, None, None)
    pre_attn, attn_in, inj = attn_hc.mix(streams)
    attn_out = attn_in * 2.0 + 0.25
    post_attn, mlp_in, mlp_inj = mlp_hc.combine_and_mix(pre_attn, attn_out, inj)
    mlp_out = mlp_in * -1.5 + 0.5
    block_out = _combine(post_attn, mlp_out, mlp_inj)

    def mean(x):
        return x.unflatten(-1, (HC, H)).mean(dim=-2)

    return {
        "pre_attn": mean(pre_attn),
        "post_attn": mean(post_attn),
        "post_block": mean(block_out),
        "mlp_in": mlp_in,
        "mlp_out": mlp_out,
        "mhc_streams_pre_attn": pre_attn,
        "mhc_streams_pre_mlp": post_attn,
        "_returned": (post_attn, mlp_out, mlp_inj),
        "_block_out": block_out,
        "_attn_in": attn_in,
    }


# ---------------------------------------------------------------------------
# Hook-site wiring
# ---------------------------------------------------------------------------


@PLATFORMS
def test_unhooked_layer_runs_upstream_forward_and_never_captures(model_mod, capture):
    """MTP layers and servers without interventions keep upstream's forward."""
    layer = _make_layer(model_mod, hooked=False)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)

    out = _run(layer, streams, block, logits)

    for got, want in zip(out, expected["_returned"]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert capture.calls == []


@PLATFORMS
@pytest.mark.parametrize("pending", [False, True], ids=["first", "delayed_combine"])
def test_capture_sites(model_mod, capture, pending):
    layer = _make_layer(model_mod)
    streams, block, logits = _inputs(pending=pending)
    expected = _expected_sites(layer, streams, block, logits)

    out = _run(layer, streams, block, logits)

    seen = capture.by_hook()
    assert set(seen) == set(ALL_LAYER_HOOKS)
    for name in ALL_LAYER_HOOKS:
        layer_idx, tensor = seen[name]
        assert layer_idx == LAYER_IDX
        torch.testing.assert_close(tensor, expected[name], msg=name)
    # Capture only reads: the layer's outputs are unchanged.
    for got, want in zip(out, expected["_returned"]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@PLATFORMS
def test_pre_attn_sees_ple_output(model_mod, capture):
    """PLE adds to the streams inside the layer, before the attention mix."""
    ple_shift = torch.linspace(-1.0, 1.0, HC * H)

    def ple(streams, *args):
        return streams + ple_shift

    layer = _make_layer(model_mod, ple=ple)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)

    _run(layer, streams, block, logits, ple_inputs=True)

    seen = capture.by_hook()
    torch.testing.assert_close(seen["pre_attn"][1], expected["pre_attn"])
    torch.testing.assert_close(
        seen["mhc_streams_pre_attn"][1], expected["mhc_streams_pre_attn"]
    )


def test_stream_hook_capture_is_stream_major_full_width(capture):
    layer = _make_layer(nvidia_model)
    streams, _, _ = _inputs()
    hc_iv.intervene_hc_residual(layer, streams, HP.MHC_STREAMS_PRE_ATTN)
    ((_, name, tensor),) = capture.calls
    assert name == "mhc_streams_pre_attn"
    assert tensor.shape == (T, HC * H)
    torch.testing.assert_close(
        tensor.unflatten(-1, (HC, H))[:, 2], streams[:, 2 * H : 3 * H]
    )


# ---------------------------------------------------------------------------
# Steering semantics
# ---------------------------------------------------------------------------


@PLATFORMS
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_registered_but_inactive_or_zero_rows_are_bitwise_noops(model_mod, dtype):
    layer = _make_layer(model_mod, dtype=dtype)
    bare = _make_layer(model_mod, hooked=False)
    streams, block, logits = _inputs(dtype=dtype, pending=True)
    baseline = _run(bare, streams, block, logits)

    inactive = _run(layer, streams, block, logits)
    for hook in HP:
        if hasattr(layer, HOOK_POINT_TABLE_ATTR[hook]):
            width = getattr(layer, HOOK_POINT_TABLE_ATTR[hook]).shape[1]
            _set_vector(layer, hook, torch.zeros(width))
    zero = _run(layer, streams, block, logits)

    for got in (inactive, zero):
        for g, b in zip(got, baseline):
            assert torch.equal(g, b)


@PLATFORMS
def test_pre_attn_vector_is_added_to_every_carried_stream(model_mod, capture):
    layer = _make_layer(model_mod)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)
    vec = torch.randn(H)
    _set_vector(layer, HP.PRE_ATTN, vec)

    out_streams, _, _ = _run(layer, streams, block, logits)

    # The attention mix already read the unsteered streams ...
    torch.testing.assert_close(layer.linear_attn.inputs[-1], expected["_attn_in"])
    # ... and the carried residual gained ``vec`` on every stream.
    shift = (out_streams - expected["mhc_streams_pre_mlp"]).unflatten(-1, (HC, H))
    torch.testing.assert_close(shift, vec.expand(T, HC, H))
    # So the next residual capture moves by exactly ``vec``.
    seen = capture.by_hook()
    torch.testing.assert_close(seen["post_attn"][1], expected["post_attn"] + vec)
    # The pre_attn capture itself is pristine.
    torch.testing.assert_close(seen["pre_attn"][1], expected["pre_attn"])


@PLATFORMS
@pytest.mark.parametrize(
    ("mean_hook", "stream_hook"),
    [
        (HP.PRE_ATTN, HP.MHC_STREAMS_PRE_ATTN),
        (HP.POST_ATTN, HP.MHC_STREAMS_PRE_MLP),
    ],
)
def test_mean_hook_equals_stream_hook_with_repeated_vector(
    model_mod, mean_hook, stream_hook
):
    streams, block, logits = _inputs(pending=True)
    vec = torch.randn(H)

    by_mean = _make_layer(model_mod)
    _set_vector(by_mean, mean_hook, vec)
    by_stream = _make_layer(model_mod)
    _set_vector(by_stream, stream_hook, vec.repeat(HC))

    for a, b in zip(
        _run(by_mean, streams, block, logits), _run(by_stream, streams, block, logits)
    ):
        torch.testing.assert_close(a, b)


@PLATFORMS
def test_stream_hook_steers_one_stream(model_mod):
    layer = _make_layer(model_mod)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)
    row = torch.zeros(HC, H)
    row[1] = torch.randn(H)
    _set_vector(layer, HP.MHC_STREAMS_PRE_MLP, row.flatten())

    out_streams, mlp_out, _ = _run(layer, streams, block, logits)

    shift = (out_streams - expected["mhc_streams_pre_mlp"]).unflatten(-1, (HC, H))
    torch.testing.assert_close(shift, row.expand(T, HC, H))
    # The MoE mix ran before the stream edit, so the branch is untouched.
    torch.testing.assert_close(mlp_out, expected["mlp_out"])


@PLATFORMS
def test_post_block_vector_rides_the_pending_combine(model_mod):
    layer = _make_layer(model_mod)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)
    vec = torch.randn(H)
    _set_vector(layer, HP.POST_BLOCK, vec)

    out_streams, mlp_out, mlp_inj = _run(layer, streams, block, logits)

    torch.testing.assert_close(mlp_out, expected["mlp_out"])
    block_out = _combine(out_streams, mlp_out, mlp_inj)
    shift = (block_out - expected["_block_out"]).unflatten(-1, (HC, H))
    torch.testing.assert_close(shift, vec.expand(T, HC, H))


@PLATFORMS
def test_mlp_branch_hooks_edit_the_moe_input_and_output(model_mod):
    layer = _make_layer(model_mod)
    streams, block, logits = _inputs(pending=True)
    expected = _expected_sites(layer, streams, block, logits)
    vec_in, vec_out = torch.randn(H), torch.randn(H)
    _set_vector(layer, HP.MLP_IN, vec_in)
    _set_vector(layer, HP.MLP_OUT, vec_out)

    out_streams, mlp_out, _ = _run(layer, streams, block, logits)

    torch.testing.assert_close(layer.mlp.inputs[-1], expected["mlp_in"] + vec_in)
    torch.testing.assert_close(
        mlp_out, (expected["mlp_in"] + vec_in) * -1.5 + 0.5 + vec_out
    )
    torch.testing.assert_close(out_streams, expected["mhc_streams_pre_mlp"])


@PLATFORMS
def test_captured_difference_of_means_steers_by_itself(model_mod, capture):
    """A diff-of-means captured at a residual hook is a usable steering vector."""
    layer = _make_layer(model_mod)
    a, _, _ = _inputs(seed=1)
    b, _, _ = _inputs(seed=2)

    _run(layer, a)
    mean_a = capture.by_hook()["post_attn"][1]
    capture.calls.clear()
    _run(layer, b)
    mean_b = capture.by_hook()["post_attn"][1]
    capture.calls.clear()

    # Steer b at post_attn by the difference; post_block then sees b's
    # block output shifted by exactly that difference.
    _set_vector(layer, HP.POST_ATTN, (mean_a - mean_b).mean(dim=0))
    _run(layer, b)
    steered = capture.by_hook()["post_attn"][1]
    torch.testing.assert_close(steered, mean_b)  # capture precedes the edit

    capture.calls.clear()
    plain = _make_layer(model_mod)
    _run(plain, b)
    base_block = capture.by_hook()["post_block"][1]
    capture.calls.clear()
    _run(layer, b)
    torch.testing.assert_close(
        capture.by_hook()["post_block"][1],
        base_block + (mean_a - mean_b).mean(dim=0),
    )


def test_bf16_mean_hook_delta_stays_close_to_the_vector():
    """The stream-mean edit costs one extra bf16 rounding, not more."""
    layer = _make_layer(nvidia_model, dtype=torch.bfloat16)
    streams = (torch.randn(T, HC * H) * 8).to(torch.bfloat16)
    vec = torch.randn(H) * 0.3
    _set_vector(layer, HP.PRE_ATTN, vec)

    out = hc_iv.intervene_hc_residual(layer, streams, HP.PRE_ATTN)

    shift = (out.float() - streams.float()).unflatten(-1, (HC, H))
    # Two bf16 roundings at |x| <= ~32 are each at most 0.125.
    torch.testing.assert_close(
        shift, vec.to(torch.bfloat16).float().expand(T, HC, H), atol=0.26, rtol=0
    )


# ---------------------------------------------------------------------------
# Patching and the final-streams hook
# ---------------------------------------------------------------------------


def test_patch_replaces_the_stream_mean_and_keeps_stream_differences():
    layer = _make_layer(nvidia_model, steering=False)
    register_patch_buffers(
        layer,
        H,
        max_patch_tokens=T,
        max_patch_slots=2,
        dtype=torch.float32,
        hook_widths={HP.PRE_ATTN: H},
    )
    source = torch.randn(H)
    getattr(layer, PATCH_TABLE_ATTR[HP.PRE_ATTN])[1] = source
    getattr(layer, PATCH_ALPHA_ATTR[HP.PRE_ATTN])[1] = 1.0
    getattr(layer, PATCH_INDEX_ATTR[HP.PRE_ATTN])[:T] = 1
    getattr(layer, PATCH_ANY_ACTIVE_ATTR[HP.PRE_ATTN]).fill_(True)
    streams, _, _ = _inputs()

    out = hc_iv.intervene_hc_residual(layer, streams, HP.PRE_ATTN)

    per_out, per_in = out.unflatten(-1, (HC, H)), streams.unflatten(-1, (HC, H))
    torch.testing.assert_close(per_out.mean(dim=-2), source.expand(T, H))
    torch.testing.assert_close(
        per_out - per_out.mean(dim=-2, keepdim=True),
        per_in - per_in.mean(dim=-2, keepdim=True),
    )


def test_final_streams_hook_edits_the_mixer_input_through_the_pending_combine(
    capture,
):
    layer = _make_layer(nvidia_model, layer_idx=NUM_LAYERS - 1)
    streams, block, logits = _inputs(pending=True)
    delta = torch.randn(HC * H)
    _set_vector(layer, HP.MHC_STREAMS_FINAL, delta)

    out = hc_iv.intervene_final_streams(layer, streams, block, logits)

    ((layer_idx, name, tensor),) = capture.calls
    assert (layer_idx, name) == (NUM_LAYERS - 1, "mhc_streams_final")
    torch.testing.assert_close(tensor, _combine(streams, block, logits))
    torch.testing.assert_close(
        _combine(out, block, logits), _combine(streams, block, logits) + delta
    )


def test_final_streams_hook_only_on_last_layer():
    geometry = hc_iv.HCStreamGeometry(hc_count=HC, hidden_size=H)
    assert HP.MHC_STREAMS_FINAL in hc_iv.qwen4_exp_hook_widths(
        geometry, is_last_layer=True
    )
    widths = hc_iv.qwen4_exp_hook_widths(geometry, is_last_layer=False)
    assert widths == {
        HP.PRE_ATTN: H,
        HP.POST_ATTN: H,
        HP.POST_BLOCK: H,
        HP.MLP_IN: H,
        HP.MLP_OUT: H,
        HP.MHC_STREAMS_PRE_ATTN: HC * H,
        HP.MHC_STREAMS_PRE_MLP: HC * H,
    }


def test_view_matches_the_reference_gated_residual_combine():
    """``hc_residual_view`` is the upstream torch reference combine."""
    torch.manual_seed(0)
    ref = ReferenceGatedResidual(
        HyperConnectionConfig(
            hc_count=HC,
            hidden_size=H,
            params_dtype=torch.float32,
            hc_lowrank=4,
            hc_per_branch_norm=True,
        )
    )
    for p in ref.parameters():
        nn.init.normal_(p, std=0.3)
    streams = torch.randn(T, HC * H)
    block = torch.randn(T, H)
    _, residuals = ref.mix(streams)
    logits = F.linear(residuals[1], ref.block_inject_weight.weight)

    view = hc_iv.hc_residual_view(streams, block, logits, HC, reduce_mean=False)
    torch.testing.assert_close(view, ref.combine(block, residuals))
    mean = hc_iv.hc_residual_view(streams, block, logits, HC, reduce_mean=True)
    torch.testing.assert_close(
        mean, ref.combine(block, residuals).unflatten(-1, (HC, H)).mean(dim=-2)
    )


# ---------------------------------------------------------------------------
# Model-level wiring and pipeline parallelism
# ---------------------------------------------------------------------------


class _StubMixer:
    def combine_and_mix(self, streams, block, logits):
        combined = streams if block is None else _combine(streams, block, logits)
        return combined, combined.unflatten(-1, (HC, H)).mean(dim=-2), None


def _make_model(model_mod, monkeypatch, *, is_last_rank: bool):
    """Two-layer model; a non-last rank owns only layer 0 (PP=2)."""
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=is_last_rank),
    )
    model = object.__new__(model_mod.Qwen4ExpModel)
    nn.Module.__init__(model)
    # The ROCm model is @support_torch_compile; run its forward eagerly.
    model.do_not_compile = True
    model.config = SimpleNamespace(hc_count=HC, hidden_size=H)
    model.use_hc_sequence_parallel = False
    model.use_sequence_parallel = False
    model._mtp_hidden_buffer = None
    layer0 = _make_layer(model_mod, layer_idx=0)
    if is_last_rank:
        layers = [layer0, _make_layer(model_mod, layer_idx=1)]
        model.hyper_connection_mixer = _StubMixer()
    else:
        layers = [layer0, PPMissingLayer()]
        model.hyper_connection_mixer = None
    model.layers = nn.ModuleList(layers)
    model.start_layer, model.end_layer = 0, 2 if is_last_rank else 1
    share_steering_index_across_layers(model.layers)
    return model


@PLATFORMS
def test_final_hook_fires_once_on_last_rank(model_mod, monkeypatch, capture):
    model = _make_model(model_mod, monkeypatch, is_last_rank=True)
    embeds = torch.randn(T, H)

    model(None, torch.arange(T), inputs_embeds=embeds)

    finals = [(i, t) for i, n, t in capture.calls if n == "mhc_streams_final"]
    assert len(finals) == 1 and finals[0][0] == NUM_LAYERS - 1
    per_layer = {(i, n) for i, n, _ in capture.calls if n != "mhc_streams_final"}
    assert per_layer == {(i, n) for i in range(NUM_LAYERS) for n in ALL_LAYER_HOOKS}


@PLATFORMS
def test_final_vector_reaches_the_sampled_hidden_state(model_mod, monkeypatch):
    embeds = torch.randn(T, H)
    plain = _make_model(model_mod, monkeypatch, is_last_rank=True)
    base = plain(None, torch.arange(T), inputs_embeds=embeds)

    steered = _make_model(model_mod, monkeypatch, is_last_rank=True)
    delta = torch.randn(HC * H)
    _set_vector(steered.layers[1], HP.MHC_STREAMS_FINAL, delta)
    out = steered(None, torch.arange(T), inputs_embeds=embeds)

    torch.testing.assert_close(out, base + delta.unflatten(-1, (HC, H)).mean(dim=0))


@PLATFORMS
def test_non_last_pp_rank_ships_steered_streams_without_final_hook(
    model_mod, monkeypatch, capture
):
    embeds = torch.randn(T, H)
    plain = _make_model(model_mod, monkeypatch, is_last_rank=False)
    base = plain(None, torch.arange(T), inputs_embeds=embeds)
    capture.calls.clear()

    model = _make_model(model_mod, monkeypatch, is_last_rank=False)
    vec = torch.randn(H)
    _set_vector(model.layers[0], HP.POST_BLOCK, vec)
    out = model(None, torch.arange(T), inputs_embeds=embeds)

    assert isinstance(out, IntermediateTensors)
    assert all(n != "mhc_streams_final" for _, n, _ in capture.calls)
    assert {i for i, _, _ in capture.calls} == {0}
    shift = (out["hidden_states"] - base["hidden_states"]).unflatten(-1, (HC, H))
    torch.testing.assert_close(shift, vec.expand(T, HC, H))


def test_steering_index_is_shared_across_the_ranks_local_layers():
    layers = nn.ModuleList(
        [
            PPMissingLayer(),
            _make_layer(nvidia_model, layer_idx=1),
            _make_layer(nvidia_model, layer_idx=2),
        ]
    )
    share_steering_index_across_layers(layers)
    assert layers[1].steering_index is layers[2].steering_index


# ---------------------------------------------------------------------------
# Registration and configuration
# ---------------------------------------------------------------------------


def _vllm_config(*, steering: bool, hc_sp: bool = False, tp: int = 1):
    return SimpleNamespace(
        steering_config=(
            SimpleNamespace(max_steering_configs=2, max_dynamic_steering_configs=0)
            if steering
            else None
        ),
        patch_config=None,
        capture_consumers_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            enable_hc_sp=hc_sp,
            use_sequence_parallel_moe=False,
        ),
    )


def _init(layer_idx: int, vllm_config):
    layer = nn.Module()
    geometry = hc_iv.init_qwen4_exp_interventions(
        layer,
        vllm_config,
        layer_idx=layer_idx,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=H,
        hc_count=HC,
    )
    return layer, geometry


def _table_widths(layer) -> dict[SteeringHookPoint, int]:
    return {
        hp: getattr(layer, attr).shape[1]
        for hp, attr in HOOK_POINT_TABLE_ATTR.items()
        if hasattr(layer, attr)
    }


def test_init_registers_hook_widths_and_final_only_on_last_layer():
    cfg = _vllm_config(steering=True)
    first, geometry = _init(0, cfg)
    last, _ = _init(NUM_LAYERS - 1, cfg)

    assert geometry == hc_iv.HCStreamGeometry(hc_count=HC, hidden_size=H)
    assert _table_widths(first) == hc_iv.qwen4_exp_hook_widths(
        geometry, is_last_layer=False
    )
    assert _table_widths(last)[HP.MHC_STREAMS_FINAL] == HC * H
    assert first.steering_index.shape == (16,)
    assert getattr(first, HOOK_POINT_TABLE_ATTR[HP.PRE_ATTN]).dtype == torch.bfloat16


def test_init_leaves_mtp_layers_and_unconfigured_servers_unhooked():
    mtp, mtp_geometry = _init(NUM_LAYERS, _vllm_config(steering=True))
    plain, plain_geometry = _init(0, _vllm_config(steering=False))
    assert mtp_geometry is None and plain_geometry is None
    assert _table_widths(mtp) == {} and _table_widths(plain) == {}


def test_init_rejects_hc_sequence_parallel_with_interventions():
    with pytest.raises(hc_iv.Qwen4ExpInterventionConfigError):
        _init(0, _vllm_config(steering=True, hc_sp=True, tp=2))
    # Without interventions HC SP stays available.
    _, geometry = _init(0, _vllm_config(steering=False, hc_sp=True, tp=2))
    assert geometry is None


def test_unknown_hook_is_rejected():
    layer = _make_layer(nvidia_model)
    with pytest.raises(ValueError):
        hc_iv.intervene_hc_residual(layer, _inputs()[0], HP.MLP_IN)
    with pytest.raises(ValueError):
        hc_iv.intervene_branch(layer, torch.zeros(T, H), HP.PRE_ATTN)


# ---------------------------------------------------------------------------
# Capture schema
# ---------------------------------------------------------------------------


def test_capture_schema_for_qwen4_exp_keeps_standard_hooks_and_adds_streams():
    text = SimpleNamespace(model_type="qwen4_exp_text", hc_count=HC)
    schema = build_model_hook_schema(
        H, torch.bfloat16, SimpleNamespace(model_type="qwen4_exp"), text
    )
    for name in ("pre_attn", "post_attn", "post_block", "mlp_in", "mlp_out"):
        assert schema[name].width == H and schema[name].logical_shape == (H,)
    for name in ("mhc_streams_pre_attn", "mhc_streams_pre_mlp", "mhc_streams_final"):
        assert schema[name].width == HC * H
        assert schema[name].logical_shape == (HC, H)
        assert schema[name].dtype == torch.bfloat16
    assert not any(name.endswith("_mix") for name in schema)


def test_capture_schema_other_models_unchanged():
    dsv4 = SimpleNamespace(hc_mult=HC)
    assert build_model_hook_schema(H, torch.bfloat16, dsv4, dsv4) == build_hook_schema(
        H, torch.bfloat16, HC
    )
    llama = SimpleNamespace(model_type="llama")
    assert build_model_hook_schema(
        H, torch.bfloat16, llama, llama
    ) == default_hook_schema(H, torch.bfloat16)
    assert build_model_hook_schema(H, torch.bfloat16, None, None) == (
        default_hook_schema(H, torch.bfloat16)
    )
