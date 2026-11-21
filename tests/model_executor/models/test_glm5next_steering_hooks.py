# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (``glm5next``) decoder layers emit the intervention hooks.

The decoder layer (shared by NVIDIA and ROCm in ``common/model.py``) has two
shapes. mHC layers carry ``hc_mult`` residual streams and take the
DeepSeek-V4 mHC hook set (single-stream ``pre_attn``/``post_attn``/
``mlp_in``/``mlp_out`` plus ``mhc_streams_pre_attn``/``mhc_streams_pre_mlp``
and, on the last layer, ``mhc_streams_final``); single-residual layers take
the standard five hooks. MTP draft blocks reuse the class and stay unhooked.

The mHC TileLang/FlashInfer kernels need a GPU, so the layer's ``hc_pre`` /
``hc_fused_post_pre`` / ``hc_post`` are replaced by small torch stand-ins with
the same signatures and the same additive residual semantics.
"""

import ast
import inspect
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
from vllm.models.glm5next.common import interventions as glm_iv
from vllm.models.glm5next.common import model as glm
from vllm.models.glm5next.nvidia.specialized_decode import runtime as sd_runtime

N, H, T = 2, 4, 3
MHC_HOOKS = [
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


class _Norm(Scale):
    """Scale stub that also exposes the RMSNorm fields the mHC path reads."""

    def __init__(self, factor: float = 1.0):
        super().__init__(factor)
        self.weight = torch.nn.Parameter(torch.ones(H), requires_grad=False)
        self.variance_epsilon = 1e-6


class _Attn(Const):
    def __init__(self, value: float = 3.0):
        super().__init__(value)
        self.o_proj = SimpleNamespace(reduce_results=True)


def _hc_pre(x, fn, scale, base, norm_weight=None, norm_eps=0.0):
    t, n, _ = x.shape
    post = torch.full((t, n, 1), 0.5)
    comb = torch.eye(n).expand(t, n, n).clone()
    return post, comb, x.sum(1)


def _hc_fused_post_pre(
    x, residual, post, comb, fn, scale, base, norm_weight=None, norm_eps=0.0,
    fn_split=None,
):  # fmt: skip
    new = residual + post * x.unsqueeze(1)
    return new, post, comb, new.sum(1)


def _hc_post(x, residual, post, comb):
    return residual + post * x.unsqueeze(1)


def _install_hc_stubs(layer):
    layer.hc_pre = _hc_pre
    layer.hc_fused_post_pre = _hc_fused_post_pre
    layer.hc_post = _hc_post


def _record(monkeypatch):
    rec = record_hooks(monkeypatch, glm, glm_iv)

    def _streams(module, streams, hook_point):
        rec.calls.append((hook_point.value, streams.detach().clone()))
        return streams

    for mod in (glm, glm_iv):
        if hasattr(mod, "apply_layer_steering_streams"):
            monkeypatch.setattr(mod, "apply_layer_steering_streams", _streams)
    return rec


def _bare_mhc(layer_idx=1, num_layers=3):
    layer = bare(
        glm.Glm5NextDecoderLayer,
        layer_idx=layer_idx,
        hidden_size=H,
        num_hidden_layers=num_layers,
        mhc=True,
        is_mtp_layer=False,
        is_sequence_parallel=False,
        _mlp_is_moe=False,
        n=N,
        hc_mult=N,
        input_layernorm=_Norm(1.0),
        post_attention_layernorm=_Norm(1.0),
        self_attn=_Attn(3.0),
        mlp=Const(5.0),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        hc_ffn_fn=None,
        hc_ffn_scale=None,
        hc_ffn_base=None,
        hc_attn_fn_split=None,
        hc_ffn_fn_split=None,
    )
    _install_hc_stubs(layer)
    return layer


def _bare_plain(is_mtp_layer=False):
    return bare(
        glm.Glm5NextDecoderLayer,
        layer_idx=0,
        hidden_size=H,
        num_hidden_layers=3,
        mhc=False,
        is_mtp_layer=is_mtp_layer,
        is_sequence_parallel=False,
        _mlp_is_moe=False,
        input_layernorm=Scale(1.0),
        self_attn=Const(3.0),
        post_attention_layernorm=Scale(2.0),
        mlp=Const(5.0),
    )


def _completed(out):
    """The layer's completed residual streams (its deferred hc_post applied)."""
    x, residual, post, comb = out
    return _hc_post(x, residual, post, comb)


# ---------------------------------------------------------------------------
# Hook order and values (stubs)
# ---------------------------------------------------------------------------


def test_plain_layer_hook_order_and_values(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_plain()
    out, residual, post, comb = layer(
        torch.zeros(T, dtype=torch.long), torch.ones(T, H)
    )

    assert rec.order == ALL_HOOKS
    assert torch.equal(rec.seen("pre_attn"), torch.full((T, H), 1.0))
    # post_attn: residual after the fused add (1 + attention 3).
    assert torch.equal(rec.seen("post_attn"), torch.full((T, H), 4.0))
    assert torch.equal(rec.seen("mlp_in"), torch.full((T, H), 8.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((T, H), 5.0))
    assert torch.equal(rec.seen("post_block"), torch.full((T, H), 9.0))
    # The single-residual path returns the summed stream.
    assert torch.equal(out, torch.full((T, H), 9.0))
    assert post is None and comb is None


def test_mtp_layer_emits_no_hooks(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_plain(is_mtp_layer=True)
    layer.mhc = True
    out, residual, _, _ = layer(torch.zeros(T, dtype=torch.long), torch.ones(T, H))
    assert rec.order == []
    # MTP keeps returning the unsummed pair for shared_head's fused norm.
    assert torch.equal(out + residual, torch.full((T, H), 9.0))


def test_mhc_middle_layer_hook_order_and_values(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_mhc(layer_idx=1)
    out = layer(torch.zeros(T, dtype=torch.long), torch.ones(T, N, H))

    assert rec.order == MHC_HOOKS
    assert torch.equal(rec.seen("mhc_streams_pre_attn"), torch.ones(T, N, H))
    # Normed single-stream attention input (the pre op's reduction).
    assert torch.equal(rec.seen("pre_attn"), torch.full((T, H), 2.0))
    assert torch.equal(rec.seen("post_attn"), torch.full((T, H), 3.0))
    # Streams entering the FFN: 1 + 0.5 * attention 3.
    assert torch.equal(rec.seen("mhc_streams_pre_mlp"), torch.full((T, N, H), 2.5))
    assert torch.equal(rec.seen("mlp_in"), torch.full((T, H), 5.0))
    assert torch.equal(rec.seen("mlp_out"), torch.full((T, H), 5.0))
    # Non-last layers defer their hc_post into the next layer's fused pre.
    assert len(out) == 4 and out[1] is not None
    assert torch.equal(_completed(out), torch.full((T, N, H), 5.0))


def test_mhc_first_layer_expands_then_hooks(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_mhc(layer_idx=0)
    layer(torch.zeros(T, dtype=torch.long), torch.ones(T, H))
    assert rec.order == MHC_HOOKS
    assert torch.equal(rec.seen("mhc_streams_pre_attn"), torch.ones(T, N, H))


def test_mhc_layer_with_incoming_state_uses_fused_pre(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_mhc(layer_idx=1)
    residual = torch.ones(T, N, H)
    post = torch.full((T, N, 1), 0.5)
    comb = torch.eye(N).expand(T, N, N).clone()
    layer(
        torch.zeros(T, dtype=torch.long), torch.full((T, H), 2.0), residual, post, comb
    )
    # The previous layer's deferred branch (0.5 * 2) is folded in first.
    assert torch.equal(rec.seen("mhc_streams_pre_attn"), torch.full((T, N, H), 2.0))
    assert torch.equal(rec.seen("pre_attn"), torch.full((T, H), 4.0))


def test_mhc_last_layer_steers_final_streams(monkeypatch):
    rec = _record(monkeypatch)
    layer = _bare_mhc(layer_idx=2, num_layers=3)
    out, residual, post, comb = layer(
        torch.zeros(T, dtype=torch.long), torch.ones(T, N, H)
    )
    assert rec.order == [*MHC_HOOKS, "mhc_streams_final"]
    assert torch.equal(rec.seen("mhc_streams_final"), torch.full((T, N, H), 5.0))
    # The last layer materializes hc_post and contracts to one stream.
    assert residual is None and post is None and comb is None
    assert torch.equal(out, torch.full((T, H), 5.0))


# ---------------------------------------------------------------------------
# Real __init__ registration
# ---------------------------------------------------------------------------


def _vllm_config(*, steering=True, sp=False):
    cfg = fake_vllm_config(steering=steering)
    cfg.cache_config = None
    cfg.quant_config = None
    cfg.parallel_config = SimpleNamespace(use_sequence_parallel_moe=sp)
    cfg.kernel_config = SimpleNamespace(enable_jit_warmup=False)
    return cfg


def _hf_config(*, mhc=True, num_layers=3):
    return SimpleNamespace(
        hidden_size=H,
        n_routed_experts=None,
        num_hidden_layers=num_layers,
        rms_norm_eps=1e-6,
        mhc=mhc,
        layer_types=None,
        mlp_layer_types=["dense"] * num_layers,
        hc_mult=N,
        hc_eps=1e-6,
        hc_sinkhorn_iters=1,
        v_head_dim=H,
        kv_lora_rank=H,
        num_attention_heads=1,
        qk_nope_head_dim=H,
        qk_rope_head_dim=0,
        q_lora_rank=None,
        max_position_embeddings=16,
        mla_use_nope=True,
        intermediate_size=8,
        hidden_act="silu",
        swiglu_limit=None,
    )


def _built_layer(
    monkeypatch, *, layer_idx=1, mhc=True, steering=True, sp=False,
    is_mtp_layer=False,
):  # fmt: skip
    monkeypatch.setattr(glm, "Glm5NextMLAAttention", lambda *a, **k: _Attn(3.0))
    monkeypatch.setattr(glm, "Glm5NextLinearAttention", lambda *a, **k: _Attn(3.0))
    monkeypatch.setattr(glm, "Glm5NextMLP", lambda *a, **k: Const(5.0))
    monkeypatch.setattr(glm, "RMSNorm", lambda *a, **k: _Norm(1.0))
    for op in ("MHCPreOp", "MHCPostOp", "MHCFusedPostPreOp"):
        monkeypatch.setattr(glm, op, torch.nn.Identity)
    monkeypatch.setattr(
        glm.current_platform, "get_device_capability", lambda *a, **k: None
    )
    layer = glm.Glm5NextDecoderLayer(
        vllm_config=_vllm_config(steering=steering, sp=sp),
        config=_hf_config(mhc=mhc),
        layer_idx=layer_idx,
        prefix=f"model.layers.{layer_idx}",
        is_mtp_layer=is_mtp_layer,
    )
    if mhc and not is_mtp_layer:
        _install_hc_stubs(layer)
    return layer


def _table(layer, hook):
    return getattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)])


def test_init_registers_mhc_hook_set(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=1)
    assert layer.layer_idx == 1
    for hook in ("pre_attn", "post_attn", "mlp_in", "mlp_out"):
        assert _table(layer, hook).shape[-1] == H, hook
    for hook in ("mhc_streams_pre_attn", "mhc_streams_pre_mlp"):
        assert _table(layer, hook).shape[-1] == N * H, hook
    # No single-stream post_block, and the final stream table is the last
    # layer's alone.
    assert not hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK])
    assert not hasattr(
        layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.MHC_STREAMS_FINAL]
    )


def test_init_last_layer_owns_final_stream_table(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=2)
    assert _table(layer, "mhc_streams_final").shape[-1] == N * H


def test_init_registers_standard_hooks_without_mhc(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=2, mhc=False)
    assert layer.layer_idx == 2
    for hook in ALL_HOOKS:
        assert _table(layer, hook).shape[-1] == H, hook


@pytest.mark.parametrize("mhc", [True, False])
def test_init_without_steering_registers_nothing(monkeypatch, mhc):
    layer = _built_layer(monkeypatch, mhc=mhc, steering=False)
    assert not hasattr(layer, "steering_index")


def test_mtp_layer_registers_nothing(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=3, is_mtp_layer=True)
    assert not hasattr(layer, "steering_index")


# ---------------------------------------------------------------------------
# Real steering op on CPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hook", ["mhc_streams_pre_attn", "mhc_streams_pre_mlp"])
def test_stream_vector_shifts_completed_streams(monkeypatch, hook):
    layer = _built_layer(monkeypatch, layer_idx=1)
    pos = torch.zeros(T, dtype=torch.long)
    h = torch.ones(T, N, H)
    base = _completed(layer(pos, h.clone()))

    vec = torch.arange(N * H, dtype=torch.float32) - 3.0
    set_global_vector(layer, hook, vec, num_tokens=T)
    got = _completed(layer(pos, h.clone()))
    assert torch.allclose(got - base, vec.view(N, H).expand(T, N, H))

    set_global_vector(layer, hook, torch.zeros(N * H), num_tokens=T)
    assert torch.equal(_completed(layer(pos, h.clone())), base)


def test_final_stream_vector_shifts_last_layer_output(monkeypatch):
    layer = _built_layer(monkeypatch, layer_idx=2)
    pos = torch.zeros(T, dtype=torch.long)
    h = torch.ones(T, N, H)
    base = layer(pos, h.clone())[0]

    vec = torch.arange(N * H, dtype=torch.float32)
    set_global_vector(layer, "mhc_streams_final", vec, num_tokens=T)
    got = layer(pos, h.clone())[0]
    assert torch.allclose(got - base, vec.view(N, H).mean(0).expand(T, H))

    set_global_vector(layer, "mhc_streams_final", torch.zeros(N * H), num_tokens=T)
    assert torch.equal(layer(pos, h.clone())[0], base)


@pytest.mark.parametrize("hook", ["pre_attn", "post_attn", "post_block"])
def test_plain_residual_vector_shifts_block_output(monkeypatch, hook):
    layer = _built_layer(monkeypatch, layer_idx=1, mhc=False)
    pos = torch.zeros(T, dtype=torch.long)
    h = torch.ones(T, H)
    base = layer(pos, h.clone())[0]

    vec = torch.tensor([1.0, -2.0, 0.5, 4.0])
    set_global_vector(layer, hook, vec, num_tokens=T)
    assert torch.allclose(layer(pos, h.clone())[0] - base, vec.expand(T, H))

    set_global_vector(layer, hook, torch.zeros(H), num_tokens=T)
    assert torch.equal(layer(pos, h.clone())[0], base)


# ---------------------------------------------------------------------------
# Model wiring and fused/specialized paths
# ---------------------------------------------------------------------------


def test_index_shared_across_layers():
    assert calls((glm.Glm5NextModel, "__init__"), "share_steering_index_across_layers")


def test_vl_wrapper_builds_the_hooked_language_model():
    src = inspect.getsource(glm.Glm5NextForConditionalGeneration.__init__)
    assert "init_vllm_registered_model" in src
    assert '"Glm5NextForCausalLM"' in src


def test_sequence_parallel_moe_off_under_interventions():
    def gate(*, steering, sp):
        cfg = _vllm_config(steering=steering, sp=sp)
        return glm._use_sequence_parallel_moe(cfg.parallel_config, cfg)

    assert gate(steering=False, sp=True)
    assert not gate(steering=True, sp=True)
    assert not gate(steering=False, sp=False)


def test_every_sequence_parallel_switch_goes_through_the_gate():
    """The layer, MoE, and model must agree on SP; only the gate reads it."""
    tree = ast.parse(inspect.getsource(glm))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "use_sequence_parallel_moe"
    ]
    assert len(reads) == 1
    for cls in (glm.Glm5NextDecoderLayer, glm.Glm5NextMoE, glm.Glm5NextModel):
        assert calls((cls, "__init__"), "_use_sequence_parallel_moe"), cls


@pytest.mark.parametrize("steering", [False, True])
def test_layer_sequence_parallel_follows_gate(monkeypatch, steering):
    layer = _built_layer(monkeypatch, steering=steering, sp=True)
    assert layer.is_sequence_parallel is (not steering)
    # Without SP the attention output projection reduces normally.
    assert layer.self_attn.o_proj.reduce_results is steering


def _decode_config(backend, *, steering):
    cfg = _vllm_config(steering=steering)
    cfg.additional_config = {"decode_backend": backend}
    cfg.speculative_config = None
    cfg.model_config = SimpleNamespace(dtype=torch.bfloat16, max_model_len=64)
    cfg.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[1, 2])
    return cfg


def test_specialized_decode_kept_without_interventions(monkeypatch):
    monkeypatch.setattr(sd_runtime, "unsupported", lambda cfg: [])
    got = sd_runtime.specialized_decode_config(_decode_config("auto", steering=False))
    assert got is not None and got.capture_sizes == (1, 2)


def test_specialized_decode_off_under_interventions(monkeypatch):
    monkeypatch.setattr(sd_runtime, "unsupported", lambda cfg: [])
    assert (
        sd_runtime.specialized_decode_config(_decode_config("auto", steering=True))
        is None
    )
    with pytest.raises(ValueError, match="steering"):
        sd_runtime.specialized_decode_config(
            _decode_config("latency_specialized", steering=True)
        )
