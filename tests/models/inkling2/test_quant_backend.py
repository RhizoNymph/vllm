# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the Inkling2 `QUANT_METHOD` backend (SM8x / non-native
quantization), pinning the contract of `notes/inkling2-expert-interface.md`:

- which backend a quantization config selects on which device,
- the router and the experts' activation (vLLM `SITU` with the softcaps),
- what `Inkling2QuantMoE` hands to `quant_method.apply`,
- the checkpoint names the experts and the split attention linears load.

Modules are built with `object.__new__` and stubbed collaborators where a
real constructor would need a device or a distributed group.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.models.inkling2.configs import Inkling2ModelConfig
from vllm.models.inkling2.nvidia import backend as backend_mod
from vllm.models.inkling2.nvidia import experts_quant
from vllm.models.inkling2.nvidia.backend import (
    Inkling2Backend,
    Inkling2LayerQuant,
    Inkling2UnsupportedError,
    select_inkling2_backend,
)
from vllm.models.inkling2.nvidia.moe import Inkling2Gate, Inkling2RoutingCapture
from vllm.models.inkling2.quant_config import Inkling2QuantConfig

pytestmark = pytest.mark.skip_global_cleanup

HIDDEN, EXPERTS, TOPK, INTER = 64, 8, 2, 32


class _StubQuantConfig(QuantizationConfig):
    """A non-Inkling2 quantization config (as the GGUF plugin's)."""

    def get_name(self):
        return "stub"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 60

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        return None


def _text_config(**overrides) -> Inkling2ModelConfig:
    kwargs = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        route_scale=8.0,
        local_layer_ids=[0],
        vocab_size=128,
    )
    kwargs.update(overrides)
    return Inkling2ModelConfig(**kwargs)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

_NATIVE_QC = Inkling2QuantConfig({"model": {"parameter_groups": []}})


@pytest.mark.parametrize("sm8x", [False, True])
def test_non_inkling_quant_config_uses_quant_methods(sm8x):
    qc = _StubQuantConfig()
    quant = select_inkling2_backend(qc, 1, sm8x=sm8x)
    assert quant.backend is Inkling2Backend.QUANT_METHOD
    assert quant.linear is qc and quant.native is None
    assert quant.split_attention_linears


def test_native_checkpoint_keeps_upstream_backend_on_sm90_plus():
    quant = select_inkling2_backend(_NATIVE_QC, 4, sm8x=False)
    assert quant == Inkling2LayerQuant(Inkling2Backend.NATIVE, native=_NATIVE_QC)


def test_native_checkpoint_on_sm8x_fails_clearly():
    with pytest.raises(Inkling2UnsupportedError, match="SM90 or newer"):
        select_inkling2_backend(_NATIVE_QC, 1, sm8x=True)


def test_unquantized_is_native_on_sm90_plus_and_bf16_reference_on_sm8x():
    assert select_inkling2_backend(None, 2, sm8x=False).backend is (
        Inkling2Backend.NATIVE
    )
    quant = select_inkling2_backend(None, 1, sm8x=True)
    assert quant.backend is Inkling2Backend.QUANT_METHOD
    assert quant.linear is None
    # BF16 keeps the fused qkvr projection (plain F.linear on SM8x).
    assert not quant.split_attention_linears


@pytest.mark.parametrize("qc", [None, _StubQuantConfig()])
def test_quant_method_backend_rejects_tensor_parallel(qc):
    with pytest.raises(Inkling2UnsupportedError, match="tensor_parallel_size=1"):
        select_inkling2_backend(qc, 2, sm8x=True)


def test_backend_device_check_defaults_to_the_platform(monkeypatch):
    monkeypatch.setattr(backend_mod, "is_sm8x", lambda: True)
    assert select_inkling2_backend(None).uses_quant_method
    monkeypatch.setattr(backend_mod, "is_sm8x", lambda: False)
    assert not select_inkling2_backend(None).uses_quant_method


# --------------------------------------------------------------------------
# Router and activation
# --------------------------------------------------------------------------


def test_route_reference_selects_by_biased_sigmoid_and_weights_unbiased():
    torch.manual_seed(0)
    x = torch.randn(5, HIDDEN)
    weight = torch.randn(EXPERTS, HIDDEN)
    bias = torch.zeros(EXPERTS)
    # A large bias on expert 3 forces it into every selection without
    # changing its (unbiased) routing weight.
    bias[3] = 100.0
    ids, weights = experts_quant.route_reference(x, weight, bias, TOPK, 8.0)
    assert ids.dtype == torch.int32 and ids.shape == (5, TOPK)
    assert weights.dtype == torch.float32 and weights.shape == (5, TOPK)
    assert (ids == 3).any(dim=1).all()
    probs = torch.sigmoid(x @ weight.t())
    selected = probs.gather(1, ids.long())
    torch.testing.assert_close(weights, selected / selected.sum(-1, True) * 8.0)
    torch.testing.assert_close(weights.sum(-1), torch.full((5,), 8.0))


def test_situ_reference_is_vllm_situ_with_the_softcaps():
    """`softcap(g, 4) * sigmoid(g) * softcap(u, 25)` (the native `_situ`) is
    the Kimi SITU kernel's formula with beta = 4, linear_beta = 25."""
    torch.manual_seed(0)
    gate_up = torch.randn(7, 2 * INTER) * 6
    g, u = gate_up.chunk(2, dim=-1)
    beta, linear_beta = 4.0, 25.0
    kimi = (
        (0.5 * beta)
        * torch.tanh(g / beta)
        * (1 + torch.tanh(g * 0.5))
        * (linear_beta * torch.tanh(u / linear_beta))
    )
    native = (
        beta
        * torch.tanh(g / beta)
        * torch.sigmoid(g)
        * (linear_beta * torch.tanh(u / linear_beta))
    )
    out = experts_quant.situ_reference(gate_up, beta, linear_beta)
    torch.testing.assert_close(out, kimi, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out, native, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# Inkling2QuantMoE -> quant_method.apply
# --------------------------------------------------------------------------


class _RecordingMethod:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def apply(self, **kwargs):
        self.calls.append(kwargs)
        if self.result is not None:
            return self.result
        return torch.zeros_like(kwargs["x"])


def _quant_moe(method, visits=(0, 12, 24)):
    moe = object.__new__(experts_quant.Inkling2QuantMoE)
    nn.Module.__init__(moe)
    config = _text_config()
    moe.topk, moe.route_scale = TOPK, config.route_scale
    moe.gate = Inkling2Gate(config)
    torch.manual_seed(1)
    moe.gate.weight.data.copy_(torch.randn(EXPERTS, HIDDEN))
    moe.gate.bias.data.copy_(torch.randn(EXPERTS))
    moe.experts = SimpleNamespace(quant_method=method)
    moe.routing_captures = nn.ModuleList(Inkling2RoutingCapture(v) for v in visits)
    moe.register_buffer("all_experts", torch.tensor([0, EXPERTS], dtype=torch.int32))
    return moe


def test_quant_moe_passes_inkling_routing_to_quant_method():
    method = _RecordingMethod()
    moe = _quant_moe(method)
    x = torch.randn(6, 2 * HIDDEN)[:, :HIDDEN]  # non-contiguous input
    out = moe(x)
    (call,) = method.calls
    assert call["layer"] is moe.experts
    assert call["x"].is_contiguous()
    torch.testing.assert_close(call["x"], x)
    assert call["shared_experts"] is None and call["shared_experts_input"] is None
    ids, weights = experts_quant.route_reference(
        x, moe.gate.weight, moe.gate.bias, TOPK, 8.0
    )
    assert torch.equal(call["topk_ids"], ids)
    torch.testing.assert_close(call["topk_weights"], weights)
    assert out.shape == x.shape


@pytest.mark.parametrize("pass_idx", [0, 1, 2])
def test_quant_moe_records_routing_capture_at_the_visit_row(pass_idx):
    moe = _quant_moe(_RecordingMethod())
    seen = {}
    for p, capture in enumerate(moe.routing_captures):
        capture.capture_fn = lambda ids, p=p: seen.setdefault(p, ids)
    x = torch.randn(3, HIDDEN)
    moe(x, pass_idx=pass_idx)
    assert list(seen) == [pass_idx]
    assert torch.equal(seen[pass_idx], moe.routed_expert_ids(x))


def test_quant_moe_rejects_unfinalized_outputs_and_unreduced_requests():
    moe = _quant_moe(_RecordingMethod(result=object()))
    with pytest.raises(TypeError, match="finalized tensor"):
        moe(torch.randn(2, HIDDEN))
    moe = _quant_moe(_RecordingMethod())
    with pytest.raises(NotImplementedError):
        moe(torch.randn(2, HIDDEN), reduce=False)
    with pytest.raises(NotImplementedError):
        moe(torch.randn(2, HIDDEN), lora=object())


def test_quant_moe_matches_dense_reference_with_a_dense_method():
    """With a dense stand-in method (vLLM's decomposed MoE contract: per
    slot `w2 @ SITU(w13 @ x)`, scaled by the topk weight, summed), the
    layer reproduces the Inkling2 expert MLP: the routing weight inside the
    SwiGLU (native) equals scaling after the down projection."""
    torch.manual_seed(2)
    w13 = torch.randn(EXPERTS, 2 * INTER, HIDDEN) * 0.2
    w2 = torch.randn(EXPERTS, HIDDEN, INTER) * 0.2

    class _Dense:
        def apply(self, layer, x, topk_weights, topk_ids, **_):
            out = torch.zeros_like(x)
            for t in range(x.shape[0]):
                for k in range(topk_ids.shape[1]):
                    e = int(topk_ids[t, k])
                    act = experts_quant.situ_reference(w13[e] @ x[t], 4.0, 25.0)
                    out[t] += topk_weights[t, k] * (w2[e] @ act)
            return out

    moe = _quant_moe(_Dense())
    x = torch.randn(4, HIDDEN)
    ids, weights = experts_quant.route_reference(
        x, moe.gate.weight, moe.gate.bias, TOPK, 8.0
    )
    ref = torch.zeros_like(x)
    for t in range(4):
        for k in range(TOPK):
            e = int(ids[t, k])
            # Native order: gamma inside the activation, then down.
            act = experts_quant.situ_reference(w13[e] @ x[t], 4.0, 25.0) * weights[t, k]
            ref[t] += w2[e] @ act
    torch.testing.assert_close(moe(x), ref, rtol=1e-4, atol=1e-4)


def test_quant_moe_requires_a_gate_softcap(monkeypatch):
    monkeypatch.setattr(experts_quant, "make_routed_experts", lambda *a: None)
    with pytest.raises(ValueError, match="mlp_gate_softcap"):
        experts_quant.Inkling2QuantMoE(
            _text_config(mlp_gate_softcap=0.0), None, "model.layers.0.mlp"
        )


def test_make_routed_experts_declares_situ_with_the_softcaps(monkeypatch):
    """The FusedMoEConfig the quant method sees: SITU activation whose betas
    are the checkpoint softcaps, custom (decomposed) routing, no bias."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(expert_placement_strategy="linear"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device_config=SimpleNamespace(device="cpu"),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        lora_config=None,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=None),
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    parallel = FusedMoEParallelConfig.make_no_parallel()
    built = {}

    class _ExpertMap:
        local_num_experts = EXPERTS

        def __init__(self, **kwargs):
            built["expert_map"] = kwargs

    def _routed_experts(name, dtype, moe_config, quant_config, **kwargs):
        built.update(name=name, dtype=dtype, moe=moe_config, qc=quant_config, kw=kwargs)
        return "experts"

    monkeypatch.setattr(experts_quant, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(experts_quant, "make_parallel_config", lambda **_: parallel)
    monkeypatch.setattr(experts_quant, "ExpertMapManager", _ExpertMap)
    monkeypatch.setattr(experts_quant, "RoutedExperts", _routed_experts)
    qc = _StubQuantConfig()
    result = experts_quant.make_routed_experts(
        _text_config(), qc, "model.layers.3.mlp.experts"
    )
    assert result == "experts"
    assert built["name"] == "model.layers.3.mlp.experts" and built["qc"] is qc
    moe = built["moe"]
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    assert moe.activation is MoEActivation.SITU
    assert moe.activation_situ_beta == 4.0
    assert moe.activation_situ_linear_beta == 25.0
    assert moe.routing_method is RoutingMethodType.Custom
    assert moe.require_decomposed_backend and not moe.has_bias
    assert (moe.num_experts, moe.experts_per_token) == (EXPERTS, TOPK)
    assert (moe.hidden_dim, moe.intermediate_size_per_partition) == (HIDDEN, INTER)
    # Weights arrive pre-scaled by route_scale.
    assert built["kw"]["routed_scaling_factor"] == 1.0
    assert built["expert_map"]["global_num_experts"] == EXPERTS


# --------------------------------------------------------------------------
# Checkpoint names
# --------------------------------------------------------------------------


class _LoaderParam:
    def __init__(self):
        self.calls = []

    def weight_loader(self, param, weight, name, shard_id, expert_id):
        self.calls.append((name, shard_id, expert_id, weight.clone()))


@pytest.mark.parametrize("suffix", ["weight", "weight_type", "expert_scale"])
@pytest.mark.parametrize("layer_prefix", ["model.layers.7", "model.llm.layers.7"])
def test_per_expert_names_map_to_routed_expert_shards(suffix, layer_prefix):
    params = {
        f"{layer_prefix}.mlp.experts.w13_{suffix}": _LoaderParam(),
        f"{layer_prefix}.mlp.experts.w2_{suffix}": _LoaderParam(),
    }
    base = f"{layer_prefix}.mlp.experts"
    for proj, fused, shard in (
        ("gate_proj", "w13", "w1"),
        ("up_proj", "w13", "w3"),
        ("down_proj", "w2", "w2"),
    ):
        name = f"{base}.5.{proj}.{suffix}"
        tensor = torch.full((3,), float(len(proj)))
        loaded = experts_quant.load_quant_expert_weight(params, name, tensor)
        assert loaded == f"{base}.{fused}_{suffix}"
        call = params[loaded].calls[-1]
        assert call[:3] == (name, shard, 5)
        assert torch.equal(call[3], tensor)


def test_fused_interleaved_bf16_experts_are_deinterleaved():
    params = {
        "model.layers.0.mlp.experts.w13_weight": _LoaderParam(),
        "model.layers.0.mlp.experts.w2_weight": _LoaderParam(),
    }
    w13 = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).view(2, 4, 3)
    name = "model.layers.0.mlp.experts.w13_weight"
    assert experts_quant.load_quant_expert_weight(params, name, w13) == name
    calls = params[name].calls
    assert [(c[1], c[2]) for c in calls] == [("w1", 0), ("w3", 0), ("w1", 1), ("w3", 1)]
    assert torch.equal(calls[0][3], w13[0, 0::2])  # gate rows
    assert torch.equal(calls[1][3], w13[0, 1::2])  # up rows
    w2 = torch.randn(2, 3, 2, dtype=torch.bfloat16)
    name2 = "model.layers.0.mlp.experts.w2_weight"
    experts_quant.load_quant_expert_weight(params, name2, w2)
    assert [(c[1], c[2]) for c in params[name2].calls] == [("w2", 0), ("w2", 1)]


def test_non_expert_and_quantized_fused_names_are_left_to_the_caller():
    params = {"model.layers.0.mlp.experts.w13_weight": _LoaderParam()}
    load = experts_quant.load_quant_expert_weight
    assert load(params, "model.layers.0.mlp.gate.weight", torch.zeros(1)) is None
    # NVFP4 bytes under the fused name are not this backend's.
    nvfp4 = torch.zeros(2, 4, 3, dtype=torch.uint8)
    assert load(params, "model.layers.0.mlp.experts.w13_weight", nvfp4) is None
    # A per-expert name without a matching parameter is not consumed either.
    assert load(params, "model.layers.0.mlp.experts.0.gate_proj.foo", nvfp4) is None


def test_dense_weights_are_cast_but_quantized_ones_untouched():
    from torch.nn.parameter import UninitializedParameter

    from vllm.models.inkling2.nvidia.backend import cast_dense_weight

    bf16 = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False)
    f32 = torch.ones(2, dtype=torch.float32)
    assert cast_dense_weight(bf16, f32).dtype == torch.bfloat16
    packed = torch.ones(2, dtype=torch.uint8)
    assert cast_dense_weight(bf16, packed) is packed
    lazy = UninitializedParameter(requires_grad=False)
    assert cast_dense_weight(lazy, f32) is f32
    u8 = nn.Parameter(torch.zeros(2, dtype=torch.uint8), requires_grad=False)
    assert cast_dense_weight(u8, f32) is f32


# --------------------------------------------------------------------------
# Split attention linears
# --------------------------------------------------------------------------


def test_split_attention_projects_q_side_k_and_v_from_their_inputs():
    from vllm.models.inkling2.nvidia.attention import Inkling2Attention

    torch.manual_seed(3)
    attn = object.__new__(Inkling2Attention)
    nn.Module.__init__(attn)
    attn.split_linears = True
    attn.split_sizes = [8, 4, 8, 2, 6, 6]
    wq, wk, wv = torch.randn(22, 16), torch.randn(6, 16), torch.randn(6, 16)
    attn.q_proj = lambda x: x @ wq.t()
    attn.k_proj = lambda x: x @ wk.t()
    attn.v_proj = lambda x: x @ wv.t()
    xq, xk, xv = torch.randn(3, 16), torch.randn(3, 16), torch.randn(3, 16)
    q, r, gate, sinks, k, v = attn._project(xq, xk, xv)
    full = xq @ wq.t()
    torch.testing.assert_close(q, full[:, :8])
    torch.testing.assert_close(r, full[:, 8:12])
    torch.testing.assert_close(gate, full[:, 12:20])
    torch.testing.assert_close(sinks, full[:, 20:22])
    torch.testing.assert_close(k, xk @ wk.t())
    torch.testing.assert_close(v, xv @ wv.t())


def _record_linear(kind, built):
    class _Linear(nn.Module):
        def __init__(
            self, input_size, output_size, *, bias, quant_config, prefix, **kwargs
        ):
            super().__init__()
            built[prefix] = (kind, input_size, output_size, quant_config, kwargs)

    return _Linear


@pytest.mark.parametrize("is_local", [True, False])
def test_split_attention_linears_take_the_quant_config(monkeypatch, is_local):
    """The quantizable linears and their prefixes are the ones listed in
    notes/inkling2-expert-interface.md (Inkling 2 Mini sizes)."""
    from vllm.models.inkling2.nvidia import attention as attn_mod
    from vllm.models.inkling2.nvidia import sconv as sconv_mod

    built = {}
    monkeypatch.setattr(
        attn_mod, "MergedColumnParallelLinear", _record_linear("merged", built)
    )
    monkeypatch.setattr(
        attn_mod, "ColumnParallelLinear", _record_linear("column", built)
    )
    monkeypatch.setattr(attn_mod, "RowParallelLinear", _record_linear("row", built))
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(sconv_mod, "get_tensor_model_parallel_rank", lambda: 0)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
    )
    monkeypatch.setattr(attn_mod, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_mod, "max_decode_query_len", lambda _: 1)
    config = Inkling2ModelConfig(
        hidden_size=2304,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=8,
        head_dim=128,
        d_rel=16,
        local_layer_ids=[0],
    )
    qc = _StubQuantConfig()
    prefix = "model.layers.0.attn" if is_local else "model.layers.5.attn"
    attn = attn_mod.Inkling2Attention(
        config,
        is_local,
        SimpleNamespace(),
        prefix,
        quant_config=qc,
        split_linears=True,
    )
    assert not hasattr(attn, "qkvr")
    assert built == {
        f"{prefix}.q_proj": (
            "merged",
            2304,
            [3072, 384, 3072, 24],
            qc,
            {"return_bias": False},
        ),
        f"{prefix}.k_proj": ("column", 2304, 1024, qc, {"return_bias": False}),
        f"{prefix}.v_proj": ("column", 2304, 1024, qc, {"return_bias": False}),
        f"{prefix}.wo_ud": ("row", 3072, 2304, qc, {"reduce_results": False}),
    }


# --------------------------------------------------------------------------
# Inkling2ForConditionalGeneration.load_weights (QUANT_METHOD names)
# --------------------------------------------------------------------------


def _tree(names_to_params):
    """An nn.Module whose `named_parameters()` are `names_to_params`."""
    root = nn.Module()
    for name, param in names_to_params.items():
        *path, leaf = name.split(".")
        module = root
        for part in path:
            if not hasattr(module, part):
                module.add_module(part, nn.Module())
            module = getattr(module, part)
        module.register_parameter(leaf, param)
    return root


def _param(shape, dtype, calls, name, sharded=False, moe=False):
    p = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    if moe:
        p.weight_loader = lambda param, w, wname, shard_id, expert_id: calls.append(
            (name, wname, shard_id, expert_id, w.dtype)
        )
    elif sharded:
        p.weight_loader = lambda param, w, shard_id: calls.append(
            (name, shard_id, w.dtype)
        )
    else:
        p.weight_loader = lambda param, w: calls.append((name, None, w.dtype))
    return p


def test_load_weights_maps_quant_method_checkpoint_names():
    from vllm.models.inkling2.nvidia.model import Inkling2ForConditionalGeneration

    calls = []
    u8, bf16, f32 = torch.uint8, torch.bfloat16, torch.float32
    L = "model.layers.0"
    params = {
        f"{L}.attn.q_proj.qweight": _param(4, u8, calls, "q", sharded=True),
        f"{L}.attn.k_proj.qweight": _param(4, u8, calls, "k"),
        f"{L}.attn.v_proj.qweight": _param(4, u8, calls, "v"),
        f"{L}.attn.wo_ud.qweight": _param(4, u8, calls, "o"),
        f"{L}.attn_norm.weight": _param((3, 4), bf16, calls, "norm", sharded=True),
        f"{L}.mlp.gate.weight": _param((2, 4), bf16, calls, "gate"),
        f"{L}.mlp.experts.w13_qweight": _param(4, u8, calls, "w13", moe=True),
        f"{L}.mlp.experts.w2_qweight": _param(4, u8, calls, "w2", moe=True),
        "model.embed_tokens.qweight": _param(4, u8, calls, "embed"),
        "lm_head.qweight": _param(4, u8, calls, "head"),
    }
    model = object.__new__(Inkling2ForConditionalGeneration)
    nn.Module.__init__(model)
    tree = _tree(params)
    model.model, model.lm_head = tree.model, tree.lm_head
    model.quant = select_inkling2_backend(_StubQuantConfig(), 1, sm8x=True)
    model.text_config = SimpleNamespace(num_hidden_layers=1)
    model.visual = None
    C = "model.llm.layers.0"
    q8 = torch.zeros(4, dtype=u8)
    weights = [
        *(
            (f"{C}.attn.{n}.qweight", q8)
            for n in (
                "wq_du",
                "wr_du",
                "gate_attn",
                "attn_sink_proj",
                "wk_dv",
                "wv_dv",
                "wo_ud",
            )
        ),
        *((f"{C}.{n}_input_norm.weight", torch.ones(4, dtype=f32)) for n in "qkv"),
        (f"{C}.mlp.gate.weight", torch.ones(2, 4, dtype=f32)),
        *(
            (f"{C}.mlp.experts.{e}.{p}.qweight", q8)
            for e in range(2)
            for p in ("gate_proj", "up_proj", "down_proj")
        ),
        ("model.llm.embed.qweight", q8),
        ("model.llm.unembed.qweight", q8),
    ]
    loaded = model.load_weights(weights)
    assert loaded == set(params)
    by_name = {}
    for call in calls:
        by_name.setdefault(call[0], []).append(call[1:])
    assert by_name["q"] == [(0, u8), (1, u8), (2, u8), (3, u8)]
    assert by_name["k"] == [(None, u8)] and by_name["v"] == [(None, u8)]
    # Dense F32 checkpoint tensors are cast to the parameter dtype.
    assert by_name["norm"] == [(0, bf16), (1, bf16), (2, bf16)]
    assert by_name["gate"] == [(None, bf16)]
    assert [c[1:3] for c in by_name["w13"]] == [
        ("w1", 0),
        ("w3", 0),
        ("w1", 1),
        ("w3", 1),
    ]
    assert [c[1:3] for c in by_name["w2"]] == [("w2", 0), ("w2", 1)]
    assert by_name["embed"] == [(None, u8)] and by_name["head"] == [(None, u8)]
