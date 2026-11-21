# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 on SM8x (Ampere / Ada): the `QUANT_METHOD` backend end to end.

GPU tests for an RTX 3090-class device (TP=1). Upstream's `test_model.py` /
`test_dspark.py` skip below Hopper; these cover the same model paths on
SM8x with the BF16 reference experts (`RoutedExperts` +
`UnquantizedFusedMoEMethod`, vLLM's SITU activation) and the Triton
attention / conv / AttnRes kernels:

- kernel correctness: upstream's attention tests, re-run on SM8x,
- one MoE layer against a dense torch reference,
- tiny random-weight Inkling 2 Small / Mini models: generation, CUDA graphs
  vs eager, decode vs prefill (per-pass caches), routed-experts capture per
  layer visit, DSpark speculative decoding,
- an out-of-tree quantization config (unquantized methods, so vLLM's
  quantization plumbing runs the split attention linears, embeddings and
  routed experts the way the GGUF plugin's methods do).
"""

import copy
import json
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(80):
    pytest.skip("SM8x (Ampere / Ada) tests", allow_module_level=True)

from tests.models.inkling2 import test_attention as upstream_attention  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.distributed import cleanup_dist_env_and_memory  # noqa: E402

TINY_CONFIG: dict[str, Any] = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}

# Inkling 2 Mini's shape in miniature: prelude layer 0, core layers 1-2 run
# three passes, coda layer 3 (8 layer visits).
TINY_LOOP_CONFIG: dict[str, Any] = {
    **TINY_CONFIG,
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "text_config": {
        **TINY_CONFIG["text_config"],
        "hidden_size": 2304,
        "num_hidden_layers": 4,
        "local_layer_ids": [0, 1, 3],
        "n_layers_per_attn_res_block": 1,
        "loop_config": {
            "core_start": 1,
            "core_end": 3,
            "num_passes": 3,
            "gate_rank": 16,
            "is_varloop": False,
        },
    },
    "mtp_config": {
        **TINY_CONFIG["mtp_config"],
        "hidden_size": 2304,
        "target_layer_ids": [3, 8],
    },
}


def _write(tmp_path, config: dict[str, Any]) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _llm(model: str, looped: bool, **kwargs) -> LLM:
    kwargs.setdefault("gpu_memory_utilization", 0.25 if looped else 0.10)
    return LLM(
        model=model,
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        max_model_len=2048,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 0},
        **kwargs,
    )


def _greedy(llm: LLM, prompts: list[list[int]], max_tokens: int = 16):
    return [
        out.outputs[0]
        for out in llm.generate(
            [{"prompt_token_ids": p} for p in prompts],
            SamplingParams(
                max_tokens=max_tokens,
                temperature=0,
                logprobs=0,
                ignore_eos=True,
                detokenize=False,
            ),
        )
    ]


# --------------------------------------------------------------------------
# Kernels (upstream's references, on SM8x)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left,scenario",
    [
        (24, 8, 16, None, "prefill"),
        (24, 8, 128, 1023, "prefill"),
        (24, 8, 16, 1023, "chunked_prefill"),
        (12, 2, 128, None, "chunked_prefill"),
        (48, 8, 128, 1023, "decode"),
        (12, 2, 16, None, "decode"),
    ],
)
def test_rel_attention_on_sm8x(
    num_heads, num_kv_heads, block_size, window_left, scenario
):
    upstream_attention.test_inkling2_rel_attention(
        num_heads, num_kv_heads, block_size, window_left, scenario, 1, True
    )


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,window_left", [(24, 8, None), (24, 8, 1023), (6, 1, 1023)]
)
@pytest.mark.parametrize(
    "query_lens,kv_lens",
    [([1], [5000]), ([1] * 4, [1, 200, 1025, 3000]), ([8, 8, 3], [8, 1500, 2100])],
)
@pytest.mark.parametrize("block_size,num_splits", [(128, 1), (128, 9), (16, 9)])
def test_rel_decode_attention_on_sm8x(
    num_heads, num_kv_heads, window_left, query_lens, kv_lens, block_size, num_splits
):
    upstream_attention.test_inkling2_rel_decode_attention(
        num_heads,
        num_kv_heads,
        window_left,
        query_lens,
        kv_lens,
        block_size,
        num_splits,
    )


@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("num_splits", [1, 4])
def test_decode_attention_after_prologue_on_sm8x(is_local, num_splits):
    upstream_attention.test_inkling2_decode_attention_after_prologue(
        is_local, num_splits
    )


# --------------------------------------------------------------------------
# One MoE layer against a dense reference
# --------------------------------------------------------------------------


@torch.inference_mode()
def test_quant_moe_bf16_matches_dense_reference(dist_init, default_vllm_config):
    """`Inkling2QuantMoE` with BF16 weights (UnquantizedFusedMoEMethod's
    Triton experts, vLLM SITU) against `sum_k w_k * W2 @ situ(W13 @ x)` with
    the torch router."""
    from vllm.models.inkling2.configs import Inkling2ModelConfig
    from vllm.models.inkling2.nvidia.experts_quant import (
        Inkling2QuantMoE,
        load_quant_expert_weight,
        route_reference,
        situ_reference,
    )

    config = Inkling2ModelConfig(
        hidden_size=2304,
        num_hidden_layers=1,
        n_routed_experts=128,
        num_experts_per_tok=8,
        intermediate_size=1152,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        route_scale=8.0,
    )
    torch.set_default_device("cuda")
    try:
        moe = Inkling2QuantMoE(config, None, "model.layers.0.mlp", [0]).cuda()
    finally:
        torch.set_default_device("cpu")
    g = torch.Generator(device="cuda").manual_seed(0)
    e, inter, hidden = 128, 1152, 2304
    w13 = (
        torch.randn(e, 2 * inter, hidden, generator=g, device="cuda") * 0.03
    ).bfloat16()
    w2 = (torch.randn(e, hidden, inter, generator=g, device="cuda") * 0.03).bfloat16()
    params = dict(moe.named_parameters(prefix="model.layers.0.mlp"))
    load_quant_expert_weight(params, "model.layers.0.mlp.experts.w13_weight", w13)
    load_quant_expert_weight(params, "model.layers.0.mlp.experts.w2_weight", w2)
    moe.gate.weight.copy_(torch.randn(e, hidden, generator=g, device="cuda").bfloat16())
    moe.gate.bias.copy_(torch.randn(e, generator=g, device="cuda") * 0.1)
    moe.experts.quant_method.process_weights_after_loading(moe.experts)
    x = torch.randn(37, hidden, generator=g, device="cuda").bfloat16()
    out = moe(x)
    ids, weights = moe.route(x)
    ref_ids, ref_weights = route_reference(x, moe.gate.weight, moe.gate.bias, 8, 8.0)
    assert torch.equal(ids.sort(-1).values, ref_ids.sort(-1).values)
    torch.testing.assert_close(
        weights.gather(1, ids.argsort(-1)),
        ref_weights.gather(1, ref_ids.argsort(-1)),
        rtol=1e-4,
        atol=1e-4,
    )
    ref = torch.zeros(x.shape, dtype=torch.float32, device="cuda")
    for slot in range(8):
        expert = ids[:, slot].long()
        gate_up = torch.einsum("th,tnh->tn", x.float(), w13[expert].float())
        # Interleaved checkpoint rows: (gate c, up c) pairs.
        gate_up = torch.cat([gate_up[:, 0::2], gate_up[:, 1::2]], dim=-1)
        act = situ_reference(gate_up, 4.0, 25.0)
        down = torch.einsum("tn,thn->th", act, w2[expert].float())
        ref += weights[:, slot, None] * down
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)


# --------------------------------------------------------------------------
# Tiny models
# --------------------------------------------------------------------------


def _backend(model: torch.nn.Module) -> tuple[str, str, str]:
    from vllm.models.inkling2.nvidia.experts_quant import Inkling2QuantMoE

    mlp = model.model.layers[0].mlp
    assert isinstance(mlp, Inkling2QuantMoE)
    return (
        model.quant.backend.value,
        type(mlp.experts.quant_method).__name__,
        type(model.model.layers[0].attn).__name__,
    )


@pytest.mark.parametrize("looped", [False, True])
def test_tiny_model_selects_quant_method_backend(tmp_path, enable_pickle, looped):
    llm = _llm(
        _write(tmp_path, TINY_LOOP_CONFIG if looped else TINY_CONFIG),
        looped,
        enforce_eager=True,
    )
    backend, method, _ = llm.apply_model(_backend)[0]
    assert (backend, method) == ("quant_method", "UnquantizedFusedMoEMethod")


@pytest.mark.parametrize("looped", [False, True])
def test_tiny_model_generates_and_graphs_match_eager(tmp_path, looped):
    """Dummy weights decode in range with CUDA graphs, and the graphs give
    the eager tokens (the 298-token prompt chunks over two steps)."""
    model = _write(tmp_path, TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    prompts = [list(range(2, 300)), [5, 5, 5]]
    runs = []
    for eager in (False, True):
        llm = _llm(model, looped, enforce_eager=eager, max_num_batched_tokens=256)
        outputs = _greedy(llm, prompts, 24)
        for out in outputs:
            assert len(out.token_ids) == 24
            assert all(0 <= t < 1000 for t in out.token_ids)
        runs.append([list(out.token_ids) for out in outputs])
        del llm
        cleanup_dist_env_and_memory()
    assert runs[0] == runs[1]


def test_tiny_looped_model_decode_matches_prefill(tmp_path):
    """Per-pass KV and conv-state caches on SM8x: greedy decode steps give
    the tokens and logprobs of one prefill over the same sequence."""
    llm = _llm(_write(tmp_path, TINY_LOOP_CONFIG), True, max_num_batched_tokens=256)
    prompt = list(range(2, 202))
    decoded = _greedy(llm, [prompt])[0]
    full = llm.generate(
        [{"prompt_token_ids": prompt + list(decoded.token_ids)}],
        SamplingParams(
            max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False
        ),
    )[0]
    assert full.prompt_logprobs is not None and decoded.logprobs is not None
    for step, token in enumerate(decoded.token_ids):
        prefill = full.prompt_logprobs[len(prompt) + step]
        assert prefill is not None and token in prefill
        assert (
            abs(prefill[token].logprob - decoded.logprobs[step][token].logprob) < 2e-2
        )
        assert prefill[token].rank == 1


def _redraw_routers(model: torch.nn.Module) -> None:
    """Dummy weights route every token alike and leave every loop pass
    alike: redraw the routers and message controllers, and record each MoE
    call's (sorted) torch reference routing in call order."""
    from vllm.models.inkling2.nvidia.experts_quant import (
        Inkling2QuantMoE,
        route_reference,
    )
    from vllm.models.inkling2.nvidia.loop import Inkling2LoopDecoderLayer
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    calls: list[torch.Tensor] = []
    model.routing_calls = calls
    generator = torch.Generator(device="cuda").manual_seed(0)

    def hook(moe, args):
        ids, _ = route_reference(
            args[0], moe.gate.weight, moe.gate.bias, moe.topk, moe.route_scale
        )
        with gpu_sync_allowed():
            calls.append(ids.sort(-1).values.cpu())

    for module in model.modules():
        if isinstance(module, Inkling2LoopDecoderLayer):
            down, gain_forget = module.controller_down, module.controller_gain_forget
            down.normal_(std=down.shape[-1] ** -0.5, generator=generator)
            gain_forget.normal_(std=gain_forget.shape[-1] ** -0.5, generator=generator)
        if isinstance(module, Inkling2QuantMoE):
            module.gate.weight.normal_(generator=generator)
            module.gate.bias.zero_()
            module.register_forward_pre_hook(hook)


def _routing_calls(model: torch.nn.Module) -> list[torch.Tensor]:
    return model.routing_calls


@pytest.mark.parametrize("looped", [False, True])
def test_tiny_model_routed_experts_per_visit(tmp_path, enable_pickle, looped):
    """Routed-experts capture keeps a row per layer visit in execution order
    on the quant-method experts too (a loop core layer has one per pass)."""
    llm = _llm(
        _write(tmp_path, TINY_LOOP_CONFIG if looped else TINY_CONFIG),
        looped,
        enforce_eager=True,
        enable_return_routed_experts=True,
    )
    num_visits = 8 if looped else 2
    llm.apply_model(_redraw_routers)
    prompt = list(range(2, 202))
    output = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
    )[0]
    calls = llm.apply_model(_routing_calls)[0]
    assert len(calls) == num_visits
    experts = output.outputs[0].routed_experts
    assert experts is not None and experts.shape[1:] == (num_visits, 8)
    n = len(prompt) - 1
    captured = torch.from_numpy(experts[:n].astype("int64")).sort(-1).values
    for visit, call in enumerate(calls):
        # The Triton router and the torch reference may break exact ties
        # differently; random gates leave none in practice.
        torch.testing.assert_close(captured[:, visit], call[:n], atol=0, rtol=0)


@pytest.mark.parametrize("looped", [False, True])
@pytest.mark.parametrize("draft_sample_method", ["greedy", "probabilistic"])
def test_tiny_model_dspark_speculates_on_sm8x(tmp_path, looped, draft_sample_method):
    from tests.v1.e2e.spec_decode.utils import get_spec_decode_metric_value

    config = copy.deepcopy(TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    llm = _llm(
        _write(tmp_path, config),
        looped,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 7,
            "draft_sample_method": draft_sample_method,
        },
        max_num_batched_tokens=256,
        disable_log_stats=False,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in (list(range(2, 300)), [5, 5, 5])],
        SamplingParams(
            max_tokens=24,
            temperature=1.0 if draft_sample_method == "probabilistic" else 0,
            ignore_eos=True,
            detokenize=False,
        ),
    )
    for output in outputs:
        assert len(output.outputs[0].token_ids) == 24
    metrics = llm.get_metrics()
    drafted = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_draft_tokens")
    accepted = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    assert drafted > 0 and 0 <= accepted <= drafted


# --------------------------------------------------------------------------
# An out-of-tree quantization config (the GGUF plugin's integration point)
# --------------------------------------------------------------------------

_OOT_NAME = "inkling2_test_unquantized"


def _register_oot_quant_config() -> None:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    if _OOT_NAME in QUANTIZATION_METHODS:
        return

    @register_quantization_config(_OOT_NAME)
    class _UnquantizedPlugin(QuantizationConfig):
        """Hands out vLLM's unquantized methods, as a plugin would its own."""

        seen: list[str] = []

        def get_name(self):
            return _OOT_NAME

        def get_supported_act_dtypes(self):
            return [torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            return 80

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config):
            return cls()

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            return _OOT_NAME if hf_quant_cfg.get("quant_method") == _OOT_NAME else None

        def get_quant_method(self, layer, prefix):
            self.seen.append(prefix)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            if isinstance(layer, VocabParallelEmbedding):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, RoutedExperts):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return None


def _quant_prefixes(model: torch.nn.Module) -> tuple[str, list[str], str]:
    attn = model.model.layers[0].attn
    return (
        model.quant.backend.value,
        sorted(
            name
            for name, module in model.named_modules()
            if getattr(module, "quant_method", None) is not None
        ),
        type(attn.q_proj.quant_method).__name__,
    )


@pytest.mark.parametrize("looped", [False, True])
def test_oot_quant_config_runs_split_linears_and_routed_experts(
    tmp_path, monkeypatch, enable_pickle, looped
):
    # The engine core and worker run in this process, where the config is
    # registered.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _register_oot_quant_config()
    config = copy.deepcopy(TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    config["quantization_config"] = {"quant_method": _OOT_NAME}
    model = _write(tmp_path, config)
    llm = _llm(model, looped, enforce_eager=True, max_num_batched_tokens=256)
    backend, names, q_method = llm.apply_model(_quant_prefixes)[0]
    assert backend == "quant_method" and q_method == "UnquantizedLinearMethod"
    expected = {
        "model.embed_tokens",
        "lm_head",
        "model.layers.0.attn.q_proj",
        "model.layers.0.attn.k_proj",
        "model.layers.0.attn.v_proj",
        "model.layers.0.attn.wo_ud",
        "model.layers.0.mlp.experts",
    }
    assert expected <= set(names)
    prompt = list(range(2, 202))
    decoded = _greedy(llm, [prompt])[0]
    assert len(decoded.token_ids) == 16
    full = llm.generate(
        [{"prompt_token_ids": prompt + list(decoded.token_ids)}],
        SamplingParams(
            max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False
        ),
    )[0]
    for step, token in enumerate(decoded.token_ids):
        prefill = full.prompt_logprobs[len(prompt) + step]
        assert (
            abs(prefill[token].logprob - decoded.logprobs[step][token].logprob) < 2e-2
        )
