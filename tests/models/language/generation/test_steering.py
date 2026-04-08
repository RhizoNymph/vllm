# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation steering on Gemma 3.

Covers global steering via the worker API, per-request steering via
SamplingParams, concurrent batching with CUDA graphs, and three-tier
prefill/decode phase-specific steering.
"""

import math

import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_VECTOR_ATTR,
)
from ...registry import HF_EXAMPLE_MODELS

MODEL = "google/gemma-3-4b-it"

_SMALL_DECODER_OVERRIDES = {
    "num_hidden_layers": 1,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
}

_QWEN_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_experts": 2,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 0,
}

_QWEN3_NEXT_OVERRIDES = {
    **_QWEN_MOE_OVERRIDES,
    "num_hidden_layers": 2,
    "head_dim": 64,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "layer_types": ["full_attention", "linear_attention"],
}

_LOOPCODER_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "loop_num": 2,
    "loop_window_size": 32,
}

_STABLELM_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
}

_GROK_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
}

_STEP3P5_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "num_attention_groups": 8,
    "head_dim": 64,
    "moe_layers_enum": "1",
    "moe_num_experts": 2,
    "moe_top_k": 1,
    "moe_intermediate_size": 1024,
    "share_expert_dim": 0,
}

_MINIMAX_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "head_dim": 64,
    "rotary_dim": 64,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
}

_MINIMAX_TEXT_OVERRIDES = {
    **_MINIMAX_OVERRIDES,
    "num_local_experts": 1,
}

_AXK1_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "qk_nope_head_dim": 0,
    "qk_rope_head_dim": 0,
    "v_head_dim": 64,
    "first_k_dense_replace": 999,
    "moe_layer_freq": 1,
    "n_routed_experts": None,
}

_TRUST_REMOTE_EAGER = {"trust_remote_code": True, "enforce_eager": True}
_EAGER_ONLY = {"enforce_eager": True}

PHASE1_DISCOVERY_CASES = [
    pytest.param("Qwen/Qwen3-0.6B", None, None, id="qwen3"),
    pytest.param(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        _QWEN3_NEXT_OVERRIDES,
        {"enforce_eager": True, "enable_chunked_prefill": True},
        id="qwen3-next",
    ),
    pytest.param(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen2-moe",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen3-moe",
    ),
    pytest.param(
        "ByteDance/Ouro-1.4B",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="ouro",
    ),
    pytest.param(
        "ByteDance-Seed/Seed-OSS-36B-Instruct",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="seed-oss",
    ),
    pytest.param(
        "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct",
        _LOOPCODER_OVERRIDES,
        {"enforce_eager": True},
        id="loopcoder",
    ),
]

PHASE1_GENERATION_CASES = [
    pytest.param("Qwen/Qwen3-0.6B", None, None, id="qwen3"),
    pytest.param(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        _QWEN3_NEXT_OVERRIDES,
        {"enforce_eager": True, "enable_chunked_prefill": True},
        id="qwen3-next",
    ),
    pytest.param(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen2-moe",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen3-moe",
    ),
    pytest.param(
        "ByteDance-Seed/Seed-OSS-36B-Instruct",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="seed-oss",
    ),
    pytest.param(
        "ByteDance/Ouro-1.4B",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="ouro",
    ),
    pytest.param(
        "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct",
        _LOOPCODER_OVERRIDES,
        {"enforce_eager": True},
        id="loopcoder",
    ),
]

PHASE2_DISCOVERY_CASES = [
    pytest.param(
        "baichuan-inc/Baichuan2-7B-chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="baichuan",
    ),
    pytest.param(
        "internlm/internlm2-chat-7b",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="internlm2",
    ),
    pytest.param(
        "OrionStarAI/Orion-14B-Chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="orion",
    ),
    pytest.param(
        "upstage/solar-pro-preview-instruct",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="solar",
    ),
    pytest.param(
        "stabilityai/stablelm-3b-4e1t",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="stablelm",
    ),
    pytest.param(
        "nvidia/Minitron-8B-Base",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="nemotron",
    ),
    pytest.param(
        "arcee-ai/AFM-4.5B-Base",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="arcee",
    ),
    pytest.param(
        "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="hyperclovax",
    ),
    pytest.param(
        "zai-org/GLM-4-9B-0414",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="glm4",
    ),
    pytest.param(
        "hpcai-tech/grok-1",
        _GROK_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="grok1",
    ),
    pytest.param(
        "CohereLabs/c4ai-command-r7b-12-2024",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="commandr",
    ),
    pytest.param(
        "FreedomIntelligence/openPangu-Embedded-7B-V1.1",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="openpangu",
    ),
]

PHASE2_GENERATION_CASES = [
    pytest.param(
        "baichuan-inc/Baichuan2-7B-chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="baichuan",
    ),
    pytest.param(
        "OrionStarAI/Orion-14B-Chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="orion",
    ),
    pytest.param(
        "CohereLabs/c4ai-command-r7b-12-2024",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="commandr",
    ),
    pytest.param(
        "zai-org/GLM-4-9B-0414",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="glm4",
    ),
    pytest.param(
        "hpcai-tech/grok-1",
        _GROK_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="grok1",
    ),
]

PHASE3_DISCOVERY_CASES = [
    pytest.param(
        "swiss-ai/Apertus-8B-Instruct-2509",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="apertus",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-Text-01",
        _MINIMAX_TEXT_OVERRIDES,
        _EAGER_ONLY,
        id="minimax-text",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-M2",
        _MINIMAX_OVERRIDES,
        _EAGER_ONLY,
        id="minimax-m2",
    ),
    pytest.param(
        "skt/A.X-K1",
        _AXK1_OVERRIDES,
        _EAGER_ONLY,
        id="axk1",
    ),
]

PHASE3_GENERATION_CASES = [
    pytest.param(
        "swiss-ai/Apertus-8B-Instruct-2509",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="apertus",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-M2",
        _MINIMAX_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="minimax-m2",
    ),
]

# Shorthand
_HP = DEFAULT_HOOK_POINT.value
_VEC_ATTR = HOOK_POINT_VECTOR_ATTR[DEFAULT_HOOK_POINT]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_layers(llm):
    """Return (target_layer, hidden_size) for the default hook point."""

    def _discover(worker):
        layers = {}
        model_inst = worker.model_runner.get_model()
        for mod in model_inst.modules():
            if hasattr(mod, _VEC_ATTR) and hasattr(mod, "layer_idx"):
                layers[mod.layer_idx] = getattr(mod, _VEC_ATTR).shape[1]
        return layers

    layer_info = llm.llm.collective_rpc(_discover)[0]
    target_layer = max(layer_info.keys()) // 2
    hidden_size = layer_info[target_layer]
    return target_layer, hidden_size


def _gen_tokens(llm, prompt, sampling):
    """Generate and return token ids list."""
    result = llm.llm.generate([prompt], sampling)
    return list(result[0].outputs[0].token_ids)


def _gen_tokens_and_cumulative_logprob(llm, prompt, sampling):
    """Generate and return token ids with cumulative logprob."""
    result = llm.llm.generate([prompt], sampling)
    output = result[0].outputs[0]
    return list(output.token_ids), output.cumulative_logprob


def _runner_kwargs(hf_overrides: dict | None,
                   extra_runner_kwargs: dict | None = None) -> dict:
    kwargs = {
        "load_format": "dummy",
        "max_model_len": 256,
        "enable_steering": True,
        "max_steering_configs": 4,
    }
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    if extra_runner_kwargs is not None:
        kwargs.update(extra_runner_kwargs)
    return kwargs


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs"),
                         PHASE1_DISCOVERY_CASES)
def test_steering_layers_discovered_for_supported_families(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Steering buffers should be discoverable beyond Gemma3."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(model,
                         **_runner_kwargs(hf_overrides,
                                          extra_runner_kwargs)) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs"),
                         PHASE1_GENERATION_CASES)
def test_phase1_steering_changes_output(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Representative phase-1 families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                f"Steering should change output for {model}"
            )

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs"),
                         PHASE2_DISCOVERY_CASES)
def test_phase2_steering_layers_discovered_for_supported_families(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Phase-2 decoder families should expose steerable layers."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(model,
                         **_runner_kwargs(hf_overrides,
                                          extra_runner_kwargs)) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs",
                          "vector_scale"),
                         PHASE2_GENERATION_CASES)
def test_phase2_steering_changes_output(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Representative phase-2 families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [vector_scale] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert (
                steered_tokens != baseline_tokens
                or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), (
                f"Steering should change output or logprob for {model}"
            )

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Clearing steering should restore baseline logprob for {model}"


def test_phase2_stablelm_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """StableLM needs a real checkpoint to produce a meaningful steering signal."""
    model = "stabilityai/stablelm-3b-4e1t"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert (
                steered_tokens != baseline_tokens
                or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), "Steering should change output or logprob for StableLM"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs"),
                         PHASE3_DISCOVERY_CASES)
def test_phase3_steering_layers_discovered_for_supported_families(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Phase-3 norm-variant decoder families should expose steerable layers."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(model,
                         **_runner_kwargs(hf_overrides,
                                          extra_runner_kwargs)) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(("model", "hf_overrides", "extra_runner_kwargs",
                          "vector_scale"),
                         PHASE3_GENERATION_CASES)
def test_phase3_steering_changes_output(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Representative phase-3 families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [vector_scale] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert (
                steered_tokens != baseline_tokens
                or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), (
                f"Steering should change output or logprob for {model}"
            )

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Clearing steering should restore baseline logprob for {model}"


def test_phase3_step3p5_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """Step-3.5 needs a real checkpoint for stable validation."""
    model = "stepfun-ai/Step-3.5-Flash"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    if not torch.cuda.is_available():
        pytest.skip("Step-3.5 real-weights steering test requires CUDA.")
    total_memory_gib = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
    )
    if total_memory_gib < 40:
        pytest.skip(
            "Step-3.5 real-weights steering test requires a GPU with at least "
            "40 GiB of memory."
        )

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert (
                steered_tokens != baseline_tokens
                or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), "Steering should change output or logprob for Step-3.5"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


# ---------------------------------------------------------------------------
# Existing tests (updated for kwargs-based collective_rpc API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL])
def test_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that non-zero steering vectors change model output
    and that clearing them restores the original behaviour."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
        ) as llm:
            # 1. Baseline (zero steering buffers)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            # Clear the clean prompt from APC so the steered run has to
            # prefill and write its own KV entries before the unsteered
            # replay.
            assert llm.llm.reset_prefix_cache()

            # 2. Discover hidden_size and pick a middle layer.
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set steering via WorkerBase (same path as HTTP API).
            #    With dummy (random) weights the magnitude must be large
            #    enough to overcome noise in the logit space.
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Non-zero steering should change model output"
            )

            # 4. Clear steering and verify output matches baseline
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Clearing steering should restore original output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_via_sampling_params(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that per-request steering_vectors in SamplingParams
    changes output and that different steering produces different results."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with per-request steering via SamplingParams
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, steered_sampling)

            assert steered_tokens != baseline_tokens, (
                "Per-request steering should change model output"
            )

            # 4. Verify baseline is unchanged (no contamination)
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Per-request steering should not contaminate other requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_concurrent_with_cuda_graphs(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Test that different per-request steering configs in the same batch
    produce different outputs, and that CUDA graph replays correctly pick
    up updated steering buffers between steps.

    This sends multiple requests simultaneously so they land in the same
    batch during decode, exercising the request-indexed gather with
    CUDA graphs active.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=False,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 2. Create three requests: no steering, positive, negative
            no_steer = SamplingParams(max_tokens=10, temperature=0.0)
            steer_pos = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )
            steer_neg = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
            )

            # 3. Send all three simultaneously so they batch together
            outputs = llm.llm.generate(
                [prompt, prompt, prompt],
                [no_steer, steer_pos, steer_neg],
            )

            tokens_none = list(outputs[0].outputs[0].token_ids)
            tokens_pos = list(outputs[1].outputs[0].token_ids)
            tokens_neg = list(outputs[2].outputs[0].token_ids)

            # Positive and negative steering should produce different output
            assert tokens_pos != tokens_neg, (
                "Opposite steering vectors should produce different outputs"
            )

            # At least one steered output should differ from unsteered
            assert tokens_pos != tokens_none or tokens_neg != tokens_none, (
                "At least one steered request should differ from unsteered"
            )

            # 4. Run again without steering to verify CUDA graph replays
            #    pick up updated (cleared) buffer contents
            outputs2 = llm.llm.generate(
                [prompt, prompt],
                [no_steer, no_steer],
            )

            tokens_none2 = list(outputs2[0].outputs[0].token_ids)
            tokens_none3 = list(outputs2[1].outputs[0].token_ids)

            # Unsteered should be consistent across runs
            assert tokens_none2 == tokens_none, (
                "Unsteered output should be deterministic across runs"
            )
            assert tokens_none3 == tokens_none, (
                "Both unsteered requests should match baseline"
            )


# ---------------------------------------------------------------------------
# Prefill steering tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that prefill-specific steering via SamplingParams changes
    output and does not contaminate subsequent unsteered requests."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with prefill-specific steering
            prefill_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, prefill_steered)

            assert steered_tokens != baseline_tokens, (
                "Prefill steering should change model output"
            )

            # 4. Reset and verify no contamination
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Prefill steering should not contaminate subsequent requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_decode_only_steering_via_new_field(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that decode_steering_vectors (the new field) changes output
    compared to an unsteered baseline."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with decode-specific steering
            decode_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                decode_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, decode_steered)

            assert steered_tokens != baseline_tokens, (
                "Decode-only steering should change model output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_and_decode_different_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that using different vectors for prefill vs decode produces
    different output than using the same vector for both phases."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Same vector for both phases via base steering_vectors
            both_same = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            result_both = _gen_tokens(llm, prompt, both_same)

            assert llm.llm.reset_prefix_cache()

            # 2. Different vectors for prefill vs decode
            split = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
            )

            result_split = _gen_tokens(llm, prompt, split)

            assert result_both != result_split, (
                "Different prefill vs decode steering should produce "
                "different output than uniform steering"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_additive_composition(vllm_runner, monkeypatch, model: str) -> None:
    """Verify the three-tier additive model works correctly.

    To test additive composition we must ensure BOTH prefill and decode
    effective vectors match between the two approaches.  We use:

    Approach A:  prefill_steering=P, steering_vectors=X, decode_steering=Y
        → prefill_effective = P + X,  decode_effective = X + Y

    Approach B:  prefill_steering=P+X, decode_steering=X+Y
        → prefill_effective = P + X,  decode_effective = X + Y

    Both should produce identical output.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # Approach A: three-tier additive
            # prefill = P(200) + base(300) = 500
            # decode  = base(300) + D(200) = 500
            approach_a = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [200.0] * H},
                },
                steering_vectors={
                    _HP: {target_layer: [300.0] * H},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [200.0] * H},
                },
            )

            result_a = _gen_tokens(llm, prompt, approach_a)

            assert llm.llm.reset_prefix_cache()

            # Approach B: phase-specific only (no base), same totals
            # prefill = 500, decode = 500
            approach_b = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_b = _gen_tokens(llm, prompt, approach_b)

            assert result_a == result_b, (
                "Three-tier additive (P=200 + base=300 + D=200) should "
                "produce same output as direct (P=500, D=500)"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefix_cache_respects_prefill_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that prefix cache correctly separates different prefill
    steering: same prompt with different prefill steering should produce
    different outputs, but same prompt with same prefill steering should
    hit the cache and produce identical output."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling_no_steer = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Request A: with prefill steering
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            tokens_a = _gen_tokens(llm, prompt, steered_sampling)

            # 2. Request B: same prompt, NO prefill steering
            tokens_b = _gen_tokens(llm, prompt, sampling_no_steer)

            # Different prefill steering means different KV cache
            assert tokens_a != tokens_b, (
                "Different prefill steering should produce different output "
                "even with prefix caching enabled"
            )

            # 3. Request C: same prompt, same prefill steering as A
            #    Should hit prefix cache and produce identical output
            tokens_c = _gen_tokens(llm, prompt, steered_sampling)

            assert tokens_c == tokens_a, (
                "Same prefill steering should hit prefix cache and produce "
                "identical output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_co_located_scale(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that the co-located scale format produces the same result
    as a pre-scaled bare vector: [500]*H should equal {"vector": [250]*H,
    "scale": 2.0}."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # 1. Bare vector at magnitude 500
            bare = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_bare = _gen_tokens(llm, prompt, bare)

            assert llm.llm.reset_prefix_cache()

            # 2. Co-located scale: vector=250, scale=2.0 => effective 500
            scaled = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {
                        target_layer: {
                            "vector": [250.0] * H,
                            "scale": 2.0,
                        },
                    },
                },
            )

            result_scaled = _gen_tokens(llm, prompt, scaled)

            assert result_bare == result_scaled, (
                "Co-located scale (250 * 2.0) should produce same output "
                "as bare vector (500)"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_global_prefill_steering_via_worker_api(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify global three-tier steering via the worker API: setting
    prefill-specific global vectors changes output, and clearing them
    restores the baseline."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
        ) as llm:
            # 1. Baseline (no global steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set global prefill-specific vectors via worker API
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "prefill_vectors": {_HP: {target_layer: vec}},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Global prefill steering should change model output"
            )

            # 4. Clear and verify restoration
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Clearing global prefill steering should restore baseline"
            )
