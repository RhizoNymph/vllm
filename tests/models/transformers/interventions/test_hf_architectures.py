# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intervention hooks on the real Hugging Face decoder layers the registry
routes to the Transformers backend, built from tiny random configs (CPU only).

The layers run their own (eager) attention here rather than vLLM's, so this
checks the source analysis and rewrite against the actual modeling code: the
hooked model must match the unhooked one, and every hook point must observe
the tensor its native definition names, cross-checked against the hidden
states Transformers itself records.
"""

import pytest
import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel

from tests.models.transformers.interventions.helpers import (
    HOOKS,
    activate,
    intercept_capture,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.model_executor.models.transformers.interventions import (
    BufferSizes,
    ParallelResidual,
    analyze_layer,
    install_interventions,
)

HIDDEN = 64
LAYERS = 3
TOKENS = 6

_SMALL = dict(
    hidden_size=HIDDEN,
    intermediate_size=128,
    num_hidden_layers=LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
_SMALL_MOE = dict(
    num_experts=4,
    num_local_experts=4,
    num_experts_per_tok=2,
    moe_topk=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
)

# (model_type, overrides) for every architecture `_TRANSFORMERS_SUPPORTED_MODELS`
# routes to the backend whose language model is a standard decoder, with both
# arms of the parallel-residual flag where one exists.
ARCHITECTURES = [
    pytest.param("arcee", {}, id="arcee"),
    pytest.param("cwm", {}, id="cwm"),
    pytest.param("flex_olmo", _SMALL_MOE, id="flex_olmo"),
    pytest.param("gemma", {"head_dim": 16}, id="gemma"),
    pytest.param("glm", {"head_dim": 16}, id="glm"),
    pytest.param("gpt_bigcode", {}, id="gpt_bigcode"),
    pytest.param("gpt_neox", {"use_parallel_residual": True}, id="gpt_neox-parallel"),
    pytest.param("gpt_neox", {"use_parallel_residual": False}, id="gpt_neox"),
    pytest.param("hunyuan_v1_dense", {"head_dim": 16}, id="hunyuan_v1_dense"),
    pytest.param(
        "hunyuan_v1_moe", {"head_dim": 16, **_SMALL_MOE}, id="hunyuan_v1_moe"
    ),
    pytest.param("jais2", {}, id="jais2"),
    pytest.param("kolibri1", _SMALL_MOE, id="kolibri1"),
    pytest.param("llama", {}, id="llama"),
    pytest.param("mellum", _SMALL_MOE, id="mellum"),
    pytest.param("mistral", {}, id="mistral"),
    pytest.param("olmo", {}, id="olmo"),
    pytest.param("olmo2", {}, id="olmo2"),
    pytest.param("olmo3", {}, id="olmo3"),
    pytest.param("olmoe", _SMALL_MOE, id="olmoe"),
    pytest.param("phi", {}, id="phi"),
    pytest.param("phi3", {}, id="phi3"),
    pytest.param("qwen2", {}, id="qwen2"),
    pytest.param("qwen2_moe", _SMALL_MOE, id="qwen2_moe"),
    pytest.param("seed_oss", {"head_dim": 16}, id="seed_oss"),
    pytest.param("smollm3", {}, id="smollm3"),
    pytest.param("stablelm", {"use_parallel_residual": True}, id="stablelm-parallel"),
    pytest.param("stablelm", {"use_parallel_residual": False}, id="stablelm"),
    pytest.param("starcoder2", {}, id="starcoder2"),
    pytest.param("vaultgemma", {"head_dim": 16}, id="vaultgemma"),
    # Composite models: the language model the backend hooks.
    pytest.param("aria_text", _SMALL_MOE, id="aria_text"),
]


def is_hf_attention(module: nn.Module) -> bool:
    """Stands in for the backend's predicate (a vLLM attention layer inside the
    module): here the layers dispatch their own eager attention."""
    return type(module).__name__.endswith("Attention")


def _build(model_type: str, overrides: dict) -> nn.Module:
    try:
        config = AutoConfig.for_model(model_type)
    except (KeyError, ValueError):
        pytest.skip(f"transformers {transformers.__version__} has no {model_type}")
    for key, value in {**_SMALL, **overrides}.items():
        if key in _SMALL_MOE and not hasattr(config, key):
            continue
        setattr(config, key, value)
    torch.manual_seed(0)
    model = AutoModel.from_config(
        config, attn_implementation="eager", dtype=torch.float32
    ).eval()
    # MoE inits can leave routers at zero; random weights keep routing real.
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:
                param.normal_(0.0, 0.2)
    return model


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model
    return next(
        m for m in decoder.modules() if isinstance(m, nn.ModuleList) and len(m) == LAYERS
    )


def _run(model: nn.Module, x: torch.Tensor):
    out = model(
        inputs_embeds=x,
        position_ids=torch.arange(TOKENS)[None],
        output_hidden_states=True,
        use_cache=False,
    )
    return out.last_hidden_state, out.hidden_states


@pytest.mark.parametrize(("model_type", "overrides"), ARCHITECTURES)
def test_hf_architecture(model_type, overrides, monkeypatch):
    model = _build(model_type, overrides)
    layers = _decoder_layers(model)
    plans = [analyze_layer(layer, is_hf_attention) for layer in layers]
    parallel = any(
        isinstance(layout, ParallelResidual) for plan in plans for layout in plan.layouts
    )
    mlp_child = plans[0].layouts[0].mlp_input.child

    torch.manual_seed(1)
    x = torch.randn(1, TOKENS, HIDDEN)
    with torch.no_grad():
        expected, _ = _run(model, x)

    install_interventions(
        layers,
        range(LAYERS),
        hidden_size=HIDDEN,
        sizes=BufferSizes(max_tokens=TOKENS, max_configs=2, dtype=torch.float32),
        is_attention=is_hf_attention,
    )

    mlp_inputs: dict[int, torch.Tensor] = {}
    for i, layer in enumerate(layers):

        def record(module, args, kwargs, i=i):
            value = args[0] if args else next(iter(kwargs.values()))
            mlp_inputs[i] = value.reshape(-1, HIDDEN).clone()

        getattr(layer, mlp_child).register_forward_pre_hook(record, with_kwargs=True)

    seen = intercept_capture(monkeypatch)
    with torch.no_grad():
        actual, hidden_states = _run(model, x)

    # Hooking with nothing active is exact on sequential blocks; parallel
    # blocks re-associate `residual + attn + mlp`.
    if parallel:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(actual, expected)

    captured = {(layer_idx, hook): t for layer_idx, hook, t in seen}
    assert len(captured) == len(seen) == LAYERS * len(HOOKS)
    for i in range(LAYERS):
        get = lambda hook, i=i: captured[(i, hook.value)]  # noqa: E731
        # pre_attn is the layer input: Transformers' hidden_states[i].
        assert torch.equal(get(SteeringHookPoint.PRE_ATTN), hidden_states[i][0])
        if i + 1 < LAYERS:
            # post_block is the layer output: hidden_states[i + 1].
            assert torch.equal(
                get(SteeringHookPoint.POST_BLOCK), hidden_states[i + 1][0]
            )
        assert torch.equal(get(SteeringHookPoint.MLP_IN), mlp_inputs[i])
        torch.testing.assert_close(
            get(SteeringHookPoint.POST_BLOCK),
            get(SteeringHookPoint.POST_ATTN) + get(SteeringHookPoint.MLP_OUT),
            rtol=1e-5,
            atol=1e-5,
        )

    monkeypatch.undo()
    with torch.no_grad():
        activate(layers[0], SteeringHookPoint.POST_BLOCK, torch.zeros(HIDDEN), TOKENS)
        zero_steered, _ = _run(model, x)
        # Not a constant vector: a final LayerNorm would subtract it out.
        vec = torch.linspace(-4.0, 4.0, HIDDEN)
        activate(layers[0], SteeringHookPoint.POST_BLOCK, vec, TOKENS)
        steered, _ = _run(model, x)
    assert torch.equal(zero_steered, actual)
    assert not torch.allclose(steered, actual, atol=1e-3)
