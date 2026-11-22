# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`Base` hooks its decoder layers exactly when an intervention is configured.

Builds real Transformers-backend models (with vLLM attention layers) on CPU
from tiny configs. Nothing is run: building is what decides which layers get
intervention buffers and a hooked forward.
"""

import contextlib
import os
import tempfile

import pytest
import torch
from transformers import AutoConfig

from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    STANDARD_STEERING_HOOKS,
)
from vllm.model_executor.models.transformers.interventions import (
    UnsupportedInterventionLayout,
)

# The module-scoped `single_rank` group must outlive each test; the global
# per-test cleanup would tear it down after the first one.
pytestmark = pytest.mark.skip_global_cleanup

LAYERS = 3


def _allow_cpu_config(patch: pytest.MonkeyPatch) -> None:
    """The CPU platform refuses to configure vLLM without triton-cpu. Nothing
    here runs a kernel, so on CPU its checks are skipped to build the model."""
    from vllm.platforms import current_platform

    if current_platform.is_cpu():
        patch.setattr(
            type(current_platform),
            "check_and_update_config",
            classmethod(lambda *_: None),
        )


@pytest.fixture(scope="module")
def single_rank():
    """One-rank distributed state, so vLLM layers can be constructed."""
    with pytest.MonkeyPatch.context() as patch:
        _allow_cpu_config(patch)
        yield from _single_rank()


def _single_rank():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
        yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def _tiny(model_type: str, path: str, **overrides) -> str:
    """Save a tiny `model_type` config to `path`; an override of `None` drops
    that default."""
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    kwargs.update(overrides)
    config = AutoConfig.for_model(
        model_type, **{k: v for k, v in kwargs.items() if v is not None}
    )
    config.save_pretrained(path)
    return path


def _build(path: str, model_cls_name: str, monkeypatch, **intervention_configs):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.models import transformers as backend

    _allow_cpu_config(monkeypatch)
    model_config = ModelConfig(
        model=path,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
    )
    vllm_config = VllmConfig(model_config=model_config, **intervention_configs)
    with set_current_vllm_config(vllm_config), torch.device("cpu"):
        return getattr(backend, model_cls_name)(vllm_config=vllm_config)


def _layers(model):
    layers = model._decoder_layers
    assert layers is not None and len(layers) == LAYERS
    return layers


def _steering():
    from vllm.config.steering import SteeringConfig

    return {"steering_config": SteeringConfig(max_steering_configs=2)}


def test_unconfigured_server_keeps_upstream_forward(single_rank, monkeypatch, tmp_path):
    model = _build(
        _tiny("llama", str(tmp_path)), "TransformersForCausalLM", monkeypatch
    )
    for layer in _layers(model):
        assert "forward" not in vars(layer)
        assert not hasattr(layer, "steering_index")


def test_steering_hooks_every_layer(single_rank, monkeypatch, tmp_path):
    model = _build(
        _tiny("llama", str(tmp_path)),
        "TransformersForCausalLM",
        monkeypatch,
        **_steering(),
    )
    layers = _layers(model)
    for i, layer in enumerate(layers):
        assert "forward" in vars(layer)
        assert layer.layer_idx == i
        for hook in STANDARD_STEERING_HOOKS:
            table = getattr(layer, HOOK_POINT_TABLE_ATTR[hook])
            # Tables are in the model dtype so the gather needs no cast.
            assert table.dtype == torch.bfloat16
            assert table.shape[1] == 64
    assert layers[0].steering_index is layers[-1].steering_index
    # The runner discovers steerable layers by `layer_idx` + table buffers.
    steerable = {
        m.layer_idx
        for m in model.modules()
        if hasattr(m, "layer_idx")
        and any(hasattr(m, attr) for attr in HOOK_POINT_TABLE_ATTR.values())
    }
    assert steerable == set(range(LAYERS))


def test_capture_only_hooks_without_tables(single_rank, monkeypatch, tmp_path):
    from vllm.config import CaptureConsumersConfig

    model = _build(
        _tiny("llama", str(tmp_path)),
        "TransformersForCausalLM",
        monkeypatch,
        capture_consumers_config=CaptureConsumersConfig(consumers=[]),
    )
    for layer in _layers(model):
        assert "forward" in vars(layer)
        assert not hasattr(layer, HOOK_POINT_TABLE_ATTR[STANDARD_STEERING_HOOKS[0]])


def test_only_pipeline_local_layers_are_hooked(single_rank, monkeypatch, tmp_path):
    """Layers outside this rank's `_pp_indices` are not built and not hooked."""
    from vllm.model_executor.models.transformers.base import Base

    monkeypatch.setattr(Base, "_pp_indices", lambda self: (1, LAYERS))
    model = _build(
        _tiny("llama", str(tmp_path)),
        "TransformersForCausalLM",
        monkeypatch,
        **_steering(),
    )
    layers = _layers(model)
    assert "forward" not in vars(layers[0])
    assert [layers[i].layer_idx for i in range(1, LAYERS)] == [1, 2]


def test_moe_model_is_hooked(single_rank, monkeypatch, tmp_path):
    path = _tiny(
        "qwen2_moe",
        str(tmp_path),
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )
    model = _build(path, "TransformersMoEForCausalLM", monkeypatch, **_steering())
    for layer in _layers(model):
        assert "forward" in vars(layer)


def test_multimodal_language_model_is_hooked(single_rank, monkeypatch, tmp_path):
    """Llava built through the backend: its Llama language model is hooked."""
    from transformers import LlavaConfig

    text = AutoConfig.for_model(
        "llama",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=160,
        pad_token_id=0,
    )
    vision = AutoConfig.for_model(
        "clip_vision_model",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
        projection_dim=32,
    )
    LlavaConfig(
        text_config=text,
        vision_config=vision,
        image_token_index=150,
        architectures=["LlavaForConditionalGeneration"],
    ).save_pretrained(str(tmp_path))
    model = _build(
        str(tmp_path), "TransformersMultiModalForCausalLM", monkeypatch, **_steering()
    )
    layers = _layers(model)
    for i, layer in enumerate(layers):
        assert type(layer).__name__ == "LlamaDecoderLayer"
        assert layer.layer_idx == i
        assert "forward" in vars(layer)
    # The vision tower is not an intervention site.
    steerable = [m for m in model.modules() if hasattr(m, "steering_index")]
    assert len(steerable) == LAYERS


def test_unsupported_layer_fails_loudly(single_rank, monkeypatch, tmp_path):
    """A decoder layer that cannot be hooked must fail the build when an
    intervention is configured rather than leave it inert, and still build
    when none is. No real Transformers-backed architecture is unsupported, so
    the analysis is made to refuse."""

    def refuse(layer, is_attention=None):
        raise UnsupportedInterventionLayout(type(layer).__name__, "test refusal")

    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.interventions.install.analyze_layer",
        refuse,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.interventions.install._HOOKED_FORWARDS",
        {},
    )
    path = _tiny("llama", str(tmp_path))
    with pytest.raises(UnsupportedInterventionLayout, match="LlamaDecoderLayer"):
        _build(path, "TransformersForCausalLM", monkeypatch, **_steering())
    _build(path, "TransformersForCausalLM", monkeypatch)
