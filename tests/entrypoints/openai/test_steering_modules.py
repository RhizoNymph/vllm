# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for steering module registry and merge_steering_specs helper."""

from __future__ import annotations

import json
import math
import os
import tempfile
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.datastructures import State

from vllm.config.steering_types import (
    SteeringVectorSpec,
    merge_steering_specs,
)
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.steering.registry import (
    SteeringModuleRegistry,
    _convert_layer_keys,
)

# ---------------------------------------------------------------------------
# merge_steering_specs tests
# ---------------------------------------------------------------------------


class TestMergeSteeringSpecs:
    """Tests for :func:`merge_steering_specs`."""

    def test_both_none_returns_none(self):
        assert merge_steering_specs(None, None) is None

    def test_both_empty_returns_none(self):
        assert merge_steering_specs({}, {}) is None

    def test_first_none_second_has_data(self):
        spec: SteeringVectorSpec = {
            "post_mlp": {14: [1.0, 2.0, 3.0]},
        }
        result = merge_steering_specs(None, spec)
        assert result is not None
        # Values should be pre-scaled (scale=1.0 for bare list)
        assert result["post_mlp"][14] == [1.0, 2.0, 3.0]

    def test_first_has_data_second_none(self):
        spec: SteeringVectorSpec = {
            "pre_attn": {5: [0.5, 0.6]},
        }
        result = merge_steering_specs(spec, None)
        assert result is not None
        assert result["pre_attn"][5] == [0.5, 0.6]

    def test_non_overlapping_hooks_both_preserved(self):
        a: SteeringVectorSpec = {"post_mlp": {14: [1.0, 2.0]}}
        b: SteeringVectorSpec = {"pre_attn": {10: [3.0, 4.0]}}
        result = merge_steering_specs(a, b)
        assert result is not None
        assert result["post_mlp"][14] == [1.0, 2.0]
        assert result["pre_attn"][10] == [3.0, 4.0]

    def test_non_overlapping_layers_same_hook(self):
        a: SteeringVectorSpec = {"post_mlp": {14: [1.0, 2.0]}}
        b: SteeringVectorSpec = {"post_mlp": {15: [3.0, 4.0]}}
        result = merge_steering_specs(a, b)
        assert result is not None
        assert result["post_mlp"][14] == [1.0, 2.0]
        assert result["post_mlp"][15] == [3.0, 4.0]

    def test_overlapping_hook_layer_added(self):
        a: SteeringVectorSpec = {"post_mlp": {14: [1.0, 2.0, 3.0]}}
        b: SteeringVectorSpec = {"post_mlp": {14: [0.5, 0.5, 0.5]}}
        result = merge_steering_specs(a, b)
        assert result is not None
        assert result["post_mlp"][14] == [1.5, 2.5, 3.5]

    def test_overlapping_with_scaled_entries(self):
        a: SteeringVectorSpec = {
            "post_mlp": {
                14: {"vector": [1.0, 2.0], "scale": 2.0},
            }
        }
        b: SteeringVectorSpec = {
            "post_mlp": {
                14: {"vector": [3.0, 4.0], "scale": 0.5},
            }
        }
        result = merge_steering_specs(a, b)
        assert result is not None
        # a scaled: [2.0, 4.0], b scaled: [1.5, 2.0], sum: [3.5, 6.0]
        assert result["post_mlp"][14] == [3.5, 6.0]

    def test_one_scaled_one_bare(self):
        a: SteeringVectorSpec = {
            "post_mlp": {
                14: {"vector": [1.0, 2.0], "scale": 3.0},
            }
        }
        b: SteeringVectorSpec = {
            "post_mlp": {
                14: [0.5, 0.5],
            }
        }
        result = merge_steering_specs(a, b)
        assert result is not None
        # a scaled: [3.0, 6.0], b scaled: [0.5, 0.5], sum: [3.5, 6.5]
        assert result["post_mlp"][14] == [3.5, 6.5]

    def test_passthrough_entry_is_prescaled(self):
        """Non-overlapping scaled entry should still be pre-scaled."""
        spec: SteeringVectorSpec = {
            "post_mlp": {
                14: {"vector": [1.0, 2.0], "scale": 0.5},
            }
        }
        result = merge_steering_specs(spec, None)
        assert result is not None
        assert result["post_mlp"][14] == [0.5, 1.0]


# ---------------------------------------------------------------------------
# _convert_layer_keys tests
# ---------------------------------------------------------------------------


class TestConvertLayerKeys:
    """Tests for the helper that converts JSON string keys to int."""

    def test_none_returns_none(self):
        assert _convert_layer_keys(None, field_name="vectors") is None

    def test_empty_dict_returns_none(self):
        assert _convert_layer_keys({}, field_name="vectors") is None

    def test_converts_string_keys_to_int(self):
        spec = {"post_mlp": {"14": [1.0, 2.0], "15": [3.0, 4.0]}}
        result = _convert_layer_keys(spec, field_name="vectors")
        assert result is not None
        assert 14 in result["post_mlp"]
        assert 15 in result["post_mlp"]
        assert result["post_mlp"][14] == [1.0, 2.0]

    def test_rejects_non_dict_layers(self):
        spec = {"post_mlp": "not_a_dict"}
        with pytest.raises(ValueError, match="must be a JSON object mapping"):
            _convert_layer_keys(spec, field_name="vectors")

    def test_rejects_non_dict_spec(self):
        with pytest.raises(ValueError, match="field 'vectors' must be a JSON object"):
            _convert_layer_keys(["not", "a", "dict"], field_name="vectors")


# ---------------------------------------------------------------------------
# SteeringModuleRegistry tests
# ---------------------------------------------------------------------------


class TestSteeringModuleRegistry:
    """Tests for :class:`SteeringModuleRegistry`."""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            name="test_mod",
            vectors={"post_mlp": {14: [1.0, 2.0]}},
        )
        module = registry.get("test_mod")
        assert module is not None
        assert module.name == "test_mod"
        assert module.vectors == {"post_mlp": {14: [1.0, 2.0]}}

    @pytest.mark.asyncio
    async def test_register_overwrites_existing(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            name="mod",
            vectors={"post_mlp": {14: [1.0]}},
        )
        await registry.register(
            name="mod",
            vectors={"pre_attn": {5: [2.0]}},
        )
        module = registry.get("mod")
        assert module is not None
        assert "pre_attn" in module.vectors
        assert "post_mlp" not in module.vectors

    @pytest.mark.asyncio
    async def test_unregister_existing_returns_true(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            name="mod",
            vectors={"post_mlp": {14: [1.0]}},
        )
        assert await registry.unregister("mod") is True
        assert registry.get("mod") is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_returns_false(self):
        registry = SteeringModuleRegistry()
        assert await registry.unregister("nope") is False

    def test_get_nonexistent_returns_none(self):
        registry = SteeringModuleRegistry()
        assert registry.get("missing") is None

    @pytest.mark.asyncio
    async def test_list_modules_sorted(self):
        registry = SteeringModuleRegistry()
        await registry.register("charlie", vectors={"post_mlp": {0: [1.0]}})
        await registry.register("alpha", vectors={"post_mlp": {0: [1.0]}})
        await registry.register("bravo", vectors={"post_mlp": {0: [1.0]}})
        assert registry.list_modules() == ["alpha", "bravo", "charlie"]

    @pytest.mark.asyncio
    async def test_list_modules_empty(self):
        registry = SteeringModuleRegistry()
        assert registry.list_modules() == []

    @pytest.mark.asyncio
    async def test_register_no_vectors_raises(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="has no vectors in any tier"):
            await registry.register(name="empty")

    @pytest.mark.asyncio
    async def test_register_invalid_hook_point_raises(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="Invalid hook point name"):
            await registry.register(
                name="bad_hook",
                vectors={"totally_invalid": {0: [1.0]}},
            )

    @pytest.mark.asyncio
    async def test_register_unknown_layer_index_raises(self):
        registry = SteeringModuleRegistry(valid_layer_indices={0, 1})
        with pytest.raises(ValueError, match="unknown layer index 99"):
            await registry.register(
                name="bad_layer",
                vectors={"post_mlp": {99: [1.0]}},
            )

    @pytest.mark.asyncio
    async def test_register_malformed_entry_raises(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(TypeError):
            await registry.register(
                name="bad_entry",
                vectors={"post_mlp": {0: "not_a_list_or_dict"}},
            )

    @pytest.mark.asyncio
    async def test_register_invalid_vector_contents_raise(self):
        registry = SteeringModuleRegistry()

        with pytest.raises(ValueError, match="must be a finite float"):
            await registry.register(
                name="bad_values",
                vectors={
                    "post_mlp": {
                        0: {
                            "vector": ["bad", 1.0],
                            "scale": 1.0,
                        }
                    }
                },
            )

        with pytest.raises(ValueError, match="must be finite"):
            await registry.register(
                name="bad_scale",
                vectors={
                    "post_mlp": {
                        0: {
                            "vector": [1.0, 2.0],
                            "scale": math.nan,
                        }
                    }
                },
            )

    # --- load_from_file tests ---

    @pytest.mark.asyncio
    async def test_load_from_file_valid_json(self):
        registry = SteeringModuleRegistry()
        data = {
            "vectors": {"post_mlp": {"14": [0.1, 0.2, 0.3]}},
            "prefill_vectors": {"pre_attn": {"5": [0.4, 0.5, 0.6]}},
            "decode_vectors": None,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            await registry.load_from_file("loaded", tmp_path)
            module = registry.get("loaded")
            assert module is not None
            assert module.name == "loaded"
            # Layer keys should be ints
            assert 14 in module.vectors["post_mlp"]
            assert 5 in module.prefill_vectors["pre_attn"]
            assert module.decode_vectors is None
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_from_file_missing_raises(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(FileNotFoundError, match="not found"):
            await registry.load_from_file("missing", "/nonexistent/path.json")

    @pytest.mark.asyncio
    async def test_load_from_file_converts_string_keys(self):
        registry = SteeringModuleRegistry()
        data = {
            "vectors": {
                "post_mlp": {"0": [1.0], "99": [2.0]},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            await registry.load_from_file("conv_keys", tmp_path)
            module = registry.get("conv_keys")
            assert module is not None
            assert 0 in module.vectors["post_mlp"]
            assert 99 in module.vectors["post_mlp"]
            # String keys should NOT be present
            assert "0" not in module.vectors["post_mlp"]
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_from_file_non_dict_raises(self):
        registry = SteeringModuleRegistry()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="JSON object"):
                await registry.load_from_file("bad_fmt", tmp_path)
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_from_file_invalid_vector_contents_raise(self):
        registry = SteeringModuleRegistry()
        data = {
            "vectors": {
                "post_mlp": {
                    "14": {
                        "vector": [1.0, "bad"],
                        "scale": 1.0,
                    }
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="must be a finite float"):
                await registry.load_from_file("bad_values", tmp_path)
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_load_from_file_rejects_non_dict_hook_payload(self):
        registry = SteeringModuleRegistry()
        data = {
            "vectors": {
                "post_mlp": [1.0, 2.0],
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="must be a JSON object mapping"):
                await registry.load_from_file("bad_hook_payload", tmp_path)
        finally:
            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_init_app_state_only_sets_registry_when_steering_enabled():
    engine_client = MagicMock()
    engine_client.vllm_config = SimpleNamespace(lora_config=None)
    engine_client.model_config = MagicMock()
    engine_client.renderer = MagicMock()
    engine_client.io_processor = MagicMock()
    engine_client.collective_rpc = AsyncMock(
        return_value=[{0: ["post_mlp"], 1: ["post_mlp"]}]
    )

    args = Namespace(
        served_model_name=None,
        model="test-model",
        enable_log_requests=False,
        max_log_len=None,
        disable_log_stats=False,
        chat_template=None,
        lora_modules=None,
        enable_steering=False,
        steering_modules=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        enable_auto_tool_choice=False,
        exclude_tools_when_tool_choice_none=False,
        tool_call_parser=None,
        default_chat_template_kwargs=None,
        log_error_stack=False,
        enable_server_load_tracking=False,
    )

    models = MagicMock()
    models.registry = MagicMock()
    models.init_static_loras = AsyncMock()

    state = State()

    with (
        patch(
            "vllm.entrypoints.openai.api_server.load_chat_template",
            return_value=None,
        ),
        patch(
            "vllm.entrypoints.openai.api_server.process_lora_modules",
            return_value=[],
        ),
        patch(
            "vllm.entrypoints.openai.api_server.OpenAIServingModels",
            return_value=models,
        ),
        patch("vllm.entrypoints.openai.api_server.OpenAIServingRender"),
        patch("vllm.entrypoints.openai.api_server.OpenAIServingTokenization"),
    ):
        await init_app_state(
            engine_client,
            state,
            args,
            supported_tasks=(),
        )

    assert not hasattr(state, "steering_module_registry")

    args.enable_steering = True

    with (
        patch(
            "vllm.entrypoints.openai.api_server.load_chat_template",
            return_value=None,
        ),
        patch(
            "vllm.entrypoints.openai.api_server.process_lora_modules",
            return_value=[],
        ),
        patch(
            "vllm.entrypoints.openai.api_server.OpenAIServingModels",
            return_value=models,
        ),
        patch("vllm.entrypoints.openai.api_server.OpenAIServingRender"),
        patch("vllm.entrypoints.openai.api_server.OpenAIServingTokenization"),
    ):
        await init_app_state(
            engine_client,
            state,
            args,
            supported_tasks=(),
        )

    assert hasattr(state, "steering_module_registry")
    engine_client.collective_rpc.assert_awaited_once_with("list_steerable_layers")


# ---------------------------------------------------------------------------
# resolve_for_request tests
# ---------------------------------------------------------------------------


class TestResolveForRequest:
    """Tests for :meth:`SteeringModuleRegistry.resolve_for_request`."""

    @pytest.mark.asyncio
    async def test_unknown_name_returns_error(self):
        registry = SteeringModuleRegistry()
        v, p, d, err = registry.resolve_for_request("unknown", None, None, None)
        assert v is None
        assert p is None
        assert d is None
        assert err is not None
        assert "Unknown steering module" in err

    @pytest.mark.asyncio
    async def test_known_name_no_inline(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            "my_mod",
            vectors={"post_mlp": {14: [1.0, 2.0]}},
            prefill_vectors={"pre_attn": {5: [0.5, 0.6]}},
        )
        v, p, d, err = registry.resolve_for_request("my_mod", None, None, None)
        assert err is None
        # Vectors are pre-scaled (scale=1.0 bare lists)
        assert v is not None
        assert v["post_mlp"][14] == [1.0, 2.0]
        assert p is not None
        assert p["pre_attn"][5] == [0.5, 0.6]
        assert d is None

    @pytest.mark.asyncio
    async def test_known_name_with_inline_merge(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            "base",
            vectors={"post_mlp": {14: [1.0, 2.0]}},
        )
        inline: SteeringVectorSpec = {"post_mlp": {14: [0.5, 0.5]}}
        v, p, d, err = registry.resolve_for_request("base", inline, None, None)
        assert err is None
        assert v is not None
        assert v["post_mlp"][14] == [1.5, 2.5]

    @pytest.mark.asyncio
    async def test_named_one_tier_inline_different_tier(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            "named",
            vectors={"post_mlp": {14: [1.0, 2.0]}},
        )
        inline_prefill: SteeringVectorSpec = {"pre_attn": {5: [0.3, 0.4]}}
        v, p, d, err = registry.resolve_for_request("named", None, inline_prefill, None)
        assert err is None
        # Named vectors tier
        assert v is not None
        assert v["post_mlp"][14] == [1.0, 2.0]
        # Inline prefill tier
        assert p is not None
        assert p["pre_attn"][5] == [0.3, 0.4]
        # Decode tier untouched
        assert d is None

    @pytest.mark.asyncio
    async def test_error_message_lists_available_modules(self):
        registry = SteeringModuleRegistry()
        await registry.register("a", vectors={"post_mlp": {0: [1.0]}})
        await registry.register("b", vectors={"post_mlp": {0: [1.0]}})
        _, _, _, err = registry.resolve_for_request("missing", None, None, None)
        assert err is not None
        assert "['a', 'b']" in err

    @pytest.mark.asyncio
    async def test_dimension_mismatch_returns_error(self):
        registry = SteeringModuleRegistry()
        await registry.register(
            "named",
            vectors={"post_mlp": {14: [1.0, 2.0]}},
        )

        inline: SteeringVectorSpec = {"post_mlp": {14: [0.5]}}
        v, p, d, err = registry.resolve_for_request("named", inline, None, None)

        assert v is None
        assert p is None
        assert d is None
        assert err is not None
        assert "Invalid steering composition for module 'named'" in err
        assert "different lengths: 2 vs 1" in err
