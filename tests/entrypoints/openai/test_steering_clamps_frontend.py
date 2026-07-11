# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend plumbing tests for directional-clamp fields.

Covers:
- ``ChatCompletionRequest`` / ``CompletionRequest`` accept the three clamp
  tier fields and ``to_sampling_params`` int-coerces JSON string layer keys
  and lands canonical (unit-normalized) entries on ``SamplingParams``
- ``coerce_clamp_spec`` layer-key coercion and error cases
- ``SteeringModuleRegistry`` clamps tier: registration, validation,
  broadcast dump, at-least-one-tier check
- ``InputProcessor._validate_steering`` clamp gates: disabled steering,
  disabled clamping (K=0), width mismatch, per-site K cap
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.config.steering import SteeringConfig
from vllm.config.steering_types import coerce_clamp_spec
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.steering.registry import SteeringModuleRegistry
from vllm.sampling_params import SamplingParams

_HIDDEN = 8

_CHAT_BASE = {
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "test-model",
}
_COMPLETION_BASE = {"prompt": "Hello", "model": "test-model"}


def _clamp_entry(axis: int = 0, value: float = 2.0, hidden: int = _HIDDEN) -> dict:
    vec = [0.0] * hidden
    vec[axis] = 2.0  # non-unit on purpose: server must normalize
    return {"vector": vec, "value": value}


def _clamps(layer_key="5", hidden: int = _HIDDEN) -> dict:
    return {"post_block": {layer_key: [_clamp_entry(hidden=hidden)]}}


def _make_chat(**extra):
    return ChatCompletionRequest.model_validate({**_CHAT_BASE, **extra})


def _make_completion(**extra):
    return CompletionRequest.model_validate({**_COMPLETION_BASE, **extra})


class TestCoerceClampSpec:
    def test_none_and_empty(self):
        assert coerce_clamp_spec(None) is None
        assert coerce_clamp_spec({}) is None

    def test_string_layer_keys_coerced(self):
        result = coerce_clamp_spec(_clamps("5"))
        assert 5 in result["post_block"]

    def test_int_layer_keys_pass_through(self):
        result = coerce_clamp_spec({"post_block": {5: [_clamp_entry()]}})
        assert 5 in result["post_block"]

    def test_bad_layer_key_raises(self):
        with pytest.raises(ValueError, match="layer index"):
            coerce_clamp_spec({"post_block": {"abc": [_clamp_entry()]}})

    def test_non_dict_hook_value_raises(self):
        with pytest.raises(ValueError, match="must map"):
            coerce_clamp_spec({"post_block": [_clamp_entry()]})


class TestChatProtocolClamps:
    def test_fields_default_none(self):
        req = _make_chat()
        assert req.steering_clamps is None
        assert req.prefill_steering_clamps is None
        assert req.decode_steering_clamps is None

    def test_to_sampling_params_coerces_and_normalizes(self):
        req = _make_chat(
            steering_clamps=_clamps("5"),
            decode_steering_clamps={"post_block": {"6": [_clamp_entry(1, 3.0)]}},
        )
        sp = req.to_sampling_params(max_tokens=100, default_sampling_params={})
        entry = sp.steering_clamps["post_block"][5][0]
        # Canonical form: unit direction, resolved bounds.
        assert np.isclose(np.linalg.norm(entry["vector"]), 1.0)
        assert entry["min"] == 2.0
        assert entry["max"] == 2.0
        assert 6 in sp.decode_steering_clamps["post_block"]
        assert sp.prefill_steering_clamps is None

    def test_invalid_entry_rejected_at_sampling_params(self):
        req = _make_chat(
            steering_clamps={
                "post_block": {"5": [{"vector": [0.0] * _HIDDEN, "value": 1.0}]}
            }
        )
        with pytest.raises(ValueError, match="zero"):
            req.to_sampling_params(max_tokens=100, default_sampling_params={})


class TestCompletionProtocolClamps:
    def test_to_sampling_params_coerces(self):
        req = _make_completion(steering_clamps=_clamps("7"))
        sp = req.to_sampling_params(max_tokens=100, default_sampling_params={})
        assert 7 in sp.steering_clamps["post_block"]


class TestRegistryClamps:
    @pytest.mark.asyncio
    async def test_register_clamp_only_module(self):
        registry = SteeringModuleRegistry(expected_row_width=_HIDDEN)
        await registry.register(name="c", clamps=_clamps("5"))
        module = registry.get("c")
        assert module.clamps is not None
        assert 5 in module.clamps["post_block"]

    @pytest.mark.asyncio
    async def test_dump_for_broadcast_carries_clamps(self):
        registry = SteeringModuleRegistry(expected_row_width=_HIDDEN)
        await registry.register(
            name="c",
            clamps=_clamps("5"),
            decode_clamps={"post_block": {"6": [_clamp_entry(1, 0.0)]}},
        )
        payload = registry.dump_for_broadcast()["c"]
        assert payload["clamps"] is not None
        assert payload["decode_clamps"] is not None
        assert payload["prefill_clamps"] is None
        assert payload["vectors"] is None

    @pytest.mark.asyncio
    async def test_empty_module_rejected(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="no clamps"):
            await registry.register(name="empty")

    @pytest.mark.asyncio
    async def test_wrong_width_clamp_rejected(self):
        registry = SteeringModuleRegistry(expected_row_width=_HIDDEN)
        with pytest.raises(ValueError, match="width"):
            await registry.register(name="c", clamps=_clamps("5", hidden=4))

    @pytest.mark.asyncio
    async def test_malformed_entry_rejected(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="at least one"):
            await registry.register(
                name="c",
                clamps={"post_block": {"5": [{"vector": [1.0] * _HIDDEN}]}},
            )

    @pytest.mark.asyncio
    async def test_invalid_hook_rejected(self):
        registry = SteeringModuleRegistry()
        with pytest.raises(ValueError, match="hook point"):
            await registry.register(name="c", clamps={"bogus": {"5": [_clamp_entry()]}})


def _make_processor(steering_config=None, hidden_size=_HIDDEN):
    proc = MagicMock()
    proc.steering_config = steering_config
    proc.model_config.get_hidden_size.return_value = hidden_size
    from vllm.v1.engine.input_processor import InputProcessor

    proc._validate_steering = InputProcessor._validate_steering.__get__(proc)
    return proc


class TestInputProcessorClampGates:
    def _params(self, **kwargs):
        return SamplingParams(**kwargs)

    def test_clamps_rejected_when_steering_disabled(self):
        proc = _make_processor(steering_config=None)
        params = self._params(steering_clamps={"post_block": {5: [_clamp_entry()]}})
        with pytest.raises(ValueError, match="not enabled"):
            proc._validate_steering(params)

    def test_clamps_rejected_when_clamping_disabled(self):
        proc = _make_processor(steering_config=SteeringConfig(max_clamp_directions=0))
        params = self._params(steering_clamps={"post_block": {5: [_clamp_entry()]}})
        with pytest.raises(ValueError, match="max_clamp_directions"):
            proc._validate_steering(params)

    def test_clamps_accepted_when_enabled(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = self._params(steering_clamps={"post_block": {5: [_clamp_entry()]}})
        proc._validate_steering(params)  # should not raise

    def test_wrong_width_rejected(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=16)
        params = self._params(steering_clamps={"post_block": {5: [_clamp_entry()]}})
        with pytest.raises(ValueError, match="width"):
            proc._validate_steering(params)

    def test_per_site_k_cap_enforced(self):
        proc = _make_processor(steering_config=SteeringConfig(max_clamp_directions=2))
        params = self._params(
            steering_clamps={"post_block": {5: [_clamp_entry(), _clamp_entry(1)]}},
            decode_steering_clamps={"post_block": {5: [_clamp_entry(2)]}},
        )
        with pytest.raises(ValueError, match="max_clamp_directions"):
            proc._validate_steering(params)

    def test_vectors_plus_clamps_ok(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = self._params(
            steering_vectors={"post_block": {5: [1.0] * _HIDDEN}},
            steering_clamps={"post_block": {5: [_clamp_entry()]}},
        )
        proc._validate_steering(params)  # should not raise
