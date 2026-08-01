# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for steering validation in InputProcessor."""

from unittest.mock import MagicMock

import pytest

from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec
from vllm.config.steering import SteeringConfig
from vllm.config.steering_types import maybe_pack_inline_steering_for_request
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


def _make_processor(steering_config=None, hidden_size=3):
    """Build a lightweight stand-in with the attributes needed by
    ``_validate_steering`` (steering config + model hidden size)."""
    proc = MagicMock()
    proc.steering_config = steering_config
    proc.model_config.get_hidden_size.return_value = hidden_size
    # Bind the real method to the mock so we exercise the actual logic.
    from vllm.v1.engine.input_processor import InputProcessor

    proc._validate_steering = InputProcessor._validate_steering.__get__(proc)
    return proc


_SAMPLE_SPEC = {"pre_attn": {0: [0.1, 0.2, 0.3]}}


def _sample_sae_clamp_spec() -> SAEClampSpec:
    return SAEClampSpec(
        module_name="m",
        clamps={
            "post_block": {0: (SAEClampEntry(feature_idx=0, kind="absolute", value=1.0),)}
        },
    )


class TestSteeringRejectedWhenDisabled:
    """Per-request steering must raise ValueError when steering_config is
    None."""

    def test_steering_vectors(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)

    def test_prefill_steering_vectors(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(prefill_steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)

    def test_decode_steering_vectors(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(decode_steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)

    def test_sae_clamp_specs(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(sae_clamp_specs=(_sample_sae_clamp_spec(),))
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)

    def test_steering_module_ref(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(steering_module_ref=("mod", 1.0))
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)

    def test_packed_inline_steering(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams(steering_vectors=_SAMPLE_SPEC)
        maybe_pack_inline_steering_for_request(params, "float32")
        assert params.steering_vectors is None
        assert params._effective_prefill_steering_packed is not None
        with pytest.raises(ValueError, match="steering is not enabled"):
            proc._validate_steering(params)


class TestSteeringAcceptedWhenEnabled:
    """Per-request steering should pass validation when steering is enabled."""

    def test_steering_vectors(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams(steering_vectors=_SAMPLE_SPEC)
        proc._validate_steering(params)  # should not raise

    def test_prefill_steering_vectors(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams(prefill_steering_vectors=_SAMPLE_SPEC)
        proc._validate_steering(params)  # should not raise

    def test_decode_steering_vectors(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams(decode_steering_vectors=_SAMPLE_SPEC)
        proc._validate_steering(params)  # should not raise

    def test_sae_clamp_specs(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams(sae_clamp_specs=(_sample_sae_clamp_spec(),))
        proc._validate_steering(params)  # should not raise

    def test_steering_module_ref(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams(steering_module_ref=("mod", 1.0))
        proc._validate_steering(params)  # should not raise


class TestNoSteeringAlwaysPasses:
    """Requests without steering vectors pass regardless of config."""

    def test_no_steering_disabled(self):
        proc = _make_processor(steering_config=None)
        params = SamplingParams()
        proc._validate_steering(params)  # should not raise

    def test_no_steering_enabled(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = SamplingParams()
        proc._validate_steering(params)  # should not raise


class TestPoolingParamsSkipped:
    """PoolingParams should skip steering validation entirely."""

    def test_pooling_params_disabled(self):
        proc = _make_processor(steering_config=None)
        params = PoolingParams()
        proc._validate_steering(params)  # should not raise

    def test_pooling_params_enabled(self):
        proc = _make_processor(steering_config=SteeringConfig())
        params = PoolingParams()
        proc._validate_steering(params)  # should not raise


class TestSteeringWidthValidation:
    """Inline vectors whose width differs from the model hidden size must be
    rejected at admission — they would otherwise shape-crash the worker's
    steering table population and kill the engine."""

    def test_wrong_width_base_tier_rejected(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=8)
        params = SamplingParams(steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(ValueError, match="width 3 != expected"):
            proc._validate_steering(params)

    def test_wrong_width_prefill_tier_rejected(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=8)
        params = SamplingParams(prefill_steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(
            ValueError, match=r"prefill_steering_vectors\['pre_attn'\]\[0\]"
        ):
            proc._validate_steering(params)

    def test_wrong_width_decode_tier_rejected(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=8)
        params = SamplingParams(decode_steering_vectors=_SAMPLE_SPEC)
        with pytest.raises(ValueError, match="width 3"):
            proc._validate_steering(params)

    def test_wrong_width_scaled_entry_rejected(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=8)
        params = SamplingParams(
            steering_vectors={
                "post_block": {2: {"vector": [1.0, 2.0], "scale": 0.5}}
            }
        )
        with pytest.raises(ValueError, match="width 2"):
            proc._validate_steering(params)

    def test_matching_width_accepted(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=3)
        params = SamplingParams(steering_vectors=_SAMPLE_SPEC)
        proc._validate_steering(params)  # should not raise

    def test_module_ref_only_skips_width_check(self):
        proc = _make_processor(steering_config=SteeringConfig(), hidden_size=8)
        params = SamplingParams(steering_module_ref=("mod", 1.0))
        proc._validate_steering(params)  # no inline vectors -> nothing to check
