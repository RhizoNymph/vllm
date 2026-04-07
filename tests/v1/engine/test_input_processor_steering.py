# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for steering validation in InputProcessor."""

from unittest.mock import MagicMock

import pytest

from vllm.config.steering import SteeringConfig
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


def _make_processor(steering_config=None):
    """Build a lightweight stand-in with only the attribute needed by
    ``_validate_steering``."""
    proc = MagicMock()
    proc.steering_config = steering_config
    # Bind the real method to the mock so we exercise the actual logic.
    from vllm.v1.engine.input_processor import InputProcessor

    proc._validate_steering = InputProcessor._validate_steering.__get__(proc)
    return proc


_SAMPLE_SPEC = {"pre_attn": {0: [0.1, 0.2, 0.3]}}


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
