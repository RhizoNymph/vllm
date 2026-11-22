# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InputProcessor model-dependent validation of ``neuron_clamps``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from vllm.config.neuron_clamp_types import NeuronClampValidationError
from vllm.config.steering import SteeringConfig
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_test


_DEFAULT = object()


def _proc(steering_config=_DEFAULT, arch="Qwen3ForCausalLM", layers=36, inter=12288):
    from vllm.v1.engine.input_processor import InputProcessor

    proc = MagicMock()
    proc.steering_config = (
        SteeringConfig() if steering_config is _DEFAULT else steering_config
    )
    proc.model_config = SimpleNamespace(
        architectures=[arch],
        hf_text_config=SimpleNamespace(intermediate_size=inter),
        get_total_num_hidden_layers=lambda: layers,
    )
    proc._validate_steering = InputProcessor._validate_steering.__get__(proc)
    proc._validate_neuron_clamps = InputProcessor._validate_neuron_clamps.__get__(proc)
    return proc


def _sp(**kw):
    clamp = {"layer": 9, "neuron": 3757, "mode": "set", "value": 0.0}
    clamp.update(kw)
    return SamplingParams(neuron_clamps=[clamp])


class TestInputProcessorNeuronClamps:
    def test_ok(self):
        _proc()._validate_steering(_sp())

    def test_qwen2_rejected(self):
        # Qwen2ForCausalLM runs on the Transformers backend (no clamp op).
        with pytest.raises(NeuronClampValidationError, match="not supported"):
            _proc(arch="Qwen2ForCausalLM")._validate_steering(_sp())

    def test_no_clamps_always_ok(self):
        _proc(steering_config=None)._validate_steering(SamplingParams())

    def test_steering_disabled(self):
        with pytest.raises(NeuronClampValidationError, match="not enabled"):
            _proc(steering_config=None)._validate_steering(_sp())

    def test_unsupported_architecture(self):
        with pytest.raises(NeuronClampValidationError, match="not supported"):
            _proc(arch="LlamaForCausalLM")._validate_steering(_sp())

    def test_layer_range(self):
        with pytest.raises(NeuronClampValidationError, match="layer 36"):
            _proc()._validate_steering(_sp(layer=36))

    def test_neuron_range(self):
        with pytest.raises(NeuronClampValidationError, match="neuron 12288"):
            _proc()._validate_steering(_sp(neuron=12288))

    def test_disabled_by_zero_width(self):
        with pytest.raises(NeuronClampValidationError, match="disabled"):
            _proc(
                steering_config=SteeringConfig(max_neuron_clamps_per_layer=0)
            )._validate_steering(_sp())

    def test_per_layer_limit(self):
        sp = SamplingParams(
            neuron_clamps=[
                {"layer": 9, "neuron": n, "mode": "set", "value": 0.0} for n in range(3)
            ]
        )
        _proc(
            steering_config=SteeringConfig(max_neuron_clamps_per_layer=3)
        )._validate_steering(sp)
        with pytest.raises(NeuronClampValidationError, match="exceed"):
            _proc(
                steering_config=SteeringConfig(max_neuron_clamps_per_layer=2)
            )._validate_steering(sp)

    def test_error_is_value_error_with_param(self):
        with pytest.raises(ValueError) as ei:
            _proc()._validate_steering(_sp(neuron=10**6))
        assert ei.value.parameter == "neuron_clamps"


class TestSteeringConfig:
    def test_default_width(self):
        assert SteeringConfig().max_neuron_clamps_per_layer == 16

    def test_width_in_compile_hash(self):
        assert SteeringConfig(max_neuron_clamps_per_layer=4).compute_hash() != (
            SteeringConfig(max_neuron_clamps_per_layer=8).compute_hash()
        )

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            SteeringConfig(max_neuron_clamps_per_layer=-1)

    def test_cli_flag(self):
        from vllm.engine.arg_utils import EngineArgs
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        args = parser.parse_args(["--max-neuron-clamps-per-layer", "5"])
        assert EngineArgs.from_cli_args(args).max_neuron_clamps_per_layer == 5
