# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``neuron_clamps`` through the OpenAI-compatible request models (HTTP 400s)."""

from http import HTTPStatus

import pytest
from pydantic import ValidationError

from vllm.config.neuron_clamp_types import NeuronClamp, NeuronClampValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)

pytestmark = pytest.mark.cpu_test

_CLAMP = {"layer": 9, "neuron": 3757, "mode": "set", "value": 0.0, "phase": "both"}


def _chat(**kw):
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kw
    )


def _completion(**kw):
    return CompletionRequest(model="m", prompt="hi", **kw)


@pytest.mark.parametrize("make", [_chat, _completion])
class TestProtocol:
    def test_absent_is_none(self, make):
        sp = make().to_sampling_params(8, {})
        assert sp.neuron_clamps is None
        assert sp.prefill_steering_config_hash == 0

    def test_roundtrip_to_sampling_params(self, make):
        req = make(
            neuron_clamps=[
                _CLAMP,
                {
                    "layer": 18,
                    "neuron": 7807,
                    "mode": "add",
                    "value": -2,
                    "phase": "decode",
                },
            ]
        )
        sp = req.to_sampling_params(8, {})
        assert sp.parsed_neuron_clamps == (
            NeuronClamp(9, 3757, "set", 0.0, "both"),
            NeuronClamp(18, 7807, "add", -2.0, "decode"),
        )
        assert sp.decode_steering_config_hash != 0

    def test_phase_defaults_to_both(self, make):
        clamp = dict(_CLAMP)
        del clamp["phase"]
        sp = make(neuron_clamps=[clamp]).to_sampling_params(8, {})
        assert sp.parsed_neuron_clamps[0].phase == "both"

    def test_json_body(self, make):
        cls = type(make())
        base = (
            {"messages": [{"role": "user", "content": "hi"}]}
            if cls is ChatCompletionRequest
            else {"prompt": "hi"}
        )
        req = cls.model_validate({"model": "m", **base, "neuron_clamps": [_CLAMP]})
        assert req.neuron_clamps[0].neuron == 3757

    @pytest.mark.parametrize(
        "bad",
        [
            {**_CLAMP, "mode": "zero"},
            {**_CLAMP, "phase": "all"},
            {**_CLAMP, "layer": -1},
            {**_CLAMP, "neuron": -5},
            {**_CLAMP, "layer": "9"},
            {**_CLAMP, "layer": 9.5},
            {**_CLAMP, "value": "0"},
            {**_CLAMP, "value": float("nan")},
            {**_CLAMP, "value": float("inf")},
            {**_CLAMP, "extra": 1},
            {k: v for k, v in _CLAMP.items() if k != "value"},
            {k: v for k, v in _CLAMP.items() if k != "mode"},
        ],
    )
    def test_malformed_rejected_by_schema(self, make, bad):
        with pytest.raises(ValidationError):
            make(neuron_clamps=[bad])

    def test_overlapping_duplicate_rejected(self, make):
        req = make(neuron_clamps=[_CLAMP, {**_CLAMP, "value": 1.0, "phase": "decode"}])
        with pytest.raises(NeuronClampValidationError) as ei:
            req.to_sampling_params(8, {})
        resp = create_error_response(ei.value)
        assert resp.error.code == HTTPStatus.BAD_REQUEST
        assert resp.error.param == "neuron_clamps"

    def test_disjoint_phases_allowed(self, make):
        req = make(
            neuron_clamps=[
                {**_CLAMP, "phase": "prefill"},
                {**_CLAMP, "phase": "decode", "value": 1.0},
            ]
        )
        sp = req.to_sampling_params(8, {})
        assert sp.prefill_steering_config_hash != sp.decode_steering_config_hash


def test_range_error_maps_to_400():
    resp = create_error_response(
        NeuronClampValidationError("neuron clamp layer 99 out of range")
    )
    assert resp.error.code == HTTPStatus.BAD_REQUEST
    assert resp.error.type == "BadRequestError"
