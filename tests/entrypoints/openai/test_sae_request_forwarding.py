# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the OpenAI request models forward ``sae_clamp_specs``.

The Phase-0 admission guard only fires if the field actually reaches
the worker.  The OpenAI request models must therefore declare the
field and propagate it through ``to_sampling_params`` — without that
plumbing, ``extra_body={"sae_clamp_specs": ...}`` is silently dropped
and the SAE state never enters the request hash or the worker
admission check.
"""

from __future__ import annotations

from vllm.config.sae_steering_types import SAEClampSpec
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

_RAW_SAE_PAYLOAD = [
    {
        "module_name": "g",
        "phase": "both",
        "clamps": {
            "post_block": {
                "20": [
                    {
                        "feature_idx": 34,
                        "kind": "absolute",
                        "value": 5.0,
                        "only_if_active": False,
                    }
                ]
            }
        },
    }
]


class TestCompletionRequestForwarding:
    def test_sae_field_round_trips(self):
        req = CompletionRequest(
            model="m",
            prompt="hi",
            sae_clamp_specs=_RAW_SAE_PAYLOAD,
        )
        sp = req.to_sampling_params(
            max_tokens=4,
            default_sampling_params={},
        )
        assert sp.sae_clamp_specs is not None
        assert isinstance(sp.sae_clamp_specs[0], SAEClampSpec)
        assert sp.sae_clamp_specs[0].module_name == "g"
        # Layer key was a JSON string; coercion produced an int.
        assert 20 in sp.sae_clamp_specs[0].clamps["post_block"]

    def test_field_optional(self):
        """No field on the wire keeps SamplingParams.sae_clamp_specs at None.
        Protects the no-SAE deployment hash bit-identity contract."""
        req = CompletionRequest(model="m", prompt="hi")
        sp = req.to_sampling_params(
            max_tokens=4,
            default_sampling_params={},
        )
        assert sp.sae_clamp_specs is None


class TestChatCompletionRequestForwarding:
    def test_sae_field_round_trips(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            sae_clamp_specs=_RAW_SAE_PAYLOAD,
        )
        sp = req.to_sampling_params(
            max_tokens=4,
            default_sampling_params={},
        )
        assert sp.sae_clamp_specs is not None
        assert isinstance(sp.sae_clamp_specs[0], SAEClampSpec)
        assert sp.sae_clamp_specs[0].module_name == "g"

    def test_field_optional(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
        sp = req.to_sampling_params(
            max_tokens=4,
            default_sampling_params={},
        )
        assert sp.sae_clamp_specs is None
