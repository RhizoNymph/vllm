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

from vllm.config.sae_steering_types import (
    SAEClampSpec,
    SAEFullReconstructionSpec,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
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


_RAW_FR_PAYLOAD = [
    {"module_name": "fr", "phase": "both"},
    {
        "module_name": "fr2",
        "phase": "decode",
        "clamps": {
            "post_block": {"20": [{"feature_idx": 3, "kind": "absolute", "value": 1.0}]}
        },
    },
]


class TestFullReconstructionForwarding:
    def test_completion_fr_field_round_trips(self):
        req = CompletionRequest(
            model="m",
            prompt="hi",
            sae_full_reconstruction_specs=_RAW_FR_PAYLOAD,
        )
        sp = req.to_sampling_params(max_tokens=4, default_sampling_params={})
        specs = sp.sae_full_reconstruction_specs
        assert specs is not None
        assert isinstance(specs[0], SAEFullReconstructionSpec)
        assert specs[0].module_name == "fr"
        assert specs[0].clamps == {}
        # Layer key was a JSON string; coercion produced an int.
        assert 20 in specs[1].clamps["post_block"]

    def test_chat_fr_field_round_trips(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            sae_full_reconstruction_specs=_RAW_FR_PAYLOAD,
        )
        sp = req.to_sampling_params(max_tokens=4, default_sampling_params={})
        specs = sp.sae_full_reconstruction_specs
        assert specs is not None
        assert specs[1].phase == "decode"

    def test_fr_field_optional(self):
        req = CompletionRequest(model="m", prompt="hi")
        sp = req.to_sampling_params(max_tokens=4, default_sampling_params={})
        assert sp.sae_full_reconstruction_specs is None

    def test_batch_model_declares_fr_field(self):
        batch = BatchChatCompletionRequest(
            model="m",
            messages=[[{"role": "user", "content": "hi"}]],
            sae_full_reconstruction_specs=_RAW_FR_PAYLOAD,
        )
        single = batch.to_chat_completion_request(batch.messages[0])
        assert single.sae_full_reconstruction_specs == _RAW_FR_PAYLOAD


class TestRegistryFrValidation:
    def _registry_with_fr_module(self):
        import asyncio

        from vllm.config.sae_steering_types import (
            SAEActivation,
            SteeringModuleKind,
        )
        from vllm.entrypoints.openai.steering.registry import (
            SAEModuleManifest,
            SteeringModuleRegistry,
        )

        registry = SteeringModuleRegistry()
        manifest = SAEModuleManifest(
            d_model=8,
            d_sae=4,
            activation=SAEActivation.RELU,
            layers=((20, "post_block"),),
            clampable_features=(0, 3),
            activation_params={},
        )
        asyncio.run(
            registry.register(
                name="fr",
                kind=SteeringModuleKind.SAE_FULL_RECONSTRUCTION,
                sae_manifest=manifest,
            )
        )
        return registry

    def test_valid_fr_spec_passes(self):
        registry = self._registry_with_fr_module()
        assert (
            registry.validate_sae_full_reconstruction_specs([{"module_name": "fr"}])
            is None
        )
        assert (
            registry.validate_sae_full_reconstruction_specs(
                [
                    {
                        "module_name": "fr",
                        "clamps": {
                            "post_block": {
                                "20": [
                                    {
                                        "feature_idx": 3,
                                        "kind": "absolute",
                                        "value": 1.0,
                                    }
                                ]
                            }
                        },
                    }
                ]
            )
            is None
        )

    def test_unknown_module_rejected(self):
        registry = self._registry_with_fr_module()
        err = registry.validate_sae_full_reconstruction_specs([{"module_name": "nope"}])
        assert err is not None and "unknown module" in err

    def test_wrong_kind_rejected(self):
        registry = self._registry_with_fr_module()
        err = registry.validate_sae_clamp_specs(
            [
                {
                    "module_name": "fr",
                    "clamps": {
                        "post_block": {
                            "20": [
                                {
                                    "feature_idx": 3,
                                    "kind": "absolute",
                                    "value": 1.0,
                                }
                            ]
                        }
                    },
                }
            ]
        )
        assert err is not None and "sae_delta" in err

    def test_uncovered_site_rejected(self):
        registry = self._registry_with_fr_module()
        err = registry.validate_sae_full_reconstruction_specs(
            [
                {
                    "module_name": "fr",
                    "clamps": {
                        "post_block": {
                            "5": [
                                {
                                    "feature_idx": 3,
                                    "kind": "absolute",
                                    "value": 1.0,
                                }
                            ]
                        }
                    },
                }
            ]
        )
        assert err is not None and "not declared" in err

    def test_non_clampable_feature_rejected(self):
        registry = self._registry_with_fr_module()
        err = registry.validate_sae_full_reconstruction_specs(
            [
                {
                    "module_name": "fr",
                    "clamps": {
                        "post_block": {
                            "20": [
                                {
                                    "feature_idx": 2,
                                    "kind": "absolute",
                                    "value": 1.0,
                                }
                            ]
                        }
                    },
                }
            ]
        )
        assert err is not None and "clampable_features" in err
