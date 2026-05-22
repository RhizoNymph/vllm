# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side admission tests for SAE-kind broadcast + SamplingParams.

The mixin's full state graph is heavy to construct; these tests
exercise only the two SAE-specific entry points:

* ``register_steering_modules`` dispatching by ``kind``
* ``_assert_sae_clamps_can_be_applied`` validating registered modules,
  covered sites, and clampable features
"""

from __future__ import annotations

import pytest

from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.exceptions import SteeringVectorError
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
)


class _MixinHarness(SteeringModelRunnerMixin):
    """Bare instance of the mixin with the two registries initialized.

    ``_init_steering_state`` requires a fully-loaded model and a
    ``vllm_config``; for these tests we only need the dispatch table.
    """

    def __init__(self):
        self._steering_module_registry = {}
        self._steering_module_resolved_cache = {}
        self._sae_module_registry = {}
        self._sae_steerable_sites = {}
        self._req_sae_phase = {}
        self._sae_clamp_manager = None


def _sae_payload(
    *,
    name_unused: str = "g",
    d_model: int = 4096,
    d_sae: int = 65536,
    activation: str = "jumprelu",
    layers=((20, "post_mlp"),),
    clampable_features=(0, 1, 2, 34),
    activation_params: dict | None = None,
) -> dict:
    if activation_params is None and activation == "jumprelu":
        activation_params = {"threshold": 0.0}
    elif activation_params is None and activation == "topk":
        activation_params = {"k": 1.0}
    else:
        activation_params = activation_params or {}
    return {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": activation,
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable_features),
            "activation_params": activation_params,
            "weights_uri": None,
        },
    }


def _additive_payload(*, vec=None) -> dict:
    return {
        "kind": "additive",
        "vectors": {"post_mlp": {0: vec or [0.1, 0.2]}},
        "prefill_vectors": None,
        "decode_vectors": None,
    }


class TestRegisterDispatch:
    def test_additive_lands_in_additive_registry(self):
        h = _MixinHarness()
        h.register_steering_modules({"m": _additive_payload()})
        assert "m" in h._steering_module_registry
        assert "m" in h._steering_module_resolved_cache
        assert "m" not in h._sae_module_registry

    def test_additive_named_module_uses_resolved_cache(self):
        h = _MixinHarness()
        h.register_steering_modules({"m": _additive_payload(vec=[0.1, 0.2])})

        sp = SamplingParams(steering_module_ref=("m", 1.0))
        resolved = h._resolve_request_steering(sp, "prefill")

        assert resolved is h._steering_module_resolved_cache["m"][0]
        assert resolved is not None
        assert resolved["post_mlp"][0].tolist() == pytest.approx([0.1, 0.2])

    def test_scaled_named_module_uses_resolved_cache_values(self):
        h = _MixinHarness()
        h.register_steering_modules({"m": _additive_payload(vec=[0.1, 0.2])})

        sp = SamplingParams(steering_module_ref=("m", 2.0))
        resolved = h._resolve_request_steering(sp, "prefill")

        assert resolved is not h._steering_module_resolved_cache["m"][0]
        assert resolved is not None
        assert resolved["post_mlp"][0].tolist() == pytest.approx([0.2, 0.4])

    def test_sae_lands_in_sae_registry(self):
        h = _MixinHarness()
        h.register_steering_modules({"g": _sae_payload()})
        assert "g" in h._sae_module_registry
        assert "g" not in h._steering_module_registry
        manifest = h._sae_module_registry["g"]
        assert manifest.d_model == 4096
        assert manifest.layers == ((20, "post_mlp"),)

    def test_legacy_payload_without_kind_treated_as_additive(self):
        """Backwards compatibility: a payload with no ``kind`` field
        (older API server) routes to the additive registry."""
        h = _MixinHarness()
        legacy = {
            "vectors": {"post_mlp": {0: [0.1]}},
            "prefill_vectors": None,
            "decode_vectors": None,
        }
        h.register_steering_modules({"m": legacy})
        assert "m" in h._steering_module_registry

    def test_re_register_swaps_kind(self):
        """Registering the same name as a different kind drops the old
        entry from the wrong registry."""
        h = _MixinHarness()
        h.register_steering_modules({"x": _additive_payload()})
        assert "x" in h._steering_module_registry
        assert "x" in h._steering_module_resolved_cache
        h.register_steering_modules({"x": _sae_payload()})
        assert "x" in h._sae_module_registry
        assert "x" not in h._steering_module_registry
        assert "x" not in h._steering_module_resolved_cache

    def test_replace_clears_both_registries(self):
        h = _MixinHarness()
        h.register_steering_modules({"m": _additive_payload(), "g": _sae_payload()})
        h.register_steering_modules({"only": _additive_payload()}, replace=True)
        assert list(h._steering_module_registry.keys()) == ["only"]
        assert list(h._steering_module_resolved_cache.keys()) == ["only"]
        assert h._sae_module_registry == {}

    def test_unknown_kind_raises(self):
        h = _MixinHarness()
        with pytest.raises(SteeringVectorError, match="unknown kind"):
            h.register_steering_modules({"m": {"kind": "unknown"}})

    def test_sae_payload_missing_manifest_raises(self):
        h = _MixinHarness()
        with pytest.raises(SteeringVectorError, match="sae_manifest"):
            h.register_steering_modules({"g": {"kind": "sae_delta"}})

    def test_sae_payload_with_empty_additive_fields_raises(self):
        h = _MixinHarness()
        payload = _sae_payload()
        payload["vectors"] = {}
        with pytest.raises(SteeringVectorError, match="additive vector fields"):
            h.register_steering_modules({"g": payload})

    def test_additive_payload_with_sae_manifest_raises(self):
        h = _MixinHarness()
        payload = _additive_payload()
        payload["sae_manifest"] = _sae_payload()["sae_manifest"]
        with pytest.raises(SteeringVectorError, match="sae_manifest"):
            h.register_steering_modules({"m": payload})

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (
                _sae_payload(clampable_features=(0, 1, 0)),
                "clampable_features.*duplicates",
            ),
            (_sae_payload(clampable_features=()), "clampable_features.*empty"),
            (
                _sae_payload(layers=((20, "post_mlp"), (20, "post_mlp"))),
                "duplicate.*sites",
            ),
            (_sae_payload(layers=((20, "bogus"),)), "unknown hook point"),
            (_sae_payload(d_sae=2, clampable_features=(3,)), "out of range"),
            (_sae_payload(d_model=True), "d_model.*int"),
            (_sae_payload(d_sae=True), "d_sae.*int"),
            (_sae_payload(layers=((True, "post_mlp"),)), "layers.*int"),
            (_sae_payload(clampable_features=(True,)), "clampable_features.*int"),
            (_sae_payload(layers=((2**31, "post_mlp"),)), "2147483647"),
            (
                _sae_payload(d_sae=2**31 + 1, clampable_features=(2**31,)),
                "2147483647",
            ),
            (
                _sae_payload(activation_params={}),
                "activation_params.*threshold",
            ),
            (
                _sae_payload(activation_params={"threshold": True}),
                "activation_params.*threshold",
            ),
            (
                _sae_payload(activation="topk", activation_params={}),
                "activation_params.*k",
            ),
            (
                _sae_payload(activation="topk", activation_params={"k": 0.0}),
                "activation_params.*k",
            ),
            (
                _sae_payload(activation="topk", activation_params={"k": 1.5}),
                "activation_params.*k",
            ),
            (
                _sae_payload(activation="relu", activation_params={"k": 1.0}),
                "activation_params.*relu",
            ),
        ],
    )
    def test_sae_payload_invalid_manifest_raises(self, payload, match):
        h = _MixinHarness()
        with pytest.raises(SteeringVectorError, match=match):
            h.register_steering_modules({"g": payload})

    def test_unregister_removes_from_both(self):
        h = _MixinHarness()
        h.register_steering_modules({"m": _additive_payload(), "g": _sae_payload()})
        h.unregister_steering_modules(["m", "g"])
        assert h._steering_module_registry == {}
        assert h._sae_module_registry == {}


class TestSAEClampAdmissionGuard:
    def test_no_specs_no_op(self):
        h = _MixinHarness()
        sp = SamplingParams()
        # Must not raise.
        h._assert_sae_clamps_can_be_applied(sp)

    def test_unknown_module_raises_steering_error(self):
        h = _MixinHarness()
        sp = SamplingParams(
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="missing",
                    clamps={"post_mlp": {0: (SAEClampEntry(0, "absolute", 1.0),)}},
                ),
            ),
        )
        with pytest.raises(SteeringVectorError, match="unknown module 'missing'"):
            h._assert_sae_clamps_can_be_applied(sp)

    def test_known_module_with_valid_spec_admits(self):
        """A spec naming a registered module, a covered (layer, hook),
        and a clampable feature must pass admission."""
        h = _MixinHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="g",
                    clamps={"post_mlp": {20: (SAEClampEntry(34, "absolute", 5.0),)}},
                ),
            ),
        )
        # Must not raise.
        h._assert_sae_clamps_can_be_applied(sp)

    def test_uncovered_layer_hook_raises(self):
        h = _MixinHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="g",
                    # layer 21 is not in the manifest's coverage (only 20).
                    clamps={"post_mlp": {21: (SAEClampEntry(34, "absolute", 5.0),)}},
                ),
            ),
        )
        with pytest.raises(SteeringVectorError, match="not declared"):
            h._assert_sae_clamps_can_be_applied(sp)

    def test_unclampable_feature_raises(self):
        h = _MixinHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="g",
                    # feature_idx=999 not in clampable_features=(0, 1, 2, 34)
                    clamps={"post_mlp": {20: (SAEClampEntry(999, "absolute", 5.0),)}},
                ),
            ),
        )
        with pytest.raises(SteeringVectorError, match="clampable_features"):
            h._assert_sae_clamps_can_be_applied(sp)
