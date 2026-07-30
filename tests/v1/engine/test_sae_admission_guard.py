# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EngineCore SAE admission guard.

The guard exists for frontends that do not pre-validate SAE specs (the
Rust frontend, raw ZMQ clients): ``preprocess_add_request`` raises a
:class:`SteeringVectorError` — a request-scoped validation error — for
specs the worker mixin would otherwise fail on mid-execute, where an
exception is fatal to the engine core.  The mirror that backs the check
is maintained as a side effect of ``collective_rpc``, after the worker
broadcast succeeded.

The methods are exercised unbound on a minimal stand-in — constructing
a full ``EngineCore`` needs an executor and a model, none of which the
guard logic reads.
"""

from types import SimpleNamespace

import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
    SAEFullReconstructionSpec,
)
from vllm.exceptions import SteeringVectorError, VLLMValidationError
from vllm.v1.engine.core import EngineCore


def _delta_payload(
    layers: list[list] | None = None,
    clampable: list[int] | None = None,
) -> dict:
    return {
        "kind": "sae_delta",
        "sae_manifest": {
            "layers": layers if layers is not None else [[12, "post_block"]],
            "clampable_features": clampable if clampable is not None else [0, 7],
        },
    }


def _fr_payload() -> dict:
    payload = _delta_payload()
    payload["kind"] = "sae_full_reconstruction"
    return payload


def _core(mirror: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(_sae_admission_mirror=mirror or {})


def _update(core, method, kwargs) -> None:
    EngineCore._maybe_update_sae_admission_mirror(core, method, kwargs)


def _validate(core, sampling_params) -> None:
    EngineCore._validate_sae_specs_for_admission(core, sampling_params)


def _params(
    clamp_specs: tuple = (),
    fr_specs: tuple = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        sae_clamp_specs=clamp_specs,
        sae_full_reconstruction_specs=fr_specs,
    )


def _clamp_spec(
    module: str = "sae",
    layer: int = 12,
    hook: str = "post_block",
    feature: int = 0,
) -> SAEClampSpec:
    entry = SAEClampEntry(feature_idx=feature, kind="absolute", value=1.0)
    return SAEClampSpec(module_name=module, clamps={hook: {layer: (entry,)}})


class TestMirrorUpdate:
    def test_register_adds_delta_and_fr_entries(self):
        core = _core()
        _update(
            core,
            "register_steering_modules",
            {"modules": {"d": _delta_payload(), "f": _fr_payload()}},
        )
        assert core._sae_admission_mirror == {
            "d": ("sae_delta", frozenset({(12, "post_block")}), frozenset({0, 7})),
            "f": (
                "sae_full_reconstruction",
                frozenset({(12, "post_block")}),
                frozenset({0, 7}),
            ),
        }

    def test_reregister_as_additive_drops_entry(self):
        core = _core()
        _update(core, "register_steering_modules", {"modules": {"d": _delta_payload()}})
        _update(
            core,
            "register_steering_modules",
            {"modules": {"d": {"kind": "additive", "vectors": {}}}},
        )
        assert core._sae_admission_mirror == {}

    def test_replace_rebuilds_mirror_from_scratch(self):
        core = _core()
        _update(core, "register_steering_modules", {"modules": {"a": _delta_payload()}})
        _update(
            core,
            "register_steering_modules",
            {"modules": {"b": _delta_payload()}, "replace": True},
        )
        assert set(core._sae_admission_mirror) == {"b"}

    def test_unregister_removes_only_named(self):
        core = _core()
        _update(
            core,
            "register_steering_modules",
            {"modules": {"a": _delta_payload(), "b": _fr_payload()}},
        )
        _update(core, "unregister_steering_modules", {"names": ["a"]})
        assert set(core._sae_admission_mirror) == {"b"}

    def test_unrelated_rpc_leaves_mirror_untouched(self):
        core = _core()
        _update(core, "register_steering_modules", {"modules": {"a": _delta_payload()}})
        before = core._sae_admission_mirror
        _update(core, "attach_sae_weights", {"module_name": "a", "weights": {}})
        assert core._sae_admission_mirror is before

    def test_non_dict_payload_ignored(self):
        core = _core()
        _update(core, "register_steering_modules", {"modules": {"a": "junk"}})
        assert core._sae_admission_mirror == {}

    def test_update_swaps_dict_instead_of_mutating(self):
        """``preprocess_add_request`` reads the mirror from the
        input-processing thread; updates must swap a rebuilt dict in
        one assignment, never mutate the published one."""
        core = _core()
        _update(core, "register_steering_modules", {"modules": {"a": _delta_payload()}})
        published = core._sae_admission_mirror
        snapshot = dict(published)
        _update(core, "register_steering_modules", {"modules": {"b": _fr_payload()}})
        assert core._sae_admission_mirror is not published
        assert published == snapshot


class TestAdmissionValidation:
    def _mirrored_core(self) -> SimpleNamespace:
        core = _core()
        _update(
            core,
            "register_steering_modules",
            {"modules": {"d": _delta_payload(), "f": _fr_payload()}},
        )
        return core

    def test_valid_specs_pass(self):
        core = self._mirrored_core()
        _validate(
            core,
            _params(
                clamp_specs=(_clamp_spec(module="d"),),
                fr_specs=(SAEFullReconstructionSpec(module_name="f"),),
            ),
        )

    def test_unknown_module_raises_request_scoped_error(self):
        core = self._mirrored_core()
        with pytest.raises(SteeringVectorError, match="unknown sae_delta"):
            _validate(core, _params(clamp_specs=(_clamp_spec(module="ghost"),)))

    def test_wrong_kind_reference_raises(self):
        """A clamp spec naming an FR module (and vice versa) must fail
        admission — the worker mixin would fail it fatally otherwise."""
        core = self._mirrored_core()
        with pytest.raises(SteeringVectorError, match="unknown sae_delta"):
            _validate(core, _params(clamp_specs=(_clamp_spec(module="f"),)))
        with pytest.raises(SteeringVectorError, match="unknown sae_full_reconstruction"):
            _validate(
                core,
                _params(fr_specs=(SAEFullReconstructionSpec(module_name="d"),)),
            )

    def test_uncovered_site_raises(self):
        core = self._mirrored_core()
        with pytest.raises(SteeringVectorError, match="not covered"):
            _validate(core, _params(clamp_specs=(_clamp_spec(module="d", layer=3),)))

    def test_unclampable_feature_raises(self):
        core = self._mirrored_core()
        with pytest.raises(SteeringVectorError, match="clampable set"):
            _validate(
                core, _params(clamp_specs=(_clamp_spec(module="d", feature=99),))
            )

    def test_error_is_request_scoped_validation_error(self):
        """The whole point of the guard: the raise must be a
        ``VLLMValidationError`` subtype so the input-processing thread
        finishes the request with an error instead of killing the
        engine core (see test_preprocess_error_handling.py for the
        generic preprocess-error-to-FinishReason.ERROR path)."""
        core = self._mirrored_core()
        with pytest.raises(VLLMValidationError):
            _validate(core, _params(clamp_specs=(_clamp_spec(module="ghost"),)))
