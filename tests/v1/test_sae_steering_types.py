# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SAE-based steering type helpers.

Covers:
- SAEClampEntry __post_init__ validation invariants
- SAEClampEntry.requires_encoder_pass derivation
- SAEClampSpec __post_init__ validation invariants
- coerce_sae_clamp_specs for raw-dict input
- hash_sae_clamp_specs determinism + ordering invariance
- hash_steering_config bit-identity for non-SAE callers
- hash_steering_config combined with SAE specs
"""

import pytest

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampEntry,
    SAEClampSpec,
    SteeringModuleKind,
    coerce_sae_clamp_specs,
    hash_sae_clamp_specs,
)
from vllm.config.steering_types import hash_steering_config


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_steering_module_kind_values(self):
        assert SteeringModuleKind.ADDITIVE.value == "additive"
        assert SteeringModuleKind.SAE_DELTA.value == "sae_delta"

    def test_sae_activation_values(self):
        assert SAEActivation.RELU.value == "relu"
        assert SAEActivation.JUMPRELU.value == "jumprelu"
        assert SAEActivation.TOPK.value == "topk"


# ---------------------------------------------------------------------------
# SAEClampEntry
# ---------------------------------------------------------------------------


class TestSAEClampEntry:
    def test_absolute_clamp_basic(self):
        e = SAEClampEntry(feature_idx=42, kind="absolute", value=5.0)
        assert e.feature_idx == 42
        assert e.kind == "absolute"
        assert e.value == 5.0
        assert e.only_if_active is False
        assert e.requires_encoder_pass is True

    def test_additive_clamp_basic(self):
        e = SAEClampEntry(feature_idx=7, kind="additive", value=2.5)
        assert e.kind == "additive"
        assert e.requires_encoder_pass is False

    def test_additive_only_if_active_requires_encoder(self):
        e = SAEClampEntry(
            feature_idx=7, kind="additive", value=2.5, only_if_active=True
        )
        assert e.requires_encoder_pass is True

    def test_negative_feature_idx_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SAEClampEntry(feature_idx=-1, kind="absolute", value=1.0)

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            SAEClampEntry(feature_idx=0, kind="bogus", value=1.0)  # type: ignore[arg-type]

    def test_non_finite_value_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            SAEClampEntry(feature_idx=0, kind="absolute", value=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            SAEClampEntry(feature_idx=0, kind="absolute", value=float("nan"))

    def test_non_int_feature_idx_rejected(self):
        # Pydantic coerces ``"0"`` to int 0, so we use a non-coercible
        # string to exercise the type validation path.
        with pytest.raises((ValueError, TypeError)):
            SAEClampEntry(
                feature_idx="not-a-number",  # type: ignore[arg-type]
                kind="absolute",
                value=1.0,
            )

    def test_frozen(self):
        e = SAEClampEntry(feature_idx=0, kind="absolute", value=1.0)
        with pytest.raises(Exception):
            e.feature_idx = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SAEClampSpec
# ---------------------------------------------------------------------------


def _spec(
    module_name: str = "m",
    *,
    hook: str = "post_mlp",
    layer: int = 0,
    feature_idx: int = 0,
    kind: str = "absolute",
    value: float = 1.0,
    phase: str = "both",
) -> SAEClampSpec:
    return SAEClampSpec(
        module_name=module_name,
        phase=phase,  # type: ignore[arg-type]
        clamps={
            hook: {
                layer: (
                    SAEClampEntry(
                        feature_idx=feature_idx, kind=kind, value=value
                    ),  # type: ignore[arg-type]
                )
            }
        },
    )


class TestSAEClampSpec:
    def test_basic_construction(self):
        s = _spec()
        assert s.module_name == "m"
        assert s.phase == "both"
        assert "post_mlp" in s.clamps

    def test_empty_module_name_rejected(self):
        with pytest.raises(ValueError, match="module_name"):
            SAEClampSpec(
                module_name="",
                clamps={
                    "post_mlp": {0: (SAEClampEntry(0, "absolute", 1.0),)}
                },
            )

    def test_invalid_phase_rejected(self):
        with pytest.raises(ValueError, match="phase"):
            _spec(phase="now")

    def test_empty_clamps_rejected(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            SAEClampSpec(module_name="m", clamps={})

    def test_invalid_hook_rejected(self):
        with pytest.raises(ValueError, match="hook point"):
            SAEClampSpec(
                module_name="m",
                clamps={"bogus": {0: (SAEClampEntry(0, "absolute", 1.0),)}},
            )

    def test_negative_layer_idx_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SAEClampSpec(
                module_name="m",
                clamps={
                    "post_mlp": {-1: (SAEClampEntry(0, "absolute", 1.0),)}
                },
            )

    def test_duplicate_feature_idx_rejected(self):
        with pytest.raises(ValueError, match="duplicate feature_idx"):
            SAEClampSpec(
                module_name="m",
                clamps={
                    "post_mlp": {
                        0: (
                            SAEClampEntry(5, "absolute", 1.0),
                            SAEClampEntry(5, "additive", 2.0),
                        )
                    }
                },
            )

    def test_empty_entries_tuple_rejected(self):
        with pytest.raises(ValueError, match="non-empty tuple"):
            SAEClampSpec(
                module_name="m",
                clamps={"post_mlp": {0: ()}},
            )

    def test_list_entries_coerced_to_tuple(self):
        # Pydantic auto-coerces ``list`` to ``tuple`` for fields typed as
        # ``tuple[SAEClampEntry, ...]``.  Validate that a list payload
        # round-trips successfully (the runtime contract is "non-empty
        # ordered sequence of entries", not literally a tuple).
        s = SAEClampSpec(
            module_name="m",
            clamps={"post_mlp": {0: [SAEClampEntry(0, "absolute", 1.0)]}},  # type: ignore[dict-item]
        )
        assert isinstance(s.clamps["post_mlp"][0], tuple)
        assert len(s.clamps["post_mlp"][0]) == 1


# ---------------------------------------------------------------------------
# coerce_sae_clamp_specs
# ---------------------------------------------------------------------------


class TestCoerceSAEClampSpecs:
    def test_none_returns_none(self):
        assert coerce_sae_clamp_specs(None) is None

    def test_empty_list_returns_none(self):
        assert coerce_sae_clamp_specs([]) is None

    def test_single_spec_dict_round_trip(self):
        raw = [
            {
                "module_name": "g",
                "clamps": {
                    "post_mlp": {
                        "20": [
                            {
                                "feature_idx": 34,
                                "kind": "absolute",
                                "value": 5.0,
                            }
                        ]
                    }
                },
            }
        ]
        specs = coerce_sae_clamp_specs(raw)
        assert specs is not None
        assert len(specs) == 1
        spec = specs[0]
        assert spec.module_name == "g"
        # JSON layer key was a string; coercion converted to int.
        assert 20 in spec.clamps["post_mlp"]
        entries = spec.clamps["post_mlp"][20]
        assert len(entries) == 1 and isinstance(entries[0], SAEClampEntry)
        assert entries[0].feature_idx == 34

    def test_passthrough_for_already_typed(self):
        s = _spec()
        out = coerce_sae_clamp_specs((s,))
        assert out is not None
        # Same instance survives — no re-allocation.
        assert out[0] is s

    def test_missing_clamps_field(self):
        with pytest.raises(ValueError, match="'clamps'"):
            coerce_sae_clamp_specs([{"module_name": "g"}])

    def test_non_list_input_rejected(self):
        with pytest.raises(ValueError, match="must be a list"):
            coerce_sae_clamp_specs({"module_name": "g"})


# ---------------------------------------------------------------------------
# hash_sae_clamp_specs
# ---------------------------------------------------------------------------


class TestHashSAEClampSpecs:
    def test_none_returns_zero(self):
        assert hash_sae_clamp_specs(None) == 0
        assert hash_sae_clamp_specs(()) == 0

    def test_deterministic(self):
        a = (_spec(),)
        b = (_spec(),)
        assert hash_sae_clamp_specs(a) == hash_sae_clamp_specs(b)

    def test_different_module_name_diff_hash(self):
        assert hash_sae_clamp_specs((_spec(module_name="a"),)) != hash_sae_clamp_specs(
            (_spec(module_name="b"),)
        )

    def test_different_value_diff_hash(self):
        assert hash_sae_clamp_specs((_spec(value=1.0),)) != hash_sae_clamp_specs(
            (_spec(value=2.0),)
        )

    def test_different_kind_same_value_diff_hash(self):
        # Make sure the kind discriminator byte actually fires.
        a = hash_sae_clamp_specs((_spec(kind="absolute", value=3.0),))
        b = hash_sae_clamp_specs((_spec(kind="additive", value=3.0),))
        assert a != b

    def test_only_if_active_matters(self):
        sp_off = SAEClampSpec(
            module_name="m",
            clamps={
                "post_mlp": {
                    0: (SAEClampEntry(0, "additive", 1.0, only_if_active=False),)
                }
            },
        )
        sp_on = SAEClampSpec(
            module_name="m",
            clamps={
                "post_mlp": {
                    0: (SAEClampEntry(0, "additive", 1.0, only_if_active=True),)
                }
            },
        )
        assert hash_sae_clamp_specs((sp_off,)) != hash_sae_clamp_specs((sp_on,))

    def test_spec_order_independent(self):
        """Two requests carrying the same set of specs in different order
        produce the same hash, so module-name list ordering doesn't
        leak into the prefix-cache key."""
        a = (_spec(module_name="alpha"), _spec(module_name="beta"))
        b = (_spec(module_name="beta"), _spec(module_name="alpha"))
        assert hash_sae_clamp_specs(a) == hash_sae_clamp_specs(b)

    def test_entry_order_independent(self):
        """Entries within the same (hook, layer) hash the same regardless
        of the order the user passed them in."""
        e1 = SAEClampEntry(1, "absolute", 1.0)
        e2 = SAEClampEntry(2, "additive", 2.0)
        a = SAEClampSpec(
            module_name="m", clamps={"post_mlp": {0: (e1, e2)}}
        )
        b = SAEClampSpec(
            module_name="m", clamps={"post_mlp": {0: (e2, e1)}}
        )
        assert hash_sae_clamp_specs((a,)) == hash_sae_clamp_specs((b,))

    def test_returns_positive_int63(self):
        h = hash_sae_clamp_specs((_spec(),))
        assert h >= 0
        assert h <= 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# hash_steering_config: SAE composition + bit-identity for non-SAE callers
# ---------------------------------------------------------------------------


class TestHashSteeringConfigWithSAE:
    def test_no_sae_arg_bit_identical(self):
        """Critical: callers that don't pass sae_clamp_specs must hash
        bit-identically to before this argument existed.  This is the
        prefix-cache reuse contract for non-SAE deployments."""
        vecs = {"post_mlp": {0: [0.1, 0.2]}}
        ref = ("mod", 1.5)
        assert hash_steering_config(vecs) == hash_steering_config(
            vecs, sae_clamp_specs=None
        )
        assert hash_steering_config(vecs, module_ref=ref) == hash_steering_config(
            vecs, module_ref=ref, sae_clamp_specs=None
        )
        # Empty-but-non-None tuple also maps to "no SAE block".
        assert hash_steering_config(vecs) == hash_steering_config(
            vecs, sae_clamp_specs=()
        )

    def test_sae_only_request_nonzero_hash(self):
        """A request with only SAE clamps and no additive vectors
        must still produce a non-zero hash (otherwise it would alias
        prefix-cache entries with the no-steering case)."""
        h = hash_steering_config(None, sae_clamp_specs=(_spec(),))
        assert h != 0

    def test_sae_state_changes_hash(self):
        vecs = {"post_mlp": {0: [0.1, 0.2]}}
        h_no_sae = hash_steering_config(vecs)
        h_with_sae = hash_steering_config(
            vecs, sae_clamp_specs=(_spec(),)
        )
        assert h_no_sae != h_with_sae

    def test_returns_positive_int63(self):
        h = hash_steering_config(
            {"post_mlp": {0: [0.1]}}, sae_clamp_specs=(_spec(),)
        )
        assert h >= 0
        assert h <= 0x7FFFFFFFFFFFFFFF
