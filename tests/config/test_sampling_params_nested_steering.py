# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-role steering support on SamplingParams (PR 5b).

Covers nested-form acceptance, per-role cached properties, per-role
hashes, and the **flat-form backward-compat guarantee**: a flat
``steering_vectors`` spec produces byte-identical hashes to the
pre-PR behaviour (no prefix-cache invalidation for legacy clients).
"""

from __future__ import annotations

import pytest

from vllm.config.steering_types import (
    hash_steering_config,
    normalize_to_per_role,
)
from vllm.sampling_params import SamplingParams

_FLAT_SPEC = {"post_mlp": {0: [1.0, 2.0, 3.0]}}
_NESTED_MAIN_ONLY = {"main": {"post_mlp": {0: [1.0, 2.0, 3.0]}}}
_NESTED_BOTH = {
    "main": {"post_mlp": {0: [1.0, 2.0, 3.0]}},
    "draft": {"post_mlp": {0: [10.0, 20.0, 30.0]}},
}


class TestNestedFormAcceptance:
    def test_flat_form_accepted(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert sp.steering_vectors == _FLAT_SPEC

    def test_nested_main_only_accepted(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_MAIN_ONLY)
        assert sp.steering_vectors == _NESTED_MAIN_ONLY

    def test_nested_both_roles_accepted(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_BOTH)
        assert sp.steering_vectors == _NESTED_BOTH

    def test_unknown_top_level_key_treated_as_hook_point(self):
        """A top-level key that is not ``main``/``draft`` falls back to
        flat-form interpretation. With a non-hook-point key, flat
        validation rejects it (the key is not in VALID_HOOK_POINT_NAMES).
        """
        with pytest.raises(ValueError):
            SamplingParams(
                max_tokens=8,
                steering_vectors={"not_a_hook": {0: [1.0]}},
            )

    def test_inner_spec_must_be_dict_when_nested(self):
        with pytest.raises(ValueError):
            SamplingParams(
                max_tokens=8,
                steering_vectors={"main": "not a dict"},  # type: ignore[dict-item]
            )


class TestEffectiveProperties:
    def test_flat_main_matches_flat_spec(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert sp.effective_prefill_steering_main == _FLAT_SPEC
        assert sp.effective_decode_steering_main == _FLAT_SPEC

    def test_flat_draft_tags_along(self):
        """Flat specs apply to both roles."""
        sp = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert sp.effective_prefill_steering_draft == _FLAT_SPEC
        assert sp.effective_decode_steering_draft == _FLAT_SPEC

    def test_nested_main_only_has_no_draft_effective(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_MAIN_ONLY)
        assert sp.effective_prefill_steering_main is not None
        assert sp.effective_prefill_steering_draft is None

    def test_nested_both_roles_have_distinct_effectives(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_BOTH)
        assert sp.effective_prefill_steering_main == _NESTED_BOTH["main"]
        assert sp.effective_prefill_steering_draft == _NESTED_BOTH["draft"]


class TestPerRoleHashes:
    def test_flat_main_hash_matches_old_contract(self):
        """Flat-form spec produces the same hash as a direct
        ``hash_steering_config`` call on the spec. Guarantees that
        pre-PR flat-form clients see byte-identical hashes and do
        NOT incur a prefix-cache miss on deploy.
        """
        sp = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert sp.prefill_steering_config_hash_main == hash_steering_config(_FLAT_SPEC)
        # Backward-compat alias mirrors _main.
        assert sp.prefill_steering_config_hash == sp.prefill_steering_config_hash_main

    def test_flat_main_and_draft_hashes_match(self):
        """Under tags-along (flat form), main and draft hashes are equal."""
        sp = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert (
            sp.prefill_steering_config_hash_main
            == sp.prefill_steering_config_hash_draft
        )

    def test_nested_both_roles_distinct_hashes(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_BOTH)
        assert (
            sp.prefill_steering_config_hash_main
            != sp.prefill_steering_config_hash_draft
        )

    def test_nested_main_only_draft_hash_zero(self):
        sp = SamplingParams(max_tokens=8, steering_vectors=_NESTED_MAIN_ONLY)
        assert sp.prefill_steering_config_hash_main != 0
        assert sp.prefill_steering_config_hash_draft == 0

    def test_no_steering_all_hashes_zero(self):
        sp = SamplingParams(max_tokens=8)
        assert sp.prefill_steering_config_hash_main == 0
        assert sp.prefill_steering_config_hash_draft == 0
        assert sp.decode_steering_config_hash_main == 0
        assert sp.decode_steering_config_hash_draft == 0

    def test_flat_hash_stable_across_sampling_params_instances(self):
        """Two SamplingParams with the same flat spec share the hash.

        Backward-compat alias must also equal the legacy computation.
        """
        sp1 = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        sp2 = SamplingParams(max_tokens=8, steering_vectors=_FLAT_SPEC)
        assert sp1.prefill_steering_config_hash == sp2.prefill_steering_config_hash


class TestNestedFormNormalization:
    def test_normalize_flat_to_tags_along(self):
        result = normalize_to_per_role(_FLAT_SPEC)
        assert result is not None
        assert set(result.keys()) == {"main", "draft"}
        assert result["main"] is _FLAT_SPEC
        assert result["draft"] is _FLAT_SPEC

    def test_normalize_nested_main_drops_draft(self):
        result = normalize_to_per_role(_NESTED_MAIN_ONLY)
        assert result is not None
        assert set(result.keys()) == {"main"}

    def test_normalize_nested_both_passes_through(self):
        result = normalize_to_per_role(_NESTED_BOTH)
        assert result is not None
        assert set(result.keys()) == {"main", "draft"}
        assert result["main"] is _NESTED_BOTH["main"]
        assert result["draft"] is _NESTED_BOTH["draft"]


class TestCrossValidateDimensions:
    def test_nested_per_role_overlapping_dimensions_validated(self):
        """Base and phase specs must agree on dimensions within a role.

        Mismatched dimensions on ``main`` should raise independently
        of what ``draft`` carries.
        """
        with pytest.raises(ValueError, match="dimension"):
            SamplingParams(
                max_tokens=8,
                steering_vectors={
                    "main": {"post_mlp": {0: [1.0, 2.0, 3.0]}},
                    "draft": {"post_mlp": {0: [1.0, 2.0]}},
                },
                prefill_steering_vectors={
                    "main": {"post_mlp": {0: [1.0, 2.0]}},  # wrong size vs main base
                },
            )

    def test_nested_cross_role_dimensions_are_independent(self):
        """Main with hidden=3 and draft with hidden=2 is legal — the
        two managers are decoupled and each validates its own specs.
        """
        sp = SamplingParams(
            max_tokens=8,
            steering_vectors={
                "main": {"post_mlp": {0: [1.0, 2.0, 3.0]}},
                "draft": {"post_mlp": {0: [1.0, 2.0]}},
            },
        )
        assert sp.effective_prefill_steering_main == {"post_mlp": {0: [1.0, 2.0, 3.0]}}
        assert sp.effective_prefill_steering_draft == {"post_mlp": {0: [1.0, 2.0]}}
