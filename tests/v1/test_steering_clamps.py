# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for directional-clamp steering types and request plumbing.

Covers:
- normalize_clamp_entry: sugar / bounds / normalization / rejection cases
- resolve_effective_clamps: tier concatenation and the K cap
- hash_steering_config: clamp segment determinism, domain separation, and
  bit-for-bit stability of vector-only hashes
- SamplingParams: field validation, in-place unit normalization, effective
  clamp properties, hash properties, and pack-path survival
"""

import math

import numpy as np
import pytest

from vllm.config.steering_types import (
    hash_steering_config,
    maybe_pack_inline_steering_for_request,
    normalize_clamp_entry,
    resolve_effective_clamps,
    validate_clamp_row_widths,
)
from vllm.sampling_params import SamplingParams

# -----------------------------------------------------------------------
# normalize_clamp_entry
# -----------------------------------------------------------------------


class TestNormalizeClampEntry:
    def test_min_max_form(self):
        vec, lo, hi, strength = normalize_clamp_entry(
            {"vector": [3.0, 4.0], "min": -1.0, "max": 2.0}
        )
        assert np.allclose(vec, [0.6, 0.8])
        assert lo == -1.0
        assert hi == 2.0
        assert strength == 1.0

    def test_unit_normalization_float64(self):
        vec, _, _, _ = normalize_clamp_entry(
            {"vector": [1e-3, 0.0, 0.0], "max": 1.0}
        )
        assert vec.dtype == np.float64
        assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-12)
        assert vec[0] == 1.0

    def test_value_sugar_pins_both_bounds(self):
        _, lo, hi, _ = normalize_clamp_entry({"vector": [1.0], "value": 5.0})
        assert lo == 5.0
        assert hi == 5.0

    def test_one_sided_min_none_is_neg_inf(self):
        _, lo, hi, _ = normalize_clamp_entry({"vector": [1.0], "max": 4.0})
        assert lo == -math.inf
        assert hi == 4.0

    def test_one_sided_max_none_is_pos_inf(self):
        _, lo, hi, _ = normalize_clamp_entry({"vector": [1.0], "min": -4.0})
        assert lo == -4.0
        assert hi == math.inf

    def test_explicit_strength(self):
        _, _, _, strength = normalize_clamp_entry(
            {"vector": [1.0], "value": 0.0, "strength": 0.25}
        )
        assert strength == 0.25

    def test_no_bounds_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            normalize_clamp_entry({"vector": [1.0]})

    def test_value_plus_min_max_rejected(self):
        with pytest.raises(ValueError, match="value"):
            normalize_clamp_entry({"vector": [1.0], "value": 1.0, "max": 2.0})

    def test_min_greater_than_max_rejected(self):
        with pytest.raises(ValueError, match="min"):
            normalize_clamp_entry({"vector": [1.0], "min": 2.0, "max": 1.0})

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError, match="zero"):
            normalize_clamp_entry({"vector": [0.0, 0.0], "value": 1.0})

    def test_nonfinite_vector_rejected(self):
        with pytest.raises(ValueError):
            normalize_clamp_entry({"vector": [1.0, math.inf], "value": 1.0})

    def test_nonfinite_bound_value_rejected(self):
        with pytest.raises(ValueError):
            normalize_clamp_entry({"vector": [1.0], "value": math.nan})

    def test_strength_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="strength"):
            normalize_clamp_entry({"vector": [1.0], "value": 1.0, "strength": 1.5})
        with pytest.raises(ValueError, match="strength"):
            normalize_clamp_entry({"vector": [1.0], "value": 1.0, "strength": -0.1})

    def test_unexpected_keys_rejected(self):
        with pytest.raises(ValueError, match="unexpected"):
            normalize_clamp_entry({"vector": [1.0], "value": 1.0, "scale": 2.0})

    def test_missing_vector_rejected(self):
        with pytest.raises(ValueError, match="vector"):
            normalize_clamp_entry({"value": 1.0})

    def test_non_dict_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            normalize_clamp_entry([1.0, 2.0])


# -----------------------------------------------------------------------
# resolve_effective_clamps
# -----------------------------------------------------------------------


def _entry(direction, value, strength=1.0):
    return {"vector": direction, "value": value, "strength": strength}


class TestResolveEffectiveClamps:
    def test_both_none(self):
        assert resolve_effective_clamps(None, None) is None

    def test_base_only_passthrough(self):
        base = {"post_block": {3: [_entry([1.0], 2.0)]}}
        result = resolve_effective_clamps(base, None)
        assert result == base

    def test_phase_only_passthrough(self):
        phase = {"post_block": {3: [_entry([1.0], 2.0)]}}
        assert resolve_effective_clamps(None, phase) == phase

    def test_concat_base_first(self):
        base = {"post_block": {3: [_entry([1.0, 0.0], 1.0)]}}
        phase = {"post_block": {3: [_entry([0.0, 1.0], 2.0)]}}
        result = resolve_effective_clamps(base, phase)
        entries = result["post_block"][3]
        assert len(entries) == 2
        assert entries[0]["vector"] == [1.0, 0.0]
        assert entries[1]["vector"] == [0.0, 1.0]

    def test_non_overlapping_sites_merge(self):
        base = {"post_block": {3: [_entry([1.0], 1.0)]}}
        phase = {"pre_attn": {5: [_entry([1.0], 2.0)]}}
        result = resolve_effective_clamps(base, phase)
        assert set(result.keys()) == {"post_block", "pre_attn"}

    def test_cap_exceeded_raises(self):
        base = {"post_block": {3: [_entry([1.0], 1.0)] * 3}}
        phase = {"post_block": {3: [_entry([1.0], 2.0)] * 2}}
        with pytest.raises(ValueError, match="max_clamp_directions"):
            resolve_effective_clamps(base, phase, max_directions=4)

    def test_cap_boundary_ok(self):
        base = {"post_block": {3: [_entry([1.0], 1.0)] * 2}}
        phase = {"post_block": {3: [_entry([1.0], 2.0)] * 2}}
        result = resolve_effective_clamps(base, phase, max_directions=4)
        assert len(result["post_block"][3]) == 4

    def test_no_cap_when_max_directions_none(self):
        base = {"post_block": {3: [_entry([1.0], 1.0)] * 8}}
        result = resolve_effective_clamps(base, None)
        assert len(result["post_block"][3]) == 8


# -----------------------------------------------------------------------
# validate_clamp_row_widths
# -----------------------------------------------------------------------


class TestValidateClampRowWidths:
    def test_ok(self):
        spec = {"post_block": {3: [_entry([1.0, 0.0, 0.0], 1.0)]}}
        validate_clamp_row_widths(spec, 3, field_name="steering_clamps")

    def test_wrong_width_raises(self):
        spec = {"post_block": {3: [_entry([1.0, 0.0], 1.0)]}}
        with pytest.raises(ValueError, match="width"):
            validate_clamp_row_widths(spec, 3, field_name="steering_clamps")

    def test_none_ok(self):
        validate_clamp_row_widths(None, 3, field_name="steering_clamps")


# -----------------------------------------------------------------------
# hash_steering_config with clamps
# -----------------------------------------------------------------------

_VEC = {"post_block": {0: [1.0, 2.0, 3.0]}}
_CLAMP = {"post_block": {0: [_entry([1.0, 0.0, 0.0], 4.0)]}}


class TestHashSteeringClamps:
    def test_vector_only_hash_bit_for_bit_unchanged(self):
        """The clamps kwarg must not perturb existing hashes."""
        assert hash_steering_config(_VEC) == hash_steering_config(_VEC, clamps=None)

    def test_clamp_only_hash_nonzero(self):
        """A clamp-only request must reserve a steering row (nonzero hash)."""
        assert hash_steering_config(None, clamps=_CLAMP) != 0

    def test_empty_everything_hashes_zero(self):
        assert hash_steering_config(None, clamps=None) == 0
        assert hash_steering_config(None, clamps={}) == 0

    def test_clamps_change_hash(self):
        assert hash_steering_config(_VEC) != hash_steering_config(
            _VEC, clamps=_CLAMP
        )

    def test_bounds_change_hash(self):
        a = {"post_block": {0: [_entry([1.0, 0.0, 0.0], 4.0)]}}
        b = {"post_block": {0: [_entry([1.0, 0.0, 0.0], 5.0)]}}
        assert hash_steering_config(None, clamps=a) != hash_steering_config(
            None, clamps=b
        )

    def test_strength_changes_hash(self):
        a = {"post_block": {0: [_entry([1.0, 0.0, 0.0], 4.0, strength=1.0)]}}
        b = {"post_block": {0: [_entry([1.0, 0.0, 0.0], 4.0, strength=0.5)]}}
        assert hash_steering_config(None, clamps=a) != hash_steering_config(
            None, clamps=b
        )

    def test_entry_order_changes_hash(self):
        e1 = _entry([1.0, 0.0, 0.0], 1.0)
        e2 = _entry([0.0, 1.0, 0.0], 2.0)
        a = {"post_block": {0: [e1, e2]}}
        b = {"post_block": {0: [e2, e1]}}
        assert hash_steering_config(None, clamps=a) != hash_steering_config(
            None, clamps=b
        )

    def test_deterministic(self):
        assert hash_steering_config(_VEC, clamps=_CLAMP) == hash_steering_config(
            _VEC, clamps=_CLAMP
        )

    def test_domain_separation_vs_module_ref(self):
        """A clamp segment must not collide with a module_ref segment."""
        with_clamp = hash_steering_config(None, clamps=_CLAMP)
        with_ref = hash_steering_config(None, module_ref=("post_block", 1.0))
        assert with_clamp != with_ref

    def test_one_sided_bounds_distinct(self):
        a = {"post_block": {0: [{"vector": [1.0, 0.0, 0.0], "max": 4.0}]}}
        b = {"post_block": {0: [{"vector": [1.0, 0.0, 0.0], "min": 4.0}]}}
        assert hash_steering_config(None, clamps=a) != hash_steering_config(
            None, clamps=b
        )


# -----------------------------------------------------------------------
# SamplingParams clamp fields
# -----------------------------------------------------------------------


class TestSamplingParamsClamps:
    def test_fields_default_none(self):
        sp = SamplingParams()
        assert sp.steering_clamps is None
        assert sp.prefill_steering_clamps is None
        assert sp.decode_steering_clamps is None

    def test_valid_clamps_accepted_and_normalized_in_place(self):
        sp = SamplingParams(
            steering_clamps={"post_block": {2: [_entry([3.0, 4.0], 1.0)]}}
        )
        entry = sp.steering_clamps["post_block"][2][0]
        assert np.allclose(entry["vector"], [0.6, 0.8])
        assert entry["min"] == 1.0
        assert entry["max"] == 1.0
        assert entry["strength"] == 1.0
        assert "value" not in entry

    def test_invalid_hook_rejected(self):
        with pytest.raises(ValueError, match="hook"):
            SamplingParams(
                steering_clamps={"not_a_hook": {0: [_entry([1.0], 1.0)]}}
            )

    def test_negative_layer_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SamplingParams(
                steering_clamps={"post_block": {-1: [_entry([1.0], 1.0)]}}
            )

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError, match="zero"):
            SamplingParams(
                steering_clamps={"post_block": {0: [_entry([0.0, 0.0], 1.0)]}}
            )

    def test_entries_must_be_list(self):
        with pytest.raises(ValueError, match="list"):
            SamplingParams(
                steering_clamps={"post_block": {0: _entry([1.0], 1.0)}}
            )

    def test_effective_prefill_clamps_concat(self):
        sp = SamplingParams(
            steering_clamps={"post_block": {2: [_entry([1.0, 0.0], 1.0)]}},
            prefill_steering_clamps={"post_block": {2: [_entry([0.0, 1.0], 2.0)]}},
        )
        eff = sp.effective_prefill_clamps
        assert len(eff["post_block"][2]) == 2
        assert sp.effective_decode_clamps == sp.steering_clamps

    def test_effective_none_when_no_clamps(self):
        sp = SamplingParams()
        assert sp.effective_prefill_clamps is None
        assert sp.effective_decode_clamps is None

    def test_hash_includes_clamps(self):
        plain = SamplingParams()
        clamped = SamplingParams(
            steering_clamps={"post_block": {2: [_entry([1.0], 1.0)]}}
        )
        assert plain.prefill_steering_config_hash == 0
        assert clamped.prefill_steering_config_hash != 0
        assert clamped.decode_steering_config_hash != 0

    def test_hash_phase_specific(self):
        sp = SamplingParams(
            decode_steering_clamps={"post_block": {2: [_entry([1.0], 1.0)]}}
        )
        assert sp.prefill_steering_config_hash == 0
        assert sp.decode_steering_config_hash != 0

    def test_vector_only_hash_unchanged_by_clamp_support(self):
        """Vector-only requests hash identically to the legacy path."""
        sp = SamplingParams(steering_vectors={"post_block": {0: [1.0, 2.0]}})
        assert sp.prefill_steering_config_hash == hash_steering_config(
            sp.effective_prefill_steering
        )

    def test_pack_path_leaves_clamps_intact(self):
        """maybe_pack_inline_steering_for_request must not clear clamp
        fields, and the hashes must be primed before vector clearing."""
        import torch

        sp = SamplingParams(
            steering_vectors={"post_block": {0: [1.0, 2.0]}},
            steering_clamps={"post_block": {0: [_entry([1.0, 0.0], 1.0)]}},
        )
        expected_prefill = sp.prefill_steering_config_hash
        sp2 = SamplingParams(
            steering_vectors={"post_block": {0: [1.0, 2.0]}},
            steering_clamps={"post_block": {0: [_entry([1.0, 0.0], 1.0)]}},
        )
        maybe_pack_inline_steering_for_request(sp2, torch.float32)
        assert sp2.steering_vectors is None
        assert sp2.steering_clamps is not None
        assert sp2.prefill_steering_config_hash == expected_prefill

    def test_pack_path_clamp_width_gate(self):
        import torch

        sp = SamplingParams(
            steering_clamps={"post_block": {0: [_entry([1.0, 0.0], 1.0)]}}
        )
        with pytest.raises(ValueError, match="width"):
            maybe_pack_inline_steering_for_request(
                sp, torch.float32, expected_row_width=4
            )

    def test_clamp_only_pack_path_noop_but_validates(self):
        """Clamp-only requests have no inline vectors to pack; the pack
        helper must still width-validate the clamp spec."""
        import torch

        sp = SamplingParams(
            steering_clamps={"post_block": {0: [_entry([1.0, 0.0], 1.0)]}}
        )
        maybe_pack_inline_steering_for_request(
            sp, torch.float32, expected_row_width=2
        )
        assert sp.steering_clamps is not None

    def test_from_optional_threads_clamps(self):
        sp = SamplingParams.from_optional(
            steering_clamps={"post_block": {0: [_entry([1.0], 1.0)]}},
            decode_steering_clamps={"post_block": {1: [_entry([1.0], 0.0)]}},
        )
        assert sp.steering_clamps is not None
        assert sp.decode_steering_clamps is not None
