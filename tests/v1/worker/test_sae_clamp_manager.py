# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``SAEClampManager``.

The manager is a pure data structure that mirrors ``SteeringManager``
but without the global-vector tier (SAE feature surgery has no
"global" clamp): row 0 is the no-op sentinel and rows 1..max are
per-(spec-hash, phase) configurations.

These tests exercise the manager in isolation — no GPU buffers, no
worker mixin, no actual encoding/decoding.  Buffer materialization
lives in a sibling module and is tested separately.
"""

from __future__ import annotations

import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager


def _make_spec(
    *,
    module_name: str = "golden_gate",
    feature_idx: int = 34,
    value: float = 5.0,
    kind: str = "absolute",
    layer: int = 20,
    hook: str = "post_mlp",
    phase: str = "both",
    only_if_active: bool = False,
) -> SAEClampSpec:
    return SAEClampSpec(
        module_name=module_name,
        phase=phase,  # type: ignore[arg-type]
        clamps={
            hook: {
                layer: (
                    SAEClampEntry(
                        feature_idx=feature_idx,
                        kind=kind,  # type: ignore[arg-type]
                        value=value,
                        only_if_active=only_if_active,
                    ),
                )
            }
        },
    )


class TestRowAllocation:
    """Row 0 is no-op; allocated rows start at 1, monotonically rising."""

    def test_initial_state_has_no_rows(self):
        m = SAEClampManager(max_sae_configs=4)
        assert m.config_to_row == {}
        assert m.free_rows == [4, 3, 2, 1]

    def test_register_returns_row_one_first(self):
        m = SAEClampManager(max_sae_configs=4)
        row = m.register_clamp_spec(123, (_make_spec(),), "prefill")
        assert row == 1
        assert m.config_to_row == {(123, "prefill"): 1}

    def test_register_consecutive_rows(self):
        m = SAEClampManager(max_sae_configs=4)
        r1 = m.register_clamp_spec(1, (_make_spec(value=1.0),), "prefill")
        r2 = m.register_clamp_spec(2, (_make_spec(value=2.0),), "prefill")
        r3 = m.register_clamp_spec(3, (_make_spec(value=3.0),), "prefill")
        assert (r1, r2, r3) == (1, 2, 3)

    def test_same_hash_same_phase_returns_existing_row_and_increments_refcount(self):
        m = SAEClampManager(max_sae_configs=4)
        r1 = m.register_clamp_spec(1, (_make_spec(),), "prefill")
        r2 = m.register_clamp_spec(1, (_make_spec(),), "prefill")
        assert r1 == r2 == 1
        assert m.config_refcounts[(1, "prefill")] == 2

    def test_same_sae_content_different_request_hash_aliases_row(self):
        m = SAEClampManager(max_sae_configs=4)
        spec = _make_spec()

        r1 = m.register_clamp_spec(1, (spec,), "prefill")
        r2 = m.register_clamp_spec(2, (spec,), "prefill")

        assert r1 == r2 == 1
        assert m.config_to_row[(1, "prefill")] == 1
        assert m.config_to_row[(2, "prefill")] == 1
        assert m.config_refcounts[(1, "prefill")] == 1
        assert m.config_refcounts[(2, "prefill")] == 1
        assert len(list(m.active_rows())) == 1

    def test_both_and_prefill_specs_alias_prefill_row(self):
        m = SAEClampManager(max_sae_configs=4)
        both = _make_spec(phase="both")
        prefill = _make_spec(phase="prefill")

        r1 = m.register_clamp_spec(1, (both,), "prefill")
        r2 = m.register_clamp_spec(2, (prefill,), "prefill")

        assert r1 == r2 == 1
        assert len(list(m.active_rows())) == 1

    def test_same_hash_different_phase_gets_new_row(self):
        m = SAEClampManager(max_sae_configs=4)
        r_prefill = m.register_clamp_spec(1, (_make_spec(),), "prefill")
        r_decode = m.register_clamp_spec(1, (_make_spec(),), "decode")
        assert r_prefill != r_decode
        assert {r_prefill, r_decode} == {1, 2}


class TestCapacityContract:
    """Strict-capacity contract: registration past capacity raises."""

    def test_register_at_capacity_raises(self):
        m = SAEClampManager(max_sae_configs=2)
        m.register_clamp_spec(1, (_make_spec(value=1.0),), "prefill")
        m.register_clamp_spec(2, (_make_spec(value=2.0),), "prefill")
        with pytest.raises(RuntimeError, match="No free SAE clamp"):
            m.register_clamp_spec(3, (_make_spec(value=3.0),), "prefill")

    def test_release_then_register_reuses_freed_row(self):
        m = SAEClampManager(max_sae_configs=2)
        r1 = m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.register_clamp_spec(2, (_make_spec(),), "prefill")
        m.release_clamp_spec(1, "prefill")
        # Lowest free row gets reused — deterministic across ranks.
        r3 = m.register_clamp_spec(3, (_make_spec(),), "prefill")
        assert r3 == r1

    def test_max_zero_raises_on_first_register(self):
        m = SAEClampManager(max_sae_configs=0)
        with pytest.raises(RuntimeError):
            m.register_clamp_spec(1, (_make_spec(),), "prefill")


class TestRefcounting:
    """Multiple requests share the same row when their hash and phase match."""

    def test_release_decrements_refcount_keeps_row_when_positive(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.release_clamp_spec(1, "prefill")
        assert m.config_refcounts[(1, "prefill")] == 1
        assert (1, "prefill") in m.config_to_row

    def test_release_to_zero_frees_row(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.release_clamp_spec(1, "prefill")
        assert (1, "prefill") not in m.config_to_row
        assert (1, "prefill") not in m.config_refcounts

    def test_release_unknown_is_noop(self):
        m = SAEClampManager(max_sae_configs=4)
        # Must not raise, must not alter free_rows.
        m.release_clamp_spec(999, "prefill")
        assert m.free_rows == [4, 3, 2, 1]


class TestGetRowForConfig:
    """Lookup mirrors SteeringManager: hash 0 → no-op row, registered → assigned."""

    def test_hash_zero_returns_row_zero_for_both_phases(self):
        m = SAEClampManager(max_sae_configs=4)
        assert m.get_row_for_config(0, is_prefill=True) == 0
        assert m.get_row_for_config(0, is_prefill=False) == 0

    def test_registered_hash_returns_assigned_row(self):
        m = SAEClampManager(max_sae_configs=4)
        row_p = m.register_clamp_spec(7, (_make_spec(),), "prefill")
        row_d = m.register_clamp_spec(7, (_make_spec(),), "decode")
        assert m.get_row_for_config(7, is_prefill=True) == row_p
        assert m.get_row_for_config(7, is_prefill=False) == row_d

    def test_unregistered_nonzero_hash_raises(self):
        m = SAEClampManager(max_sae_configs=4)
        with pytest.raises(RuntimeError, match="not registered"):
            m.get_row_for_config(42, is_prefill=True)

    def test_phase_mismatch_raises(self):
        # Registered for prefill only — decode lookup must fail loud.
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(7, (_make_spec(),), "prefill")
        with pytest.raises(RuntimeError, match="not registered"):
            m.get_row_for_config(7, is_prefill=False)


class TestActiveRowsEnumeration:
    """``active_rows`` is the contract used by the buffer populator."""

    def test_iterates_in_row_order(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(10, (_make_spec(module_name="a"),), "prefill")
        m.register_clamp_spec(20, (_make_spec(module_name="b"),), "decode")
        m.register_clamp_spec(30, (_make_spec(module_name="c"),), "prefill")
        rows = list(m.active_rows())
        # (row, hash, phase, specs) tuples; sorted by row index.
        assert [r[0] for r in rows] == [1, 2, 3]
        assert [r[2] for r in rows] == ["prefill", "decode", "prefill"]
        assert [r[3][0].module_name for r in rows] == ["a", "b", "c"]

    def test_yields_specs_tuple_for_repopulation(self):
        m = SAEClampManager(max_sae_configs=4)
        spec = _make_spec(value=99.0)
        m.register_clamp_spec(7, (spec,), "prefill")
        rows = list(m.active_rows())
        assert len(rows) == 1
        _, _, _, returned_specs = rows[0]
        assert returned_specs == (spec,)

    def test_yields_full_specs_tuple(self):
        m = SAEClampManager(max_sae_configs=4)
        s_a = _make_spec(module_name="a")
        s_b = _make_spec(module_name="b")
        m.register_clamp_spec(7, (s_a, s_b), "prefill")
        rows = list(m.active_rows())
        _, _, _, returned_specs = rows[0]
        # Both modules' specs share one row — populator iterates per-spec.
        assert returned_specs == (s_a, s_b)

    def test_empty_specs_tuple_rejected(self):
        m = SAEClampManager(max_sae_configs=4)
        with pytest.raises(ValueError, match="empty specs"):
            m.register_clamp_spec(7, (), "prefill")

    def test_overlapping_specs_rejected(self):
        m = SAEClampManager(max_sae_configs=4)
        spec_a = _make_spec(phase="both", value=1.0)
        spec_b = _make_spec(phase="prefill", value=2.0)
        with pytest.raises(ValueError, match="overlapping clamps"):
            m.register_clamp_spec(7, (spec_a, spec_b), "prefill")

    def test_same_feature_disjoint_phases_allowed(self):
        m = SAEClampManager(max_sae_configs=4)
        spec_a = _make_spec(phase="prefill", value=1.0)
        spec_b = _make_spec(phase="decode", value=2.0)
        row = m.register_clamp_spec(7, (spec_a, spec_b), "prefill")
        assert row == 1

    def test_specs_with_no_entries_for_phase_rejected(self):
        m = SAEClampManager(max_sae_configs=4)
        with pytest.raises(ValueError, match="do not apply"):
            m.register_clamp_spec(7, (_make_spec(phase="decode"),), "prefill")

    def test_freed_row_is_not_yielded(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(10, (_make_spec(),), "prefill")
        m.register_clamp_spec(20, (_make_spec(),), "decode")
        m.release_clamp_spec(10, "prefill")
        rows = list(m.active_rows())
        assert [(r[2], r[3][0].module_name) for r in rows] == [
            ("decode", "golden_gate")
        ]


class TestDirtyFlag:
    """``_tables_dirty`` flips on every state mutation."""

    def test_starts_dirty(self):
        m = SAEClampManager(max_sae_configs=4)
        # Initial state must trigger first populate.
        assert m._tables_dirty is True

    def test_register_sets_dirty(self):
        m = SAEClampManager(max_sae_configs=4)
        m.mark_tables_clean()
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        assert m._tables_dirty is True

    def test_register_refcount_hit_does_not_set_dirty(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.mark_tables_clean()
        # Same (hash, phase) — refcount bump only, no row content change.
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        assert m._tables_dirty is False

    def test_release_to_zero_sets_dirty(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.mark_tables_clean()
        m.release_clamp_spec(1, "prefill")
        assert m._tables_dirty is True

    def test_release_decrement_only_does_not_set_dirty(self):
        m = SAEClampManager(max_sae_configs=4)
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.register_clamp_spec(1, (_make_spec(),), "prefill")
        m.mark_tables_clean()
        m.release_clamp_spec(1, "prefill")
        # Refcount went 2 -> 1; row still occupied; no row content change.
        assert m._tables_dirty is False
