# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SAEFullReconstructionManager — row allocation + refcount.

Mirrors :mod:`tests.v1.worker.test_sae_clamp_manager` for the Phase-4
full-reconstruction path: row 0 = no-op sentinel, rows 1+ per-config
with refcounting and strict capacity, deterministic-replay across
ranks via the same register / release contract.
"""

from __future__ import annotations

import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEFullReconstructionSpec,
)
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)


def _spec(
    *,
    module_name: str = "m",
    clamps: dict | None = None,
    phase: str = "both",
) -> SAEFullReconstructionSpec:
    return SAEFullReconstructionSpec(
        module_name=module_name,
        clamps=clamps or {},
        phase=phase,
    )


class TestConstruction:
    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SAEFullReconstructionManager(-1)

    def test_zero_capacity_constructs(self):
        # Disabled-mode engines instantiate the manager with capacity 0.
        m = SAEFullReconstructionManager(0)
        assert m.max_recon_configs == 0
        assert m.free_rows == []

    def test_positive_capacity_layout(self):
        m = SAEFullReconstructionManager(4)
        # Free rows contain 1..4, with pop() returning low rows first.
        assert m.free_rows == [4, 3, 2, 1]


class TestRegisterReleaseRoundTrip:
    def test_first_register_returns_lowest_row(self):
        m = SAEFullReconstructionManager(4)
        row = m.register_recon_spec(0xABC, (_spec(),), "prefill")
        assert row == 1

    def test_repeated_register_increments_refcount(self):
        m = SAEFullReconstructionManager(4)
        first = m.register_recon_spec(0xABC, (_spec(),), "prefill")
        second = m.register_recon_spec(0xABC, (_spec(),), "prefill")
        assert first == second
        assert m.config_refcounts[(0xABC, "prefill")] == 2

    def test_distinct_phases_allocate_separately(self):
        m = SAEFullReconstructionManager(4)
        r1 = m.register_recon_spec(0xABC, (_spec(),), "prefill")
        r2 = m.register_recon_spec(0xABC, (_spec(),), "decode")
        assert r1 != r2

    def test_release_reduces_refcount_then_frees(self):
        m = SAEFullReconstructionManager(4)
        m.register_recon_spec(1, (_spec(),), "prefill")
        m.register_recon_spec(1, (_spec(),), "prefill")
        m.release_recon_spec(1, "prefill")
        # Still held by refcount=1.
        assert (1, "prefill") in m.config_to_row
        m.release_recon_spec(1, "prefill")
        assert (1, "prefill") not in m.config_to_row

    def test_release_unregistered_is_silent_no_op(self):
        # Worker request-completion path calls this unconditionally.
        m = SAEFullReconstructionManager(4)
        m.release_recon_spec(0xDEAD, "prefill")  # no error

    def test_capacity_exceeded_raises(self):
        m = SAEFullReconstructionManager(2)
        m.register_recon_spec(1, (_spec(),), "prefill")
        m.register_recon_spec(2, (_spec(),), "prefill")
        with pytest.raises(RuntimeError, match="No free SAE full"):
            m.register_recon_spec(3, (_spec(),), "prefill")


class TestRegisterValidation:
    def test_zero_hash_rejected(self):
        m = SAEFullReconstructionManager(4)
        with pytest.raises(ValueError, match="reserved"):
            m.register_recon_spec(0, (_spec(),), "prefill")

    def test_invalid_phase_rejected(self):
        m = SAEFullReconstructionManager(4)
        with pytest.raises(ValueError, match="phase"):
            m.register_recon_spec(1, (_spec(),), "bogus")

    def test_empty_specs_rejected(self):
        # Pure reconstruction lives inside SAEFullReconstructionSpec
        # (clamps may be empty); but registering with NO spec is a
        # caller bug.
        m = SAEFullReconstructionManager(4)
        with pytest.raises(ValueError, match="empty specs"):
            m.register_recon_spec(1, (), "prefill")


class TestPureReconstructionRegistration:
    """Specs with empty clamps must still register and round-trip.

    Pure reconstruction is meaningful for this kind: a non-zero row
    with empty clamps means "reconstruct without modifying any
    activations".  The dispatch shim's recon_mask = (recon_index != 0)
    triggers the reconstruction even when the clamp tables are zero.
    """

    def test_empty_clamps_still_allocates_row(self):
        m = SAEFullReconstructionManager(4)
        bare = SAEFullReconstructionSpec(module_name="m")
        row = m.register_recon_spec(0xCAFE, (bare,), "prefill")
        assert row >= 1
        assert m.config_specs[(0xCAFE, "prefill")] == (bare,)

    def test_mixed_specs_with_and_without_clamps(self):
        m = SAEFullReconstructionManager(4)
        modified = SAEFullReconstructionSpec(
            module_name="m1",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=3, kind="absolute", value=5.0),)
                }
            },
        )
        bare = SAEFullReconstructionSpec(module_name="m2")
        row = m.register_recon_spec(0x1234, (modified, bare), "decode")
        assert row >= 1
        specs = m.config_specs[(0x1234, "decode")]
        assert len(specs) == 2


class TestGetRowForConfig:
    def test_zero_hash_returns_zero(self):
        m = SAEFullReconstructionManager(4)
        assert m.get_row_for_config(0, is_prefill=True) == 0
        assert m.get_row_for_config(0, is_prefill=False) == 0

    def test_returns_assigned_row(self):
        m = SAEFullReconstructionManager(4)
        r = m.register_recon_spec(0xAB, (_spec(),), "decode")
        assert m.get_row_for_config(0xAB, is_prefill=False) == r

    def test_unregistered_raises(self):
        m = SAEFullReconstructionManager(4)
        with pytest.raises(RuntimeError, match="not registered"):
            m.get_row_for_config(0xAB, is_prefill=True)


class TestActiveRowsIteration:
    def test_iterates_in_row_order(self):
        m = SAEFullReconstructionManager(4)
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        m.register_recon_spec(0xB, (_spec(),), "prefill")
        m.register_recon_spec(0xC, (_spec(),), "decode")
        rows = [r for r, *_ in m.active_rows()]
        assert rows == sorted(rows)
        assert len(rows) == 3


class TestDirtyFlag:
    def test_starts_dirty(self):
        m = SAEFullReconstructionManager(4)
        assert m._tables_dirty is True

    def test_register_marks_dirty(self):
        m = SAEFullReconstructionManager(4)
        m.mark_tables_clean()
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        assert m._tables_dirty is True

    def test_release_at_zero_marks_dirty(self):
        m = SAEFullReconstructionManager(4)
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        m.mark_tables_clean()
        m.release_recon_spec(0xA, "prefill")
        assert m._tables_dirty is True

    def test_release_above_zero_keeps_clean(self):
        # Refcount drop without freeing the row doesn't change content.
        m = SAEFullReconstructionManager(4)
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        m.mark_tables_clean()
        m.release_recon_spec(0xA, "prefill")
        assert m._tables_dirty is False

    def test_register_existing_keeps_clean(self):
        m = SAEFullReconstructionManager(4)
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        m.mark_tables_clean()
        m.register_recon_spec(0xA, (_spec(),), "prefill")
        assert m._tables_dirty is False
