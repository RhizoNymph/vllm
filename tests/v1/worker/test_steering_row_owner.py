# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed RowOwner state + single purge path + dirty-state grouping.

Covers the ownership refactor:

* ``release_config`` at refcount 0 purges the config's owner-keyed runtime
  state (per-config scale + per-row monitors) — the closed leak/sticky-
  semantics bug — but NOT before the last live registration releases.
* A scale pre-armed for a not-yet-registered hash survives until the first
  live->0 transition (pre-arming still works).
* ``_purge_owner`` clears every store in the owner-store registry
  (parametrized so a future unpurged store fails).
* The APC decode signature drops a monitor fold after release/re-register
  (the intended new contract: a monitor no longer survives a hash's release).
* ``_DirtyState`` implication matrix (membership => content; scales-only
  eligibility; full-populate clear).

CPU-only, no engine.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.v1.worker.steering_manager import SteeringManager, _DirtyState
from vllm.v1.worker.steering_owner import RowOwner

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
NUM_ROWS = MAX_STATIC + MAX_DYNAMIC + 3
HOOK = "post_block"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "steering_table_post_block", torch.zeros(NUM_ROWS, HIDDEN)
        )
        self.register_buffer(
            "steering_table_post_block_any_active", torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer("steering_scales", torch.ones(NUM_ROWS))


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _vec(value: float = 0.0) -> dict:
    return {HOOK: {0: np.full(HIDDEN, value, dtype=np.float32)}}


def _reg_decode(mgr, config_hash) -> int:
    return mgr.register_config(config_hash, _vec(), phase="decode")


# ---------------------------------------------------------------------------
# refcount-0 purge of config-owned scale + row monitor
# ---------------------------------------------------------------------------


def test_release_config_refcount0_purges_scale_and_monitor():
    mgr = _mgr()
    ch = 55
    owner = RowOwner.config(ch, "decode")
    _reg_decode(mgr, ch)
    mgr.set_row_scale(ch, "decode", 2.0)
    mgr.set_row_monitor(HOOK, 0, owner, torch.ones(HIDDEN), 0.0, 1.0)
    assert owner in mgr._row_scales
    assert mgr.has_row_monitor is True

    mgr.release_config(ch, "decode")  # live -> 0

    assert owner not in mgr._row_scales
    assert mgr.has_row_monitor is False


def test_purge_only_on_live_to_zero_not_before():
    mgr = _mgr()
    ch = 55
    owner = RowOwner.config(ch, "decode")
    _reg_decode(mgr, ch)  # refcount 1
    _reg_decode(mgr, ch)  # refcount 2 (same row)
    mgr.set_row_scale(ch, "decode", 2.0)
    mgr.set_row_monitor(HOOK, 0, owner, torch.ones(HIDDEN), 0.0, 1.0)

    mgr.release_config(ch, "decode")  # refcount 1, still live -> no purge
    assert owner in mgr._row_scales
    assert mgr.has_row_monitor is True

    mgr.release_config(ch, "decode")  # refcount 0 -> purge
    assert owner not in mgr._row_scales
    assert mgr.has_row_monitor is False


def test_prearmed_scale_survives_until_first_live_to_zero():
    mgr = _mgr()
    ch = 77
    owner = RowOwner.config(ch, "decode")
    # Pre-arm a scale for a hash that is not registered yet.
    mgr.set_row_scale(ch, "decode", 3.0)
    assert owner in mgr._row_scales

    # Releasing an unregistered hash is a no-op: the pre-armed scale survives.
    mgr.release_config(ch, "decode")
    assert owner in mgr._row_scales

    # Register (scale still armed), then release to 0 -> now purged.
    _reg_decode(mgr, ch)
    assert owner in mgr._row_scales
    mgr.release_config(ch, "decode")
    assert owner not in mgr._row_scales


def test_prefill_release_leaves_decode_owner_intact():
    mgr = _mgr()
    ch = 88
    mgr.register_config(ch, _vec(), phase="prefill")
    mgr.register_config(ch, _vec(), phase="decode")
    mgr.set_row_scale(ch, "decode", 2.0)
    mgr.set_row_monitor(
        HOOK, 0, RowOwner.config(ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )

    # Releasing the prefill registration purges only the prefill owner.
    mgr.release_config(ch, "prefill")
    assert RowOwner.config(ch, "decode") in mgr._row_scales
    assert mgr.has_row_monitor is True


# ---------------------------------------------------------------------------
# _purge_owner clears every registered store (registry-parametrized)
# ---------------------------------------------------------------------------


def test_purge_owner_clears_every_registered_store():
    mgr = _mgr()
    owner = RowOwner.dyn(42)
    stores = mgr._owner_stores()
    assert stores, "owner-store registry must be non-empty"

    for store in stores:
        store.install_dummy(owner)
    for store in stores:
        assert store.contains(owner), f"install_dummy failed for {store.name}"

    mgr._purge_owner(owner)

    for store in mgr._owner_stores():
        assert not store.contains(owner), f"{store.name} was not purged"


@pytest.mark.parametrize("store_name", [s.name for s in _mgr()._owner_stores()])
def test_each_store_is_purged(store_name):
    mgr = _mgr()
    stores = {s.name: s for s in mgr._owner_stores()}
    store = stores[store_name]
    owner = RowOwner.config(123, "decode")
    store.install_dummy(owner)
    assert store.contains(owner)
    mgr._purge_owner(owner)
    assert not store.contains(owner)


# ---------------------------------------------------------------------------
# APC signature drops the monitor fold after release/re-register
# ---------------------------------------------------------------------------


def test_apc_signature_drops_monitor_after_release_reregister():
    mgr = _mgr()
    ch = 999
    _reg_decode(mgr, ch)
    mgr.set_row_monitor(
        HOOK, 0, RowOwner.config(ch, "decode"), torch.ones(HIDDEN), 0.0, 1.0
    )
    sig_with_monitor = mgr.effective_decode_signature(None, ch)
    assert sig_with_monitor is not None

    # Release (purges the monitor) and re-register the SAME content hash.
    mgr.release_config(ch, "decode")
    _reg_decode(mgr, ch)

    # The intended new contract: the monitor did not survive the release, so
    # nothing dynamic applies and the signature is None (distinct from before).
    sig_after = mgr.effective_decode_signature(None, ch)
    assert sig_after is None
    assert sig_after != sig_with_monitor


# ---------------------------------------------------------------------------
# _DirtyState implication matrix
# ---------------------------------------------------------------------------


def test_dirtystate_membership_implies_content():
    d = _DirtyState(content=False, membership=False, scales=False)
    d.mark_membership()
    assert d.membership is True
    assert d.content is True  # membership => content


def test_dirtystate_mark_content_and_scales_are_narrow():
    d = _DirtyState(content=False, membership=False, scales=False)
    d.mark_content()
    assert d.content is True and d.membership is False and d.scales is False
    d.mark_scales()
    assert d.scales is True and d.membership is False


@pytest.mark.parametrize(
    "content,membership,scales,eligible",
    [
        (False, False, True, True),  # only scales dirty -> cheap path
        (True, False, True, False),  # content dirty -> full populate
        (False, True, True, False),  # membership dirty -> full populate
        (False, False, False, False),  # nothing dirty
    ],
)
def test_dirtystate_scales_only_eligibility(content, membership, scales, eligible):
    d = _DirtyState(content=content, membership=membership, scales=scales)
    assert d.scales_only_eligible is eligible


def test_dirtystate_full_populate_clears_all():
    d = _DirtyState(content=True, membership=True, scales=True)
    d.clear_after_full_populate()
    assert not (d.content or d.membership or d.scales)


def test_new_manager_starts_all_dirty():
    mgr = _mgr()
    assert mgr._tables_dirty is True
    assert mgr._scales_dirty is True
    assert mgr._indices_dirty is True


# ---------------------------------------------------------------------------
# populate_steering_scales stale-indices fallback (membership dirty)
# ---------------------------------------------------------------------------


def test_scales_populate_falls_back_to_full_when_membership_dirty():
    mgr = _mgr()
    layers = {0: _Layer()}
    # First full populate clears every flag and caches the indices scratch.
    mgr.populate_steering_tables(layers)
    assert not mgr._indices_dirty

    # A new config change makes membership (indices) dirty. The cheap scales
    # path must fall back to a full populate that rebuilds the indices.
    _reg_decode(mgr, 314)
    mgr.set_row_scale(314, "decode", 2.0)
    assert mgr._indices_dirty and mgr._scales_dirty
    mgr.populate_steering_scales(layers)

    # Full populate ran: every flag cleared and the new row's scale landed.
    assert not (mgr._tables_dirty or mgr._scales_dirty or mgr._indices_dirty)
    row = mgr.get_row_for_config(314, is_prefill=False)
    assert layers[0].steering_scales[row].item() == 2.0


def test_scales_only_cheap_path_clears_only_scales():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.populate_steering_tables(layers)
    mgr.set_global_scale("decode", 0.25)
    assert mgr._scales_dirty and not mgr._tables_dirty and not mgr._indices_dirty
    mgr.populate_steering_scales(layers)
    assert not mgr._scales_dirty
    assert layers[0].steering_scales[2].item() == 0.25


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
