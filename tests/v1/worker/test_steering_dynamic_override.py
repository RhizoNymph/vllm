# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic-override row pool and per-request routing
(Phase 1a M2).

Covers: ``SteeringManager`` dynamic pool (allocation, release,
exhaustion, deterministic ids, populate composition, update-in-place,
indices-cache invalidation), the mixin's ``RequestSteeringOverride``
apply/validate matrix, steering-index routing, and cleanup hooks.
CPU-only, no engine.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import get_steering_buffer_config
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
    SteeringVectorUpdate,
)
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
NUM_ROWS = MAX_STATIC + MAX_DYNAMIC + 3
_HP = "post_block"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("steering_table_post_block", torch.zeros(NUM_ROWS, HIDDEN))
        self.register_buffer(
            "steering_table_post_block_any_active", torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer(
            "steering_table_post_block_dynvec", torch.zeros(HIDDEN)
        )
        self.register_buffer("steering_index", torch.zeros(16, dtype=torch.long))
        self.register_buffer("steering_token_scales", torch.zeros(16))
        self.register_buffer("steering_row_gate", torch.ones(16))
        self.register_buffer("steering_decode_mask", torch.zeros(16))
        self.register_buffer(
            "steering_table_post_block_monitor_probe", torch.zeros(HIDDEN)
        )
        self.register_buffer(
            "steering_table_post_block_monitor_params",
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        )
        self.register_buffer(
            "steering_table_post_block_monitor_active", torch.zeros(1, dtype=torch.bool)
        )


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _vec(value: float = 1.0) -> dict:
    return {_HP: {0: np.full(HIDDEN, value, dtype=np.float32)}}


# ---------------------------------------------------------------------------
# Manager: dynamic pool mechanics
# ---------------------------------------------------------------------------


def test_dynamic_rows_come_from_dedicated_pool():
    mgr = _mgr()
    dyn_id, row = mgr.register_dynamic_config(_vec())
    # Dynamic rows live strictly above the static pool.
    assert row >= MAX_STATIC + 3
    assert dyn_id == 1
    assert mgr.has_dynamic
    # Static pool is untouched.
    assert len(mgr.free_rows) == MAX_STATIC


def test_dynamic_ids_monotonic_never_reused():
    mgr = _mgr()
    id1, row1 = mgr.register_dynamic_config(_vec())
    mgr.release_dynamic_config(id1)
    id2, row2 = mgr.register_dynamic_config(_vec())
    assert id2 == id1 + 1
    # Row IS reused (pool), id is not.
    assert row2 == row1
    with pytest.raises(RuntimeError, match="not registered"):
        mgr.get_dynamic_row(id1)


def test_dynamic_pool_exhaustion_raises():
    mgr = _mgr()
    for _ in range(MAX_DYNAMIC):
        mgr.register_dynamic_config(_vec())
    with pytest.raises(RuntimeError, match="No free dynamic steering rows"):
        mgr.register_dynamic_config(_vec())


def test_dynamic_release_is_idempotent():
    mgr = _mgr()
    dyn_id, _row = mgr.register_dynamic_config(_vec())
    mgr.release_dynamic_config(dyn_id)
    mgr.release_dynamic_config(dyn_id)  # no-op
    assert not mgr.has_dynamic
    assert len(mgr._dynamic_free_rows) == MAX_DYNAMIC


def test_static_pool_isolated_from_dynamic_exhaustion():
    """Exhausting the dynamic pool must not consume static rows, and
    vice versa."""
    mgr = _mgr()
    for _ in range(MAX_DYNAMIC):
        mgr.register_dynamic_config(_vec())
    # Static registrations still succeed up to their own pool size.
    for i in range(MAX_STATIC):
        mgr.register_config(config_hash=100 + i, vectors=_vec(), phase="decode")
    with pytest.raises(RuntimeError, match="No free steering table rows"):
        mgr.register_config(config_hash=999, vectors=_vec(), phase="decode")


def test_zero_dynamic_pool_disables_feature():
    mgr = SteeringManager(max_steering_configs=MAX_STATIC, device=None)
    assert mgr.max_dynamic_steering_configs == 0
    with pytest.raises(RuntimeError, match="No free dynamic steering rows"):
        mgr.register_dynamic_config(_vec())


# ---------------------------------------------------------------------------
# Manager: populate composition
# ---------------------------------------------------------------------------


def test_populate_composes_dynamic_row_with_global_decode():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, torch.full((HIDDEN,), 2.0), phase="decode")
    dyn_id, row = mgr.register_dynamic_config(_vec(5.0))
    mgr.populate_steering_tables(layers)

    table = layers[0].steering_table_post_block
    assert torch.all(table[row] == 7.0)  # global decode (2) + override (5)
    assert torch.all(table[2] == 2.0)  # global decode row unaffected
    assert bool(layers[0].steering_table_post_block_any_active.item())


def test_populate_dynamic_row_without_globals():
    mgr = _mgr()
    layers = {0: _Layer()}
    _dyn_id, row = mgr.register_dynamic_config(_vec(3.0))
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_block[row] == 3.0)


def test_update_dynamic_config_rewrites_same_row():
    mgr = _mgr()
    layers = {0: _Layer()}
    dyn_id, row = mgr.register_dynamic_config(_vec(1.0))
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_block[row] == 1.0)

    mgr.update_dynamic_config(dyn_id, _vec(9.0))
    assert mgr._tables_dirty
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_block[row] == 9.0)
    assert mgr.get_dynamic_row(dyn_id) == row


def test_update_unknown_dynamic_config_raises():
    mgr = _mgr()
    with pytest.raises(KeyError):
        mgr.update_dynamic_config(42, _vec())


def test_register_populate_release_register_cycle():
    """Indices-cache invalidation: a release+register cycle must write
    the new content, never leave stale rows addressed by a live index."""
    mgr = _mgr()
    layers = {0: _Layer()}
    id1, row1 = mgr.register_dynamic_config(_vec(1.0))
    mgr.populate_steering_tables(layers)
    mgr.release_dynamic_config(id1)
    id2, row2 = mgr.register_dynamic_config(_vec(4.0))
    assert row2 == row1
    mgr.populate_steering_tables(layers)
    assert torch.all(layers[0].steering_table_post_block[row2] == 4.0)


def test_dynamic_and_static_rows_coexist_in_populate():
    mgr = _mgr()
    layers = {0: _Layer()}
    static_row = mgr.register_config(config_hash=7, vectors=_vec(2.0), phase="decode")
    _dyn_id, dyn_row = mgr.register_dynamic_config(_vec(5.0))
    mgr.populate_steering_tables(layers)
    table = layers[0].steering_table_post_block
    assert torch.all(table[static_row] == 2.0)
    assert torch.all(table[dyn_row] == 5.0)
    assert static_row != dyn_row


# ---------------------------------------------------------------------------
# Buffer sizing
# ---------------------------------------------------------------------------


def test_buffer_config_includes_dynamic_pool():
    cfg = MagicMock()
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.steering_config.max_steering_configs = MAX_STATIC
    cfg.steering_config.max_dynamic_steering_configs = MAX_DYNAMIC
    assert get_steering_buffer_config(cfg) == (128, MAX_STATIC + MAX_DYNAMIC)


def test_buffer_config_zero_dynamic_matches_legacy():
    cfg = MagicMock()
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.steering_config.max_steering_configs = MAX_STATIC
    cfg.steering_config.max_dynamic_steering_configs = 0
    assert get_steering_buffer_config(cfg) == (128, MAX_STATIC)


def test_buffer_config_steering_disabled():
    cfg = MagicMock()
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.steering_config = None
    assert get_steering_buffer_config(cfg) == (128, 0)


# ---------------------------------------------------------------------------
# Mixin: override apply / validate matrix
# ---------------------------------------------------------------------------


class _FakeInputBatch:
    def __init__(self, reqs: list[dict]):
        self.num_reqs = len(reqs)
        self.req_ids = [r["req_id"] for r in reqs]
        self.req_id_to_index = {r["req_id"]: i for i, r in enumerate(reqs)}
        self.num_computed_tokens_cpu = np.array(
            [r["num_computed"] for r in reqs], dtype=np.int32
        )
        self.num_prompt_tokens = np.array(
            [r["num_prompt"] for r in reqs], dtype=np.int32
        )
        self.request_prefill_steering_hash = np.array(
            [r.get("prefill_hash", 0) for r in reqs], dtype=np.int64
        )
        self.request_decode_steering_hash = np.array(
            [r.get("decode_hash", 0) for r in reqs], dtype=np.int64
        )


class _MixinHost(SteeringModelRunnerMixin):
    def __init__(self, reqs: list[dict], manager: SteeringManager | None = None):
        self._steering_manager = manager if manager is not None else _mgr()
        self._steerable_layers_cache = {0: _Layer()}
        self._locally_owned_layers = frozenset({0})
        self._dynamic_steering_stats = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self.input_batch = _FakeInputBatch(reqs)
        self.requests = {}
        # Scratch for _update_steering_buffers.
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(16, dtype=torch.long)
        self._steering_tier_gain_scratch = np.zeros(8, dtype=np.float32)
        self._steering_token_scales_pinned = torch.zeros(16)
        self._steering_decode_mask_scratch = np.zeros(8, dtype=np.float32)
        self._steering_decode_mask_pinned = torch.zeros(16)
        # APC effective-decode-signature reporting state.
        self._req_decode_sig_reported = {}
        self._pending_decode_sigs = {}


def _override(req: str = "r1", value: float | None = 5.0) -> RequestSteeringOverride:
    vectors = None if value is None else _vec(value)
    return RequestSteeringOverride(req_id=req, vectors=vectors, source="test")


def _decode_req(req_id: str = "r1", **kw) -> dict:
    return {"req_id": req_id, "num_computed": 12, "num_prompt": 8, **kw}


def _prefill_req(req_id: str = "r1") -> dict:
    return {"req_id": req_id, "num_computed": 2, "num_prompt": 8}


def test_override_applies_to_decode_request():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions([_override()], source="s")
    assert (applied, rejected) == (1, 0)
    assert "r1" in host._req_dynamic_decode
    assert host._steering_manager.has_dynamic


def test_override_rejected_for_prefill_request():
    host = _MixinHost([_prefill_req()])
    applied, rejected = host._apply_steering_actions([_override()], source="s")
    assert (applied, rejected) == (0, 1)
    assert host._req_dynamic_decode == {}


def test_override_rejected_for_unknown_request():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions(
        [_override(req="ghost")], source="s"
    )
    assert (applied, rejected) == (0, 1)


def test_override_rejected_for_bad_vectors():
    host = _MixinHost([_decode_req()])
    bad = RequestSteeringOverride(
        req_id="r1",
        vectors={_HP: {0: np.ones(HIDDEN + 1, dtype=np.float32)}},
        source="test",
    )
    applied, rejected = host._apply_steering_actions([bad], source="s")
    assert (applied, rejected) == (0, 1)


def test_override_rejected_when_pool_disabled():
    host = _MixinHost(
        [_decode_req()],
        manager=SteeringManager(max_steering_configs=MAX_STATIC, device=None),
    )
    applied, rejected = host._apply_steering_actions([_override()], source="s")
    assert (applied, rejected) == (0, 1)


def test_override_pool_exhaustion_keeps_previous_state():
    host = _MixinHost([_decode_req("r1"), _decode_req("r2"), _decode_req("r3")])
    assert host._apply_steering_actions([_override("r1")], source="s")[0] == 1
    assert host._apply_steering_actions([_override("r2")], source="s")[0] == 1
    # Pool (size 2) exhausted: r3 rejected, r1/r2 untouched.
    applied, rejected = host._apply_steering_actions([_override("r3")], source="s")
    assert (applied, rejected) == (0, 1)
    assert set(host._req_dynamic_decode) == {"r1", "r2"}


def test_override_reemit_updates_in_place():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override(value=1.0)], source="s")
    dyn_id = host._req_dynamic_decode["r1"]
    host._apply_steering_actions([_override(value=2.0)], source="s")
    # Same dyn_id (update-in-place), still one pool slot used.
    assert host._req_dynamic_decode["r1"] == dyn_id
    assert host._steering_manager.num_active_dynamic_configs == 1


def test_override_clear_releases_row():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override()], source="s")
    applied, rejected = host._apply_steering_actions(
        [_override(value=None)], source="s"
    )
    assert (applied, rejected) == (1, 0)
    assert host._req_dynamic_decode == {}
    assert not host._steering_manager.has_dynamic


def test_override_clear_without_override_is_noop_success():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions(
        [_override(value=None)], source="s"
    )
    assert (applied, rejected) == (1, 0)


def test_mixed_update_and_override_in_one_batch():
    host = _MixinHost([_decode_req()])
    update = SteeringVectorUpdate(vectors=_vec(1.0))
    applied, rejected = host._apply_steering_actions([update, _override()], source="s")
    assert (applied, rejected) == (2, 0)


# ---------------------------------------------------------------------------
# Mixin: steering-index routing + admitted-state isolation
# ---------------------------------------------------------------------------


class _FakeSchedulerOutput:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = dict(scheduled)


def test_decode_only_request_defeats_nothing_active_short_circuit():
    """Regression: a decode-only per-request steered request (prefill_hash=0,
    decode_hash != 0) must NOT be dropped by the nothing-active short-circuit.

    Its decode config is registered lazily at the prefill->decode transition
    inside ``_update_steering_buffers``; if the short-circuit returns first
    (empty manager state, no globals), that transition never runs and the
    steering is silently dropped forever. The batch-pending-steering guard
    must keep the loop alive so the config registers.
    """
    from types import SimpleNamespace

    decode_hash = 4242
    host = _MixinHost(
        [
            {
                "req_id": "r1",
                "num_computed": 6,
                "num_prompt": 8,
                "prefill_hash": 0,
                "decode_hash": decode_hash,
            }
        ]
    )
    sp = SamplingParams(
        max_tokens=4,
        decode_steering_vectors={_HP: {0: [1.0] * HIDDEN}},
    )
    host.requests = {"r1": SimpleNamespace(sampling_params=sp)}
    # Admission populates the canonical _steering_reqs store (prefill_hash=0
    # decode-only ⇒ nothing registered yet, phase "prefill").
    host._steering_add_request(
        SimpleNamespace(
            req_id="r1",
            sampling_params=sp,
            prefill_steering_config_hash=0,
            decode_steering_config_hash=decode_hash,
            prompt_token_ids=list(range(8)),
            prompt_embeds=None,
            num_computed_tokens=6,
        )
    )
    # Nothing registered yet — pre-fix this short-circuits and never registers.
    assert not host._steering_manager.config_to_row
    # 6 + 2 >= 8 ⇒ prefill completes this step ⇒ transition fires.
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 2}))
    assert (decode_hash, "decode") in host._steering_manager.config_to_row, (
        "decode-only request's config was never registered — the short-circuit "
        "swallowed the prefill->decode transition"
    )
    assert host._steering_reqs["r1"].phase == "decode"


def test_decode_routing_uses_dynamic_row_and_admitted_state_untouched():
    host = _MixinHost([_decode_req(decode_hash=77)])
    mgr = host._steering_manager
    admitted_row = mgr.register_config(
        config_hash=77, vectors=_vec(1.0), phase="decode"
    )
    assert mgr.config_refcounts[(77, "decode")] == 1

    host._apply_steering_actions([_override(value=5.0)], source="s")
    dyn_row = mgr.get_dynamic_row(host._req_dynamic_decode["r1"])

    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    steering_index = host._steerable_layers_cache[0].steering_index
    assert int(steering_index[0]) == dyn_row
    # Admitted config untouched: still registered, same row, refcount 1.
    assert mgr.config_to_row[(77, "decode")] == admitted_row
    assert mgr.config_refcounts[(77, "decode")] == 1

    # Clearing the override reverts routing to the admitted row.
    host._apply_steering_actions([_override(value=None)], source="s")
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    assert int(steering_index[0]) == admitted_row


def test_decode_routing_without_override_unchanged():
    host = _MixinHost([_decode_req()])  # decode_hash=0 → global row 2
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    # No steering state at all → nothing-active short-circuit leaves
    # index zeroed.
    steering_index = host._steerable_layers_cache[0].steering_index
    assert int(steering_index[0]) == 0


def test_override_only_state_defeats_nothing_active_short_circuit():
    """An override on a request with no other steering state must still
    populate tables and route (the short-circuit must see has_dynamic)."""
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override(value=5.0)], source="s")
    dyn_row = host._steering_manager.get_dynamic_row(host._req_dynamic_decode["r1"])
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    steering_index = host._steerable_layers_cache[0].steering_index
    assert int(steering_index[0]) == dyn_row
    table = host._steerable_layers_cache[0].steering_table_post_block
    assert torch.all(table[dyn_row] == 5.0)


def test_monitor_only_state_defeats_nothing_active_short_circuit():
    """A monitor-only config (no operator/per-request/tier vectors) must
    still populate so the monitor's active flag is written (Phase 2)."""
    host = _MixinHost([_decode_req()])
    host._steering_manager.set_monitor(
        "post_block", 0, torch.ones(HIDDEN), threshold=0.0, sharpness=4.0
    )
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    layer = host._steerable_layers_cache[0]
    assert bool(layer.steering_table_post_block_monitor_active.item())
    torch.testing.assert_close(
        layer.steering_table_post_block_monitor_params,
        torch.tensor([0.0, 4.0, 0.0], dtype=torch.float32),
    )


def test_monitor_update_dispatches_through_apply_path():
    """A SteeringMonitorUpdate routes to manager.set_monitor; probe=None
    clears it. Exercises the _apply_steering_actions dispatch + validation."""
    host = _MixinHost([_decode_req()])
    probe = np.arange(HIDDEN, dtype=np.float32)
    applied, rejected = host._apply_steering_actions(
        [
            SteeringMonitorUpdate(
                hook="post_block",
                layer=0,
                probe=probe,
                threshold=1.0,
                sharpness=3.0,
                source="s",
            )
        ],
        source="s",
    )
    assert (applied, rejected) == (1, 0)
    cfg = host._steering_manager.monitor_configs["post_block"][0]
    assert cfg["threshold"] == 1.0 and cfg["sharpness"] == 3.0
    # Clear.
    host._apply_steering_actions(
        [SteeringMonitorUpdate(hook="post_block", layer=0, probe=None, source="s")],
        source="s",
    )
    assert not host._steering_manager.has_monitor


def test_monitor_update_bad_probe_rejected():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions(
        [
            SteeringMonitorUpdate(
                hook="post_block",
                layer=0,
                probe=np.ones(HIDDEN + 5, np.float32),
                source="s",
            )
        ],
        source="s",
    )
    assert (applied, rejected) == (0, 1)
    assert not host._steering_manager.has_monitor


def test_clearing_monitor_deactivates_flag_on_transition():
    """Once the monitor is the last active state and is cleared, the
    nothing-active transition must zero its active flag."""
    host = _MixinHost([_decode_req()])
    host._steering_manager.set_monitor(
        "post_block", 0, torch.ones(HIDDEN), 0.0, 4.0
    )
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    layer = host._steerable_layers_cache[0]
    assert bool(layer.steering_table_post_block_monitor_active.item())
    # Clear and step again → short-circuit transition deactivates.
    host._steering_manager.clear_monitor()
    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    assert not bool(layer.steering_table_post_block_monitor_active.item())


# ---------------------------------------------------------------------------
# Mixin: cleanup hooks + leak test
# ---------------------------------------------------------------------------


def test_finished_request_releases_override():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override()], source="s")
    host._steering_finish_requests({"r1"})
    assert host._req_dynamic_decode == {}
    assert not host._steering_manager.has_dynamic


def test_resumption_into_prefill_drops_override():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override()], source="s")
    # Resume re-enters prefill: release-at-preemption already freed the config,
    # so re-registration drops any surviving override. sampling_params=None +
    # zero hashes means nothing is re-registered — the override drop is what we
    # assert here (intent unchanged from the pre-de-fork test).
    req_state = MagicMock()
    req_state.num_prompt_tokens = 8
    req_state.sampling_params = None
    req_state.prefill_steering_config_hash = 0
    req_state.decode_steering_config_hash = 0
    host._reset_steering_for_resumption("r1", req_state, new_num_computed_tokens=0)
    assert host._req_dynamic_decode == {}
    assert not host._steering_manager.has_dynamic


def test_streaming_refresh_drops_override():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override()], source="s")
    # Streaming re-adds route through the canonical admission path; a re-add
    # with no steering (sampling_params=None, zero hashes) still drops the
    # prior decode-run override.
    new_req_data = SimpleNamespace(
        req_id="r1",
        sampling_params=None,
        prefill_steering_config_hash=0,
        decode_steering_config_hash=0,
        prompt_token_ids=list(range(8)),
        prompt_embeds=None,
        num_computed_tokens=0,
    )
    host._steering_add_request(new_req_data)
    assert host._req_dynamic_decode == {}
    assert not host._steering_manager.has_dynamic


def test_no_leaks_after_override_churn():
    host = _MixinHost([_decode_req("r1"), _decode_req("r2")])
    mgr = host._steering_manager
    for _ in range(5):
        host._apply_steering_actions(
            [_override("r1", 1.0), _override("r2", 2.0)], source="s"
        )
        host._apply_steering_actions([_override("r1", None)], source="s")
        host._steering_finish_requests({"r2"})
    assert host._req_dynamic_decode == {}
    assert not mgr.has_dynamic
    assert len(mgr._dynamic_free_rows) == MAX_DYNAMIC
    assert len(mgr.free_rows) == MAX_STATIC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# SteeringScaleUpdate dispatch through the mixin apply path (§5.3)
# ---------------------------------------------------------------------------


def test_scale_update_global_decode_dispatches():
    host = _MixinHost([_decode_req("r1")])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=0.5, source="s")], source="s"
    )
    assert (applied, rejected) == (1, 0)
    assert host._steering_manager._global_scales == {"decode": 0.5}
    assert host._steering_manager._scales_dirty


def test_scale_update_dynamic_row_dispatches():
    host = _MixinHost([_decode_req("r1")])
    dyn_id, _row = host._steering_manager.register_dynamic_config(_vec(1.0))
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=3.0, dyn_id=dyn_id, source="s")], source="s"
    )
    assert (applied, rejected) == (1, 0)
    assert host._steering_manager._dynamic_scales[dyn_id] == 3.0


def test_scale_update_unknown_dynamic_rejected():
    host = _MixinHost([_decode_req("r1")])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=2.0, dyn_id=999, source="s")], source="s"
    )
    assert (applied, rejected) == (0, 1)


def test_scale_update_by_req_id_resolves_to_override_row():
    """A consumer scales a per-request override by req_id (never sees the
    internal dyn_id); the mixin resolves req_id -> dyn_id -> set_dynamic_scale."""
    host = _MixinHost([_decode_req("r1")])
    host._apply_steering_actions([_override(value=5.0)], source="s")
    dyn_id = host._req_dynamic_decode["r1"]
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=0.25, req_id="r1", source="s")], source="s"
    )
    assert (applied, rejected) == (1, 0)
    assert host._steering_manager._dynamic_scales[dyn_id] == 0.25


def test_scale_update_by_req_id_without_override_rejected():
    host = _MixinHost([_decode_req("r1")])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=0.5, req_id="r1", source="s")], source="s"
    )
    assert (applied, rejected) == (0, 1)  # no live override to scale


def test_scale_update_negative_rejected():
    host = _MixinHost([_decode_req("r1")])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=-1.0, source="s")], source="s"
    )
    assert (applied, rejected) == (0, 1)


# ---------------------------------------------------------------------------
# Dynamic tier: per-token gate (token_scales) + short-circuit fix (§5.4)
# ---------------------------------------------------------------------------


def test_token_scales_gate_decode_gets_gain_prefill_zero():
    """token_scales is the tier gain for decode tokens, 0 for prefill."""
    host = _MixinHost(
        [
            _decode_req("r1"),  # 1 decode token
            {"req_id": "p1", "num_computed": 0, "num_prompt": 8},  # prefill
        ]
    )
    mgr = host._steering_manager
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 1.0))
    mgr.set_dynamic_tier_gain(2.0)

    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1, "p1": 3}))
    tscales = host._steerable_layers_cache[0].steering_token_scales
    assert float(tscales[0]) == 2.0  # r1 decode token → gain
    assert torch.all(tscales[1:4] == 0.0)  # p1 prefill tokens → 0
    # dvec buffer carries the tier vector.
    assert torch.all(
        host._steerable_layers_cache[0].steering_table_post_block_dynvec == 1.0
    )


def test_tier_only_state_defeats_nothing_active_short_circuit():
    """A tier-only manager state (no operator/config/override) must NOT be
    short-circuited — the latent bug this slice fixes."""
    host = _MixinHost([_decode_req()])  # decode_hash=0, no config registered
    mgr = host._steering_manager
    mgr.update_dynamic_tier(_HP, 0, torch.full((HIDDEN,), 4.0))
    mgr.set_dynamic_tier_gain(1.0)

    host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
    tscales = host._steerable_layers_cache[0].steering_token_scales
    assert float(tscales[0]) == 1.0  # gate written → not short-circuited
    assert torch.all(
        host._steerable_layers_cache[0].steering_table_post_block_dynvec == 4.0
    )


def test_tier_gain_action_dispatches():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=3.0, tier_gain=True, source="s")], source="s"
    )
    assert (applied, rejected) == (1, 0)
    assert host._steering_manager.dynamic_tier_gain == 3.0


def test_scale_update_rejects_multiple_targets():
    host = _MixinHost([_decode_req()])
    applied, rejected = host._apply_steering_actions(
        [SteeringScaleUpdate(scale=1.0, dyn_id=1, tier_gain=True, source="s")],
        source="s",
    )
    assert (applied, rejected) == (0, 1)


# ---------------------------------------------------------------------------
# Applied-action determinism checksum (§6.1)
# ---------------------------------------------------------------------------


def _scale(req: str = "r1", scale: float = 2.0) -> SteeringScaleUpdate:
    return SteeringScaleUpdate(scale=scale, req_id=req, source="test")


def test_checksum_starts_zero_and_counts_applied():
    host = _MixinHost([_decode_req()])
    assert host._steering_action_checksum == 0
    assert host._steering_action_count == 0
    host._apply_steering_actions([_override()], source="s")
    assert host._steering_action_count == 1
    assert host._steering_action_checksum != 0


def test_checksum_deterministic_across_fresh_hosts():
    """Two fresh hosts applying the same sequence reach the same value."""
    seq = lambda: [_override("r1", 5.0)]  # noqa: E731
    h1 = _MixinHost([_decode_req()])
    h2 = _MixinHost([_decode_req()])
    h1._apply_steering_actions(seq(), source="s")
    h2._apply_steering_actions(seq(), source="s")
    assert h1._steering_action_checksum == h2._steering_action_checksum
    assert h1._steering_action_count == h2._steering_action_count == 1


def test_checksum_fixed_sequence_pythonhashseed_independent():
    """Hardcoded expected value pins the algorithm: no ``hash()``, so
    the checksum is independent of ``PYTHONHASHSEED`` (this value was
    verified identical across seeds 0/1/12345)."""
    host = _MixinHost([_decode_req()])
    # Batch 1: override; batch 2: scale on the same request's dyn row.
    host._apply_steering_actions(
        [RequestSteeringOverride(req_id="r1", vectors=_vec(5.0), source="test")],
        source="s",
    )
    host._apply_steering_actions([_scale("r1", 2.0)], source="s")
    assert host._steering_action_count == 2
    assert f"{host._steering_action_checksum:016x}" == "8bff19a99224e701"


def test_checksum_sensitive_to_content():
    h1 = _MixinHost([_decode_req()])
    h2 = _MixinHost([_decode_req()])
    h1._apply_steering_actions([_override("r1", 5.0)], source="s")
    h2._apply_steering_actions([_override("r1", 7.0)], source="s")
    assert h1._steering_action_checksum != h2._steering_action_checksum


def test_checksum_sensitive_to_order():
    h1 = _MixinHost([_decode_req("r1"), _decode_req("r2")])
    h2 = _MixinHost([_decode_req("r1"), _decode_req("r2")])
    # Same two actions, opposite order → different checksum.
    h1._apply_steering_actions(
        [_override("r1", 5.0), _override("r2", 6.0)], source="s"
    )
    h2._apply_steering_actions(
        [_override("r2", 6.0), _override("r1", 5.0)], source="s"
    )
    assert h1._steering_action_checksum != h2._steering_action_checksum


def test_checksum_sensitive_to_step_boundary():
    """Same actions, one batch vs two, differ via the drain-batch ordinal."""
    h1 = _MixinHost([_decode_req("r1"), _decode_req("r2")])
    h2 = _MixinHost([_decode_req("r1"), _decode_req("r2")])
    h1._apply_steering_actions(
        [_override("r1", 5.0), _override("r2", 6.0)], source="s"
    )
    h2._apply_steering_actions([_override("r1", 5.0)], source="s")
    h2._apply_steering_actions([_override("r2", 6.0)], source="s")
    assert h1._steering_action_count == h2._steering_action_count == 2
    assert h1._steering_action_checksum != h2._steering_action_checksum


def test_checksum_rejected_actions_do_not_fold():
    host = _MixinHost([_prefill_req()])  # override rejected for prefill req
    applied, rejected = host._apply_steering_actions([_override()], source="s")
    assert (applied, rejected) == (0, 1)
    assert host._steering_action_count == 0
    assert host._steering_action_checksum == 0


def test_checksum_unknown_action_type_does_not_fold():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([object()], source="s")
    assert host._steering_action_count == 0
    assert host._steering_action_checksum == 0


def test_status_dict_exposes_checksum_fields():
    host = _MixinHost([_decode_req()])
    host._apply_steering_actions([_override()], source="s")
    status = host.get_dynamic_steering_status()
    assert status["action_count"] == 1
    assert status["action_checksum"] == (
        f"{host._steering_action_checksum:016x}"
    )
    # Hex u64: 16 chars, valid hex.
    assert len(status["action_checksum"]) == 16
    int(status["action_checksum"], 16)
