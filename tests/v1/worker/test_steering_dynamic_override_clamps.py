# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-override directional clamps in the dynamic-override pool.

A live dynamic override can carry its OWN clamps (not just the inherited
global-decode composition), so a closed-loop controller can tighten/loosen
bounds mid-decode. Covers:

- ``SteeringManager`` dynamic-clamp storage (register/update/release),
  replace/keep/clear semantics on ``update_dynamic_config(clamps=...)``,
  the dynamic-row composition ``concat(global base, global decode,
  override)`` capped at K with a loud error naming the dynamic id, and
  the ``has_any_clamps`` / ``_site_has_clamps`` gating so a dynamic-only
  clamp still writes buffers.
- The mixin ``RequestSteeringOverride.clamps`` threading through
  ``_apply_request_override`` (validation, compose-admitted, keep/clear).

CPU-only, no engine.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.clamp import (
    CLAMP_ANY_ACTIVE_ATTR,
    CLAMP_BOUNDS_ATTR,
    CLAMP_DIRS_ATTR,
    CLAMP_STRENGTH_ATTR,
)
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_ANY_ACTIVE_ATTR,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_action_queue import RequestSteeringOverride
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
    _SteeringReqState,
)

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
K = 4
NUM_ROWS = MAX_STATIC + MAX_DYNAMIC + 3
_HP = DEFAULT_HOOK_POINT.value  # "post_block"
_HP_ENUM = DEFAULT_HOOK_POINT
INF = float("inf")


def _clamp_entry(axis: int, value: float, strength: float = 1.0) -> dict:
    vec = [0.0] * HIDDEN
    vec[axis] = 2.0  # non-unit: the manager must normalize
    return {"vector": vec, "value": value, "strength": strength}


def _axis(i: int) -> torch.Tensor:
    v = torch.zeros(HIDDEN)
    v[i] = 1.0
    return v


def _vec(value: float = 1.0) -> dict:
    return {_HP: {0: np.full(HIDDEN, value, dtype=np.float32)}}


class _ClampLayer(nn.Module):
    """Fake decoder layer with steering table + clamp + control buffers."""

    def __init__(self, num_rows: int = NUM_ROWS, k: int = K):
        super().__init__()
        self.register_buffer("steering_table_post_block", torch.zeros(num_rows, HIDDEN))
        self.register_buffer(
            HOOK_POINT_ANY_ACTIVE_ATTR[_HP_ENUM], torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer("steering_table_post_block_dynvec", torch.zeros(HIDDEN))
        self.register_buffer("steering_index", torch.zeros(16, dtype=torch.long))
        self.register_buffer("steering_token_scales", torch.zeros(16))
        self.register_buffer("steering_row_gate", torch.ones(16))
        self.register_buffer("steering_decode_mask", torch.zeros(16))
        self.register_buffer(
            CLAMP_DIRS_ATTR[_HP_ENUM], torch.zeros(num_rows, k, HIDDEN)
        )
        bounds = torch.empty(num_rows, k, 2)
        bounds[..., 0] = -INF
        bounds[..., 1] = INF
        self.register_buffer(CLAMP_BOUNDS_ATTR[_HP_ENUM], bounds)
        self.register_buffer(CLAMP_STRENGTH_ATTR[_HP_ENUM], torch.ones(num_rows, k))
        self.register_buffer(
            CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM], torch.zeros(1, dtype=torch.bool)
        )


def _mgr(k: int = K) -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
        max_clamp_directions=k,
    )


def _buffers(layer):
    return (
        getattr(layer, CLAMP_DIRS_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_BOUNDS_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_STRENGTH_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM]),
    )


# ---------------------------------------------------------------------------
# Manager: dynamic-clamp storage + composition
# ---------------------------------------------------------------------------


class TestDynamicClampStorage:
    def test_register_dynamic_with_clamps_stores(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        assert dyn_id in mgr._dynamic_clamps
        assert mgr.has_any_clamps

    def test_register_dynamic_without_clamps_no_entry(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(_vec())
        assert dyn_id not in mgr._dynamic_clamps
        assert not mgr.has_any_clamps

    def test_release_dynamic_drops_clamps(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.release_dynamic_config(dyn_id)
        assert dyn_id not in mgr._dynamic_clamps
        assert not mgr.has_any_clamps

    def test_register_too_many_directions_rejects_without_leaking_row(self):
        mgr = _mgr(k=2)
        before = list(mgr._dynamic_free_rows)
        with pytest.raises(ValueError, match="max_clamp_directions"):
            mgr.register_dynamic_config(
                _vec(), clamps={_HP: {0: [_clamp_entry(0, 1.0)] * 3}}
            )
        # No row leaked, no id burned into the pool map.
        assert list(mgr._dynamic_free_rows) == before
        assert not mgr._dynamic_to_row

    def test_locally_owned_filter(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(),
            clamps={_HP: {0: [_clamp_entry(0, 1.0)], 5: [_clamp_entry(1, 2.0)]}},
            locally_owned_layers=frozenset({5}),
        )
        stored = mgr._dynamic_clamps[dyn_id]
        assert 0 not in stored.get(_HP, {})
        assert 5 in stored[_HP]


class TestDynamicClampUpdateSemantics:
    def test_update_replace(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.update_dynamic_config(
            dyn_id, _vec(), clamps={_HP: {0: [_clamp_entry(1, 9.0)]}}
        )
        payload = mgr._dynamic_clamps[dyn_id][_HP][0]
        torch.testing.assert_close(payload.dirs[0], _axis(1))
        assert payload.bounds[0, 0].item() == 9.0

    def test_update_none_keeps_previous(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.update_dynamic_config(dyn_id, _vec(2.0))  # clamps defaults None => keep
        payload = mgr._dynamic_clamps[dyn_id][_HP][0]
        torch.testing.assert_close(payload.dirs[0], _axis(0))
        assert payload.bounds[0, 0].item() == 5.0

    def test_update_empty_clears(self):
        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.update_dynamic_config(dyn_id, _vec(), clamps={})
        assert dyn_id not in mgr._dynamic_clamps
        assert not mgr.has_any_clamps

    def test_update_clamps_marks_content_not_membership(self):
        mgr = _mgr()
        layers = {0: _ClampLayer()}
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.populate_steering_tables(layers)
        cached = mgr._cached_indices
        assert not mgr._tables_dirty
        mgr.update_dynamic_config(
            dyn_id, _vec(), clamps={_HP: {0: [_clamp_entry(1, 3.0)]}}
        )
        assert mgr._tables_dirty
        assert not mgr._indices_dirty  # membership untouched
        assert mgr._cached_indices is cached  # cached indices still valid


class TestDynamicClampPopulate:
    def test_dynamic_only_clamps_populate_and_flag(self):
        """No global, no per-request config clamp: a dynamic override's OWN
        clamps must still write the buffers and raise the active flag."""
        mgr = _mgr()
        layers = {0: _ClampLayer()}
        _dyn_id, row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0, strength=0.5)]}}
        )
        assert not mgr.has_global_clamps
        assert not any(mgr.config_clamps.values())
        assert mgr.has_any_clamps  # dynamic-only still counts
        mgr.populate_steering_tables(layers)
        dirs, bounds, strength, active = _buffers(layers[0])
        assert bool(active.item())
        torch.testing.assert_close(dirs[row, 0], _axis(0))
        assert bounds[row, 0, 0].item() == 5.0
        assert strength[row, 0].item() == 0.5

    def test_dynamic_row_concats_global_decode_then_override(self):
        mgr = _mgr()
        layers = {0: _ClampLayer()}
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(2, 1.0)], phase="base")
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(3, 2.0)], phase="decode")
        _dyn_id, row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(1, 7.0)]}}
        )
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, _ = _buffers(layers[0])
        # concat order: global base, global decode, override.
        torch.testing.assert_close(dirs[row, 0], _axis(2))
        assert bounds[row, 0, 0].item() == 1.0
        torch.testing.assert_close(dirs[row, 1], _axis(3))
        assert bounds[row, 1, 0].item() == 2.0
        torch.testing.assert_close(dirs[row, 2], _axis(1))
        assert bounds[row, 2, 0].item() == 7.0

    def test_overflow_names_dynamic_id(self):
        mgr = _mgr(k=2)
        layers = {0: _ClampLayer(k=2)}
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 1.0)], phase="base")
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(1, 1.0)], phase="decode")
        dyn_id, _row = mgr.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(2, 2.0)]}}
        )
        with pytest.raises(ValueError, match=f"dynamic id={dyn_id}"):
            mgr.populate_steering_tables(layers)

    def test_update_keep_survives_repopulate(self):
        mgr = _mgr()
        layers = {0: _ClampLayer()}
        dyn_id, row = mgr.register_dynamic_config(
            _vec(1.0), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.populate_steering_tables(layers)
        mgr.update_dynamic_config(dyn_id, _vec(9.0))  # keep clamps, new vectors
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, active = _buffers(layers[0])
        assert bool(active.item())
        torch.testing.assert_close(dirs[row, 0], _axis(0))
        assert bounds[row, 0, 0].item() == 5.0
        # Vectors were rewritten.
        assert torch.all(layers[0].steering_table_post_block[row] == 9.0)


class TestDynamicClampSignature:
    def test_sig_sensitive_to_override_clamps(self):
        m1 = _mgr()
        m2 = _mgr()
        id1, _ = m1.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        id2, _ = m2.register_dynamic_config(
            _vec(), clamps={_HP: {0: [_clamp_entry(0, 6.0)]}}
        )
        assert m1._dynamic_sig[id1] != m2._dynamic_sig[id2]

    def test_sig_vector_only_unchanged_by_feature(self):
        """A clamp-free override hashes exactly as vectors-only (bit-for-bit
        the pre-clamp behavior)."""
        from vllm.config.steering_types import hash_steering_config

        mgr = _mgr()
        dyn_id, _row = mgr.register_dynamic_config(_vec(3.0))
        assert mgr._dynamic_sig[dyn_id] == hash_steering_config(_vec(3.0))


# ---------------------------------------------------------------------------
# Mixin: RequestSteeringOverride.clamps threading
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


class _FakeSchedulerOutput:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = dict(scheduled)


class _Host(SteeringModelRunnerMixin):
    def __init__(self, reqs: list[dict]):
        self._steering_manager = _mgr()
        self._max_clamp_directions = K
        self._steerable_layers_cache = {0: _ClampLayer()}
        self._locally_owned_layers = frozenset({0})
        self._dynamic_steering_stats = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self._steering_module_clamps = {}
        self._steering_module_clamps_effective = {}
        self.input_batch = _FakeInputBatch(reqs)
        self.requests = {}
        self._steering_action_checksum = 0
        self._steering_action_count = 0
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(16, dtype=torch.long)
        self._steering_tier_gain_scratch = np.zeros(8, dtype=np.float32)
        self._steering_token_scales_pinned = torch.zeros(16)
        self._steering_decode_mask_scratch = np.zeros(8, dtype=np.float32)
        self._steering_decode_mask_pinned = torch.zeros(16)
        self._req_decode_sig_reported = {}
        self._pending_decode_sigs = {}


def _decode_req(req_id: str = "r1", **kw) -> dict:
    return {"req_id": req_id, "num_computed": 12, "num_prompt": 8, **kw}


def _clamps_spec(axis: int = 0, value: float = 2.0) -> dict:
    return {_HP: {0: [_clamp_entry(axis, value)]}}


class TestOverrideClampThreading:
    def test_override_with_clamps_registers_and_composes(self):
        host = _Host([_decode_req()])
        action = RequestSteeringOverride(
            req_id="r1", vectors=_vec(5.0), clamps=_clamps_spec(0, 3.0), source="t"
        )
        applied, rejected = host._apply_steering_actions([action], source="s")
        assert (applied, rejected) == (1, 0)
        mgr = host._steering_manager
        dyn_id = host._req_dynamic_decode["r1"]
        assert dyn_id in mgr._dynamic_clamps
        # Populate + verify the dyn row carries the clamp.
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        layer = host._steerable_layers_cache[0]
        row = mgr.get_dynamic_row(dyn_id)
        dirs, bounds, _, active = _buffers(layer)
        assert bool(active.item())
        torch.testing.assert_close(dirs[row, 0], _axis(0))
        assert bounds[row, 0, 0].item() == 3.0

    def test_override_only_clamps_defeats_short_circuit(self):
        """No global, no per-request steering — the override's own clamps
        must still reach never-written buffers (has_dynamic defeats the
        short-circuit; has_any_clamps must include dynamic clamps)."""
        host = _Host([_decode_req()])
        action = RequestSteeringOverride(
            req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(2, 4.0), source="t"
        )
        host._apply_steering_actions([action], source="s")
        mgr = host._steering_manager
        assert not mgr.has_global_clamps
        assert not any(mgr.config_clamps.values())
        assert mgr.has_any_clamps
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        layer = host._steerable_layers_cache[0]
        row = mgr.get_dynamic_row(host._req_dynamic_decode["r1"])
        dirs, bounds, _, active = _buffers(layer)
        assert bool(active.item())
        torch.testing.assert_close(dirs[row, 0], _axis(2))
        assert bounds[row, 0, 0].item() == 4.0

    def test_reemit_none_keeps_clamps(self):
        host = _Host([_decode_req()])
        host._apply_steering_actions(
            [
                RequestSteeringOverride(
                    req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(0, 5.0)
                )
            ],
            source="s",
        )
        dyn_id = host._req_dynamic_decode["r1"]
        # Re-emit with no clamps => keep.
        host._apply_steering_actions(
            [RequestSteeringOverride(req_id="r1", vectors=_vec(2.0))], source="s"
        )
        assert host._req_dynamic_decode["r1"] == dyn_id
        payload = host._steering_manager._dynamic_clamps[dyn_id][_HP][0]
        assert payload.bounds[0, 0].item() == 5.0

    def test_reemit_replaces_clamps(self):
        host = _Host([_decode_req()])
        host._apply_steering_actions(
            [
                RequestSteeringOverride(
                    req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(0, 5.0)
                )
            ],
            source="s",
        )
        dyn_id = host._req_dynamic_decode["r1"]
        host._apply_steering_actions(
            [
                RequestSteeringOverride(
                    req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(1, 8.0)
                )
            ],
            source="s",
        )
        payload = host._steering_manager._dynamic_clamps[dyn_id][_HP][0]
        torch.testing.assert_close(payload.dirs[0], _axis(1))
        assert payload.bounds[0, 0].item() == 8.0

    def test_reemit_empty_clears_clamps(self):
        host = _Host([_decode_req()])
        host._apply_steering_actions(
            [
                RequestSteeringOverride(
                    req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(0, 5.0)
                )
            ],
            source="s",
        )
        dyn_id = host._req_dynamic_decode["r1"]
        host._apply_steering_actions(
            [RequestSteeringOverride(req_id="r1", vectors=_vec(1.0), clamps={})],
            source="s",
        )
        assert dyn_id not in host._steering_manager._dynamic_clamps

    def test_override_bad_clamp_width_rejected(self):
        host = _Host([_decode_req()])
        bad = RequestSteeringOverride(
            req_id="r1",
            vectors=_vec(1.0),
            clamps={_HP: {0: [{"vector": [1.0, 0.0], "value": 1.0}]}},
        )
        applied, rejected = host._apply_steering_actions([bad], source="s")
        assert (applied, rejected) == (0, 1)
        assert "r1" not in host._req_dynamic_decode

    def test_override_too_many_clamp_directions_rejected(self):
        host = _Host([_decode_req()])
        bad = RequestSteeringOverride(
            req_id="r1",
            vectors=_vec(1.0),
            clamps={_HP: {0: [_clamp_entry(i % HIDDEN, 1.0) for i in range(K + 1)]}},
        )
        applied, rejected = host._apply_steering_actions([bad], source="s")
        assert (applied, rejected) == (0, 1)

    def test_override_clear_releases_clamps(self):
        host = _Host([_decode_req()])
        host._apply_steering_actions(
            [
                RequestSteeringOverride(
                    req_id="r1", vectors=_vec(1.0), clamps=_clamps_spec(0, 5.0)
                )
            ],
            source="s",
        )
        host._apply_steering_actions(
            [RequestSteeringOverride(req_id="r1", vectors=None)], source="s"
        )
        assert host._req_dynamic_decode == {}
        assert not host._steering_manager.has_any_clamps

    def test_compose_admitted_concats_admitted_decode_clamps(self):
        host = _Host([_decode_req()])
        sp = SamplingParams(
            max_tokens=4,
            decode_steering_clamps={_HP: {0: [_clamp_entry(3, 1.0)]}},
        )
        host._steering_reqs["r1"] = _SteeringReqState(
            sampling_params=sp,
            prefill_hash=0,
            decode_hash=sp.decode_steering_config_hash,
            num_prompt_tokens=8,
            phase="decode",
        )
        action = RequestSteeringOverride(
            req_id="r1",
            vectors=_vec(1.0),
            clamps=_clamps_spec(1, 6.0),
            compose_admitted=True,
        )
        applied, rejected = host._apply_steering_actions([action], source="s")
        assert (applied, rejected) == (1, 0)
        dyn_id = host._req_dynamic_decode["r1"]
        site = host._steering_manager._dynamic_clamps[dyn_id][_HP][0]
        # admitted decode clamp first (axis 3), then override (axis 1).
        assert site.dirs.shape[0] == 2
        torch.testing.assert_close(site.dirs[0], _axis(3))
        torch.testing.assert_close(site.dirs[1], _axis(1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
