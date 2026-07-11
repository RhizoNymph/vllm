# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SteeringManager directional-clamp storage and populate.

Clamps ride the steering row machinery: per-config clamp payloads are
stored at ``register_config`` time, global clamps live in three tiers
(base/prefill/decode), and ``populate_steering_tables`` writes the
per-(hook, layer) clamp dirs/bounds/strength buffers in the same
row-position order (and with the same scatter indices) as the steering
tables.  Row 0 stays the all-zero sentinel; rows 1/2 carry the global
effective clamps; config rows carry concat(global base, global phase,
per-request), capped at K with a loud error.
"""

import math

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
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN_SIZE = 8
MAX_CONFIGS = 4
K = 4
_HP = DEFAULT_HOOK_POINT.value  # "post_block"
_HP_ENUM = DEFAULT_HOOK_POINT
_TABLE_ATTR = "steering_table_post_block"

INF = float("inf")


class FakeClampLayer(nn.Module):
    """Fake decoder layer with a steering table and clamp buffers."""

    def __init__(self, num_rows: int, hidden_size: int, k: int = K):
        super().__init__()
        self.register_buffer(_TABLE_ATTR, torch.zeros(num_rows, hidden_size))
        self.register_buffer(
            HOOK_POINT_ANY_ACTIVE_ATTR[_HP_ENUM], torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer(
            CLAMP_DIRS_ATTR[_HP_ENUM], torch.zeros(num_rows, k, hidden_size)
        )
        bounds = torch.empty(num_rows, k, 2)
        bounds[..., 0] = -INF
        bounds[..., 1] = INF
        self.register_buffer(CLAMP_BOUNDS_ATTR[_HP_ENUM], bounds)
        self.register_buffer(CLAMP_STRENGTH_ATTR[_HP_ENUM], torch.ones(num_rows, k))
        self.register_buffer(
            CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM], torch.zeros(1, dtype=torch.bool)
        )


def _make_manager(max_configs: int = MAX_CONFIGS, k: int = K) -> SteeringManager:
    return SteeringManager(max_steering_configs=max_configs, max_clamp_directions=k)


def _make_layers(mgr: SteeringManager, layer_indices=None, k: int = K):
    if layer_indices is None:
        layer_indices = [0, 1]
    num_rows = mgr.max_steering_configs + 3
    return {idx: FakeClampLayer(num_rows, HIDDEN_SIZE, k=k) for idx in layer_indices}


def _clamp_entry(axis: int, value: float, strength: float = 1.0) -> dict:
    vec = [0.0] * HIDDEN_SIZE
    vec[axis] = 2.0  # non-unit on purpose: the manager must normalize
    return {"vector": vec, "value": value, "strength": strength}


def _axis(i: int) -> torch.Tensor:
    v = torch.zeros(HIDDEN_SIZE)
    v[i] = 1.0
    return v


def _buffers(layer):
    return (
        getattr(layer, CLAMP_DIRS_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_BOUNDS_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_STRENGTH_ATTR[_HP_ENUM]),
        getattr(layer, CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM]),
    )


class TestClampRegistration:
    def test_register_stores_clamps(self):
        mgr = _make_manager()
        row = mgr.register_config(
            config_hash=7,
            vectors={},
            phase="decode",
            clamps={_HP: {0: [_clamp_entry(0, 5.0)]}},
        )
        assert row >= 3
        assert (7, "decode") in mgr.config_clamps
        assert mgr.has_any_clamps

    def test_release_purges_clamps(self):
        mgr = _make_manager()
        mgr.register_config(
            config_hash=7,
            vectors={},
            phase="decode",
            clamps={_HP: {0: [_clamp_entry(0, 5.0)]}},
        )
        mgr.release_config(7, "decode")
        assert (7, "decode") not in mgr.config_clamps
        assert not mgr.has_any_clamps

    def test_refcount_hit_keeps_single_payload(self):
        mgr = _make_manager()
        clamps = {_HP: {0: [_clamp_entry(0, 5.0)]}}
        r1 = mgr.register_config(7, {}, phase="decode", clamps=clamps)
        r2 = mgr.register_config(7, {}, phase="decode", clamps=clamps)
        assert r1 == r2
        mgr.release_config(7, "decode")
        assert (7, "decode") in mgr.config_clamps
        mgr.release_config(7, "decode")
        assert (7, "decode") not in mgr.config_clamps

    def test_register_too_many_directions_raises(self):
        mgr = _make_manager(k=2)
        with pytest.raises(ValueError, match="max_clamp_directions"):
            mgr.register_config(
                config_hash=7,
                vectors={},
                phase="decode",
                clamps={_HP: {0: [_clamp_entry(0, 1.0)] * 3}},
            )

    def test_register_clamps_when_disabled_raises(self):
        mgr = SteeringManager(max_steering_configs=4, max_clamp_directions=0)
        with pytest.raises(ValueError, match="max_clamp_directions"):
            mgr.register_config(
                config_hash=7,
                vectors={},
                phase="decode",
                clamps={_HP: {0: [_clamp_entry(0, 1.0)]}},
            )

    def test_locally_owned_layers_filter(self):
        mgr = _make_manager()
        mgr.register_config(
            config_hash=7,
            vectors={},
            phase="decode",
            clamps={_HP: {0: [_clamp_entry(0, 1.0)], 5: [_clamp_entry(1, 2.0)]}},
            locally_owned_layers=frozenset({5}),
        )
        stored = mgr.config_clamps[(7, "decode")]
        assert 0 not in stored.get(_HP, {})
        assert 5 in stored[_HP]


class TestClampPopulate:
    def test_per_request_row_written(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        row = mgr.register_config(
            config_hash=7,
            vectors={},
            phase="decode",
            clamps={_HP: {0: [_clamp_entry(0, 5.0, strength=0.5)]}},
        )
        mgr.populate_steering_tables(layers)
        dirs, bounds, strength, active = _buffers(layers[0])
        assert bool(active.item())
        torch.testing.assert_close(dirs[row, 0], _axis(0))  # unit-normalized
        assert bounds[row, 0, 0].item() == 5.0
        assert bounds[row, 0, 1].item() == 5.0
        assert strength[row, 0].item() == 0.5
        # Unused K slots at the row keep no-op defaults.
        torch.testing.assert_close(dirs[row, 1:], torch.zeros(K - 1, HIDDEN_SIZE))
        assert bounds[row, 1, 0].item() == -INF
        assert bounds[row, 1, 1].item() == INF
        # Sentinel row untouched.
        torch.testing.assert_close(dirs[0], torch.zeros(K, HIDDEN_SIZE))

    def test_flag_false_when_no_clamps(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.register_config(7, {_HP: {0: [1.0] * HIDDEN_SIZE}}, phase="decode")
        mgr.populate_steering_tables(layers)
        _, _, _, active = _buffers(layers[0])
        assert not bool(active.item())
        # Steering itself is active.
        assert bool(getattr(layers[0], HOOK_POINT_ANY_ACTIVE_ATTR[_HP_ENUM]).item())

    def test_flag_clears_after_release(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.register_config(
            7, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.populate_steering_tables(layers)
        _, _, _, active = _buffers(layers[0])
        assert bool(active.item())
        mgr.release_config(7, "decode")
        mgr.populate_steering_tables(layers)
        assert not bool(active.item())

    def test_global_clamps_reach_rows_1_and_2(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(1, 3.0)], phase="base")
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, active = _buffers(layers[0])
        assert bool(active.item())
        for row in (1, 2):
            torch.testing.assert_close(dirs[row, 0], _axis(1))
            assert bounds[row, 0, 0].item() == 3.0

    def test_phase_specific_global_clamps(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 1.0)], phase="prefill")
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(1, 2.0)], phase="decode")
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, _ = _buffers(layers[0])
        torch.testing.assert_close(dirs[1, 0], _axis(0))
        assert bounds[1, 0, 0].item() == 1.0
        torch.testing.assert_close(dirs[2, 0], _axis(1))
        assert bounds[2, 0, 0].item() == 2.0

    def test_config_row_concats_global_then_per_request(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 1.0)], phase="base")
        row = mgr.register_config(
            7, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(1, 2.0)]}}
        )
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, _ = _buffers(layers[0])
        torch.testing.assert_close(dirs[row, 0], _axis(0))  # global first
        assert bounds[row, 0, 0].item() == 1.0
        torch.testing.assert_close(dirs[row, 1], _axis(1))  # per-request after
        assert bounds[row, 1, 0].item() == 2.0

    def test_populate_overflow_names_site(self):
        mgr = _make_manager(k=2)
        layers = _make_layers(mgr, [0], k=2)
        mgr.update_global_clamps(
            _HP, 0, [_clamp_entry(0, 1.0), _clamp_entry(1, 1.0)], phase="base"
        )
        mgr.register_config(
            7, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(2, 2.0)]}}
        )
        with pytest.raises(ValueError, match=_HP):
            mgr.populate_steering_tables(layers)

    def test_stale_row_rewritten_on_reassignment(self):
        """A released row's clamp content must be overwritten when the row
        is reassigned to a clamp-free config."""
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        row1 = mgr.register_config(
            7, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.populate_steering_tables(layers)
        mgr.release_config(7, "decode")
        row2 = mgr.register_config(8, {_HP: {0: [1.0] * HIDDEN_SIZE}}, phase="decode")
        assert row1 == row2  # freed row reused
        # A second clamp elsewhere keeps the site active so rows are built.
        mgr.register_config(
            9, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(1, 2.0)]}}
        )
        mgr.populate_steering_tables(layers)
        dirs, _, _, active = _buffers(layers[0])
        assert bool(active.item())
        torch.testing.assert_close(dirs[row2], torch.zeros(K, HIDDEN_SIZE))

    def test_dynamic_rows_get_global_decode_clamps(self):
        mgr = SteeringManager(
            max_steering_configs=MAX_CONFIGS,
            max_dynamic_steering_configs=2,
            max_clamp_directions=K,
        )
        num_rows = MAX_CONFIGS + 2 + 3
        layers = {0: FakeClampLayer(num_rows, HIDDEN_SIZE)}
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 4.0)], phase="decode")
        _, dyn_row = mgr.register_dynamic_config({_HP: {0: [1.0] * HIDDEN_SIZE}})
        mgr.populate_steering_tables(layers)
        dirs, bounds, _, _ = _buffers(layers[0])
        torch.testing.assert_close(dirs[dyn_row, 0], _axis(0))
        assert bounds[dyn_row, 0, 0].item() == 4.0

    def test_clear_global_clamps(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 1.0)], phase="base")
        mgr.populate_steering_tables(layers)
        mgr.clear_global_clamps()
        assert not mgr.has_global_clamps
        mgr.populate_steering_tables(layers)
        _, _, _, active = _buffers(layers[0])
        assert not bool(active.item())

    def test_layer_without_clamp_buffers_skipped(self):
        """Layers lacking clamp buffers (e.g. K resolved to 0 at build, or
        a vector-only fake) must not crash populate."""

        class VectorOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    _TABLE_ATTR, torch.zeros(MAX_CONFIGS + 3, HIDDEN_SIZE)
                )

        mgr = _make_manager()
        layers = {0: VectorOnlyLayer()}
        mgr.register_config(
            7, {}, phase="decode", clamps={_HP: {0: [_clamp_entry(0, 5.0)]}}
        )
        mgr.populate_steering_tables(layers)  # no crash

    def test_one_sided_bounds_roundtrip(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        entry = {"vector": [2.0] + [0.0] * (HIDDEN_SIZE - 1), "max": 4.0}
        row = mgr.register_config(7, {}, phase="decode", clamps={_HP: {0: [entry]}})
        mgr.populate_steering_tables(layers)
        _, bounds, _, _ = _buffers(layers[0])
        assert bounds[row, 0, 0].item() == -INF
        assert bounds[row, 0, 1].item() == 4.0

    def test_update_global_clamps_marks_content_dirty(self):
        mgr = _make_manager()
        layers = _make_layers(mgr, [0])
        mgr.populate_steering_tables(layers)
        assert not mgr._tables_dirty
        mgr.update_global_clamps(_HP, 0, [_clamp_entry(0, 1.0)], phase="base")
        assert mgr._tables_dirty


class TestClampDirectionNormalization:
    def test_directions_unit_normalized_in_payload(self):
        mgr = _make_manager()
        mgr.register_config(
            7,
            {},
            phase="decode",
            clamps={_HP: {0: [_clamp_entry(0, 5.0)]}},
        )
        payload = mgr.config_clamps[(7, "decode")][_HP][0]
        norm = float(payload.dirs[0].norm())
        assert math.isclose(norm, 1.0, rel_tol=1e-6)
