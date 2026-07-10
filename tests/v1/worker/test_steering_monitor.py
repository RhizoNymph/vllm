# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-side tests for the in-graph monitor (Phase 2, §8).

The monitor config lives in the manager keyed by ``(hook, layer)``; at
populate time the manager writes it into the per-layer
``*_monitor_probe`` / ``*_monitor_params`` / ``*_monitor_active`` buffers.
A configured site flips ``active`` True and loads the probe + [threshold,
sharpness]; an unconfigured site is deactivated so the monitor op there is
a no-op (leaving the runner's flat decode gate intact). CPU-only, no
engine. See docs/design/dynamic_steering.md §8.
"""

import torch
import torch.nn as nn

from vllm.v1.worker.steering_manager import SteeringManager

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
        self.register_buffer("steering_table_post_block_dynvec", torch.zeros(HIDDEN))
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


def _probe(layers, idx=0):
    return layers[idx].steering_table_post_block_monitor_probe


def _params(layers, idx=0):
    return layers[idx].steering_table_post_block_monitor_params


def _active(layers, idx=0):
    return bool(layers[idx].steering_table_post_block_monitor_active.item())


# ---------------------------------------------------------------------------
# Manager: monitor store
# ---------------------------------------------------------------------------


def test_set_and_has_monitor():
    mgr = _mgr()
    assert not mgr.has_monitor
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), threshold=0.5, sharpness=2.0)
    assert mgr.has_monitor
    assert mgr._tables_dirty


def test_clear_monitor_all():
    mgr = _mgr()
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.0, 1.0)
    mgr._tables_dirty = False
    mgr.clear_monitor()
    assert not mgr.has_monitor
    assert mgr._tables_dirty


def test_clear_monitor_single_site():
    mgr = _mgr()
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.0, 1.0)
    mgr.set_monitor(_HP, 1, torch.ones(HIDDEN), 0.0, 1.0)
    mgr.clear_monitor(_HP, 0)
    assert mgr.has_monitor  # layer 1 still configured
    assert 0 not in mgr.monitor_configs[_HP]
    assert 1 in mgr.monitor_configs[_HP]


def test_set_monitor_respects_locally_owned_layers():
    mgr = _mgr()
    mgr.set_monitor(
        _HP, 5, torch.ones(HIDDEN), 0.0, 1.0, locally_owned_layers=frozenset({0, 1})
    )
    assert not mgr.has_monitor  # layer 5 not owned -> no-op


# ---------------------------------------------------------------------------
# Populate: config lands in the per-layer monitor buffers
# ---------------------------------------------------------------------------


def test_populate_writes_probe_params_active():
    mgr = _mgr()
    layers = {0: _Layer()}
    probe = torch.arange(HIDDEN, dtype=torch.float32)
    mgr.set_monitor(_HP, 0, probe, threshold=1.5, sharpness=3.0)
    mgr.populate_steering_tables(layers)
    assert _active(layers)
    torch.testing.assert_close(_probe(layers), probe)
    torch.testing.assert_close(
        _params(layers), torch.tensor([1.5, 3.0, 0.0], dtype=torch.float32)
    )


def test_populate_writes_gate_rows_flag():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.5, 2.0, gate_rows=True)
    mgr.populate_steering_tables(layers)
    # params[2] == 1.0 ⇒ the monitor op will also gate per-request rows.
    assert _params(layers)[2].item() == 1.0


def test_populate_deactivates_unconfigured_site():
    mgr = _mgr()
    layers = {0: _Layer()}
    # Pre-activate, then clear, then populate -> must be deactivated.
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.0, 1.0)
    mgr.populate_steering_tables(layers)
    assert _active(layers)
    mgr.clear_monitor()
    mgr.populate_steering_tables(layers)
    assert not _active(layers)


def test_monitor_independent_of_table_rows():
    """A monitor-only config still populates (the dynvec/tier path drives
    the actual steering; the monitor only writes the gate)."""
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.set_monitor(_HP, 0, torch.ones(HIDDEN), 0.0, 5.0)
    mgr.populate_steering_tables(layers)
    # Monitor active; table rows untouched (all zero — no operator steering).
    assert _active(layers)
    assert torch.all(layers[0].steering_table_post_block == 0.0)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
