# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per-row steering scale (§5.3) — SteeringManager side.

The scale is a runtime "how much" knob: a per-row float32 multiplier the
kernel applies to the gathered steering vector (`out = hidden +
table[row] * scales[row]`), settable without re-uploading vectors. The
manager keys scales by logical owner (so they survive row reuse), writes
them into each layer's `steering_scales` buffer in populate, and supports
a cheap scales-only write (`populate_steering_scales`) gated by
`_scales_dirty`. Prefill rows are pinned to 1.0 (cache safety, §7).
CPU-only, no engine.
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN = 8
MAX_STATIC = 4
MAX_DYNAMIC = 2
NUM_ROWS = MAX_STATIC + MAX_DYNAMIC + 3
_HP = "post_mlp"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("steering_table_post_mlp", torch.zeros(NUM_ROWS, HIDDEN))
        self.register_buffer(
            "steering_table_post_mlp_any_active", torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer("steering_scales", torch.ones(NUM_ROWS))


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _vec(value: float) -> dict:
    return {_HP: {0: np.full(HIDDEN, value, dtype=np.float32)}}


def _scales(layers, idx=0):
    return layers[idx].steering_scales


# ---------------------------------------------------------------------------
# Defaults + global / dynamic scales
# ---------------------------------------------------------------------------


def test_default_scales_are_one_after_populate():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, torch.full((HIDDEN,), 2.0), phase="decode")
    mgr.populate_steering_tables(layers)
    assert torch.all(_scales(layers) == 1.0)


def test_global_decode_scale_written_to_row2():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.set_global_scale("decode", 0.5)
    mgr.populate_steering_tables(layers)
    s = _scales(layers)
    assert s[2].item() == 0.5  # row 2 = global decode
    assert s[0].item() == 1.0 and s[1].item() == 1.0  # sentinel + prefill


def test_prefill_scale_pinned_to_one():
    """A prefill scale is stored but forced to 1.0 at populate (cache safety)."""
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.set_global_scale("prefill", 9.0)
    cfg_row = mgr.register_config(config_hash=7, vectors=_vec(1.0), phase="prefill")
    mgr.set_row_scale(7, "prefill", 9.0)
    mgr.populate_steering_tables(layers)
    s = _scales(layers)
    assert s[1].item() == 1.0  # row 1 prefill never scaled
    assert s[cfg_row].item() == 1.0  # prefill config row never scaled


def test_config_and_dynamic_scales_target_their_rows():
    mgr = _mgr()
    layers = {0: _Layer()}
    cfg_row = mgr.register_config(config_hash=11, vectors=_vec(1.0), phase="decode")
    dyn_id, dyn_row = mgr.register_dynamic_config(_vec(1.0))
    mgr.set_row_scale(11, "decode", 2.0)
    mgr.set_dynamic_scale(dyn_id, 3.0)
    mgr.populate_steering_tables(layers)
    s = _scales(layers)
    assert s[cfg_row].item() == 2.0
    assert s[dyn_row].item() == 3.0


# ---------------------------------------------------------------------------
# Cheap path + row reuse + clear
# ---------------------------------------------------------------------------


def test_scales_only_cheap_path_does_not_touch_table():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, torch.full((HIDDEN,), 4.0), phase="decode")
    mgr.populate_steering_tables(layers)
    table_before = layers[0].steering_table_post_mlp.clone()

    # A scale-only change sets _scales_dirty (not _tables_dirty).
    mgr.set_global_scale("decode", 0.25)
    assert mgr._scales_dirty and not mgr._tables_dirty
    mgr.populate_steering_scales(layers)

    assert _scales(layers)[2].item() == 0.25
    # The table content is untouched (vector not re-uploaded).
    assert torch.equal(layers[0].steering_table_post_mlp, table_before)
    assert not mgr._scales_dirty


def test_scale_follows_owner_across_row_reuse():
    mgr = _mgr()
    layers = {0: _Layer()}
    id1, row1 = mgr.register_dynamic_config(_vec(1.0))
    mgr.set_dynamic_scale(id1, 5.0)
    mgr.populate_steering_tables(layers)
    assert _scales(layers)[row1].item() == 5.0

    # Release id1, register id2 reusing the row: the stale scale must NOT
    # carry over (id2 has no scale → default 1.0).
    mgr.release_dynamic_config(id1)
    id2, row2 = mgr.register_dynamic_config(_vec(1.0))
    assert row2 == row1
    mgr.populate_steering_tables(layers)
    assert _scales(layers)[row2].item() == 1.0


def test_clear_scales_resets_to_one():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.set_global_scale("decode", 0.5)
    mgr.populate_steering_tables(layers)
    assert _scales(layers)[2].item() == 0.5
    mgr.clear_scales()
    mgr.populate_steering_scales(layers)
    assert torch.all(_scales(layers) == 1.0)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
