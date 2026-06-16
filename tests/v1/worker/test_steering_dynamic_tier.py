# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic additive tier (§5.4, dedicated-gather) — manager side.

The tier is a decode-only global steering contribution. As of the
dedicated-gather design it is NOT folded into table rows; instead the
manager writes it into a per-(layer, hook) ``steering_table_{hook}_dynvec``
buffer, and the kernel adds ``dynamic_vec * token_scales[token]`` on top
of the row gather (the per-token gate is the runner's job, tested
elsewhere). So at the manager level: a tier vector lands in the dynvec
buffer, table rows (incl. row 2 / decode config / override rows) carry
ONLY operator + per-request content, and the two compose at the kernel.
CPU-only, no engine. See docs/design/dynamic_steering.md §5.4.
"""

import numpy as np
import torch
import torch.nn as nn

from vllm.v1.worker.steering_action_queue import (
    SteeringVectorUpdate,
    apply_steering_updates,
)
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
        self.register_buffer("steering_table_post_mlp_dynvec", torch.zeros(HIDDEN))


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _full(value: float) -> torch.Tensor:
    return torch.full((HIDDEN,), value)


def _dynvec(layers, idx=0):
    return layers[idx].steering_table_post_mlp_dynvec


def _table(layers, idx=0):
    return layers[idx].steering_table_post_mlp


# ---------------------------------------------------------------------------
# Manager: tier store + gain
# ---------------------------------------------------------------------------


def test_update_and_clear_dynamic_tier():
    mgr = _mgr()
    assert not mgr.has_dynamic_tier
    mgr.update_dynamic_tier(_HP, 0, _full(3.0))
    assert mgr.has_dynamic_tier
    assert mgr._tables_dirty

    mgr._tables_dirty = False
    mgr.clear_dynamic_tier()
    assert not mgr.has_dynamic_tier
    assert mgr._tables_dirty


def test_set_dynamic_tier_gain():
    mgr = _mgr()
    assert mgr.dynamic_tier_gain == 1.0
    mgr.set_dynamic_tier_gain(4.0)
    assert mgr.dynamic_tier_gain == 4.0


def test_update_dynamic_tier_respects_locally_owned_layers():
    mgr = _mgr()
    mgr.update_dynamic_tier(_HP, 5, _full(1.0), locally_owned_layers=frozenset({0, 1}))
    assert not mgr.has_dynamic_tier  # layer 5 not owned -> no-op


# ---------------------------------------------------------------------------
# Populate: tier lands in the dedicated buffer, NOT in rows
# ---------------------------------------------------------------------------


def test_tier_lands_in_dynvec_buffer_not_rows():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_dynamic_tier(_HP, 0, _full(5.0))
    mgr.populate_steering_tables(layers)
    assert torch.all(_dynvec(layers) == 5.0)  # dedicated buffer holds the tier
    assert torch.all(_table(layers) == 0.0)  # NO table row carries it
    # Tier-only still marks the layer active so the kernel runs.
    assert bool(layers[0].steering_table_post_mlp_any_active.item())


def test_tier_does_not_touch_operator_decode_row():
    """Operator decode (row 2) and the tier are independent now."""
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(2.0), phase="decode")  # operator
    mgr.update_dynamic_tier(_HP, 0, _full(5.0))  # dynamic
    mgr.populate_steering_tables(layers)
    assert torch.all(_table(layers)[2] == 2.0)  # row 2 = operator only
    assert torch.all(_dynvec(layers) == 5.0)  # tier in dedicated buffer
    # Source state untouched on both sides.
    assert torch.all(mgr.global_decode_vectors[_HP][0] == 2.0)
    assert torch.all(mgr.dynamic_tier_vectors[_HP][0] == 5.0)


def test_tier_absent_from_decode_config_and_override_rows():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(1.0), phase="decode")
    mgr.update_dynamic_tier(_HP, 0, _full(4.0))
    dec_row = mgr.register_config(
        config_hash=22, vectors={_HP: {0: _full(3.0).tolist()}}, phase="decode"
    )
    _dyn_id, ovr_row = mgr.register_dynamic_config(
        {_HP: {0: np.full(HIDDEN, 5.0, np.float32)}}
    )
    mgr.populate_steering_tables(layers)
    table = _table(layers)
    # Decode config row = operator decode (1) + per_req (3); NO tier.
    assert torch.all(table[dec_row] == 4.0)
    # Override row = operator decode (1) + override (5); NO tier.
    assert torch.all(table[ovr_row] == 6.0)
    # Tier lives only in the dedicated buffer.
    assert torch.all(_dynvec(layers) == 4.0)


def test_clear_tier_zeros_dynvec_buffer():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_dynamic_tier(_HP, 0, _full(6.0))
    mgr.populate_steering_tables(layers)
    assert torch.all(_dynvec(layers) == 6.0)
    mgr.clear_dynamic_tier()
    mgr.populate_steering_tables(layers)
    assert torch.all(_dynvec(layers) == 0.0)


# ---------------------------------------------------------------------------
# Apply path: decode updates route to the tier buffer
# ---------------------------------------------------------------------------


def test_apply_decode_update_routes_to_tier_buffer():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(2.0), phase="decode")  # operator
    update = SteeringVectorUpdate(
        vectors={_HP: {0: np.full(HIDDEN, 5.0, np.float32)}}, phase="decode"
    )
    applied, rejected = apply_steering_updates([update], mgr, layers)
    assert (applied, rejected) == (1, 0)
    assert mgr.has_dynamic_tier
    assert torch.all(mgr.global_decode_vectors[_HP][0] == 2.0)  # operator intact

    mgr.populate_steering_tables(layers)
    assert torch.all(_table(layers)[2] == 2.0)  # operator row unchanged
    assert torch.all(_dynvec(layers) == 5.0)  # tier in dedicated buffer


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
