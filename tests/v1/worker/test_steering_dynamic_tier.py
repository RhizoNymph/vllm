# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic additive tier (Phase 1b, populate-folding).

The dynamic tier is a decode-only global steering contribution that is
folded ADDITIVELY into the decode-effective vector at populate time, so
dynamic global steering composes with operator-set (``/v1/steering/set``)
decode steering instead of overwriting ``global_decode_vectors``. It must
reach every decode-derived row (row 2, decode per-request rows,
dynamic-override rows) and NO prefill row (decode-only cache safety, §7).
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


def _mgr() -> SteeringManager:
    return SteeringManager(
        max_steering_configs=MAX_STATIC,
        device=None,
        max_dynamic_steering_configs=MAX_DYNAMIC,
    )


def _full(value: float) -> torch.Tensor:
    return torch.full((HIDDEN,), value)


def _table(layers, idx=0):
    return layers[idx].steering_table_post_mlp


# ---------------------------------------------------------------------------
# Manager: tier store mechanics
# ---------------------------------------------------------------------------


def test_update_and_clear_dynamic_tier():
    mgr = _mgr()
    assert not mgr.has_dynamic_tier
    mgr.update_dynamic_tier(_HP, 0, _full(3.0))
    assert mgr.has_dynamic_tier
    assert mgr.dynamic_tier_vectors[_HP][0].sum().item() == 24.0
    assert mgr._tables_dirty

    mgr._tables_dirty = False
    mgr.clear_dynamic_tier()
    assert not mgr.has_dynamic_tier
    assert mgr._tables_dirty


def test_update_dynamic_tier_respects_locally_owned_layers():
    mgr = _mgr()
    mgr.update_dynamic_tier(_HP, 5, _full(1.0), locally_owned_layers=frozenset({0, 1}))
    assert not mgr.has_dynamic_tier  # layer 5 not owned -> no-op


# ---------------------------------------------------------------------------
# Populate: tier folds into decode-effective rows only
# ---------------------------------------------------------------------------


def test_tier_folds_into_row2_decode_not_row1_prefill():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_dynamic_tier(_HP, 0, _full(5.0))
    mgr.populate_steering_tables(layers)
    table = _table(layers)
    assert torch.all(table[2] == 5.0)  # row 2 (decode) gets the tier
    assert torch.all(table[1] == 0.0)  # row 1 (prefill) does NOT
    assert bool(layers[0].steering_table_post_mlp_any_active.item())


def test_tier_composes_additively_with_operator_decode():
    """The core no-clobber property: operator decode + dynamic tier."""
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(2.0), phase="decode")  # operator
    mgr.update_dynamic_tier(_HP, 0, _full(5.0))  # dynamic
    mgr.populate_steering_tables(layers)
    # Row 2 = operator decode (2) + tier (5); the operator vector is NOT
    # clobbered.
    assert torch.all(_table(layers)[2] == 7.0)
    # And the operator store is untouched by the tier write.
    assert torch.all(mgr.global_decode_vectors[_HP][0] == 2.0)
    assert torch.all(mgr.dynamic_tier_vectors[_HP][0] == 5.0)


def test_tier_excluded_from_prefill_rows_but_in_decode_rows():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(1.0), phase="base")
    mgr.update_dynamic_tier(_HP, 0, _full(4.0))
    pre_row = mgr.register_config(
        config_hash=11, vectors={_HP: {0: _full(2.0).tolist()}}, phase="prefill"
    )
    dec_row = mgr.register_config(
        config_hash=22, vectors={_HP: {0: _full(3.0).tolist()}}, phase="decode"
    )
    mgr.populate_steering_tables(layers)
    table = _table(layers)
    # Prefill config row = base(1) + per_req(2); NO tier.
    assert torch.all(table[pre_row] == 3.0)
    # Decode config row = base(1) + tier(4) + per_req(3).
    assert torch.all(table[dec_row] == 8.0)


def test_tier_reaches_dynamic_override_row():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_global_vectors(_HP, 0, _full(2.0), phase="decode")
    mgr.update_dynamic_tier(_HP, 0, _full(4.0))
    _dyn_id, row = mgr.register_dynamic_config(
        {_HP: {0: np.full(HIDDEN, 5.0, np.float32)}}
    )
    mgr.populate_steering_tables(layers)
    # Override row = global_decode_effective (operator 2 + tier 4) + override 5.
    assert torch.all(_table(layers)[row] == 11.0)


def test_clear_tier_restores_rows():
    mgr = _mgr()
    layers = {0: _Layer()}
    mgr.update_dynamic_tier(_HP, 0, _full(6.0))
    mgr.populate_steering_tables(layers)
    assert torch.all(_table(layers)[2] == 6.0)
    mgr.clear_dynamic_tier()
    mgr.populate_steering_tables(layers)
    assert torch.all(_table(layers)[2] == 0.0)


# ---------------------------------------------------------------------------
# Apply path: decode updates route to the tier, not global_decode
# ---------------------------------------------------------------------------


def test_apply_decode_update_routes_to_tier_and_composes():
    mgr = _mgr()
    layers = {0: _Layer()}
    # Operator sets a global decode vector directly (the /v1/steering/set
    # path, bypassing the action queue).
    mgr.update_global_vectors(_HP, 0, _full(2.0), phase="decode")

    update = SteeringVectorUpdate(
        vectors={_HP: {0: np.full(HIDDEN, 5.0, np.float32)}}, phase="decode"
    )
    applied, rejected = apply_steering_updates([update], mgr, layers)
    assert (applied, rejected) == (1, 0)
    # The dynamic update landed on the tier, NOT on global_decode_vectors.
    assert mgr.has_dynamic_tier
    assert torch.all(mgr.dynamic_tier_vectors[_HP][0] == 5.0)
    assert torch.all(mgr.global_decode_vectors[_HP][0] == 2.0)  # operator intact

    mgr.populate_steering_tables(layers)
    assert torch.all(_table(layers)[2] == 7.0)  # composed, not clobbered


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
