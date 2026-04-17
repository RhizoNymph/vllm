# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``locally_owned_layers`` filter on SteeringManager.

Under pipeline parallelism, each worker physically owns only a contiguous
subset of decoder layers.  The ``SteeringManager`` still needs to allocate
rows for every config on every rank (so that row IDs in ``steering_index``
stay coherent across ranks — the distributed-steering determinism
contract), but it should not materialize per-request vector tensors for
layers it can't write.  These tests pin that behavior.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_TABLE_ATTR,
)
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN_SIZE = 8
MAX_CONFIGS = 4
_HP = DEFAULT_HOOK_POINT.value  # "post_mlp"
_TABLE_ATTR = HOOK_POINT_TABLE_ATTR[DEFAULT_HOOK_POINT]


class FakeSteerableLayer(nn.Module):
    """Minimal module with every hook-point ``steering_table_*`` buffer.

    ``populate_steering_tables`` checks ``hasattr`` on the first hook
    point (``pre_attn``) to gate early-return, so the fake must register
    every hook's buffer — not just the one we assert against — to
    exercise the populate path.
    """

    def __init__(self, num_rows: int, hidden_size: int):
        super().__init__()
        for attr in HOOK_POINT_TABLE_ATTR.values():
            self.register_buffer(
                attr,
                torch.zeros(num_rows, hidden_size),
            )


def _make_manager(
    max_configs: int = MAX_CONFIGS,
    device: "torch.device | None" = None,
) -> SteeringManager:
    return SteeringManager(max_steering_configs=max_configs, device=device)


def _make_layers(
    manager: SteeringManager,
    layer_indices: list[int],
    hidden_size: int = HIDDEN_SIZE,
) -> dict[int, nn.Module]:
    num_rows = manager.max_steering_configs + 3
    return {idx: FakeSteerableLayer(num_rows, hidden_size) for idx in layer_indices}


def _vec(value: float) -> list[float]:
    return [value] * HIDDEN_SIZE


class TestOwnershipFilter:
    """Filtering behavior when ``locally_owned_layers`` is provided."""

    def test_register_config_skips_non_local_layer_tensors(self):
        """Tensors are only materialized for locally-owned layers,
        but the row is still allocated."""
        mgr = _make_manager()
        vectors = {_HP: {5: _vec(1.0), 10: _vec(2.0), 20: _vec(3.0)}}
        row = mgr.register_config(
            config_hash=42,
            vectors=vectors,
            phase="prefill",
            locally_owned_layers=frozenset({5, 10}),
        )
        # Row allocation proceeds regardless of ownership.
        assert row >= 3
        stored = mgr.config_vectors[(42, "prefill")]
        assert _HP in stored
        assert set(stored[_HP].keys()) == {5, 10}
        # The non-local layer is not materialized.
        assert 20 not in stored[_HP]

    def test_register_config_none_owners_keeps_all_layers(self):
        """When ``locally_owned_layers`` is ``None`` (test default),
        no filtering — preserves existing behavior."""
        mgr = _make_manager()
        vectors = {_HP: {5: _vec(1.0), 10: _vec(2.0), 20: _vec(3.0)}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        stored = mgr.config_vectors[(42, "prefill")]
        assert set(stored[_HP].keys()) == {5, 10, 20}

    def test_register_config_row_allocation_is_ownership_independent(self):
        """Determinism contract: two managers with disjoint ownership
        sets must assign identical rows when given the same sequence of
        ``(config_hash, phase)`` pairs in the same order."""
        mgr_a = _make_manager()
        mgr_b = _make_manager()

        vectors = {_HP: {5: _vec(1.0), 25: _vec(2.0)}}
        sequence = [
            (100, "prefill"),
            (200, "decode"),
            (300, "prefill"),
        ]

        rows_a = [
            mgr_a.register_config(
                config_hash=h,
                vectors=vectors,
                phase=p,
                locally_owned_layers=frozenset(range(0, 16)),
            )
            for h, p in sequence
        ]
        rows_b = [
            mgr_b.register_config(
                config_hash=h,
                vectors=vectors,
                phase=p,
                locally_owned_layers=frozenset(range(16, 32)),
            )
            for h, p in sequence
        ]

        assert rows_a == rows_b, (
            "row allocation must be independent of which layers a rank owns; "
            "this invariant is the foundation of the distributed-steering "
            "determinism contract"
        )
        # Sanity: each manager only stored its own layers.
        assert set(mgr_a.config_vectors[(100, "prefill")][_HP].keys()) == {5}
        assert set(mgr_b.config_vectors[(100, "prefill")][_HP].keys()) == {25}

    def test_update_global_vectors_skips_non_local_layer(self):
        """Updates for layers the rank doesn't own are no-ops."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE)
        mgr.update_global_vectors(
            _HP,
            layer_idx=5,
            vector=vec,
            phase="base",
            locally_owned_layers=frozenset({10}),
        )
        # Layer 5 isn't owned, so no entry should exist.
        assert 5 not in mgr.global_base_vectors.get(_HP, {})

    def test_update_global_vectors_none_owners_keeps_all(self):
        """``locally_owned_layers=None`` preserves the unfiltered
        behavior relied on by existing tests and the manager's test-
        fixture usage."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE)
        mgr.update_global_vectors(_HP, layer_idx=5, vector=vec, phase="base")
        assert 5 in mgr.global_base_vectors[_HP]

    def test_populate_steering_tables_tolerates_partial_ownership(self):
        """After registering with partial ownership, populate writes
        only the owned-layer rows and does not raise on absent
        non-local entries."""
        mgr = _make_manager()
        owned = {5, 10}
        vectors = {_HP: {5: _vec(1.0), 10: _vec(2.0), 20: _vec(3.0)}}
        row = mgr.register_config(
            config_hash=42,
            vectors=vectors,
            phase="prefill",
            locally_owned_layers=frozenset(owned),
        )
        steerable = _make_layers(mgr, layer_indices=list(owned))
        # Should not raise — non-local layer 20 has no tensor and no
        # buffer on this rank, but the manager simply doesn't try to
        # write it.
        mgr.populate_steering_tables(steerable)

        # Owned-layer rows should now contain the per-request values.
        for layer_idx in owned:
            table = getattr(steerable[layer_idx], _TABLE_ATTR)
            row_content = table[row]
            expected = vectors[_HP][layer_idx][0]  # all entries equal
            assert torch.allclose(
                row_content, torch.full((HIDDEN_SIZE,), expected, dtype=table.dtype)
            )
