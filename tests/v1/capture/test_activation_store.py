# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the capture ActivationStore (roadmap step A, core).

The store is a bounded, LRU, drop-on-eviction CPU-RAM cache of captured
residual rows keyed by content. These tests exercise the data structure in
isolation with tiny float32 rows (16 bytes each) so byte budgets are exact;
the scheduler / capture-manager wiring is tested separately.
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.capture.activation_store import (
    ActivationStore,
    activation_key,
    get_active_activation_store,
    set_active_activation_store,
)
from vllm.v1.capture.manager import (
    CaptureManager,
    _DispatchPacket,
    _RequestCaptureState,
)
from vllm.v1.capture.plan import CapturePositionEntry

ROW_BYTES = 4 * 4  # torch.float32, hidden=4


def _row(val: float, hidden: int = 4) -> torch.Tensor:
    return torch.full((hidden,), float(val), dtype=torch.float32)


def _key(i: int):
    return (bytes([i]), 0, 0, "post_mlp")


class TestBasics:
    def test_put_get_roundtrip(self) -> None:
        store = ActivationStore(max_bytes=ROW_BYTES)
        store.put(_key(0), _row(1.0))
        got = store.get(_key(0))
        assert got is not None
        assert torch.equal(got, _row(1.0))

    def test_miss_returns_none_and_counts(self) -> None:
        store = ActivationStore(max_bytes=ROW_BYTES)
        assert store.get(_key(9)) is None
        assert _key(9) not in store
        assert store.stats().misses == 1
        assert store.stats().hits == 0

    def test_contains_and_len(self) -> None:
        store = ActivationStore(max_bytes=4 * ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0))
        assert _key(0) in store
        assert len(store) == 2

    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ActivationStore(max_bytes=-1)

    def test_cuda_row_rejected(self) -> None:
        store = ActivationStore(max_bytes=ROW_BYTES)
        cpu = _row(1.0)

        class _FakeCuda:
            is_cuda = True

            def numel(self) -> int:  # pragma: no cover - not reached
                return cpu.numel()

            def element_size(self) -> int:  # pragma: no cover - not reached
                return cpu.element_size()

        with pytest.raises(ValueError, match="CPU tensors"):
            store.put(_key(0), _FakeCuda())  # type: ignore[arg-type]


class TestEviction:
    def test_lru_evicts_oldest(self) -> None:
        store = ActivationStore(max_bytes=2 * ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0))
        store.put(_key(2), _row(2.0))  # over budget -> evict key0
        assert _key(0) not in store
        assert _key(1) in store
        assert _key(2) in store
        stats = store.stats()
        assert stats.evictions == 1
        assert stats.entries == 2
        assert stats.resident_bytes == 2 * ROW_BYTES

    def test_get_touches_lru_order(self) -> None:
        store = ActivationStore(max_bytes=2 * ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0))
        # Touch key0 so key1 becomes the least-recently-used.
        assert store.get(_key(0)) is not None
        store.put(_key(2), _row(2.0))  # evicts key1, not key0
        assert _key(0) in store
        assert _key(1) not in store
        assert _key(2) in store

    def test_replace_same_key_no_spurious_eviction(self) -> None:
        store = ActivationStore(max_bytes=2 * ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0))
        store.put(_key(0), _row(5.0))  # replace, not grow
        stats = store.stats()
        assert stats.evictions == 0
        assert stats.entries == 2
        assert stats.resident_bytes == 2 * ROW_BYTES
        got = store.get(_key(0))
        assert got is not None and torch.equal(got, _row(5.0))

    def test_row_larger_than_budget_is_skipped(self) -> None:
        store = ActivationStore(max_bytes=ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0, hidden=8))  # 32 bytes > 16 budget
        assert _key(1) not in store
        assert _key(0) in store  # untouched
        stats = store.stats()
        assert stats.skipped_too_large == 1
        assert stats.resident_bytes == ROW_BYTES

    def test_oversize_replacement_drops_stale_entry(self) -> None:
        # A too-large put under an existing key must remove the stale row so
        # a later read cannot return a wrong residual for that content key.
        store = ActivationStore(max_bytes=ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(0), _row(9.0, hidden=8))  # 32 bytes > 16 budget
        assert _key(0) not in store
        stats = store.stats()
        assert stats.skipped_too_large == 1
        assert stats.resident_bytes == 0


class TestInvalidation:
    def test_invalidate_all_clears(self) -> None:
        store = ActivationStore(max_bytes=4 * ROW_BYTES)
        store.put(_key(0), _row(0.0))
        store.put(_key(1), _row(1.0))
        store.invalidate_all()
        assert len(store) == 0
        stats = store.stats()
        assert stats.resident_bytes == 0
        assert stats.invalidations == 1
        assert _key(0) not in store


class TestGlobalAccessor:
    def test_set_get_clear(self) -> None:
        assert get_active_activation_store() is None
        store = ActivationStore(max_bytes=ROW_BYTES)
        set_active_activation_store(store)
        try:
            assert get_active_activation_store() is store
        finally:
            set_active_activation_store(None)
        assert get_active_activation_store() is None


class TestActivationKey:
    def test_basic_key(self) -> None:
        # hash_block_size=2: position 3 -> block 1, offset 1.
        assert activation_key([b"blk0", b"blk1"], 2, 3, 5, "post_mlp") == (
            b"blk1",
            1,
            5,
            "post_mlp",
        )

    def test_position_zero(self) -> None:
        assert activation_key([b"a"], 2, 0, 0, "post_attn") == (b"a", 0, 0, "post_attn")

    def test_partial_block_unhashed_is_none(self) -> None:
        # position 4 -> block 2, but only 2 blocks are hashed.
        assert activation_key([b"a", b"b"], 2, 4, 0, "post_mlp") is None

    def test_negative_position_is_none(self) -> None:
        assert activation_key([b"a"], 2, -1, 0, "h") is None

    def test_zero_block_size_is_none(self) -> None:
        assert activation_key([b"a"], 0, 0, 0, "h") is None


def _manager() -> CaptureManager:
    return CaptureManager(
        consumers=(),
        consumer_specs=(),
        num_hidden_layers=4,
        hidden_size=2,
        model_dtype=torch.float32,
    )


def _state(num_prompt_tokens: int, block_hashes, hash_block_size: int):
    return _RequestCaptureState(
        req_id="r",
        consumer_specs={},
        position_kind={},
        static_positions={},
        num_prompt_tokens=num_prompt_tokens,
        block_hashes=block_hashes,
        hash_block_size=hash_block_size,
    )


def _entries(positions, layer=1, hook="post_mlp"):
    return [
        CapturePositionEntry(
            request_id="r",
            layer=layer,
            hook=hook,
            logical_pos=p,
            scratch_row=i,
            step_index=0,
            consumer_mask=1,
        )
        for i, p in enumerate(positions)
    ]


def _packet(entries, view, layer=1, hook="post_mlp"):
    return _DispatchPacket(
        entries=entries,
        scratch_pinned={(layer, hook): (None, view)},
        cuda_event=None,
    )


class TestWriteThrough:
    def test_stores_prompt_rows_keyed_by_content(self) -> None:
        mgr = _manager()
        mgr._requests["r"] = _state(4, [b"blk0", b"blk1"], 2)
        # One row per prompt position 0..3, hidden=2.
        view = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        store = ActivationStore(max_bytes=10_000)
        set_active_activation_store(store)
        try:
            mgr._write_through_to_store(_packet(_entries([0, 1, 2, 3]), view))
        finally:
            set_active_activation_store(None)
            mgr.shutdown(timeout=2.0)
        # positions 0,1 -> blk0 off 0,1 ; positions 2,3 -> blk1 off 0,1.
        assert (b"blk0", 0, 1, "post_mlp") in store
        assert (b"blk1", 1, 1, "post_mlp") in store
        got = store.get((b"blk1", 1, 1, "post_mlp"))
        assert got is not None and torch.equal(got, view[3])

    def test_skips_generated_positions(self) -> None:
        mgr = _manager()
        # Only the first 2 positions are prompt; 2 and 3 are generated.
        mgr._requests["r"] = _state(2, [b"blk0", b"blk1"], 2)
        view = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        store = ActivationStore(max_bytes=10_000)
        set_active_activation_store(store)
        try:
            mgr._write_through_to_store(_packet(_entries([0, 1, 2, 3]), view))
        finally:
            set_active_activation_store(None)
            mgr.shutdown(timeout=2.0)
        assert (b"blk0", 0, 1, "post_mlp") in store
        # Position 2 (generated) must not be stored.
        assert (b"blk1", 0, 1, "post_mlp") not in store
        assert len(store) == 2

    def test_rows_are_cloned_off_buffer(self) -> None:
        mgr = _manager()
        mgr._requests["r"] = _state(2, [b"blk0"], 2)
        view = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        store = ActivationStore(max_bytes=10_000)
        set_active_activation_store(store)
        try:
            mgr._write_through_to_store(_packet(_entries([0, 1]), view))
        finally:
            set_active_activation_store(None)
            mgr.shutdown(timeout=2.0)
        # Mutating the (recycled) pinned buffer must not corrupt the store.
        before = store.get((b"blk0", 0, 1, "post_mlp")).clone()
        view.zero_()
        after = store.get((b"blk0", 0, 1, "post_mlp"))
        assert torch.equal(after, before)

    def test_noop_without_store(self) -> None:
        mgr = _manager()
        mgr._requests["r"] = _state(2, [b"blk0"], 2)
        view = torch.zeros(2, 2, dtype=torch.float32)
        set_active_activation_store(None)
        try:
            mgr._write_through_to_store(_packet(_entries([0, 1]), view))  # no error
        finally:
            mgr.shutdown(timeout=2.0)

    def test_noop_when_block_hashes_absent(self) -> None:
        mgr = _manager()
        mgr._requests["r"] = _state(2, None, 0)  # offline / no hashes
        view = torch.zeros(2, 2, dtype=torch.float32)
        store = ActivationStore(max_bytes=10_000)
        set_active_activation_store(store)
        try:
            mgr._write_through_to_store(_packet(_entries([0, 1]), view))
        finally:
            set_active_activation_store(None)
            mgr.shutdown(timeout=2.0)
        assert len(store) == 0
