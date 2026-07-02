# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the run-id-keyed patch source store.

The store parks clean-run activations under a run handle for cheap reference
across a sweep. Unlike the content-addressed ActivationStore, it is
authoritative (misses are errors) and evicts whole runs (never partial).
"""

import torch

from vllm.v1.capture.source_store import (
    PatchSourceStore,
    get_active_patch_source_store,
    set_active_patch_source_store,
)


def _row(h: int, fill: float) -> torch.Tensor:
    return torch.full((h,), fill)


class TestPutGet:
    def test_put_then_get_row(self):
        store = PatchSourceStore(max_bytes=0)  # unbounded
        store.put_row("R1", 3, "post_block", 5, _row(8, 1.5), num_prompt_tokens=10)
        out = store.get_row("R1", 3, "post_block", 5)
        assert out is not None
        assert torch.allclose(out, _row(8, 1.5))

    def test_get_row_returns_clone(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 0, "post_block", 0, _row(4, 2.0), num_prompt_tokens=4)
        a = store.get_row("R1", 0, "post_block", 0)
        a += 100.0  # mutate the returned copy
        b = store.get_row("R1", 0, "post_block", 0)
        assert torch.allclose(b, _row(4, 2.0))  # store unaffected

    def test_get_rows_all_or_nothing(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 1, "post_block", 0, _row(4, 1.0), num_prompt_tokens=3)
        store.put_row("R1", 1, "post_block", 1, _row(4, 2.0), num_prompt_tokens=3)
        rows = store.get_rows("R1", [(1, "post_block", 0), (1, "post_block", 1)])
        assert rows is not None and len(rows) == 2
        assert torch.allclose(rows[0], _row(4, 1.0))
        assert torch.allclose(rows[1], _row(4, 2.0))
        # One missing site -> whole batch returns None.
        partial = store.get_rows("R1", [(1, "post_block", 0), (1, "post_block", 9)])
        assert partial is None

    def test_miss_on_unknown_run(self):
        store = PatchSourceStore(max_bytes=0)
        assert store.get_row("nope", 0, "post_block", 0) is None
        assert store.get_rows("nope", [(0, "post_block", 0)]) is None
        assert not store.has_run("nope")


class TestEviction:
    def test_whole_run_lru_eviction(self):
        h = 16
        row_bytes = _row(h, 0.0).numel() * _row(h, 0.0).element_size()
        # Budget for ~1.5 rows -> a second run forces the first out wholesale.
        store = PatchSourceStore(max_bytes=int(row_bytes * 1.5))
        store.put_row("R1", 0, "post_block", 0, _row(h, 1.0), num_prompt_tokens=1)
        assert store.has_run("R1")
        store.put_row("R2", 0, "post_block", 0, _row(h, 2.0), num_prompt_tokens=1)
        # R1 evicted whole; R2 retained.
        assert not store.has_run("R1")
        assert store.has_run("R2")

    def test_touch_protects_from_eviction(self):
        h = 16
        row_bytes = _row(h, 0.0).numel() * _row(h, 0.0).element_size()
        store = PatchSourceStore(max_bytes=int(row_bytes * 2.5))
        store.put_row("R1", 0, "post_block", 0, _row(h, 1.0), num_prompt_tokens=1)
        store.put_row("R2", 0, "post_block", 0, _row(h, 2.0), num_prompt_tokens=1)
        # Touch R1 so it becomes MRU; adding R3 should evict R2, not R1.
        assert store.get_row("R1", 0, "post_block", 0) is not None
        store.put_row("R3", 0, "post_block", 0, _row(h, 3.0), num_prompt_tokens=1)
        assert store.has_run("R1")
        assert not store.has_run("R2")
        assert store.has_run("R3")

    def test_max_runs_cap(self):
        store = PatchSourceStore(max_bytes=0, max_runs=2)
        for i in range(3):
            store.put_row(f"R{i}", 0, "post_block", 0, _row(4, i), num_prompt_tokens=1)
        # Oldest (R0) evicted to honor max_runs=2.
        assert not store.has_run("R0")
        assert store.has_run("R1")
        assert store.has_run("R2")

    def test_single_oversize_run_kept(self):
        h = 16
        row_bytes = _row(h, 0.0).numel() * _row(h, 0.0).element_size()
        store = PatchSourceStore(max_bytes=row_bytes // 2)  # smaller than one row
        store.put_row("R1", 0, "post_block", 0, _row(h, 1.0), num_prompt_tokens=1)
        # A single run exceeding budget is kept (eviction can't help).
        assert store.has_run("R1")


class TestManifestAndLifecycle:
    def test_manifest(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 2, "post_block", 0, _row(8, 1.0), num_prompt_tokens=7)
        store.put_row("R1", 2, "pre_attn", 1, _row(8, 2.0), num_prompt_tokens=7)
        m = store.manifest("R1")
        assert m is not None
        assert m.run_id == "R1"
        assert m.num_prompt_tokens == 7
        assert m.hidden_size == 8
        assert ("post_block", 2) in m.hook_layers
        assert ("pre_attn", 2) in m.hook_layers
        assert m.positions == [0, 1]
        assert store.manifest("missing") is None

    def test_drop_run(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        assert store.drop_run("R1")
        assert not store.has_run("R1")
        assert not store.drop_run("R1")  # idempotent-ish: False second time

    def test_invalidate_all(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.put_row("R2", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.invalidate_all()
        assert not store.has_run("R1")
        assert not store.has_run("R2")
        assert store.stats().resident_bytes == 0

    def test_stats(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("R1", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.get_row("R1", 0, "post_block", 0)
        store.get_row("R1", 9, "post_block", 0)  # miss
        s = store.stats()
        assert s.rows_put == 1
        assert s.run_hits == 1
        assert s.run_misses == 1
        assert s.runs == 1


class TestActiveAccessor:
    def test_set_get_active(self):
        prev = get_active_patch_source_store()
        try:
            store = PatchSourceStore(max_bytes=0)
            set_active_patch_source_store(store)
            assert get_active_patch_source_store() is store
            set_active_patch_source_store(None)
            assert get_active_patch_source_store() is None
        finally:
            set_active_patch_source_store(prev)


class TestLeases:
    """Leased runs must survive eviction pressure (admission→resolution race)."""

    def test_leased_run_not_evicted(self):
        row_bytes = 4 * 4  # 4 fp32 elements
        store = PatchSourceStore(max_bytes=int(row_bytes * 1.5))
        store.put_row("A", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.lease_runs(["A"], ttl_seconds=60.0)
        # B's write would normally evict LRU run A; the lease must protect it
        # (soft-exceeding the budget instead).
        store.put_row("B", 0, "post_block", 0, _row(4, 2.0), num_prompt_tokens=1)
        assert store.get_row("A", 0, "post_block", 0) is not None

    def test_expired_lease_evictable(self):
        row_bytes = 4 * 4
        store = PatchSourceStore(max_bytes=int(row_bytes * 1.5))
        store.put_row("A", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.lease_runs(["A"], ttl_seconds=0.0)  # immediately expired
        store.put_row("B", 0, "post_block", 0, _row(4, 2.0), num_prompt_tokens=1)
        assert store.get_row("A", 0, "post_block", 0) is None  # evicted
        assert store.get_row("B", 0, "post_block", 0) is not None

    def test_unleased_lru_evicted_before_leased(self):
        row_bytes = 4 * 4
        store = PatchSourceStore(max_bytes=int(row_bytes * 2.5))
        store.put_row("A", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.put_row("B", 0, "post_block", 0, _row(4, 2.0), num_prompt_tokens=1)
        store.lease_runs(["A"], ttl_seconds=60.0)
        # C pushes over budget: A is older but leased -> B (unleased) evicts.
        store.put_row("C", 0, "post_block", 0, _row(4, 3.0), num_prompt_tokens=1)
        assert store.get_row("A", 0, "post_block", 0) is not None
        assert store.get_row("B", 0, "post_block", 0) is None
        assert store.get_row("C", 0, "post_block", 0) is not None

    def test_lease_renewal_extends(self):
        store = PatchSourceStore(max_bytes=0)
        store.put_row("A", 0, "post_block", 0, _row(4, 1.0), num_prompt_tokens=1)
        store.lease_runs(["A"], ttl_seconds=0.0)
        store.lease_runs(["A"], ttl_seconds=60.0)  # renewal wins (max expiry)
        assert store._leased_locked("A") is True


class TestResolutionFailureRegistry:
    def test_record_and_pop(self):
        from vllm.v1.worker.gpu.patch_resolve import (
            pop_resolution_failures,
            record_resolution_failure,
        )

        pop_resolution_failures()  # drain any prior state
        record_resolution_failure("req-1", "source missing: run=X")
        record_resolution_failure("req-1", "source missing: run=Y")
        record_resolution_failure("req-2", "no active source store on rank")
        failures = pop_resolution_failures()
        assert set(failures) == {"req-1", "req-2"}
        assert len(failures["req-1"]) == 2
        assert pop_resolution_failures() == {}  # drained
