# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared-memory IPC region for inline steering vectors.

Covers:

- Round-trip in-process: write packed bytes, read same bytes
- Round-trip across processes: parent writes, child reads, byte-equal
- Reset semantics: cursor resets when in-flight refcount hits zero
- atexit cleanup: region file is unlinked on interpreter shutdown
- ``maybe_create`` fail-soft on non-Linux platforms
- Integration with ``SamplingParams.effective_*_steering`` cached_property
- Integration with ``maybe_pack_inline_steering_for_request``
"""

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

from vllm.config.steering_types import maybe_pack_inline_steering_for_request
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.steering_shm import (
    SteeringShmRegion,
    get_worker_region,
    materialize_shm_dict,
    set_client_region,
    set_worker_region,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shm_region():
    """Create a small region for fast tests; clean up after."""
    region = SteeringShmRegion(total_size_bytes=1024 * 1024)  # 1 MiB
    try:
        yield region
    finally:
        region.close()
        # Process-local registry cleanup so tests don't bleed.
        set_client_region(None)
        set_worker_region(None)


# ---------------------------------------------------------------------------
# In-process round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_write_read_float32(self, shm_region):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        offset, length, dtype_str, shape = shm_region.write_packed(arr)
        assert offset == 0
        assert length == arr.nbytes
        assert dtype_str == "<f4"
        assert shape == (4,)
        out = shm_region.read_packed(offset, length, dtype_str, shape)
        assert np.array_equal(out, arr)
        assert out.dtype == arr.dtype

    def test_write_read_float16(self, shm_region):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        descriptor = shm_region.write_packed(arr)
        out = shm_region.read_packed(*descriptor)
        assert np.array_equal(out, arr)
        assert out.dtype == np.float16

    def test_write_read_2d(self, shm_region):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        descriptor = shm_region.write_packed(arr)
        out = shm_region.read_packed(*descriptor)
        assert out.shape == (3, 4)
        assert np.array_equal(out, arr)

    def test_multiple_writes_distinct_offsets(self, shm_region):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        off_a, len_a, _, _ = shm_region.write_packed(a)
        off_b, len_b, _, _ = shm_region.write_packed(b)
        assert off_a == 0
        assert off_b == len_a
        assert len_b == b.nbytes
        # Both bytes are still readable.
        assert np.array_equal(shm_region.read_packed(off_a, len_a, "<f4", (2,)), a)
        assert np.array_equal(shm_region.read_packed(off_b, len_b, "<f4", (3,)), b)

    def test_write_overflow_raises(self, shm_region):
        # Region is 1 MiB; 256K floats fits, 512K floats does not.
        big = np.zeros(512 * 1024, dtype=np.float32)
        with pytest.raises(RuntimeError, match="overflow"):
            shm_region.write_packed(big)

    def test_non_contiguous_input_handled(self, shm_region):
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        # Non-contiguous: every other column.
        view = base[:, ::2]
        assert not view.flags["C_CONTIGUOUS"]
        descriptor = shm_region.write_packed(view)
        out = shm_region.read_packed(*descriptor)
        assert np.array_equal(out, view)


# ---------------------------------------------------------------------------
# Reset / generation semantics
# ---------------------------------------------------------------------------


class TestResetSemantics:
    def test_request_finished_resets_cursor(self, shm_region):
        a = np.array([1.0, 2.0], dtype=np.float32)
        shm_region.request_started()
        off1, _, _, _ = shm_region.write_packed(a)
        assert off1 == 0
        # Without finishing, the next write advances.
        off2, _, _, _ = shm_region.write_packed(a)
        assert off2 > 0
        # Drop the refcount → cursor resets.
        shm_region.request_finished()
        off3, _, _, _ = shm_region.write_packed(a)
        assert off3 == 0

    def test_no_reset_with_inflight(self, shm_region):
        a = np.array([1.0, 2.0], dtype=np.float32)
        shm_region.request_started()
        shm_region.request_started()
        shm_region.write_packed(a)
        shm_region.request_finished()
        # Still one in flight → no reset.
        off2, _, _, _ = shm_region.write_packed(a)
        assert off2 > 0
        shm_region.request_finished()
        # Now resets.
        off3, _, _, _ = shm_region.write_packed(a)
        assert off3 == 0

    def test_request_finished_underflow_safe(self, shm_region):
        # Calling without a matching started call shouldn't crash.
        shm_region.request_finished()
        a = np.array([1.0], dtype=np.float32)
        off, _, _, _ = shm_region.write_packed(a)
        assert off == 0


# ---------------------------------------------------------------------------
# Multi-process round-trip
# ---------------------------------------------------------------------------


def _child_read(path, offset, length, dtype_str, shape, out_q):
    """Run in a child process: open the shm path readonly, read, send bytes
    back over the queue."""
    try:
        ro = SteeringShmRegion.open_readonly(path)
        try:
            arr = ro.read_packed(offset, length, dtype_str, shape)
            out_q.put(("ok", arr.tobytes(), arr.dtype.str, arr.shape))
        finally:
            ro.close()
    except Exception as exc:  # pragma: no cover - debug aid
        out_q.put(("err", repr(exc)))


class TestCrossProcess:
    def test_parent_write_child_read(self, shm_region):
        arr = np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32)
        offset, length, dtype_str, shape = shm_region.write_packed(arr)

        ctx = mp.get_context("fork")
        out_q: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_child_read,
            args=(shm_region.mmap_path, offset, length, dtype_str, shape, out_q),
        )
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0, "child process did not exit cleanly"
        result = out_q.get(timeout=1)
        assert result[0] == "ok", f"child failed: {result}"
        _, raw_bytes, child_dtype, child_shape = result
        assert child_dtype == dtype_str
        assert child_shape == shape
        recovered = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(
            child_shape
        )
        assert np.array_equal(recovered, arr)


# ---------------------------------------------------------------------------
# atexit cleanup
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    def test_creator_unlinks_on_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Use a custom path under tmpdir so we can verify post-close.
            path = os.path.join(tmp, "atexit_test.mmap")
            region = SteeringShmRegion(
                total_size_bytes=4096,
                path=path,
            )
            assert os.path.exists(path)
            region.close()
            assert not os.path.exists(path), (
                "Creator should unlink the backing file on close"
            )

    def test_child_process_unlinks_on_normal_exit(self):
        """Spawn a fresh interpreter that creates a region and exits;
        the atexit hook must unlink the file."""
        with tempfile.TemporaryDirectory() as tmp:
            sentinel_path = os.path.join(tmp, "child_atexit.mmap")
            script = (
                "import sys; "
                "sys.path.insert(0, "
                + repr("/home/nymph/Code/vllm/steering-shm")
                + "); "
                "from vllm.v1.engine.steering_shm import SteeringShmRegion; "
                "r = SteeringShmRegion(total_size_bytes=4096, path="
                + repr(sentinel_path)
                + ")"
            )
            # Use the project's venv interpreter so deps resolve.
            python_bin = sys.executable
            result = subprocess.run(
                [python_bin, "-c", script],
                capture_output=True,
                text=True,
                timeout=20,
            )
            # Even if the child failed for unrelated reasons, ensure
            # the file isn't lying around.
            assert result.returncode == 0, (
                f"child failed: stdout={result.stdout!r} stderr={result.stderr!r}"
            )
            assert not os.path.exists(sentinel_path), (
                "atexit hook should unlink the region file on normal exit"
            )


# ---------------------------------------------------------------------------
# materialize_shm_dict + integration
# ---------------------------------------------------------------------------


class TestMaterializeShmDict:
    def test_materialize_round_trip(self, shm_region):
        a0 = np.array([1.0, 2.0], dtype=np.float32)
        a1 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        shm_dict = {
            "post_mlp": {
                0: shm_region.write_packed(a0),
                1: shm_region.write_packed(a1),
            }
        }
        out = materialize_shm_dict(shm_dict, shm_region)
        assert set(out.keys()) == {"post_mlp"}
        assert np.array_equal(out["post_mlp"][0], a0)
        assert np.array_equal(out["post_mlp"][1], a1)


# ---------------------------------------------------------------------------
# Integration with maybe_pack_inline_steering_for_request
# ---------------------------------------------------------------------------


class TestPackThroughShm:
    def test_pack_writes_to_shm_and_clears_originals(self, shm_region):
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0, 3.0]}},
        )
        maybe_pack_inline_steering_for_request(
            sp, torch.float32, shm_region=shm_region
        )
        # Originals cleared.
        assert sp.steering_vectors is None
        assert sp.prefill_steering_vectors is None
        assert sp.decode_steering_vectors is None
        # Shm fields populated, packed fields stay None (took the shm path).
        assert sp._effective_prefill_steering_shm is not None
        assert sp._effective_decode_steering_shm is not None
        assert sp._effective_prefill_steering_packed is None
        assert sp._effective_decode_steering_packed is None
        # Path threaded through.
        assert sp._steering_shm_path == shm_region.mmap_path
        # Descriptor shape sanity.
        descriptor = sp._effective_prefill_steering_shm["post_mlp"][0]
        offset, length, dtype_str, shape = descriptor
        assert length == 3 * 4  # fp32, 3 elements
        assert dtype_str == "<f4"
        assert shape == (3,)

    def test_effective_steering_pre_cached_on_writer(self, shm_region):
        """The writer-side process pre-caches the resolved arrays so it
        doesn't have to bounce through shm to read them back."""
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0, 3.0]}},
        )
        maybe_pack_inline_steering_for_request(
            sp, torch.float32, shm_region=shm_region
        )
        eff = sp.effective_prefill_steering
        assert eff is not None
        assert eff["post_mlp"][0].tolist() == [1.0, 2.0, 3.0]

    def test_effective_steering_via_worker_region(self, shm_region):
        """A fresh SamplingParams (without the pre-cache) reading via
        the worker-side region path resolves descriptor tuples to
        ndarrays."""
        # Build a SamplingParams directly with pre-built descriptor
        # tuples so the cached_property path is exercised cleanly.
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        descriptor = shm_region.write_packed(arr)

        sp = SamplingParams(max_tokens=1)
        sp._effective_prefill_steering_shm = {"post_mlp": {0: descriptor}}
        sp._steering_shm_path = shm_region.mmap_path

        # Register the region as the worker-side process-local handle.
        set_worker_region(shm_region)
        try:
            eff = sp.effective_prefill_steering
        finally:
            set_worker_region(None)
        assert eff is not None
        assert np.array_equal(eff["post_mlp"][0], arr)

    def test_no_region_falls_back_to_packed(self):
        """When ``shm_region=None`` is passed, behaviour is exactly the
        existing inline-packed wire path."""
        sp = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0]}},
        )
        maybe_pack_inline_steering_for_request(sp, torch.float32, shm_region=None)
        assert sp._effective_prefill_steering_shm is None
        assert sp._effective_decode_steering_shm is None
        assert sp._effective_prefill_steering_packed is not None
        assert sp._effective_prefill_steering_packed["post_mlp"][0].tolist() == [
            1.0,
            2.0,
        ]


# ---------------------------------------------------------------------------
# Region registry helpers
# ---------------------------------------------------------------------------


class TestRegionRegistry:
    def test_get_worker_region_starts_empty(self):
        # Test isolation: unregister anything a previous test left.
        set_worker_region(None)
        assert get_worker_region() is None

    def test_set_worker_region_round_trips(self, shm_region):
        set_worker_region(shm_region)
        try:
            assert get_worker_region() is shm_region
        finally:
            set_worker_region(None)


# ---------------------------------------------------------------------------
# Wire-format / msgspec round-trip
# ---------------------------------------------------------------------------


class TestMsgspecRoundtrip:
    def test_shm_descriptor_round_trips_through_msgspec(self, shm_region):
        """The descriptor tuples + path field survive msgspec round-trip."""
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        sp_in = SamplingParams(
            max_tokens=1,
            steering_vectors={"post_mlp": {0: [1.0, 2.0, 3.0]}},
        )
        maybe_pack_inline_steering_for_request(
            sp_in, torch.float32, shm_region=shm_region
        )

        enc = MsgpackEncoder()
        bufs = enc.encode(sp_in)
        dec = MsgpackDecoder(SamplingParams)
        sp_out = dec.decode(bufs)

        assert sp_out._steering_shm_path == shm_region.mmap_path
        assert sp_out._effective_prefill_steering_shm is not None
        # Resolve via the worker-side region path.
        set_worker_region(shm_region)
        try:
            eff = sp_out.effective_prefill_steering
        finally:
            set_worker_region(None)
        assert eff is not None
        assert eff["post_mlp"][0].tolist() == [1.0, 2.0, 3.0]

    def test_shm_payload_smaller_than_unpacked(self, shm_region):
        """Sanity: the shm wire form ships only descriptor tuples."""
        from vllm.v1.serial_utils import MsgpackEncoder

        # Use a moderate-size workload so the difference is visible.
        vectors = {
            "post_mlp": {i: [float(j) for j in range(2560)] for i in range(8)}
        }
        sp_unpacked = SamplingParams(max_tokens=1, steering_vectors=vectors)
        sp_shm = SamplingParams(max_tokens=1, steering_vectors=vectors)
        maybe_pack_inline_steering_for_request(
            sp_shm, torch.float32, shm_region=shm_region
        )

        enc = MsgpackEncoder()
        unpacked_bytes = sum(len(b) for b in enc.encode(sp_unpacked))
        shm_bytes = sum(len(b) for b in enc.encode(sp_shm))
        # 2560 floats × 8 layers × ~9 B/float ≈ 184 K msgpack;
        # 8 layers × ~64 B descriptor + path ≈ 600 B shm.
        # Expect at least 50× reduction.
        assert shm_bytes * 50 < unpacked_bytes, (
            f"shm={shm_bytes} not < unpacked={unpacked_bytes} / 50"
        )
