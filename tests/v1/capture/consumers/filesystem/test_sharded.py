# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the cross-request ``sharded`` layout + its reader.

WIP — written alongside the sharded implementation; intended to be run in
the morning (the implementation is unverified at commit time). Covers:
  * reader round-trip from hand-built shard files (incl. a request split
    across two shards);
  * consumer e2e: many requests interleaved into shared shards, reader
    reconstructs each request's tensors byte-exact;
  * size-based sealing/rotation produces multiple shard files;
  * shutdown seals the final open shard;
  * per-request result is ``ok`` with the shard file(s) as payload.
"""

from __future__ import annotations

import json
import pathlib
import time
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.reader import read_sharded
from vllm.v1.capture.consumers.filesystem.types import (
    FilesystemCaptureRequest,
    shard_bin_name,
    shard_index_name,
)
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    VllmInternalRequestId,
)

# ---------------------------------------------------------------------------
# Reader round-trip from hand-built shards
# ---------------------------------------------------------------------------


def _write_shard(tag_dir, shard_idx, seq, entries_with_arrays, dtype):
    """entries_with_arrays: list of (request_id, layer, hook, np.ndarray)."""
    tag_dir.mkdir(parents=True, exist_ok=True)
    blob = b""
    entries = []
    for rid, layer, hook, arr in entries_with_arrays:
        payload = arr.tobytes()
        entries.append(
            {
                "request_id": rid,
                "layer": layer,
                "hook": hook,
                "offset": len(blob),
                "nbytes": len(payload),
                "shape": list(arr.shape),
            }
        )
        blob += payload
    (tag_dir / shard_bin_name(shard_idx, seq)).write_bytes(blob)
    (tag_dir / shard_index_name(shard_idx, seq)).write_text(
        json.dumps(
            {
                "layout": "sharded",
                "shard_idx": shard_idx,
                "seq": seq,
                "dtype": dtype,
                "entries": entries,
            }
        )
    )


class TestShardedReader:
    def test_round_trip_multi_request(self, tmp_path: pathlib.Path) -> None:
        tag = tmp_path / "t"
        a = np.random.randn(2, 8).astype(np.float32)  # reqA L0
        b = np.random.randn(3, 8).astype(np.float32)  # reqB L0
        c = np.random.randn(1, 8).astype(np.float32)  # reqA L1
        _write_shard(
            tag,
            0,
            0,
            [
                ("reqA", 0, "post_mlp", a),
                ("reqB", 0, "post_mlp", b),
                ("reqA", 1, "post_mlp", c),
            ],
            "float32",
        )
        got = read_sharded(tag)
        assert set(got) == {"reqA", "reqB"}
        np.testing.assert_array_equal(got["reqA"][(0, "post_mlp")].array, a)
        np.testing.assert_array_equal(got["reqA"][(1, "post_mlp")].array, c)
        np.testing.assert_array_equal(got["reqB"][(0, "post_mlp")].array, b)

    def test_request_spanning_two_shards(self, tmp_path: pathlib.Path) -> None:
        # reqA L0 has rows in seq 0 then seq 1 (sealed mid-request); reader
        # must concatenate them in (seq, offset) order.
        tag = tmp_path / "t"
        a0 = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
        a1 = (np.arange(3 * 8, dtype=np.float32) + 100).reshape(3, 8)
        _write_shard(tag, 0, 0, [("reqA", 0, "post_mlp", a0)], "float32")
        _write_shard(tag, 0, 1, [("reqA", 0, "post_mlp", a1)], "float32")
        got = read_sharded(tag)
        np.testing.assert_array_equal(
            got["reqA"][(0, "post_mlp")].array, np.concatenate([a0, a1])
        )


# ---------------------------------------------------------------------------
# Consumer e2e
# ---------------------------------------------------------------------------


def _ctx(req_id: str, *, num_hidden_layers: int = 4) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId(req_id),
        num_prompt_tokens=10,
        num_computed_tokens=0,
        num_hidden_layers=num_hidden_layers,
        hidden_size=8,
        element_size_bytes=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def _consumer(tmp_path: pathlib.Path, **params: object) -> FilesystemConsumer:
    p: dict[str, object] = {"root": str(tmp_path), "default_layout": "per_file"}
    p.update(params)
    return FilesystemConsumer(vllm_config=MagicMock(), params=p)


def _register(c, req_id, hooks, tag="t"):
    c.validate_client_spec(
        FilesystemCaptureRequest(
            request_id=req_id,
            tag=tag,
            hooks=hooks,
            positions="last_prompt",
            layout="sharded",
        ),
        _ctx(req_id),
    )


def _chunk(req_id, layer, hook, tensor, step):
    return CaptureChunk(
        key=(VllmInternalRequestId(req_id), layer, hook),
        tensor=tensor,
        dtype=tensor.dtype,
        row_offset=0,
        step_index=step,
        metadata={},
    )


def _finalize(req_id, layer, hook):
    return CaptureFinalize(key=(VllmInternalRequestId(req_id), layer, hook), sidecar={})


def _wait(c, key: CaptureKey, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = c.get_result(key)
        if r is not None and r.status != "pending":
            return r
        time.sleep(0.01)
    return c.get_result(key)


class TestShardedConsumer:
    def test_many_requests_one_shard_round_trip(self, tmp_path: pathlib.Path) -> None:
        # num_shards=1 -> all requests land in the same shard, interleaved.
        c = _consumer(tmp_path, num_shards=1)
        reqs = [f"req{i}" for i in range(6)]
        expected: dict[str, dict[tuple[int, str], np.ndarray]] = {}
        try:
            for rid in reqs:
                _register(c, rid, {"post_mlp": [0, 1]})
            # Interleave chunks across requests and layers; 2 steps each.
            tensors: dict = {}
            for step in range(2):
                for rid in reqs:
                    for layer in (0, 1):
                        t = torch.randn(2, 8, dtype=torch.float32)
                        tensors.setdefault((rid, layer), []).append(t)
                        c.submit_chunk(_chunk(rid, layer, "post_mlp", t, step))
            for rid in reqs:
                for layer in (0, 1):
                    c.submit_finalize(_finalize(rid, layer, "post_mlp"))
                    expected.setdefault(rid, {})[(layer, "post_mlp")] = torch.cat(
                        tensors[(rid, layer)]
                    ).numpy()
            # results are ok before seal (data captured, readable after seal)
            r = _wait(c, (VllmInternalRequestId("req0"), 0, "post_mlp"))
            assert r is not None and r.status == "ok"
            assert r.payload and all("shard-" in p for p in r.payload)
        finally:
            c.shutdown(timeout=5.0)  # seals the open shard

        got = read_sharded(tmp_path / "t")
        assert set(got) == set(reqs)
        for rid in reqs:
            for layer in (0, 1):
                np.testing.assert_array_equal(
                    got[rid][(layer, "post_mlp")].array,
                    expected[rid][(layer, "post_mlp")],
                )

    def test_size_based_sealing_rotates(self, tmp_path: pathlib.Path) -> None:
        # Tiny shard_max_bytes forces multiple shard files (seq 0,1,...).
        # Each row is 8*4=32 bytes; cap at 200 bytes -> seal every ~6 rows.
        c = _consumer(tmp_path, num_shards=1, shard_max_bytes=200)
        try:
            _register(c, "r", {"post_mlp": [0]})
            tensors = []
            for step in range(20):
                t = torch.randn(1, 8, dtype=torch.float32)
                tensors.append(t)
                c.submit_chunk(_chunk("r", 0, "post_mlp", t, step))
            c.submit_finalize(_finalize("r", 0, "post_mlp"))
            assert _wait(c, (VllmInternalRequestId("r"), 0, "post_mlp")).status == "ok"
        finally:
            c.shutdown(timeout=5.0)
        tag = tmp_path / "t"
        shards = sorted(tag.glob("shard-*.bin"))
        assert len(shards) >= 2, f"expected rotation into multiple shards, got {shards}"
        got = read_sharded(tag)
        np.testing.assert_array_equal(
            got["r"][(0, "post_mlp")].array, torch.cat(tensors).numpy()
        )


# ---------------------------------------------------------------------------
# Sharded under pipeline parallelism (per-stage shard files, merged on read)
# ---------------------------------------------------------------------------


class _FakePPConfig:
    """``VllmConfig`` stand-in exposing the parallel/model geometry the
    consumer reads to derive its pipeline-parallel stage + layer slice."""

    def __init__(self, *, pp_size: int, pp_rank: int, total_layers: int) -> None:
        from vllm.distributed.utils import get_pp_indices

        class _Parallel:
            pipeline_parallel_size = pp_size
            tensor_parallel_size = 1
            rank = pp_rank  # tp=1 → rank == pp_rank

        class _Model:
            def get_layers_start_end_indices(self, parallel_config: object):
                pp_rank_ = (
                    parallel_config.rank // parallel_config.tensor_parallel_size
                ) % parallel_config.pipeline_parallel_size
                return get_pp_indices(total_layers, pp_rank_, pp_size)

        self.parallel_config = _Parallel()
        self.model_config = _Model()


def _pp_consumer(
    tmp_path: pathlib.Path, *, pp_rank: int, **params: object
) -> FilesystemConsumer:
    p: dict[str, object] = {"root": str(tmp_path), "default_layout": "per_file"}
    p.update(params)
    cfg = _FakePPConfig(pp_size=2, pp_rank=pp_rank, total_layers=4)
    return FilesystemConsumer(vllm_config=cfg, params=p)


class TestShardedPipelineParallel:
    def test_two_stage_shards_merge(self, tmp_path: pathlib.Path) -> None:
        # pp_size=2, 4 layers → stage 0 owns [0,2), stage 1 owns [2,4). Each
        # stage seals its own shard-pp{rank} files; read_sharded merges by
        # request across both, recovering the full layer set.
        req = "req"
        hooks = {"post_mlp": [0, 1, 2, 3]}
        tensors = {layer: torch.randn(2, 8, dtype=torch.float32) for layer in range(4)}
        c0 = _pp_consumer(tmp_path, pp_rank=0, num_shards=1)
        c1 = _pp_consumer(tmp_path, pp_rank=1, num_shards=1)
        try:
            _register(c0, req, hooks)
            _register(c1, req, hooks)
            for layer in (0, 1):
                c0.submit_chunk(_chunk(req, layer, "post_mlp", tensors[layer], 0))
                c0.submit_finalize(_finalize(req, layer, "post_mlp"))
            for layer in (2, 3):
                c1.submit_chunk(_chunk(req, layer, "post_mlp", tensors[layer], 0))
                c1.submit_finalize(_finalize(req, layer, "post_mlp"))
        finally:
            c0.shutdown(timeout=5.0)  # seal each stage's open shard
            c1.shutdown(timeout=5.0)

        tag = tmp_path / "t"
        # Both stages' shard files live in the tag dir, distinguished by rank.
        assert sorted(p.name for p in tag.glob("shard-pp00-*.bin"))
        assert sorted(p.name for p in tag.glob("shard-pp01-*.bin"))
        got = read_sharded(tag)
        assert set(got) == {req}
        assert set(got[req]) == {(layer, "post_mlp") for layer in range(4)}
        for layer in range(4):
            np.testing.assert_array_equal(
                got[req][(layer, "post_mlp")].array, tensors[layer].numpy()
            )

    def test_stage_owning_no_layers_creates_no_state(
        self, tmp_path: pathlib.Path
    ) -> None:
        req = "req-skip"
        c1 = _pp_consumer(tmp_path, pp_rank=1, num_shards=1)
        try:
            _register(c1, req, {"post_mlp": [0, 1]})  # all on stage 0
            assert req not in c1._sharded_requests
        finally:
            c1.shutdown(timeout=5.0)
