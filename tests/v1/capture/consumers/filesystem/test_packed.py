# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the packed on-disk layout and its reference reader.

Split in two parts:
  * ``TestReader`` — builds both layouts by hand and round-trips them
    through ``reader.py``. Verifies the on-disk contract independently
    of the consumer.
  * ``TestPackedConsumer`` — drives ``FilesystemConsumer`` in ``packed``
    mode end-to-end (added with the consumer implementation).
"""

from __future__ import annotations

import json
import pathlib

# Imports for the consumer-level tests (torch + framework types).
import time
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.reader import (
    read_packed,
    read_per_file,
    read_request,
)
from vllm.v1.capture.consumers.filesystem.types import (
    PACKED_BIN_NAME,
    PACKED_INDEX_NAME,
    FilesystemCaptureRequest,
    packed_bin_name,
    packed_index_name,
)
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureSpec,
    VllmInternalRequestId,
)


def _write_per_file(
    req_dir: pathlib.Path, layer: int, hook: str, arr: np.ndarray, dtype: str
) -> None:
    req_dir.mkdir(parents=True, exist_ok=True)
    bin_path = req_dir / f"{layer}_{hook}.bin"
    bin_path.write_bytes(arr.tobytes())
    bin_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "request_id": req_dir.name,
                "layer": layer,
                "hook": hook,
                "shape": list(arr.shape),
                "dtype": dtype,
            }
        )
    )


def _write_packed(
    req_dir: pathlib.Path,
    tensors: list[tuple[int, str, np.ndarray]],
    dtype: str,
) -> None:
    req_dir.mkdir(parents=True, exist_ok=True)
    blob = b""
    entries = []
    for layer, hook, arr in tensors:
        payload = arr.tobytes()
        entries.append(
            {
                "layer": layer,
                "hook": hook,
                "offset": len(blob),
                "nbytes": len(payload),
                "shape": list(arr.shape),
            }
        )
        blob += payload
    (req_dir / PACKED_BIN_NAME).write_bytes(blob)
    (req_dir / PACKED_INDEX_NAME).write_text(
        json.dumps(
            {
                "request_id": req_dir.name,
                "layout": "packed",
                "dtype": dtype,
                "entries": entries,
            }
        )
    )


class TestReader:
    def test_per_file_round_trip(self, tmp_path: pathlib.Path) -> None:
        arr = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
        _write_per_file(tmp_path / "req-1", 3, "post_block", arr, "float32")
        entry = read_per_file(tmp_path / "req-1" / "3_post_block.bin")
        assert entry.layer == 3
        assert entry.hook == "post_block"
        assert entry.dtype == "float32"
        np.testing.assert_array_equal(entry.array, arr)

    def test_packed_round_trip(self, tmp_path: pathlib.Path) -> None:
        a = np.random.randn(4, 16).astype(np.float32)
        b = np.random.randn(1, 16).astype(np.float32)
        c = np.random.randn(7, 16).astype(np.float32)
        tensors = [(0, "post_block", a), (5, "post_block", b), (5, "post_attn", c)]
        _write_packed(tmp_path / "req-2", tensors, "float32")

        got = read_packed(tmp_path / "req-2")
        assert set(got) == {(0, "post_block"), (5, "post_block"), (5, "post_attn")}
        np.testing.assert_array_equal(got[(0, "post_block")].array, a)
        np.testing.assert_array_equal(got[(5, "post_block")].array, b)
        np.testing.assert_array_equal(got[(5, "post_attn")].array, c)

    def test_packed_accepts_index_or_bin_or_dir(self, tmp_path: pathlib.Path) -> None:
        arr = np.random.randn(2, 4).astype(np.float32)
        _write_packed(tmp_path / "r", [(1, "post_block", arr)], "float32")
        for target in (
            tmp_path / "r",
            tmp_path / "r" / PACKED_INDEX_NAME,
            tmp_path / "r" / PACKED_BIN_NAME,
        ):
            got = read_packed(target)
            np.testing.assert_array_equal(got[(1, "post_block")].array, arr)

    def test_read_request_autodetects_layout(self, tmp_path: pathlib.Path) -> None:
        # per_file dir
        pf = tmp_path / "pf"
        _write_per_file(pf, 0, "post_block", np.ones((2, 4), np.float32), "float32")
        _write_per_file(pf, 1, "post_block", np.zeros((3, 4), np.float32), "float32")
        got_pf = read_request(pf)
        assert set(got_pf) == {(0, "post_block"), (1, "post_block")}
        # packed dir
        pk = tmp_path / "pk"
        _write_packed(pk, [(0, "post_block", np.ones((2, 4), np.float32))], "float32")
        got_pk = read_request(pk)
        assert set(got_pk) == {(0, "post_block")}

    def test_bfloat16_returns_uint16(self, tmp_path: pathlib.Path) -> None:
        # bf16 is stored as raw uint16; the reader returns it as uint16.
        raw = np.array([1, 2, 3, 4], dtype=np.uint16).reshape(2, 2)
        _write_per_file(tmp_path / "bf", 0, "post_block", raw, "bfloat16")
        entry = read_per_file(tmp_path / "bf" / "0_post_block.bin")
        assert entry.dtype == "bfloat16"
        assert entry.array.dtype == np.uint16
        np.testing.assert_array_equal(entry.array, raw)

    def test_truncated_packed_raises(self, tmp_path: pathlib.Path) -> None:
        arr = np.random.randn(4, 8).astype(np.float32)
        d = tmp_path / "trunc"
        _write_packed(d, [(0, "post_block", arr)], "float32")
        # Corrupt: truncate the bin so the entry's bytes are missing.
        (d / PACKED_BIN_NAME).write_bytes(b"\x00\x00")
        try:
            read_packed(d)
        except ValueError as e:
            assert "truncated" in str(e)
        else:
            raise AssertionError("expected ValueError on truncated packed bin")


# ---------------------------------------------------------------------------
# Consumer-level packed tests
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
    p: dict[str, object] = {"root": str(tmp_path)}
    p.update(params)
    return FilesystemConsumer(vllm_config=MagicMock(), params=p)


class _FakePPConfig:
    """Minimal ``VllmConfig`` stand-in exposing the parallel/model fields
    ``FilesystemConsumer`` reads to derive its pipeline-parallel geometry.

    Mirrors production: ``get_layers_start_end_indices`` defers to the same
    ``get_pp_indices`` the real ``ModelConfig`` / ``CaptureManager`` use, so
    the consumer's local layer slice matches what the manager would feed it.
    """

    def __init__(self, *, pp_size: int, pp_rank: int, total_layers: int) -> None:
        from vllm.distributed.utils import get_pp_indices

        tp_size = 1

        class _Parallel:
            pipeline_parallel_size = pp_size
            tensor_parallel_size = tp_size
            rank = pp_rank * tp_size  # layout DP x PP x TP; tp=1 → rank==pp_rank

        class _Model:
            def get_layers_start_end_indices(self, parallel_config: object):
                pp_rank_ = (
                    parallel_config.rank // parallel_config.tensor_parallel_size
                ) % parallel_config.pipeline_parallel_size
                return get_pp_indices(total_layers, pp_rank_, pp_size)

        self.parallel_config = _Parallel()
        self.model_config = _Model()


def _pp_consumer(
    tmp_path: pathlib.Path,
    *,
    pp_size: int,
    pp_rank: int,
    total_layers: int,
    **params: object,
) -> FilesystemConsumer:
    p: dict[str, object] = {"root": str(tmp_path)}
    p.update(params)
    cfg = _FakePPConfig(pp_size=pp_size, pp_rank=pp_rank, total_layers=total_layers)
    return FilesystemConsumer(vllm_config=cfg, params=p)


def _register_packed(consumer: FilesystemConsumer, req_id: str, hooks: dict) -> None:
    raw = FilesystemCaptureRequest(
        request_id=req_id,
        tag="t",
        hooks=hooks,
        positions="last_prompt",
        layout="packed",
    )
    consumer.validate_client_spec(raw, _ctx(req_id))


def _chunk(req_id: str, layer: int, hook: str, tensor: torch.Tensor, step: int):
    return CaptureChunk(
        key=(VllmInternalRequestId(req_id), layer, hook),
        tensor=tensor,
        dtype=tensor.dtype,
        row_offset=0,
        step_index=step,
        metadata={},
    )


def _finalize(req_id: str, layer: int, hook: str) -> CaptureFinalize:
    return CaptureFinalize(key=(VllmInternalRequestId(req_id), layer, hook), sidecar={})


def _wait(consumer: FilesystemConsumer, key: CaptureKey, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = consumer.get_result(key)
        if r is not None and r.status != "pending":
            return r
        time.sleep(0.01)
    return consumer.get_result(key)


class TestPackedConsumer:
    def test_packed_round_trip(self, tmp_path: pathlib.Path) -> None:
        req = "req-pk"
        c = _consumer(tmp_path)
        try:
            _register_packed(c, req, {"post_block": [0, 2]})
            # (0,post_block) spans 2 steps; (2,post_block) one step. Submit
            # interleaved across keys to exercise per-chunk indexing.
            a0 = torch.randn(2, 8, dtype=torch.float32)
            a1 = torch.randn(3, 8, dtype=torch.float32)
            b0 = torch.randn(1, 8, dtype=torch.float32)
            c.submit_chunk(_chunk(req, 0, "post_block", a0, 0))
            c.submit_chunk(_chunk(req, 2, "post_block", b0, 0))
            c.submit_chunk(_chunk(req, 0, "post_block", a1, 1))
            for layer in (0, 2):
                c.submit_finalize(_finalize(req, layer, "post_block"))

            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"

            req_dir = tmp_path / "t" / req
            assert (req_dir / PACKED_BIN_NAME).exists()
            assert (req_dir / PACKED_INDEX_NAME).exists()
            assert not list(req_dir.glob("*_post_block.bin")), "no per-file bins"

            got = read_request(req_dir)
            assert set(got) == {(0, "post_block"), (2, "post_block")}
            np.testing.assert_array_equal(
                got[(0, "post_block")].array, torch.cat([a0, a1]).numpy()
            )
            np.testing.assert_array_equal(got[(2, "post_block")].array, b0.numpy())
        finally:
            c.shutdown(timeout=5.0)

    def test_submit_chunk_batch_round_trip(self, tmp_path: pathlib.Path) -> None:
        # Batched submit: a step's worth of (layer) chunks handed over in
        # one call must produce the same packed file as per-chunk submits.
        # Two steps batched; (0,post_block) spans both, (2,post_block) only
        # step 0 — concatenation order must follow submission order.
        req = "req-batch"
        c = _consumer(tmp_path)
        try:
            _register_packed(c, req, {"post_block": [0, 2]})
            a0 = torch.randn(2, 8, dtype=torch.float32)
            b0 = torch.randn(1, 8, dtype=torch.float32)
            a1 = torch.randn(3, 8, dtype=torch.float32)
            # step 0: both layers, in one batch
            c.submit_chunk_batch(
                [_chunk(req, 0, "post_block", a0, 0), _chunk(req, 2, "post_block", b0, 0)]
            )
            # step 1: only layer 0
            c.submit_chunk_batch([_chunk(req, 0, "post_block", a1, 1)])
            for layer in (0, 2):
                c.submit_finalize(_finalize(req, layer, "post_block"))

            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"

            req_dir = tmp_path / "t" / req
            # One packed file for the whole request, no per-file bins.
            assert (req_dir / PACKED_BIN_NAME).exists()
            assert not list(req_dir.glob("*_post_block.bin"))

            got = read_request(req_dir)
            assert set(got) == {(0, "post_block"), (2, "post_block")}
            np.testing.assert_array_equal(
                got[(0, "post_block")].array, torch.cat([a0, a1]).numpy()
            )
            np.testing.assert_array_equal(got[(2, "post_block")].array, b0.numpy())
        finally:
            c.shutdown(timeout=5.0)

    def test_batch_matches_per_chunk_bytes(self, tmp_path: pathlib.Path) -> None:
        # The batched and per-chunk paths must write byte-identical packed
        # files for the same chunks.
        layers = [0, 1, 2, 3]
        tensors = {layer: torch.randn(2, 8, dtype=torch.float32) for layer in layers}

        def run(req: str, batched: bool) -> bytes:
            c = _consumer(tmp_path)
            try:
                _register_packed(c, req, {"post_block": layers})
                step_chunks = [
                    _chunk(req, layer, "post_block", tensors[layer], 0)
                    for layer in layers
                ]
                if batched:
                    c.submit_chunk_batch(step_chunks)
                else:
                    for ch in step_chunks:
                        c.submit_chunk(ch)
                for layer in layers:
                    c.submit_finalize(_finalize(req, layer, "post_block"))
                key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
                assert _wait(c, key0).status == "ok"
                return (tmp_path / "t" / req / PACKED_BIN_NAME).read_bytes()
            finally:
                c.shutdown(timeout=5.0)

        assert run("req-perchunk", batched=False) == run("req-batched", batched=True)

    def test_finalize_aggregation_waits_for_all_keys(
        self, tmp_path: pathlib.Path
    ) -> None:
        req = "req-agg"
        c = _consumer(tmp_path)
        try:
            _register_packed(c, req, {"post_block": [0, 1]})
            c.submit_chunk(_chunk(req, 0, "post_block", torch.randn(2, 8), 0))
            c.submit_chunk(_chunk(req, 1, "post_block", torch.randn(2, 8), 0))
            # Finalize only the first key — packed file must NOT publish.
            c.submit_finalize(_finalize(req, 0, "post_block"))
            time.sleep(0.2)
            req_dir = tmp_path / "t" / req
            assert not (req_dir / PACKED_INDEX_NAME).exists(), (
                "packed index published before all keys finalized"
            )
            # Finalize the last expected key — now it publishes.
            c.submit_finalize(_finalize(req, 1, "post_block"))
            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"
            assert (req_dir / PACKED_INDEX_NAME).exists()
        finally:
            c.shutdown(timeout=5.0)

    def test_zero_chunk_key(self, tmp_path: pathlib.Path) -> None:
        # A spec key that never receives a chunk still gets finalized;
        # the packed file completes, the reader just lacks that key.
        req = "req-zero"
        c = _consumer(tmp_path)
        try:
            _register_packed(c, req, {"post_block": [0, 1]})
            c.submit_chunk(_chunk(req, 0, "post_block", torch.randn(2, 8), 0))
            for layer in (0, 1):  # key 1 had no chunk
                c.submit_finalize(_finalize(req, layer, "post_block"))
            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"
            got = read_request(tmp_path / "t" / req)
            assert set(got) == {(0, "post_block")}
        finally:
            c.shutdown(timeout=5.0)

    def test_per_file_default_unchanged(self, tmp_path: pathlib.Path) -> None:
        # layout unset -> per_file: per-(layer,hook) files, no packed.*.
        req = "req-pf"
        c = _consumer(tmp_path)
        try:
            raw = FilesystemCaptureRequest(
                request_id=req,
                tag="t",
                hooks={"post_block": [0]},
                positions="last_prompt",
            )
            c.validate_client_spec(raw, _ctx(req))
            t0 = torch.randn(2, 8, dtype=torch.float32)
            c.submit_chunk(_chunk(req, 0, "post_block", t0, 0))
            c.submit_finalize(_finalize(req, 0, "post_block"))
            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"
            req_dir = tmp_path / "t" / req
            assert (req_dir / "0_post_block.bin").exists()
            assert not (req_dir / PACKED_INDEX_NAME).exists()
            entry = read_per_file(req_dir / "0_post_block.bin")
            np.testing.assert_array_equal(entry.array, t0.numpy())
            assert entry.dtype == "float32"  # sidecar now self-describing
        finally:
            c.shutdown(timeout=5.0)

    # --- entrypoint path: raw dict spec (as SamplingParams.capture carries) ---

    def test_dict_spec_layout_packed(self, tmp_path: pathlib.Path) -> None:
        # _admit_capture passes the raw dict straight to validate_client_spec;
        # a dict carrying layout="packed" must produce the packed layout.
        req = "req-dict-pk"
        c = _consumer(tmp_path)
        try:
            c.validate_client_spec(
                {
                    "request_id": req,
                    "tag": "t",
                    "hooks": {"post_block": [0, 1]},
                    "positions": "last_prompt",
                    "layout": "packed",
                },
                _ctx(req),
            )
            for layer in (0, 1):
                c.submit_chunk(_chunk(req, layer, "post_block", torch.randn(2, 8), 0))
            for layer in (0, 1):
                c.submit_finalize(_finalize(req, layer, "post_block"))
            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"
            req_dir = tmp_path / "t" / req
            assert (req_dir / PACKED_INDEX_NAME).exists()
            assert set(read_request(req_dir)) == {(0, "post_block"), (1, "post_block")}
        finally:
            c.shutdown(timeout=5.0)

    def test_dict_spec_defaults_per_file(self, tmp_path: pathlib.Path) -> None:
        # A dict without layout keeps the default (per_file).
        req = "req-dict-pf"
        c = _consumer(tmp_path)
        try:
            c.validate_client_spec(
                {
                    "request_id": req,
                    "tag": "t",
                    "hooks": {"post_block": [0]},
                    "positions": "last_prompt",
                },
                _ctx(req),
            )
            c.submit_chunk(_chunk(req, 0, "post_block", torch.randn(2, 8), 0))
            c.submit_finalize(_finalize(req, 0, "post_block"))
            key0: CaptureKey = (VllmInternalRequestId(req), 0, "post_block")
            assert _wait(c, key0).status == "ok"
            req_dir = tmp_path / "t" / req
            assert (req_dir / "0_post_block.bin").exists()
            assert not (req_dir / PACKED_INDEX_NAME).exists()
        finally:
            c.shutdown(timeout=5.0)

    def test_invalid_layout_rejected(self, tmp_path: pathlib.Path) -> None:
        # Bad layout must raise CaptureValidationError (→ HTTP 400 at the
        # entrypoint), not silently fall through.
        from vllm.v1.capture.errors import CaptureValidationError

        req = "req-bad"
        c = _consumer(tmp_path)
        try:
            raised = False
            try:
                c.validate_client_spec(
                    {
                        "request_id": req,
                        "tag": "t",
                        "hooks": {"post_block": [0]},
                        "positions": "last_prompt",
                        "layout": "bogus",
                    },
                    _ctx(req),
                )
            except CaptureValidationError as e:
                raised = True
                assert "layout" in str(e)
            assert raised, "expected CaptureValidationError for invalid layout"
        finally:
            c.shutdown(timeout=5.0)


# ---------------------------------------------------------------------------
# Packed under pipeline parallelism (per-stage files, merged on read)
# ---------------------------------------------------------------------------


def _pp_raw(req: str, hooks: dict) -> FilesystemCaptureRequest:
    return FilesystemCaptureRequest(
        request_id=req, tag="t", hooks=hooks, positions="last_prompt", layout="packed"
    )


class TestPackedPipelineParallel:
    def test_two_stage_split_and_merge(self, tmp_path: pathlib.Path) -> None:
        # pp_size=2, 4 layers → stage 0 owns [0,2), stage 1 owns [2,4). Each
        # stage writes its own packed-pp{rank} file; the reader merges them
        # back into the request's full layer set.
        req = "req-pp"
        total = 4
        hooks = {"post_block": [0, 1, 2, 3]}
        tensors = {layer: torch.randn(2, 8, dtype=torch.float32) for layer in range(4)}
        c0 = _pp_consumer(tmp_path, pp_size=2, pp_rank=0, total_layers=total)
        c1 = _pp_consumer(tmp_path, pp_size=2, pp_rank=1, total_layers=total)
        try:
            c0.validate_client_spec(_pp_raw(req, hooks), _ctx(req))
            c1.validate_client_spec(_pp_raw(req, hooks), _ctx(req))
            # The manager only feeds each stage its owned layers.
            for layer in (0, 1):
                c0.submit_chunk(_chunk(req, layer, "post_block", tensors[layer], 0))
            for layer in (0, 1):
                c0.submit_finalize(_finalize(req, layer, "post_block"))
            for layer in (2, 3):
                c1.submit_chunk(_chunk(req, layer, "post_block", tensors[layer], 0))
            for layer in (2, 3):
                c1.submit_finalize(_finalize(req, layer, "post_block"))

            assert _wait(c0, (VllmInternalRequestId(req), 0, "post_block")).status == "ok"
            assert _wait(c1, (VllmInternalRequestId(req), 2, "post_block")).status == "ok"

            req_dir = tmp_path / "t" / req
            # Per-stage files exist; the pp-agnostic packed.json does not.
            assert (req_dir / packed_index_name(0)).exists()
            assert (req_dir / packed_bin_name(0)).exists()
            assert (req_dir / packed_index_name(1)).exists()
            assert (req_dir / packed_bin_name(1)).exists()
            assert not (req_dir / PACKED_INDEX_NAME).exists()
            assert not (req_dir / PACKED_BIN_NAME).exists()

            got = read_request(req_dir)
            assert set(got) == {(layer, "post_block") for layer in range(4)}
            for layer in range(4):
                np.testing.assert_array_equal(
                    got[(layer, "post_block")].array, tensors[layer].numpy()
                )
        finally:
            c0.shutdown(timeout=5.0)
            c1.shutdown(timeout=5.0)

    def test_stage_owning_no_layers_writes_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        # A request whose layers all live on stage 0: stage 1 must create no
        # packed state and write no file (the manager drops the consumer for
        # this request on that stage).
        req = "req-pp-skip"
        c1 = _pp_consumer(tmp_path, pp_size=2, pp_rank=1, total_layers=4)
        try:
            spec = c1.validate_client_spec(
                _pp_raw(req, {"post_block": [0, 1]}), _ctx(req)
            )
            assert isinstance(spec, CaptureSpec)
            # No accumulation state for a stage that owns none of the layers.
            assert req not in c1._packed_states
        finally:
            c1.shutdown(timeout=5.0)

    def test_expected_keys_filtered_to_local_slice(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Stage 1 ([2,4)) given a global spec must track only its own layers,
        # else the packed index would wait forever on stage 0's layers.
        req = "req-pp-expected"
        c1 = _pp_consumer(tmp_path, pp_size=2, pp_rank=1, total_layers=4)
        try:
            c1.validate_client_spec(_pp_raw(req, {"post_block": [0, 1, 2, 3]}), _ctx(req))
            state = c1._packed_states[req]
            assert state.expected_keys == {(2, "post_block"), (3, "post_block")}
        finally:
            c1.shutdown(timeout=5.0)
