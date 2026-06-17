# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-row position recording + speculative-decode dedup.

The manager stamps each captured row with its absolute logical token
position. The filesystem consumer persists that to the sidecar so a reader
can map rows back to positions and, under speculative decoding, dedup: a
verify step captures all candidate positions including drafts that are
later rejected and re-forwarded, so a generated position can appear in
several rows. Rows are written in step order, so the last row for a
position is the canonical (accepted) one — :func:`latest_per_position`
collapses to it.

These tests are framework-general (standard ``post_mlp`` hook); the mHC
work just rides on the same machinery.
"""

from __future__ import annotations

import pathlib
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.reader import (
    latest_per_position,
    read_per_file,
    read_request,
)
from vllm.v1.capture.consumers.filesystem.types import FilesystemCaptureRequest
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    VllmInternalRequestId,
)

HIDDEN = 4
HOOK = "post_mlp"


def _ctx(req_id: str) -> CaptureContext:
    # No hook_schema → validation falls back to the standard wired hooks,
    # which include post_mlp.
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId(req_id),
        num_prompt_tokens=4,
        num_computed_tokens=0,
        num_hidden_layers=4,
        hidden_size=HIDDEN,
        element_size_bytes=4,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def _consumer(tmp_path: pathlib.Path) -> FilesystemConsumer:
    return FilesystemConsumer(vllm_config=MagicMock(), params={"root": str(tmp_path)})


def _rows(values: list[float]) -> torch.Tensor:
    """One row per value, broadcast across HIDDEN, so a row is identifiable."""
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 1).repeat(1, HIDDEN)


def _chunk(
    req_id: str, layer: int, positions: list[int], values: list[float], step: int
) -> CaptureChunk:
    return CaptureChunk(
        key=(VllmInternalRequestId(req_id), layer, HOOK),
        tensor=_rows(values),
        dtype=torch.float32,
        row_offset=0,
        step_index=step,
        metadata={"positions": positions},
    )


def _wait(consumer: FilesystemConsumer, key: CaptureKey, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = consumer.get_result(key)
        if r is not None and r.status != "pending":
            return r
        time.sleep(0.01)
    return consumer.get_result(key)


# Two verify steps over the same request: step 0 captures positions 5,6,7;
# 6 and 7 are then rejected and re-forwarded in step 1 alongside the new
# position 8. The canonical rows are pos5@step0 and pos6,7,8@step1.
_STEP0_POS, _STEP0_VAL = [5, 6, 7], [50.0, 60.0, 70.0]
_STEP1_POS, _STEP1_VAL = [6, 7, 8], [61.0, 71.0, 81.0]
_DEDUP_POS = [5, 6, 7, 8]
_DEDUP_VAL = [50.0, 61.0, 71.0, 81.0]  # last-write wins per position


class TestPositionsPerFile:
    def test_positions_recorded_and_dedup(self, tmp_path: pathlib.Path) -> None:
        req = "r-pf"
        c = _consumer(tmp_path)
        try:
            raw = FilesystemCaptureRequest(
                request_id=req,
                tag="t",
                hooks={HOOK: [0]},
                positions="all_generated",
                layout="per_file",
            )
            c.validate_client_spec(raw, _ctx(req))
            c.submit_chunk(_chunk(req, 0, _STEP0_POS, _STEP0_VAL, 0))
            c.submit_chunk(_chunk(req, 0, _STEP1_POS, _STEP1_VAL, 1))
            c.submit_finalize(
                CaptureFinalize(key=(VllmInternalRequestId(req), 0, HOOK))
            )

            key: CaptureKey = (VllmInternalRequestId(req), 0, HOOK)
            assert _wait(c, key).status == "ok"

            entry = read_per_file(tmp_path / "t" / req / f"0_{HOOK}.bin")
            # Raw capture keeps every candidate row, in write order.
            assert entry.positions == _STEP0_POS + _STEP1_POS
            assert entry.array.shape == (6, HIDDEN)

            deduped = latest_per_position(entry)
            assert deduped.positions == _DEDUP_POS
            np.testing.assert_array_equal(deduped.array[:, 0], np.array(_DEDUP_VAL))
        finally:
            c.shutdown(timeout=5.0)


class TestPositionsPacked:
    def test_positions_recorded_and_dedup(self, tmp_path: pathlib.Path) -> None:
        req = "r-pk"
        c = _consumer(tmp_path)
        try:
            raw = FilesystemCaptureRequest(
                request_id=req,
                tag="t",
                hooks={HOOK: [0]},
                positions="all_generated",
                layout="packed",
            )
            c.validate_client_spec(raw, _ctx(req))
            c.submit_chunk(_chunk(req, 0, _STEP0_POS, _STEP0_VAL, 0))
            c.submit_chunk(_chunk(req, 0, _STEP1_POS, _STEP1_VAL, 1))
            c.submit_finalize(
                CaptureFinalize(key=(VllmInternalRequestId(req), 0, HOOK))
            )

            key: CaptureKey = (VllmInternalRequestId(req), 0, HOOK)
            assert _wait(c, key).status == "ok"

            entry = read_request(tmp_path / "t" / req)[(0, HOOK)]
            assert entry.positions == _STEP0_POS + _STEP1_POS

            deduped = latest_per_position(entry)
            assert deduped.positions == _DEDUP_POS
            np.testing.assert_array_equal(deduped.array[:, 0], np.array(_DEDUP_VAL))
        finally:
            c.shutdown(timeout=5.0)


def test_latest_per_position_requires_positions() -> None:
    from vllm.v1.capture.consumers.filesystem.reader import CaptureEntry

    entry = CaptureEntry(
        layer=0, hook=HOOK, array=np.zeros((2, HIDDEN)), dtype="float32", positions=None
    )
    with pytest.raises(ValueError, match="no per-row positions"):
        latest_per_position(entry)
