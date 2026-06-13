# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem round-trip + validation for DeepSeek-V4 mHC capture hooks.

mHC hooks differ from the standard residual hooks in two ways the on-disk
layout must preserve: width (the multi-stream residual is ``hc_mult`` times
wider) and dtype (the Sinkhorn ``res_mix`` / ``post_mix`` coefficients are
fp32, not the bf16 model dtype). The sidecar therefore records a per-row
``row_shape`` and a per-entry ``dtype`` so the reader reshapes each row back
to its logical shape and decodes the right dtype — including when a single
request packs hooks of mixed dtype into one file.
"""

from __future__ import annotations

import pathlib
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.reader import read_per_file, read_request
from vllm.v1.capture.consumers.filesystem.types import FilesystemCaptureRequest
from vllm.v1.capture.consumers.filesystem.validation import (
    validate_filesystem_request,
)
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    VllmInternalRequestId,
    build_hook_schema,
)

HC_MULT = 4
HIDDEN = 8
STREAM_W = HC_MULT * HIDDEN
RES_MIX_W = HC_MULT * HC_MULT


def _ctx(req_id: str, *, hc_mult: int | None = HC_MULT) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId(req_id),
        num_prompt_tokens=10,
        num_computed_tokens=0,
        num_hidden_layers=4,
        hidden_size=HIDDEN,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        hook_schema=build_hook_schema(HIDDEN, torch.bfloat16, hc_mult),
    )


def _consumer(tmp_path: pathlib.Path) -> FilesystemConsumer:
    return FilesystemConsumer(vllm_config=MagicMock(), params={"root": str(tmp_path)})


def _chunk(
    req_id: str,
    layer: int,
    hook: str,
    tensor: torch.Tensor,
    row_shape: tuple[int, ...],
) -> CaptureChunk:
    return CaptureChunk(
        key=(VllmInternalRequestId(req_id), layer, hook),
        tensor=tensor,
        dtype=tensor.dtype,
        row_offset=0,
        step_index=0,
        metadata={"row_shape": row_shape},
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMhcValidation:
    def test_rejects_mhc_hook_on_non_mhc_model(self, tmp_path: pathlib.Path) -> None:
        raw = FilesystemCaptureRequest(
            request_id="r", tag="t", hooks={"mhc_attn_res_mix": [0]}, positions="all"
        )
        # A non-mHC model's schema has no mhc_* hooks → rejected.
        with pytest.raises(CaptureValidationError, match="not a hook point this model"):
            validate_filesystem_request(raw, MagicMock(), _ctx("r", hc_mult=None))

    def test_accepts_mhc_hook_on_mhc_model(self, tmp_path: pathlib.Path) -> None:
        raw = FilesystemCaptureRequest(
            request_id="r", tag="t", hooks={"mhc_attn_res_mix": [0]}, positions="all"
        )
        spec = validate_filesystem_request(raw, MagicMock(), _ctx("r"))
        assert spec.hooks["mhc_attn_res_mix"] == [0]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestMhcRoundTrip:
    def test_per_file_fp32_res_mix(self, tmp_path: pathlib.Path) -> None:
        """fp32 Sinkhorn matrix round-trips with its (hc_mult, hc_mult) shape."""
        req = "r-pf"
        c = _consumer(tmp_path)
        try:
            raw = FilesystemCaptureRequest(
                request_id=req,
                tag="t",
                hooks={"mhc_attn_res_mix": [0]},
                positions="last_prompt",
                layout="per_file",
            )
            c.validate_client_spec(raw, _ctx(req))
            res = torch.arange(RES_MIX_W, dtype=torch.float32).reshape(1, RES_MIX_W)
            c.submit_chunk(_chunk(req, 0, "mhc_attn_res_mix", res, (HC_MULT, HC_MULT)))
            c.submit_finalize(_finalize(req, 0, "mhc_attn_res_mix"))

            key: CaptureKey = (VllmInternalRequestId(req), 0, "mhc_attn_res_mix")
            assert _wait(c, key).status == "ok"

            entry = read_per_file(tmp_path / "t" / req / "0_mhc_attn_res_mix.bin")
            assert entry.dtype == "float32"
            assert entry.array.shape == (1, HC_MULT, HC_MULT)
            np.testing.assert_array_equal(
                entry.array, res.reshape(1, HC_MULT, HC_MULT).numpy()
            )
        finally:
            c.shutdown(timeout=5.0)

    def test_packed_mixed_dtype_streams_and_coeffs(
        self, tmp_path: pathlib.Path
    ) -> None:
        """One packed request mixing bf16 streams + fp32 coefficients.

        The packed file holds both dtypes; the per-entry dtype + row_shape
        let the reader decode and reshape each hook correctly.
        """
        req = "r-pk"
        c = _consumer(tmp_path)
        try:
            raw = FilesystemCaptureRequest(
                request_id=req,
                tag="t",
                hooks={"mhc_streams_pre_attn": [1], "mhc_attn_res_mix": [0]},
                positions="last_prompt",
                layout="packed",
            )
            c.validate_client_spec(raw, _ctx(req))

            streams = torch.arange(STREAM_W, dtype=torch.bfloat16).reshape(1, STREAM_W)
            res = (torch.arange(RES_MIX_W, dtype=torch.float32) + 0.5).reshape(
                1, RES_MIX_W
            )
            c.submit_chunk(
                _chunk(req, 1, "mhc_streams_pre_attn", streams, (HC_MULT, HIDDEN))
            )
            c.submit_chunk(_chunk(req, 0, "mhc_attn_res_mix", res, (HC_MULT, HC_MULT)))
            c.submit_finalize(_finalize(req, 1, "mhc_streams_pre_attn"))
            c.submit_finalize(_finalize(req, 0, "mhc_attn_res_mix"))

            key: CaptureKey = (VllmInternalRequestId(req), 0, "mhc_attn_res_mix")
            assert _wait(c, key).status == "ok"

            got = read_request(tmp_path / "t" / req)
            assert set(got) == {(1, "mhc_streams_pre_attn"), (0, "mhc_attn_res_mix")}

            res_entry = got[(0, "mhc_attn_res_mix")]
            assert res_entry.dtype == "float32"
            assert res_entry.array.shape == (1, HC_MULT, HC_MULT)
            np.testing.assert_array_equal(
                res_entry.array, res.reshape(1, HC_MULT, HC_MULT).numpy()
            )

            stream_entry = got[(1, "mhc_streams_pre_attn")]
            # bf16 comes back as uint16 (its on-disk representation), reshaped.
            assert stream_entry.dtype == "bfloat16"
            assert stream_entry.array.shape == (1, HC_MULT, HIDDEN)
            recovered = torch.from_numpy(stream_entry.array.copy()).view(torch.bfloat16)
            torch.testing.assert_close(recovered, streams.reshape(1, HC_MULT, HIDDEN))
        finally:
            c.shutdown(timeout=5.0)
