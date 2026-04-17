# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone smoke test for ActivationRewardProducer.

Runs the CaptureSink lifecycle end-to-end against a MagicMock
``VllmConfig`` and a tempfile reference vector. No engine required.

Usage: ``python test.py``
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

from activation_reward_producer import ActivationRewardProducer
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    VllmInternalRequestId,
)


HIDDEN = 128
NUM_LAYERS = 32


def _mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.get_hidden_size.return_value = HIDDEN
    cfg.model_config.hf_config.num_hidden_layers = NUM_LAYERS
    return cfg


def _ctx(tp: int = 1, pp: int = 1) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId("req-1"),
        num_prompt_tokens=16,
        num_computed_tokens=0,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        element_size_bytes=2,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


def _tempfile_vector(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


def test_payload_shape_and_lifecycle(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    producer = ActivationRewardProducer(
        _mock_config(),
        {
            "layer": 12,
            "hook": "post_mlp",
            "vector_path": str(vec_path),
            "position_slice": {"start": 2, "end": None, "stride": 1},
            "scale": 5.0,
            "nonlinearity": "tanh",
        },
    )

    # Validator returns the pinned spec.
    spec = producer.validate_client_spec({}, _ctx())
    assert spec.hooks == {"post_mlp": [12]}
    assert spec.positions == "all_generated"

    # Two chunks across two steps; total 6 rows; slice starts at 2.
    key = (VllmInternalRequestId("req-1"), 12, "post_mlp")
    chunk_a = CaptureChunk(
        key=key,
        tensor=torch.randn(3, HIDDEN),
        dtype=torch.float32,
        row_offset=0,
        step_index=0,
    )
    chunk_b = CaptureChunk(
        key=key,
        tensor=torch.randn(3, HIDDEN),
        dtype=torch.float32,
        row_offset=3,
        step_index=1,
    )
    producer.submit_chunk(chunk_a)
    producer.submit_chunk(chunk_b)
    producer.submit_finalize(CaptureFinalize(key=key))

    result = producer.get_result(key)
    assert result is not None, "no result surfaced"
    assert result.status == "ok", f"unexpected status: {result.status}"
    payload = result.payload
    assert set(payload.keys()) == {
        "reward",
        "cos",
        "act_norm",
        "num_positions",
        "status",
    }
    assert payload["status"] == "ok"
    assert payload["num_positions"] == 6 - 2, (
        "expected 4 positions after slice start=2 over 6 rows"
    )
    assert math.isfinite(payload["reward"])
    assert math.isfinite(payload["cos"])
    assert math.isfinite(payload["act_norm"])
    assert abs(payload["cos"]) <= 1.0 + 1e-6
    assert abs(payload["reward"]) <= math.tanh(5.0) + 1e-6


def test_empty_window_payload(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    producer = ActivationRewardProducer(
        _mock_config(),
        {
            "layer": 0,
            "hook": "post_mlp",
            "vector_path": str(vec_path),
            "position_slice": {"start": 100, "end": None, "stride": 1},
        },
    )
    key = (VllmInternalRequestId("short"), 0, "post_mlp")
    producer.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=torch.randn(4, HIDDEN),
            dtype=torch.float32,
            row_offset=0,
            step_index=0,
        )
    )
    producer.submit_finalize(CaptureFinalize(key=key))
    payload = producer.get_result(key).payload
    assert payload["status"] == "empty_window"
    assert payload["num_positions"] == 0
    assert math.isnan(payload["reward"])


def test_no_chunks_partial_error(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    producer = ActivationRewardProducer(
        _mock_config(),
        {"layer": 0, "hook": "post_mlp", "vector_path": str(vec_path)},
    )
    key = (VllmInternalRequestId("ghost"), 0, "post_mlp")
    producer.submit_finalize(CaptureFinalize(key=key))
    result = producer.get_result(key)
    assert result.status == "partial_error"
    assert result.error is not None


def test_non_empty_client_spec_rejected(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    producer = ActivationRewardProducer(
        _mock_config(),
        {"layer": 0, "hook": "post_mlp", "vector_path": str(vec_path)},
    )
    try:
        producer.validate_client_spec({"layer": 99}, _ctx())
    except CaptureValidationError as e:
        assert "empty per-request spec" in str(e)
    else:
        raise AssertionError("expected CaptureValidationError")


def test_tp_pp_rejected(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    producer = ActivationRewardProducer(
        _mock_config(),
        {"layer": 0, "hook": "post_mlp", "vector_path": str(vec_path)},
    )
    try:
        producer.validate_client_spec({}, _ctx(tp=2))
    except CaptureValidationError as e:
        assert "tensor_parallel_size=1" in str(e)
    else:
        raise AssertionError("expected CaptureValidationError for TP>1")


def test_bad_layer_rejected(tmp: Path) -> None:
    ref = torch.randn(HIDDEN)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    try:
        ActivationRewardProducer(
            _mock_config(),
            {"layer": NUM_LAYERS + 5, "hook": "post_mlp",
             "vector_path": str(vec_path)},
        )
    except ValueError as e:
        assert "out of range" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-range layer")


def test_vector_hidden_size_mismatch(tmp: Path) -> None:
    ref = torch.randn(HIDDEN + 1)
    vec_path = tmp / "ref.pt"
    _tempfile_vector(vec_path, ref)

    try:
        ActivationRewardProducer(
            _mock_config(),
            {"layer": 0, "hook": "post_mlp", "vector_path": str(vec_path)},
        )
    except ValueError as e:
        assert "hidden_size" in str(e)
    else:
        raise AssertionError("expected ValueError for hidden_size mismatch")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tests = [
            test_payload_shape_and_lifecycle,
            test_empty_window_payload,
            test_no_chunks_partial_error,
            test_non_empty_client_spec_rejected,
            test_tp_pp_rejected,
            test_bad_layer_rejected,
            test_vector_hidden_size_mismatch,
        ]
        for fn in tests:
            fn(tmp)
            print(f"ok  {fn.__name__}")
    print("all good")


if __name__ == "__main__":
    main()
