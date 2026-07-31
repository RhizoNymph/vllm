# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for driver-side capture consumers via ``LLM``.

Drives a real engine with a pre-constructed ``location="driver"``
consumer instance (``LLM(capture_consumers=[instance])``). The engine
runs in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the driver
bridge's receiver delivers captures back to the caller's instance —
under engine-core multiprocessing the caller-side instance is a pickled
copy and never sees them.

Run: ``pytest tests/v1/capture/test_driver_consumer_e2e.py -v``
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Literal

import pytest
import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey, CaptureSpec

MODEL = "facebook/opt-125m"


class _E2ERecordingConsumer(CaptureConsumer):
    """A trivial driver consumer that records calls for assertion."""

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(self) -> None:
        self.captures: list[tuple[CaptureKey, torch.Tensor, dict[str, Any]]] = []

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks={"post_block": [0]}, positions="last_prompt")

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.captures.append((key, tensor.clone(), dict(sidecar)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_llm_with_driver_capture_consumer(monkeypatch):
    """``LLM(capture_consumers=[instance])`` wires a driver-side consumer
    so ``on_capture`` fires for each request."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM, SamplingParams
    from vllm.distributed import cleanup_dist_env_and_memory

    consumer = _E2ERecordingConsumer()
    try:
        llm = LLM(
            model=MODEL,
            enforce_eager=True,
            gpu_memory_utilization=0.25,
            capture_consumers=[consumer],
        )
    except OSError as exc:
        msg = str(exc).lower()
        if "connection error" in msg or "read timeout" in msg:
            pytest.skip(f"{MODEL} unavailable: {exc}")
        raise
    try:
        outputs = llm.generate(
            ["Hello world"],
            SamplingParams(max_tokens=8),
            use_tqdm=False,
        )
        assert len(outputs) == 1

        # A request's capture finalize is processed on a subsequent engine
        # step; the in-process engine idles once generate() returns, so pump
        # one throwaway generation to flush it. (Warmup traffic used to
        # provide this pumping as a side effect — and its garbage captures
        # satisfied this assertion — before warmup requests were excluded
        # from the capture pipeline.)
        llm.generate(["ping"], SamplingParams(max_tokens=1), use_tqdm=False)

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if consumer.captures:
                break
            time.sleep(0.2)
        assert len(consumer.captures) > 0
        assert all(
            not key[0].startswith("_warmup") for key, _t, _s in consumer.captures
        ), "warmup requests must not reach capture consumers"
        for captured_key, tensor, _sidecar in consumer.captures:
            _request_id, layer, hook = captured_key
            assert layer == 0
            assert hook == "post_block"
            assert tensor.ndim == 2
            assert tensor.shape[0] >= 1  # last_prompt: one row
            assert tensor.shape[1] > 0  # hidden_size
    finally:
        del llm
        cleanup_dist_env_and_memory()
