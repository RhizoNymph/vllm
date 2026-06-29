# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client request id attribution surfaced into the capture sidecar.

These CPU-only tests pin the end-to-end plumbing of the original
client-supplied request id (``EngineCoreRequest.external_req_id``) through
the scheduler ``Request`` → ``NewRequestData`` → capture sidecar
(``client_request_id``) → filesystem consumer JSON. The capture runner
mixin and engine-core machinery require CUDA / heavy deps, so the worker
registration step is exercised by replicating the sidecar it builds.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import CaptureSpec, VllmInternalRequestId
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str,
    external_req_id: str | None,
) -> Request:
    """Build a minimal generative scheduler ``Request`` (no heavy config)."""
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        external_req_id=external_req_id,
    )


def _engine_core_request(request_id: str, external_req_id: str | None):
    """Build a stand-in carrying only the fields ``from_engine_core_request``
    reads, so we avoid the heavy ``EngineCoreRequest`` constructor."""
    from types import SimpleNamespace

    return SimpleNamespace(
        request_id=request_id,
        external_req_id=external_req_id,
        client_index=0,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        prompt_is_token_ids=None,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        priority=0,
        trace_headers=None,
        resumable=False,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
        abort_immediately=False,
        request_metadata=None,
    )


def _capture_sidecar_fields(new_req_data: NewRequestData) -> dict:
    """Replicate the sidecar ``capture_runner_mixin._capture_add_request``
    builds at registration time (the CUDA-bound mixin can't run on CPU)."""
    return {
        "vllm_internal_request_id": new_req_data.req_id,
        "client_request_id": (
            new_req_data.client_request_id
            if new_req_data.client_request_id is not None
            else new_req_data.req_id
        ),
        "request_id_slug": new_req_data.req_id,
        "tag_slug": "default",
    }


class _FakeVllmConfig:
    def __init__(self) -> None:
        self.capture_consumers_config = None


def _wait_for_status(consumer, key, *, timeout: float = 5.0) -> None:
    capture_key = (VllmInternalRequestId(key[0]), key[1], key[2])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = consumer.get_result(capture_key)
        if result is not None and result.status in ("ok", "error", "partial_error"):
            return
        time.sleep(0.005)
    pytest.fail(f"timeout waiting for {key} to finalize")


# ---------------------------------------------------------------------------
# 1. Request carries external_req_id
# ---------------------------------------------------------------------------


def test_request_from_engine_core_copies_external_req_id() -> None:
    eng = _engine_core_request("client-7-ab12cd34", "client-7")
    req = Request.from_engine_core_request(eng, block_hasher=None)
    assert req.external_req_id == "client-7"
    assert req.request_id == "client-7-ab12cd34"


def test_request_external_req_id_defaults_none() -> None:
    req = _make_request("internal-only", external_req_id=None)
    assert req.external_req_id is None


# ---------------------------------------------------------------------------
# 2. NewRequestData carries the client id
# ---------------------------------------------------------------------------


def test_new_request_data_carries_client_request_id() -> None:
    req = _make_request("client-7-ab12cd34", external_req_id="client-7")
    nrd = NewRequestData.from_request(req, block_ids=([],))
    assert nrd.req_id == "client-7-ab12cd34"
    assert nrd.client_request_id == "client-7"


def test_new_request_data_client_id_none_safe() -> None:
    req = _make_request("internal-only", external_req_id=None)
    nrd = NewRequestData.from_request(req, block_ids=([],))
    assert nrd.client_request_id is None


def test_new_request_data_client_id_equals_internal_when_randomization_off() -> None:
    # With VLLM_DISABLE_REQUEST_ID_RANDOMIZATION the internal id is not
    # suffixed, so external == internal.
    req = _make_request("client-7", external_req_id="client-7")
    nrd = NewRequestData.from_request(req, block_ids=([],))
    assert nrd.client_request_id == nrd.req_id == "client-7"


# ---------------------------------------------------------------------------
# 3. Capture sidecar contains client_request_id (always present)
# ---------------------------------------------------------------------------


def test_capture_sidecar_contains_client_request_id() -> None:
    req = _make_request("client-9-ffeedd00", external_req_id="client-9")
    nrd = NewRequestData.from_request(req, block_ids=([],))
    sidecar = _capture_sidecar_fields(nrd)
    assert sidecar["client_request_id"] == "client-9"
    assert sidecar["vllm_internal_request_id"] == "client-9-ffeedd00"


def test_capture_sidecar_client_id_falls_back_to_internal() -> None:
    req = _make_request("internal-only", external_req_id=None)
    nrd = NewRequestData.from_request(req, block_ids=([],))
    sidecar = _capture_sidecar_fields(nrd)
    # None-safe: falls back to the internal id so attribution always works.
    assert sidecar["client_request_id"] == "internal-only"


# ---------------------------------------------------------------------------
# 4. Filesystem consumer JSON metadata includes client_request_id
# ---------------------------------------------------------------------------


def test_filesystem_json_includes_client_request_id(tmp_path: pathlib.Path) -> None:
    consumer = FilesystemConsumer(
        vllm_config=_FakeVllmConfig(),
        params={"root": str(tmp_path), "writer_threads": 2},
    )
    mgr = CaptureManager(
        consumers=(consumer,),
        consumer_specs=(None,),
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
        device="cpu",
    )

    internal_id = "client-9-ffeedd00"
    client_id = "client-9"

    # Drive registration with the sidecar the runner mixin would build.
    req = _make_request(internal_id, external_req_id=client_id)
    nrd = NewRequestData.from_request(req, block_ids=([],))
    sidecar_fields = _capture_sidecar_fields(nrd)

    mgr.register_request(
        internal_id,
        client_specs={0: CaptureSpec(hooks={"post_block": [1]}, positions="last_prompt")},
        num_prompt_tokens=3,
        sidecar_fields=sidecar_fields,
    )

    batch_view = CaptureBatchView(
        req_ids=[internal_id],
        num_prompt_tokens=[3],
        num_computed_tokens=[0],
        num_scheduled_tokens=[3],
        token_offsets=[0],
    )
    plan = mgr.build_step_plan(batch_view)
    mgr.on_hook(1, "post_block", torch.arange(24, dtype=torch.float32).reshape(3, 8))
    mgr.dispatch_step_captures(plan)
    mgr.finalize_request(internal_id)

    _wait_for_status(consumer, (internal_id, 1, "post_block"))
    consumer.shutdown()

    # Directory naming is unchanged (keyed by request_id_slug == internal id).
    sidecar_path = tmp_path / "default" / internal_id / "1_post_block.json"
    assert sidecar_path.exists(), f"missing sidecar {sidecar_path}"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["client_request_id"] == client_id
    assert sidecar["request_id"] == internal_id
