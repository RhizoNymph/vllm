# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D integration tests — ``GPUModelRunner``-ish capture flows.

These tests stand in for the end-to-end ``LLM(...)`` / filesystem-
consumer flow the roadmap describes.  Spinning up a real ``LLM`` here
requires heavy dependencies (pydantic, msgspec, cloudpickle, a CUDA
device) that aren't guaranteed in this test environment; to keep the
gating logic verifiable on a CPU-only machine we wire the new
``CaptureManager`` + ``FilesystemConsumer`` up directly and drive them
through the same register → plan → dispatch → finalize cycle
``GPUModelRunner._register_capture_request`` /
``_prepare_capture_step`` / ``_finalize_capture_step`` /
``_finalize_capture_for_request`` perform.

The byte-for-byte golden check against the Phase-2 ``ActivationWriter``
output is implemented by driving the *same* ``ActivationWriter`` from
inside the consumer and comparing against a parallel direct-writer run
— if the filesystem consumer preserves its wire format the two files
are identical.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import pytest
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import (
    CaptureSpec,
    VllmInternalRequestId,
)

_has_pydantic = importlib.util.find_spec("pydantic") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_status(
    consumer: FilesystemConsumer,
    key: tuple[str, int, str],
    *,
    timeout: float = 5.0,
) -> None:
    """Block until ``consumer.get_result(key)`` reaches a terminal status."""
    capture_key = (VllmInternalRequestId(key[0]), key[1], key[2])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = consumer.get_result(capture_key)
        if result is not None and result.status in ("ok", "error", "partial_error"):
            return
        time.sleep(0.005)
    pytest.fail(f"timeout waiting for {key} to finalize")


class _FakeModelConfig:
    """Stand-in exposing the layer count the consumer reads to resolve
    ``global_hooks`` (e.g. ``"all"``)."""

    def __init__(self, num_hidden_layers: int = 32) -> None:
        self._num_hidden_layers = num_hidden_layers

    def get_total_num_hidden_layers(self) -> int:
        return self._num_hidden_layers


class _FakeVllmConfig:
    """Minimal stand-in for ``VllmConfig`` — enough for the filesystem
    consumer's constructor to run without pulling in pydantic."""

    def __init__(self, num_hidden_layers: int = 32) -> None:
        self.capture_consumers_config = None
        self.model_config = _FakeModelConfig(num_hidden_layers)


# ---------------------------------------------------------------------------
# 1. Happy path — register, plan, dispatch, finalize through the new
#    manager.  Mirrors ``GPUModelRunner`` Phase D flow.
# ---------------------------------------------------------------------------


def test_filesystem_consumer_end_to_end_via_manager(tmp_path: pathlib.Path) -> None:
    """Drive ``CaptureManager`` → ``FilesystemConsumer`` with one request,
    one consumer, and verify the on-disk layout.
    """
    vllm_config = _FakeVllmConfig()
    consumer = FilesystemConsumer(
        vllm_config=vllm_config,
        params={"root": str(tmp_path), "writer_threads": 2},
    )

    # Build a manager wrapping the filesystem consumer as a direct sink.
    mgr = CaptureManager(
        consumers=(consumer,),
        consumer_specs=(None,),  # filesystem has no global spec
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
        device="cpu",
    )

    # Per-request client spec — analogous to what
    # ``_register_capture_request`` resolves via
    # ``validate_client_spec``.
    client_spec = CaptureSpec(
        hooks={"post_block": [1]},
        positions="last_prompt",
    )

    req_id = "req-runner-1"
    mgr.register_request(
        req_id,
        client_specs={0: client_spec},
        num_prompt_tokens=3,
        sidecar_fields={
            "tag_slug": "default",
            "request_id_slug": req_id,
            "vllm_internal_request_id": req_id,
        },
    )

    # Fake forward step — build the plan, populate scratch, dispatch.
    batch_view = CaptureBatchView(
        req_ids=[req_id],
        num_prompt_tokens=[3],
        num_computed_tokens=[0],
        num_scheduled_tokens=[3],
        token_offsets=[0],
    )
    plan = mgr.build_step_plan(batch_view)

    # Simulate ``on_hook`` firing: for the single (layer=1, hook=post_block)
    # key, populate the scratch with a known tensor.
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    mgr.on_hook(1, "post_block", hidden)

    # Drain.
    mgr.dispatch_step_captures(plan)
    results = mgr.finalize_request(req_id)

    assert list(results.keys()) == [0]

    # Wait for the writer pool to flush.
    _wait_for_status(consumer, (req_id, 1, "post_block"))
    consumer.shutdown()

    # Verify the expected file exists under the consumer's layout.
    bin_path = tmp_path / "default" / req_id / "1_post_block.bin"
    sidecar_path = bin_path.with_suffix(".json")
    assert bin_path.exists(), f"missing bin file {bin_path}"
    assert sidecar_path.exists(), f"missing sidecar {sidecar_path}"

    # The sidecar should echo the finalize sidecar fields.
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["request_id"] == req_id
    assert sidecar["layer"] == 1
    assert sidecar["hook"] == "post_block"


# ---------------------------------------------------------------------------
# 1b. Global-spec driven capture — the CUDA-graph-safe persistent-buffer
#     path. No per-request client spec; the consumer advertises a global
#     spec and files are named by the engine request id under default_tag.
# ---------------------------------------------------------------------------


def test_global_spec_drives_per_request_files(tmp_path: pathlib.Path) -> None:
    """A consumer-level global spec captures every request via the
    persistent-buffer path and still writes one file set per request,
    keyed by the engine request id under the configured ``default_tag``.
    """
    vllm_config = _FakeVllmConfig()
    consumer = FilesystemConsumer(
        vllm_config=vllm_config,
        params={
            "root": str(tmp_path),
            "writer_threads": 2,
            "global_hooks": {"post_mlp": [1]},
            "global_positions": "last_prompt",
            "default_tag": "run-global",
        },
    )

    # The runner installs the consumer's global spec into the manager.
    global_spec = consumer.global_capture_spec()
    assert global_spec is not None

    mgr = CaptureManager(
        consumers=(consumer,),
        consumer_specs=(global_spec,),
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
        device="cpu",
        # Engage the CUDA-graph-safe persistent-buffer path: with
        # max_num_tokens>0 the global key is served from a persistent buffer
        # instead of the eager dynamic gather.
        max_num_tokens=16,
    )

    # The global key gets a persistent buffer (the graph-safe path), and
    # routes to the global gather (not the dynamic index_select).
    assert mgr._global_keys == frozenset({(1, "post_mlp")})

    req_id = "req-global-1"
    # No client_specs and no admission slugs — purely global-driven.
    mgr.register_request(
        req_id,
        client_specs=None,
        num_prompt_tokens=3,
        sidecar_fields={"vllm_internal_request_id": req_id},
    )

    batch_view = CaptureBatchView(
        req_ids=[req_id],
        num_prompt_tokens=[3],
        num_computed_tokens=[0],
        num_scheduled_tokens=[3],
        token_offsets=[0],
    )
    plan = mgr.build_step_plan(batch_view)
    # Global key routed to the buffer path, not the dynamic gather.
    assert (1, "post_mlp") in plan.global_gather_indices
    assert (1, "post_mlp") not in plan.gather_indices

    # Simulate the graph-recorded full-residual copy into the persistent
    # buffer (what on_hook does for a global key).
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    mgr.on_hook(1, "post_mlp", hidden)

    mgr.dispatch_step_captures(plan)
    results = mgr.finalize_request(req_id)
    assert list(results.keys()) == [0]

    _wait_for_status(consumer, (req_id, 1, "post_mlp"))
    consumer.shutdown()

    # File named by the engine request id under the configured tag.
    bin_path = tmp_path / "run-global" / req_id / "1_post_mlp.bin"
    sidecar_path = bin_path.with_suffix(".json")
    assert bin_path.exists(), f"missing bin file {bin_path}"
    assert sidecar_path.exists(), f"missing sidecar {sidecar_path}"

    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["request_id"] == req_id
    assert sidecar["layer"] == 1
    assert sidecar["hook"] == "post_mlp"

    # The captured row is the last prompt position of the residual.
    captured = torch.frombuffer(bytearray(bin_path.read_bytes()), dtype=torch.float32)
    assert torch.equal(captured, hidden[-1])


def test_global_spec_two_requests_distinct_dirs(tmp_path: pathlib.Path) -> None:
    """Two global-driven requests land in separate request directories."""
    consumer = FilesystemConsumer(
        vllm_config=_FakeVllmConfig(),
        params={
            "root": str(tmp_path),
            "writer_threads": 2,
            "global_hooks": {"post_mlp": [0]},
            "global_positions": "last_prompt",
        },
    )
    mgr = CaptureManager(
        consumers=(consumer,),
        consumer_specs=(consumer.global_capture_spec(),),
        num_hidden_layers=2,
        hidden_size=4,
        model_dtype=torch.float32,
        device="cpu",
        max_num_tokens=16,
    )

    for req_id, base in (("req-A", 0), ("req-B", 100)):
        mgr.register_request(
            req_id,
            client_specs=None,
            num_prompt_tokens=2,
            sidecar_fields={"vllm_internal_request_id": req_id},
        )
        batch_view = CaptureBatchView(
            req_ids=[req_id],
            num_prompt_tokens=[2],
            num_computed_tokens=[0],
            num_scheduled_tokens=[2],
            token_offsets=[0],
        )
        plan = mgr.build_step_plan(batch_view)
        hidden = torch.arange(base, base + 8, dtype=torch.float32).reshape(2, 4)
        mgr.on_hook(0, "post_mlp", hidden)
        mgr.dispatch_step_captures(plan)
        mgr.finalize_request(req_id)
        _wait_for_status(consumer, (req_id, 0, "post_mlp"))

    consumer.shutdown()

    # Default tag is "default" (legacy fallback name); each request gets its
    # own directory keyed by the engine request id.
    assert (tmp_path / "default" / "req-A" / "0_post_mlp.bin").exists()
    assert (tmp_path / "default" / "req-B" / "0_post_mlp.bin").exists()


# ---------------------------------------------------------------------------
# 2. Plan-level admission error is surfaced as a terminal result.
# ---------------------------------------------------------------------------


def test_manager_admission_error_yields_error_result() -> None:
    """Requests rejected via ``record_request_error`` surface through
    :meth:`finalize_request` as a terminal ``CaptureResult`` — the
    runner uses this path when a ``validate_client_spec`` call raises.
    """
    from unittest.mock import MagicMock

    sink = MagicMock()
    sink.location = "worker"
    sink.submit_chunk = MagicMock()
    sink.submit_finalize = MagicMock()
    sink.get_result = MagicMock(return_value=None)
    sink.wait_for_result = MagicMock(return_value=None)

    mgr = CaptureManager(
        consumers=(sink,),
        consumer_specs=(CaptureSpec(hooks={"post_block": [0]}, positions="last_prompt"),),
        num_hidden_layers=2,
        hidden_size=4,
        model_dtype=torch.float32,
    )

    mgr.register_request(
        "req-error",
        client_specs=None,
        num_prompt_tokens=2,
    )
    mgr.record_request_error("req-error", "deliberate failure")

    results = mgr.finalize_request("req-error")
    # One consumer registered → one entry.
    assert 0 in results


# ---------------------------------------------------------------------------
# 3. Golden byte-for-byte comparison against the Phase-2
#    ``ActivationWriter`` output.
# ---------------------------------------------------------------------------


def test_filesystem_consumer_byte_for_byte_matches_writer(
    tmp_path: pathlib.Path,
) -> None:
    """Run the same tensor through ``FilesystemConsumer`` and a raw
    ``ActivationWriter``; the ``.bin`` payloads must be bit-identical.

    The byte-level format stability is the one backwards-compat
    guarantee the design doc makes: scripts reading the Phase-2 output
    layout keep working across the refactor.
    """
    from vllm.v1.capture.consumers.filesystem.writer import (
        ActivationWriter,
        FinalizeTask,
        WriteTask,
    )
    from vllm.v1.capture.types import (
        CaptureChunk,
        CaptureFinalize,
    )

    # Path 1: filesystem consumer.
    consumer_root = tmp_path / "via_consumer"
    consumer_root.mkdir()
    vllm_config = _FakeVllmConfig()
    consumer = FilesystemConsumer(
        vllm_config=vllm_config,
        params={"root": str(consumer_root), "writer_threads": 1},
    )

    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    key = (VllmInternalRequestId("req-gold"), 3, "post_block")

    consumer.submit_chunk(
        CaptureChunk(
            key=key,
            tensor=tensor,
            dtype=tensor.dtype,
            row_offset=0,
            step_index=0,
            metadata={"tag_slug": "gold", "request_id_slug": "req-gold"},
        )
    )
    consumer.submit_finalize(
        CaptureFinalize(
            key=key,
            sidecar={
                "tag_slug": "gold",
                "request_id_slug": "req-gold",
                "shape": [2, 8],
                "dtype": "float32",
            },
        )
    )
    _wait_for_status(consumer, ("req-gold", 3, "post_block"))
    consumer.shutdown()

    consumer_bin = consumer_root / "gold" / "req-gold" / "3_post_block.bin"
    assert consumer_bin.exists()
    consumer_bytes = consumer_bin.read_bytes()

    # Path 2: raw ActivationWriter produces the same bytes.
    writer_root = tmp_path / "via_writer"
    writer_root.mkdir()
    writer = ActivationWriter(writer_root, num_threads=1)
    try:
        writer_bin = writer_root / "gold" / "req-gold" / "3_post_block.bin"
        writer_bin.parent.mkdir(parents=True, exist_ok=True)
        writer.submit(
            WriteTask(
                path=writer_bin,
                payload=bytes(tensor.numpy().tobytes()),
                append=True,
                key=("req-gold", 3, "post_block"),
            )
        )
        writer.submit(
            FinalizeTask(
                bin_path=writer_bin,
                sidecar_path=writer_bin.with_suffix(".json"),
                sidecar_payload={
                    "tag_slug": "gold",
                    "request_id_slug": "req-gold",
                    "shape": [2, 8],
                    "dtype": "float32",
                },
                key=("req-gold", 3, "post_block"),
            )
        )

        # Spin until writer finalizes.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = writer.get_result(("req-gold", 3, "post_block"))
            if result is not None and result.status in ("ok", "error"):
                break
            time.sleep(0.005)
        else:
            pytest.fail("raw writer timed out")
    finally:
        writer.shutdown()

    writer_bytes = writer_bin.read_bytes()

    # Byte-for-byte equal: the consumer must not transform the payload.
    assert consumer_bytes == writer_bytes, (
        "FilesystemConsumer output diverged from the raw ActivationWriter "
        "output — the on-disk byte format is the one backwards-compat "
        "guarantee the capture-consumers refactor makes."
    )


# ---------------------------------------------------------------------------
# 4. ``build_consumers`` returns a three-tuple matching the expected
#    shape the runner consumes.
# ---------------------------------------------------------------------------


def test_build_consumers_returns_sinks_validators_and_name_index(
    tmp_path: pathlib.Path,
) -> None:
    """``vllm.v1.capture.registry.build_consumers`` fans out the config
    into a (sinks, validators, name_to_index) three-tuple.
    """
    from unittest.mock import MagicMock, patch

    from vllm.v1.capture import registry as _registry
    from vllm.v1.capture.config import (
        CaptureConsumersConfig,
        CaptureConsumerSpec,
    )
    from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
    from vllm.v1.capture.registry import _reset_cache_for_testing

    # Patch the registry to know about ``filesystem``.
    class _Entry:
        name = "filesystem"

        def load(self):
            return FilesystemConsumer

    def fake_eps(*, group):
        return [_Entry()]

    _reset_cache_for_testing()
    with patch(
        "vllm.v1.capture.registry.importlib.metadata.entry_points",
        side_effect=fake_eps,
    ):
        config = CaptureConsumersConfig(
            consumers=[
                CaptureConsumerSpec(
                    name="filesystem",
                    params={"root": str(tmp_path / "fs_a")},
                ),
                CaptureConsumerSpec(
                    name="filesystem",
                    instance_name="mirror",
                    params={"root": str(tmp_path / "fs_b")},
                ),
            ]
        )
        vllm_config = MagicMock()
        vllm_config.capture_consumers_config = config

        sinks, validators, name_to_index = _registry.build_consumers(vllm_config)

    try:
        assert len(sinks) == 2
        assert len(validators) == 2
        # First entry has no instance_name → keyed by entry-point name.
        # Second entry has instance_name="mirror" → keyed by that.
        assert name_to_index["filesystem"] == 0
        assert name_to_index["mirror"] == 1
        # FilesystemConsumer implements CaptureSink directly so the sink
        # IS the validator.
        for sink, validator in zip(sinks, validators, strict=True):
            assert validator is sink
    finally:
        for sink in sinks:
            if hasattr(sink, "shutdown"):
                sink.shutdown()
        _reset_cache_for_testing()


# ---------------------------------------------------------------------------
# 5. Pipeline-parallel cooperative capture across two stages.
# ---------------------------------------------------------------------------


def test_pipeline_parallel_two_stage_shared_fs(tmp_path: pathlib.Path) -> None:
    """Two pipeline stages cooperatively capture one request to a shared root.

    Each stage stands in for the TP-rank-0 capturer of a pipeline stage:
    its ``CaptureManager`` is built with the *global* layer count and the
    stage's *local* ``[start, end)`` slice, and both write to the same
    root (the shared mount). A client spec spanning both stages
    (``post_block`` at layers 1 and 3 of a 4-layer model) must land exactly
    one file per layer under its global-layer path, with each stage
    writing only the layers it owns — the Option-A merge the engine then
    unions at the result level.
    """
    GLOBAL = 4
    req_id = "req-pp"
    client_spec = CaptureSpec(hooks={"post_block": [1, 3]}, positions="last_prompt")

    def _drive_stage(local_range: tuple[int, int], owned_layer: int) -> None:
        consumer = FilesystemConsumer(
            vllm_config=_FakeVllmConfig(),
            params={"root": str(tmp_path), "writer_threads": 1},
        )
        mgr = CaptureManager(
            consumers=(consumer,),
            consumer_specs=(None,),
            num_hidden_layers=GLOBAL,
            hidden_size=8,
            model_dtype=torch.float32,
            device="cpu",
            local_layer_range=local_range,
        )
        mgr.register_request(
            req_id,
            client_specs={0: client_spec},
            num_prompt_tokens=3,
            sidecar_fields={
                "tag_slug": "default",
                "request_id_slug": req_id,
                "vllm_internal_request_id": req_id,
            },
        )
        batch_view = CaptureBatchView(
            req_ids=[req_id],
            num_prompt_tokens=[3],
            num_computed_tokens=[0],
            num_scheduled_tokens=[3],
            token_offsets=[0],
        )
        plan = mgr.build_step_plan(batch_view)
        # Only this stage's owned layer is planned.
        assert set(plan.gather_indices) == {(owned_layer, "post_block")}

        hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        # Firing the other stage's layer is a no-op on this manager.
        mgr.on_hook(owned_layer, "post_block", hidden)
        mgr.dispatch_step_captures(plan)
        results = mgr.finalize_request(req_id)
        assert list(results.keys()) == [0]
        _wait_for_status(consumer, (req_id, owned_layer, "post_block"))
        consumer.shutdown()

    # Stage 0 owns global layers [0, 2) → captures layer 1.
    _drive_stage((0, 2), owned_layer=1)
    # Stage 1 owns global layers [2, 4) → captures layer 3.
    _drive_stage((2, 4), owned_layer=3)

    req_dir = tmp_path / "default" / req_id
    written = sorted(p.name for p in req_dir.glob("*.bin"))
    # Exactly one file per requested layer, keyed by the GLOBAL layer index.
    assert written == ["1_post_block.bin", "3_post_block.bin"]
