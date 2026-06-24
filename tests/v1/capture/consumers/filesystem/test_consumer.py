# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the filesystem capture consumer.

Exercises the full lifecycle: chunk submission, finalization, golden-path
byte-for-byte comparison with raw ``ActivationWriter``, validation
delegation, and shutdown forwarding.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import time
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.types import (
    FilesystemCaptureRequest,
)
from vllm.v1.capture.consumers.filesystem.writer import (
    ActivationWriter,
    FinalizeTask,
    WriteResult,
    WriteTask,
)
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureContext,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
    VllmInternalRequestId,
)

# Validation tests require pydantic (pulled in by vllm.config). Skip
# gracefully when running in a lightweight test environment.
_has_pydantic = importlib.util.find_spec("pydantic") is not None
_skip_no_pydantic = pytest.mark.skipif(
    not _has_pydantic,
    reason="pydantic not installed — validation tests require full vllm.config",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vllm_config(
    *,
    root_path: str | None = "/tmp/activations",
    max_bytes: int = 0,
) -> MagicMock:
    """Build a minimal mock ``VllmConfig`` for the filesystem consumer.

    ``root_path`` / ``max_bytes`` are accepted for backwards compatibility
    with older call sites but are no longer consulted by the validator;
    the consumer receives its ``root`` via its constructor ``params``.
    """
    del root_path, max_bytes
    return MagicMock()


def _make_context(
    *,
    request_id: str = "req-1",
    num_prompt_tokens: int = 10,
    num_computed_tokens: int = 0,
    num_hidden_layers: int = 32,
    hidden_size: int = 4096,
    element_size_bytes: int = 2,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 1,
) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId(request_id),
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        element_size_bytes=element_size_bytes,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        data_parallel_size=dp,
    )


def _make_consumer(
    tmp_path: pathlib.Path,
    **extra_params: object,
) -> FilesystemConsumer:
    """Create a ``FilesystemConsumer`` pointed at ``tmp_path``."""
    params: dict[str, object] = {"root": str(tmp_path)}
    params.update(extra_params)
    return FilesystemConsumer(
        vllm_config=_make_vllm_config(root_path=str(tmp_path)),
        params=params,
    )


def test_root_expanduser(tmp_path, monkeypatch) -> None:
    """A ``~`` in the configured root expands to ``$HOME``.

    The shell does not expand ``~`` in
    ``--capture-consumers filesystem:root=~/path`` (it is not at a word
    boundary), so the consumer must ``expanduser`` it — otherwise captures
    land in a literal ``~`` directory under the server's cwd.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    consumer = FilesystemConsumer(
        vllm_config=_make_vllm_config(),
        params={"root": "~/caps"},
    )
    try:
        assert consumer._root == tmp_path / "caps"
        assert "~" not in str(consumer._root)
    finally:
        consumer.shutdown(timeout=5.0)


def _wait_for_result(
    consumer: FilesystemConsumer,
    key: CaptureKey,
    *,
    timeout: float = 5.0,
) -> CaptureResult | None:
    """Poll ``get_result`` until a terminal status or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = consumer.get_result(key)
        if result is not None and result.status != "pending":
            return result
        time.sleep(0.01)
    return consumer.get_result(key)


def _wait_for_write_result(
    writer: ActivationWriter,
    key: tuple[str, int, str],
    *,
    timeout: float = 5.0,
) -> WriteResult | None:
    """Poll ``get_result`` on a raw writer until terminal or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = writer.get_result(key)
        if result is not None and result.status != "pending":
            return result
        time.sleep(0.01)
    return writer.get_result(key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicLifecycle:
    """Create a consumer, submit chunks, finalize, verify files."""

    def test_single_key_lifecycle(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            request_id = VllmInternalRequestId("req-abc")
            key: CaptureKey = (request_id, 0, "post_attn")

            # Two chunks for the same key.
            t1 = torch.randn(2, 8, dtype=torch.float32)
            t2 = torch.randn(3, 8, dtype=torch.float32)

            for step, (tensor, offset) in enumerate([(t1, 0), (t2, 2)]):
                consumer.submit_chunk(
                    CaptureChunk(
                        key=key,
                        tensor=tensor,
                        dtype=tensor.dtype,
                        row_offset=offset,
                        step_index=step,
                        metadata={
                            "tag_slug": "my-tag",
                            "request_id_slug": "req-abc",
                        },
                    )
                )

            consumer.submit_finalize(
                CaptureFinalize(
                    key=key,
                    sidecar={
                        "tag_slug": "my-tag",
                        "request_id_slug": "req-abc",
                    },
                )
            )

            result = _wait_for_result(consumer, key)
            assert result is not None
            assert result.status == "ok", f"Expected ok, got {result}"

            # Verify files exist.
            bin_path = tmp_path / "my-tag" / "req-abc" / "0_post_attn.bin"
            json_path = bin_path.with_suffix(".json")
            assert bin_path.exists(), f"{bin_path} does not exist"
            assert json_path.exists(), f"{json_path} does not exist"

            # Verify bin contents match concatenation of both tensors.
            expected_bytes = t1.numpy().tobytes() + t2.numpy().tobytes()
            actual_bytes = bin_path.read_bytes()
            assert actual_bytes == expected_bytes

            # Verify sidecar contains our metadata.
            sidecar = json.loads(json_path.read_text())
            assert sidecar["request_id"] == "req-abc"
            assert sidecar["layer"] == 0
            assert sidecar["hook"] == "post_attn"
        finally:
            consumer.shutdown(timeout=5.0)


class TestGoldenComparison:
    """Write through both FilesystemConsumer and raw ActivationWriter,
    compare byte-for-byte."""

    def test_golden_bytes(self, tmp_path: pathlib.Path) -> None:
        consumer_root = tmp_path / "consumer"
        writer_root = tmp_path / "writer"
        consumer_root.mkdir()
        writer_root.mkdir()

        consumer = _make_consumer(consumer_root)
        writer = ActivationWriter(writer_root, num_threads=1)

        try:
            request_id = "req-golden"
            layer = 5
            hook = "pre_attn"
            tag_slug = "golden-tag"
            request_id_slug = "req-golden"

            # Shared tensor data.
            t1 = torch.randn(4, 16, dtype=torch.float32)
            t2 = torch.randn(2, 16, dtype=torch.float32)
            payload1 = t1.numpy().tobytes()
            payload2 = t2.numpy().tobytes()

            # --- Consumer path ---
            cap_key: CaptureKey = (
                VllmInternalRequestId(request_id),
                layer,
                hook,
            )
            metadata = {
                "tag_slug": tag_slug,
                "request_id_slug": request_id_slug,
            }
            consumer.submit_chunk(
                CaptureChunk(
                    key=cap_key,
                    tensor=t1,
                    dtype=t1.dtype,
                    row_offset=0,
                    step_index=0,
                    metadata=metadata,
                )
            )
            consumer.submit_chunk(
                CaptureChunk(
                    key=cap_key,
                    tensor=t2,
                    dtype=t2.dtype,
                    row_offset=4,
                    step_index=1,
                    metadata=metadata,
                )
            )
            consumer.submit_finalize(CaptureFinalize(key=cap_key, sidecar=metadata))

            # --- Raw writer path ---
            writer_key = (request_id, layer, hook)
            bin_path = writer_root / tag_slug / request_id_slug / f"{layer}_{hook}.bin"
            sidecar_path = bin_path.with_suffix(".json")

            writer.submit(
                WriteTask(
                    path=bin_path,
                    payload=payload1,
                    append=True,
                    key=writer_key,
                )
            )
            writer.submit(
                WriteTask(
                    path=bin_path,
                    payload=payload2,
                    append=True,
                    key=writer_key,
                )
            )
            writer.submit(
                FinalizeTask(
                    bin_path=bin_path,
                    sidecar_path=sidecar_path,
                    sidecar_payload={
                        "request_id": request_id,
                        "layer": layer,
                        "hook": hook,
                    },
                    key=writer_key,
                )
            )

            # Wait for both.
            cap_result = _wait_for_result(consumer, cap_key)
            write_result = _wait_for_write_result(writer, writer_key)

            assert cap_result is not None and cap_result.status == "ok"
            assert write_result is not None and write_result.status == "ok"

            # Compare bin files byte-for-byte.
            consumer_bin = (
                consumer_root / tag_slug / request_id_slug / f"{layer}_{hook}.bin"
            )
            writer_bin = (
                writer_root / tag_slug / request_id_slug / f"{layer}_{hook}.bin"
            )
            assert consumer_bin.read_bytes() == writer_bin.read_bytes()
        finally:
            consumer.shutdown(timeout=5.0)
            writer.shutdown(timeout=5.0)


@_skip_no_pydantic
class TestValidateClientSpec:
    """Exercise the ``validate_client_spec`` delegation."""

    def test_valid_request_returns_capture_spec(self) -> None:
        vllm_config = _make_vllm_config()
        consumer = FilesystemConsumer(
            vllm_config=vllm_config,
            params={"root": "/tmp/test"},
        )
        try:
            ctx = _make_context()
            raw = FilesystemCaptureRequest(
                request_id="my-req",
                tag="my-tag",
                hooks={"post_attn": [0, 1, 2]},
                positions="last_prompt",
            )
            spec = consumer.validate_client_spec(raw, ctx)
            assert isinstance(spec, CaptureSpec)
            assert "post_attn" in spec.hooks
            assert spec.hooks["post_attn"] == [0, 1, 2]
        finally:
            consumer.shutdown(timeout=5.0)

    def test_dict_input_also_accepted(self) -> None:
        vllm_config = _make_vllm_config()
        consumer = FilesystemConsumer(
            vllm_config=vllm_config,
            params={"root": "/tmp/test"},
        )
        try:
            ctx = _make_context()
            raw_dict = {
                "request_id": "dict-req",
                "tag": "dict-tag",
                "hooks": {"pre_attn": "all"},
                "positions": "all_prompt",
            }
            spec = consumer.validate_client_spec(raw_dict, ctx)
            assert isinstance(spec, CaptureSpec)
            assert "pre_attn" in spec.hooks
            assert len(spec.hooks["pre_attn"]) == 32  # all layers
        finally:
            consumer.shutdown(timeout=5.0)

    @pytest.mark.parametrize(
        "tp,pp,ep,dp",
        [
            (2, 1, 1, 1),  # tensor parallel
            (1, 2, 1, 1),  # pipeline parallel
            (1, 1, 4, 1),  # expert parallel
            (1, 1, 1, 2),  # data parallel
            (2, 2, 4, 2),  # all axes together
        ],
    )
    def test_parallel_sizes_accepted_for_residual_hooks(
        self, tp: int, pp: int, ep: int, dp: int
    ) -> None:
        # The replicated residual hooks are supported under every
        # parallel axis — admission no longer rejects TP/PP/EP/DP > 1.
        vllm_config = _make_vllm_config()
        consumer = FilesystemConsumer(
            vllm_config=vllm_config,
            params={"root": "/tmp/test"},
        )
        try:
            ctx = _make_context(tp=tp, pp=pp, ep=ep, dp=dp)
            raw = FilesystemCaptureRequest(
                request_id="par-req",
                tag="par-tag",
                hooks={"post_block": [0, 1, 2]},
                positions="last_prompt",
            )
            spec = consumer.validate_client_spec(raw, ctx)
            assert isinstance(spec, CaptureSpec)
            assert spec.hooks["post_block"] == [0, 1, 2]
        finally:
            consumer.shutdown(timeout=5.0)

    @pytest.mark.parametrize("layout", ["packed", "sharded", "per_file"])
    def test_layouts_accepted_under_pipeline_parallelism(self, layout: str) -> None:
        # packed/sharded now split their per-request/per-tag files by
        # pipeline stage rank, so admission accepts them under PP rather
        # than rejecting (see test_packed.py / test_sharded.py for the
        # per-stage on-disk behavior).
        consumer = FilesystemConsumer(
            vllm_config=_make_vllm_config(),
            params={"root": "/tmp/test"},
        )
        try:
            ctx = _make_context(pp=2)
            raw = FilesystemCaptureRequest(
                request_id="pp-accepts",
                tag="pp-accepts",
                hooks={"post_block": [0, 1]},
                positions="last_prompt",
                layout=layout,
            )
            assert isinstance(consumer.validate_client_spec(raw, ctx), CaptureSpec)
        finally:
            consumer.shutdown(timeout=5.0)


@_skip_no_pydantic
class TestPPGeometry:
    """Validate ``_pp_geometry`` against the *real* ``ParallelConfig`` and
    the real ``ModelConfig.get_layers_start_end_indices`` layer-split logic
    — the production path the other PP tests stub with a fake config."""

    @staticmethod
    def _geom(pp: int, tp: int, rank: int, total_layers: int):
        from types import SimpleNamespace

        from vllm.config.model import ModelConfig
        from vllm.config.parallel import ParallelConfig
        from vllm.v1.capture.consumers.filesystem.consumer import _pp_geometry

        # Bind the real (unbound) ModelConfig method to a stub carrying only
        # the layer count it needs, so the real split arithmetic runs.
        real_split = ModelConfig.get_layers_start_end_indices

        class _ModelStub:
            def get_total_num_hidden_layers(self) -> int:
                return total_layers

            def get_layers_start_end_indices(self, parallel_config: object):
                return real_split(self, parallel_config)

        # distributed_executor_backend="ray" skips ParallelConfig's
        # single-node GPU-count validation, so the test runs on a box with
        # fewer GPUs than world_size (e.g. CI / a 1-GPU node).
        pc = ParallelConfig(
            pipeline_parallel_size=pp,
            tensor_parallel_size=tp,
            distributed_executor_backend="ray",
        )
        pc.rank = rank
        cfg = SimpleNamespace(parallel_config=pc, model_config=_ModelStub())
        return _pp_geometry(cfg)

    def test_pp_disabled_is_inert(self) -> None:
        # pp_size == 1 → legacy filenames, no local-range filtering.
        assert self._geom(pp=1, tp=1, rank=0, total_layers=4) == (1, 0, None)

    def test_two_stage_tp1_splits_layers(self) -> None:
        assert self._geom(pp=2, tp=1, rank=0, total_layers=4) == (2, 0, (0, 2))
        assert self._geom(pp=2, tp=1, rank=1, total_layers=4) == (2, 1, (2, 4))

    def test_tp_replicas_share_their_stage_slice(self) -> None:
        # stage = (rank // tp) % pp: ranks 0,1 are stage 0; ranks 2,3 stage 1.
        assert self._geom(pp=2, tp=2, rank=0, total_layers=4) == (2, 0, (0, 2))
        assert self._geom(pp=2, tp=2, rank=1, total_layers=4) == (2, 0, (0, 2))
        assert self._geom(pp=2, tp=2, rank=2, total_layers=4) == (2, 1, (2, 4))
        assert self._geom(pp=2, tp=2, rank=3, total_layers=4) == (2, 1, (2, 4))

    def test_four_stage_even_split(self) -> None:
        for rank in range(4):
            pp_size, pp_rank, rng = self._geom(pp=4, tp=1, rank=rank, total_layers=32)
            assert (pp_size, pp_rank) == (4, rank)
            assert rng == (rank * 8, rank * 8 + 8)

    def test_non_config_falls_back_to_legacy(self) -> None:
        from vllm.v1.capture.consumers.filesystem.consumer import _pp_geometry

        # A MagicMock (its attrs don't coerce to int) → pp disabled.
        assert _pp_geometry(MagicMock()) == (1, 0, None)


class TestGetResultLifecycle:
    """Verify the result transitions: None -> pending -> ok."""

    def test_none_before_any_submission(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            key: CaptureKey = (
                VllmInternalRequestId("unknown"),
                0,
                "post_attn",
            )
            assert consumer.get_result(key) is None
        finally:
            consumer.shutdown(timeout=5.0)

    def test_ok_after_finalize(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            request_id = VllmInternalRequestId("result-test")
            key: CaptureKey = (request_id, 1, "post_block")

            tensor = torch.randn(1, 4, dtype=torch.float32)
            consumer.submit_chunk(
                CaptureChunk(
                    key=key,
                    tensor=tensor,
                    dtype=tensor.dtype,
                    row_offset=0,
                    step_index=0,
                    metadata={
                        "tag_slug": "tag",
                        "request_id_slug": "result-test",
                    },
                )
            )
            consumer.submit_finalize(
                CaptureFinalize(
                    key=key,
                    sidecar={
                        "tag_slug": "tag",
                        "request_id_slug": "result-test",
                    },
                )
            )

            result = _wait_for_result(consumer, key)
            assert result is not None
            assert result.status == "ok"
            assert result.payload is not None
            assert isinstance(result.payload, list)
            assert len(result.payload) >= 1
        finally:
            consumer.shutdown(timeout=5.0)


class TestWaitForResult:
    """Verify the event-based ``wait_for_result`` method."""

    def test_returns_ok_after_finalize(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            request_id = VllmInternalRequestId("wait-ok")
            key: CaptureKey = (request_id, 2, "post_block")

            tensor = torch.randn(1, 4, dtype=torch.float32)
            consumer.submit_chunk(
                CaptureChunk(
                    key=key,
                    tensor=tensor,
                    dtype=tensor.dtype,
                    row_offset=0,
                    step_index=0,
                    metadata={"tag_slug": "t", "request_id_slug": "wait-ok"},
                )
            )
            consumer.submit_finalize(
                CaptureFinalize(
                    key=key,
                    sidecar={"tag_slug": "t", "request_id_slug": "wait-ok"},
                )
            )

            result = consumer.wait_for_result(key, timeout=5.0)
            assert result is not None
            assert result.status == "ok"
        finally:
            consumer.shutdown(timeout=5.0)

    def test_returns_immediately_if_already_terminal(
        self, tmp_path: pathlib.Path
    ) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            request_id = VllmInternalRequestId("wait-already-done")
            key: CaptureKey = (request_id, 0, "pre_attn")

            tensor = torch.zeros(1, 4, dtype=torch.float32)
            consumer.submit_chunk(
                CaptureChunk(
                    key=key,
                    tensor=tensor,
                    dtype=tensor.dtype,
                    row_offset=0,
                    step_index=0,
                    metadata={"tag_slug": "t", "request_id_slug": "wait-already-done"},
                )
            )
            consumer.submit_finalize(
                CaptureFinalize(
                    key=key,
                    sidecar={"tag_slug": "t", "request_id_slug": "wait-already-done"},
                )
            )

            # Poll until terminal so we know the result is already ready.
            ready = _wait_for_result(consumer, key)
            assert ready is not None and ready.status == "ok"

            # Second call via wait_for_result must return immediately (no timeout).
            t0 = time.monotonic()
            result = consumer.wait_for_result(key, timeout=5.0)
            elapsed = time.monotonic() - t0

            assert result is not None
            assert result.status == "ok"
            assert elapsed < 1.0, f"Expected immediate return, took {elapsed:.3f}s"
        finally:
            consumer.shutdown(timeout=5.0)

    def test_returns_none_on_timeout_for_unknown_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            key: CaptureKey = (VllmInternalRequestId("never-submitted"), 0, "post_attn")
            result = consumer.wait_for_result(key, timeout=0.05)
            # No WriteTask or FinalizeTask was ever submitted, so the
            # writer has no result row — get_result returns None.
            assert result is None
        finally:
            consumer.shutdown(timeout=5.0)


class TestShutdown:
    """Verify shutdown forwards to the underlying writer."""

    def test_shutdown_is_idempotent(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        consumer.shutdown(timeout=5.0)
        # Second call should not raise.
        consumer.shutdown(timeout=5.0)

    def test_global_capture_spec_is_none(self, tmp_path: pathlib.Path) -> None:
        consumer = _make_consumer(tmp_path)
        try:
            assert consumer.global_capture_spec() is None
        finally:
            consumer.shutdown(timeout=5.0)


class TestClassVars:
    """Verify the ClassVar metadata on FilesystemConsumer."""

    def test_location_is_worker(self) -> None:
        assert FilesystemConsumer.location == "worker"

    def test_reads_client_spec_is_true(self) -> None:
        assert FilesystemConsumer.reads_client_spec is True
