# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ``ActivationCaptureManager.finalize_request``.

Phase 4 stands up the finalize path entirely inside the capture manager:
no torch-cuda, no real model runner, no real filesystem writes. These
tests install a fake writer that records submits and returns canned
results so we can assert the finalize invariants end-to-end without a
GPU.

Coverage (binding list from the Phase 4 prompt):

1. Sidecar payload exposes the exact keys the spec requires.
2. Two (layer x hook) pairs produce two ``FinalizeTask``s.
3. Forced writer submit error → ``CaptureResult.status == "partial_error"``.
4. Atomic visibility: a finalize task carries both ``bin_path`` and
   ``sidecar_path`` referring to the same ``(layer, hook)`` tuple, and
   the payload is JSON-serializable on construction.
5. Terminal results flow onto ``ModelRunnerOutput.capture_results`` when
   they are written to a ``dict``.
6. Admission-time errors recorded via ``record_request_error`` surface
   as ``CaptureResult.status == "error"``.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import pytest
import torch

from vllm.config.activation_storing_types import (
    ActivationStoringSpec,
    CaptureResult,
)
from vllm.entrypoints.openai.activation_storing_validation import (
    ResolvedActivationStoringSpec,
)
from vllm.model_executor.layers.activation_capture import ActivationCaptureManager
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.activation_writer import (
    FinalizeTask,
    WriteError,
    WriteTask,
)


HIDDEN_SIZE = 8
NUM_HIDDEN_LAYERS = 32


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWriter:
    """Records ``submit`` calls without touching the filesystem.

    ``submit_raises`` lets tests force the next ``submit`` to raise a
    ``WriteError`` so we can exercise the partial-error path.
    """

    submitted: list = field(default_factory=list)
    submit_raises: dict = field(default_factory=dict)

    def submit(self, task) -> None:
        key = task.key
        if key in self.submit_raises:
            err = self.submit_raises.pop(key)
            self.submitted.append(("error", task, err))
            raise err
        self.submitted.append(("ok", task, None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager() -> ActivationCaptureManager:
    return ActivationCaptureManager(
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=torch.float32,
        device="cpu",
    )


def _make_resolved_spec(
    request_id: str,
    tag: str,
    hooks: dict[str, list[int]],
    positions: list[int] | str,
    position_kind: str,
) -> ResolvedActivationStoringSpec:
    raw = ActivationStoringSpec(
        request_id=request_id,
        tag=tag,
        hooks={hook: list(layers) for hook, layers in hooks.items()},
        positions=list(positions) if isinstance(positions, list) else positions,
    )
    return ResolvedActivationStoringSpec(
        request_id_slug=request_id,
        tag_slug=tag,
        hooks={hook: list(layers) for hook, layers in hooks.items()},
        positions=positions,
        position_kind=position_kind,
        estimated_bytes=0,
        raw=raw,
    )


def _register(
    mgr: ActivationCaptureManager,
    req_id: str,
    *,
    tag: str = "t",
    hooks: dict[str, list[int]] | None = None,
    positions: list[int] | str | None = None,
    position_kind: str = "last_prompt",
    num_prompt_tokens: int = 4,
) -> None:
    if hooks is None:
        hooks = {"post_mlp": [6]}
    if positions is None:
        positions = [num_prompt_tokens - 1]
    resolved = _make_resolved_spec(
        request_id=req_id,
        tag=tag,
        hooks=hooks,
        positions=positions,
        position_kind=position_kind,
    )
    mgr.register_request(
        req_id,
        resolved,
        num_prompt_tokens=num_prompt_tokens,
        model_name="test-org/test-model",
        model_dtype_str="float32",
        element_size_bytes=4,
        vllm_internal_request_id=f"internal-{req_id}",
        prompt_token_ids=[1, 2, 3, 4][:num_prompt_tokens],
    )


# ---------------------------------------------------------------------------
# 1. Sidecar payload keys
# ---------------------------------------------------------------------------


_REQUIRED_SIDECAR_KEYS = {
    "request_id",
    "tag",
    "model",
    "model_dtype",
    "layer",
    "hook",
    "shape",
    "dtype",
    "element_size",
    "positions",
    "position_kind",
    "last_prompt_token_index",
    "prompt_token_ids",
    "generated_token_ids",
    "created_at",
    "finalized_at",
    "vllm_internal_request_id",
    "capture_status",
    "capture_error",
}


def test_sidecar_payload_keys(tmp_path: pathlib.Path) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        tag="probe",
        hooks={"post_mlp": [6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])
    mgr.record_generated_token_ids("r0", [11, 22])

    writer = _FakeWriter()
    result = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    assert result.status == "ok"
    assert len(writer.submitted) == 1
    status, task, _ = writer.submitted[0]
    assert status == "ok"
    assert isinstance(task, FinalizeTask)
    payload = task.sidecar_payload
    assert set(payload.keys()) == _REQUIRED_SIDECAR_KEYS
    assert payload["request_id"] == "r0"
    assert payload["tag"] == "probe"
    assert payload["model"] == "test-org/test-model"
    assert payload["model_dtype"] == "float32"
    assert payload["layer"] == 6
    assert payload["hook"] == "post_mlp"
    assert payload["shape"] == [1, HIDDEN_SIZE]
    assert payload["dtype"] == "float32"
    assert payload["element_size"] == 4
    assert payload["positions"] == [3]
    assert payload["position_kind"] == "last_prompt"
    assert payload["last_prompt_token_index"] == 3
    assert payload["prompt_token_ids"] == [1, 2, 3, 4]
    assert payload["generated_token_ids"] == [11, 22]
    assert payload["vllm_internal_request_id"] == "internal-r0"
    assert payload["capture_status"] == "ok"
    assert payload["capture_error"] is None
    assert isinstance(payload["created_at"], str)
    assert isinstance(payload["finalized_at"], str)


# ---------------------------------------------------------------------------
# 2. One finalize task per (layer, hook)
# ---------------------------------------------------------------------------


def test_finalize_request_enqueues_finalize_task_per_layer_hook(
    tmp_path: pathlib.Path,
) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        hooks={"post_mlp": [3, 6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 3, "post_mlp", [3])
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    writer = _FakeWriter()
    result = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    assert result.status == "ok"
    # Two layers × one hook = two finalize tasks
    assert len(writer.submitted) == 2
    seen_keys = {task.key for _, task, _ in writer.submitted}
    assert seen_keys == {("r0", 3, "post_mlp"), ("r0", 6, "post_mlp")}

    # Each task has both bin_path and sidecar_path and they share the
    # same parent directory (spec path layout).
    for _, task, _ in writer.submitted:
        assert task.bin_path.suffix == ".bin"
        assert task.sidecar_path.suffix == ".json"
        assert task.bin_path.parent == task.sidecar_path.parent
        assert task.bin_path.stem == task.sidecar_path.stem
        assert tmp_path in task.bin_path.parents


# ---------------------------------------------------------------------------
# 3. Partial error from forced writer failure
# ---------------------------------------------------------------------------


def test_finalize_request_with_partial_error(tmp_path: pathlib.Path) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        hooks={"post_mlp": [3, 6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 3, "post_mlp", [3])
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    writer = _FakeWriter()
    # Make the layer-3 submit fail; layer-6 should still succeed.
    writer.submit_raises[("r0", 3, "post_mlp")] = WriteError(
        "forced failure for test",
        key=("r0", 3, "post_mlp"),
    )

    result = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    assert result.status == "partial_error"
    assert result.error is not None
    assert "forced failure for test" in result.error
    # Layer 6 succeeded: its paths should appear in the result.
    assert any(".bin" in p and "/6/" in p for p in result.paths)
    # Layer 3 failed before submission completed: its paths should NOT
    # appear (we don't echo paths for submits that threw).
    assert not any(".bin" in p and "/3/" in p for p in result.paths)


# ---------------------------------------------------------------------------
# 4. Atomic visibility: bin + sidecar ride on the same FinalizeTask
# ---------------------------------------------------------------------------


def test_atomic_visibility_between_bin_and_json(tmp_path: pathlib.Path) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        hooks={"pre_attn": [6], "post_mlp": [6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 6, "pre_attn", [3])
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    writer = _FakeWriter()
    result = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    assert result.status == "ok"
    # Each submitted FinalizeTask carries both paths so the writer can
    # rename atomically. The writer (Phase 2) enforces the rename order
    # on disk; here we only assert that the finalize contract is
    # providing both targets together.
    for _, task, _ in writer.submitted:
        assert task.bin_path is not None
        assert task.sidecar_path is not None
        assert task.bin_path.parent == task.sidecar_path.parent
    # Every task also carries a serializable payload; the writer will
    # re-validate on its side but we check the contract here.
    for _, task, _ in writer.submitted:
        import json

        json.dumps(task.sidecar_payload)


# ---------------------------------------------------------------------------
# 5. CaptureResult propagation onto ModelRunnerOutput.capture_results
# ---------------------------------------------------------------------------


def test_capture_status_propagates_to_model_runner_output(
    tmp_path: pathlib.Path,
) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        hooks={"post_mlp": [6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    writer = _FakeWriter()
    result = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
    output.capture_results["r0"] = result

    assert "r0" in output.capture_results
    assert output.capture_results["r0"] is result
    assert output.capture_results["r0"].status == "ok"


def test_model_runner_output_capture_results_defaults_empty() -> None:
    """An untouched ``ModelRunnerOutput`` exposes an empty dict — no None."""
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert output.capture_results == {}
    # Writing after the fact works and doesn't aliasing-bite the class
    # default.
    output.capture_results["r0"] = CaptureResult(status="ok", error=None, paths=[])
    fresh = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert fresh.capture_results == {}


# ---------------------------------------------------------------------------
# 6. Admission-time error surfaces as CaptureResult.status == "error"
# ---------------------------------------------------------------------------


def test_request_error_during_admission_surfaces_as_error_status(
    tmp_path: pathlib.Path,
) -> None:
    mgr = _make_manager()
    # Request r0: good admission, should still finalize successfully.
    _register(
        mgr,
        "r0",
        hooks={"post_mlp": [6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    # Request r1: admission-time error, never registered. Phase 4's
    # admission path calls ``record_request_error`` to stash the
    # message; finalize_request later returns status="error".
    mgr.record_request_error("r1", "layer 99 is out of range")

    writer = _FakeWriter()
    result_r0 = mgr.finalize_request("r0", writer=writer, root=tmp_path)
    result_r1 = mgr.finalize_request("r1", writer=writer, root=tmp_path)

    assert result_r0.status == "ok"
    assert result_r1.status == "error"
    assert result_r1.error == "layer 99 is out of range"
    assert result_r1.paths == []

    # r1's error did not bleed into r0's submissions.
    submitted_keys = {task.key for _, task, _ in writer.submitted}
    assert submitted_keys == {("r0", 6, "post_mlp")}


# ---------------------------------------------------------------------------
# Bonus: finalize twice is idempotent
# ---------------------------------------------------------------------------


def test_finalize_request_twice_is_no_double_submit(tmp_path: pathlib.Path) -> None:
    mgr = _make_manager()
    _register(
        mgr,
        "r0",
        hooks={"post_mlp": [6]},
        positions=[3],
        position_kind="last_prompt",
        num_prompt_tokens=4,
    )
    mgr.record_captured_rows("r0", 6, "post_mlp", [3])

    writer = _FakeWriter()
    first = mgr.finalize_request("r0", writer=writer, root=tmp_path)
    second = mgr.finalize_request("r0", writer=writer, root=tmp_path)

    assert first.status == "ok"
    # Second call should return an "error" terminal result with the
    # "never registered" message rather than re-submit to the writer.
    assert second.status == "error"
    assert second.error is not None
    assert "never registered" in second.error
    # Writer got exactly one task.
    assert len(writer.submitted) == 1


def test_write_task_and_finalize_task_import_roundtrip() -> None:
    """Smoke check that Phase 2 task types import cleanly from Phase 4."""
    assert WriteTask is not None
    assert FinalizeTask is not None
    assert WriteError is not None
