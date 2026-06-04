# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm.v1.capture.manager.CaptureManager``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from types import SimpleNamespace

from vllm.v1.capture.manager import (
    CaptureManager,
    _aggregate_capture_results,
    merge_capture_results,
)
from vllm.v1.capture.plan import CaptureBatchView, StepCapturePlan
from vllm.v1.capture.types import (
    CaptureKey,
    CaptureResult,
    CaptureSpec,
    VllmInternalRequestId,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
HIDDEN_SIZE = 8
MODEL_DTYPE = torch.float32


def _make_sink(name: str = "sink") -> MagicMock:
    """Create a MagicMock that quacks like a CaptureSink."""
    sink = MagicMock()
    sink.location = "worker"
    sink.submit_chunk = MagicMock()
    # ``submit_chunk_batch`` is an optional sink method; the manager calls it
    # when present and otherwise falls back to per-chunk ``submit_chunk``.
    # A bare MagicMock would auto-expose it and divert the manager off the
    # fallback path these tests assert on, so model a submit_chunk-only sink.
    sink.submit_chunk_batch = None
    sink.submit_finalize = MagicMock()
    sink.get_result = MagicMock(return_value=None)
    sink.wait_for_result = MagicMock(return_value=None)
    sink.shutdown = MagicMock()
    return sink


def _make_manager(
    sinks: tuple[MagicMock, ...] | None = None,
    specs: tuple[CaptureSpec | None, ...] | None = None,
) -> tuple[CaptureManager, tuple[MagicMock, ...]]:
    """Build a CaptureManager with one or more mock sinks."""
    if sinks is None:
        sinks = (_make_sink(),)
    if specs is None:
        specs = (
            CaptureSpec(
                hooks={"post_mlp": [0, 1]},
                positions="last_prompt",
            ),
        ) * len(sinks)
    mgr = CaptureManager(
        consumers=sinks,
        consumer_specs=specs,
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=MODEL_DTYPE,
    )
    return mgr, sinks


def _batch_view(
    req_ids: list[str],
    num_prompt_tokens: list[int],
    num_computed_tokens: list[int],
    num_scheduled_tokens: list[int],
    token_offsets: list[int] | None = None,
) -> CaptureBatchView:
    """Convenience constructor for CaptureBatchView."""
    if token_offsets is None:
        # Auto-compute offsets as cumulative sum of scheduled tokens.
        offsets = []
        running = 0
        for n in num_scheduled_tokens:
            offsets.append(running)
            running += n
        token_offsets = offsets
    return CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=num_computed_tokens,
        num_scheduled_tokens=num_scheduled_tokens,
        token_offsets=token_offsets,
    )


def _populate_scratch(plan, hidden_size: int = HIDDEN_SIZE):
    """Simulate the forward pass populating scratch tensors.

    Allocates a deterministic scratch tensor for every ``(layer, hook)``
    in ``plan.gather_indices`` and stores it on the plan, mimicking what
    :meth:`CaptureManager.on_hook` does inside the compiled forward
    graph.  Tests can then call :meth:`dispatch_step_captures` and
    assert the right rows reached each sink.
    """
    for key, idx in plan.gather_indices.items():
        n_rows = idx.shape[0]
        scratch = torch.zeros((n_rows, hidden_size), dtype=MODEL_DTYPE)
        for r in range(n_rows):
            scratch[r] = torch.arange(hidden_size, dtype=MODEL_DTYPE) + r * 100
        plan.scratch_gpu[key] = scratch


# ---------------------------------------------------------------------------
# Single consumer with global spec
# ---------------------------------------------------------------------------


class TestSingleConsumerGlobalSpec:
    def test_register_build_dispatch_finalize(self):
        mgr, (sink,) = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)

        # The spec asks for post_mlp at layers [0, 1] and "last_prompt"
        # which is position 9 for a 10-token prompt.
        assert (0, "post_mlp") in plan.gather_indices
        assert (1, "post_mlp") in plan.gather_indices
        assert len(plan.entries) == 2  # one entry per layer

        for entry in plan.entries:
            assert entry.logical_pos == 9
            assert entry.consumer_mask & 1

        # Simulate forward pass.
        _populate_scratch(plan)
        mgr.dispatch_step_captures(plan)
        # Dispatch is async; wait for the dispatch thread to fan the
        # chunks out to the sink before asserting on call counts.
        mgr._drain_dispatch_queue()

        # Sink should have received two chunks (one per layer).
        assert sink.submit_chunk.call_count == 2

        # Finalize.
        results = mgr.finalize_request("r1")
        assert 0 in results
        assert sink.submit_finalize.call_count == 2  # one per (layer, hook)

    def test_no_entries_when_position_outside_window(self):
        """Position 9 is outside a decode window [10, 11)."""
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[10],
            num_scheduled_tokens=[1],
        )
        plan = mgr.build_step_plan(view)
        assert len(plan.entries) == 0
        assert len(plan.gather_indices) == 0


# ---------------------------------------------------------------------------
# Two consumers with overlapping global specs
# ---------------------------------------------------------------------------


class TestTwoConsumersOverlapping:
    def test_union_gather_both_dispatched(self):
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        spec = CaptureSpec(
            hooks={"post_mlp": [0]},
            positions="last_prompt",
        )

        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(spec, spec),
        )
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)

        # Union: only one entry for (layer=0, post_mlp, pos=9), but the
        # consumer_mask should have bits 0 and 1 set.
        assert len(plan.entries) == 1
        entry = plan.entries[0]
        assert entry.consumer_mask == 0b11

        _populate_scratch(plan)
        mgr.dispatch_step_captures(plan)
        mgr._drain_dispatch_queue()

        # Both sinks received a chunk.
        assert sink0.submit_chunk.call_count == 1
        assert sink1.submit_chunk.call_count == 1

    def test_different_layers_produce_separate_entries(self):
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        spec0 = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        spec1 = CaptureSpec(hooks={"post_mlp": [1]}, positions="last_prompt")

        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(spec0, spec1),
        )
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)

        # Two entries: (layer=0, post_mlp) for consumer 0,
        # (layer=1, post_mlp) for consumer 1.
        assert len(plan.entries) == 2
        masks = {e.layer: e.consumer_mask for e in plan.entries}
        assert masks[0] == 0b01  # only consumer 0
        assert masks[1] == 0b10  # only consumer 1


# ---------------------------------------------------------------------------
# Per-request client spec
# ---------------------------------------------------------------------------


class TestPerRequestClientSpec:
    def test_client_spec_overrides_global(self):
        sink = _make_sink()
        global_spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        client_spec = CaptureSpec(hooks={"pre_attn": [2]}, positions="all_prompt")
        mgr, _ = _make_manager(sinks=(sink,), specs=(global_spec,))

        # Register with a client spec that overrides consumer 0.
        mgr.register_request("r1", client_specs={0: client_spec}, num_prompt_tokens=5)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[0],
            num_scheduled_tokens=[5],
        )
        plan = mgr.build_step_plan(view)

        # Should use client spec: pre_attn at layer 2, all_prompt = [0..4].
        assert (2, "pre_attn") in plan.gather_indices
        assert (0, "post_mlp") not in plan.gather_indices
        assert len(plan.entries) == 5

    def test_client_spec_for_specific_consumer_only(self):
        """Only consumer 1 gets a client spec; consumer 0 uses global."""
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        global0 = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(global0, None),
        )

        client1 = CaptureSpec(hooks={"post_mlp": [0]}, positions="all_prompt")
        mgr.register_request(
            "r1",
            client_specs={1: client1},
            num_prompt_tokens=5,
        )

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[0],
            num_scheduled_tokens=[5],
        )
        plan = mgr.build_step_plan(view)

        # Consumer 0 wants position 4 (last_prompt), consumer 1 wants [0..4].
        # The union at (layer=0, post_mlp) should be [0, 1, 2, 3, 4].
        assert (0, "post_mlp") in plan.gather_indices
        assert len(plan.entries) == 5  # positions 0,1,2,3,4

        # Position 4 should have both consumers' bits.
        entry_pos4 = [e for e in plan.entries if e.logical_pos == 4]
        assert len(entry_pos4) == 1
        assert entry_pos4[0].consumer_mask == 0b11

        # Positions 0-3 should only have consumer 1's bit.
        for e in plan.entries:
            if e.logical_pos < 4:
                assert e.consumer_mask == 0b10


# ---------------------------------------------------------------------------
# Consumer isolation
# ---------------------------------------------------------------------------


class TestConsumerIsolation:
    def test_failing_submit_chunk_does_not_block_other_consumer(self):
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        sink0.submit_chunk.side_effect = RuntimeError("sink0 exploded")

        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(spec, spec),
        )
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)
        _populate_scratch(plan)

        # Should not raise.
        mgr.dispatch_step_captures(plan)
        mgr._drain_dispatch_queue()

        # sink0 was called (and raised), but sink1 still received its chunk.
        assert sink0.submit_chunk.call_count == 1
        assert sink1.submit_chunk.call_count == 1

    def test_failing_submit_finalize_does_not_block_other_consumer(self):
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        sink0.submit_finalize.side_effect = RuntimeError("finalize boom")

        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(spec, spec),
        )
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        # Should not raise.
        results = mgr.finalize_request("r1")

        # Both consumers should have results.
        assert 0 in results
        assert 1 in results

        # sink1 should have received its finalize.
        assert sink1.submit_finalize.call_count == 1


# ---------------------------------------------------------------------------
# Leak test: 1000 requests
# ---------------------------------------------------------------------------


class TestLeakTest:
    def test_1000_request_register_finalize_cleans_state(self):
        mgr, _ = _make_manager()

        for i in range(1000):
            req_id = f"r{i}"
            mgr.register_request(req_id, client_specs=None, num_prompt_tokens=10)
            mgr.finalize_request(req_id)

        # Internal state should be empty.
        assert not mgr.is_active()
        assert len(mgr._requests) == 0


# ---------------------------------------------------------------------------
# finalize_request returns dict of results per consumer
# ---------------------------------------------------------------------------


class TestFinalizeResults:
    def test_returns_dict_keyed_by_consumer_index(self):
        sink0 = _make_sink("sink0")
        sink1 = _make_sink("sink1")
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")

        # Make sink0 return a specific result.
        expected_key = (VllmInternalRequestId("r1"), 0, "post_mlp")
        sink0.wait_for_result.return_value = CaptureResult(
            key=expected_key, status="ok", payload={"path": "/tmp/test"}
        )
        sink1.wait_for_result.return_value = CaptureResult(
            key=expected_key,
            status="ok",
        )

        mgr, _ = _make_manager(
            sinks=(sink0, sink1),
            specs=(spec, spec),
        )
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        results = mgr.finalize_request("r1")
        assert 0 in results
        assert 1 in results
        assert results[0].status == "ok"
        assert results[0].payload == {"path": "/tmp/test"}

    def test_finalize_unknown_request_returns_empty(self):
        mgr, _ = _make_manager()
        results = mgr.finalize_request("nonexistent")
        assert results == {}

    def test_finalize_aggregates_all_keys_and_preserves_payloads(self):
        sink = _make_sink("sink0")
        spec = CaptureSpec(
            hooks={"post_mlp": [0, 1]},
            positions="last_prompt",
        )
        key0 = (VllmInternalRequestId("r1"), 0, "post_mlp")
        key1 = (VllmInternalRequestId("r1"), 1, "post_mlp")
        payload0 = {"path": "/tmp/layer0"}
        payload1 = {"path": "/tmp/layer1"}

        def _wait_for_result(key: CaptureKey, timeout: float) -> CaptureResult:
            assert timeout == 5.0
            if key == key0:
                return CaptureResult(key=key, status="ok", payload=payload0)
            if key == key1:
                return CaptureResult(key=key, status="ok", payload=payload1)
            raise AssertionError(f"unexpected key {key!r}")

        sink.wait_for_result.side_effect = _wait_for_result

        mgr, _ = _make_manager(sinks=(sink,), specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        results = mgr.finalize_request("r1")

        assert sink.submit_finalize.call_count == 2
        assert sink.wait_for_result.call_count == 2
        assert results[0].status == "ok"
        assert results[0].key == key0
        assert results[0].error is None
        assert results[0].payload == {
            key0: payload0,
            key1: payload1,
        }

    def test_finalize_uses_worst_key_result(self):
        sink = _make_sink("sink0")
        spec = CaptureSpec(
            hooks={"post_mlp": [0, 1]},
            positions="last_prompt",
        )
        key0 = (VllmInternalRequestId("r1"), 0, "post_mlp")
        key1 = (VllmInternalRequestId("r1"), 1, "post_mlp")

        def _wait_for_result(key: CaptureKey, timeout: float) -> CaptureResult:
            if key == key0:
                return CaptureResult(key=key, status="ok", payload="first")
            if key == key1:
                return CaptureResult(
                    key=key,
                    status="error",
                    error=f"boom at {key}",
                    payload="second",
                )
            raise AssertionError(f"unexpected key {key!r}")

        sink.wait_for_result.side_effect = _wait_for_result

        mgr, _ = _make_manager(sinks=(sink,), specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        result = mgr.finalize_request("r1")[0]

        assert result.status == "error"
        assert result.key == key1
        assert result.error is not None
        assert str(key1) in result.error
        assert result.payload == {
            key0: "first",
            key1: "second",
        }

    def test_finalize_timeout_becomes_error(self):
        sink = _make_sink("sink0")
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        key = (VllmInternalRequestId("r1"), 0, "post_mlp")
        sink.wait_for_result.return_value = None

        mgr, _ = _make_manager(sinks=(sink,), specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        result = mgr.finalize_request("r1")[0]

        assert result.status == "error"
        assert result.key == key
        assert result.error == f"finalize timed out for {key}"


class TestAggregateCaptureResults:
    def test_prefers_error_over_partial_error_over_ok(self):
        key_ok = (VllmInternalRequestId("r1"), 0, "post_mlp")
        key_partial = (VllmInternalRequestId("r1"), 1, "post_mlp")
        key_error = (VllmInternalRequestId("r1"), 2, "post_mlp")

        result = _aggregate_capture_results(
            [
                CaptureResult(key=key_ok, status="ok", payload="ok"),
                CaptureResult(
                    key=key_partial,
                    status="partial_error",
                    error="partial",
                    payload="partial",
                ),
                CaptureResult(
                    key=key_error,
                    status="error",
                    error="fatal",
                    payload="error",
                ),
            ]
        )

        assert result.status == "error"
        assert result.key == key_error
        assert result.error == "partial; fatal"
        assert result.payload == {
            key_ok: "ok",
            key_partial: "partial",
            key_error: "error",
        }

    def test_single_result_preserves_payload_shape(self):
        key = (VllmInternalRequestId("r1"), 0, "post_mlp")
        payload = ["/tmp/capture.bin"]

        result = _aggregate_capture_results(
            [CaptureResult(key=key, status="ok", payload=payload)]
        )

        assert result.status == "ok"
        assert result.key == key
        assert result.payload == payload


# ---------------------------------------------------------------------------
# unregister_request
# ---------------------------------------------------------------------------


class TestUnregisterRequest:
    def test_unregister_removes_state(self):
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        assert mgr.has_request("r1")

        mgr.unregister_request("r1")
        assert not mgr.has_request("r1")
        assert not mgr.is_active()

    def test_unregister_unknown_is_noop(self):
        mgr, _ = _make_manager()
        mgr.unregister_request("nonexistent")  # should not raise

    def test_finalize_after_unregister_returns_empty(self):
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.unregister_request("r1")

        results = mgr.finalize_request("r1")
        assert results == {}


# ---------------------------------------------------------------------------
# Position expansion
# ---------------------------------------------------------------------------


class TestPositionExpansion:
    def test_last_prompt(self):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)
        positions = [e.logical_pos for e in plan.entries]
        assert positions == [9]

    def test_all_prompt(self):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="all_prompt")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=5)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[0],
            num_scheduled_tokens=[5],
        )
        plan = mgr.build_step_plan(view)
        positions = sorted(e.logical_pos for e in plan.entries)
        assert positions == [0, 1, 2, 3, 4]

    def test_all_generated(self):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="all_generated")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=5)

        # Prefill step: positions [5..10) don't exist yet, 0 generated.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[0],
            num_scheduled_tokens=[5],
        )
        plan = mgr.build_step_plan(view)
        # No generated tokens in prefill.
        assert len(plan.entries) == 0

        # First decode step: position 5 is generated.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[5],
            num_scheduled_tokens=[1],
        )
        plan = mgr.build_step_plan(view)
        positions = [e.logical_pos for e in plan.entries]
        assert positions == [5]

    def test_all(self):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="all")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=3)

        # Prefill: all 3 prompt tokens.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[3],
            num_computed_tokens=[0],
            num_scheduled_tokens=[3],
        )
        plan = mgr.build_step_plan(view)
        positions = sorted(e.logical_pos for e in plan.entries)
        assert positions == [0, 1, 2]

        # First decode: position 3.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[3],
            num_computed_tokens=[3],
            num_scheduled_tokens=[1],
        )
        plan = mgr.build_step_plan(view)
        positions = [e.logical_pos for e in plan.entries]
        assert positions == [3]

    def test_explicit_list(self):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions=[2, 7])
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)
        positions = sorted(e.logical_pos for e in plan.entries)
        assert positions == [2, 7]


# ---------------------------------------------------------------------------
# Step window intersection
# ---------------------------------------------------------------------------


class TestStepWindowIntersection:
    def test_positions_outside_window_excluded(self):
        """Explicit list with some positions outside the current window."""
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions=[0, 5, 9])
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        # Window is [3, 7): only position 5 is inside.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[3],
            num_scheduled_tokens=[4],
        )
        plan = mgr.build_step_plan(view)
        positions = [e.logical_pos for e in plan.entries]
        assert positions == [5]

    def test_all_prompt_only_captures_scheduled_window(self):
        """all_prompt is [0..9] but window [0, 3) only captures 0,1,2."""
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="all_prompt")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[3],
        )
        plan = mgr.build_step_plan(view)
        positions = sorted(e.logical_pos for e in plan.entries)
        assert positions == [0, 1, 2]

    def test_decode_step_window(self):
        """During decode, the window is [N, N+1) for one token."""
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="all")
        mgr, _ = _make_manager(specs=(spec,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=5)

        # Decode step at position 7.
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[5],
            num_computed_tokens=[7],
            num_scheduled_tokens=[1],
        )
        plan = mgr.build_step_plan(view)
        positions = [e.logical_pos for e in plan.entries]
        assert positions == [7]


# ---------------------------------------------------------------------------
# record_request_error
# ---------------------------------------------------------------------------


class TestRecordRequestError:
    def test_error_surfaced_in_plan(self):
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        mgr.record_request_error("r1", "something went wrong")

        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)
        assert "r1" in plan.request_errors
        assert plan.request_errors["r1"] == "something went wrong"
        # No entries should be planned for the errored request.
        assert len(plan.entries) == 0

    def test_error_on_unknown_request_is_noop(self):
        mgr, _ = _make_manager()
        mgr.record_request_error("nonexistent", "boom")  # should not raise


# ---------------------------------------------------------------------------
# is_active and has_request
# ---------------------------------------------------------------------------


class TestActiveAndHasRequest:
    def test_is_active_with_requests(self):
        mgr, _ = _make_manager()
        assert not mgr.is_active()

        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        assert mgr.is_active()

        mgr.unregister_request("r1")
        assert not mgr.is_active()

    def test_has_request(self):
        mgr, _ = _make_manager()
        assert not mgr.has_request("r1")

        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        assert mgr.has_request("r1")

        mgr.unregister_request("r1")
        assert not mgr.has_request("r1")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_consumers_with_specs_skips_registration(self):
        """When all consumer_specs are None and no client_specs, register is a no-op."""
        sink = _make_sink()
        mgr, _ = _make_manager(sinks=(sink,), specs=(None,))
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        # Request should not be tracked.
        assert not mgr.has_request("r1")
        assert not mgr.is_active()

    def test_register_duplicate_raises(self):
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        with pytest.raises(ValueError, match="already registered"):
            mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

    def test_register_nonpositive_prompt_tokens_raises(self):
        mgr, _ = _make_manager()
        with pytest.raises(ValueError, match="non-positive"):
            mgr.register_request("r1", client_specs=None, num_prompt_tokens=0)

    def test_batch_view_length_mismatch_raises(self):
        mgr, _ = _make_manager()
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

        view = CaptureBatchView(
            req_ids=["r1"],
            num_prompt_tokens=[10, 20],  # length mismatch
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
            token_offsets=[0],
        )
        with pytest.raises(ValueError, match="list lengths"):
            mgr.build_step_plan(view)

    def test_dispatch_empty_plan_is_noop(self):
        mgr, (sink,) = _make_manager()
        plan = StepCapturePlan(
            gather_indices={},
            scratch_gpu={},
            scratch_dtype={},
            entries=[],
        )
        mgr.dispatch_step_captures(plan)
        assert sink.submit_chunk.call_count == 0

    def test_client_spec_out_of_range_raises(self):
        mgr, _ = _make_manager()
        with pytest.raises(ValueError, match="out of range"):
            mgr.register_request(
                "r1",
                client_specs={
                    99: CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
                },
                num_prompt_tokens=10,
            )

    def test_layer_out_of_range_raises(self):
        mgr, _ = _make_manager()
        with pytest.raises(ValueError, match="out of range"):
            mgr.register_request(
                "r1",
                client_specs={
                    0: CaptureSpec(hooks={"post_mlp": [999]}, positions="last_prompt")
                },
                num_prompt_tokens=10,
            )


# ---------------------------------------------------------------------------
# Pipeline-parallel local layer-range filtering
# ---------------------------------------------------------------------------

GLOBAL_LAYERS = 8


def _make_pp_manager(
    local_layer_range,
    spec: CaptureSpec,
) -> tuple[CaptureManager, MagicMock]:
    """A manager over an 8-layer model owning ``local_layer_range``."""
    sink = _make_sink()
    mgr = CaptureManager(
        consumers=(sink,),
        consumer_specs=(spec,),
        num_hidden_layers=GLOBAL_LAYERS,
        hidden_size=HIDDEN_SIZE,
        model_dtype=MODEL_DTYPE,
        local_layer_range=local_layer_range,
    )
    return mgr, sink


def _captured_layers(mgr: CaptureManager) -> set[int]:
    """Register 'r1' then build a plan; return the global layers planned."""
    mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
    if not mgr.has_request("r1"):
        return set()
    view = _batch_view(
        req_ids=["r1"],
        num_prompt_tokens=[10],
        num_computed_tokens=[0],
        num_scheduled_tokens=[10],
    )
    plan = mgr.build_step_plan(view)
    return {layer for (layer, _hook) in plan.gather_indices}


class TestLocalLayerRangeFiltering:
    def test_first_stage_keeps_only_its_layers(self):
        spec = CaptureSpec(hooks={"post_mlp": [2, 6]}, positions="last_prompt")
        mgr, sink = _make_pp_manager((0, 4), spec)
        assert _captured_layers(mgr) == {2}
        # Finalize touches only the in-range layer (layer 2), not layer 6.
        mgr.finalize_request("r1")
        finalized_layers = {
            call.args[0].key[1] for call in sink.submit_finalize.call_args_list
        }
        assert finalized_layers == {2}

    def test_second_stage_keeps_only_its_layers(self):
        spec = CaptureSpec(hooks={"post_mlp": [2, 6]}, positions="last_prompt")
        mgr, _ = _make_pp_manager((4, 8), spec)
        assert _captured_layers(mgr) == {6}

    def test_none_range_keeps_all_layers(self):
        spec = CaptureSpec(hooks={"post_mlp": [2, 6]}, positions="last_prompt")
        mgr, _ = _make_pp_manager(None, spec)
        assert _captured_layers(mgr) == {2, 6}

    def test_all_layers_out_of_local_range_inactive(self):
        spec = CaptureSpec(hooks={"post_mlp": [6, 7]}, positions="last_prompt")
        mgr, _ = _make_pp_manager((0, 4), spec)
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        # No requested layer lives on this stage → request not registered.
        assert not mgr.has_request("r1")
        assert mgr.finalize_request("r1") == {}

    def test_out_of_global_range_still_raises_per_stage(self):
        # A genuinely out-of-range layer is rejected even though it is also
        # outside this stage's local slice.
        spec = CaptureSpec(hooks={"post_mlp": [100]}, positions="last_prompt")
        mgr, _ = _make_pp_manager((0, 4), spec)
        with pytest.raises(ValueError, match="out of range"):
            mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)

    def test_partial_hook_layers_filtered(self):
        # Multiple hooks, each split across the stage boundary.
        spec = CaptureSpec(
            hooks={"post_mlp": [1, 5], "post_attn": [3, 7]},
            positions="last_prompt",
        )
        mgr, _ = _make_pp_manager((0, 4), spec)
        mgr.register_request("r1", client_specs=None, num_prompt_tokens=10)
        view = _batch_view(
            req_ids=["r1"],
            num_prompt_tokens=[10],
            num_computed_tokens=[0],
            num_scheduled_tokens=[10],
        )
        plan = mgr.build_step_plan(view)
        assert set(plan.gather_indices) == {(1, "post_mlp"), (3, "post_attn")}

    @pytest.mark.parametrize("bad_range", [(-1, 4), (4, 2), (0, 9)])
    def test_invalid_local_range_rejected(self, bad_range):
        spec = CaptureSpec(hooks={"post_mlp": [0]}, positions="last_prompt")
        with pytest.raises(ValueError, match="local_layer_range"):
            _make_pp_manager(bad_range, spec)


# ---------------------------------------------------------------------------
# Cross-rank capture_results merge (executor aggregation path)
# ---------------------------------------------------------------------------


def _result(req: str, layer: int, status: str = "ok", payload=None) -> CaptureResult:
    return CaptureResult(
        key=(VllmInternalRequestId(req), layer, "post_mlp"),
        status=status,
        payload=payload,
    )


def _fake_output(capture_results):
    # Duck-typed stand-in for ModelRunnerOutput (only .capture_results used).
    return SimpleNamespace(capture_results=capture_results)


class TestMergeCaptureResults:
    def test_disjoint_requests_unioned_into_output_rank(self):
        # rank 0 (stage 0) captured req-a; rank 1 (output_rank) captured req-b.
        stage0 = _fake_output({"req-a": {"filesystem": _result("req-a", 1)}})
        stage1 = _fake_output({"req-b": {"filesystem": _result("req-b", 3)}})
        outputs = [stage0, stage1]
        merge_capture_results(outputs, output_rank=1)
        assert set(outputs[1].capture_results) == {"req-a", "req-b"}
        assert "filesystem" in outputs[1].capture_results["req-a"]
        assert "filesystem" in outputs[1].capture_results["req-b"]

    def test_same_request_across_stages_aggregated(self):
        # Both stages captured different layers of the same request.
        stage0 = _fake_output(
            {"req": {"filesystem": _result("req", 1, payload=["a.bin"])}}
        )
        stage1 = _fake_output(
            {"req": {"filesystem": _result("req", 5, payload=["b.bin"])}}
        )
        outputs = [stage0, stage1]
        merge_capture_results(outputs, output_rank=1)
        merged = outputs[1].capture_results["req"]["filesystem"]
        # Aggregated payload keeps both stages' keys.
        assert merged.status == "ok"
        assert isinstance(merged.payload, dict)
        assert len(merged.payload) == 2

    def test_worst_status_wins(self):
        stage0 = _fake_output({"req": {"fs": _result("req", 1, status="error")}})
        stage1 = _fake_output({"req": {"fs": _result("req", 5, status="ok")}})
        outputs = [stage0, stage1]
        merge_capture_results(outputs, output_rank=1)
        assert outputs[1].capture_results["req"]["fs"].status == "error"

    def test_single_rank_result_passes_through_unchanged(self):
        only = _result("req", 2)
        stage0 = _fake_output({})  # non-capturing stage
        stage1 = _fake_output({"req": {"fs": only}})
        outputs = [stage0, stage1]
        merge_capture_results(outputs, output_rank=1)
        # Exactly one contributor → same object, not re-wrapped.
        assert outputs[1].capture_results["req"]["fs"] is only

    def test_none_outputs_skipped(self):
        stage0 = None
        stage1 = _fake_output({"req": {"fs": _result("req", 1)}})
        outputs = [stage0, stage1]
        merge_capture_results(outputs, output_rank=1)
        assert set(outputs[1].capture_results) == {"req"}

    def test_none_target_is_noop(self):
        outputs = [_fake_output({"req": {"fs": _result("req", 1)}}), None]
        # output_rank points at the None entry → no crash, no-op.
        merge_capture_results(outputs, output_rank=1)
        assert outputs[1] is None
