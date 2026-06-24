# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``capture_wait`` delivery path.

Capture writes finalize asynchronously, so a request's results may land
after its final token -- after the per-request output stream is closed.
The scheduler then has no ``EngineCoreOutput`` to carry them and instead
surfaces whatever remains in ``capture_results`` batch-level via
``EngineCoreOutputs.late_capture_results``. The output processor stashes
those late results so an ``AsyncLLM.wait_for_capture_results`` waiter can
collect them.

These tests cover the engine-internal pieces without spinning up a real
engine or model:

- ``EngineCoreOutputs`` carries the new ``late_capture_results`` field and
  round-trips it through msgspec (the wire that delivers late results).
- The scheduler's batch-level fallback logic: results still pending after
  per-request outputs are built are delivered batch-level (and ``pop``
  prevents double-delivery for the in-band case).
- ``OutputProcessor.process_outputs`` routes late results into the stash,
  sets any waiter event, indexes under both the suffixed and client-facing
  request id, and bounds the stash so uncollected entries cannot leak.
"""

from __future__ import annotations

import asyncio

import msgspec

from vllm.v1.capture.types import CaptureResult
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.output_processor import OutputProcessor


def _result(req_id: str = "r1") -> CaptureResult:
    return CaptureResult(
        key=(req_id, 0, "post_mlp"),
        status="ok",
        error=None,
        payload=["/tmp/a.bin", "/tmp/a.json"],
    )


# ---------------------------------------------------------------------------
# EngineCoreOutputs wire field
# ---------------------------------------------------------------------------


class TestEngineCoreOutputsField:
    def test_defaults_empty(self) -> None:
        assert EngineCoreOutputs().late_capture_results == {}

    def test_msgspec_roundtrip(self) -> None:
        late = {"r1": {"fs": _result("r1")}}
        out = EngineCoreOutputs(late_capture_results=late)
        wire = msgspec.msgpack.encode(out)
        decoded = msgspec.msgpack.decode(wire, type=EngineCoreOutputs)
        assert set(decoded.late_capture_results) == {"r1"}
        got = decoded.late_capture_results["r1"]["fs"]
        assert got.status == "ok"
        # Payloads cross msgspec as str (not Path), which the response
        # builder's ``_coerce`` then normalizes.
        assert list(got.payload) == ["/tmp/a.bin", "/tmp/a.json"]


# ---------------------------------------------------------------------------
# Scheduler batch-level fallback logic
# ---------------------------------------------------------------------------


def _emulate_scheduler_capture_delivery(
    capture_results: dict[str, dict],
    finished_in_band: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Mirror the scheduler's capture delivery in ``update_from_output``.

    Per-request outputs ``pop`` their results in-band; whatever is left
    (results that finalized after the owning request finished) is delivered
    batch-level. Returns ``(in_band_per_req, late_batch_level)``.
    """
    in_band: dict[str, dict] = {}
    for req_id in finished_in_band:
        # The real loop calls ``capture_results.pop(req_id, {})``.
        in_band[req_id] = capture_results.pop(req_id, {})

    late = {rid: res for rid, res in capture_results.items() if res}
    return in_band, late


class TestSchedulerBatchLevelDelivery:
    def test_pop_prevents_double_delivery(self) -> None:
        # A request whose results were ready in time is delivered in-band
        # and must NOT also appear batch-level.
        cr = {"r1": {"fs": _result("r1")}}
        in_band, late = _emulate_scheduler_capture_delivery(cr, {"r1"})
        assert in_band["r1"] == {"fs": _result("r1")}
        assert late == {}

    def test_late_finalize_delivered_batch_level(self) -> None:
        # ``r2`` finalized after it finished: no in-band output exists, so
        # it is surfaced batch-level instead of being dropped.
        cr = {
            "r1": {"fs": _result("r1")},
            "r2": {"fs": _result("r2")},
        }
        in_band, late = _emulate_scheduler_capture_delivery(cr, {"r1"})
        assert "r1" in in_band
        assert set(late) == {"r2"}
        assert late["r2"]["fs"].status == "ok"

    def test_empty_results_not_delivered(self) -> None:
        cr: dict[str, dict] = {"r3": {}}
        _, late = _emulate_scheduler_capture_delivery(cr, set())
        assert late == {}


# ---------------------------------------------------------------------------
# OutputProcessor late-result routing
# ---------------------------------------------------------------------------


def _processor() -> OutputProcessor:
    return OutputProcessor(tokenizer=None, log_stats=False)


class TestOutputProcessorLateRouting:
    def test_stash_and_pop(self) -> None:
        op = _processor()
        op.process_outputs(
            [],
            late_capture_results={"r1": {"fs": _result("r1")}},
        )
        popped = op.pop_late_capture_results("r1")
        assert popped == {"fs": _result("r1")}
        # Second pop yields nothing (cleared).
        assert op.pop_late_capture_results("r1") is None

    def test_event_set_on_arrival(self) -> None:
        op = _processor()
        event = op.register_late_capture_event("r1")
        assert not event.is_set()
        op.process_outputs(
            [],
            late_capture_results={"r1": {"fs": _result("r1")}},
        )
        assert event.is_set()

    def test_indexed_under_client_facing_and_suffixed_id(self) -> None:
        # Engine-internal ids may carry a random suffix on top of the
        # client-facing id the waiter holds. Late results are indexed under
        # both so the waiter (holding the unsuffixed id) can collect them.
        op = _processor()
        op.process_outputs(
            [],
            late_capture_results={"cmpl-x-0-abcdef": {"fs": _result()}},
        )
        assert op.pop_late_capture_results("cmpl-x-0") is not None

    def test_stash_is_bounded(self) -> None:
        op = _processor()
        # Insert well past the 4096 bound; oldest entries must be evicted.
        big = {f"r{i}": {"fs": _result()} for i in range(5000)}
        op.process_outputs([], late_capture_results=big)
        assert len(op._late_capture_results) <= 4096

    def test_no_late_results_is_noop(self) -> None:
        op = _processor()
        out = op.process_outputs([])
        assert out.request_outputs == []
        assert op._late_capture_results == {}


# ---------------------------------------------------------------------------
# wait_for_capture_results semantics (event-driven, no engine)
# ---------------------------------------------------------------------------


class _FakeAsyncLLM:
    """Just the ``wait_for_capture_results`` method bound to a real
    ``OutputProcessor``; mirrors ``AsyncLLM.wait_for_capture_results``.
    """

    def __init__(self) -> None:
        self.output_processor = _processor()

    wait_for_capture_results = (
        __import__("vllm.v1.engine.async_llm", fromlist=["AsyncLLM"])
        .AsyncLLM.wait_for_capture_results
    )


class TestWaitForCaptureResults:
    def test_returns_already_present_results(self) -> None:
        llm = _FakeAsyncLLM()
        llm.output_processor.process_outputs(
            [], late_capture_results={"r1": {"fs": _result()}}
        )

        async def run():
            return await llm.wait_for_capture_results("r1", timeout=1.0)

        got = asyncio.run(run())
        assert got is not None and "fs" in got

    def test_times_out_to_none(self) -> None:
        llm = _FakeAsyncLLM()

        async def run():
            return await llm.wait_for_capture_results("missing", timeout=0.05)

        assert asyncio.run(run()) is None

    def test_wakes_on_late_arrival(self) -> None:
        llm = _FakeAsyncLLM()

        async def run():
            waiter = asyncio.create_task(
                llm.wait_for_capture_results("r1", timeout=2.0)
            )
            await asyncio.sleep(0.05)
            # Late results arrive while the waiter is blocked.
            llm.output_processor.process_outputs(
                [], late_capture_results={"r1": {"fs": _result()}}
            )
            return await waiter

        got = asyncio.run(run())
        assert got is not None and "fs" in got
