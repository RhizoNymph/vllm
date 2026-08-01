# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dispatch-queue backpressure / overload policies.

The dispatch queue is the single GPU-facing backpressure point. These
tests block the dispatch thread (so the bounded queue saturates
deterministically), then exercise each ``overload_policy`` via
``_enqueue_packet`` with empty fake packets.
"""

from __future__ import annotations

import pathlib
import threading
import time

import torch
from unittest.mock import MagicMock

from vllm.v1.capture.manager import CaptureManager, _DispatchPacket
from vllm.v1.capture.plan import CapturePositionEntry

NUM_LAYERS, HIDDEN, DTYPE = 4, 8, torch.float32


def _sink() -> MagicMock:
    s = MagicMock()
    s.location = "worker"
    s.get_result = MagicMock(return_value=None)
    s.wait_for_result = MagicMock(return_value=None)
    s.shutdown = MagicMock()
    return s


def _mgr(**kw) -> CaptureManager:
    return CaptureManager(
        consumers=(_sink(),),
        consumer_specs=(None,),
        num_hidden_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        model_dtype=DTYPE,
        **kw,
    )


def _empty_packet() -> _DispatchPacket:
    return _DispatchPacket(entries=[], scratch_pinned={}, cuda_event=None)


def _block_dispatch(mgr: CaptureManager):
    """Replace fan-out with a gate so the dispatch thread parks on the first
    packet it pulls. Returns (release_event, started_event)."""
    release = threading.Event()
    started = threading.Event()

    def fake_fan_out(_packet):
        started.set()
        release.wait(timeout=10)

    mgr._fan_out_to_consumers = fake_fan_out  # type: ignore[assignment]
    return release, started


class TestOverloadPolicy:
    def test_drop_discards_when_queue_full(self) -> None:
        mgr = _mgr(dispatch_queue_size=2, overload_policy="drop")
        release, started = _block_dispatch(mgr)
        try:
            mgr._enqueue_packet(_empty_packet())  # picked up, thread blocks
            assert started.wait(timeout=5)
            mgr._enqueue_packet(_empty_packet())  # queued (1/2)
            mgr._enqueue_packet(_empty_packet())  # queued (2/2, full)
            mgr._enqueue_packet(_empty_packet())  # full -> dropped
            mgr._enqueue_packet(_empty_packet())  # full -> dropped
            assert mgr.dropped_packets == 2
        finally:
            release.set()
            mgr._drain_dispatch_queue()
        # pending settles to zero after the backlog drains
        assert mgr._pending_dispatches == 0

    def test_block_never_drops(self) -> None:
        # With block policy the producer waits; nothing is dropped. Use a
        # background releaser so the (blocking) enqueues eventually proceed.
        mgr = _mgr(dispatch_queue_size=2, overload_policy="block")
        release, started = _block_dispatch(mgr)
        mgr._enqueue_packet(_empty_packet())
        assert started.wait(timeout=5)
        # Release shortly so the blocking puts can drain.
        threading.Timer(0.2, release.set).start()
        for _ in range(5):
            mgr._enqueue_packet(_empty_packet())  # may block, never drops
        mgr._drain_dispatch_queue()
        assert mgr.dropped_packets == 0
        assert mgr._pending_dispatches == 0

    def test_unbounded_legacy_never_drops(self) -> None:
        # dispatch_queue_size<=0 -> unbounded, drop policy is moot.
        mgr = _mgr(dispatch_queue_size=0, overload_policy="drop")
        release, started = _block_dispatch(mgr)
        try:
            for _ in range(20):
                mgr._enqueue_packet(_empty_packet())
            assert mgr.dropped_packets == 0  # unbounded: no drops
        finally:
            release.set()
            mgr._drain_dispatch_queue()
        assert mgr._pending_dispatches == 0

    def test_invalid_policy_rejected(self) -> None:
        raised = False
        try:
            _mgr(overload_policy="bogus")
        except ValueError as e:
            raised = "overload_policy" in str(e)
        assert raised


class _RecordingSink:
    """Sink that records, in arrival order, the marker each chunk carries
    (its scratch value), with an optional gate to block the first call so
    the dispatch queue saturates deterministically."""

    location = "worker"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.seqs: list[int] = []
        self._gate = gate
        self.started = threading.Event()
        self._first = True

    def submit_chunk_batch(self, chunks) -> None:
        if self._first and self._gate is not None:
            self._first = False
            self.started.set()
            self._gate.wait(timeout=10)
        for c in chunks:
            self.seqs.append(int(c.tensor[0, 0].item()))

    def submit_chunk(self, chunk) -> None:
        self.seqs.append(int(chunk.tensor[0, 0].item()))

    def submit_finalize(self, finalize) -> None: ...
    def get_result(self, key): return None
    def wait_for_result(self, key, timeout): return None
    def shutdown(self, timeout: float = 30.0) -> None: ...


def _marked_packet(seq: int, layer: int = 0, hook: str = "h") -> _DispatchPacket:
    """A packet whose single scratch row holds ``seq`` as its value, so the
    sink can recover submission order from the fanned-out chunk."""
    entry = CapturePositionEntry(
        request_id="r", layer=layer, hook=hook, logical_pos=0,
        scratch_row=0, step_index=seq, consumer_mask=1,
    )
    scratch = torch.full((1, HIDDEN), float(seq), dtype=DTYPE)
    return _DispatchPacket(
        entries=[entry], scratch_pinned={(layer, hook): (None, scratch)},
        cuda_event=None,
    )


class TestSpill:
    def test_spill_preserves_order_and_loses_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        gate = threading.Event()
        sink = _RecordingSink(gate=gate)
        mgr = CaptureManager(
            consumers=(sink,), consumer_specs=(None,),
            num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, model_dtype=DTYPE,
            dispatch_queue_size=2, overload_policy="spill",
            spill_dir=str(tmp_path),
        )
        try:
            mgr._enqueue_packet(_marked_packet(0))  # picked up, blocks in sink
            assert sink.started.wait(timeout=5)
            for seq in range(1, 8):  # 1,2 queue (size 2); 3..7 spill
                mgr._enqueue_packet(_marked_packet(seq))
            assert mgr.spilled_packets >= 1
            gate.set()
            mgr._drain_dispatch_queue()
        finally:
            gate.set()
            mgr.shutdown(timeout=10)
        # Every packet arrived exactly once, in submission order.
        assert sink.seqs == list(range(8)), sink.seqs
        assert mgr._pending_dispatches == 0

    def test_spill_syncs_cuda_event_before_serializing(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Spilling serializes the pinned views on the producer thread,
        # bypassing the dispatch loop's event sync — the packet's copies
        # must be awaited before the bytes are snapshotted.
        sink = _RecordingSink()
        mgr = CaptureManager(
            consumers=(sink,), consumer_specs=(None,),
            num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, model_dtype=DTYPE,
            dispatch_queue_size=1, overload_policy="spill",
            spill_dir=str(tmp_path),
        )
        order: list[str] = []
        real_serialize = mgr._serialize_packet

        def recording_serialize(packet):
            order.append("serialize")
            return real_serialize(packet)

        mgr._serialize_packet = recording_serialize  # type: ignore[assignment]
        event = MagicMock()
        event.synchronize = MagicMock(side_effect=lambda: order.append("sync"))
        base = _marked_packet(0)
        packet = _DispatchPacket(
            entries=base.entries,
            scratch_pinned=base.scratch_pinned,
            cuda_event=event,
        )
        try:
            with mgr._pending_cond:
                mgr._pending_dispatches += 1
            mgr._spill_packet(packet)
            assert order == ["sync", "serialize"], order
            mgr._drain_dispatch_queue()
        finally:
            mgr.shutdown(timeout=10)
        # The replayed packet still reached the sink.
        assert sink.seqs == [0], sink.seqs
        assert mgr._pending_dispatches == 0

    def test_spill_entry_published_only_after_file_written(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        # The dispatch loop pops a ``_spill_pending`` entry and reads its
        # file without holding the lock — an entry visible before its bytes
        # are on disk would race the replay path.
        sink = _RecordingSink()
        mgr = CaptureManager(
            consumers=(sink,), consumer_specs=(None,),
            num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, model_dtype=DTYPE,
            dispatch_queue_size=1, overload_policy="spill",
            spill_dir=str(tmp_path),
        )
        visible_at_write: list[bool] = []
        real_write = pathlib.Path.write_bytes

        def checking_write(self_path, data):
            with mgr._spill_lock:
                pending = [p for p, _ in mgr._spill_pending]
            visible_at_write.append(self_path in pending)
            return real_write(self_path, data)

        monkeypatch.setattr(pathlib.Path, "write_bytes", checking_write)
        try:
            with mgr._pending_cond:
                mgr._pending_dispatches += 1
            mgr._spill_packet(_marked_packet(0))
            assert visible_at_write == [False], visible_at_write
            mgr._drain_dispatch_queue()
        finally:
            mgr.shutdown(timeout=10)
        # The spilled packet was still replayed to the sink exactly once.
        assert sink.seqs == [0], sink.seqs
        assert mgr._pending_dispatches == 0

    def test_spill_cap_falls_back_to_block_no_loss(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Tiny spill cap forces the cap-full -> block fallback; nothing lost.
        gate = threading.Event()
        sink = _RecordingSink(gate=gate)
        mgr = CaptureManager(
            consumers=(sink,), consumer_specs=(None,),
            num_hidden_layers=NUM_LAYERS, hidden_size=HIDDEN, model_dtype=DTYPE,
            dispatch_queue_size=1, overload_policy="spill",
            spill_dir=str(tmp_path), spill_max_bytes=1,  # ~always "full"
        )

        def feed():
            for seq in range(6):
                mgr._enqueue_packet(_marked_packet(seq))

        feeder = threading.Thread(target=feed)
        try:
            feeder.start()
            assert sink.started.wait(timeout=5)
            time.sleep(0.2)  # let the feeder block on the spill cap
            gate.set()
            feeder.join(timeout=10)
            mgr._drain_dispatch_queue()
        finally:
            gate.set()
            mgr.shutdown(timeout=10)
        assert sink.seqs == list(range(6)), sink.seqs
        assert mgr._pending_dispatches == 0
