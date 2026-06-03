# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dispatch-queue backpressure / overload policies.

The dispatch queue is the single GPU-facing backpressure point. These
tests block the dispatch thread (so the bounded queue saturates
deterministically), then exercise each ``overload_policy`` via
``_enqueue_packet`` with empty fake packets.
"""

from __future__ import annotations

import threading
import time

import torch
from unittest.mock import MagicMock

from vllm.v1.capture.manager import CaptureManager, _DispatchPacket

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
