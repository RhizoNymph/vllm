# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``SteeringController`` — the plumbing it owns on behalf of a
policy-only subclass: per-request lifecycle (prune on finish), conversation
scoping + bounded eviction, and the trigger -> sticky override -> bridge latch.

A tiny ``_ThresholdController`` supplies only the policy (fire above a
threshold) so the tests exercise the base, not a specific consumer.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.capture.controller import SteeringController
from vllm.v1.capture.step_view import StepCaptureView, StepRequestView
from vllm.v1.capture.types import CaptureSpec
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringAction,
    SteeringVectorUpdate,
)

HIDDEN = 8
LAYER = 0
HOOK = "post_mlp"
KEY = (LAYER, HOOK)


class _ThresholdController(SteeringController):
    """Fires a per-request override when the residual's mean exceeds
    ``threshold``. The residual rows are pre-set so ``mean`` is deterministic.
    """

    def __init__(self, vllm_config, params):
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", 0.5))
        self._vec = np.ones(HIDDEN, dtype=np.float32)

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks={HOOK: [LAYER]}, positions="all_generated")

    def decide(self, request_view, residual) -> SteeringAction | None:
        if float(residual.float().mean()) <= self._threshold:
            return None
        return RequestSteeringOverride(
            req_id=request_view.req_id,
            vectors={HOOK: {LAYER: self._vec}},
            source="threshold_controller",
        )


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.get_hidden_size.return_value = HIDDEN
    return cfg


def _view(reqs):
    """``reqs``: list of ``(req_id, conversation_id, phase, value)``.

    Each request gets one residual row filled with ``value`` (so its mean is
    ``value``, and ``value > threshold`` is the trigger condition).
    """
    rows, views = [], []
    for i, (rid, cid, phase, value) in enumerate(reqs):
        rows.append(np.full(HIDDEN, value, dtype=np.float32))
        views.append(
            StepRequestView(
                req_id=rid,
                start=i,
                end=i + 1,
                phase=phase,
                token_ids=np.empty(0, dtype=np.int64),
                conversation_id=cid,
            )
        )
    tensor = torch.tensor(np.stack(rows), dtype=torch.float32)
    return StepCaptureView(step=0, tensors={KEY: tensor}, requests=views)


def _ctl(**params):
    p = {"threshold": 0.5}
    p.update(params)
    return _ThresholdController(_cfg(), p)


# --- latch: trigger -> sticky -> bridge ------------------------------------


def test_trigger_emits_override_and_latches_conversation():
    ctl = _ctl()
    acts = ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r1"
    assert "c1" in ctl._latched and "r1" in ctl._armed
    assert ctl.status()["triggers"] == 1


def test_no_trigger_below_threshold():
    ctl = _ctl()
    assert ctl.on_step(_view([("r1", "c1", "decode", 0.0)])) is None
    assert ctl._latched == {} and ctl._armed == set()


def test_bridge_steers_new_request_of_latched_conversation():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))  # trigger
    # New request r2 of c1, residual BELOW threshold -> bridged anyway.
    acts = ctl.on_step(_view([("r2", "c1", "decode", 0.0)]))
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r2"
    # Bridged override re-applies the latched vectors.
    np.testing.assert_array_equal(acts[0].vectors[HOOK][LAYER], ctl._vec)
    assert ctl.status()["bridges"] == 1


def test_bridge_preserves_all_override_fields():
    """A bridged turn must steer identically to the trigger turn: every field
    of the latched override is carried over, only ``req_id`` is rebound. Guards
    against field-by-field rebuilds silently dropping ``compose_admitted`` (so
    the trigger composes with admitted steering but later turns replace it).
    """

    class _ComposeController(_ThresholdController):
        def decide(self, request_view, residual) -> SteeringAction | None:
            if float(residual.float().mean()) <= self._threshold:
                return None
            return RequestSteeringOverride(
                req_id=request_view.req_id,
                vectors={HOOK: {LAYER: self._vec}},
                compose_admitted=True,
                source="compose_controller",
            )

    ctl = _ComposeController(_cfg(), {"threshold": 0.5})
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))  # trigger + latch
    latched = ctl._latched["c1"]
    assert latched.compose_admitted is True
    acts = ctl.on_step(_view([("r2", "c1", "decode", 0.0)]))  # bridge
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    bridged = acts[0]
    assert bridged.req_id == "r2"
    # Every field except req_id matches the latched override.
    assert bridged.compose_admitted is True
    assert bridged.source == latched.source
    assert bridged.vectors is latched.vectors
    assert replace(bridged, req_id="r1") == latched


def test_other_conversations_untouched():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))  # latch c1
    assert ctl.on_step(_view([("r2", "c2", "decode", 0.0)])) is None
    assert "c2" not in ctl._latched


# --- per-request lifecycle: emit-once + prune-on-finish --------------------


def test_emit_once_per_request():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    # Same live request next step: already armed -> no duplicate override.
    assert ctl.on_step(_view([("r1", "c1", "decode", 1.0)])) is None


def test_armed_state_pruned_when_request_finishes():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    assert ctl._armed == {"r1"}
    # r1 gone from the view -> pruned; the conversation latch persists.
    ctl.on_step(_view([("r9", "c9", "decode", 0.0)]))
    assert "r1" not in ctl._armed
    assert "c1" in ctl._latched


# --- conversation scoping --------------------------------------------------


def test_untagged_and_prefill_rows_are_skipped():
    ctl = _ctl()
    acts = ctl.on_step(
        _view(
            [
                ("r1", None, "decode", 1.0),  # untagged conversation
                ("r2", "c1", "prefill", 1.0),  # prefill phase
            ]
        )
    )
    assert acts is None and ctl._latched == {}


def test_latch_map_is_bounded_fifo():
    ctl = _ctl(max_conversations=2)
    for i in range(3):
        ctl.on_step(_view([(f"r{i}", f"c{i}", "decode", 1.0)]))
    assert len(ctl._latched) == 2 and "c0" not in ctl._latched


# --- non-override policy actions are emitted but not latched ----------------


def test_non_override_action_is_emitted_without_latching():
    class _GlobalController(_ThresholdController):
        def decide(self, request_view, residual) -> SteeringAction | None:
            if float(residual.float().mean()) <= self._threshold:
                return None
            return SteeringVectorUpdate(
                vectors={HOOK: {LAYER: self._vec}}, phase="decode", source="g"
            )

    ctl = _GlobalController(_cfg(), {"threshold": 0.5})
    acts = ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    assert len(acts) == 1 and isinstance(acts[0], SteeringVectorUpdate)
    # Not a RequestSteeringOverride -> no conversation latch installed.
    assert ctl._latched == {} and "r1" in ctl._armed


def test_controller_requires_decide_and_global_capture_spec():
    class _Incomplete(SteeringController):
        pass

    with pytest.raises(TypeError):
        _Incomplete(_cfg(), {})  # type: ignore[abstract]


# --- latch byte bound ------------------------------------------------------

# One trigger override latches a single (LAYER, HOOK) float32 vector of length
# HIDDEN, so each latched conversation accounts for exactly this many bytes.
VEC_BYTES = HIDDEN * 4


def test_latched_bytes_tracks_payload():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    ctl.on_step(_view([("r2", "c2", "decode", 1.0)]))
    assert ctl._latched_bytes_total == 2 * VEC_BYTES
    assert ctl._latched_bytes == {"c1": VEC_BYTES, "c2": VEC_BYTES}
    assert ctl.status()["latched_bytes"] == 2 * VEC_BYTES


def test_count_cap_evicts_and_updates_byte_total():
    ctl = _ctl(max_conversations=2)
    for i in range(3):
        ctl.on_step(_view([(f"r{i}", f"c{i}", "decode", 1.0)]))
    # Count cap holds and the byte total tracks only the surviving latches.
    assert len(ctl._latched) == 2 and "c0" not in ctl._latched
    assert ctl._latched_bytes_total == 2 * VEC_BYTES
    assert set(ctl._latched_bytes) == {"c1", "c2"}


def test_byte_cap_evicts_oldest_until_fit():
    # Room for exactly two latches; count cap left generous so the byte cap
    # is what forces eviction.
    ctl = _ctl(max_conversations=100, max_latched_bytes=2 * VEC_BYTES)
    for i in range(3):
        ctl.on_step(_view([(f"r{i}", f"c{i}", "decode", 1.0)]))
    assert set(ctl._latched) == {"c1", "c2"} and "c0" not in ctl._latched
    assert ctl._latched_bytes_total == 2 * VEC_BYTES


def test_oversized_single_latch_refused_without_state_corruption(caplog):
    # Byte cap below a single override's payload: latching is impossible.
    ctl = _ctl(max_latched_bytes=VEC_BYTES - 1)
    logging.getLogger("vllm").propagate = True
    with caplog.at_level(logging.WARNING, logger="vllm.v1.capture.controller"):
        acts = ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    # The triggering request still steers this turn ...
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r1"
    assert "r1" in ctl._armed and ctl.status()["triggers"] == 1
    # ... but nothing is latched and byte accounting stays consistent (no leak).
    assert ctl._latched == {} and ctl._latched_bytes == {}
    assert ctl._latched_bytes_total == 0
    assert ctl.status()["latched_bytes"] == 0
    assert ctl._oversize_logged is True
    assert any("latch refused" in r.getMessage() for r in caplog.records)
    # A later turn of the same conversation is not bridged (nothing latched);
    # below threshold it simply does nothing, and the engine does not crash.
    assert ctl.on_step(_view([("r2", "c1", "decode", 0.0)])) is None


def test_oversized_refusal_logged_once():
    ctl = _ctl(max_latched_bytes=VEC_BYTES - 1)
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))
    assert ctl._oversize_logged is True
    # Second oversized refusal must not reset or re-arm the rate-limit flag.
    ctl.on_step(_view([("r2", "c2", "decode", 1.0)]))
    assert ctl._oversize_logged is True
    assert ctl._latched == {} and ctl._latched_bytes_total == 0


def test_relatch_same_conversation_does_not_double_count():
    ctl = _ctl()
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))  # latch c1
    # r1 finishes; a fresh request re-triggers on the SAME conversation. The
    # old latch's bytes must be released before the new latch is accounted.
    ctl.on_step(_view([("r2", "c1", "decode", 1.0)]))
    assert list(ctl._latched) == ["c1"]
    assert ctl._latched_bytes_total == VEC_BYTES


def test_lru_bridge_refreshes_recency_before_eviction():
    # Count cap of 2. Latch c0, c1; then BRIDGE c0 (a new request of c0
    # refreshes its recency). Latching c2 must then evict the least-recently-
    # used — c1 — leaving c0 alive. Pure FIFO would have evicted c0.
    ctl = _ctl(max_conversations=2)
    ctl.on_step(_view([("r0", "c0", "decode", 1.0)]))  # latch c0
    ctl.on_step(_view([("r1", "c1", "decode", 1.0)]))  # latch c1
    # A new request of c0, below threshold -> bridged (refreshes c0 recency).
    acts = ctl.on_step(_view([("r0b", "c0", "decode", 0.0)]))
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert ctl.status()["bridges"] == 1
    # Now c1 is least-recently-used; latching c2 evicts c1, not c0.
    ctl.on_step(_view([("r2", "c2", "decode", 1.0)]))
    assert set(ctl._latched) == {"c0", "c2"}


def test_eviction_is_deterministic_across_identical_sequences():
    seq = [(f"r{i}", f"c{i % 4}", "decode", 1.0) for i in range(12)]
    finals = []
    for _ in range(2):
        ctl = _ctl(max_conversations=3, max_latched_bytes=2 * VEC_BYTES)
        for row in seq:
            ctl.on_step(_view([row]))
        finals.append(
            (
                tuple(ctl._latched),
                dict(ctl._latched_bytes),
                ctl._latched_bytes_total,
            )
        )
    # Same latch sequence -> identical final latch set, order, and byte total.
    assert finals[0] == finals[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
