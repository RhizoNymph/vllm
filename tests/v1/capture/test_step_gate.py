# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the rank-replicated capture force-eager gate.

Covers :func:`vllm.v1.capture.manager.selector_hits_window` (the shared
position/window predicate) and :class:`CaptureStepGate` (the per-step
force-eager decision). Global specs no longer force eager — they ride the
CUDA-graph-safe persistent-buffer path — so the gate fires only on client
captures; ``force_all`` remains only as a manual escape hatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.capture.manager import selector_hits_window
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.step_gate import (
    CaptureStepGate,
    _extract_selectors,
)

# ---------------------------------------------------------------------------
# selector_hits_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positions, num_prompt, num_computed, num_scheduled, expected",
    [
        # last_prompt: the final prompt token is index num_prompt-1 == 9.
        ("last_prompt", 10, 0, 10, True),  # prefill covers [0,10) → hits 9
        ("last_prompt", 10, 0, 5, False),  # first chunk [0,5) → misses 9
        ("last_prompt", 10, 5, 5, True),  # second chunk [5,10) → hits 9
        ("last_prompt", 10, 10, 1, False),  # decode step [10,11) → no capture
        ("last_prompt", 10, 25, 1, False),  # later decode → no capture
        # all_prompt: any prompt position in window.
        ("all_prompt", 10, 0, 5, True),
        ("all_prompt", 10, 10, 1, False),  # past the prompt
        # all_generated: only generated positions (>= num_prompt).
        ("all_generated", 10, 0, 10, False),  # pure prefill, no gen yet
        ("all_generated", 10, 10, 1, True),  # decode token at pos 10
        # all: every step with scheduled tokens captures.
        ("all", 10, 0, 4, True),
        ("all", 10, 30, 1, True),
        # explicit list.
        ([3, 7], 10, 0, 5, True),  # 3 in [0,5)
        ([3, 7], 10, 5, 5, True),  # 7 in [5,10)
        ([3, 7], 10, 10, 1, False),  # neither in [10,11)
        # empty window never captures.
        ("all", 10, 5, 0, False),
    ],
)
def test_selector_hits_window(
    positions, num_prompt, num_computed, num_scheduled, expected
):
    assert (
        selector_hits_window(positions, num_prompt, num_computed, num_scheduled)
        is expected
    )


def test_selector_hits_window_unknown_selector_raises():
    with pytest.raises(ValueError):
        selector_hits_window("not_a_selector", 10, 0, 10)


# ---------------------------------------------------------------------------
# _extract_selectors
# ---------------------------------------------------------------------------


def test_extract_selectors_none_and_empty():
    assert _extract_selectors(None) == []
    assert _extract_selectors({}) == []


def test_extract_selectors_dict_spec():
    raw = {"filesystem": {"hooks": {"post_mlp": "all"}, "positions": "last_prompt"}}
    assert _extract_selectors(raw) == ["last_prompt"]


def test_extract_selectors_object_spec():
    raw = {"c": SimpleNamespace(positions=[1, 2, 3])}
    assert _extract_selectors(raw) == [[1, 2, 3]]


def test_extract_selectors_missing_positions_is_conservative():
    # A capture spec with no positions field must still force eager.
    assert _extract_selectors({"c": {"hooks": {}}}) == ["all"]


def test_extract_selectors_unionizes_multiple_consumers():
    raw = {
        "fs": {"positions": "last_prompt"},
        "log": {"positions": "all"},
    }
    assert set(map(str, _extract_selectors(raw))) == {"last_prompt", "all"}


# ---------------------------------------------------------------------------
# CaptureStepGate
# ---------------------------------------------------------------------------


def _view(rows):
    """Build a CaptureBatchView from (req_id, num_prompt, num_computed,
    num_scheduled) tuples."""
    req_ids, npt, ncomp, nsched, offsets = [], [], [], [], []
    off = 0
    for req_id, p, c, s in rows:
        req_ids.append(req_id)
        npt.append(p)
        ncomp.append(c)
        nsched.append(s)
        offsets.append(off)
        off += s
    return CaptureBatchView(
        req_ids=req_ids,
        num_prompt_tokens=npt,
        num_computed_tokens=ncomp,
        num_scheduled_tokens=nsched,
        token_offsets=offsets,
    )


def test_gate_empty_when_nothing_registered():
    gate = CaptureStepGate()
    assert gate.step_captures(_view([("a", 10, 0, 10)])) is False


def test_gate_last_prompt_fires_only_on_prefill_chunk():
    gate = CaptureStepGate()
    gate.register("a", {"fs": {"positions": "last_prompt"}})
    # Prefill chunk covering the final prompt token → eager.
    assert gate.step_captures(_view([("a", 10, 0, 10)])) is True
    # Decode steps → no capture, cudagraph eligible.
    assert gate.step_captures(_view([("a", 10, 10, 1)])) is False
    assert gate.step_captures(_view([("a", 10, 42, 1)])) is False


def test_gate_chunked_prefill_fires_on_correct_chunk():
    gate = CaptureStepGate()
    gate.register("a", {"fs": {"positions": "last_prompt"}})
    assert gate.step_captures(_view([("a", 10, 0, 5)])) is False  # chunk [0,5)
    assert gate.step_captures(_view([("a", 10, 5, 5)])) is True  # chunk [5,10)


def test_gate_plain_request_never_forces_eager():
    # A request with no capture spec must not force eager — this is the
    # benchmarked "capture configured but this request doesn't capture"
    # speedup.
    gate = CaptureStepGate()
    gate.register("a", None)
    assert gate.tracked_requests() == 0
    assert gate.step_captures(_view([("a", 10, 0, 10)])) is False
    assert gate.step_captures(_view([("a", 10, 10, 1)])) is False


def test_gate_mixed_batch_forces_eager_when_any_captures():
    gate = CaptureStepGate()
    gate.register("cap", {"fs": {"positions": "last_prompt"}})
    gate.register("plain", None)
    # Batch with the capture request at its capturing prefill chunk.
    assert gate.step_captures(_view([("plain", 8, 0, 8), ("cap", 10, 0, 10)])) is True
    # Same batch a step later: both in decode, capture request done.
    assert gate.step_captures(_view([("plain", 8, 8, 1), ("cap", 10, 10, 1)])) is False


def test_gate_drop_stops_forcing():
    gate = CaptureStepGate()
    gate.register("a", {"fs": {"positions": "all"}})
    assert gate.step_captures(_view([("a", 10, 5, 1)])) is True
    gate.drop("a")
    assert gate.tracked_requests() == 0
    assert gate.step_captures(_view([("a", 10, 5, 1)])) is False


def test_gate_force_all_always_eager():
    # ``force_all`` is now a manual escape hatch (no longer auto-enabled by
    # global specs), but when set it still forces eager unconditionally.
    gate = CaptureStepGate(force_all=True)
    # register is a no-op under force_all, but the gate still forces eager.
    gate.register("a", None)
    assert gate.step_captures(_view([("a", 10, 10, 1)])) is True
    assert gate.step_captures(_view([("a", 10, 99, 1)])) is True
