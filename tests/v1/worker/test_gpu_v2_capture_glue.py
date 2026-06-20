# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the v2 runner's capture control-plane glue.

These cover the v2-specific projection logic (``CaptureBatchView`` builders and
the finalized-result drain) without a CUDA device or a real model. The data
plane and managers are exercised separately in ``tests/v1/capture``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.capture_runner_mixin import CaptureRunnerMixin


class _Glue(CaptureRunnerMixin):
    """Minimal host exposing only what the view builders / drain read."""

    def __init__(self, req_states):
        self.req_states = req_states
        self._capture_feature_enabled = True
        self._capture_step_gate = None
        self._capture_manager = None
        self._pending_capture_results = {}
        self._pending_capture_results_lock = threading.Lock()


def _req_states(prompt_len, num_computed, req_id_to_index):
    return SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.asarray(prompt_len, dtype=np.int32)),
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32),
        req_id_to_index=req_id_to_index,
    )


def test_gate_view_from_scheduler_output():
    # req_state slots: a->1, b->0 (deliberately not in batch order).
    rs = _req_states(
        prompt_len=[7, 5],
        num_computed=[5, 0],
        req_id_to_index={"a": 1, "b": 0},
    )
    glue = _Glue(rs)
    # Scheduler dict order is the iteration order used for the gate view.
    sched = SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 5})

    view = glue._build_capture_gate_view(sched)

    assert view.req_ids == ["a", "b"]
    assert view.num_prompt_tokens == [5, 7]  # a->idx1=5, b->idx0=7
    assert view.num_computed_tokens == [0, 5]
    assert view.num_scheduled_tokens == [2, 5]
    assert view.token_offsets == [0, 2]  # cumulative scheduled tokens


def test_gate_view_unknown_request_defaults_zero():
    rs = _req_states([3], [0], {"a": 0})
    glue = _Glue(rs)
    sched = SimpleNamespace(num_scheduled_tokens={"ghost": 4})

    view = glue._build_capture_gate_view(sched)

    assert view.req_ids == ["ghost"]
    assert view.num_prompt_tokens == [0]
    assert view.num_computed_tokens == [0]
    assert view.num_scheduled_tokens == [4]


def test_batch_view_uses_input_batch_offsets():
    # req_state slots: d->0, p->1; batch is decode-first so idx_mapping=[0, 1].
    rs = _req_states(
        prompt_len=[10, 20],
        num_computed=[10, 0],
        req_id_to_index={"d": 0, "p": 1},
    )
    glue = _Glue(rs)
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["d", "p"],
        idx_mapping_np=np.asarray([0, 1], dtype=np.int32),
        num_scheduled_tokens=np.asarray([1, 20], dtype=np.int32),
        # query_start_loc_np carries one extra (cumulative) entry; the builder
        # slices [:num_reqs].
        query_start_loc_np=np.asarray([0, 1, 21], dtype=np.int32),
    )

    view = glue._build_capture_batch_view(input_batch)

    assert view.req_ids == ["d", "p"]
    assert view.num_prompt_tokens == [10, 20]
    assert view.num_computed_tokens == [10, 0]
    assert view.num_scheduled_tokens == [1, 20]
    assert view.token_offsets == [0, 1]  # from query_start_loc_np[:2]


def test_gate_decision_false_without_gate():
    glue = _Glue(_req_states([1], [0], {"a": 0}))
    sched = SimpleNamespace(num_scheduled_tokens={"a": 1})

    assert glue._capture_gate_decision(sched) is False


def test_drain_capture_results_empties_buffer():
    glue = _Glue(_req_states([1], [0], {"a": 0}))
    glue._pending_capture_results = {"a": {"c": object()}}

    drained = glue._drain_capture_results()

    assert set(drained) == {"a"}
    assert glue._drain_capture_results() == {}  # buffer cleared


def test_drain_disabled_returns_empty():
    glue = _Glue(_req_states([1], [0], {"a": 0}))
    glue._capture_feature_enabled = False
    glue._pending_capture_results = {"a": {"c": object()}}

    assert glue._drain_capture_results() == {}


# ---- re-add / preemption-resume admission ---------------------------------


class _FakeGate:
    def __init__(self):
        self.registered = {}
        self.dropped = []

    def register(self, req_id, raw):
        self.registered[req_id] = raw

    def drop(self, req_id):
        self.dropped.append(req_id)
        self.registered.pop(req_id, None)


class _FakeManager:
    def __init__(self):
        self._reqs = set()
        self.unregistered = []

    def has_request(self, req_id):
        return req_id in self._reqs

    def unregister_request(self, req_id):
        self.unregistered.append(req_id)
        self._reqs.discard(req_id)


class _AddGlue(CaptureRunnerMixin):
    """Host that records admissions and isolates the re-add branching."""

    def __init__(self, gate, mgr):
        self._capture_feature_enabled = True
        self._capture_step_gate = gate
        self._capture_manager = mgr
        self.registered_calls = []

    # Stub the full registration machinery; only the branching is under test.
    def _register_capture_request(self, new_req_data):
        self.registered_calls.append(new_req_data.req_id)
        if self._capture_manager is not None:
            self._capture_manager._reqs.add(new_req_data.req_id)


def _new_req(req_id, capture):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SimpleNamespace(capture=capture),
    )


def test_capture_add_fresh_request_registers():
    gate, mgr = _FakeGate(), _FakeManager()
    glue = _AddGlue(gate, mgr)

    glue._capture_add_request(_new_req("a", {"c": {}}), was_present=False)

    assert glue.registered_calls == ["a"]
    assert gate.registered == {"a": {"c": {}}}
    assert gate.dropped == []
    assert mgr.unregistered == []


def test_capture_add_streaming_readd_discards_and_reregisters():
    gate, mgr = _FakeGate(), _FakeManager()
    # Prior chunk already admitted.
    mgr._reqs.add("a")
    gate.registered["a"] = {"c": {"positions": "last_prompt"}}
    glue = _AddGlue(gate, mgr)

    # Re-add (still live) with a different capture spec.
    glue._capture_add_request(
        _new_req("a", {"c": {"positions": "all_generated"}}), was_present=True
    )

    # Stale registration dropped, then re-registered against the new prompt.
    assert gate.dropped == ["a"]
    assert mgr.unregistered == ["a"]
    assert glue.registered_calls == ["a"]
    assert gate.registered["a"] == {"c": {"positions": "all_generated"}}


def test_capture_add_preemption_resume_keeps_registration():
    gate, mgr = _FakeGate(), _FakeManager()
    # Registration survived preemption (finish_requests did not finalize it).
    mgr._reqs.add("a")
    gate.registered["a"] = {"c": {}}
    glue = _AddGlue(gate, mgr)

    # Resumed req is folded into scheduled_new_reqs on v2, but was_present is
    # False because finish_requests removed it from req_states on preempt.
    glue._capture_add_request(_new_req("a", {"c": {}}), was_present=False)

    # No discard, no re-registration — the open registration is reused.
    assert gate.dropped == []
    assert mgr.unregistered == []
    assert glue.registered_calls == []


def test_capture_add_readd_on_non_capturer_rank_no_manager():
    gate = _FakeGate()
    gate.registered["a"] = {"c": {}}
    glue = _AddGlue(gate, None)  # non-capturer rank: no manager

    # Streaming re-add still refreshes the rank-replicated gate, no crash.
    glue._capture_add_request(_new_req("a", {"c": {"positions": "all"}}), True)

    assert gate.dropped == ["a"]
    assert gate.registered["a"] == {"c": {"positions": "all"}}
    assert glue.registered_calls == []


def test_capture_add_disabled_is_noop():
    gate, mgr = _FakeGate(), _FakeManager()
    glue = _AddGlue(gate, mgr)
    glue._capture_feature_enabled = False

    glue._capture_add_request(_new_req("a", {"c": {}}), was_present=True)

    assert gate.registered == {} and gate.dropped == []
    assert glue.registered_calls == []
