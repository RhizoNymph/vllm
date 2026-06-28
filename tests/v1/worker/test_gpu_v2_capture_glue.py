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


def test_drain_pending_capture_results_public_target():
    # ``capture_wait``'s idle-loop drains via the public method (collective_rpc
    # target on the v2 runner); it must exist and surface buffered results.
    glue = _Glue(_req_states([1], [0], {"a": 0}))
    glue._pending_capture_results = {"a": {"c": object()}}

    drained = glue.drain_pending_capture_results()

    assert set(drained) == {"a"}
    assert glue.drain_pending_capture_results() == {}  # buffer cleared


class _CaptureCB:
    """Fake manager that captures the finalize callback to fire it late."""

    def __init__(self):
        self.cb = None

    def finalize_request_async(self, req_id, on_complete):
        self.cb = on_complete


def test_late_finalize_not_orphaned_by_drain():
    # Regression for the capture_wait hang on v2: the finalize-thread callback
    # must stash into the LIVE _pending_capture_results, not a reference
    # captured at closure-creation time that an intervening drain has already
    # swapped away.
    glue = _Glue(_req_states([1], [0], {"r1": 0}))
    mgr = _CaptureCB()
    glue._capture_manager = mgr
    glue._capture_index_to_name = {0: "filesystem"}

    glue._finalize_capture_for_request_async("r1")
    # An idle-loop drain swaps the buffer BEFORE the finalize thread fires.
    assert glue.drain_pending_capture_results() == {}
    # Finalize thread fires late (after writer fsync).
    sentinel = object()
    mgr.cb({0: sentinel})
    # The next drain must surface it — not lose it in an orphaned dict.
    assert glue.drain_pending_capture_results() == {"r1": {"filesystem": sentinel}}


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


# ---- sync-execution consumers ---------------------------------------------


class _FakeSyncBuffers:
    """Stand-in for the (slim or full) CaptureManager owning global buffers."""

    def __init__(self, buffers):
        self._buffers = buffers

    def global_buffer(self, key):
        return self._buffers.get(key)


class _RecordingConsumer:
    """Sync consumer that records the views it sees and returns canned actions."""

    def __init__(self, actions=None, raises=False):
        self._actions = actions
        self._raises = raises
        self.seen = []

    def on_step(self, view):
        self.seen.append(view)
        if self._raises:
            raise RuntimeError("boom")
        return self._actions


class _SyncGlue(CaptureRunnerMixin):
    """Host exposing only the sync-consumer surface + a stubbed steering apply."""

    def __init__(self, req_states, consumers, buffers, monitor_keys):
        self.req_states = req_states
        self._sync_consumers = consumers
        self._sync_capture_buffers = _FakeSyncBuffers(buffers)
        self._sync_monitor_keys = monitor_keys
        self._sync_consumer_stats = {}
        self._sync_step_counter = 0
        self._sync_timing_events = None  # CPU: wall-time accounting path
        self.applied = []

    # The steering mixin provides this on the real runner (resolved via MRO).
    def _apply_steering_actions(self, actions, source):
        self.applied.append((source, actions))


def _sync_input_batch():
    # Batch is decode-first: d (1 decode token) then p (3 prefill tokens).
    return SimpleNamespace(
        num_tokens=4,
        num_reqs=2,
        req_ids=["d", "p"],
        idx_mapping_np=np.asarray([0, 1], dtype=np.int32),
        query_start_loc_np=np.asarray([0, 1, 4], dtype=np.int32),
    )


def test_step_capture_view_slices_buffers_and_spans():
    rs = _req_states(
        prompt_len=[5, 10],
        num_computed=[5, 0],
        req_id_to_index={"d": 0, "p": 1},
    )
    key = (3, "post_block")
    buf = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    glue = _SyncGlue(rs, [], {key: buf}, [key])
    sched = SimpleNamespace(
        num_scheduled_tokens={"d": 1, "p": 3}, total_num_scheduled_tokens=4
    )

    view = glue._build_step_capture_view(sched, _sync_input_batch())

    # Tensor sliced to the unpadded forward token count.
    assert view.tensors[key].shape[0] == 4
    # Per-request spans from query_start_loc_np; phase from req_states.
    assert [(r.req_id, r.start, r.end, r.phase) for r in view.requests] == [
        ("d", 0, 1, "decode"),  # num_computed 5 >= prompt 5
        ("p", 1, 4, "prefill"),  # num_computed 0 < prompt 10
    ]
    # v2 exposes no input-token window (GPU-resident); token_ids is empty.
    assert all(r.token_ids.size == 0 for r in view.requests)


def test_run_sync_consumers_applies_actions_and_counts_steps():
    rs = _req_states([5], [5], {"d": 0})
    actions = [object()]
    consumer = _RecordingConsumer(actions=actions)
    glue = _SyncGlue(rs, [("probe", consumer)], {}, [])
    sched = SimpleNamespace(
        num_scheduled_tokens={"d": 1}, total_num_scheduled_tokens=1
    )
    ib = SimpleNamespace(
        num_tokens=1,
        num_reqs=1,
        req_ids=["d"],
        idx_mapping_np=np.asarray([0], dtype=np.int32),
        query_start_loc_np=np.asarray([0, 1], dtype=np.int32),
    )

    glue._run_sync_consumers(sched, ib)

    assert glue._sync_step_counter == 1
    assert len(consumer.seen) == 1
    # Returned actions routed through the steering apply with the consumer name.
    assert glue.applied == [("probe", actions)]
    assert glue._sync_consumer_stats["probe"]["steps"] == 1


def test_run_sync_consumers_isolates_exceptions():
    rs = _req_states([5], [5], {"d": 0})
    consumer = _RecordingConsumer(raises=True)
    glue = _SyncGlue(rs, [("probe", consumer)], {}, [])
    sched = SimpleNamespace(
        num_scheduled_tokens={"d": 1}, total_num_scheduled_tokens=1
    )
    ib = SimpleNamespace(
        num_tokens=1,
        num_reqs=1,
        req_ids=["d"],
        idx_mapping_np=np.asarray([0], dtype=np.int32),
        query_start_loc_np=np.asarray([0, 1], dtype=np.int32),
    )

    # A raising consumer must not abort the step nor apply any actions.
    glue._run_sync_consumers(sched, ib)

    assert glue.applied == []
    assert glue._sync_consumer_stats["probe"]["steps"] == 1
