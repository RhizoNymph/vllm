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
