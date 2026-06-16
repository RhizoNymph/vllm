# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the sync-consumer steering loop (Phase 1a M1).

Drives the unbound ``GPUModelRunner`` / ``SteeringModelRunnerMixin``
methods against duck-typed stubs — no engine, no GPU. Covers:
``_build_step_capture_view`` (spans/phase/token_ids/tensor slicing),
``_run_sync_consumers`` (timing, isolation, action routing), and
``_apply_steering_actions`` (the single apply path shared by the async
queue and sync returns).
"""

import time

import numpy as np
import pytest
import torch

from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.types import CaptureSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.steering_action_queue import SteeringVectorUpdate
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

HIDDEN = 8
MAX_TOKENS = 32


class _FakeInputBatch:
    def __init__(self, reqs: list[dict]):
        # reqs: [{req_id, num_computed, num_prompt}]
        self.num_reqs = len(reqs)
        self.req_ids = [r["req_id"] for r in reqs]
        self.req_id_to_index = {r["req_id"]: i for i, r in enumerate(reqs)}
        self.num_computed_tokens_cpu = np.array(
            [r["num_computed"] for r in reqs], dtype=np.int32
        )
        self.num_prompt_tokens = np.array(
            [r["num_prompt"] for r in reqs], dtype=np.int32
        )
        # Token table: row r holds 1000*r + position so tests can assert
        # exact window contents.
        self.token_ids_cpu = np.stack(
            [np.arange(64, dtype=np.int64) + 1000 * i for i in range(len(reqs))]
        )


class _FakeSchedulerOutput:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = dict(scheduled)
        self.total_num_scheduled_tokens = sum(scheduled.values())


def _slim_manager(keys=((1, "post_mlp"),)) -> CaptureManager:
    hooks: dict[str, list[int]] = {}
    for layer, hook in keys:
        hooks.setdefault(hook, []).append(layer)
    return CaptureManager(
        consumers=(),
        consumer_specs=(),
        extra_global_specs=(CaptureSpec(hooks=hooks, positions="all"),),
        num_hidden_layers=4,
        hidden_size=HIDDEN,
        model_dtype=torch.float32,
        device="cpu",
        max_num_tokens=MAX_TOKENS,
        slim=True,
    )


class _RunnerStub:
    """Duck-typed receiver for the unbound runner methods under test."""

    def __init__(self, reqs, monitor_keys=((1, "post_mlp"),)):
        self.input_batch = _FakeInputBatch(reqs)
        self._sync_capture_buffers = _slim_manager(monitor_keys)
        self._sync_monitor_keys = sorted(monitor_keys)
        self._sync_consumers = []
        self._sync_consumer_stats = {}
        self._sync_step_counter = 0
        self.applied_calls = []

    def _build_step_capture_view(self, scheduler_output):
        return GPUModelRunner._build_step_capture_view(self, scheduler_output)

    def _run_sync_consumers(self, scheduler_output):
        return GPUModelRunner._run_sync_consumers(self, scheduler_output)

    def _apply_steering_actions(self, actions, *, source):
        self.applied_calls.append((source, list(actions)))
        return len(actions), 0


# ---------------------------------------------------------------------------
# _build_step_capture_view
# ---------------------------------------------------------------------------


def test_step_view_spans_phases_and_token_ids():
    stub = _RunnerStub(
        reqs=[
            # Mid-prefill: 4 of 10 prompt tokens computed, 3 scheduled.
            {"req_id": "a", "num_computed": 4, "num_prompt": 10},
            # Decode: prompt done, one token per step.
            {"req_id": "b", "num_computed": 12, "num_prompt": 8},
        ]
    )
    out = stub._build_step_capture_view(_FakeSchedulerOutput({"a": 3, "b": 1}))

    assert out.step == 0
    assert set(out.tensors.keys()) == {(1, "post_mlp")}
    assert out.tensors[(1, "post_mlp")].shape == (4, HIDDEN)

    a, b = out.requests
    assert (a.req_id, a.start, a.end, a.phase) == ("a", 0, 3, "prefill")
    np.testing.assert_array_equal(a.token_ids, [4, 5, 6])
    assert (b.req_id, b.start, b.end, b.phase) == ("b", 3, 4, "decode")
    np.testing.assert_array_equal(b.token_ids, [1012])


def test_step_view_skips_zero_token_and_unknown_requests():
    stub = _RunnerStub(
        reqs=[
            {"req_id": "a", "num_computed": 0, "num_prompt": 4},
            {"req_id": "b", "num_computed": 9, "num_prompt": 8},
        ]
    )
    # "a" has zero scheduled tokens this step.
    out = stub._build_step_capture_view(_FakeSchedulerOutput({"a": 0, "b": 1}))
    assert [r.req_id for r in out.requests] == ["b"]
    assert (out.requests[0].start, out.requests[0].end) == (0, 1)


def test_step_view_tensor_is_zero_copy_of_buffer():
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    mgr = stub._sync_capture_buffers
    hidden = torch.randn(1, HIDDEN)
    mgr.on_hook(1, "post_mlp", hidden)

    out = stub._build_step_capture_view(_FakeSchedulerOutput({"a": 1}))
    view_tensor = out.tensors[(1, "post_mlp")]
    torch.testing.assert_close(view_tensor, hidden)
    # Zero-copy: writing through the buffer is visible in the view.
    mgr.global_buffer((1, "post_mlp"))[0].fill_(7.0)
    assert bool((view_tensor[0] == 7.0).all())


# ---------------------------------------------------------------------------
# _run_sync_consumers
# ---------------------------------------------------------------------------


class _ActionsConsumer:
    def __init__(self, actions):
        self.actions = actions
        self.views = []

    def on_step(self, view):
        self.views.append(view)
        return self.actions


class _BoomConsumer:
    def on_step(self, view):
        raise RuntimeError("boom")


def test_run_sync_consumers_routes_actions_and_isolates_failures():
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    update = SteeringVectorUpdate(
        vectors={"post_mlp": {1: np.ones(HIDDEN, dtype=np.float32)}}
    )
    good = _ActionsConsumer([update])
    stub._sync_consumers = [("boom", _BoomConsumer()), ("good", good)]

    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))

    # The failing consumer did not block the good one.
    assert stub.applied_calls == [("good", [update])]
    assert len(good.views) == 1
    # Both consumers timed; counter advanced.
    assert stub._sync_consumer_stats["boom"]["steps"] == 1
    assert stub._sync_consumer_stats["good"]["steps"] == 1
    assert stub._sync_step_counter == 1

    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    assert stub._sync_consumer_stats["good"]["steps"] == 2
    assert good.views[1].step == 2


def test_run_sync_consumers_skips_apply_when_no_actions():
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    stub._sync_consumers = [("idle", _ActionsConsumer(None))]
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    assert stub.applied_calls == []


class _FakeEvent:
    """Stand-in for ``torch.cuda.Event(enable_timing=True)``.

    The start event of a pair carries the elapsed-ms reading so the
    runner's ``events[0].elapsed_time(events[1])`` returns a fixed value.
    """

    def __init__(self, elapsed_ms: float = 0.0):
        self._elapsed_ms = elapsed_ms
        self.records = 0

    def record(self):
        self.records += 1

    def elapsed_time(self, _other):
        return self._elapsed_ms


def test_gpu_event_timing_is_deferred_one_step():
    # GPU work measured at 10 ms/step; wall is ~0.
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    stub._sync_consumers = [("probe", _ActionsConsumer(None))]
    stub._sync_timing_events = {"probe": (_FakeEvent(10.0), _FakeEvent())}

    # Step 1: events recorded but not yet readable -> no GPU sample.
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    s = stub._sync_consumer_stats["probe"]
    assert s["steps"] == 1
    assert s["gpu_steps"] == 0
    assert s["gpu_last_ms"] is None

    # Steps 2,3: the prior step's events are read (one-step deferral).
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    s = stub._sync_consumer_stats["probe"]
    assert s["gpu_steps"] == 2
    assert s["gpu_total_ms"] == pytest.approx(20.0)
    assert s["gpu_max_ms"] == pytest.approx(10.0)
    assert s["gpu_last_ms"] == 10.0
    # The start event was recorded once per step.
    assert stub._sync_timing_events["probe"][0].records == 3


def test_budget_charges_gpu_time_not_wall_time():
    # Slow wall (5 ms) over a 1 ms budget, but cheap GPU work (0.1 ms):
    # the budget must charge the GPU reading, so nothing is over budget.
    class _SlowWall:
        sync_budget_ms = 1.0

        def on_step(self, view):
            time.sleep(0.005)
            return None

    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    stub._sync_consumers = [("probe", _SlowWall())]
    stub._sync_timing_events = {"probe": (_FakeEvent(0.1), _FakeEvent())}
    for _ in range(3):
        stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))

    s = stub._sync_consumer_stats["probe"]
    assert s["total_ms"] > 1.0  # wall blew the budget...
    assert s["over_budget_steps"] == 0  # ...but the GPU charge did not


# ---------------------------------------------------------------------------
# _apply_steering_actions (mixin)
# ---------------------------------------------------------------------------


class _MixinStub(SteeringModelRunnerMixin):
    """Bare mixin host with the state _apply_steering_actions needs."""

    def __init__(self, manager, layers):
        self._steering_manager = manager
        self._steerable_layers_cache = layers
        self._dynamic_steering_stats = {}


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("steering_table_post_mlp", torch.zeros(7, HIDDEN))
        self.register_buffer("steering_table_post_mlp_dynvec", torch.zeros(HIDDEN))


def test_apply_actions_routes_vector_updates_to_manager():
    mgr = SteeringManager(max_steering_configs=4, device=None)
    host = _MixinStub(mgr, {1: _Layer()})
    update = SteeringVectorUpdate(
        vectors={"post_mlp": {1: np.full(HIDDEN, 2.0, dtype=np.float32)}}
    )
    applied, rejected = host._apply_steering_actions([update], source="sync:test")
    assert (applied, rejected) == (1, 0)
    assert host._dynamic_steering_stats["sync:test"] == {
        "applied": 1,
        "rejected": 0,
    }

    mgr.populate_steering_tables(host._steerable_layers_cache)
    # Decode updates now land on the dedicated dynamic tier (§5.4), not
    # the global-decode row — applied by the kernel via token_scales.
    layer = host._steerable_layers_cache[1]
    assert torch.all(layer.steering_table_post_mlp[2] == 0.0)
    assert torch.all(layer.steering_table_post_mlp_dynvec == 2.0)


def test_apply_actions_rejects_unknown_action_type():
    mgr = SteeringManager(max_steering_configs=4, device=None)
    host = _MixinStub(mgr, {1: _Layer()})
    applied, rejected = host._apply_steering_actions([object()], source="sync:test")
    assert (applied, rejected) == (0, 1)
    assert host._dynamic_steering_stats["sync:test"]["rejected"] == 1


def test_apply_actions_rejects_when_steering_uninitialized():
    host = _MixinStub(None, {})
    update = SteeringVectorUpdate(
        vectors={"post_mlp": {1: np.ones(HIDDEN, dtype=np.float32)}}
    )
    applied, rejected = host._apply_steering_actions([update], source="s")
    assert (applied, rejected) == (0, 1)


def test_apply_actions_mixes_good_and_bad():
    mgr = SteeringManager(max_steering_configs=4, device=None)
    host = _MixinStub(mgr, {1: _Layer()})
    good = SteeringVectorUpdate(
        vectors={"post_mlp": {1: np.ones(HIDDEN, dtype=np.float32)}}
    )
    bad_layer = SteeringVectorUpdate(
        vectors={"post_mlp": {3: np.ones(HIDDEN, dtype=np.float32)}}
    )
    applied, rejected = host._apply_steering_actions(
        [good, object(), bad_layer], source="s"
    )
    assert (applied, rejected) == (1, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Observability: ring buffer, budget warning, status RPC
# ---------------------------------------------------------------------------


class _SlowConsumer:
    sync_budget_ms = 0.0  # everything is over budget

    def on_step(self, view):
        return None


def test_ring_buffer_bounded_and_ordered():
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    stub._sync_consumers = [("idle", _ActionsConsumer(None))]
    for _ in range(300):
        stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    ring = stub._sync_consumer_stats["idle"]["ring"]
    assert len(ring) == 256  # bounded
    steps = [entry[0] for entry in ring]
    assert steps == sorted(steps)
    assert steps[-1] == 300


def test_over_budget_steps_counted():
    stub = _RunnerStub(reqs=[{"req_id": "a", "num_computed": 8, "num_prompt": 8}])
    stub._sync_consumers = [("slow", _SlowConsumer())]
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    stub._run_sync_consumers(_FakeSchedulerOutput({"a": 1}))
    stats = stub._sync_consumer_stats["slow"]
    assert stats["over_budget_steps"] == 2


def test_dynamic_steering_status_is_picklable():
    import pickle

    class _StatusConsumer:
        def on_step(self, view):
            return None

        def status(self):
            return {"engaged": True, "gain": 1.5}

    host = _MixinStub(
        SteeringManager(max_steering_configs=4, device=None), {1: _Layer()}
    )
    # Graft the runner-side sync state the status method reads.
    host._sync_consumers = [("probe", _StatusConsumer())]
    host._sync_consumer_stats = {
        "probe": {
            "steps": 3,
            "total_ms": 1.5,
            "max_ms": 0.9,
            "over_budget_steps": 1,
            "ring": __import__("collections").deque(
                [(1, 0.5, 0), (2, 0.4, 1)], maxlen=256
            ),
            "_last_budget_warn": 123.0,
        }
    }
    host._req_dynamic_decode = {}

    status = host.get_dynamic_steering_status()
    blob = pickle.dumps(status)
    assert blob
    assert status["steering_initialized"] is True
    assert status["sync_consumers"]["probe"]["steps"] == 3
    assert status["sync_consumers"]["probe"]["ring"] == [[1, 0.5, 0], [2, 0.4, 1]]
    assert status["sync_consumers"]["probe"]["status"] == {
        "engaged": True,
        "gain": 1.5,
    }
    assert status["dynamic_pool"] == {
        "capacity": 0,
        "in_use": 0,
        "overrides": {},
    }


def test_dynamic_steering_status_consumer_error_isolated():
    class _BoomStatus:
        def on_step(self, view):
            return None

        def status(self):
            raise RuntimeError("nope")

    host = _MixinStub(
        SteeringManager(max_steering_configs=4, device=None), {1: _Layer()}
    )
    host._sync_consumers = [("boom", _BoomStatus())]
    host._sync_consumer_stats = {
        "boom": {
            "steps": 1,
            "total_ms": 0.1,
            "max_ms": 0.1,
            "over_budget_steps": 0,
            "ring": __import__("collections").deque(maxlen=256),
            "_last_budget_warn": 0.0,
        }
    }
    status = host.get_dynamic_steering_status()
    assert "error" in status["sync_consumers"]["boom"]["status"]
