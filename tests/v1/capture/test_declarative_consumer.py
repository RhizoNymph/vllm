# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the built-in declarative steering consumer's on_step
dispatch (``when × scope × apply`` → substrate actions).

Exercises the consumer against hand-built ``StepCaptureView`` objects (no
engine) and asserts the emitted action sequence for each gate combination,
plus the arming/latch/pulse lifecycle. CPU-only.
"""

import numpy as np
import torch

from vllm.v1.capture.declarative import (
    _PROBE_CACHE_MAX,
    DeclarativeSteeringConsumer,
)
from vllm.v1.capture.step_view import StepCaptureView, StepRequestView
from vllm.v1.steering_schema import ResolvedGate
from vllm.v1.worker.steering_action_queue import (
    DECLARATIVE_SOURCE,
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
)

HIDDEN = 4
SITE = (5, "post_block")


def _add(scope, when="always"):
    return ResolvedGate(
        scope=scope,
        when_kind=when,
        probe_site=SITE if when == "probe" else None,
        probe_vec=np.ones(HIDDEN, dtype=np.float32) if when == "probe" else None,
        threshold=0.0 if when == "probe" else None,
        sharpness=1.0 if when == "probe" else None,
        apply_kind="add",
        steer_vectors={"post_block": {5: np.ones(HIDDEN, dtype=np.float32)}},
        strength=2.0,
    )


def _att(scope, strength=0.0):
    return ResolvedGate(
        scope=scope, when_kind="always", probe_site=None, probe_vec=None,
        threshold=None, sharpness=None, apply_kind="attenuate",
        steer_vectors=None, strength=strength,
    )


def _req(rid, gates, *, phase="decode", cid=None):
    return StepRequestView(
        req_id=rid, start=0, end=1, phase=phase,
        token_ids=np.array([]), conversation_id=cid, steering=gates,
    )


def _view(reqs, tensors=None, step=0):
    return StepCaptureView(step=step, tensors=tensors or {}, requests=reqs)


def _consumer(**params):
    return DeclarativeSteeringConsumer(None, params)


def _types(acts):
    return [type(a).__name__ for a in (acts or [])]


def test_add_always_installs_override_once():
    c = _consumer()
    acts = c.on_step(_view([_req("r", [_add("rest_of_request")])]))
    assert _types(acts) == ["RequestSteeringOverride"]
    ovr = acts[0]
    assert ovr.compose_admitted is True
    assert ovr.source == DECLARATIVE_SOURCE
    # already armed -> no re-emit
    assert c.on_step(_view([_req("r", [_add("rest_of_request")])])) is None


def test_this_token_probe_attaches_row_monitor():
    c = _consumer()
    acts = c.on_step(_view([_req("r", [_add("this_token", "probe")])]))
    assert _types(acts) == ["RequestSteeringOverride", "SteeringMonitorUpdate"]
    mon = acts[1]
    assert isinstance(mon, SteeringMonitorUpdate)
    assert mon.req_id == "r"
    assert (mon.layer, mon.hook) == SITE


def test_rest_of_conversation_latches_and_bridges():
    c = _consumer(probe_sites=["5:post_block"])
    tensors = {SITE: torch.ones(1, HIDDEN) * 10.0}  # high proj -> fires
    a1 = c.on_step(_view([_req("a", [_add("rest_of_conversation", "probe")], cid="k")],
                         tensors))
    assert _types(a1) == ["RequestSteeringOverride"]
    assert len(c._latched) == 1
    # a NEW request of the same conversation is bridged (no re-trigger)
    a2 = c.on_step(_view([_req("b", [_add("rest_of_conversation", "probe")], cid="k")],
                         tensors))
    assert _types(a2) == ["RequestSteeringOverride"]
    assert c._bridges == 1


def test_host_probe_does_not_fire_below_threshold():
    c = _consumer(probe_sites=["5:post_block"])
    tensors = {SITE: torch.ones(1, HIDDEN) * -10.0}
    acts = c.on_step(_view([_req("r", [_add("rest_of_request", "probe")], cid="k")],
                          tensors))
    assert acts is None


def test_host_probe_site_not_captured_is_skipped():
    # probe site not in probe_sites -> not in view.tensors -> gate skipped
    c = _consumer(probe_sites=["0:post_block"])
    acts = c.on_step(_view([_req("r", [_add("rest_of_request", "probe")], cid="k")],
                          tensors={(0, "post_block"): torch.ones(1, HIDDEN)}))
    assert acts is None


def test_attenuate_only_installs_admitted_override_and_scale():
    c = _consumer()
    acts = c.on_step(_view([_req("r", [_att("rest_of_request", 0.0)])]))
    assert _types(acts) == ["RequestSteeringOverride", "SteeringScaleUpdate"]
    assert acts[0].vectors == {}  # admitted-only override
    assert acts[0].compose_admitted is True
    assert isinstance(acts[1], SteeringScaleUpdate)
    assert acts[1].scale == 0.0 and acts[1].req_id == "r"


def test_next_step_pulse_installs_then_clears():
    c = _consumer()
    a1 = c.on_step(_view([_req("r", [_add("next_step")])]))
    assert _types(a1) == ["RequestSteeringOverride"]
    assert len(c._next_step_pending) == 1
    a2 = c.on_step(_view([_req("r", [_add("next_step")])]))
    assert _types(a2) == ["RequestSteeringOverride"]
    assert a2[0].vectors is None  # cleared
    assert len(c._next_step_pending) == 0


def test_prefill_and_no_gates_are_ignored():
    c = _consumer()
    assert c.on_step(_view([_req("r", [_add("rest_of_request")], phase="prefill")])) is None
    assert c.on_step(_view([_req("r", None)])) is None


def test_multi_add_merge_single_override():
    c = _consumer()
    acts = c.on_step(_view([_req("r", [_add("rest_of_request"), _add("rest_of_request")])]))
    # two add gates -> ONE override row (vectors merged), not two
    assert _types(acts) == ["RequestSteeringOverride"]
    # merged ones + ones = 2*ones at (post_block, 5) (strength already folded
    # into steer_vectors upstream by resolve_gates; the helper uses raw ones)
    np.testing.assert_allclose(
        np.asarray(acts[0].vectors["post_block"][5]), np.full(HIDDEN, 2.0)
    )


def test_conversation_bridges_a_later_gateless_turn():
    # A later turn of a latched conversation carries NO gates of its own but
    # must still be bridged (regression: the no-gate short-circuit used to run
    # before the bridge check, so gateless turns were skipped).
    c = _consumer(probe_sites=["5:post_block"])
    tensors = {SITE: torch.ones(1, HIDDEN) * 10.0}
    c.on_step(_view([_req("t1", [_add("rest_of_conversation", "probe")], cid="k")],
                    tensors))
    assert len(c._latched) == 1
    # turn 2: same conversation, steering=None -> bridged, not skipped
    acts = c.on_step(_view([_req("t2", None, cid="k")]))
    assert _types(acts) == ["RequestSteeringOverride"]
    assert acts[0].req_id == "t2"
    assert c._bridges == 1


def test_finished_request_state_pruned():
    c = _consumer()
    c.on_step(_view([_req("r", [_add("rest_of_request")])]))
    assert "r" in c._armed
    # request no longer live -> pruned
    c.on_step(_view([_req("other", None)]))
    assert "r" not in c._armed


# -- host-probe tensor cache -------------------------------------------------

_DEV = torch.device("cpu")


def test_probe_cache_equal_content_shares_one_entry():
    # Two distinct ndarray objects with equal content collapse to one entry
    # (id()-keyed code would store two).
    c = _consumer()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert a is not b
    ta = c._probe_tensor(a, _DEV)
    tb = c._probe_tensor(b, _DEV)
    assert len(c._probe_tensor_cache) == 1
    assert ta is tb
    torch.testing.assert_close(ta, torch.tensor([1.0, 2.0, 3.0]))


def test_probe_cache_different_content_distinct_tensors():
    c = _consumer()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    ta = c._probe_tensor(a, _DEV)
    tb = c._probe_tensor(b, _DEV)
    assert len(c._probe_tensor_cache) == 2
    assert ta is not tb
    torch.testing.assert_close(ta, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(tb, torch.tensor([4.0, 5.0, 6.0]))


def test_probe_cache_is_bounded_and_evicts_oldest():
    c = _consumer()
    probes = [
        np.full(2, float(i), dtype=np.float32)
        for i in range(_PROBE_CACHE_MAX + 5)
    ]
    for p in probes:
        c._probe_tensor(p, _DEV)
    assert len(c._probe_tensor_cache) == _PROBE_CACHE_MAX
    # The 5 oldest were evicted; the newest MAX remain and read correctly.
    survivor = c._probe_tensor(probes[-1], _DEV)
    torch.testing.assert_close(
        survivor, torch.full((2,), float(_PROBE_CACHE_MAX + 4))
    )


def test_probe_cache_hit_after_eviction_reuploads():
    c = _consumer()
    first = np.full(2, -1.0, dtype=np.float32)
    c._probe_tensor(first, _DEV)  # LRU-oldest
    # Evict `first` by inserting MAX newer distinct probes.
    for i in range(_PROBE_CACHE_MAX):
        c._probe_tensor(np.full(2, float(i), dtype=np.float32), _DEV)
    assert len(c._probe_tensor_cache) == _PROBE_CACHE_MAX
    # A fresh, value-equal array re-uploads with the correct value.
    again = c._probe_tensor(np.full(2, -1.0, dtype=np.float32), _DEV)
    torch.testing.assert_close(again, torch.full((2,), -1.0))
    assert len(c._probe_tensor_cache) == _PROBE_CACHE_MAX
