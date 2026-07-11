# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the minimal single-purpose dynamic-steering examples — that
each emits exactly the intended action type / uses the intended transport.
CPU-only, no engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from dynamic_steering_controller.e2e_stub import (  # noqa: E402
    ConfigurableOverrideStub,
)
from dynamic_steering_controller.minimal_examples import (  # noqa: E402
    AsyncTierExample,
    ConversationLatchExample,
    GlobalTierExample,
    MonitorRowGateExample,
    MonitorTierExample,
    OverrideExample,
    PerRequestScaleExample,
    TierScaleExample,
)

from vllm.v1.capture.step_view import StepCaptureView, StepRequestView  # noqa: E402
from vllm.v1.worker.steering_action_queue import (  # noqa: E402
    RequestSteeringOverride,
    SteeringActionQueue,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
    SteeringVectorUpdate,
    get_steering_action_queue,
    install_steering_action_queue,
)

HIDDEN = 16
LAYER = 0


def _cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.get_hidden_size.return_value = HIDDEN
    cfg.parallel_config.pipeline_parallel_size = 1
    return cfg


def _decode_view(step: int = 0, req_id: str = "r1") -> StepCaptureView:
    req = StepRequestView(
        req_id=req_id, start=0, end=1, phase="decode",
        token_ids=np.array([7], dtype=np.int64),
    )
    import torch
    return StepCaptureView(
        step=step, tensors={(LAYER, "post_mlp"): torch.zeros(1, HIDDEN)},
        requests=[req],
    )


def _params(**kw):
    p = {"steer_layer": LAYER, "steer_norm": 4.0}
    p.update(kw)
    return p


def test_override_example_emits_override():
    ex = OverrideExample(_cfg(), _params())
    acts = ex.on_step(_decode_view())
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r1"
    assert ex.on_step(_decode_view(step=1)) is None  # once only


def test_global_tier_example_emits_vector_update():
    ex = GlobalTierExample(_cfg(), _params())
    acts = ex.on_step(_decode_view())
    assert len(acts) == 1 and isinstance(acts[0], SteeringVectorUpdate)
    assert acts[0].phase == "decode"


def test_tier_scale_example_installs_then_scales():
    ex = TierScaleExample(_cfg(), _params())
    acts = ex.on_step(_decode_view())
    kinds = {type(a) for a in acts}
    assert kinds == {SteeringVectorUpdate, SteeringScaleUpdate}
    sc = next(a for a in acts if isinstance(a, SteeringScaleUpdate))
    assert sc.tier_gain is True


def test_per_request_scale_example_installs_then_scales_by_req_id():
    ex = PerRequestScaleExample(_cfg(), _params())
    a1 = ex.on_step(_decode_view(step=0))
    assert len(a1) == 1 and isinstance(a1[0], RequestSteeringOverride)
    a2 = ex.on_step(_decode_view(step=1))
    assert len(a2) == 1 and isinstance(a2[0], SteeringScaleUpdate)
    assert a2[0].req_id == "r1"  # scales by req_id (not dyn_id)
    assert ex.on_step(_decode_view(step=2)) is None


def test_monitor_tier_example_installs_probe_and_tier():
    ex = MonitorTierExample(_cfg(), _params(threshold=0.2))
    acts = ex.on_step(_decode_view())
    kinds = {type(a) for a in acts}
    assert kinds == {SteeringVectorUpdate, SteeringScaleUpdate, SteeringMonitorUpdate}
    mon = next(a for a in acts if isinstance(a, SteeringMonitorUpdate))
    assert mon.gate_rows is False and mon.layer == LAYER


def test_monitor_rowgate_example_sets_gate_rows():
    ex = MonitorRowGateExample(_cfg(), _params())
    acts = ex.on_step(_decode_view())
    assert len(acts) == 1 and isinstance(acts[0], SteeringMonitorUpdate)
    assert acts[0].gate_rows is True


def test_async_example_submits_to_queue():
    prev = get_steering_action_queue()
    install_steering_action_queue(SteeringActionQueue())
    try:
        ex = AsyncTierExample(_cfg(), _params())
        ex.on_capture(("k",), None, {})
        queue = get_steering_action_queue()
        drained = queue.drain()
        assert len(drained) == 1 and isinstance(drained[0], SteeringVectorUpdate)
        assert ex.status()["submitted"] == 1
    finally:
        install_steering_action_queue(prev)


def test_async_example_noop_without_queue():
    prev = get_steering_action_queue()
    install_steering_action_queue(None)
    try:
        ex = AsyncTierExample(_cfg(), _params())
        ex.on_capture(("k",), None, {})  # must not raise
        assert ex.status()["submitted"] == 0
    finally:
        install_steering_action_queue(prev)


def _stub(**kw):
    p = {"steer_layer": LAYER, "steer_hook": "post_mlp", "steer_norm": 8.0}
    p.update(kw)
    return ConfigurableOverrideStub(_cfg(), p)


def test_cfg_stub_override_mode_emits_only_override():
    acts = _stub(mode="override").on_step(_decode_view())
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)


def test_cfg_stub_rowgate_emits_override_then_gate_rows_monitor():
    for gate_on, sign in ((True, -1.0), (False, 1.0)):
        acts = _stub(mode="rowgate", gate_on=gate_on).on_step(_decode_view())
        assert [type(a) for a in acts] == [
            RequestSteeringOverride, SteeringMonitorUpdate]
        mon = acts[1]
        assert mon.gate_rows is True
        # Threshold saturated in the gate direction.
        assert (mon.threshold < 0) is (sign < 0)


def test_cfg_stub_reqscale_emits_override_then_req_id_scale():
    acts = _stub(mode="reqscale", scale=0.0).on_step(_decode_view(req_id="r1"))
    assert [type(a) for a in acts] == [
        RequestSteeringOverride, SteeringScaleUpdate]
    assert acts[1].req_id == "r1" and acts[1].scale == 0.0
    # Override is first so the req_id scale can resolve the fresh dyn_id.
    assert acts[0].req_id == "r1"


# ---------------------------------------------------------------------------
# ConversationLatchExample — trigger / bridge / prune
# ---------------------------------------------------------------------------

_PROBE_KEY = (LAYER, "post_block")


def _latch_params(**kw):
    p = {"steer_layer": LAYER, "steer_norm": 4.0, "threshold": 0.5}
    p.update(kw)
    return p


def _latch_view(ex, reqs):
    """Build a decode-step view for ConversationLatchExample.

    ``reqs``: list of ``(req_id, conversation_id, phase, value)``. Each request
    gets one residual row set to ``value * probe``; since the probe is a unit
    vector the mean projection equals ``value`` (so ``value > threshold`` is
    the trigger condition).
    """
    import torch

    rows, views = [], []
    for i, (rid, cid, phase, value) in enumerate(reqs):
        rows.append(value * ex._probe)
        views.append(
            StepRequestView(
                req_id=rid, start=i, end=i + 1, phase=phase,
                token_ids=np.empty(0, dtype=np.int64), conversation_id=cid,
            )
        )
    tensor = torch.tensor(np.stack(rows), dtype=torch.float32)
    return StepCaptureView(step=0, tensors={_PROBE_KEY: tensor}, requests=views)


def test_latch_trigger_overrides_and_latches_conversation():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    # value 1.0 > threshold 0.5 -> trigger.
    acts = ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 1.0)]))
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r1"
    assert "c1" in ex._latched and "r1" in ex._armed
    assert ex.status()["triggers"] == 1


def test_latch_does_not_trigger_below_threshold():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    # value 0.0 < threshold -> no trigger, conversation not latched.
    assert ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 0.0)])) is None
    assert ex._latched == {} and ex._armed == set()


def test_latch_bridges_later_request_of_same_conversation():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    # Turn 1: r1 triggers and latches c1.
    ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 1.0)]))
    # Turn 2: a NEW request r2 of c1, residual BELOW threshold -> bridged
    # (steered without re-triggering) because the conversation is latched.
    acts = ex.on_step(_latch_view(ex, [("r2", "c1", "decode", 0.0)]))
    assert len(acts) == 1 and isinstance(acts[0], RequestSteeringOverride)
    assert acts[0].req_id == "r2"
    assert ex.status()["bridges"] == 1
    # r1 finished -> pruned from armed; r2 is now armed.
    assert ex._armed == {"r2"}


def test_latch_leaves_other_conversations_untouched():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 1.0)]))  # latch c1
    # A different conversation c2, below threshold -> untouched.
    assert ex.on_step(_latch_view(ex, [("r2", "c2", "decode", 0.0)])) is None
    assert "c2" not in ex._latched


def test_latch_emit_once_per_request_and_prunes_finished():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 1.0)]))
    # Same live request, next step: already armed -> no duplicate override.
    assert ex.on_step(_latch_view(ex, [("r1", "c1", "decode", 1.0)])) is None
    # r1 gone -> pruned from armed.
    ex.on_step(_latch_view(ex, [("r9", "c9", "decode", 0.0)]))
    assert "r1" not in ex._armed


def test_latch_ignores_untagged_and_prefill_rows():
    ex = ConversationLatchExample(_cfg(), _latch_params())
    acts = ex.on_step(
        _latch_view(
            ex,
            [
                ("r1", None, "decode", 1.0),   # untagged conversation
                ("r2", "c1", "prefill", 1.0),  # prefill phase
            ],
        )
    )
    assert acts is None and ex._latched == {}


def test_latch_map_is_bounded():
    ex = ConversationLatchExample(_cfg(), _latch_params(max_conversations=2))
    for i in range(3):
        ex.on_step(_latch_view(ex, [(f"r{i}", f"c{i}", "decode", 1.0)]))
    # Oldest conversation evicted; map capped at 2.
    assert len(ex._latched) == 2 and "c0" not in ex._latched


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
