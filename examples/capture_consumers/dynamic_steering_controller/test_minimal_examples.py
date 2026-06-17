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

from dynamic_steering_controller.minimal_examples import (  # noqa: E402
    AsyncTierExample,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
