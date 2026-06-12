# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for DynamicSteeringController (sync execution).

Drives ``on_step`` against synthetic ``StepCaptureView``s with a
MagicMock ``VllmConfig`` and tempfile probe/steering vectors. No engine
or GPU required.

Usage: ``python test.py`` or ``pytest test.py``.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from dynamic_steering_controller import (  # noqa: E402
    DynamicSteeringController,
    PolicyConfig,
    ProbePolicy,
)

from vllm.v1.capture.step_view import (  # noqa: E402
    StepCaptureView,
    StepRequestView,
)
from vllm.v1.worker.steering_action_queue import (  # noqa: E402
    RequestSteeringOverride,
    SteeringVectorUpdate,
)

HIDDEN = 32
MONITOR = (3, "post_mlp")


# ---------------------------------------------------------------------------
# ProbePolicy
# ---------------------------------------------------------------------------


def _policy(**kwargs) -> ProbePolicy:
    defaults = dict(threshold=0.5, hysteresis=0.2, ema_alpha=1.0, gain=2.0)
    defaults.update(kwargs)
    return ProbePolicy(PolicyConfig(**defaults))


def test_policy_engages_above_threshold():
    p = _policy()
    p.observe("r1", np.array([0.6]))
    assert p.current_gain("r1") == 2.0
    assert p.engaged("r1")


def test_policy_stays_disengaged_below_threshold():
    p = _policy()
    p.observe("r1", np.array([0.49]))
    assert p.current_gain("r1") == 0.0
    assert not p.engaged("r1")


def test_policy_hysteresis_holds_engagement():
    p = _policy()
    p.observe("r1", np.array([0.6]))
    assert p.current_gain("r1") == 2.0
    # Dips below threshold but above threshold - hysteresis: stays on.
    p.observe("r1", np.array([0.4]))
    assert p.current_gain("r1") == 2.0
    # Below the disengage level: off; re-engage needs full threshold.
    p.observe("r1", np.array([0.29]))
    assert p.current_gain("r1") == 0.0
    p.observe("r1", np.array([0.4]))
    assert p.current_gain("r1") == 0.0


def test_policy_states_are_independent_per_request():
    p = _policy()
    p.observe("hot", np.array([0.9]))
    p.observe("cold", np.array([0.1]))
    assert p.current_gain("hot") == 2.0
    assert p.current_gain("cold") == 0.0
    assert p.engaged("hot") and not p.engaged("cold")


def test_policy_ema_smoothing():
    p = _policy(ema_alpha=0.5, hysteresis=0.0)
    p.observe("r1", np.array([1.0]))
    p.observe("r1", np.array([0.0]))
    assert p.score("r1") == pytest.approx(0.5)
    p.observe("r1", np.array([0.0]))
    assert p.score("r1") == pytest.approx(0.25)


def test_policy_multi_token_chunk_folds_in_order():
    p = _policy(ema_alpha=0.5)
    p.observe("r1", np.array([1.0, 0.0]))
    assert p.score("r1") == pytest.approx(0.5)


def test_policy_aggregate_max_vs_mean():
    p_max = _policy(aggregate="max")
    p_max.observe("r1", np.array([0.2]))
    p_max.observe("r2", np.array([0.8]))
    assert p_max.aggregate_score() == pytest.approx(0.8)

    p_mean = _policy(aggregate="mean")
    p_mean.observe("r1", np.array([0.2]))
    p_mean.observe("r2", np.array([0.8]))
    assert p_mean.aggregate_score() == pytest.approx(0.5)


def test_policy_global_gain_uses_aggregate():
    p = _policy(aggregate="max")
    p.observe("r1", np.array([0.9]))
    assert p.current_global_gain() == 2.0
    p.forget("r1")
    assert p.current_global_gain() == 0.0


def test_policy_forget_and_prune():
    p = _policy()
    p.observe("r1", np.array([0.9]), step=1)
    p.observe("r2", np.array([0.9]), step=10)
    pruned = p.prune_unseen(current_step=14, max_age=5)
    assert pruned == ["r1"]
    assert p.keys() == ["r2"]
    p.forget("r2")
    assert p.keys() == []


def test_policy_proportional_gain_ramps():
    p = _policy(gain_mode="proportional", threshold=0.5, hysteresis=0.2, gain=2.0)
    p.observe("r1", np.array([0.5]))
    assert p.current_gain("r1") == pytest.approx(2.0)
    p.observe("r1", np.array([0.4]))
    assert p.current_gain("r1") == pytest.approx(1.0)
    p.observe("r1", np.array([0.9]))
    assert p.current_gain("r1") == pytest.approx(2.0)


def test_policy_emission_gating():
    p = _policy(min_emit_delta=0.5, gain_mode="proportional", gain=2.0)
    assert not p.should_emit("r1", 0.0)
    assert p.should_emit("r1", 1.0)
    p.mark_emitted("r1", 1.0)
    assert not p.should_emit("r1", 1.2)
    assert p.should_emit("r1", 1.6)
    # Engagement flip always emits.
    assert p.should_emit("r1", 0.0)


def test_policy_config_validation():
    with pytest.raises(ValueError):
        PolicyConfig(threshold=0.5, hysteresis=-0.1)
    with pytest.raises(ValueError):
        PolicyConfig(threshold=0.5, ema_alpha=0.0)
    with pytest.raises(ValueError):
        PolicyConfig(threshold=0.5, gain_mode="nope")
    with pytest.raises(ValueError):
        PolicyConfig(threshold=0.5, aggregate="median")


# ---------------------------------------------------------------------------
# Controller fixtures
# ---------------------------------------------------------------------------


def _mock_config(pp: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.get_hidden_size.return_value = HIDDEN
    cfg.parallel_config.pipeline_parallel_size = pp
    return cfg


def _make_controller(
    tmp: Path,
    probe: torch.Tensor,
    steer: torch.Tensor,
    **param_overrides,
) -> DynamicSteeringController:
    probe_path = tmp / "probe.pt"
    steer_path = tmp / "steer.pt"
    torch.save(probe, probe_path)
    torch.save(steer, steer_path)
    params = {
        "monitor_layer": MONITOR[0],
        "monitor_hook": MONITOR[1],
        "probe_path": str(probe_path),
        "steering_vector_path": str(steer_path),
        "threshold": 0.5,
        "hysteresis": 0.2,
        "ema_alpha": 1.0,
        "gain": 2.0,
    }
    params.update(param_overrides)
    return DynamicSteeringController(_mock_config(), params)


def _view(
    rows_by_req: dict[str, torch.Tensor],
    phases: dict[str, str] | None = None,
    step: int = 0,
) -> StepCaptureView:
    """Build a StepCaptureView from per-request activation rows."""
    phases = phases or {}
    requests = []
    chunks = []
    offset = 0
    for req_id, rows in rows_by_req.items():
        n = rows.shape[0]
        requests.append(
            StepRequestView(
                req_id=req_id,
                start=offset,
                end=offset + n,
                phase=phases.get(req_id, "decode"),
                token_ids=np.arange(n, dtype=np.int64),
            )
        )
        chunks.append(rows)
        offset += n
    tensor = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, HIDDEN)
    return StepCaptureView(step=step, tensors={MONITOR: tensor}, requests=requests)


def _aligned(probe: torch.Tensor, scale: float = 1.0, n: int = 1) -> torch.Tensor:
    return (probe * scale).repeat(n, 1)


@pytest.fixture()
def tmp() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def probe() -> torch.Tensor:
    p = torch.zeros(HIDDEN)
    p[0] = 1.0
    return p


# ---------------------------------------------------------------------------
# Controller: construction validation
# ---------------------------------------------------------------------------


def test_rejects_pipeline_parallel(tmp: Path, probe: torch.Tensor):
    torch.save(probe, tmp / "p.pt")
    params = {
        "monitor_layer": 0,
        "probe_path": str(tmp / "p.pt"),
        "steering_vector_path": str(tmp / "p.pt"),
        "threshold": 0.5,
    }
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        DynamicSteeringController(_mock_config(pp=2), params)


def test_rejects_bad_vectors(tmp: Path, probe: torch.Tensor):
    with pytest.raises(ValueError, match="hidden_size"):
        _make_controller(tmp, torch.randn(HIDDEN + 1), probe)
    with pytest.raises(ValueError, match="zero norm"):
        _make_controller(tmp, torch.zeros(HIDDEN), probe)
    with pytest.raises(ValueError, match="1-D"):
        _make_controller(tmp, torch.randn(2, HIDDEN), probe)


def test_rejects_capture_only_steer_hook(tmp: Path, probe: torch.Tensor):
    with pytest.raises(ValueError, match="steer_hook"):
        _make_controller(tmp, probe, probe, steer_hook="mlp_in")


def test_sync_class_attributes(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe)
    assert ctrl.execution == "sync"
    assert ctrl.location == "worker"
    assert not ctrl.reads_client_spec
    assert ctrl.sync_budget_ms == 5.0
    spec = ctrl.global_capture_spec()
    assert spec.hooks == {MONITOR[1]: [MONITOR[0]]}
    assert spec.positions == "all_generated"


# ---------------------------------------------------------------------------
# Controller: per-request actuation
# ---------------------------------------------------------------------------


def test_per_request_override_for_firing_request_only(tmp: Path, probe: torch.Tensor):
    steer = torch.full((HIDDEN,), 3.0)
    ctrl = _make_controller(tmp, probe, steer, steer_layers=[3, 5])

    hot = _aligned(probe, 4.0)  # cosine 1.0 → fires
    cold = torch.zeros(1, HIDDEN)
    cold[0, 1] = 5.0  # cosine 0.0 → silent
    actions = ctrl.on_step(_view({"hot": hot, "cold": cold}))

    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, RequestSteeringOverride)
    assert action.req_id == "hot"
    assert set(action.vectors["post_mlp"].keys()) == {3, 5}
    np.testing.assert_allclose(
        action.vectors["post_mlp"][3],
        np.full(HIDDEN, 6.0, dtype=np.float32),  # steer(3) * gain(2)
    )


def test_per_request_steady_state_does_not_reemit(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe)
    assert len(ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))) == 1
    # Same engaged state, binary gain: silent.
    assert ctrl.on_step(_view({"r1": _aligned(probe)}, step=2)) == []


def test_per_request_disengage_emits_clear(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe)
    ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))
    actions = ctrl.on_step(_view({"r1": _aligned(probe, -1.0)}, step=2))
    assert len(actions) == 1
    assert actions[0].vectors is None  # clear → revert to admitted routing


def test_per_request_prefill_observed_but_not_actuated(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe)
    actions = ctrl.on_step(
        _view({"r1": _aligned(probe, n=4)}, phases={"r1": "prefill"}, step=1)
    )
    assert actions == []
    # EMA primed during prefill: first decode step fires immediately.
    actions = ctrl.on_step(_view({"r1": _aligned(probe)}, step=2))
    assert len(actions) == 1
    assert actions[0].vectors is not None


def test_per_request_departed_state_pruned(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe, forget_after_steps=2)
    ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))
    assert ctrl._policy.score("r1") is not None
    for step in range(2, 6):
        ctrl.on_step(_view({"other": torch.zeros(1, HIDDEN)}, step=step))
    assert ctrl._policy.score("r1") is None


def test_no_monitor_tensor_is_noop(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe)
    view = StepCaptureView(step=1, tensors={}, requests=[])
    assert ctrl.on_step(view) is None


# ---------------------------------------------------------------------------
# Controller: global actuation
# ---------------------------------------------------------------------------


def test_global_mode_emits_vector_update(tmp: Path, probe: torch.Tensor):
    steer = torch.full((HIDDEN,), 3.0)
    ctrl = _make_controller(tmp, probe, steer, actuation="global")
    actions = ctrl.on_step(_view({"r1": _aligned(probe, 4.0)}, step=1))
    assert len(actions) == 1
    update = actions[0]
    assert isinstance(update, SteeringVectorUpdate)
    assert update.phase == "decode"
    np.testing.assert_allclose(
        update.vectors["post_mlp"][MONITOR[0]],
        np.full(HIDDEN, 6.0, dtype=np.float32),
    )
    # Steady engagement: no re-emit.
    assert ctrl.on_step(_view({"r1": _aligned(probe)}, step=2)) == []


def test_global_mode_disengage_emits_zero_vector(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(tmp, probe, probe, actuation="global")
    ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))
    actions = ctrl.on_step(_view({"r1": _aligned(probe, -1.0)}, step=2))
    assert len(actions) == 1
    np.testing.assert_allclose(
        actions[0].vectors["post_mlp"][MONITOR[0]],
        np.zeros(HIDDEN, dtype=np.float32),
    )


def test_dot_score_mode(tmp: Path, probe: torch.Tensor):
    ctrl = _make_controller(
        tmp, probe * 2.0, probe, score="dot", threshold=3.0
    )  # probe normalized to unit at load
    rows = torch.zeros(1, HIDDEN)
    rows[0, 0] = 2.5  # dot 2.5 < 3.0
    assert ctrl.on_step(_view({"r1": rows}, step=1)) == []
    rows2 = torch.zeros(1, HIDDEN)
    rows2[0, 0] = 3.5
    assert len(ctrl.on_step(_view({"r1": rows2}, step=2))) == 1


# ---------------------------------------------------------------------------
# Packed banks + status
# ---------------------------------------------------------------------------


def _packed_hook(vectors: dict[int, np.ndarray]) -> dict:
    layers = sorted(vectors.keys())
    stacked = np.stack([vectors[i].astype(np.float32) for i in layers])
    return {
        "dtype": "float32",
        "shape": list(stacked.shape),
        "layer_indices": layers,
        "data": base64.b64encode(np.ascontiguousarray(stacked).tobytes()).decode(
            "ascii"
        ),
    }


def test_packed_probe_and_steering_bank(tmp: Path, probe: torch.Tensor):
    probe_np = probe.numpy().astype(np.float32)
    steer_a = np.full(HIDDEN, 1.0, dtype=np.float32)
    steer_b = np.full(HIDDEN, 2.0, dtype=np.float32)

    probe_file = tmp / "probe.json"
    probe_file.write_text(
        json.dumps({"post_mlp": _packed_hook({MONITOR[0]: probe_np})})
    )
    steer_file = tmp / "steer.json"
    steer_file.write_text(
        json.dumps({"post_mlp": _packed_hook({3: steer_a, 5: steer_b})})
    )

    ctrl = DynamicSteeringController(
        _mock_config(),
        {
            "monitor_layer": MONITOR[0],
            "probe_packed_path": str(probe_file),
            "steering_packed_path": str(steer_file),
            "threshold": 0.5,
            "ema_alpha": 1.0,
            "gain": 2.0,
        },
    )
    actions = ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))
    assert len(actions) == 1
    vectors = actions[0].vectors["post_mlp"]
    np.testing.assert_allclose(vectors[3], steer_a * 2.0)
    np.testing.assert_allclose(vectors[5], steer_b * 2.0)


def test_packed_probe_missing_row_rejected(tmp: Path, probe: torch.Tensor):
    probe_file = tmp / "probe.json"
    probe_file.write_text(
        json.dumps({"post_mlp": _packed_hook({99: probe.numpy().astype(np.float32)})})
    )
    torch.save(probe, tmp / "steer.pt")
    with pytest.raises(ValueError, match="no\\s+row"):
        DynamicSteeringController(
            _mock_config(),
            {
                "monitor_layer": MONITOR[0],
                "probe_packed_path": str(probe_file),
                "steering_vector_path": str(tmp / "steer.pt"),
                "threshold": 0.5,
            },
        )


def test_status_snapshot_is_plain_data(tmp: Path, probe: torch.Tensor):
    import pickle

    ctrl = _make_controller(tmp, probe, probe)
    ctrl.on_step(_view({"r1": _aligned(probe)}, step=1))
    status = ctrl.status()
    assert status["actuation"] == "per_request"
    assert status["updates_emitted"] == 1
    assert status["policy"]["r1"]["engaged"] is True
    pickle.dumps(status)  # picklable for collective_rpc


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
