# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for DynamicSteeringController and ProbePolicy.

Runs the CaptureSink lifecycle end-to-end against a MagicMock
``VllmConfig``, tempfile probe/steering vectors, and a manually
installed ``SteeringActionQueue``. No engine or GPU required.

Usage: ``python test.py`` or ``pytest test.py``.
"""

from __future__ import annotations

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

from vllm.v1.capture.types import CaptureChunk, CaptureFinalize  # noqa: E402
from vllm.v1.worker.steering_action_queue import (  # noqa: E402
    SteeringActionQueue,
    install_steering_action_queue,
)

HIDDEN = 32


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
    assert p.current_gain() == 2.0
    assert p.engaged


def test_policy_stays_disengaged_below_threshold():
    p = _policy()
    p.observe("r1", np.array([0.49]))
    assert p.current_gain() == 0.0
    assert not p.engaged


def test_policy_hysteresis_holds_engagement():
    p = _policy()
    p.observe("r1", np.array([0.6]))
    assert p.current_gain() == 2.0
    # Dips below threshold but above threshold - hysteresis: stays on.
    p.observe("r1", np.array([0.4]))
    assert p.current_gain() == 2.0
    # Below the disengage level: off, and re-engage needs full threshold.
    p.observe("r1", np.array([0.29]))
    assert p.current_gain() == 0.0
    p.observe("r1", np.array([0.4]))
    assert p.current_gain() == 0.0


def test_policy_ema_smoothing():
    p = _policy(ema_alpha=0.5, hysteresis=0.0)
    p.observe("r1", np.array([1.0]))  # ema = 1.0
    p.observe("r1", np.array([0.0]))  # ema = 0.5
    assert p.aggregate_score() == pytest.approx(0.5)
    p.observe("r1", np.array([0.0]))  # ema = 0.25
    assert p.aggregate_score() == pytest.approx(0.25)


def test_policy_multi_token_chunk_folds_in_order():
    p = _policy(ema_alpha=0.5)
    p.observe("r1", np.array([1.0, 0.0]))
    assert p.aggregate_score() == pytest.approx(0.5)


def test_policy_aggregate_max_vs_mean():
    p_max = _policy(aggregate="max")
    p_max.observe("r1", np.array([0.2]))
    p_max.observe("r2", np.array([0.8]))
    assert p_max.aggregate_score() == pytest.approx(0.8)

    p_mean = _policy(aggregate="mean")
    p_mean.observe("r1", np.array([0.2]))
    p_mean.observe("r2", np.array([0.8]))
    assert p_mean.aggregate_score() == pytest.approx(0.5)


def test_policy_forget_drops_request_and_disengages():
    p = _policy()
    p.observe("r1", np.array([0.9]))
    assert p.current_gain() == 2.0
    p.forget("r1")
    assert p.aggregate_score() is None
    assert p.current_gain() == 0.0
    assert not p.engaged


def test_policy_proportional_gain_ramps():
    p = _policy(gain_mode="proportional", threshold=0.5, hysteresis=0.2, gain=2.0)
    # Engage at threshold: full gain.
    p.observe("r1", np.array([0.5]))
    assert p.current_gain() == pytest.approx(2.0)
    # Mid-band (0.4 between disengage 0.3 and threshold 0.5): half gain.
    p.observe("r1", np.array([0.4]))
    assert p.current_gain() == pytest.approx(1.0)
    # Above threshold clamps at full gain.
    p.observe("r1", np.array([0.9]))
    assert p.current_gain() == pytest.approx(2.0)


def test_policy_emission_gating():
    p = _policy(min_emit_delta=0.5, gain_mode="proportional", gain=2.0)
    # Silent while never engaged.
    assert not p.should_emit(0.0)
    # First non-zero gain emits.
    assert p.should_emit(1.0)
    p.mark_emitted(1.0)
    # Small move suppressed; big move emits.
    assert not p.should_emit(1.2)
    assert p.should_emit(1.6)
    # Engagement flip always emits, even with a huge min_emit_delta.
    assert p.should_emit(0.0)


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
# DynamicSteeringController
# ---------------------------------------------------------------------------


def _mock_config(tp: int = 1, pp: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.model_config.get_hidden_size.return_value = HIDDEN
    cfg.parallel_config.tensor_parallel_size = tp
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
        "monitor_layer": 3,
        "monitor_hook": "post_mlp",
        "probe_path": str(probe_path),
        "steering_vector_path": str(steer_path),
        "threshold": 0.5,
        "hysteresis": 0.2,
        "ema_alpha": 1.0,
        "gain": 2.0,
    }
    params.update(param_overrides)
    return DynamicSteeringController(_mock_config(), params)


def _chunk(rows: torch.Tensor, req: str = "req-1", step: int = 0) -> CaptureChunk:
    return CaptureChunk(
        key=(req, 3, "post_mlp"),
        tensor=rows,
        dtype=rows.dtype,
        row_offset=0,
        step_index=step,
    )


def _aligned_rows(probe: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (probe * scale).unsqueeze(0)


@pytest.fixture()
def tmp() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(autouse=True)
def action_queue():
    q = SteeringActionQueue()
    install_steering_action_queue(q)
    yield q
    install_steering_action_queue(None)


def test_rejects_multi_rank_topology(tmp: Path):
    probe = torch.randn(HIDDEN)
    torch.save(probe, tmp / "p.pt")
    params = {
        "monitor_layer": 0,
        "probe_path": str(tmp / "p.pt"),
        "steering_vector_path": str(tmp / "p.pt"),
        "threshold": 0.5,
    }
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        DynamicSteeringController(_mock_config(tp=2), params)
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        DynamicSteeringController(_mock_config(pp=2), params)


def test_rejects_bad_vectors(tmp: Path):
    probe = torch.randn(HIDDEN)
    with pytest.raises(ValueError, match="hidden_size"):
        _make_controller(tmp, torch.randn(HIDDEN + 1), probe)
    with pytest.raises(ValueError, match="zero norm"):
        _make_controller(tmp, torch.zeros(HIDDEN), probe)
    with pytest.raises(ValueError, match="1-D"):
        _make_controller(tmp, torch.randn(2, HIDDEN), probe)


def test_global_spec_targets_monitor_site(tmp: Path):
    probe = torch.randn(HIDDEN)
    ctrl = _make_controller(tmp, probe, probe, monitor_layer=7)
    spec = ctrl.global_capture_spec()
    assert spec.hooks == {"post_mlp": [7]}
    assert spec.positions == "all_generated"


def test_engage_emits_scaled_update(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    steer = torch.full((HIDDEN,), 3.0)
    ctrl = _make_controller(tmp, probe, steer, steer_layers=[3, 5])

    # Activation perfectly aligned with the probe: cosine = 1.0 > 0.5.
    ctrl.submit_chunk(_chunk(_aligned_rows(probe, scale=4.0)))

    updates = action_queue.drain()
    assert len(updates) == 1
    update = updates[0]
    assert update.phase == "decode"
    assert update.source == "dynamic_steering"
    assert set(update.vectors["post_mlp"].keys()) == {3, 5}
    np.testing.assert_allclose(
        update.vectors["post_mlp"][3],
        np.full(HIDDEN, 6.0, dtype=np.float32),  # steer(3.0) * gain(2.0)
    )


def test_below_threshold_emits_nothing(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe)
    orthogonal = torch.zeros(1, HIDDEN)
    orthogonal[0, 1] = 5.0  # cosine with probe = 0.0
    ctrl.submit_chunk(_chunk(orthogonal))
    assert action_queue.drain() == []


def test_steady_engagement_does_not_reemit(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe)
    ctrl.submit_chunk(_chunk(_aligned_rows(probe), step=0))
    assert len(action_queue.drain()) == 1
    # Same engaged state, binary gain unchanged: no second update.
    ctrl.submit_chunk(_chunk(_aligned_rows(probe), step=1))
    assert action_queue.drain() == []


def test_disengage_emits_zero_vector(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe)
    ctrl.submit_chunk(_chunk(_aligned_rows(probe), step=0))
    action_queue.drain()

    anti = _aligned_rows(probe, scale=-1.0)  # cosine = -1.0 < disengage
    ctrl.submit_chunk(_chunk(anti, step=1))
    updates = action_queue.drain()
    assert len(updates) == 1
    np.testing.assert_allclose(
        updates[0].vectors["post_mlp"][3], np.zeros(HIDDEN, dtype=np.float32)
    )


def test_finalize_forgets_request_and_disengages(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe)
    ctrl.submit_chunk(_chunk(_aligned_rows(probe)))
    action_queue.drain()

    key = ("req-1", 3, "post_mlp")
    ctrl.submit_finalize(CaptureFinalize(key=key))

    # The only engaged request finished: a zeroing update is emitted.
    updates = action_queue.drain()
    assert len(updates) == 1
    np.testing.assert_allclose(
        updates[0].vectors["post_mlp"][3], np.zeros(HIDDEN, dtype=np.float32)
    )

    result = ctrl.get_result(key)
    assert result is not None
    assert result.status == "ok"
    assert result.payload["updates_emitted"] == 2
    assert result.payload["last_score"] == pytest.approx(1.0)


def test_dot_score_mode(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 2.0  # normalized to unit at load
    ctrl = _make_controller(tmp, probe, probe, score="dot", threshold=3.0)
    rows = torch.zeros(1, HIDDEN)
    rows[0, 0] = 2.5  # dot with unit probe = 2.5 < 3.0
    ctrl.submit_chunk(_chunk(rows))
    assert action_queue.drain() == []
    rows[0, 0] = 3.5  # 3.5 >= 3.0
    ctrl.submit_chunk(_chunk(rows, step=1))
    assert len(action_queue.drain()) == 1


def test_no_queue_installed_is_graceful(tmp: Path):
    install_steering_action_queue(None)
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe)
    # Must not raise; decision is computed but unapplied.
    ctrl.submit_chunk(_chunk(_aligned_rows(probe)))


def test_chunk_batch_path(tmp: Path, action_queue):
    probe = torch.zeros(HIDDEN)
    probe[0] = 1.0
    ctrl = _make_controller(tmp, probe, probe, aggregate="mean")
    chunks = [
        _chunk(_aligned_rows(probe), req="req-a"),
        _chunk(_aligned_rows(probe), req="req-b"),
    ]
    ctrl.submit_chunk_batch(chunks)
    assert len(action_queue.drain()) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
