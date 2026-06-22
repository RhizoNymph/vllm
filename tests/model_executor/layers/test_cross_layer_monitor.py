# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the opt-in cross-layer monitor (Phase 2, §8).

When ``SteeringConfig.enable_cross_layer_monitor`` is set, the model runner
stamps ``module._cross_layer_monitor = True`` on every steerable layer and
``apply_layer_steering`` switches the in-graph monitor from the default
*same-hook fused* gate to the *cross-layer* mode: it emits the standalone
mutating ``steering_monitor`` op at every steered hook (writing the shared
``steering_token_scales``/``steering_row_gate`` buffers that later layers read)
and bypasses the fused gate by passing the always-False ``steering_monitor_off``
flag to ``apply_steering`` (so the per-token gate applies exactly once, not
twice at the probe site).

These tests mock the CUDA-only ``torch.ops.vllm.{apply_steering,
steering_monitor}`` ops (they have no CPU dispatch) and assert the branch
behaviour, plus the new buffer and the config hash/threading.
"""

import torch
import torch.nn as nn

from vllm.config.steering import SteeringConfig
from vllm.model_executor.layers.steering import (
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_PARAMS_ATTR,
    HOOK_POINT_MONITOR_PROBE_ATTR,
    SteeringHookPoint,
    apply_layer_steering,
    register_steering_buffers,
)

HIDDEN = 8
MAX_TOKENS = 4
MAX_CONFIGS = 2
HOOK = SteeringHookPoint.POST_MLP


def _make_module() -> nn.Module:
    m = nn.Module()
    m.layer_idx = 0
    register_steering_buffers(
        m,
        HIDDEN,
        max_steering_tokens=MAX_TOKENS,
        max_steering_configs=MAX_CONFIGS,
        dtype=torch.float32,
    )
    return m


class _OpRecorder:
    """Replace the CUDA-only ops with recorders so apply_layer_steering's
    control flow can be exercised on CPU."""

    def __init__(self, monkeypatch):
        self.apply_calls: list[tuple] = []
        self.monitor_calls: list[tuple] = []
        ns = torch.ops.vllm

        def fake_apply(*args):
            self.apply_calls.append(args)
            return args[0].clone()  # mirror the non-aliasing contract

        def fake_monitor(*args):
            self.monitor_calls.append(args)
            return None

        monkeypatch.setattr(ns, "apply_steering", fake_apply, raising=False)
        monkeypatch.setattr(ns, "steering_monitor", fake_monitor, raising=False)


# ---------------------------------------------------------------------------
# register_steering_buffers: the always-False bypass flag
# ---------------------------------------------------------------------------


def test_register_adds_monitor_off_flag():
    m = _make_module()
    assert hasattr(m, "steering_monitor_off")
    off = m.steering_monitor_off
    assert off.shape == (1,)
    assert off.dtype == torch.bool
    assert not bool(off.item())  # always False, never written


# ---------------------------------------------------------------------------
# apply_layer_steering branch behaviour
# ---------------------------------------------------------------------------

_APPLY_ACTIVE_IDX = 10  # position of the fused-monitor active flag in apply_steering


def test_default_mode_no_standalone_op_real_active(monkeypatch):
    """Flag off (default): no standalone monitor op; apply_steering gets the
    REAL per-hook monitor-active flag (same-hook fused gate path)."""
    rec = _OpRecorder(monkeypatch)
    m = _make_module()  # _cross_layer_monitor not set -> getattr default False
    h = torch.randn(MAX_TOKENS, HIDDEN)

    apply_layer_steering(m, h, HOOK)

    assert rec.monitor_calls == []  # standalone op NOT emitted
    assert len(rec.apply_calls) == 1
    passed_active = rec.apply_calls[0][_APPLY_ACTIVE_IDX]
    assert passed_active is getattr(m, HOOK_POINT_MONITOR_ACTIVE_ATTR[HOOK])


def test_cross_layer_mode_emits_op_and_bypasses_fused(monkeypatch):
    """Flag on: standalone mutating monitor op emitted with the per-hook
    probe/params/active + shared gate buffers; apply_steering gets the
    always-False steering_monitor_off flag (fused gate bypassed)."""
    rec = _OpRecorder(monkeypatch)
    m = _make_module()
    m._cross_layer_monitor = True
    h = torch.randn(MAX_TOKENS, HIDDEN)

    apply_layer_steering(m, h, HOOK)

    # standalone op fired exactly once, reading the right buffers
    assert len(rec.monitor_calls) == 1
    margs = rec.monitor_calls[0]
    assert margs[0] is h
    assert margs[1] is getattr(m, HOOK_POINT_MONITOR_PROBE_ATTR[HOOK])
    assert margs[2] is getattr(m, HOOK_POINT_MONITOR_PARAMS_ATTR[HOOK])
    assert margs[3] is getattr(m, HOOK_POINT_MONITOR_ACTIVE_ATTR[HOOK])
    assert margs[4] is m.steering_token_scales
    assert margs[5] is m.steering_decode_mask
    assert margs[6] is m.steering_row_gate

    # apply_steering's fused gate is bypassed via the always-False flag
    assert len(rec.apply_calls) == 1
    passed_active = rec.apply_calls[0][_APPLY_ACTIVE_IDX]
    assert passed_active is m.steering_monitor_off
    assert passed_active is not getattr(m, HOOK_POINT_MONITOR_ACTIVE_ATTR[HOOK])


def test_explicit_false_flag_matches_default(monkeypatch):
    """_cross_layer_monitor=False behaves exactly like the unset default."""
    rec = _OpRecorder(monkeypatch)
    m = _make_module()
    m._cross_layer_monitor = False
    apply_layer_steering(m, torch.randn(MAX_TOKENS, HIDDEN), HOOK)
    assert rec.monitor_calls == []
    assert (
        rec.apply_calls[0][_APPLY_ACTIVE_IDX]
        is getattr(m, HOOK_POINT_MONITOR_ACTIVE_ATTR[HOOK])
    )


def test_disabled_steering_short_circuits(monkeypatch):
    """No table buffers (steering disabled) -> no ops, returns input."""
    rec = _OpRecorder(monkeypatch)
    m = nn.Module()
    m.layer_idx = 0
    m._cross_layer_monitor = True  # must not matter when no buffers exist
    h = torch.randn(MAX_TOKENS, HIDDEN)
    out = apply_layer_steering(m, h, HOOK)
    assert out is h
    assert rec.apply_calls == [] and rec.monitor_calls == []


# ---------------------------------------------------------------------------
# config: hash + default
# ---------------------------------------------------------------------------


def test_config_default_off():
    assert SteeringConfig().enable_cross_layer_monitor is False


def test_config_hash_changes_with_flag():
    off = SteeringConfig(enable_cross_layer_monitor=False)
    on = SteeringConfig(enable_cross_layer_monitor=True)
    assert off.compute_hash() != on.compute_hash()
