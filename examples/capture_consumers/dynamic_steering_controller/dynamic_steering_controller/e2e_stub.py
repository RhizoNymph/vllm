# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal deterministic sync capture consumer — a reference/test stub.

Where :class:`DynamicSteeringController` makes probe-driven engagement
decisions, this stub is deliberately trivial and deterministic so an
engine-level test can assert the exactly-one-step actuation latency and
per-request targeting end to end:

- The *first* request it sees in decode becomes the single target.
- On that target's ``emit_after_steps``-th decode step it emits exactly
  one :class:`RequestSteeringOverride` carrying a fixed (seeded) vector.
- Every other request is left untouched — the in-batch control.

Config (``--capture-consumers dynamic_steering_e2e:key=val,...`` or the
``LLM(capture_consumers=[{"name": "dynamic_steering_e2e", ...}])`` dict):

- ``steer_layer`` (int, required): layer the override targets.
- ``steer_hook`` (str, default ``post_mlp``): steering hook point.
- ``steer_norm`` (float, default ``20.0``): L2 norm of the vector.
- ``emit_after_steps`` (int, default ``1``): emit on the target's Nth
  decode step (≥1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from vllm.v1.capture.consumer import SyncCaptureConsumer
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView
    from vllm.v1.capture.types import CaptureSpec
    from vllm.v1.worker.steering_action_queue import SteeringAction


class DeterministicOverrideStub(SyncCaptureConsumer):
    """Steers one deterministically-chosen request, exactly once."""

    location: ClassVar[Literal["worker"]] = "worker"
    execution: ClassVar[Literal["sync"]] = "sync"
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        hidden = model_config.get_hidden_size() if model_config is not None else None
        if hidden is None:
            raise ValueError("DeterministicOverrideStub needs a model hidden size.")
        self._layer = int(params["steer_layer"])
        self._hook = str(params.get("steer_hook", "post_mlp"))
        self._emit_after = max(1, int(params.get("emit_after_steps", 1)))
        norm = float(params.get("steer_norm", 20.0))

        # Seeded so the vector is identical across ranks and runs.
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(hidden).astype(np.float32)
        vec = vec / float(np.linalg.norm(vec)) * norm
        self._vector = np.ascontiguousarray(vec, dtype=np.float32)

        self._target: str | None = None
        self._target_decode_steps = 0
        self._emitted = False

    def global_capture_spec(self) -> CaptureSpec:
        from vllm.v1.capture.types import CaptureSpec

        return CaptureSpec(hooks={self._hook: [self._layer]}, positions="all_generated")

    def status(self) -> dict[str, Any]:
        return {
            "target": self._target,
            "target_decode_steps": self._target_decode_steps,
            "emitted": self._emitted,
        }

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._emitted:
            return None

        # Designate the first request observed in decode as the target.
        if self._target is None:
            for req in view.requests:
                if req.phase == "decode":
                    self._target = req.req_id
                    break
        if self._target is None:
            return None

        # Count the target's decode steps; emit on the Nth.
        for req in view.requests:
            if req.req_id != self._target or req.phase != "decode":
                continue
            self._target_decode_steps += 1
            if self._target_decode_steps >= self._emit_after:
                self._emitted = True
                return [
                    RequestSteeringOverride(
                        req_id=self._target,
                        vectors={self._hook: {self._layer: self._vector}},
                        source="dynamic_steering_e2e",
                    )
                ]
        return None


class ConfigurableOverrideStub(SyncCaptureConsumer):
    """Override stub with a configurable companion action — for engine-level
    e2e validation of per-request row gating and req_id-keyed scaling.

    Like :class:`DeterministicOverrideStub` it designates the first request
    seen in decode as the single target and, on that target's
    ``emit_after_steps``-th decode step, emits a
    :class:`RequestSteeringOverride` carrying a seeded vector. The other
    request is the untouched in-batch control. ``mode`` adds a companion
    action emitted *in the same step* (so both take effect together on the
    next step — see the runner's in-order action apply):

    - ``mode="override"`` (default): override only — equivalent to
      :class:`DeterministicOverrideStub`.
    - ``mode="rowgate"``: also emit a :class:`SteeringMonitorUpdate` with
      ``gate_rows=True``. The probe gate is forced fully on/off by the sign
      of the threshold (``gate_on``): a large negative threshold makes
      ``sigmoid(sharpness*(h·probe - threshold)) ≈ 1`` for any residual
      (gate ON → the target's per-request row is applied), a large positive
      threshold forces ``≈ 0`` (gate OFF → the row is suppressed). The
      within-run target-vs-control divergence then isolates the row gate.
    - ``mode="reqscale"``: also emit a :class:`SteeringScaleUpdate`
      targeting the override by ``req_id`` with ``scale`` (default 0.0).
      ``scale=0`` suppresses the target's row → target reverts to the
      control; the no-scale override run diverges. The contrast isolates
      the req_id→dyn_id scale path.

    Config (extends :class:`DeterministicOverrideStub`'s params):

    - ``mode`` (str, default ``override``): ``override`` | ``rowgate`` |
      ``reqscale``.
    - ``gate_on`` (bool, default ``True``): rowgate gate direction.
    - ``scale`` (float, default ``0.0``): reqscale row multiplier.
    - ``monitor_sharpness`` (float, default ``1.0``): rowgate gate slope.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    execution: ClassVar[Literal["sync"]] = "sync"
    reads_client_spec: ClassVar[bool] = False

    # A threshold this large saturates the monitor sigmoid for any plausible
    # residual·probe projection, making the gate a deterministic 0/1.
    _SATURATE = 1.0e6

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        hidden = model_config.get_hidden_size() if model_config is not None else None
        if hidden is None:
            raise ValueError("ConfigurableOverrideStub needs a model hidden size.")
        self._layer = int(params["steer_layer"])
        self._hook = str(params.get("steer_hook", "post_mlp"))
        self._emit_after = max(1, int(params.get("emit_after_steps", 1)))
        self._mode = str(params.get("mode", "override"))
        if self._mode not in ("override", "rowgate", "reqscale", "perrow"):
            raise ValueError(f"unknown mode: {self._mode!r}")
        self._gate_on = bool(params.get("gate_on", True))
        self._scale = float(params.get("scale", 0.0))
        self._sharpness = float(params.get("monitor_sharpness", 1.0))
        norm = float(params.get("steer_norm", 20.0))

        # Seeded so the vector/probe are identical across ranks and runs.
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(hidden).astype(np.float32)
        vec = vec / float(np.linalg.norm(vec)) * norm
        self._vector = np.ascontiguousarray(vec, dtype=np.float32)
        probe = np.random.default_rng(1).standard_normal(hidden).astype(np.float32)
        probe = probe / float(np.linalg.norm(probe))
        self._probe = np.ascontiguousarray(probe, dtype=np.float32)

        self._target: str | None = None
        self._target_decode_steps = 0
        self._emitted = False

    def global_capture_spec(self) -> CaptureSpec:
        from vllm.v1.capture.types import CaptureSpec

        return CaptureSpec(hooks={self._hook: [self._layer]}, positions="all_generated")

    def status(self) -> dict[str, Any]:
        return {
            "target": self._target,
            "mode": self._mode,
            "emitted": self._emitted,
        }

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def _companion(self, target: str) -> list[SteeringAction]:
        if self._mode == "rowgate":
            threshold = -self._SATURATE if self._gate_on else self._SATURATE
            return [
                SteeringMonitorUpdate(
                    hook=self._hook,
                    layer=self._layer,
                    probe=self._probe,
                    threshold=threshold,
                    sharpness=self._sharpness,
                    gate_rows=True,
                    source="dynamic_steering_e2e_cfg",
                )
            ]
        if self._mode == "perrow":
            # PER-ROW monitor: a SteeringMonitorUpdate keyed by req_id gates
            # ONLY the target's override row by its own probe (requires
            # ``enable_row_monitor``). Threshold saturated to force the gate
            # fully ON/OFF — gate ON applies the target's add, gate OFF
            # suppresses it, while the control request is untouched.
            threshold = -self._SATURATE if self._gate_on else self._SATURATE
            return [
                SteeringMonitorUpdate(
                    hook=self._hook,
                    layer=self._layer,
                    probe=self._probe,
                    threshold=threshold,
                    sharpness=self._sharpness,
                    req_id=target,
                    source="dynamic_steering_e2e_cfg",
                )
            ]
        if self._mode == "reqscale":
            return [
                SteeringScaleUpdate(
                    scale=self._scale,
                    req_id=target,
                    source="dynamic_steering_e2e_cfg",
                )
            ]
        return []

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._emitted:
            return None

        if self._target is None:
            for req in view.requests:
                if req.phase == "decode":
                    self._target = req.req_id
                    break
        if self._target is None:
            return None

        for req in view.requests:
            if req.req_id != self._target or req.phase != "decode":
                continue
            self._target_decode_steps += 1
            if self._target_decode_steps >= self._emit_after:
                self._emitted = True
                # Override first so the req_id scale can resolve the freshly
                # registered dyn_id in the same in-order apply pass.
                return [
                    RequestSteeringOverride(
                        req_id=self._target,
                        vectors={self._hook: {self._layer: self._vector}},
                        source="dynamic_steering_e2e_cfg",
                    ),
                    *self._companion(self._target),
                ]
        return None


__all__ = ["ConfigurableOverrideStub", "DeterministicOverrideStub"]
