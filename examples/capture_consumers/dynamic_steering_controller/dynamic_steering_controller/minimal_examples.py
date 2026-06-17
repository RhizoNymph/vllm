# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal, single-purpose dynamic-steering consumers — one per way the
runtime can be configured. Each is a copy-paste template that emits exactly
ONE kind of :class:`SteeringAction` (or uses one transport), deterministically
(engage on the Nth decode step seen), so the wiring is obvious and testable.

For a real, probe-gated policy that combines these knobs see
:class:`DynamicSteeringController` (the kitchen-sink controller). For the
within-run e2e latency test see :class:`DeterministicOverrideStub`.

Configuration matrix covered here (all decode-tier, cache-safe):

  | example                  | transport | action emitted                       |
  |--------------------------|-----------|--------------------------------------|
  | OverrideExample          | sync      | RequestSteeringOverride (per-request)|
  | GlobalTierExample        | sync      | SteeringVectorUpdate    (global tier)|
  | TierScaleExample         | sync      | SteeringScaleUpdate(tier_gain)       |
  | PerRequestScaleExample   | sync      | SteeringScaleUpdate(req_id)          |
  | MonitorTierExample       | sync      | SteeringMonitorUpdate   (gate tier)  |
  | MonitorRowGateExample    | sync      | SteeringMonitorUpdate(gate_rows)     |
  | AsyncTierExample         | async     | SteeringVectorUpdate via the queue   |

All read a global capture spec on ``(layer, hook)`` so the sync ones get a
``StepCaptureView`` each step; none read client specs. See
``docs/design/dynamic_steering.md`` §5 and the package README.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureSpec
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringAction,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
    SteeringVectorUpdate,
    get_steering_action_queue,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView

_HOOK = "post_mlp"


def _seeded_vector(hidden: int, norm: float) -> np.ndarray:
    """Deterministic (seed 0) unit vector scaled to ``norm`` — identical
    across ranks and runs so the example is reproducible."""
    v = np.random.default_rng(0).standard_normal(hidden).astype(np.float32)
    return np.ascontiguousarray(v / float(np.linalg.norm(v)) * norm, dtype=np.float32)


class _SyncBase:
    """Shared scaffolding for the sync examples.

    Declares the sync-consumer contract (worker-side, exactly-1-step
    latency, global spec on one site), loads engine params, and tracks the
    first request seen in decode as the deterministic ``target``.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    execution: ClassVar[Literal["sync"]] = "sync"
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        self._hidden = model_config.get_hidden_size() if model_config else None
        self._layer = int(params["steer_layer"])
        self._hook = str(params.get("steer_hook", _HOOK))
        self._norm = float(params.get("steer_norm", 8.0))
        self._emit_after = max(1, int(params.get("emit_after_steps", 1)))
        self._vec = (
            _seeded_vector(self._hidden, self._norm) if self._hidden else None
        )
        self._target: str | None = None
        self._decode_steps = 0
        self._emitted = False

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks={self._hook: [self._layer]}, positions="all_generated")

    def status(self) -> dict[str, Any]:
        return {"target": self._target, "emitted": self._emitted}

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def _ready_target(self, view: StepCaptureView) -> str | None:
        """Return the target req_id on the step it reaches ``emit_after``
        decode steps, else ``None`` (and mark emitted)."""
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
            if req.req_id == self._target and req.phase == "decode":
                self._decode_steps += 1
                if self._decode_steps >= self._emit_after:
                    self._emitted = True
                    return self._target
        return None

    def _vecs(self) -> dict[str, dict[int, np.ndarray]]:
        return {self._hook: {self._layer: self._vec}}


class OverrideExample(_SyncBase):
    """Per-request override: route the target's decode tokens to a
    dynamic-pool row holding ``global_decode + vec`` (pure routing)."""

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        target = self._ready_target(view)
        if target is None:
            return None
        return [RequestSteeringOverride(req_id=target, vectors=self._vecs(),
                                       source="override_example")]


class GlobalTierExample(_SyncBase):
    """Global dynamic tier: add ``vec`` to the decode tier for ALL requests
    (composes additively with operator-set decode steering)."""

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._ready_target(view) is None:
            return None
        return [SteeringVectorUpdate(vectors=self._vecs(), phase="decode",
                                     source="global_tier_example")]


class TierScaleExample(_SyncBase):
    """Cheap tier-gain knob: upload the tier vector ONCE, then modulate its
    strength with ``SteeringScaleUpdate(tier_gain=...)`` — no re-upload."""

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._ready_target(view) is None:
            return None
        # One-shot install + a sample strength change. A real policy would
        # emit further tier-gain updates over subsequent steps for free.
        return [
            SteeringVectorUpdate(vectors=self._vecs(), phase="decode",
                                 source="tier_scale_example"),
            SteeringScaleUpdate(scale=self._norm, tier_gain=True,
                                source="tier_scale_example"),
        ]


class PerRequestScaleExample(_SyncBase):
    """Per-request strength via the cheap knob: install a unit override,
    then scale it by ``req_id`` (the runner resolves req_id→dyn_id)."""

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._installed: str | None = None

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        # Step 1: install the override for the target (unit vector).
        if self._installed is None:
            target = self._ready_target(view)
            if target is None:
                return None
            self._installed = target
            self._emitted = False  # allow a follow-up scale next step
            return [RequestSteeringOverride(req_id=target, vectors=self._vecs(),
                                            source="per_request_scale_example")]
        # Step 2+: modulate the installed override's strength by req_id.
        if not self._emitted:
            self._emitted = True
            return [SteeringScaleUpdate(scale=0.5, req_id=self._installed,
                                        source="per_request_scale_example")]
        return None


class MonitorTierExample(_SyncBase):
    """In-graph monitor gating the TIER: install a probe + tier once; the
    kernel gates the tier per token from ``sigmoid(sharpness·(h·probe−thr))``."""

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", 0.0))
        self._sharpness = float(params.get("monitor_sharpness", 8.0))
        self._probe = _seeded_vector(self._hidden, 1.0) if self._hidden else None

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._ready_target(view) is None:
            return None
        return [
            SteeringVectorUpdate(vectors=self._vecs(), phase="decode",
                                 source="monitor_tier_example"),
            SteeringScaleUpdate(scale=self._norm, tier_gain=True,
                                source="monitor_tier_example"),
            SteeringMonitorUpdate(hook=self._hook, layer=self._layer,
                                  probe=self._probe, threshold=self._threshold,
                                  sharpness=self._sharpness,
                                  source="monitor_tier_example"),
        ]


class MonitorRowGateExample(_SyncBase):
    """In-graph monitor gating per-request ROWS: install a probe with
    ``gate_rows=True``. The kernel then gates each decode token's
    per-request row by the probe (prefill rows are never gated)."""

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._threshold = float(params.get("threshold", 0.0))
        self._sharpness = float(params.get("monitor_sharpness", 8.0))
        self._probe = _seeded_vector(self._hidden, 1.0) if self._hidden else None

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._ready_target(view) is None:
            return None
        return [
            SteeringMonitorUpdate(hook=self._hook, layer=self._layer,
                                  probe=self._probe, threshold=self._threshold,
                                  sharpness=self._sharpness, gate_rows=True,
                                  source="monitor_rowgate_example"),
        ]


class AsyncTierExample(CaptureConsumer):
    """ASYNC transport: a worker-side capture consumer that submits a global
    tier update through the action queue (drained on a later step — the
    1–3-step async latency). Unlike the sync examples it implements
    ``on_capture`` (delivered post-D2H on the dispatch thread at finalize),
    not ``on_step``; the update it submits steers *subsequent* decode steps.

    Must run worker-side (``location='worker'``) so the process-global
    action queue is reachable; ``get_steering_action_queue()`` returns
    ``None`` in multi-rank topologies, where the example degrades to a no-op.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    # execution defaults to "async".

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        self._hidden = model_config.get_hidden_size() if model_config else None
        self._layer = int(params["steer_layer"])
        self._hook = str(params.get("steer_hook", _HOOK))
        self._norm = float(params.get("steer_norm", 8.0))
        self._vec = (
            _seeded_vector(self._hidden, self._norm) if self._hidden else None
        )
        self._submitted = 0

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks={self._hook: [self._layer]}, positions="all_generated")

    def on_capture(self, key: Any, tensor: Any, sidecar: dict[str, Any]) -> None:
        # A real async policy would score ``tensor`` (CPU, post-D2H) here.
        # Submit a global decode-tier update; the runner drains it before a
        # subsequent step. Non-throwing: degrade gracefully if no queue.
        queue = get_steering_action_queue()
        if queue is None or self._vec is None:
            return
        if queue.submit(
            SteeringVectorUpdate(
                vectors={self._hook: {self._layer: self._vec}},
                phase="decode",
                source="async_tier_example",
            )
        ):
            self._submitted += 1

    def status(self) -> dict[str, Any]:
        return {"submitted": self._submitted}
