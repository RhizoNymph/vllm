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

from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView
    from vllm.v1.capture.types import CaptureSpec
    from vllm.v1.worker.steering_action_queue import SteeringAction


class DeterministicOverrideStub:
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

        return CaptureSpec(
            hooks={self._hook: [self._layer]}, positions="all_generated"
        )

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


__all__ = ["DeterministicOverrideStub"]
