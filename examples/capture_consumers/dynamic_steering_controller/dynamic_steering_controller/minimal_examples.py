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
import torch

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
    tier update through the action queue. Unlike the sync examples it
    implements ``on_capture`` (delivered post-D2H on the dispatch thread),
    not ``on_step``.

    Timing note: ``on_capture`` runs when a request *finalizes* (after its
    output is emitted), so the update a request triggers cannot steer that
    same request — it is drained at the top of a *later* request's step and
    steers that one onward. (For same-request, exactly-one-step latency use
    a sync ``on_step`` consumer instead.) Validated end to end in
    ``tests/v1/worker/test_async_steering_e2e.py``.

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


class ConversationLatchExample:
    """Per-conversation *latched* steering: once a probe trigger fires in a
    conversation, every later decode step — **including subsequent requests of
    the same conversation** — is steered the same ``particular way``.

    Requires the client to tag requests with ``SamplingParams.conversation_id``
    (surfaced on :attr:`StepRequestView.conversation_id`). It is a pure
    host-side sync consumer (no kernels): it reads the residual to detect the
    trigger and emits a sticky :class:`RequestSteeringOverride` that routes the
    request's decode tokens to a dynamic-pool row holding
    ``global_decode + steer_vec``.

    Two ways a request gets steered:

    - **TRIGGER** — a live request not yet overridden whose probe projection
      ``mean(h · probe)`` over the step's residual window exceeds
      ``threshold``. Latches the conversation and overrides this request.
    - **BRIDGE** — a *new* request of an already-latched conversation: it is
      overridden immediately (no re-trigger needed), so the steering carries
      across turns.

    Reads activations (``view.tensors``), never ``req.token_ids`` (empty on the
    v2 runner). ``RequestSteeringOverride`` auto-expires on finish / preempt /
    streaming re-add, so per-request ``armed`` state is pruned to the live set
    each step; the per-conversation ``latched`` map persists (bounded by
    ``max_conversations``, FIFO eviction).

    Params:
      ``steer_layer`` (int, required), ``steer_hook`` (default ``post_block``),
      ``steer_norm`` (default 8.0)   — the override vector ("particular way").
      ``probe_layer`` (default ``steer_layer``),
      ``probe_hook``  (default ``steer_hook``),
      ``threshold``   (default 0.0)  — the trigger detector.
      ``seed``        (default 0)    — deterministic probe + steer vectors.
      ``npz``         (optional path)— load real ``probe``/``steer`` arrays
                                       (diff-of-means etc.); falls back to the
                                       seeded vectors when absent.
      ``max_conversations`` (default 1024) — bound on the latch map.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    execution: ClassVar[Literal["sync"]] = "sync"
    reads_client_spec: ClassVar[bool] = False

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        model_config = getattr(vllm_config, "model_config", None)
        self._hidden = model_config.get_hidden_size() if model_config else None
        self._steer_layer = int(params["steer_layer"])
        self._steer_hook = str(params.get("steer_hook", "post_block"))
        self._steer_norm = float(params.get("steer_norm", 8.0))
        self._probe_layer = int(params.get("probe_layer", self._steer_layer))
        self._probe_hook = str(params.get("probe_hook", self._steer_hook))
        self._threshold = float(params.get("threshold", 0.0))
        self._max_conversations = max(1, int(params.get("max_conversations", 1024)))

        probe, steer = self._load_vectors(params)
        self._probe = probe  # unit probe (np.float32 [hidden]) or None
        self._steer = steer  # steer vector (np.float32 [hidden]) or None
        self._probe_t: torch.Tensor | None = None  # cached device tensor

        # conv_id -> steer vector to (re)apply; persists across requests.
        self._latched: dict[str, np.ndarray] = {}
        # live req_ids already overridden this run (emit-once per request).
        self._armed: set[str] = set()
        self._triggers = 0
        self._bridges = 0

    def _load_vectors(
        self, params: dict[str, Any]
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._hidden is None:
            return None, None
        npz = params.get("npz")
        if npz:
            data = np.load(npz)
            probe = np.ascontiguousarray(data["probe"], dtype=np.float32)
            steer = np.ascontiguousarray(data["steer"], dtype=np.float32)
            return probe, steer
        seed = int(params.get("seed", 0))
        probe = _seeded_vector(self._hidden, 1.0)
        # Steer along the probe direction by default: deterministic and a
        # sensible self-contained stand-in for a real diff-of-means vector.
        steer = np.ascontiguousarray(
            probe * self._steer_norm
            if seed == 0
            else _seeded_vector(self._hidden, self._steer_norm),
            dtype=np.float32,
        )
        return probe, steer

    def global_capture_spec(self) -> CaptureSpec:
        # Tap the probe site so the residual is in ``view.tensors`` each step.
        return CaptureSpec(
            hooks={self._probe_hook: [self._probe_layer]},
            positions="all_generated",
        )

    def status(self) -> dict[str, Any]:
        return {
            "latched_conversations": len(self._latched),
            "armed_requests": len(self._armed),
            "triggers": self._triggers,
            "bridges": self._bridges,
        }

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    def _steer_vectors(self, vec: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
        return {self._steer_hook: {self._steer_layer: vec}}

    def _latch(self, cid: str, vec: np.ndarray) -> None:
        """Record the conversation's steer vector, FIFO-evicting if full."""
        if cid not in self._latched and len(self._latched) >= self._max_conversations:
            # Drop the oldest conversation (dict preserves insertion order).
            self._latched.pop(next(iter(self._latched)))
        self._latched[cid] = vec

    def _projection(self, view: StepCaptureView, start: int, end: int) -> float:
        """Mean probe projection over a request's residual window."""
        x = view.tensors[(self._probe_layer, self._probe_hook)][start:end]
        if self._probe_t is None or self._probe_t.device != x.device:
            self._probe_t = torch.as_tensor(
                self._probe, dtype=torch.float32, device=x.device
            )
        return float((x.float() @ self._probe_t).mean())

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        if self._steer is None or self._probe is None:
            return None
        actions: list[SteeringAction] = []
        live: set[str] = set()
        for req in view.requests:
            live.add(req.req_id)
            cid = req.conversation_id
            # Latching keys on the conversation id and only decode tokens are
            # routed by the override, so untagged / prefill rows are skipped.
            if cid is None or req.phase != "decode":
                continue
            if req.req_id in self._armed:
                continue
            if cid in self._latched:
                # BRIDGE: a new request of an already-latched conversation.
                actions.append(
                    RequestSteeringOverride(
                        req_id=req.req_id,
                        vectors=self._steer_vectors(self._latched[cid]),
                        source="conversation_latch_example",
                    )
                )
                self._armed.add(req.req_id)
                self._bridges += 1
                continue
            # TRIGGER: probe this request's residual.
            if self._projection(view, req.start, req.end) > self._threshold:
                self._latch(cid, self._steer)
                actions.append(
                    RequestSteeringOverride(
                        req_id=req.req_id,
                        vectors=self._steer_vectors(self._steer),
                        source="conversation_latch_example",
                    )
                )
                self._armed.add(req.req_id)
                self._triggers += 1
        # Prune finished requests; the per-conversation latch persists.
        self._armed &= live
        return actions or None
