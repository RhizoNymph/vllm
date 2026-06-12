# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic steering controller — sync-execution consumer (Phase 1a).

Closes the activation→steering feedback loop with exactly-one-step
latency: it declares ``execution = "sync"``, so the engine runs
``on_step`` on the model-runner step thread immediately after every
forward, handing it zero-copy GPU views of the monitored residual
(captured via the graph-safe persistent-buffer path — no eager
forcing). Probe scores are computed with one small GEMM on the
buffer's device, a single tiny D2H brings per-token scalars to the
policy, and the returned steering actions are applied before the next
step builds its steering tables.

Two actuation modes (``actuation`` param):

- ``"per_request"`` (default): each request runs its own
  EMA/hysteresis engagement state machine; a firing request gets a
  ``RequestSteeringOverride`` routing *its* decode tokens to a
  dynamic-pool steering row. Other requests are untouched. Admission
  state, scheduler accounting, and prefix caching are unaffected.
- ``"global"``: per-request scores aggregate (max/mean) into one
  engagement state driving the global decode tier via
  ``SteeringVectorUpdate`` — the Phase 0 behavior, now at 1-step
  latency.

Determinism: sync consumers run on every TP rank with byte-identical
views (the monitored residual is read post-all-reduce). ``on_step`` is
a pure function of the view and policy state, and the state evolves
identically per rank — no RNG, no clocks, no unordered iteration.

See ``docs/design/dynamic_steering.md`` and the companion README.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch

from vllm.config.steering_types import unpack_steering_vectors
from vllm.logger import init_logger
from vllm.v1.capture.types import CaptureSpec
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringAction,
    SteeringVectorUpdate,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView

logger = init_logger(__name__)

_MONITOR_HOOKS = frozenset({"pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out"})
# Steering can only target the steering hook points.
_STEER_HOOKS = frozenset({"pre_attn", "post_attn", "post_mlp"})
_SCORE_MODES = frozenset({"cosine", "dot"})
_GAIN_MODES = frozenset({"binary", "proportional"})
_AGGREGATES = frozenset({"max", "mean"})
_ACTUATIONS = frozenset({"per_request", "global"})

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Policy (pure, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Tuning knobs for the engagement state machines.

    ``threshold`` is the engage level on the (EMA-smoothed) probe
    score; once engaged, disengage happens only below ``threshold -
    hysteresis`` (anti-flap). ``gain_mode``:

    - ``"binary"``: emit ``gain`` while engaged, ``0.0`` otherwise.
    - ``"proportional"``: while engaged, scale linearly from 0 at the
      disengage level to ``gain`` at ``threshold`` (clamped).

    ``min_emit_delta`` suppresses emissions whose gain moved less than
    this since the last one (engagement flips always emit).
    ``aggregate`` applies only to global actuation.
    """

    threshold: float
    hysteresis: float = 0.0
    ema_alpha: float = 0.25
    gain: float = 1.0
    gain_mode: str = "binary"
    aggregate: str = "max"
    min_emit_delta: float = 0.05

    def __post_init__(self) -> None:
        if self.hysteresis < 0.0:
            raise ValueError("hysteresis must be >= 0")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.gain_mode not in _GAIN_MODES:
            raise ValueError(f"gain_mode must be one of {sorted(_GAIN_MODES)}")
        if self.aggregate not in _AGGREGATES:
            raise ValueError(f"aggregate must be one of {sorted(_AGGREGATES)}")
        if self.min_emit_delta < 0.0:
            raise ValueError("min_emit_delta must be >= 0")


@dataclass
class _Engagement:
    """One hysteresis state machine over an EMA-smoothed score."""

    ema: float | None = None
    engaged: bool = False
    last_emitted_gain: float | None = None
    last_seen_step: int = 0


class ProbePolicy:
    """Per-key (request id or ``"__global__"``) engagement machines.

    Pure and deterministic; the owning consumer serializes access
    (single-threaded on the step thread).
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._cfg = config
        self._states: dict[str, _Engagement] = {}

    def observe(self, key: str, scores: np.ndarray, step: int = 0) -> None:
        """Fold ``scores`` (in token order) into ``key``'s EMA."""
        state = self._states.setdefault(key, _Engagement())
        state.last_seen_step = step
        if scores.size == 0:
            return
        a = self._cfg.ema_alpha
        ema = state.ema
        for s in scores.tolist():
            ema = s if ema is None else (1.0 - a) * ema + a * s
        state.ema = ema

    def forget(self, key: str) -> None:
        self._states.pop(key, None)

    def prune_unseen(self, current_step: int, max_age: int) -> list[str]:
        """Drop state for keys not observed within ``max_age`` steps.

        Returns the pruned keys.
        """
        stale = [
            key
            for key, state in self._states.items()
            if current_step - state.last_seen_step > max_age
        ]
        for key in stale:
            del self._states[key]
        return stale

    def keys(self) -> list[str]:
        return list(self._states.keys())

    def score(self, key: str) -> float | None:
        state = self._states.get(key)
        return None if state is None else state.ema

    def aggregate_score(self) -> float | None:
        """Aggregate across tracked request keys (global actuation)."""
        emas = [
            s.ema
            for key, s in self._states.items()
            if s.ema is not None and key != "__global__"
        ]
        if not emas:
            return None
        return max(emas) if self._cfg.aggregate == "max" else sum(emas) / len(emas)

    def _gain_from(self, state: _Engagement, score: float | None) -> float:
        t_high = self._cfg.threshold
        t_low = t_high - self._cfg.hysteresis
        if score is None:
            state.engaged = False
            return 0.0
        if state.engaged:
            if score < t_low:
                state.engaged = False
        elif score >= t_high:
            state.engaged = True
        if not state.engaged:
            return 0.0
        if self._cfg.gain_mode == "binary":
            return self._cfg.gain
        span = max(t_high - t_low, _EPS)
        frac = (score - t_low) / span
        return self._cfg.gain * min(max(frac, 0.0), 1.0)

    def current_gain(self, key: str) -> float:
        """Advance ``key``'s state machine and return its gain."""
        state = self._states.setdefault(key, _Engagement())
        return self._gain_from(state, state.ema)

    def current_global_gain(self, step: int = 0) -> float:
        """Advance the aggregate state machine (key ``"__global__"``)."""
        state = self._states.setdefault("__global__", _Engagement())
        state.last_seen_step = step
        return self._gain_from(state, self.aggregate_score())

    def should_emit(self, key: str, gain: float) -> bool:
        state = self._states.setdefault(key, _Engagement())
        last = state.last_emitted_gain
        if last is None:
            return gain != 0.0
        if (gain == 0.0) != (last == 0.0):
            return True
        return abs(gain - last) >= self._cfg.min_emit_delta

    def mark_emitted(self, key: str, gain: float) -> None:
        self._states.setdefault(key, _Engagement()).last_emitted_gain = gain

    def engaged(self, key: str) -> bool:
        state = self._states.get(key)
        return state.engaged if state is not None else False

    def snapshot(self) -> dict[str, dict[str, float | bool | None]]:
        """Picklable policy state for the status endpoint."""
        return {
            key: {
                "ema": state.ema,
                "engaged": state.engaged,
                "last_emitted_gain": state.last_emitted_gain,
            }
            for key, state in self._states.items()
        }


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SteerBank:
    """Per-layer steering vectors for one hook point (float32)."""

    hook: str
    vectors: dict[int, np.ndarray] = field(default_factory=dict)


class DynamicSteeringController:
    """Probe-gated dynamic steering, sync execution.

    Engine parameters (``--capture-consumers dynamic_steering:...``):

    - ``monitor_layer`` (int, required), ``monitor_hook`` (str, default
      ``"post_mlp"``): the probed site.
    - ``probe_path`` (str): ``torch.save``'d 1-D tensor, unit-normalized
      at load; OR ``probe_packed_path`` (str): packed-JSON
      (``SteeringHookPacked``) file whose ``monitor_hook`` entry carries
      the probe row for ``monitor_layer``.
    - ``steering_vector_path`` (str): ``torch.save``'d 1-D tensor applied
      (scaled by gain) at every ``steer_layers`` entry; OR
      ``steering_packed_path`` (str): packed-JSON file — its
      ``steer_hook`` entry defines per-layer vectors (``layer_indices``
      = the steer layers; optional per-row ``scales`` fold in at load).
    - ``steer_layers`` (list[int], default ``[monitor_layer]``),
      ``steer_hook`` (str, default ``monitor_hook``; must be a steering
      hook point). Ignored layer-wise when ``steering_packed_path``
      provides per-layer vectors.
    - ``actuation`` (``"per_request"`` default | ``"global"``).
    - ``score`` (``"cosine"`` default | ``"dot"``).
    - policy knobs: ``threshold`` (required), ``hysteresis``,
      ``ema_alpha``, ``gain``, ``gain_mode``, ``aggregate``,
      ``min_emit_delta`` — see :class:`PolicyConfig`.
    - ``forget_after_steps`` (int, default 16): per-request policy state
      is pruned after this many steps without the request appearing in
      the view.
    - ``sync_budget_ms`` (float, default 5.0): soft per-step budget
      surfaced by the engine's over-budget warning.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    execution: ClassVar[Literal["sync"]] = "sync"
    reads_client_spec: ClassVar[bool] = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        params: dict[str, Any],
    ) -> None:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        pp = int(getattr(parallel_config, "pipeline_parallel_size", 1))
        if pp != 1:
            raise ValueError(
                f"DynamicSteeringController requires pipeline_parallel_size=1 "
                f"(got pp={pp}); see docs/design/dynamic_steering.md §6."
            )

        model_config = getattr(vllm_config, "model_config", None)
        hidden_size = (
            model_config.get_hidden_size() if model_config is not None else None
        )

        self._monitor_layer = int(params["monitor_layer"])
        self._monitor_hook = str(params.get("monitor_hook", "post_mlp"))
        if self._monitor_hook not in _MONITOR_HOOKS:
            raise ValueError(
                f"monitor_hook={self._monitor_hook!r} not one of "
                f"{sorted(_MONITOR_HOOKS)}"
            )
        self._monitor_key = (self._monitor_layer, self._monitor_hook)

        self._steer_hook = str(params.get("steer_hook", self._monitor_hook))
        if self._steer_hook not in _STEER_HOOKS:
            raise ValueError(
                f"steer_hook={self._steer_hook!r} not one of {sorted(_STEER_HOOKS)}"
            )

        self._actuation = str(params.get("actuation", "per_request"))
        if self._actuation not in _ACTUATIONS:
            raise ValueError(
                f"actuation={self._actuation!r} not one of {sorted(_ACTUATIONS)}"
            )

        self._score_mode = str(params.get("score", "cosine"))
        if self._score_mode not in _SCORE_MODES:
            raise ValueError(
                f"score={self._score_mode!r} not one of {sorted(_SCORE_MODES)}"
            )

        self._probe = self._load_probe(params, hidden_size)
        self._bank = self._load_steer_bank(params, hidden_size)

        self._policy = ProbePolicy(
            PolicyConfig(
                threshold=float(params["threshold"]),
                hysteresis=float(params.get("hysteresis", 0.0)),
                ema_alpha=float(params.get("ema_alpha", 0.25)),
                gain=float(params.get("gain", 1.0)),
                gain_mode=str(params.get("gain_mode", "binary")),
                aggregate=str(params.get("aggregate", "max")),
                min_emit_delta=float(params.get("min_emit_delta", 0.05)),
            )
        )
        self._forget_after_steps = int(params.get("forget_after_steps", 16))
        self.sync_budget_ms = float(params.get("sync_budget_ms", 5.0))

        # Device-side probe cache, materialized lazily on first on_step
        # from the view tensor's device.
        self._probe_gpu: torch.Tensor | None = None
        self._updates_emitted = 0

    # ------------------------------------------------------------------
    # Vector / bank loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pt_vector(
        path: pathlib.Path, hidden_size: int | None, param_name: str
    ) -> np.ndarray:
        vector = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(vector, torch.Tensor):
            raise TypeError(
                f"DynamicSteeringController: {param_name}={path} did not "
                f"load a torch.Tensor (got {type(vector).__name__})."
            )
        if vector.dim() != 1:
            raise ValueError(
                f"DynamicSteeringController: {param_name} must be 1-D, "
                f"got shape {tuple(vector.shape)}."
            )
        if hidden_size is not None and vector.shape[0] != hidden_size:
            raise ValueError(
                f"DynamicSteeringController: {param_name} size "
                f"{vector.shape[0]} does not match model hidden_size "
                f"{hidden_size}."
            )
        return np.ascontiguousarray(
            vector.to(dtype=torch.float32).numpy(), dtype=np.float32
        )

    @staticmethod
    def _load_packed_file(path: pathlib.Path) -> dict[str, dict[int, np.ndarray]]:
        with open(path) as f:
            payload = json.load(f)
        unpacked = unpack_steering_vectors(payload)
        if not unpacked:
            raise ValueError(
                f"DynamicSteeringController: packed file {path} decoded to "
                f"an empty spec."
            )
        return unpacked

    def _load_probe(
        self, params: dict[str, Any], hidden_size: int | None
    ) -> np.ndarray:
        if "probe_packed_path" in params:
            unpacked = self._load_packed_file(
                pathlib.Path(str(params["probe_packed_path"]))
            )
            hook_rows = unpacked.get(self._monitor_hook, {})
            vec = hook_rows.get(self._monitor_layer)
            if vec is None:
                raise ValueError(
                    f"DynamicSteeringController: probe_packed_path has no "
                    f"row for ({self._monitor_hook!r}, layer "
                    f"{self._monitor_layer})."
                )
            probe = np.ascontiguousarray(vec, dtype=np.float32)
            if hidden_size is not None and probe.shape[0] != hidden_size:
                raise ValueError(
                    f"DynamicSteeringController: packed probe size "
                    f"{probe.shape[0]} does not match model hidden_size "
                    f"{hidden_size}."
                )
        elif "probe_path" in params:
            probe = self._load_pt_vector(
                pathlib.Path(str(params["probe_path"])), hidden_size, "probe_path"
            )
        else:
            raise ValueError(
                "DynamicSteeringController: one of probe_path / "
                "probe_packed_path is required."
            )
        norm = float(np.linalg.norm(probe))
        if norm == 0.0:
            raise ValueError("DynamicSteeringController: probe has zero norm.")
        return probe / norm

    def _load_steer_bank(
        self, params: dict[str, Any], hidden_size: int | None
    ) -> _SteerBank:
        if "steering_packed_path" in params:
            unpacked = self._load_packed_file(
                pathlib.Path(str(params["steering_packed_path"]))
            )
            hook_rows = unpacked.get(self._steer_hook)
            if not hook_rows:
                raise ValueError(
                    f"DynamicSteeringController: steering_packed_path has "
                    f"no {self._steer_hook!r} entry."
                )
            vectors = {
                int(layer): np.ascontiguousarray(vec, dtype=np.float32)
                for layer, vec in hook_rows.items()
            }
        elif "steering_vector_path" in params:
            vec = self._load_pt_vector(
                pathlib.Path(str(params["steering_vector_path"])),
                hidden_size,
                "steering_vector_path",
            )
            steer_layers = params.get("steer_layers")
            if steer_layers is None:
                steer_layers = [self._monitor_layer]
            layers = [int(layer) for layer in steer_layers]
            if not layers:
                raise ValueError("steer_layers must not be empty")
            vectors = {layer: vec for layer in layers}
        else:
            raise ValueError(
                "DynamicSteeringController: one of steering_vector_path / "
                "steering_packed_path is required."
            )
        return _SteerBank(hook=self._steer_hook, vectors=vectors)

    # ------------------------------------------------------------------
    # Consumer-API surface
    # ------------------------------------------------------------------

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            hooks={self._monitor_hook: [self._monitor_layer]},
            positions="all_generated",
        )

    def status(self) -> dict[str, Any]:
        """Policy snapshot for ``GET /v1/steering/dynamic``."""
        return {
            "actuation": self._actuation,
            "monitor": [self._monitor_layer, self._monitor_hook],
            "steer_hook": self._bank.hook,
            "steer_layers": sorted(self._bank.vectors.keys()),
            "updates_emitted": self._updates_emitted,
            "policy": self._policy.snapshot(),
        }

    def shutdown(self, timeout: float = 30.0) -> None:  # noqa: B027
        pass

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        acts = view.tensors.get(self._monitor_key)
        if acts is None or acts.shape[0] == 0:
            return None

        if self._probe_gpu is None or self._probe_gpu.device != acts.device:
            self._probe_gpu = torch.from_numpy(self._probe).to(acts.device)

        # One fixed-shape GEMV on the buffer's device, then a single
        # tiny D2H of per-token scalars. The view tensors are only
        # valid inside on_step, so all reads happen here.
        acts_f32 = acts.to(dtype=torch.float32)
        proj = acts_f32 @ self._probe_gpu
        if self._score_mode == "cosine":
            norms = torch.linalg.vector_norm(acts_f32, dim=-1)
            proj = proj / torch.clamp(norms, min=_EPS)
        scores = proj.cpu().numpy()

        if self._actuation == "per_request":
            return self._per_request_actions(view, scores)
        return self._global_actions(view, scores)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scaled_vectors(self, gain: float) -> dict[str, dict[int, np.ndarray]]:
        return {
            self._bank.hook: {
                layer: np.ascontiguousarray(vec * gain, dtype=np.float32)
                for layer, vec in self._bank.vectors.items()
            }
        }

    def _per_request_actions(
        self, view: StepCaptureView, scores: np.ndarray
    ) -> list[SteeringAction]:
        actions: list[SteeringAction] = []
        for req in view.requests:
            self._policy.observe(req.req_id, scores[req.start : req.end], view.step)
            if req.phase != "decode":
                # Prefill scores inform the EMA so engagement can fire
                # on the first decode step, but no action is emitted
                # (overrides are decode-only).
                continue
            gain = self._policy.current_gain(req.req_id)
            if not self._policy.should_emit(req.req_id, gain):
                continue
            vectors = None if gain == 0.0 else self._scaled_vectors(gain)
            actions.append(
                RequestSteeringOverride(
                    req_id=req.req_id,
                    vectors=vectors,
                    source="dynamic_steering",
                )
            )
            self._policy.mark_emitted(req.req_id, gain)
            self._updates_emitted += 1

        # Prune state for departed requests. No clear action is needed:
        # their overrides were already released by the engine's
        # finish/preemption hooks; only local policy state remains.
        self._policy.prune_unseen(view.step, self._forget_after_steps)
        return actions

    def _global_actions(
        self, view: StepCaptureView, scores: np.ndarray
    ) -> list[SteeringAction]:
        for req in view.requests:
            self._policy.observe(req.req_id, scores[req.start : req.end], view.step)
        self._policy.prune_unseen(view.step, self._forget_after_steps)

        gain = self._policy.current_global_gain(view.step)
        if not self._policy.should_emit("__global__", gain):
            return []
        self._policy.mark_emitted("__global__", gain)
        self._updates_emitted += 1
        return [
            SteeringVectorUpdate(
                vectors=self._scaled_vectors(gain),
                phase="decode",
                source="dynamic_steering",
            )
        ]


__all__ = [
    "DynamicSteeringController",
    "PolicyConfig",
    "ProbePolicy",
]
