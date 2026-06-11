# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic steering controller — Phase 0 prototype.

A capture consumer that closes the activation→steering feedback loop:
it observes the residual stream at a configured ``(layer, hook)`` via a
*global* capture spec (CUDA-graph-safe, no eager forcing), projects each
generated token's activation onto a probe direction, runs a
threshold/hysteresis policy over the per-request scores, and modulates
the *global decode* steering vector through the in-process
``SteeringActionQueue``.

Implements ``CaptureSink`` directly (not ``CaptureConsumer``) because
the batched adapter only delivers tensors at request finalize — far too
late for a feedback loop. Direct sinks receive ``submit_chunk`` per
step on the capture dispatch thread, giving a loop latency of one to a
few decode steps.

Scope (enforced):

- ``tensor_parallel_size == 1`` and ``pipeline_parallel_size == 1``
  (consumers exist on TP rank 0 only; pushing steering updates from one
  rank would diverge the others).
- Actuation is **global decode tier** only. Per-request and prefill
  actuation are Phase 1+ — see ``docs/design/dynamic_steering.md``.

See the companion ``README.md`` for parameters and usage.
"""

from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
)
from vllm.v1.worker.steering_action_queue import (
    SteeringVectorUpdate,
    get_steering_action_queue,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_HOOK_NAMES = frozenset({"pre_attn", "post_attn", "post_mlp", "mlp_in", "mlp_out"})
_SCORE_MODES = frozenset({"cosine", "dot"})
_GAIN_MODES = frozenset({"binary", "proportional"})
_AGGREGATES = frozenset({"max", "mean"})

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Policy (pure, unit-testable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Tuning knobs for :class:`ProbePolicy`.

    ``threshold`` is the engage level on the aggregated probe score;
    once engaged, the policy disengages only below ``threshold -
    hysteresis`` (anti-flap). ``ema_alpha`` smooths each request's
    per-token scores. ``gain`` is the maximum steering scale.
    ``gain_mode``:

    - ``"binary"``: emit ``gain`` while engaged, ``0.0`` otherwise.
    - ``"proportional"``: while engaged, scale linearly from 0 at the
      disengage level to ``gain`` at ``threshold`` (clamped), so the
      intervention strength tracks how strongly the probe fires.

    ``min_emit_delta`` suppresses updates whose gain moved less than
    this since the last emission (engagement flips always emit).
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


class ProbePolicy:
    """Hysteresis gate over per-request EMA-smoothed probe scores.

    Not thread-safe; the owning controller serializes access.
    """

    def __init__(self, config: PolicyConfig) -> None:
        self._cfg = config
        self._ema: dict[str, float] = {}
        self._engaged = False
        self._last_emitted_gain: float | None = None

    def observe(self, req_id: str, scores: np.ndarray) -> None:
        """Fold ``scores`` (per-token, in step order) into ``req_id``'s EMA."""
        if scores.size == 0:
            return
        a = self._cfg.ema_alpha
        ema = self._ema.get(req_id)
        for s in scores.tolist():
            ema = s if ema is None else (1.0 - a) * ema + a * s
        assert ema is not None
        self._ema[req_id] = ema

    def forget(self, req_id: str) -> None:
        self._ema.pop(req_id, None)

    def aggregate_score(self) -> float | None:
        """Aggregate across active requests; ``None`` when no state."""
        if not self._ema:
            return None
        values = self._ema.values()
        if self._cfg.aggregate == "max":
            return max(values)
        return sum(values) / len(values)

    def current_gain(self) -> float:
        """Advance the engagement state machine and return the gain."""
        agg = self.aggregate_score()
        t_high = self._cfg.threshold
        t_low = t_high - self._cfg.hysteresis
        if agg is None:
            # No active requests: disengage.
            self._engaged = False
            return 0.0
        if self._engaged:
            if agg < t_low:
                self._engaged = False
        else:
            if agg >= t_high:
                self._engaged = True
        if not self._engaged:
            return 0.0
        if self._cfg.gain_mode == "binary":
            return self._cfg.gain
        # Proportional: 0 at the disengage level, full gain at threshold.
        span = max(t_high - t_low, _EPS)
        frac = (agg - t_low) / span
        return self._cfg.gain * min(max(frac, 0.0), 1.0)

    def should_emit(self, gain: float) -> bool:
        last = self._last_emitted_gain
        if last is None:
            # Never emitted: only break silence for a non-zero gain.
            return gain != 0.0
        if (gain == 0.0) != (last == 0.0):
            # Engagement flip always emits so disengage cleanly zeroes
            # the steering vector.
            return True
        return abs(gain - last) >= self._cfg.min_emit_delta

    def mark_emitted(self, gain: float) -> None:
        self._last_emitted_gain = gain

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def last_emitted_gain(self) -> float | None:
        return self._last_emitted_gain


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class DynamicSteeringController:
    """Probe-gated dynamic steering over the global decode tier.

    Engine parameters (``--capture-consumers dynamic_steering:...`` flat
    scalars, or the dict form via the Python API):

    - ``monitor_layer`` (int, required): layer whose residual is probed.
    - ``monitor_hook`` (str, default ``"post_mlp"``).
    - ``probe_path`` (str, required): ``torch.save``'d 1-D float tensor;
      normalized to unit length at load.
    - ``steering_vector_path`` (str, required): ``torch.save``'d 1-D
      float tensor added (scaled by the policy gain) to the residual at
      the steer sites.
    - ``steer_layers`` (list[int], default ``[monitor_layer]``).
    - ``steer_hook`` (str, default ``monitor_hook``).
    - ``score`` (``"cosine"`` | ``"dot"``, default ``"cosine"``):
      cosine divides the probe projection by the activation norm
      (scale-invariant, thresholds live in [-1, 1]); dot is the raw
      projection onto the unit probe.
    - policy knobs: ``threshold`` (required), ``hysteresis``,
      ``ema_alpha``, ``gain``, ``gain_mode``, ``aggregate``,
      ``min_emit_delta`` — see :class:`PolicyConfig`.
    """

    location: ClassVar[Literal["worker"]] = "worker"
    reads_client_spec: ClassVar[bool] = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        params: dict[str, Any],
    ) -> None:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp = int(getattr(parallel_config, "tensor_parallel_size", 1))
        pp = int(getattr(parallel_config, "pipeline_parallel_size", 1))
        if tp != 1 or pp != 1:
            raise ValueError(
                f"DynamicSteeringController requires tensor_parallel_size=1 "
                f"and pipeline_parallel_size=1 (got tp={tp}, pp={pp}): "
                f"capture consumers run on TP rank 0 only, so steering "
                f"updates pushed from one would diverge the other ranks."
            )

        model_config = getattr(vllm_config, "model_config", None)
        hidden_size = (
            model_config.get_hidden_size() if model_config is not None else None
        )

        self._monitor_layer = int(params["monitor_layer"])
        self._monitor_hook = str(params.get("monitor_hook", "post_mlp"))
        if self._monitor_hook not in _HOOK_NAMES:
            raise ValueError(
                f"monitor_hook={self._monitor_hook!r} not one of {sorted(_HOOK_NAMES)}"
            )

        self._steer_hook = str(params.get("steer_hook", self._monitor_hook))
        if self._steer_hook not in _HOOK_NAMES:
            raise ValueError(
                f"steer_hook={self._steer_hook!r} not one of {sorted(_HOOK_NAMES)}"
            )
        steer_layers = params.get("steer_layers")
        if steer_layers is None:
            steer_layers = [self._monitor_layer]
        self._steer_layers = [int(layer) for layer in steer_layers]
        if not self._steer_layers:
            raise ValueError("steer_layers must not be empty")

        self._score_mode = str(params.get("score", "cosine"))
        if self._score_mode not in _SCORE_MODES:
            raise ValueError(
                f"score={self._score_mode!r} not one of {sorted(_SCORE_MODES)}"
            )

        self._probe = self._load_unit_vector(
            pathlib.Path(str(params["probe_path"])), hidden_size, "probe_path"
        )
        steer_vec = self._load_vector(
            pathlib.Path(str(params["steering_vector_path"])),
            hidden_size,
            "steering_vector_path",
        )
        self._steer_vec = np.ascontiguousarray(steer_vec.numpy(), dtype=np.float32)

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

        self._lock = threading.Lock()
        self._results: dict[CaptureKey, CaptureResult] = {}
        self._last_score: dict[str, float] = {}
        self._updates_emitted = 0
        self._updates_dropped = 0
        self._no_queue_logged = False

    # ------------------------------------------------------------------
    # Vector loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vector(
        path: pathlib.Path,
        hidden_size: int | None,
        param_name: str,
    ) -> torch.Tensor:
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
        return vector.to(dtype=torch.float32)

    @classmethod
    def _load_unit_vector(
        cls,
        path: pathlib.Path,
        hidden_size: int | None,
        param_name: str,
    ) -> np.ndarray:
        vector = cls._load_vector(path, hidden_size, param_name)
        norm = float(vector.norm())
        if norm == 0.0:
            raise ValueError(f"DynamicSteeringController: {param_name} has zero norm.")
        return np.ascontiguousarray((vector / norm).numpy(), dtype=np.float32)

    # ------------------------------------------------------------------
    # Consumer-API surface (registry and admission path)
    # ------------------------------------------------------------------

    def global_capture_spec(self) -> CaptureSpec | None:
        return CaptureSpec(
            hooks={self._monitor_hook: [self._monitor_layer]},
            positions="all_generated",
        )

    # ------------------------------------------------------------------
    # CaptureSink protocol
    # ------------------------------------------------------------------

    def submit_chunk(self, chunk: CaptureChunk) -> None:
        self.submit_chunk_batch([chunk])

    def submit_chunk_batch(self, chunks: list[CaptureChunk]) -> None:
        with self._lock:
            for chunk in chunks:
                self._observe_chunk(chunk)
            self._evaluate_and_emit()

    def submit_finalize(self, finalize: CaptureFinalize) -> None:
        key = finalize.key
        req_id = str(key[0])
        with self._lock:
            last_score = self._last_score.pop(req_id, float("nan"))
            self._policy.forget(req_id)
            # A finishing request can drop the aggregate below the
            # disengage level; re-evaluate so the controller zeroes the
            # steering vector instead of leaving the last gain latched
            # until some other request's chunk arrives. The payload is
            # built afterwards so its counters include that emission.
            self._evaluate_and_emit()
            payload = {
                "last_score": last_score,
                "engaged": self._policy.engaged,
                "gain": self._policy.last_emitted_gain,
                "updates_emitted": self._updates_emitted,
                "updates_dropped": self._updates_dropped,
            }
            self._results[key] = CaptureResult(key=key, status="ok", payload=payload)

    def get_result(self, key: CaptureKey) -> CaptureResult | None:
        with self._lock:
            return self._results.pop(key, None)

    def shutdown(self, timeout: float = 30.0) -> None:
        with self._lock:
            self._last_score.clear()
            self._results.clear()

    # ------------------------------------------------------------------
    # Internals (lock held)
    # ------------------------------------------------------------------

    def _observe_chunk(self, chunk: CaptureChunk) -> None:
        req_id = str(chunk.key[0])
        acts = chunk.tensor.to(dtype=torch.float32).numpy()
        if acts.size == 0:
            return
        proj = acts @ self._probe
        if self._score_mode == "cosine":
            norms = np.linalg.norm(acts, axis=-1)
            proj = proj / np.maximum(norms, _EPS)
        self._policy.observe(req_id, proj)
        self._last_score[req_id] = float(proj[-1])

    def _evaluate_and_emit(self) -> None:
        gain = self._policy.current_gain()
        if not self._policy.should_emit(gain):
            return
        queue = get_steering_action_queue()
        if queue is None:
            if not self._no_queue_logged:
                self._no_queue_logged = True
                logger.warning(
                    "DynamicSteeringController: no steering action queue "
                    "installed (steering disabled, or tp/pp > 1); "
                    "controller decisions will be computed but not applied."
                )
            return
        scaled = np.ascontiguousarray(self._steer_vec * gain, dtype=np.float32)
        update = SteeringVectorUpdate(
            vectors={self._steer_hook: {layer: scaled for layer in self._steer_layers}},
            phase="decode",
            source="dynamic_steering",
        )
        if queue.submit(update):
            self._updates_emitted += 1
            self._policy.mark_emitted(gain)
        else:
            self._updates_dropped += 1


__all__ = [
    "DynamicSteeringController",
    "PolicyConfig",
    "ProbePolicy",
]
