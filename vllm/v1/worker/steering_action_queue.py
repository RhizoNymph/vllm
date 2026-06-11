# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process action queue for dynamic steering (Phase 0).

This module is the narrow bridge between activation *observers* (capture
consumers running on the capture dispatch thread) and the steering
runtime (mutated only on the model-runner step thread). Observers submit
:class:`SteeringVectorUpdate` actions; the model runner drains the queue
at the top of ``_update_steering_buffers`` each step, so an action
submitted while step *N* executes is visible to the steering tables
built for step *N+1*.

Design contract (see ``docs/design/dynamic_steering.md``):

- **Single mutator.** The queue is drained only on the model-runner step
  thread, which already owns the ``SteeringManager``. Observers never
  touch the manager directly, so the manager needs no new locking.
- **Decode tier only by default.** Global base/prefill steering feeds
  prefix-cache keys; mutating it through this side channel would corrupt
  KV-cache identity because nothing here invalidates the prefix cache.
  Decode-tier updates are explicitly exempt from cache keying, so they
  are the only phase accepted unless ``allow_cache_unsafe_phases`` is
  set by the caller (which then owns cache invalidation).
- **TP=1 / PP=1 only.** Capture consumers are constructed on TP rank 0
  only, so updates originating from one would diverge the steering
  tables across ranks. The model runner installs a queue only in
  single-rank topologies; ``get_steering_action_queue()`` returns
  ``None`` everywhere else and observers must degrade gracefully.
- **Non-throwing submit.** A misbehaving observer must never take down
  the engine. ``submit`` returns ``False`` (and rate-limit-logs) when
  the queue is full or uninstalled; validation failures at drain time
  reject the single offending update and keep going.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    VALID_HOOK_POINT_NAMES,
    SteeringHookPoint,
)

if TYPE_CHECKING:
    from vllm.v1.worker.steering_manager import SteeringManager

logger = init_logger(__name__)

VALID_PHASES: frozenset[str] = frozenset({"base", "prefill", "decode"})

# Phases whose global vectors do not participate in prefix-cache keys.
# See docs/features/steering.md ("Prefix Caching"): decode-only steering
# must not fork prompt KV-cache entries, so mutating it requires no
# cache invalidation.
CACHE_SAFE_PHASES: frozenset[str] = frozenset({"decode"})


@dataclass(frozen=True)
class SteeringVectorUpdate:
    """One queued global-steering mutation.

    ``vectors`` maps hook-point name -> layer index -> 1-D float32
    vector of length ``hidden_size``. Application semantics match
    :meth:`SteeringManager.update_global_vectors`: each ``(hook,
    layer)`` entry *overwrites* the current global vector for ``phase``
    (set, not add). Emitting a zero vector therefore disengages
    steering at that site.
    """

    vectors: dict[str, dict[int, np.ndarray]]
    phase: str = "decode"
    # Identifies the submitting observer in logs and stats.
    source: str = ""


@dataclass
class SteeringActionQueueStats:
    submitted: int = 0
    dropped: int = 0
    applied: int = 0
    rejected: int = 0


class SteeringActionQueue:
    """Bounded thread-safe FIFO of :class:`SteeringVectorUpdate`.

    ``submit`` may be called from any thread (in practice the capture
    dispatch thread). ``drain`` must only be called from the
    model-runner step thread — the single-mutator contract above.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._dq: deque[SteeringVectorUpdate] = deque()
        self._stats = SteeringActionQueueStats()
        self._overflow_logged = False

    def submit(self, update: SteeringVectorUpdate) -> bool:
        """Enqueue ``update``; returns ``False`` if dropped (queue full).

        Never raises. Overflow means the observer is producing updates
        faster than the step loop drains them — dropping the *newest*
        keeps the already-queued (older, about-to-apply) state coherent
        and bounds memory.
        """
        with self._lock:
            self._stats.submitted += 1
            if len(self._dq) >= self._maxsize:
                self._stats.dropped += 1
                if not self._overflow_logged:
                    self._overflow_logged = True
                    logger.warning(
                        "steering action queue overflow: dropping update "
                        "(source=%s, maxsize=%d); further drops will not "
                        "be logged",
                        update.source,
                        self._maxsize,
                    )
                return False
            self._dq.append(update)
            return True

    def drain(self) -> list[SteeringVectorUpdate]:
        """Pop and return all queued updates in submission order."""
        with self._lock:
            if not self._dq:
                return []
            items = list(self._dq)
            self._dq.clear()
            return items

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return bool(self._dq)

    def stats(self) -> SteeringActionQueueStats:
        """Snapshot of queue counters (applied/rejected set by drainers)."""
        with self._lock:
            return SteeringActionQueueStats(
                submitted=self._stats.submitted,
                dropped=self._stats.dropped,
                applied=self._stats.applied,
                rejected=self._stats.rejected,
            )

    def record_applied(self, n: int) -> None:
        with self._lock:
            self._stats.applied += n

    def record_rejected(self, n: int) -> None:
        with self._lock:
            self._stats.rejected += n


# ---------------------------------------------------------------------------
# Process-global queue slot
# ---------------------------------------------------------------------------
#
# Mirrors ``set_active_capture_manager`` in
# ``vllm/model_executor/layers/activation_capture.py``: the model runner
# installs the queue once per worker (only when steering is enabled and
# the topology is single-rank); observers look it up lazily at submit
# time because consumers are constructed before steering init runs.

_ACTIVE_QUEUE: SteeringActionQueue | None = None


def install_steering_action_queue(queue: SteeringActionQueue | None) -> None:
    """Install ``queue`` as this process's dynamic-steering action queue.

    Passing ``None`` marks dynamic steering unavailable (steering
    disabled, or a multi-rank topology where consumer-originated
    updates would diverge ranks).
    """
    global _ACTIVE_QUEUE
    _ACTIVE_QUEUE = queue


def get_steering_action_queue() -> SteeringActionQueue | None:
    """Return the installed queue, or ``None`` if dynamic steering is
    unavailable in this process."""
    return _ACTIVE_QUEUE


# ---------------------------------------------------------------------------
# Drain-time validation and application
# ---------------------------------------------------------------------------


def _validate_update(
    update: SteeringVectorUpdate,
    steerable_layers: dict,
    *,
    allow_cache_unsafe_phases: bool,
) -> None:
    """Raise :class:`SteeringVectorError` if ``update`` cannot be applied.

    Mirrors the checks ``set_steering_vectors`` performs on the HTTP
    path (hook validity, layer presence, hidden-size match, finite
    values) plus the dynamic-path-specific phase restriction.
    """
    if update.phase not in VALID_PHASES:
        raise SteeringVectorError(
            f"invalid phase {update.phase!r}; must be one of {sorted(VALID_PHASES)}"
        )
    if update.phase not in CACHE_SAFE_PHASES and not allow_cache_unsafe_phases:
        raise SteeringVectorError(
            f"dynamic updates to phase {update.phase!r} are rejected: "
            f"global base/prefill vectors feed prefix-cache keys and this "
            f"path performs no cache invalidation. Use phase='decode', or "
            f"the /v1/steering/set HTTP path for base/prefill changes."
        )
    if not update.vectors:
        raise SteeringVectorError("update contains no vectors")
    for hook_name, layer_vecs in update.vectors.items():
        if hook_name not in VALID_HOOK_POINT_NAMES:
            raise SteeringVectorError(f"invalid hook point: {hook_name!r}")
        table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook_name)]
        for layer_idx, vec in layer_vecs.items():
            mod = steerable_layers.get(layer_idx)
            if mod is None:
                raise SteeringVectorError(
                    f"layer {layer_idx} is not steerable on this worker"
                )
            if not hasattr(mod, table_attr):
                raise SteeringVectorError(
                    f"hook point {hook_name!r} not active on layer {layer_idx}"
                )
            expected = getattr(mod, table_attr).shape[1]
            arr = np.asarray(vec)
            if arr.ndim != 1 or arr.shape[0] != expected:
                raise SteeringVectorError(
                    f"layer {layer_idx} ({hook_name}): expected 1-D vector "
                    f"of size {expected}, got shape {tuple(arr.shape)}"
                )
            if not np.isfinite(arr).all():
                raise SteeringVectorError(
                    f"layer {layer_idx} ({hook_name}): vector contains "
                    f"non-finite values"
                )


def apply_steering_updates(
    updates: list[SteeringVectorUpdate],
    manager: SteeringManager,
    steerable_layers: dict,
    *,
    allow_cache_unsafe_phases: bool = False,
    queue: SteeringActionQueue | None = None,
) -> tuple[int, int]:
    """Validate and apply drained ``updates`` to ``manager``.

    Returns ``(applied, rejected)`` update counts. Each update is
    validated and applied independently — one malformed update does not
    affect the others (observer-isolation contract). Must be called on
    the model-runner step thread.

    Application goes through
    :meth:`SteeringManager.update_global_vectors`, which sets
    ``_tables_dirty`` so the caller's existing populate path uploads the
    new state before the next forward.
    """
    applied = 0
    rejected = 0
    for update in updates:
        try:
            _validate_update(
                update,
                steerable_layers,
                allow_cache_unsafe_phases=allow_cache_unsafe_phases,
            )
        except SteeringVectorError as exc:
            rejected += 1
            logger.warning(
                "rejected dynamic steering update (source=%s, phase=%s): %s",
                update.source,
                update.phase,
                exc,
            )
            continue
        for hook_name, layer_vecs in update.vectors.items():
            for layer_idx, vec in layer_vecs.items():
                tensor = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
                manager.update_global_vectors(
                    hook_name,
                    layer_idx,
                    tensor,
                    phase=update.phase,
                )
        applied += 1
    if queue is not None:
        if applied:
            queue.record_applied(applied)
        if rejected:
            queue.record_rejected(rejected)
    return applied, rejected
