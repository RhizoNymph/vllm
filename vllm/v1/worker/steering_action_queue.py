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


@dataclass(frozen=True)
class RequestSteeringOverride:
    """Per-request decode-steering override (dynamic row routing).

    Routes ``req_id``'s decode tokens to a dynamic-pool table row
    holding ``global_decode_effective + vectors`` — pure routing on top
    of admission state, which stays untouched (admitted config hashes,
    refcounts, and scheduler accounting are never modified; see
    docs/design/dynamic_steering.md §5.2). While active, the override
    REPLACES the request's admitted per-request decode delta.

    ``vectors`` shape matches :class:`SteeringVectorUpdate`;
    ``vectors=None`` clears the override, reverting the request to its
    admitted decode routing. Overrides end automatically when the
    request finishes, is preempted, or is re-added by streaming
    continuation.
    """

    req_id: str
    vectors: dict[str, dict[int, np.ndarray]] | None
    # When True, the runner folds the request's admitted decode steering
    # delta into the override row so it becomes
    # ``global_decode_effective + admitted_decode_delta + vectors`` (i.e.
    # ``vectors`` is ADDED ON TOP of the request's static decode steering
    # rather than replacing it). Used by the declarative consumer's ``add``
    # gates so a client's static ``decode_steering_vectors`` are preserved.
    compose_admitted: bool = False
    # Identifies the submitting observer in logs and stats.
    source: str = ""


@dataclass(frozen=True)
class SteeringScaleUpdate:
    """Per-row strength scale change (the §5.3 "how much" knob).

    A cheap decode-tier gain adjustment: the kernel multiplies the
    gathered steering row by its scale, so changing strength needs only a
    scales-buffer write — no vector re-upload, no table recompose. Targets
    a single row, decode-only by construction (prefill rows are pinned to
    1.0 for cache safety, §7):

    - neither field set ⇒ the global decode row (row 2),
    - ``config_hash`` set ⇒ that static per-request *decode* config row,
    - ``dyn_id`` set ⇒ that dynamic-override row,
    - ``req_id`` set ⇒ the dynamic-override row of that request (resolved
      ``req_id → dyn_id`` by the runner — a consumer never sees the
      internal dyn_id, so this is how a sync consumer modulates a
      per-request override's strength cheaply),
    - ``tier_gain=True`` ⇒ the dedicated dynamic-tier scalar gain (§5.4),
      a free per-step knob independent of any row.

    ``scale`` is a non-negative multiplier (1.0 = unscaled, 0.0 = off).
    """

    scale: float
    config_hash: int | None = None
    dyn_id: int | None = None
    req_id: str | None = None
    tier_gain: bool = False
    source: str = ""


@dataclass(frozen=True)
class SteeringMonitorUpdate:
    """Configure (or clear) the in-graph monitor at a probe site (§8).

    Installs a probe at ``(hook, layer)`` so the monitor op modulates the
    §5.4 dynamic tier's per-token gate by
    ``sigmoid(sharpness*(residual@probe - threshold))`` in the same
    forward — detection at this layer conditions steering at later
    hooks/layers (Phase 2). ``probe=None`` clears the site.

    Cache-safe by construction: the monitor only scales the decode tier
    (prefill gate entries are zeroed by the runner, so ``0*gate==0``); it
    never touches prefill rows or prefix-cache keys. The probe and params
    are runtime state, never part of any config hash.

    Targeting (at most one of ``req_id``/``config_hash``/``dyn_id``):

    - none set ⇒ the GLOBAL monitor (the single probe per site above), which
      gates the dynamic tier and, when ``gate_rows``, all per-request rows;
    - ``config_hash`` set ⇒ a PER-ROW monitor on that static decode config
      row only (each row gated by its own probe — true per-request gating);
    - ``dyn_id`` set ⇒ a per-row monitor on that dynamic-override row;
    - ``req_id`` set ⇒ the dynamic-override row of that request (resolved
      ``req_id → dyn_id`` by the runner). ``probe=None`` clears the target.

    The per-row monitor requires the engine to enable it
    (``enable_row_monitor``); ``gate_rows`` is ignored for per-row targets
    (a per-row monitor always gates its own row). See
    docs/design/dynamic_steering.md.
    """

    hook: str
    layer: int
    probe: np.ndarray | None
    threshold: float = 0.0
    sharpness: float = 1.0
    # When True the monitor also gates the per-request row term (not just
    # the §5.4 tier), decode-only. See dynamic_steering_row_gating.md.
    gate_rows: bool = False
    # Per-row (per-request) targeting; at most one may be set. All None ⇒
    # the global monitor (unchanged behavior).
    req_id: str | None = None
    config_hash: int | None = None
    dyn_id: int | None = None
    source: str = ""


# Source tag stamped on every action emitted by the built-in declarative
# per-request steering consumer. The runner uses it to enforce precedence:
# an operator/server-registered consumer (any other source) WINS over a
# client's declarative gate — a declarative action targeting a request already
# owned by another source is rejected and logged.
DECLARATIVE_SOURCE = "declarative"


# Everything the apply path accepts, from either transport (queue or
# sync ``on_step`` returns).
SteeringAction = (
    SteeringVectorUpdate
    | RequestSteeringOverride
    | SteeringScaleUpdate
    | SteeringMonitorUpdate
)


@dataclass
class SteeringActionQueueStats:
    submitted: int = 0
    dropped: int = 0
    applied: int = 0
    rejected: int = 0


class SteeringActionQueue:
    """Bounded thread-safe FIFO of :data:`SteeringAction`.

    ``submit`` may be called from any thread (in practice the capture
    dispatch thread). ``drain`` must only be called from the
    model-runner step thread — the single-mutator contract above.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._dq: deque[SteeringAction] = deque()
        self._stats = SteeringActionQueueStats()
        self._overflow_logged = False

    def submit(self, update: SteeringAction) -> bool:
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

    def drain(self) -> list[SteeringAction]:
        """Pop and return all queued actions in submission order."""
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


def validate_steering_vectors(
    vectors: dict[str, dict[int, np.ndarray]],
    steerable_layers: dict,
) -> None:
    """Raise :class:`SteeringVectorError` on an unappliable vector dict.

    The vector-level subset of the checks ``set_steering_vectors``
    performs on the HTTP path: hook validity, layer presence on this
    worker, hidden-size match, finite values. Shared by the
    global-update and per-request-override apply paths.
    """
    if not vectors:
        raise SteeringVectorError("update contains no vectors")
    for hook_name, layer_vecs in vectors.items():
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


def validate_steering_scale(action: SteeringScaleUpdate) -> None:
    """Raise :class:`SteeringVectorError` on an unappliable scale update."""
    targets = sum(
        (
            action.config_hash is not None,
            action.dyn_id is not None,
            action.req_id is not None,
            action.tier_gain,
        )
    )
    if targets > 1:
        raise SteeringVectorError(
            "scale update must target at most one of config_hash / dyn_id / "
            "req_id / tier_gain"
        )
    scale = action.scale
    if not np.isfinite(scale):
        raise SteeringVectorError(f"scale must be finite, got {scale!r}")
    if scale < 0:
        raise SteeringVectorError(f"scale must be non-negative, got {scale!r}")


def validate_steering_monitor(
    action: SteeringMonitorUpdate,
    steerable_layers: dict,
) -> None:
    """Raise :class:`SteeringVectorError` on an unappliable monitor update.

    A clear (``probe=None``) only needs a valid ``(hook, layer)`` target;
    a set additionally validates the probe shape/finiteness and the
    policy params. At most one per-row target may be set.
    """
    targets = sum(
        (
            action.req_id is not None,
            action.config_hash is not None,
            action.dyn_id is not None,
        )
    )
    if targets > 1:
        raise SteeringVectorError(
            "monitor update must target at most one of req_id / config_hash / dyn_id"
        )
    if action.hook not in VALID_HOOK_POINT_NAMES:
        raise SteeringVectorError(f"invalid hook point: {action.hook!r}")
    table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(action.hook)]
    mod = steerable_layers.get(action.layer)
    if mod is None:
        raise SteeringVectorError(
            f"layer {action.layer} is not steerable on this worker"
        )
    if not hasattr(mod, table_attr):
        raise SteeringVectorError(
            f"hook point {action.hook!r} not active on layer {action.layer}"
        )
    if not np.isfinite(action.threshold) or not np.isfinite(action.sharpness):
        raise SteeringVectorError("monitor threshold/sharpness must be finite")
    if action.sharpness < 0:
        raise SteeringVectorError(
            f"monitor sharpness must be non-negative, got {action.sharpness!r}"
        )
    if action.probe is None:
        return  # clear
    expected = getattr(mod, table_attr).shape[1]
    arr = np.asarray(action.probe)
    if arr.ndim != 1 or arr.shape[0] != expected:
        raise SteeringVectorError(
            f"monitor probe for layer {action.layer} ({action.hook}): expected "
            f"1-D vector of size {expected}, got shape {tuple(arr.shape)}"
        )
    if not np.isfinite(arr).all():
        raise SteeringVectorError("monitor probe contains non-finite values")


def _validate_update(
    update: SteeringVectorUpdate,
    steerable_layers: dict,
    *,
    allow_cache_unsafe_phases: bool,
) -> None:
    """Raise :class:`SteeringVectorError` if ``update`` cannot be applied.

    Phase restriction (the dynamic-path-specific check) plus the shared
    vector-level checks of :func:`validate_steering_vectors`.
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
    validate_steering_vectors(update.vectors, steerable_layers)


def steering_update_accepted(
    update: SteeringVectorUpdate,
    steerable_layers: dict,
    *,
    allow_cache_unsafe_phases: bool = False,
) -> bool:
    """Return whether ``apply_steering_updates`` would apply ``update``.

    Single source of truth (reuses ``_validate_update``) so the caller's
    determinism checksum folds exactly the set that gets applied, without
    duplicating the phase / vector validation logic.
    """
    try:
        _validate_update(
            update,
            steerable_layers,
            allow_cache_unsafe_phases=allow_cache_unsafe_phases,
        )
    except SteeringVectorError:
        return False
    return True


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

    Decode-phase updates (the only cache-safe phase, and the only one
    accepted unless ``allow_cache_unsafe_phases``) are applied to the
    **dynamic additive tier** via
    :meth:`SteeringManager.update_dynamic_tier`, so dynamic global
    steering composes additively with operator-set decode steering
    instead of overwriting ``global_decode_vectors`` (§5.4). The rare
    base/prefill escape hatch (only reachable with
    ``allow_cache_unsafe_phases=True``) keeps its overwrite semantics on
    :meth:`SteeringManager.update_global_vectors`. Either way the manager
    sets ``_tables_dirty`` so the caller's populate path uploads the new
    state before the next forward.
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
                if update.phase == "decode":
                    # Additive tier — composes with operator-set decode
                    # steering rather than clobbering it.
                    manager.update_dynamic_tier(hook_name, layer_idx, tensor)
                else:
                    # base/prefill: cache-unsafe escape hatch only; keep
                    # the historical overwrite-on-global-tier semantics.
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
