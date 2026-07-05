# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``SteeringController`` — a higher-level sync consumer base.

A :class:`~vllm.v1.capture.consumer.SyncCaptureConsumer` subclass implements
``on_step`` directly and re-writes, by hand, the bookkeeping every dynamic
consumer needs: tracking which requests are live, scoping decisions to a
conversation, and latching a transient trigger into sticky steering that
bridges across later turns. :class:`SteeringController` owns that plumbing and
asks the subclass for only the policy: :meth:`SteeringController.decide`.

What the base owns:

- **Per-request lifecycle.** Each step it tracks the live request ids and
  prunes per-request ``armed`` state down to the live set, so state for
  finished / preempted requests is dropped automatically.
- **Conversation scoping.** Decisions are keyed on
  :attr:`StepRequestView.conversation_id`; untagged rows and prefill rows are
  skipped (overrides route decode tokens only).
- **The latch pattern.** A :class:`RequestSteeringOverride` returned by
  :meth:`decide` (the *trigger*) is recorded for the conversation and applied
  to the firing request. Every later request of that conversation is *bridged*
  — overridden with the same vectors immediately, no re-trigger — so the
  steering carries across turns. The latch map is a bounded LRU (a bridge
  refreshes a conversation's recency; eviction drops the least-recently-used).
  Operator latches store the raw override (byte accounted); the declarative
  consumer stores a :class:`ByRefLatch` (name references, ~0 bytes).

Subclasses implement :meth:`decide` (the trigger detector) and
:meth:`global_capture_spec` (the monitored probe site); the base resolves the
single monitored ``(layer, hook)`` from that spec and hands :meth:`decide` the
firing request's residual window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm.config.steering_types import merge_steering_specs
from vllm.logger import init_logger
from vllm.v1.capture.consumer import SyncCaptureConsumer
from vllm.v1.worker.steering_action_queue import (
    RequestSteeringOverride,
    SteeringAction,
)
from vllm.v1.worker.steering_vector_registry import (
    get_worker_steering_vector_registry,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView, StepRequestView

logger = init_logger(__name__)


@dataclass(frozen=True)
class LatchVectorRef:
    """A reference to one server-registered steer vector for a latch.

    Carries the registered ``name`` (in the ``kind`` namespace), the content
    ``digest`` observed when the latch was installed, and the gate's
    ``strength``. Bridging re-resolves ``name`` from the worker registry and
    verifies ``digest`` before scaling by ``strength``, so a name re-registered
    with different content mid-conversation is detected and the latch drops.
    """

    name: str
    kind: str  # "steer"
    digest: str
    strength: float


@dataclass(frozen=True)
class ByRefLatch:
    """A ``rest_of_conversation`` latch that persists *references*, not bytes.

    The declarative consumer emits this for a client's persisted-scope ``add``
    gates (guaranteed named by the frontend scope rule). It pins ~0 host bytes
    — the vectors live in the rank-replicated worker registry — so it is exempt
    from the byte cap that bounds :class:`RequestSteeringOverride`
    (operator-authored, raw-vector) latches. Bridging resolves ``refs`` at
    bridge time (:meth:`SteeringController._bridge_override`).
    """

    refs: tuple[LatchVectorRef, ...]
    compose_admitted: bool
    source: str


# A latched entry is either an operator-authored raw-vector override (byte
# accounted) or a declarative by-reference latch (~0 bytes). Making this a real
# union keeps the two provenances — trusted operator bytes vs. client-named
# references — distinct rather than a dict-with-flags.
LatchEntry = RequestSteeringOverride | ByRefLatch


def _scale_ref_vectors(
    vectors: dict[str, dict[int, np.ndarray]], strength: float
) -> dict[str, dict[int, np.ndarray]]:
    """Copy ``vectors`` scaling every row by ``strength`` (dtype-preserving).

    Matches :func:`vllm.v1.steering_schema._scale_vectors` so a bridged turn
    reproduces the trigger turn's numerics bit-for-bit.
    """
    return {
        hook: {
            layer: (arr if strength == 1.0 else arr * arr.dtype.type(strength))
            for layer, arr in layers.items()
        }
        for hook, layers in vectors.items()
    }


class SteeringController(SyncCaptureConsumer, ABC):
    """Latching, conversation-scoped sync steering consumer.

    Owns per-request lifecycle, conversation scoping, and the trigger→latch→
    bridge pattern; subclasses supply only :meth:`decide` and
    :meth:`global_capture_spec`. See the module docstring for the design.

    Recognized params:

    - ``max_conversations`` (int, default 1024): entry-count bound on the
      latch map; LRU-evicted (least-recently-used conversation dropped) when
      exceeded.
    - ``max_latched_bytes`` (int, default 256 MiB): payload-byte bound on the
      latch map. Because a :class:`RequestSteeringOverride` latch pins full
      steering vectors (all hooks x layers, float32) on every TP rank, an
      entry-count-only bound is an unbounded host-memory exposure; this caps
      the aggregate. Enforced alongside ``max_conversations`` (LRU eviction
      until both fit); a single override larger than the cap is refused, not
      latched. :class:`ByRefLatch` entries cost ~0 bytes against this cap (the
      vectors live in the worker registry, not the latch). See
      docs/design/dynamic_steering.md (Trust model and multi-tenancy).
    """

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._max_conversations = max(1, int(params.get("max_conversations", 1024)))
        # 256 MiB: a single-(hook, layer) latch at hidden=4096 float32 is
        # ~16 KiB, so this comfortably holds many hundreds of typical latches,
        # while stopping a pathological all-layer flood (a full-model latch can
        # be tens of MiB) from pinning gigabytes of host memory on every rank.
        self._max_latched_bytes = max(
            1, int(params.get("max_latched_bytes", 256 * 1024 * 1024))
        )
        # conv_id -> sticky latch entry to (re)apply; persists across requests.
        # LRU-ordered (most-recently bridged/latched last) so count/byte-cap
        # eviction drops the least-recently-used conversation.
        self._latched: OrderedDict[str, LatchEntry] = OrderedDict()
        # conv_id -> approximate payload bytes of its latched entry, and the
        # running total, so eviction can enforce the byte cap in O(1).
        # ByRefLatch entries record 0 (their vectors live in the registry).
        self._latched_bytes: dict[str, int] = {}
        self._latched_bytes_total = 0
        # Rate-limit the oversized-latch refusal warning (log once).
        self._oversize_logged = False
        # Rate-limit the by-reference bridge-failure warning (log once): a
        # latch whose name was unregistered or re-registered with different
        # content mid-conversation disengages instead of steering with
        # silently-changed content.
        self._latch_ref_failed_warned = False
        # live req_ids already overridden this run (emit-once per request).
        self._armed: set[str] = set()
        self._triggers = 0
        self._bridges = 0
        # Monitored (layer, hook) resolved lazily from global_capture_spec.
        self._residual_key: tuple[int, str] | None = None

    @abstractmethod
    def decide(
        self,
        request_view: StepRequestView,
        residual: torch.Tensor,
    ) -> SteeringAction | None:
        """Policy hook: decide whether to fire on a fresh request.

        Called only for a live, conversation-tagged, decode-phase request that
        is neither already overridden nor already latched (a latched
        conversation's later requests are bridged by the base without
        consulting the policy). ``residual`` is the request's zero-copy GPU
        activation window ``[num_tokens, hidden]`` at the monitored site (from
        :meth:`global_capture_spec`), valid only for the duration of the call.

        Returns a :class:`RequestSteeringOverride` to *trigger* — the base
        applies it to this request and latches its vectors onto the
        conversation for bridging — or ``None`` to leave the request untouched.
        Must satisfy the :meth:`on_step` determinism contract (pure function of
        the residual and the consumer's rank-identical state).
        """

    def _monitored_key(self) -> tuple[int, str]:
        """Resolve the single monitored ``(layer, hook)`` from the spec."""
        if self._residual_key is None:
            spec = self.global_capture_spec()
            keys = [
                (layer, hook) for hook, layers in spec.hooks.items() for layer in layers
            ]
            if not keys:
                raise ValueError(
                    f"{type(self).__name__}.global_capture_spec() declares no "
                    f"(layer, hook) site; a steering controller must monitor one."
                )
            self._residual_key = keys[0]
        return self._residual_key

    @staticmethod
    def _override_nbytes(override: RequestSteeringOverride) -> int:
        """Approximate host bytes an override's vectors pin (0 if clearing)."""
        if override.vectors is None:
            return 0
        return sum(
            arr.nbytes
            for layer_vecs in override.vectors.values()
            for arr in layer_vecs.values()
        )

    def _drop_latched(self, cid: str) -> None:
        """Remove a conversation's latch, releasing its accounted bytes."""
        if cid in self._latched:
            del self._latched[cid]
            self._latched_bytes_total -= self._latched_bytes.pop(cid)

    def _store_latch(self, cid: str, entry: LatchEntry, nbytes: int) -> None:
        """Store ``entry`` for ``cid`` under both caps (LRU eviction).

        Evicts least-recently-used conversations until both the entry-count cap
        (``max_conversations``) and the payload-byte cap (``max_latched_bytes``)
        admit the newcomer, then appends the newcomer as most-recently-used.
        Eviction is a pure function of the latch/bridge sequence (no time-based
        logic), so it is rank-deterministic.
        """
        # Replacing an existing latch: release its bytes before re-accounting.
        self._drop_latched(cid)
        # Evict LRU-first (front of the OrderedDict) until both caps admit it.
        while self._latched and (
            len(self._latched) + 1 > self._max_conversations
            or self._latched_bytes_total + nbytes > self._max_latched_bytes
        ):
            self._drop_latched(next(iter(self._latched)))
        # Append as most-recently-used (a replaced key keeps its old position
        # in an OrderedDict, so move_to_end makes recency unconditional).
        self._latched[cid] = entry
        self._latched.move_to_end(cid)
        self._latched_bytes[cid] = nbytes
        self._latched_bytes_total += nbytes

    def _latch(self, cid: str, override: RequestSteeringOverride) -> None:
        """Record a conversation's sticky raw-vector override (ByValue).

        Used by operator-authored :meth:`decide` subclasses (trusted raw
        vectors). A single override whose vectors alone exceed the byte cap is
        refused (rate-limit-logged): the triggering request still steers this
        turn, only the cross-turn conversation latch is dropped.
        """
        nbytes = self._override_nbytes(override)
        if nbytes > self._max_latched_bytes:
            if not self._oversize_logged:
                self._oversize_logged = True
                logger.warning(
                    "steering latch refused: override for conversation %r is "
                    "%d bytes, exceeding max_latched_bytes=%d; the request "
                    "still steers this turn but the conversation will not be "
                    "latched. Further refusals will not be logged.",
                    cid,
                    nbytes,
                    self._max_latched_bytes,
                )
            # Drop any stale latch so bridging does not resurrect an older
            # override in place of the now-oversized turn.
            self._drop_latched(cid)
            return
        self._store_latch(cid, override, nbytes)

    def _latch_by_ref(self, cid: str, entry: ByRefLatch) -> None:
        """Record a conversation's by-reference latch (declarative, ~0 bytes).

        Exempt from the byte cap (its vectors live in the worker registry, not
        the latch), so it never triggers the oversized-refusal path; the
        count-cap LRU still applies.
        """
        self._store_latch(cid, entry, 0)

    def _bridge_override(
        self, latched: LatchEntry, req_id: str
    ) -> RequestSteeringOverride | None:
        """Build the override to apply to a bridged turn, or ``None`` to drop.

        - :class:`RequestSteeringOverride` (operator ByValue): a copy rebound
          to ``req_id``, preserving every other field (``compose_admitted``,
          ``source``, ...) so a bridged turn steers identically to the trigger.
        - :class:`ByRefLatch` (declarative): re-resolve each referenced name
          from the worker registry and verify its digest. On a missing name or
          a digest mismatch (name re-registered with different content
          mid-conversation) return ``None`` — the caller disengages the latch
          rather than steering with silently-changed content.
        """
        if isinstance(latched, RequestSteeringOverride):
            return replace(latched, req_id=req_id)
        return self._resolve_by_ref(latched, req_id)

    def _resolve_by_ref(
        self, latched: ByRefLatch, req_id: str
    ) -> RequestSteeringOverride | None:
        """Resolve a :class:`ByRefLatch` to a live override, or ``None``."""
        registry = get_worker_steering_vector_registry()

        def _disengage(reason: str) -> None:
            if not self._latch_ref_failed_warned:
                self._latch_ref_failed_warned = True
                logger.warning(
                    "steering latch disengaged for a conversation: %s; the "
                    "conversation will no longer be steered. Further "
                    "disengagements will not be logged.",
                    reason,
                )

        if registry is None:
            _disengage("no worker vector registry is available")
            return None
        vectors: dict[str, dict[int, np.ndarray]] = {}
        for ref in latched.refs:
            resolved = registry.resolve_vectors(ref.name, ref.kind)
            if resolved is None:
                _disengage(f"referenced {ref.kind} vector {ref.name!r} is not "
                           "registered")
                return None
            ref_vectors, digest = resolved
            if digest != ref.digest:
                _disengage(f"referenced {ref.kind} vector {ref.name!r} was "
                           "re-registered with different content (digest "
                           "mismatch)")
                return None
            scaled = _scale_ref_vectors(ref_vectors, ref.strength)
            vectors = scaled if not vectors else merge_steering_specs(vectors, scaled)
        return RequestSteeringOverride(
            req_id=req_id,
            vectors=vectors,
            compose_admitted=latched.compose_admitted,
            source=latched.source,
        )

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        actions: list[SteeringAction] = []
        live: set[str] = set()
        key: tuple[int, str] | None = None
        for req in view.requests:
            live.add(req.req_id)
            cid = req.conversation_id
            # Scope to tagged decode rows: latching keys on the conversation id
            # and only decode tokens are routed by the override, so untagged /
            # prefill rows are skipped.
            if cid is None or req.phase != "decode":
                continue
            if req.req_id in self._armed:
                continue
            if cid in self._latched:
                # BRIDGE: a new request of an already-latched conversation.
                bridged = self._bridge_override(self._latched[cid], req.req_id)
                if bridged is None:
                    # By-reference latch that no longer resolves: disengage.
                    self._drop_latched(cid)
                    continue
                self._latched.move_to_end(cid)  # LRU: refresh recency
                actions.append(bridged)
                self._armed.add(req.req_id)
                self._bridges += 1
                continue
            # TRIGGER: hand the request's residual window to the policy.
            if key is None:
                key = self._monitored_key()
            residual = view.tensors[key][req.start : req.end]
            action = self.decide(req, residual)
            if action is None:
                continue
            if isinstance(action, RequestSteeringOverride):
                self._latch(cid, action)
            actions.append(action)
            self._armed.add(req.req_id)
            self._triggers += 1
        # Prune finished requests; the per-conversation latch persists.
        self._armed &= live
        return actions or None

    def status(self) -> dict[str, Any]:
        return {
            "latched_conversations": len(self._latched),
            "latched_bytes": self._latched_bytes_total,
            "armed_requests": len(self._armed),
            "triggers": self._triggers,
            "bridges": self._bridges,
        }
