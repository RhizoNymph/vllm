# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side named probe/steer vector registry.

Promotes the frontend named-vector registry
(:class:`~vllm.entrypoints.openai.steering.vector_registry.\
SteeringVectorRegistry`) to a worker-resident mirror so a request's
declarative gates can carry a ``NamedVec`` reference all the way to the
worker rather than being inflated to inline packed bytes at the frontend.

Two reasons the worker now needs its own copy (reversing the original
"the worker only ever sees packed bytes" decision, see
docs/design/dynamic_steering.md §8.2):

- **Smaller wire payloads.** A ``NamedVec`` rides the msgpack channel as a
  short string instead of the full base64 vector blob; resolution to numpy
  happens once at admission (:func:`vllm.v1.steering_schema.resolve_gates`).
- **Persistence semantics.** A ``rest_of_conversation`` latch persists a
  reference (name + content digest), not the client's bytes. Bridging a
  later turn re-resolves the name *here* and verifies the digest, so the
  server never pins client-supplied vectors indefinitely and a
  re-registered name cannot silently change a live conversation's steering.

Determinism: mutations arrive as rank-replicated ``collective_rpc`` calls
(the engine serializes RPCs), so every TP/PP rank's registry evolves as a
pure function of the same ordered event sequence. The registry is a
process-global singleton per worker (mirrors the dynamic-steering action
queue), so the sync steering consumer — which runs on the capture dispatch
thread with no handle to the model runner — can reach it at bridge time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config.steering_types import (
    steering_vector_content_digest,
    unpack_steering_vectors,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    import numpy as np

    from vllm.config.steering_types import SteeringHookPacked

logger = init_logger(__name__)

_KINDS = ("probe", "steer")


class WorkerSteeringVectorRegistry:
    """Rank-replicated store of named probe/steer vectors, resolved to numpy.

    Each entry holds the unpacked ``{hook: {layer: ndarray}}`` vectors and a
    content digest, both derived once at registration from the packed bytes
    the frontend broadcast. Read on the admission path
    (:func:`vllm.v1.steering_schema.resolve_gates`) and at latch-bridge time
    (:class:`~vllm.v1.capture.controller.SteeringController`).
    """

    def __init__(self) -> None:
        # kind -> name -> (unpacked vectors, content digest)
        self._vectors: dict[
            str, dict[str, tuple[dict[str, dict[int, np.ndarray]], str]]
        ] = {"probe": {}, "steer": {}}

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")

    def register(
        self,
        name: str,
        kind: str,
        packed: dict[str, SteeringHookPacked],
        digest: str | None = None,
    ) -> None:
        """Register (or replace) a named vector, unpacking it to numpy.

        Args:
            name: Vector name.
            kind: ``"probe"`` or ``"steer"``.
            packed: The ``{hook: SteeringHookPacked}`` packed spec.
            digest: Precomputed content digest (from the frontend). Recomputed
                locally when omitted; both derive from the same canonical
                serialization, so they match.
        """
        self._check_kind(kind)
        if not name:
            raise ValueError("vector name must be non-empty")
        unpacked = unpack_steering_vectors(packed)
        if not unpacked:
            raise ValueError(f"{kind} vector {name!r} decoded to no rows")
        content_digest = digest or steering_vector_content_digest(packed)
        self._vectors[kind][name] = (unpacked, content_digest)

    def unregister(self, name: str, kind: str) -> bool:
        """Remove a named vector. Returns ``True`` if it existed."""
        self._check_kind(kind)
        return self._vectors[kind].pop(name, None) is not None

    def resolve_vectors(
        self, name: str, kind: str
    ) -> tuple[dict[str, dict[int, np.ndarray]], str] | None:
        """Return ``(unpacked vectors, digest)`` for a name, or ``None``.

        The unifying resolution surface used by
        :func:`vllm.v1.steering_schema.resolve_gates`; the frontend registry
        implements the same method (unpacking on demand) so one code path
        resolves ``NamedVec`` on either side.
        """
        self._check_kind(kind)
        return self._vectors[kind].get(name)

    def status(self) -> dict[str, list[str]]:
        return {kind: sorted(self._vectors[kind]) for kind in _KINDS}

    def count(self) -> int:
        return sum(len(v) for v in self._vectors.values())


# ---------------------------------------------------------------------------
# Process-global registry slot (mirrors ``install_steering_action_queue``).
# ---------------------------------------------------------------------------

_ACTIVE_REGISTRY: WorkerSteeringVectorRegistry | None = None


def install_worker_steering_vector_registry(
    registry: WorkerSteeringVectorRegistry | None,
) -> None:
    """Install ``registry`` as this worker process's named-vector registry."""
    global _ACTIVE_REGISTRY
    _ACTIVE_REGISTRY = registry


def get_worker_steering_vector_registry() -> WorkerSteeringVectorRegistry | None:
    """Return the installed worker named-vector registry, or ``None``."""
    return _ACTIVE_REGISTRY


__all__ = [
    "WorkerSteeringVectorRegistry",
    "install_worker_steering_vector_registry",
    "get_worker_steering_vector_registry",
]
