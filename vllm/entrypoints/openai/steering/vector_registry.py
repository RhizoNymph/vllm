# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named probe/steer vector registry for declarative per-request steering.

Holds single named vectors in two namespaces — ``probe`` and ``steer`` — that
a request's declarative gates reference by name (``{"kind":"name","name":...}``).
Like :class:`~vllm.entrypoints.openai.steering.registry.\
SteeringModuleRegistry` (3-tier steering *modules*), registrations are
broadcast to every worker: the vectors router mirrors each register/unregister
to the worker-resident
:class:`~vllm.v1.worker.steering_vector_registry.\
WorkerSteeringVectorRegistry` so a ``NamedVec`` gate resolves worker-side at
admission (:func:`vllm.v1.steering_schema.resolve_gates`) rather than being
inflated to inline bytes at the frontend. This registry is the frontend mirror:
it validates registrations, answers fast existence checks for
:func:`vllm.v1.steering_schema.build_steering_gates`, and backs the listing
endpoint.

Each stored vector is the canonical ``{hook: SteeringHookPacked}`` packed form
plus a content digest, so latch-by-reference digests match on both sides of the
worker boundary.
"""

from __future__ import annotations

import asyncio

import numpy as np

from vllm.config.steering_types import (
    SteeringHookPacked,
    steering_vector_content_digest,
    unpack_steering_vectors,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)

_KINDS = ("probe", "steer")


class SteeringVectorRegistry:
    """Frontend registry of named probe/steer vectors (packed form).

    The frontend mirror of the worker-resident
    :class:`~vllm.v1.worker.steering_vector_registry.\
WorkerSteeringVectorRegistry`: it validates registrations, answers existence
    checks for :func:`vllm.v1.steering_schema.build_steering_gates`, and backs
    the ``GET /v1/steering/vectors`` listing. Registration is broadcast to the
    workers by the vectors router so both sides stay in lock-step. Each entry
    also stores a content digest so latch-by-reference digests match across
    the boundary.
    """

    def __init__(self, valid_layer_indices: set[int] | None = None) -> None:
        # kind -> name -> {hook: SteeringHookPacked}
        self._vectors: dict[str, dict[str, dict[str, SteeringHookPacked]]] = {
            "probe": {},
            "steer": {},
        }
        # kind -> name -> content digest (parallel to ``_vectors``).
        self._digests: dict[str, dict[str, str]] = {"probe": {}, "steer": {}}
        self._lock = asyncio.Lock()
        self._valid_layer_indices = valid_layer_indices

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")

    def _validate_packed(
        self, name: str, kind: str, packed: dict[str, SteeringHookPacked]
    ) -> None:
        if not isinstance(packed, dict) or not packed:
            raise ValueError(f"{kind} vector '{name}' must be a non-empty packed dict")
        invalid = set(packed.keys()) - VALID_HOOK_POINT_NAMES
        if invalid:
            raise ValueError(
                f"{kind} vector '{name}' has invalid hook(s) {sorted(invalid)}; "
                f"valid: {sorted(VALID_HOOK_POINT_NAMES)}"
            )
        # Unpack to validate shape/base64/lengths (raises ValueError on bad).
        unpacked = unpack_steering_vectors(packed)
        if not unpacked:
            raise ValueError(f"{kind} vector '{name}' decoded to no rows")
        if self._valid_layer_indices is not None:
            for hook, layers in unpacked.items():
                for layer_idx in layers:
                    if layer_idx not in self._valid_layer_indices:
                        raise ValueError(
                            f"{kind} vector '{name}' references layer {layer_idx} "
                            f"at hook {hook!r} which is not a valid model layer"
                        )
        if kind == "probe":
            n_rows = sum(len(layers) for layers in unpacked.values())
            if n_rows != 1:
                raise ValueError(
                    f"probe vector '{name}' must name exactly one (hook, layer); "
                    f"got {n_rows} rows"
                )

    async def register(
        self, name: str, kind: str, packed: dict[str, SteeringHookPacked]
    ) -> str:
        """Register (or overwrite) a named probe/steer vector.

        Returns the content digest so the caller can broadcast a matching
        value to the workers.
        """
        self._check_kind(kind)
        if not name:
            raise ValueError("vector name must be non-empty")
        self._validate_packed(name, kind, packed)
        digest = steering_vector_content_digest(packed)
        async with self._lock:
            self._vectors[kind][name] = packed
            self._digests[kind][name] = digest
        logger.info("Registered %s steering vector '%s'", kind, name)
        return digest

    async def unregister(self, name: str, kind: str) -> bool:
        """Remove a named vector. Returns True if it existed."""
        self._check_kind(kind)
        async with self._lock:
            removed = self._vectors[kind].pop(name, None)
            self._digests[kind].pop(name, None)
        if removed is not None:
            logger.info("Unregistered %s steering vector '%s'", kind, name)
        return removed is not None

    def get_packed(
        self, name: str, kind: str
    ) -> dict[str, SteeringHookPacked] | None:
        """Look up a named vector's packed form. Thread-safe for reads."""
        self._check_kind(kind)
        return self._vectors[kind].get(name)

    def get_digest(self, name: str, kind: str) -> str | None:
        """Return a named vector's content digest, or ``None`` if absent."""
        self._check_kind(kind)
        return self._digests[kind].get(name)

    def resolve_vectors(
        self, name: str, kind: str
    ) -> tuple[dict[str, dict[int, np.ndarray]], str] | None:
        """Return ``(unpacked vectors, digest)`` for a name, or ``None``.

        The uniform name-resolution surface (shared with the worker registry)
        used by :func:`vllm.v1.steering_schema.resolve_gates` during the
        frontend dry-run validation of a request's gates.
        """
        self._check_kind(kind)
        packed = self._vectors[kind].get(name)
        if packed is None:
            return None
        unpacked = unpack_steering_vectors(packed)
        if not unpacked:
            return None
        return unpacked, self._digests[kind][name]

    def list_vectors(self) -> dict[str, list[str]]:
        """Return the sorted names in each namespace."""
        return {kind: sorted(self._vectors[kind].keys()) for kind in _KINDS}

    def count(self) -> int:
        return sum(len(v) for v in self._vectors.values())


__all__ = ["SteeringVectorRegistry"]
