# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named probe/steer vector registry for declarative per-request steering.

Distinct from :class:`~vllm.entrypoints.openai.steering.registry.\
SteeringModuleRegistry` (which holds 3-tier steering *modules* replicated to
workers): this registry is **frontend-only** and holds single named vectors in
two namespaces — ``probe`` and ``steer`` — that a request's declarative gates
reference by name (``{"kind":"name","name":...}``). Names are resolved to inline
packed bytes at request-admission time (see
:func:`vllm.v1.steering_schema.build_steering_gates`), so the worker never sees a
name and no registry replication is needed.

Each stored vector is the canonical ``{hook: SteeringHookPacked}`` packed form,
so name resolution is a zero-copy inline (no re-encode) into the request's
metadata channel.
"""

from __future__ import annotations

import asyncio

from vllm.config.steering_types import SteeringHookPacked, unpack_steering_vectors
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)

_KINDS = ("probe", "steer")


class SteeringVectorRegistry:
    """Frontend registry of named probe/steer vectors (packed form)."""

    def __init__(self, valid_layer_indices: set[int] | None = None) -> None:
        # kind -> name -> {hook: SteeringHookPacked}
        self._vectors: dict[str, dict[str, dict[str, SteeringHookPacked]]] = {
            "probe": {},
            "steer": {},
        }
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
    ) -> None:
        """Register (or overwrite) a named probe/steer vector."""
        self._check_kind(kind)
        if not name:
            raise ValueError("vector name must be non-empty")
        self._validate_packed(name, kind, packed)
        async with self._lock:
            self._vectors[kind][name] = packed
        logger.info("Registered %s steering vector '%s'", kind, name)

    async def unregister(self, name: str, kind: str) -> bool:
        """Remove a named vector. Returns True if it existed."""
        self._check_kind(kind)
        async with self._lock:
            removed = self._vectors[kind].pop(name, None)
        if removed is not None:
            logger.info("Unregistered %s steering vector '%s'", kind, name)
        return removed is not None

    def get_packed(
        self, name: str, kind: str
    ) -> dict[str, SteeringHookPacked] | None:
        """Look up a named vector's packed form. Thread-safe for reads.

        Called by :func:`vllm.v1.steering_schema.build_steering_gates` to
        resolve a ``{"kind":"name"}`` gate source to inline packed bytes.
        """
        self._check_kind(kind)
        return self._vectors[kind].get(name)

    def list_vectors(self) -> dict[str, list[str]]:
        """Return the sorted names in each namespace."""
        return {kind: sorted(self._vectors[kind].keys()) for kind in _KINDS}

    def count(self) -> int:
        return sum(len(v) for v in self._vectors.values())


__all__ = ["SteeringVectorRegistry"]
