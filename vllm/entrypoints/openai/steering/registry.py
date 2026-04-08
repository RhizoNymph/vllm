# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named steering vector registry.

Provides :class:`SteeringModuleRegistry` for loading, storing, and
resolving named steering vector configurations that can be referenced
by name in API requests.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.config.steering_types import (
    SteeringVectorSpec,
    merge_steering_specs,
    normalize_layer_entry,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)


@dataclass
class SteeringModule:
    """A named steering vector configuration."""

    name: str
    vectors: SteeringVectorSpec | None = None
    prefill_vectors: SteeringVectorSpec | None = None
    decode_vectors: SteeringVectorSpec | None = None


class SteeringModuleRegistry:
    """Registry for named steering vector configurations."""

    def __init__(self) -> None:
        self._modules: dict[str, SteeringModule] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        vectors: SteeringVectorSpec | None = None,
        prefill_vectors: SteeringVectorSpec | None = None,
        decode_vectors: SteeringVectorSpec | None = None,
    ) -> None:
        """Register a named steering module. Overwrites if name exists."""
        # Validate that at least one tier has vectors
        if not vectors and not prefill_vectors and not decode_vectors:
            raise ValueError(f"Steering module '{name}' has no vectors in any tier")

        # Validate hook point names and entry format
        for spec in [vectors, prefill_vectors, decode_vectors]:
            if spec:
                invalid = set(spec.keys()) - VALID_HOOK_POINT_NAMES
                if invalid:
                    raise ValueError(
                        f"Invalid hook point name(s) in module '{name}': "
                        f"{sorted(invalid)}. "
                        f"Valid: {sorted(VALID_HOOK_POINT_NAMES)}"
                    )
                # Validate entries are well-formed
                for _hook_name, layers in spec.items():
                    for _layer_idx, entry in layers.items():
                        normalize_layer_entry(entry)  # raises on bad format

        module = SteeringModule(
            name=name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
        )
        async with self._lock:
            self._modules[name] = module
        logger.info("Registered steering module '%s'", name)

    async def unregister(self, name: str) -> bool:
        """Remove a named module. Returns True if it existed."""
        async with self._lock:
            removed = self._modules.pop(name, None)
        if removed:
            logger.info("Unregistered steering module '%s'", name)
        return removed is not None

    def get(self, name: str) -> SteeringModule | None:
        """Look up a module by name. Thread-safe for reads."""
        return self._modules.get(name)

    def list_modules(self) -> list[str]:
        """Return sorted list of registered module names."""
        return sorted(self._modules.keys())

    async def load_from_file(self, name: str, path: str) -> None:
        """Load a steering module from a JSON file and register it.

        Expected JSON format::

            {
                "vectors": {"post_mlp": {"14": [0.1, ...]}},
                "prefill_vectors": {"pre_attn": {"14": [0.3, ...]}},
                "decode_vectors": null,
            }

        JSON uses string keys for layer indices; they are converted to int.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Steering module file not found: {path}")

        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Steering module file must contain a JSON object, "
                f"got {type(data).__name__}"
            )

        # Extract three tiers, converting string layer keys to int
        vectors = _convert_layer_keys(data.get("vectors"))
        prefill_vectors = _convert_layer_keys(data.get("prefill_vectors"))
        decode_vectors = _convert_layer_keys(data.get("decode_vectors"))

        await self.register(
            name=name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
        )

    def resolve_for_request(
        self,
        steering_name: str,
        inline_vectors: SteeringVectorSpec | None,
        inline_prefill: SteeringVectorSpec | None,
        inline_decode: SteeringVectorSpec | None,
    ) -> tuple[
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        str | None,
    ]:
        """Resolve a named module and merge with inline vectors.

        Returns ``(merged_vectors, merged_prefill, merged_decode, error)``.
        On success *error* is ``None``.  On failure the first three are
        ``None``.
        """
        module = self.get(steering_name)
        if module is None:
            return (
                None,
                None,
                None,
                (
                    f"Unknown steering module '{steering_name}'. "
                    f"Available: {self.list_modules() or 'none'}"
                ),
            )

        merged_vectors = merge_steering_specs(module.vectors, inline_vectors)
        merged_prefill = merge_steering_specs(module.prefill_vectors, inline_prefill)
        merged_decode = merge_steering_specs(module.decode_vectors, inline_decode)

        return merged_vectors, merged_prefill, merged_decode, None


def _convert_layer_keys(
    spec: dict[str, Any] | None,
) -> SteeringVectorSpec | None:
    """Convert JSON string layer keys to int."""
    if not spec:
        return None
    result: SteeringVectorSpec = {}
    for hook_name, layers in spec.items():
        if not isinstance(layers, dict):
            continue
        converted: dict[int, Any] = {}
        for layer_key, entry in layers.items():
            converted[int(layer_key)] = entry
        if converted:
            result[hook_name] = converted
    return result if result else None
