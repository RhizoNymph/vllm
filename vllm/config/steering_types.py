# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type definitions and helpers for steering vector composition.

The additive composition model:

    effective_prefill[hook][layer] =
        scale(steering_vectors[hook][layer])
        + scale(prefill_steering_vectors[hook][layer])

    effective_decode[hook][layer] =
        scale(steering_vectors[hook][layer])
        + scale(decode_steering_vectors[hook][layer])

Where ``scale(entry)`` means: if entry is a bare list, scale=1.0; if entry is
``{"vector": [...], "scale": float}``, multiply vector by scale.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Per-layer entry: bare list (scale=1.0) or {"vector": [...], "scale": float}
SteeringLayerEntry = list[float] | dict[str, Any]

# Full spec: {hook_point_name: {layer_idx: SteeringLayerEntry}}
SteeringVectorSpec = dict[str, dict[int, SteeringLayerEntry]]

# Pre-resolved per-layer vector: a flat sequence of floats with no scale wrapper.
# The resolver produces ``np.ndarray`` (float32); ``register_config`` and
# ``hash_steering_config`` also accept plain ``list[float]`` so direct
# unit-test callers don't have to wrap inputs.
ResolvedLayerVector = list[float] | np.ndarray
ResolvedSteeringVectors = dict[str, dict[int, ResolvedLayerVector]]


def normalize_layer_entry(entry: SteeringLayerEntry) -> tuple[list[float], float]:
    """Return ``(vector, scale)`` from a steering layer entry.

    If *entry* is a bare ``list[float]``, returns ``(entry, 1.0)``.
    If *entry* is ``{"vector": [...], "scale": float}``, returns
    ``(entry["vector"], entry["scale"])``.
    """
    if isinstance(entry, list):
        return entry, 1.0
    if isinstance(entry, dict):
        allowed = {"vector", "scale"}
        extra = set(entry.keys()) - allowed
        if extra:
            raise ValueError(
                f"Scaled steering entry has unexpected keys: {sorted(extra)}; "
                f"allowed keys: ['scale', 'vector']"
            )
        missing = allowed - set(entry.keys())
        if missing:
            raise ValueError(
                f"Scaled steering entry missing required key(s): "
                f"{sorted(missing)}; got keys: {sorted(entry.keys())}"
            )
        return entry["vector"], float(entry["scale"])
    raise TypeError(
        f"SteeringLayerEntry must be a list or dict, got {type(entry).__name__}"
    )


def resolve_effective_vectors(
    base: SteeringVectorSpec | None,
    phase_specific: SteeringVectorSpec | None,
) -> ResolvedSteeringVectors | None:
    """Merge *base* and *phase_specific* steering specs additively.

    For each ``(hook, layer)`` pair, both the base and phase-specific entries
    are pre-scaled and then summed.  Non-overlapping entries pass through
    unchanged (pre-scaled).

    Returns pre-scaled flat ``np.ndarray`` (float32) values, no scale wrapper.
    Returns ``None`` if both inputs are ``None`` or empty.
    """
    base_empty = not base
    phase_empty = not phase_specific
    if base_empty and phase_empty:
        return None

    result: ResolvedSteeringVectors = {}

    all_hooks: set[str] = set()
    if not base_empty:
        assert base is not None
        all_hooks.update(base.keys())
    if not phase_empty:
        assert phase_specific is not None
        all_hooks.update(phase_specific.keys())

    for hook in all_hooks:
        base_layers = base.get(hook, {}) if base else {}
        phase_layers = phase_specific.get(hook, {}) if phase_specific else {}

        all_layer_idxs: set[int] = set()
        all_layer_idxs.update(base_layers.keys())
        all_layer_idxs.update(phase_layers.keys())

        if not all_layer_idxs:
            continue

        hook_result: dict[int, ResolvedLayerVector] = {}
        for layer_idx in all_layer_idxs:
            base_entry = base_layers.get(layer_idx)
            phase_entry = phase_layers.get(layer_idx)

            if base_entry is not None and phase_entry is not None:
                base_vec, base_scale = normalize_layer_entry(base_entry)
                phase_vec, phase_scale = normalize_layer_entry(phase_entry)
                base_arr = np.asarray(base_vec, dtype=np.float32)
                phase_arr = np.asarray(phase_vec, dtype=np.float32)
                if base_arr.shape != phase_arr.shape:
                    raise ValueError(
                        "Cannot add steering vectors of different lengths: "
                        f"{base_arr.shape[0]} vs {phase_arr.shape[0]}"
                    )
                hook_result[layer_idx] = base_arr * base_scale + phase_arr * phase_scale
            elif base_entry is not None:
                vec, scale = normalize_layer_entry(base_entry)
                hook_result[layer_idx] = np.asarray(vec, dtype=np.float32) * scale
            else:
                assert phase_entry is not None
                vec, scale = normalize_layer_entry(phase_entry)
                hook_result[layer_idx] = np.asarray(vec, dtype=np.float32) * scale

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def merge_steering_specs(
    a: SteeringVectorSpec | None,
    b: SteeringVectorSpec | None,
) -> SteeringVectorSpec | None:
    """Additively merge two :class:`SteeringVectorSpec` dicts.

    For overlapping ``(hook, layer)`` entries both sides are pre-scaled and
    summed.  Non-overlapping entries pass through pre-scaled.  The result
    keeps the bare-list ``SteeringVectorSpec`` shape so it can be fed back
    in as input to other spec-consuming code.

    Returns ``None`` if both inputs are ``None`` or empty.
    """
    a_empty = not a
    b_empty = not b
    if a_empty and b_empty:
        return None

    result: SteeringVectorSpec = {}

    all_hooks: set[str] = set()
    if not a_empty:
        assert a is not None
        all_hooks.update(a.keys())
    if not b_empty:
        assert b is not None
        all_hooks.update(b.keys())

    for hook in all_hooks:
        a_layers = a.get(hook, {}) if a else {}
        b_layers = b.get(hook, {}) if b else {}

        all_layer_idxs: set[int] = set()
        all_layer_idxs.update(a_layers.keys())
        all_layer_idxs.update(b_layers.keys())

        if not all_layer_idxs:
            continue

        hook_result: dict[int, SteeringLayerEntry] = {}
        for layer_idx in all_layer_idxs:
            a_entry = a_layers.get(layer_idx)
            b_entry = b_layers.get(layer_idx)

            if a_entry is not None and b_entry is not None:
                a_vec, a_scale = normalize_layer_entry(a_entry)
                b_vec, b_scale = normalize_layer_entry(b_entry)
                a_arr = np.asarray(a_vec, dtype=np.float32)
                b_arr = np.asarray(b_vec, dtype=np.float32)
                if a_arr.shape != b_arr.shape:
                    raise ValueError(
                        "Cannot add steering vectors of different lengths: "
                        f"{a_arr.shape[0]} vs {b_arr.shape[0]}"
                    )
                hook_result[layer_idx] = (
                    a_arr * a_scale + b_arr * b_scale
                ).tolist()
            elif a_entry is not None:
                vec, scale = normalize_layer_entry(a_entry)
                hook_result[layer_idx] = (
                    np.asarray(vec, dtype=np.float32) * scale
                ).tolist()
            else:
                assert b_entry is not None
                vec, scale = normalize_layer_entry(b_entry)
                hook_result[layer_idx] = (
                    np.asarray(vec, dtype=np.float32) * scale
                ).tolist()

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def hash_steering_config(
    effective_vectors: ResolvedSteeringVectors | None,
) -> int:
    """Deterministic SHA-256 hash of pre-resolved steering vectors.

    Returns 0 if *effective_vectors* is ``None`` or empty.
    The hash is masked to fit in ``np.int64``.

    Hashes the binary representation of each layer vector (via
    ``np.asarray(...).tobytes()``) instead of stringifying the raw Python
    floats. The previous ``str(sorted(...))`` approach took ~28 ms per call
    on Gemma-3-4B (87K floats) because ``str`` invokes ``float.__repr__``
    on every element; this version is ~30x faster because ``tobytes`` is a
    memcpy and ``hashlib.sha256.update`` is hardware-accelerated.
    """
    if not effective_vectors:
        return 0
    h = hashlib.sha256()
    for hook in sorted(effective_vectors.keys()):
        h.update(hook.encode())
        layer_dict = effective_vectors[hook]
        for layer_idx in sorted(layer_dict.keys()):
            entry = layer_dict[layer_idx]
            # An entry is either a bare list/array of floats or a dict
            # ``{"vector": [...], "scale": float}``. By the time we get here
            # the resolver has flattened the dict form into a plain list, so
            # we expect the bare form — but handle both for safety.
            if isinstance(entry, dict):
                vec = entry.get("vector", entry)
                scale = float(entry.get("scale", 1.0))
            else:
                vec = entry
                scale = 1.0
            arr = np.asarray(vec, dtype=np.float32)
            h.update(layer_idx.to_bytes(4, "little", signed=True))
            h.update(arr.tobytes())
            if scale != 1.0:
                h.update(np.float64(scale).tobytes())
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
