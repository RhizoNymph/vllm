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
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

# Per-layer entry: bare list (scale=1.0) or {"vector": [...], "scale": float}.
# This is the public, user-facing shape. Internally, merge helpers may
# produce np.ndarray entries that are fed back through downstream resolvers;
# normalize_layer_entry handles those without widening the public alias.
SteeringLayerEntry = list[float] | dict[str, Any]
SteeringLayerEntryInternal = SteeringLayerEntry | np.ndarray

# Full spec: {hook_point_name: {layer_idx: SteeringLayerEntry}}
SteeringVectorSpec = dict[str, dict[int, SteeringLayerEntry]]
SteeringVectorSpecInternal = dict[str, dict[int, SteeringLayerEntryInternal]]
SteeringVectorSpecLike = Mapping[str, Mapping[int, SteeringLayerEntryInternal]]
ResolvedSteeringVectorSpec = dict[str, dict[int, np.ndarray]]

STEERING_INDEX_MAX = 2**31 - 1


def validate_steering_index(value: object, label: str) -> int:
    """Validate an index that is serialized as signed int32."""
    if type(value) is not int:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value!r}.")
    if value > STEERING_INDEX_MAX:
        raise ValueError(
            f"{label} must be <= {STEERING_INDEX_MAX}, got {value!r}."
        )
    return value


def _steering_index_bytes(value: object, label: str) -> bytes:
    idx = validate_steering_index(value, label)
    return idx.to_bytes(4, "little", signed=True)


def normalize_layer_entry(
    entry: SteeringLayerEntry | np.ndarray,
) -> tuple[list[float] | np.ndarray, float]:
    """Return ``(vector, scale)`` from a steering layer entry.

    If *entry* is a bare ``list[float]`` or ``np.ndarray``, returns
    ``(entry, 1.0)``. If *entry* is ``{"vector": [...], "scale": float}``,
    returns ``(entry["vector"], entry["scale"])``.
    """
    if isinstance(entry, np.ndarray):
        return entry, 1.0
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
        f"SteeringLayerEntry must be a list, dict, or ndarray, "
        f"got {type(entry).__name__}"
    )


def _scale_vector(vec: list[float] | np.ndarray, scale: float) -> np.ndarray:
    """Multiply *vec* by *scale*, returning a float64 numpy array.

    Arithmetic is performed in float64 to match the legacy Python-list path
    bit-for-bit at the float64→float32 boundary in ``hash_steering_config``.
    """
    arr = np.asarray(vec, dtype=np.float64)
    return arr * scale


def _add_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition of two equal-length float64 vectors."""
    if a.shape != b.shape:
        raise ValueError(
            f"Cannot add steering vectors of different lengths: "
            f"{a.shape[0]} vs {b.shape[0]}"
        )
    return a + b


def resolve_effective_vectors(
    base: SteeringVectorSpecLike | None,
    phase_specific: SteeringVectorSpecLike | None,
) -> ResolvedSteeringVectorSpec | None:
    """Merge *base* and *phase_specific* steering specs additively.

    For each ``(hook, layer)`` pair, both the base and phase-specific entries
    are pre-scaled and then summed.  Non-overlapping entries pass through
    unchanged (pre-scaled).

    Returns pre-scaled flat vectors as 1-D ``np.float64`` arrays. The
    float64 dtype is required for hash-determinism parity with the legacy
    Python-list path (``hash_steering_config`` casts to float32 once at the
    SHA boundary).  Returns ``None`` if both inputs are ``None`` or empty.
    """
    base_empty = not base
    phase_empty = not phase_specific
    if base_empty and phase_empty:
        return None

    result: dict[str, dict[int, np.ndarray]] = {}

    # Collect all hook points from both specs
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

        hook_result: dict[int, np.ndarray] = {}
        for layer_idx in all_layer_idxs:
            base_entry = base_layers.get(layer_idx)
            phase_entry = phase_layers.get(layer_idx)

            if base_entry is not None and phase_entry is not None:
                base_vec, base_scale = normalize_layer_entry(base_entry)
                phase_vec, phase_scale = normalize_layer_entry(phase_entry)
                scaled_base = _scale_vector(base_vec, base_scale)
                scaled_phase = _scale_vector(phase_vec, phase_scale)
                hook_result[layer_idx] = _add_vectors(scaled_base, scaled_phase)
            elif base_entry is not None:
                vec, scale = normalize_layer_entry(base_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)
            else:
                assert phase_entry is not None
                vec, scale = normalize_layer_entry(phase_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def _torch_dtype_to_pack_dtype(torch_dtype: object) -> np.dtype:
    """Pick the numpy dtype to pack steering vectors as for *torch_dtype*."""
    name = getattr(torch_dtype, "__str__", lambda: "")().rsplit(".", 1)[-1]
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float64":
        return np.dtype(np.float64)
    if name in ("bfloat16",):
        return np.dtype(np.float32)
    return np.dtype(np.float32)


def pack_effective_steering(
    spec_base: SteeringVectorSpec | None,
    spec_phase: SteeringVectorSpec | None,
    dtype: np.dtype | str,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Resolve and pack inline steering specs in one shot."""
    if not spec_base and not spec_phase:
        return None
    np_dtype = np.dtype(dtype)
    resolved = resolve_effective_vectors(spec_base, spec_phase)
    if resolved is None:
        return None
    out: dict[str, dict[int, np.ndarray]] = {}
    for hook, layer_dict in resolved.items():
        out[hook] = {
            layer_idx: arr.astype(np_dtype, copy=False)
            for layer_idx, arr in layer_dict.items()
        }
    return out


def pack_steering_for_dtype(
    spec: SteeringVectorSpec | None,
    dtype: np.dtype | str,
) -> dict[str, dict[int, np.ndarray]] | None:
    """Pre-bake a :class:`SteeringVectorSpec` into dtype-specific arrays."""
    if not spec:
        return None
    np_dtype = np.dtype(dtype)
    result: dict[str, dict[int, np.ndarray]] = {}
    for hook, layer_dict in spec.items():
        if not layer_dict:
            continue
        packed: dict[int, np.ndarray] = {}
        for layer_idx, entry in layer_dict.items():
            vec, scale = normalize_layer_entry(entry)
            arr = np.asarray(vec, dtype=np_dtype)
            if scale != 1.0:
                arr = arr * np_dtype.type(scale)
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype, copy=False)
            packed[layer_idx] = arr
        if packed:
            result[hook] = packed
    return result if result else None


def scale_steering_spec(
    spec: SteeringVectorSpecLike | None,
    scale: float,
) -> SteeringVectorSpecInternal | None:
    """Apply a uniform multiplier to every entry in *spec*.

    Returns a new spec where each layer entry's effective magnitude has
    been multiplied by *scale*.  Per-layer ``{"vector": ..., "scale": ...}``
    entries have their inner ``scale`` field multiplied; bare-list entries
    are wrapped in the dict form with the new scale.

    Used by the worker-side named-module resolver to apply the
    request's module-level scale before merging with inline overrides.
    Returns ``None`` if *spec* is ``None`` or empty.  When *scale* equals
    ``1.0`` the input is returned unchanged.
    """
    if not spec:
        return None
    if scale == 1.0:
        return cast(SteeringVectorSpecInternal, spec)
    result: SteeringVectorSpecInternal = {}
    for hook, layer_dict in spec.items():
        if not layer_dict:
            continue
        scaled_layers: dict[int, SteeringLayerEntryInternal] = {}
        for layer_idx, entry in layer_dict.items():
            vec, sc = normalize_layer_entry(entry)
            scaled_layers[layer_idx] = {"vector": vec, "scale": sc * scale}
        if scaled_layers:
            result[hook] = scaled_layers
    return result if result else None


def merge_steering_specs(
    a: SteeringVectorSpecLike | None,
    b: SteeringVectorSpecLike | None,
) -> ResolvedSteeringVectorSpec | None:
    """Additively merge two :class:`SteeringVectorSpec` dicts.

    For overlapping ``(hook, layer)`` entries both sides are pre-scaled
    (via :func:`normalize_layer_entry` + :func:`_scale_vector`) then
    summed (via :func:`_add_vectors`).  Non-overlapping entries pass
    through pre-scaled.

    Returns ``None`` if both inputs are ``None`` or empty.
    """
    a_empty = not a
    b_empty = not b
    if a_empty and b_empty:
        return None

    result: ResolvedSteeringVectorSpec = {}

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

        hook_result: dict[int, np.ndarray] = {}
        for layer_idx in all_layer_idxs:
            a_entry = a_layers.get(layer_idx)
            b_entry = b_layers.get(layer_idx)

            if a_entry is not None and b_entry is not None:
                a_vec, a_scale = normalize_layer_entry(a_entry)
                b_vec, b_scale = normalize_layer_entry(b_entry)
                scaled_a = _scale_vector(a_vec, a_scale)
                scaled_b = _scale_vector(b_vec, b_scale)
                hook_result[layer_idx] = _add_vectors(scaled_a, scaled_b)
            elif a_entry is not None:
                vec, scale = normalize_layer_entry(a_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)
            else:
                assert b_entry is not None
                vec, scale = normalize_layer_entry(b_entry)
                hook_result[layer_idx] = _scale_vector(vec, scale)

        if hook_result:
            result[hook] = hook_result

    return result if result else None


def hash_steering_config(
    effective_vectors: SteeringVectorSpecLike | None,
    module_ref: tuple[str, float] | None = None,
    sae_clamp_specs: tuple[Any, ...] | None = None,
    sae_full_reconstruction_specs: tuple[Any, ...] | None = None,
) -> int:
    """Deterministic SHA-256 hash of pre-resolved steering vectors.

    Returns 0 if *effective_vectors*, *module_ref*, and
    *sae_clamp_specs* are all ``None`` or empty.  The hash is masked
    to fit in ``np.int64``.

    *module_ref* is an optional ``(name, scale)`` reference to a
    worker-side named steering module.  When set, the reference is
    incorporated into the hash so that two requests with the same
    ``(name, scale)`` reference plus identical inline overrides produce
    the same hash, while different references (or different scales)
    produce different hashes.  When ``module_ref`` is ``None`` this
    function reduces to the original "hash inline-only vectors" behavior
    bit-for-bit, preserving prefix-cache reuse for existing requests.

    *sae_clamp_specs* is an optional tuple of
    :class:`vllm.config.sae_steering_types.SAEClampSpec` describing
    SAE feature-surgery clamps.  When ``None`` or empty the SAE block
    is omitted from the digest, so requests that don't use SAE
    steering hash bit-for-bit identically to before this argument
    existed.  This is required for prefix-cache reuse.

    Accepts entries as either ``list[float]`` (legacy callers) or
    ``np.ndarray`` (the float64 arrays produced by
    :func:`resolve_effective_vectors`).  In both cases the float→float32
    cast happens exactly once at the ``tobytes`` boundary, so hashes are
    bit-for-bit identical regardless of the input container.
    """
    if (
        not effective_vectors
        and module_ref is None
        and not sae_clamp_specs
        and not sae_full_reconstruction_specs
    ):
        return 0
    h = hashlib.sha256()
    if effective_vectors:
        for hook in sorted(effective_vectors.keys()):
            h.update(hook.encode())
            layer_dict = effective_vectors[hook]
            for layer_idx in sorted(layer_dict.keys()):
                entry = layer_dict[layer_idx]
                # An entry is either a list/ndarray of floats or a dict
                # ``{"vector": [...], "scale": float}``. By the time we get here
                # the resolver has flattened the dict form into a plain
                # array — but handle both for safety.
                if isinstance(entry, dict):
                    vec = entry.get("vector", entry)
                    scale = float(entry.get("scale", 1.0))
                else:
                    vec = entry
                    scale = 1.0
                arr = np.asarray(vec, dtype=np.float32)
                h.update(_steering_index_bytes(layer_idx, "layer_idx"))
                h.update(arr.tobytes())
                if scale != 1.0:
                    h.update(np.float64(scale).tobytes())
    if module_ref is not None:
        # Domain-separator byte ensures a request with only a module_ref
        # cannot collide with an inline-vector request whose hook name
        # happens to match the module name.  Inline vectors are written
        # before this block, so the separator unambiguously demarcates
        # the module-ref segment of the digest.
        name, scale = module_ref
        h.update(b"\x00module_ref\x00")
        h.update(name.encode("utf-8"))
        h.update(np.float64(scale).tobytes())
    if sae_clamp_specs:
        # Defer the import to avoid a circular dependency: sae_steering_types
        # imports VALID_HOOK_POINT_NAMES from model_executor.layers.steering,
        # which transitively does not depend on this module — but the cycle
        # is fragile, and a function-local import keeps this file's import
        # graph minimal.  ``hash_sae_clamp_specs`` writes its own domain
        # separator into the digest it computes; we incorporate that result
        # via ``int.to_bytes`` to keep the composition associative.
        from vllm.config.sae_steering_types import hash_sae_clamp_specs

        sae_hash = hash_sae_clamp_specs(sae_clamp_specs)
        h.update(b"\x00sae_block\x00")
        h.update(sae_hash.to_bytes(8, "little", signed=False))
    if sae_full_reconstruction_specs:
        # Distinct domain separator from the delta block above so a
        # delta and a full-reconstruction request with identical clamp
        # content can never collide on prefix-cache keys: the two paths
        # produce different residual streams (perturbation vs
        # replacement) and must not share prefill cache.
        from vllm.config.sae_steering_types import (
            hash_sae_full_reconstruction_specs,
        )

        recon_hash = hash_sae_full_reconstruction_specs(sae_full_reconstruction_specs)
        h.update(b"\x00sae_full_recon_block\x00")
        h.update(recon_hash.to_bytes(8, "little", signed=False))
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def maybe_pack_inline_steering_for_request(
    sp: SamplingParams,
    torch_dtype: object,
) -> None:
    """Pre-resolve and pack inline steering vectors on *sp* in-place."""
    if sp.steering_module_ref is not None:
        # Named-module requests may still carry inline base / phase
        # overrides.  Those raw tiers are needed on the worker so
        # ``_resolve_request_steering`` can merge them with the named
        # module's tiers.  The packed effective inline form is only
        # valid for inline-only requests.
        return
    if (
        sp.steering_vectors is None
        and sp.prefill_steering_vectors is None
        and sp.decode_steering_vectors is None
    ):
        return
    if (
        sp._effective_prefill_steering_packed is not None
        or sp._effective_decode_steering_packed is not None
    ):
        return

    np_dtype = _torch_dtype_to_pack_dtype(torch_dtype)

    # Prime hashes before mutating inline fields. This preserves
    # prefix-cache parity between packed and unpacked submissions,
    # including the SAE hash contribution when present.  The additive-only
    # row keys must also be fixed here so the first unpromoted request and
    # later auto-promoted siblings use the same physical-row identity even
    # when packing casts vectors to fp16/bf16-compatible payloads.
    _ = sp.prefill_steering_config_hash
    _ = sp.decode_steering_config_hash
    _ = sp.prefill_additive_steering_config_hash
    _ = sp.decode_additive_steering_config_hash
    _ = sp.prefill_sae_clamp_config_hash
    _ = sp.decode_sae_clamp_config_hash

    sp._effective_prefill_steering_packed = pack_effective_steering(
        sp.steering_vectors, sp.prefill_steering_vectors, np_dtype
    )
    sp._effective_decode_steering_packed = pack_effective_steering(
        sp.steering_vectors, sp.decode_steering_vectors, np_dtype
    )
    sp.steering_vectors = None
    sp.prefill_steering_vectors = None
    sp.decode_steering_vectors = None
    sp.__dict__["effective_prefill_steering"] = sp._effective_prefill_steering_packed
    sp.__dict__["effective_decode_steering"] = sp._effective_decode_steering_packed


class SteeringAutoPromoteLRU:
    """Per-engine LRU tracking which inline specs have been seen."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError(f"LRU capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._items: OrderedDict[tuple[int, int], str | None] = OrderedDict()

    def observe(
        self, key: tuple[int, int]
    ) -> tuple[str, str | None, tuple[tuple[int, int], str | None] | None]:
        existing = self._items.get(key, ...)
        if existing is ...:
            evicted = self._put_new(key, None)
            return "first", None, evicted
        self._items.move_to_end(key)
        if existing is None:
            return "second", None, None
        return "registered", existing, None

    def mark_registered(self, key: tuple[int, int], name: str) -> None:
        if key not in self._items:
            raise KeyError(key)
        self._items[key] = name
        self._items.move_to_end(key)

    def _put_new(
        self, key: tuple[int, int], name: str | None
    ) -> tuple[tuple[int, int], str | None] | None:
        evicted: tuple[tuple[int, int], str | None] | None = None
        if len(self._items) >= self._capacity:
            for evicted_key, evicted_name in self._items.items():
                if evicted_name is None:
                    del self._items[evicted_key]
                    evicted = (evicted_key, evicted_name)
                    break
            else:
                return None
        self._items[key] = name
        return evicted

    def get(self, key: tuple[int, int]) -> str | None:
        existing = self._items.get(key)
        if existing is None:
            return None
        self._items.move_to_end(key)
        return existing

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


def _build_named_payload_from_resolved(
    sp: SamplingParams,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Build a worker broadcast payload from a request's resolved vectors."""
    payload: dict[str, dict[str, dict[int, list[float]]]] = {}

    prefill = sp.effective_prefill_steering
    decode = sp.effective_decode_steering

    def _to_list_dict(
        spec: dict[str, dict[int, np.ndarray]] | None,
    ) -> dict[str, dict[int, list[float]]] | None:
        if not spec:
            return None
        return {
            hook: {
                layer: arr.astype(np.float32, copy=False).tolist()
                for layer, arr in layer_dict.items()
            }
            for hook, layer_dict in spec.items()
        }

    prefill_payload = _to_list_dict(prefill)
    decode_payload = _to_list_dict(decode)

    if prefill_payload == decode_payload:
        if prefill_payload is not None:
            payload["vectors"] = prefill_payload
    else:
        if prefill_payload is not None:
            payload["prefill_vectors"] = prefill_payload
        if decode_payload is not None:
            payload["decode_vectors"] = decode_payload
    return payload


def _auto_promote_prep(
    sp: SamplingParams,
    registry_lru: SteeringAutoPromoteLRU,
) -> (
    tuple[
        tuple[int, int] | None,
        tuple[int, int] | None,
        str | None,
        dict[str, dict[str, dict[int, list[float]]]] | None,
        tuple[tuple[int, int], str | None] | None,
    ]
    | None
):
    if sp.steering_module_ref is not None:
        return None
    has_inline = (
        sp.steering_vectors is not None
        or sp.prefill_steering_vectors is not None
        or sp.decode_steering_vectors is not None
    )
    has_packed = (
        sp._effective_prefill_steering_packed is not None
        or sp._effective_decode_steering_packed is not None
    )
    if not has_inline and not has_packed:
        return None

    original_hashes = (
        sp.prefill_steering_config_hash,
        sp.decode_steering_config_hash,
    )
    # Scheduler admission may need SAE-only row hashes after the inline
    # fields are rewritten to a generated module reference.  Prime them now
    # so promotion does not defer SAE hashing into the scheduler hot path.
    _ = sp.prefill_sae_clamp_config_hash
    _ = sp.decode_sae_clamp_config_hash
    # Auto-promotion only registers the additive inline payload.  SAE clamps
    # stay on the request, but including them in the LRU key would duplicate
    # identical additive modules for each SAE clamp variant.
    #
    # Use the cached additive identity rather than re-hashing
    # ``effective_*_steering`` directly.  Offline LLM.add_request may reuse the
    # same SamplingParams object for multiple prompts; the first prompt can pack
    # inline vectors to fp16/bf16 before the second prompt reaches this path.
    # The cached additive hashes are primed before packing, so the second
    # observation still hits the same LRU key instead of looking like a new
    # rounded payload.
    h_prefill = sp.prefill_additive_steering_config_hash
    h_decode = sp.decode_additive_steering_config_hash
    key = (h_prefill, h_decode)
    status, name, evicted = registry_lru.observe(key)

    if status == "first":
        if evicted is not None and evicted[1] is not None:
            return None, None, None, None, evicted
        return None
    if status == "registered":
        assert name is not None
        return key, original_hashes, name, None, None

    name = f"_auto_{h_prefill:016x}_{h_decode:016x}"
    payload = _build_named_payload_from_resolved(sp)
    if not payload:
        return None
    return key, original_hashes, name, payload, evicted


def _auto_promote_apply(
    sp: SamplingParams,
    name: str,
    original_hashes: tuple[int, int],
    additive_hashes: tuple[int, int],
) -> None:
    sp.steering_module_ref = (name, 1.0)
    sp.steering_vectors = None
    sp.prefill_steering_vectors = None
    sp.decode_steering_vectors = None
    sp._effective_prefill_steering_packed = None
    sp._effective_decode_steering_packed = None
    sp._auto_promote_original_hashes = original_hashes
    sp.__dict__.pop("effective_prefill_steering", None)
    sp.__dict__.pop("effective_decode_steering", None)
    # Auto-promotion is a transport optimization: the worker reads the
    # vectors through a generated module name, but the logical request
    # identity remains the pre-promotion inline payload plus any SAE
    # clamps.  Keep these cached hashes stable so prefix-cache keys and
    # row-capacity accounting match an unpromoted request.
    sp.__dict__["prefill_steering_config_hash"] = original_hashes[0]
    sp.__dict__["decode_steering_config_hash"] = original_hashes[1]
    sp.__dict__["prefill_additive_steering_config_hash"] = additive_hashes[0]
    sp.__dict__["decode_additive_steering_config_hash"] = additive_hashes[1]


def maybe_auto_promote_steering_modules(
    sp: SamplingParams,
    rpc_fn: Callable[..., Any],
    registry_lru: SteeringAutoPromoteLRU,
) -> None:
    """Promote repeated inline steering specs to named modules."""
    prep = _auto_promote_prep(sp, registry_lru)
    if prep is None:
        return
    key, original_hashes, name, payload, evicted = prep
    if payload is not None:
        assert key is not None
        assert name is not None
        rpc_fn(
            "register_steering_modules",
            kwargs={"modules": {name: payload}, "replace": False},
        )
        registry_lru.mark_registered(key, name)
    if evicted is not None:
        _, evicted_name = evicted
        if evicted_name is not None:
            rpc_fn(
                "unregister_steering_modules",
                kwargs={"names": [evicted_name]},
            )
    if name is not None:
        assert key is not None
        assert original_hashes is not None
        _auto_promote_apply(sp, name, original_hashes, key)


async def maybe_auto_promote_steering_modules_async(
    sp: SamplingParams,
    rpc_fn: Callable[..., Any],
    registry_lru: SteeringAutoPromoteLRU,
) -> None:
    """Async variant for ``AsyncLLM.add_request``."""
    prep = _auto_promote_prep(sp, registry_lru)
    if prep is None:
        return
    key, original_hashes, name, payload, evicted = prep
    if payload is not None:
        assert key is not None
        assert name is not None
        await rpc_fn(
            "register_steering_modules",
            kwargs={"modules": {name: payload}, "replace": False},
        )
        registry_lru.mark_registered(key, name)
    if evicted is not None:
        _, evicted_name = evicted
        if evicted_name is not None:
            await rpc_fn(
                "unregister_steering_modules",
                kwargs={"names": [evicted_name]},
            )
    if name is not None:
        assert key is not None
        assert original_hashes is not None
        _auto_promote_apply(sp, name, original_hashes, key)
