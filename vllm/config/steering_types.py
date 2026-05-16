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

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

# Per-layer entry: bare list (scale=1.0) or {"vector": [...], "scale": float}.
# This is the public, user-facing shape — the type alias is exposed as a
# pydantic field type by request/response models in ``vllm.entrypoints``,
# so it must remain a narrow, schema-generable union.  Internally,
# :func:`merge_steering_specs` produces ``np.ndarray`` entries that are
# re-fed into resolvers; :func:`normalize_layer_entry` handles that
# transparently without widening this alias.
SteeringLayerEntry = list[float] | dict[str, Any]

# Full spec: {hook_point_name: {layer_idx: SteeringLayerEntry}}
SteeringVectorSpec = dict[str, dict[int, SteeringLayerEntry]]


def normalize_layer_entry(
    entry: SteeringLayerEntry | np.ndarray,
) -> tuple[list[float] | np.ndarray, float]:
    """Return ``(vector, scale)`` from a steering layer entry.

    If *entry* is a bare ``list[float]`` or ``np.ndarray``, returns
    ``(entry, 1.0)``.  If *entry* is ``{"vector": [...], "scale": float}``,
    returns ``(entry["vector"], entry["scale"])``.

    The ndarray case supports re-feeding the output of
    :func:`merge_steering_specs` (which produces pre-scaled ``np.float64``
    arrays) through downstream resolvers without converting back to lists.
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
    base: SteeringVectorSpec | None,
    phase_specific: SteeringVectorSpec | None,
) -> dict[str, dict[int, np.ndarray]] | None:
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
    """Pick the numpy dtype to pack steering vectors as for *torch_dtype*.

    Maps the model's compute dtype to a numpy dtype for the wire-format
    packing path.  Numpy lacks a native ``bfloat16`` (without the
    optional ``ml_dtypes`` package), so bf16 models fall back to
    ``float32`` — still a ~2.25× IPC reduction over msgpack-encoded
    Python float lists.
    """
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
    """Resolve and pack inline steering specs in one shot.

    Equivalent to ``resolve_effective_vectors(spec_base, spec_phase)``
    cast to *dtype*, but does the cast inline so we never allocate the
    intermediate float64 arrays purely for the cast-then-discard.
    Used by the LLM client (and HTTP server) to build the
    ``effective_*_steering_packed`` fields on :class:`SamplingParams`
    before request submission.

    Returns ``None`` when both inputs are ``None`` / empty.
    """
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
    """Pre-bake a :class:`SteeringVectorSpec` into model-dtype ``ndarray`` form.

    Converts every per-layer entry — bare-list or ``{"vector", "scale"}``
    dict — into a 1-D ``np.ndarray`` in *dtype* with the inner ``scale``
    already applied.  Returned shape:
    ``{hook: {layer_idx: ndarray[dtype]}}``.

    Used by the inline-vectors fast path: the LLM client (or HTTP server)
    converts user-supplied list-of-floats into this packed form before
    serializing into the request body, so the wire payload carries
    ``len(vec) * dtype.itemsize`` bytes per layer instead of
    ``len(vec) * 9`` bytes (msgpack-encoded floats).  The downstream
    resolver (:func:`resolve_effective_vectors` and
    :func:`merge_steering_specs`) already accepts ``ndarray`` entries
    transparently, so packed inputs flow through without further
    conversion.

    Returns ``None`` if *spec* is ``None`` or empty.
    """
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
                # Cast the scale to the target dtype so the multiply
                # stays in-place (avoiding an fp32 promotion that the
                # caller would only have to cast back).
                arr = arr * np_dtype.type(scale)
                # ``arr * scalar`` on a non-fp32 ndarray sometimes
                # returns a higher-precision result; force it back.
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype, copy=False)
            packed[layer_idx] = arr
        if packed:
            result[hook] = packed
    return result if result else None


def scale_steering_spec(
    spec: SteeringVectorSpec | None,
    scale: float,
) -> SteeringVectorSpec | None:
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
        return spec
    result: SteeringVectorSpec = {}
    for hook, layer_dict in spec.items():
        if not layer_dict:
            continue
        scaled_layers: dict[int, SteeringLayerEntry] = {}
        for layer_idx, entry in layer_dict.items():
            vec, sc = normalize_layer_entry(entry)
            scaled_layers[layer_idx] = {"vector": vec, "scale": sc * scale}
        if scaled_layers:
            result[hook] = scaled_layers
    return result if result else None


def merge_steering_specs(
    a: SteeringVectorSpec | None,
    b: SteeringVectorSpec | None,
) -> SteeringVectorSpec | None:
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
    effective_vectors: dict[str, dict[int, list[float] | np.ndarray]] | None,
    module_ref: tuple[str, float] | None = None,
) -> int:
    """Deterministic SHA-256 hash of pre-resolved steering vectors.

    Returns 0 if both *effective_vectors* and *module_ref* are ``None``
    or empty.  The hash is masked to fit in ``np.int64``.

    *module_ref* is an optional ``(name, scale)`` reference to a
    worker-side named steering module.  When set, the reference is
    incorporated into the hash so that two requests with the same
    ``(name, scale)`` reference plus identical inline overrides produce
    the same hash, while different references (or different scales)
    produce different hashes.  When ``module_ref`` is ``None`` this
    function reduces to the original "hash inline-only vectors" behavior
    bit-for-bit, preserving prefix-cache reuse for existing requests.

    Accepts entries as either ``list[float]`` (legacy callers) or
    ``np.ndarray`` (the float64 arrays produced by
    :func:`resolve_effective_vectors`).  In both cases the float→float32
    cast happens exactly once at the ``tobytes`` boundary, so hashes are
    bit-for-bit identical regardless of the input container.
    """
    if not effective_vectors and module_ref is None:
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
                h.update(layer_idx.to_bytes(4, "little", signed=True))
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
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def maybe_pack_inline_steering_for_request(
    sp: SamplingParams,
    torch_dtype: object,
) -> None:
    """Pre-resolve + pack inline steering vectors on *sp* in-place.

    Shared between :class:`vllm.entrypoints.LLM` (sync path) and
    :class:`vllm.v1.engine.async_llm.AsyncLLM` (HTTP path), called just
    before the request crosses the multiprocessing boundary.  See
    :meth:`SamplingParams._effective_prefill_steering_packed` for the
    contract.

    No-ops when:

    - all inline tier fields are ``None`` (named-only or no-steering);
    - packed fields are already set (idempotency for callers that
      pre-packed).

    Mutates *sp* by:
    1. Reading ``prefill_steering_config_hash`` and
       ``decode_steering_config_hash`` to fix the hash against the
       original fp64-resolved values (preserves prefix-cache reuse).
    2. Setting ``effective_*_steering_packed`` to the model-dtype
       ``ndarray`` form of the resolved per-phase specs.
    3. Clearing ``steering_vectors`` / ``prefill_steering_vectors`` /
       ``decode_steering_vectors`` so the wire payload doesn't carry
       both forms.
    4. Stashing the packed dicts as the cached values for the
       ``effective_*_steering`` cached_properties so worker-side reads
       return them directly without re-resolving.
    """
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

    # Prime the hash cached_properties against the original fp64 path
    # so a packed and an unpacked submission of the same logical request
    # share a prefix-cache hash.
    _ = sp.prefill_steering_config_hash
    _ = sp.decode_steering_config_hash

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


# ---------------------------------------------------------------------------
# Inline → named auto-promotion (Branch A)
# ---------------------------------------------------------------------------


class SteeringAutoPromoteLRU:
    """Per-engine LRU tracking which inline specs have been seen.

    Two-strikes promotion: on first sight of a ``(prefill_hash,
    decode_hash)`` pair we mark it as seen but do **not** broadcast
    ``register_steering_modules`` — the request flows through the
    inline-pack path normally.  On the second sight we register the
    spec under a hash-derived name and start promoting subsequent
    requests with that hash to use ``steering_module_ref``.

    This avoids regressing genuinely-unique-per-request workloads
    (research sweeps where every spec is fresh) which otherwise pay an
    extra synchronous ``collective_rpc`` per request for no benefit —
    auto-promote would only ever miss the cache.

    Entry states:

    - present with value ``None``: seen once (no broadcast yet)
    - present with value ``(name, "pending")``: a name has been allocated
      and the ``register_steering_modules`` broadcast has been dispatched
      but not yet confirmed against the worker.  Callers must NOT rewrite
      the request to use ``module_ref`` in this state — the named module
      may not be installed on the worker yet, and shipping the ref races
      the broadcast through different IPC channels (``collective_rpc`` vs
      the request submit FIFO).  The trigger request and any other
      request observed while pending fall through to the inline path.
    - present with value ``(name, "registered")``: the broadcast has been
      confirmed (or the sync RPC has returned) — subsequent requests with
      this hash are safe to rewrite to use ``steering_module_ref``.

    Eviction returns ``(key, name_or_None)`` so callers can issue a
    paired ``unregister_steering_modules`` against the worker — but only
    when the evicted entry was actually registered with a name
    (``name is not None``).  Both ``"pending"`` and ``"registered"``
    entries carry a name; ``"pending"`` evictions still need an
    unregister so the worker drops any partial state it received.

    Capacity caps the worker-side memory footprint of cached resolved
    specs.  Default 512 entries × ~520 KB of bf16 vectors = ~260 MB
    per worker.
    """

    # Per-entry value: ``None`` for first-sight, ``(name, status)`` once a
    # name has been allocated.  The status discriminates between in-flight
    # broadcast ("pending") and confirmed ("registered").
    _Entry = tuple[str, str] | None

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError(f"LRU capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._items: OrderedDict[
            tuple[int, int], tuple[str, str] | None
        ] = OrderedDict()

    def observe(
        self, key: tuple[int, int]
    ) -> tuple[str, str | None, tuple[tuple[int, int], str | None] | None]:
        """Record that *key* was seen and return ``(status, name, evicted)``.

        ``status`` is one of:

        - ``"first"``: never seen before; caller should fall through to
          inline-pack and skip the broadcast.  ``name`` is always
          ``None``.  ``evicted`` may carry the LRU's overflow.
        - ``"second"``: seen once previously without registration; this
          is the trigger to allocate a name and dispatch the register
          broadcast.  ``name`` is ``None`` (the caller picks the name and
          records it via :meth:`mark_pending`).  ``evicted`` is ``None``.
          The triggering request must NOT be rewritten — the broadcast
          has not landed on the worker yet.
        - ``"pending"``: a previous observation has dispatched the
          register broadcast but it has not yet been confirmed.  ``name``
          is the allocated name; the caller must NOT rewrite the request
          to use ``module_ref`` (the worker may not have processed the
          register RPC yet).  ``evicted`` is ``None``.
        - ``"registered"``: the broadcast has been confirmed.  ``name``
          is the stored module name; the caller MAY rewrite the request
          to use ``module_ref``.  ``evicted`` is ``None``.
        """
        existing = self._items.get(key, ...)
        if existing is ...:
            evicted = self._put_new(key, None)
            return "first", None, evicted
        # Refresh recency on any hit.
        self._items.move_to_end(key)
        if existing is None:
            return "second", None, None
        name, status = existing
        return status, name, None

    def mark_pending(self, key: tuple[int, int], name: str) -> None:
        """Promote the existing entry from "seen" to "pending".

        Called by the auto-promote helper at the moment of dispatching
        the register broadcast.  The entry carries the allocated name
        but ``observe`` will return ``status="pending"`` until
        :meth:`mark_registered` is called from the broadcast's
        done-callback.
        """
        if key not in self._items:
            raise KeyError(key)
        self._items[key] = (name, "pending")
        self._items.move_to_end(key)

    def mark_registered(self, key: tuple[int, int], name: str) -> None:
        """Promote the existing entry to "registered".

        Two callers:

        - Sync auto-promote (after ``collective_rpc`` returns
          successfully): goes directly from "seen" to "registered" since
          there is no broadcast-in-flight window.
        - Async auto-promote (from the broadcast done-callback): flips
          a previously "pending" entry to "registered".
        """
        if key not in self._items:
            raise KeyError(key)
        self._items[key] = (name, "registered")
        self._items.move_to_end(key)

    def _put_new(
        self,
        key: tuple[int, int],
        value: tuple[str, str] | None,
    ) -> tuple[tuple[int, int], str | None] | None:
        evicted: tuple[tuple[int, int], str | None] | None = None
        if len(self._items) >= self._capacity:
            evicted_key, evicted_value = self._items.popitem(last=False)
            evicted_name = evicted_value[0] if evicted_value is not None else None
            evicted = (evicted_key, evicted_name)
        self._items[key] = value
        return evicted

    def get(self, key: tuple[int, int]) -> str | None:
        """Read-only lookup of a *registered* name.

        Returns the name only when the entry is in ``"registered"``
        status — first-sight and pending entries return ``None`` so
        callers using ``get`` as a cache lookup don't accidentally
        promote a request against an unconfirmed broadcast.  Used by
        tests; production callers go through :meth:`observe`.
        """
        existing = self._items.get(key)
        if existing is None:
            return None
        self._items.move_to_end(key)
        name, status = existing
        if status != "registered":
            return None
        return name

    def get_status(self, key: tuple[int, int]) -> str | None:
        """Return the entry's status or ``None`` when absent.

        Status values mirror :meth:`observe`'s ``status`` field but
        without mutating recency: ``"first"`` for a first-sight entry,
        ``"pending"`` or ``"registered"`` for an entry with an allocated
        name.  Used by tests to assert the broadcast lifecycle without
        side effects.
        """
        existing = self._items.get(key, ...)
        if existing is ...:
            return None
        if existing is None:
            return "first"
        return existing[1]

    def forget(self, key: tuple[int, int]) -> None:
        """Drop *key* from the LRU if present.

        Used by the async broadcast path to roll back a pending entry
        when ``register_steering_modules`` fails — the next request with
        the same hash will then observe ``"first"`` status and fall back
        to the inline-pack path instead of shipping a ``module_ref`` to
        a name the worker never received.  No-op when *key* is absent.
        """
        self._items.pop(key, None)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


def _build_named_payload_from_resolved(
    sp: SamplingParams,
) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Build the broadcast payload for a request's already-resolved spec.

    ``register_steering_modules`` expects the payload shape
    ``{"vectors": {hook: {layer: list[float]}}, "prefill_vectors": ...,
    "decode_vectors": ...}``.  Auto-promotion registers the *resolved*
    effective spec under the ``vectors`` key (single-tier) — the worker
    treats it as a base spec with no phase-specific overrides.

    Reads the ``effective_*_steering`` cached_properties (which fall
    back to the packed fields if set, else resolve fresh) so the
    payload reflects what the worker would have computed anyway.
    """
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

    # The named-module worker resolver merges base + phase tiers.  When
    # prefill and decode are equal (the common case for a request whose
    # only steering is in ``steering_vectors``), put the spec under
    # ``vectors`` (base) so both phases reuse the same resolved cache
    # entry.  When they differ, place each into its phase-specific tier.
    if prefill_payload == decode_payload:
        if prefill_payload is not None:
            payload["vectors"] = prefill_payload
    else:
        if prefill_payload is not None:
            payload["prefill_vectors"] = prefill_payload
        if decode_payload is not None:
            payload["decode_vectors"] = decode_payload
    return payload


AUTO_PROMOTE_NAME_PREFIX = "_auto_"


def auto_promote_hashes_from_module_ref(
    module_ref: tuple[str, float] | None,
) -> tuple[int, int] | None:
    """Recover ``(prefill_hash, decode_hash)`` from an auto-promoted name.

    Returns ``None`` when *module_ref* is ``None`` or names a user-defined
    module.  Auto-promoted names follow the format
    ``_auto_<prefill_hex16>_<decode_hex16>`` and embed the original
    pre-promotion inline-content hashes (see
    :func:`_auto_promote_prep`), so the request hash can be recovered
    by parsing the name alone — no access to the original inline
    vectors required.  This is the linchpin of auto-promote being a
    transport-only optimization: the request identity stays equal to
    the inline-content hash on both sides of the promotion.
    """
    if module_ref is None:
        return None
    name, _ = module_ref
    if not name.startswith(AUTO_PROMOTE_NAME_PREFIX):
        return None
    body = name[len(AUTO_PROMOTE_NAME_PREFIX):]
    parts = body.split("_")
    if len(parts) != 2 or len(parts[0]) != 16 or len(parts[1]) != 16:
        return None
    try:
        return int(parts[0], 16), int(parts[1], 16)
    except ValueError:
        return None


def _auto_promote_prep(
    sp: SamplingParams,
    registry_lru: SteeringAutoPromoteLRU,
    *,
    mark_status: str = "pending",
) -> (
    tuple[
        str | None,
        dict[str, dict[str, dict[int, list[float]]]] | None,
        tuple[tuple[int, int], str | None] | None,
        tuple[int, int] | None,
    ]
    | None
):
    """Shared eligibility + two-strikes cache lookup for auto-promote.

    Returns ``None`` when:
    - *sp* is ineligible (already named, no inline vectors, already packed); or
    - this is a first sight with no eviction that needs cleanup — fall through
      to inline-pack so unique-per-request workloads don't pay a wasted RPC; or
    - this is a "pending" sight: a broadcast is in flight for the same hash
      but has not yet been confirmed, so we cannot promote *sp* yet.  The
      request flows through inline-pack until the broadcast lands.

    Otherwise returns ``(name_or_None, payload, evicted, lru_key)`` where:

    - ``name_or_None``: ``None`` at "second" sight (the trigger does NOT
      rewrite — broadcast races request submission on the async path).
      ``None`` also when the only action is to issue an
      ``unregister_steering_modules`` for an evicted registered entry.
      A string name on a "registered" cache hit — the caller installs
      this on *sp*.
    - ``payload``: the broadcast payload for ``register_steering_modules`` —
      non-``None`` only at second-sight (the trigger to register).
    - ``evicted``: an LRU eviction tuple ``(key, prior_name)`` where
      ``prior_name`` may be ``None`` for evictions of unregistered
      first-sight entries.  Caller skips the unregister RPC when
      ``prior_name`` is ``None``.
    - ``lru_key``: the ``(prefill_hash, decode_hash)`` key for the
      registered entry; ``None`` when no register broadcast is being
      dispatched.  The async caller uses this in the broadcast
      done-callback to flip the LRU from "pending" → "registered" on
      success or to ``forget`` it on failure.

    *mark_status* controls how the LRU is marked at second-sight:
    ``"pending"`` (default, async path) records the name but holds back
    SP rewrite until the broadcast confirms; ``"registered"`` (sync
    path) marks immediately because the sync caller awaits the
    ``collective_rpc`` before returning, so by request-submit time the
    worker has the module installed.

    Pure / sync — does no IO.  Caller issues the broadcast(s) and
    optionally mutates *sp* via :func:`_auto_promote_apply`.
    """
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

    # Read the cached_properties to compute the dedup key.  These are
    # primed by ``maybe_pack_inline_steering_for_request`` against the
    # original fp64 path, so when *sp* is already packed the
    # ``effective_*_steering`` falls back to the packed dicts and the
    # hash comes out bit-for-bit identical to the pre-pack value.
    h_prefill = sp.prefill_steering_config_hash
    h_decode = sp.decode_steering_config_hash
    key = (h_prefill, h_decode)

    status, name, evicted = registry_lru.observe(key)

    if status == "first":
        # Don't promote *sp*, but if we evicted a *registered* (or pending)
        # entry from the LRU tail we still need to tell the worker to drop it.
        if evicted is not None and evicted[1] is not None:
            return None, None, evicted, None
        return None
    if status == "registered":
        assert name is not None
        return name, None, None, None
    if status == "pending":
        # A previous observation's broadcast is in flight for this hash.
        # The trigger request that fired the broadcast already fell
        # through to inline (see "second" branch below); this current
        # request must also fall through to inline rather than ship a
        # ``module_ref`` to a name the worker may not have processed yet.
        # ``observe`` already refreshed recency.  No RPC to issue here.
        return None
    # status == "second": this is the moment to register.  Behavior at
    # the trigger differs between sync and async callers:
    #
    # - Async (mark_status="pending"): the broadcast races the request
    #   through different IPC channels (collective_rpc vs the request
    #   submit FIFO).  Rewriting the trigger SP now and shipping the ref
    #   would land the request on a worker that has not yet seen the
    #   register RPC, so the trigger does NOT get rewritten.  The
    #   ``"pending"`` LRU marker holds back subsequent requests too,
    #   until the broadcast done-callback flips status to "registered".
    # - Sync (mark_status="registered"): the caller awaits
    #   collective_rpc before returning, so by the time the trigger
    #   request hits the wire the worker has the module.  The trigger
    #   IS safely rewritten.
    name = AUTO_PROMOTE_NAME_PREFIX + f"{h_prefill:016x}_{h_decode:016x}"
    payload = _build_named_payload_from_resolved(sp)
    if not payload:
        return None
    if mark_status == "registered":
        registry_lru.mark_registered(key, name)
        # Returning the name signals the sync caller to apply the SP
        # rewrite after the inline broadcast succeeds.
        return name, payload, evicted, key
    registry_lru.mark_pending(key, name)
    # Async caller: do NOT rewrite the trigger SP (name=None).  The
    # payload is still returned so the broadcast goes out.
    return None, payload, evicted, key


def _auto_promote_apply(sp: SamplingParams, name: str) -> None:
    """Final mutation step: clear inline + packed fields, install module ref.

    Both inline and packed fields are cleared because *sp* may have been
    pre-packed by an earlier request in the same shared-``[sp]*N`` batch.
    The request hash is intentionally preserved at its pre-promotion
    (inline-content) value: auto-promotion is purely a worker-side
    transport optimization (the worker resolves vectors from the named
    module instead of unpacking inline blobs), and folding the auto-
    generated ``_auto_<hex>`` name into the hash would change the
    request identity in flight.  Within a shared-``[sp]*N`` batch the
    first request reaches ``_add_request`` before promotion and ships
    with the inline hash, while siblings register after promotion;
    re-deriving the hash from ``module_ref`` here would produce a
    different value for siblings, doubling the ``(hash, phase)`` row
    slots the worker's strict-capacity table needs to service the same
    logical config across that batch.
    """
    sp.steering_module_ref = (name, 1.0)
    sp.steering_vectors = None
    sp.prefill_steering_vectors = None
    sp.decode_steering_vectors = None
    sp._effective_prefill_steering_packed = None
    sp._effective_decode_steering_packed = None
    # Drop the cached resolved-vector dicts only.  The hash cached_props
    # are deliberately left intact so the pre-promotion inline-content
    # hash continues to identify this request end-to-end.
    sp.__dict__.pop("effective_prefill_steering", None)
    sp.__dict__.pop("effective_decode_steering", None)


def maybe_auto_promote_steering_modules(
    sp: SamplingParams,
    rpc_fn: Callable[..., Any],
    registry_lru: SteeringAutoPromoteLRU,
) -> None:
    """Promote an inline steering request to the named-module fast path
    (synchronous variant for ``LLM._add_request``).

    On cache miss, synchronously broadcasts ``register_steering_modules``
    so the worker has the resolved spec by the time the request lands.
    On LRU overflow the evicted name is unregistered in a paired RPC.

    No-ops when *sp* is ineligible (already named, has no inline
    vectors, or is already packed).  See :func:`_auto_promote_prep` for
    the eligibility contract.

    Closes the inline-vs-named IPC bytes gap for repeated specs:
    ``[sp]*N`` workloads ship the spec once via the register RPC and
    then 16 bytes per request, instead of the full packed blob N
    times.  Genuinely-unique-per-request workloads pay one extra
    synchronous RPC per request — bench gates this case.
    """
    prep = _auto_promote_prep(sp, registry_lru, mark_status="registered")
    if prep is None:
        return
    name, payload, evicted, _lru_key = prep
    if payload is not None:
        assert name is not None
        rpc_fn(
            "register_steering_modules",
            kwargs={"modules": {name: payload}, "replace": False},
        )
    if evicted is not None:
        _, evicted_name = evicted
        if evicted_name is not None:
            rpc_fn(
                "unregister_steering_modules",
                kwargs={"names": [evicted_name]},
            )
    if name is not None:
        _auto_promote_apply(sp, name)


async def maybe_auto_promote_steering_modules_async(
    sp: SamplingParams,
    rpc_fn: Callable[..., Any],
    registry_lru: SteeringAutoPromoteLRU,
) -> "asyncio.Task[None] | None":
    """Async variant for ``AsyncLLM.add_request``.

    Identical eligibility / LRU semantics to
    :func:`maybe_auto_promote_steering_modules`, but the
    ``register_steering_modules`` (and paired
    ``unregister_steering_modules``) collective RPCs are dispatched as a
    background ``asyncio.Task`` rather than awaited inline.  The trace
    evidence (3090, gemma-3-4b-it, NUM_PROMPTS=8 CONCURRENCY=4) shows
    these broadcasts cost 32–44 ms each and map ~1:1 to a GPU idle gap
    on the engine thread, so blocking the request that triggered the
    promotion is the single biggest avoidable stall in inline-mode
    traces.

    Ordering correctness: the broadcast and the per-request submit
    flow over *different* IPC channels (``collective_rpc`` to the
    worker vs the request-submit FIFO to engine-core), so we cannot
    rely on socket ordering to guarantee the broadcast lands before
    the request.  Instead the LRU carries a ``"pending"``/``"registered"``
    status: the triggering request is NOT rewritten to use
    ``module_ref`` while the broadcast is in flight; subsequent
    requests with the same hash also fall through to inline until
    the broadcast's done-callback flips the LRU to ``"registered"``.
    From that point on, requests are promoted as designed.  This
    trades a small (one-request) loss of optimization for correctness
    — the alternative (the previous fire-and-forget behavior) raced
    the broadcast against the request submit and produced
    ``Steering module is not registered on this worker`` errors at
    concurrency ≥ 2 with a fresh spec.

    Failure handling: if the broadcast errors, the LRU entry is
    rolled back so the *next* request observing the same hash falls
    back to the inline-pack path (its ``observe`` returns ``"first"``).
    No in-flight request was ever rewritten to use ``module_ref`` —
    "pending" gates that — so there is nothing to recover on the
    request side.

    Returns the dispatched broadcast ``asyncio.Task`` (or ``None`` when
    no RPC is needed).  Production callers ignore the return value;
    tests can ``await`` it to observe the broadcast's effect on the
    spy ``rpc_fn`` and the LRU status flip.
    """
    prep = _auto_promote_prep(sp, registry_lru, mark_status="pending")
    if prep is None:
        return None
    name, payload, evicted, lru_key = prep

    # The async path never rewrites the trigger SP: ``_auto_promote_prep``
    # returns ``name=None`` at second-sight under mark_status="pending"
    # precisely so the trigger falls through to inline.  On a "registered"
    # cache hit ``name`` is non-None — the broadcast has already landed
    # on the worker, so it's safe to rewrite.
    if name is not None:
        _auto_promote_apply(sp, name)

    has_register = payload is not None
    has_unregister = evicted is not None and evicted[1] is not None
    if not has_register and not has_unregister:
        return None

    # The LRU key for the entry we just put into "pending" status (when
    # we dispatched a register).  Used in the done-callback to flip the
    # entry to "registered" on success or ``forget`` it on failure.
    pending_key: tuple[int, int] | None = lru_key if has_register else None
    pending_name: str | None = None
    if has_register:
        # Recover the allocated name from the LRU key — auto-promoted
        # names embed ``(prefill_hash, decode_hash)`` in their suffix.
        assert pending_key is not None
        pending_name = AUTO_PROMOTE_NAME_PREFIX + (
            f"{pending_key[0]:016x}_{pending_key[1]:016x}"
        )

    evicted_name = evicted[1] if evicted is not None else None

    async def _broadcast() -> None:
        if has_register:
            assert pending_name is not None and payload is not None
            try:
                await rpc_fn(
                    "register_steering_modules",
                    kwargs={"modules": {pending_name: payload}, "replace": False},
                )
            except Exception:
                # Roll back so subsequent requests fall back to inline-pack
                # rather than wait forever on a "pending" entry that will
                # never flip to "registered".  The ``forget`` is best-effort:
                # if the entry has already been evicted by a concurrent
                # observer the no-op pop is fine.
                if pending_key is not None:
                    registry_lru.forget(pending_key)
                raise
            # Broadcast confirmed.  Flip LRU "pending" → "registered" so
            # subsequent requests with this hash get promoted.  Guard
            # against the entry having been evicted by intervening
            # observations: if it's gone, there's nothing to mark and
            # the next request will simply re-observe as "first".
            if pending_key is not None and pending_key in registry_lru:
                registry_lru.mark_registered(pending_key, pending_name)
        if has_unregister:
            assert evicted_name is not None
            await rpc_fn(
                "unregister_steering_modules",
                kwargs={"names": [evicted_name]},
            )

    task: asyncio.Task[None] = asyncio.ensure_future(_broadcast())

    def _swallow(t: "asyncio.Task[None]") -> None:
        # Reading the exception marks it as retrieved so asyncio's
        # default "exception was never retrieved" warning stays quiet.
        # The LRU rollback inside ``_broadcast`` is the actual recovery.
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.warning(
                "Auto-promote broadcast failed; subsequent requests "
                "with the same steering hash will fall back to inline. "
                "Underlying error: %s",
                exc,
            )

    task.add_done_callback(_swallow)
    return task
