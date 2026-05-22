# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type definitions for SAE-based steering (delta / feature surgery).

This module defines the per-request types for the SAE feature-surgery
path: a clamp instructs the runtime to replace one SAE feature
activation with a target value (or shift it by an additive offset),
then add the resulting decoder-direction delta back into the residual
stream.

The runtime contract is documented in
``docs/features/sae_steering.md``.  The named-module *registry* (which
holds the SAE weights themselves) lives in
``vllm.entrypoints.openai.steering.registry``; this module only
contains the request-side types and their hashing helper.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from vllm.config.steering_types import validate_steering_index
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

SAEClampKind = Literal["absolute", "additive"]
"""Discriminator for the two clamp variants.

* ``"absolute"`` — set ``f_i := value``.  Always needs the live
  encoder activation so the runtime can compute the subtraction
  ``(value − f_i) · W_dec[:, i]``.
* ``"additive"`` — shift ``f_i := f_i + value``.  When
  ``only_if_active`` is False this collapses to the precomputed
  steering-vector op ``value · W_dec[:, i]`` and the encoder pass
  can be skipped.
"""

SAEClampPhase = Literal["both", "prefill", "decode"]
"""Phase tier for a clamp spec, mirroring the additive
``steering_vectors`` / ``prefill_steering_vectors`` /
``decode_steering_vectors`` triplet."""


class SteeringModuleKind(str, Enum):
    """Discriminator on entries in the named steering-module registry.

    Existing additive modules default to ``ADDITIVE`` and keep their
    current behaviour bit-for-bit.  ``SAE_DELTA`` modules carry SAE
    encoder/decoder weights and are referenced by per-request
    :class:`SAEClampSpec` entries.
    """

    ADDITIVE = "additive"
    SAE_DELTA = "sae_delta"


class SAEActivation(str, Enum):
    """SAE encoder activation function.

    The delta runtime supports these three activation families; gated
    and batch-TopK variants are intentionally out of scope.
    """

    RELU = "relu"
    JUMPRELU = "jumprelu"
    TOPK = "topk"


@dataclass(frozen=True)
class SAEClampEntry:
    """One feature clamp inside a :class:`SAEClampSpec`.

    Invariants enforced in ``__post_init__``:

    * ``feature_idx`` is a non-negative integer.
    * ``value`` is a finite float.
    * ``kind`` is one of ``SAEClampKind`` literals.
    * ``only_if_active`` is a bool.

    The derived :pyattr:`requires_encoder_pass` property tells the
    kernel whether the encoder GEMM is needed: True for absolute
    clamps (need to compute ``value − f_i``) or for ``only_if_active``
    additive clamps (need to gate on ``f_i > 0``); False for plain
    additive clamps where the delta is independent of the live
    activation.  Future kernel work uses this flag to short-circuit
    the encoder pass when the union of clamp entries on a hook
    contains no entry that needs it.
    """

    feature_idx: int
    kind: SAEClampKind
    value: float
    only_if_active: bool = False

    def __post_init__(self) -> None:
        validate_steering_index(self.feature_idx, "SAEClampEntry.feature_idx")
        if self.kind not in ("absolute", "additive"):
            raise ValueError(
                "SAEClampEntry.kind must be 'absolute' or 'additive', "
                f"got {self.kind!r}."
            )
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
        ):
            raise ValueError(
                f"SAEClampEntry.value must be a finite float, got {self.value!r}."
            )
        if not isinstance(self.only_if_active, bool):
            raise ValueError(
                "SAEClampEntry.only_if_active must be a bool, "
                f"got {type(self.only_if_active).__name__}."
            )

    @property
    def requires_encoder_pass(self) -> bool:
        """True iff applying this clamp needs the live encoder activation."""
        return self.kind == "absolute" or self.only_if_active


# Strongly-typed runtime form of a per-module clamp spec.
SAEClampLayerMap = dict[int, tuple[SAEClampEntry, ...]]
"""``layer_idx → tuple(entries)`` for a single (module, hook)."""

SAEClampHookMap = dict[str, SAEClampLayerMap]
"""``hook_point → layer_idx → tuple(entries)`` for a single module."""


@dataclass(frozen=True)
class SAEClampSpec:
    """Per-request clamp directive for one SAE module.

    A request may carry multiple :class:`SAEClampSpec` entries (one per
    referenced module).  Each spec carries a phase tier so that
    clamps can be limited to prefill or decode, mirroring the
    additive three-tier model.  Phase ``"both"`` applies the clamps
    in both phases.
    """

    module_name: str
    clamps: SAEClampHookMap
    phase: SAEClampPhase = "both"

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not self.module_name:
            raise ValueError(
                "SAEClampSpec.module_name must be a non-empty str, "
                f"got {self.module_name!r}."
            )
        if self.phase not in ("both", "prefill", "decode"):
            raise ValueError(
                "SAEClampSpec.phase must be 'both', 'prefill', or 'decode', "
                f"got {self.phase!r}."
            )
        if not isinstance(self.clamps, dict) or not self.clamps:
            raise ValueError(
                "SAEClampSpec.clamps must be a non-empty dict mapping "
                "hook-point names to per-layer clamp lists, got "
                f"{self.clamps!r}."
            )
        # ``self.clamps`` may arrive with list-valued entries (the
        # natural shape from JSON / direct test construction); we
        # canonicalize each to a tuple so the runtime contract holds
        # invariantly.  Frozen dataclasses don't allow attribute
        # mutation, so write through ``object.__setattr__`` after
        # building a fresh dict — the field type is still
        # ``SAEClampHookMap`` so downstream code sees ``tuple``.
        coerced: SAEClampHookMap = {}
        for hook_name, layer_map in self.clamps.items():
            if hook_name not in VALID_HOOK_POINT_NAMES:
                raise ValueError(
                    f"SAEClampSpec.clamps key {hook_name!r} is not a valid "
                    f"hook point.  Valid: {sorted(VALID_HOOK_POINT_NAMES)}."
                )
            if not isinstance(layer_map, dict) or not layer_map:
                raise ValueError(
                    f"SAEClampSpec.clamps[{hook_name!r}] must be a non-empty "
                    "dict mapping layer indices to clamp-entry tuples."
                )
            coerced_layer: SAEClampLayerMap = {}
            for layer_idx, entries in layer_map.items():
                validate_steering_index(
                    layer_idx, f"SAEClampSpec.clamps[{hook_name!r}] layer key"
                )
                if not isinstance(entries, (list, tuple)) or not entries:
                    raise ValueError(
                        f"SAEClampSpec.clamps[{hook_name!r}][{layer_idx}] "
                        "must be a non-empty sequence of SAEClampEntry, got "
                        f"{type(entries).__name__}."
                    )
                feature_seen: set[int] = set()
                for entry in entries:
                    if not isinstance(entry, SAEClampEntry):
                        raise ValueError(
                            f"SAEClampSpec.clamps[{hook_name!r}]"
                            f"[{layer_idx}] entries must all be "
                            f"SAEClampEntry, got "
                            f"{type(entry).__name__}."
                        )
                    if entry.feature_idx in feature_seen:
                        raise ValueError(
                            f"SAEClampSpec.clamps[{hook_name!r}]"
                            f"[{layer_idx}] contains duplicate "
                            f"feature_idx={entry.feature_idx}; each "
                            "feature may appear at most once per "
                            "(hook, layer)."
                        )
                    feature_seen.add(entry.feature_idx)
                coerced_layer[layer_idx] = tuple(entries)
            coerced[hook_name] = coerced_layer
        object.__setattr__(self, "clamps", coerced)


def coerce_sae_clamp_specs(
    raw: object,
) -> tuple[SAEClampSpec, ...] | None:
    """Coerce a JSON-shaped payload into ``tuple[SAEClampSpec, ...]``.

    Accepts ``None``, ``[]``, or a list of dicts of the form::

        {
            "module_name": "golden_gate",
            "phase": "both",  # optional, default "both"
            "clamps": {
                "post_mlp": {
                    20: [
                        {
                            "feature_idx": 34,
                            "kind": "absolute",
                            "value": 5.0,
                            "only_if_active": false,
                        }
                    ]
                }
            },
        }

    Returns ``None`` for empty input so downstream code can short-circuit
    on ``is None`` exactly like the additive ``steering_vectors`` field.
    Layer-key strings (a JSON-isms quirk) are converted to ``int``.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "sae_clamp_specs must be a list of clamp-spec objects, got "
            f"{type(raw).__name__}."
        )
    if not raw:
        return None
    out: list[SAEClampSpec] = []
    for i, item in enumerate(raw):
        if isinstance(item, SAEClampSpec):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"sae_clamp_specs[{i}] must be a dict or SAEClampSpec, got "
                f"{type(item).__name__}."
            )
        module_name_raw = item.get("module_name")
        if not isinstance(module_name_raw, str):
            raise ValueError(
                f"sae_clamp_specs[{i}]['module_name'] must be a str, got "
                f"{type(module_name_raw).__name__}."
            )
        module_name: str = module_name_raw
        phase = item.get("phase", "both")
        clamps_raw = item.get("clamps")
        if clamps_raw is None:
            raise ValueError(f"sae_clamp_specs[{i}] missing required 'clamps' field.")
        if not isinstance(clamps_raw, dict):
            raise ValueError(
                f"sae_clamp_specs[{i}]['clamps'] must be a dict, got "
                f"{type(clamps_raw).__name__}."
            )
        clamps: SAEClampHookMap = {}
        for hook_name, layer_map_raw in clamps_raw.items():
            if not isinstance(layer_map_raw, dict):
                raise ValueError(
                    f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}] must "
                    f"be a dict mapping layer indices to entry lists, got "
                    f"{type(layer_map_raw).__name__}."
                )
            layer_map: SAEClampLayerMap = {}
            for layer_key, entries_raw in layer_map_raw.items():
                if type(layer_key) is int:
                    layer_idx = layer_key
                elif isinstance(layer_key, str):
                    try:
                        layer_idx = int(layer_key)
                    except ValueError as exc:
                        raise ValueError(
                            f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}] "
                            f"has invalid layer key {layer_key!r}; expected "
                            "an integer."
                        ) from exc
                else:
                    raise ValueError(
                        f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}] "
                        f"has invalid layer key {layer_key!r}; expected "
                        "an integer."
                    )
                if not isinstance(entries_raw, (list, tuple)):
                    raise ValueError(
                        f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}]"
                        f"[{layer_idx}] must be a list of clamp entries, "
                        f"got {type(entries_raw).__name__}."
                    )
                validate_steering_index(
                    layer_idx,
                    f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}] layer key",
                )
                if layer_idx in layer_map:
                    raise ValueError(
                        f"sae_clamp_specs[{i}]['clamps'][{hook_name!r}] "
                        f"contains duplicate layer key {layer_idx!r} after "
                        "integer normalization."
                    )
                entries = tuple(_coerce_clamp_entry(e) for e in entries_raw)
                layer_map[layer_idx] = entries
            clamps[hook_name] = layer_map
        out.append(SAEClampSpec(module_name=module_name, phase=phase, clamps=clamps))
    validate_sae_clamp_specs_no_overlap(tuple(out))
    return tuple(out)


def validate_sae_clamp_specs_no_overlap(specs: tuple[SAEClampSpec, ...]) -> None:
    """Reject overlapping clamps for the same module/site/feature.

    Multiple specs may reference the same SAE module, for example to
    express different prefill and decode clamps.  What must not happen
    is two phase-overlapping specs writing the same
    ``(module, hook, layer, feature)`` cell: the clamp-table populator
    has a single row slot for that cell, so accepting both would make
    the later spec silently overwrite the earlier one.
    """
    phase_mask = {"prefill": 0b01, "decode": 0b10, "both": 0b11}
    seen: dict[tuple[str, str, int, int], tuple[int, str]] = {}
    for spec_idx, spec in enumerate(specs):
        mask = phase_mask[spec.phase]
        for hook_name, layer_map in spec.clamps.items():
            for layer_idx, entries in layer_map.items():
                for entry in entries:
                    key = (
                        spec.module_name,
                        hook_name,
                        layer_idx,
                        entry.feature_idx,
                    )
                    prev = seen.get(key)
                    if prev is not None and prev[0] & mask:
                        raise ValueError(
                            "sae_clamp_specs contains overlapping clamps for "
                            f"module={spec.module_name!r}, hook={hook_name!r}, "
                            f"layer={layer_idx}, feature_idx={entry.feature_idx}. "
                            f"Spec {spec_idx} phase {spec.phase!r} overlaps "
                            f"an earlier spec with phase {prev[1]!r}; combine "
                            "the entries into one spec or use disjoint phases."
                        )
                    merged_mask = prev[0] | mask if prev is not None else mask
                    seen[key] = (merged_mask, spec.phase)


def _coerce_clamp_entry(raw: object) -> SAEClampEntry:
    """Convert a dict / SAEClampEntry into a validated SAEClampEntry."""
    if isinstance(raw, SAEClampEntry):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"SAE clamp entry must be a dict or SAEClampEntry, got "
            f"{type(raw).__name__}."
        )
    missing = [key for key in ("feature_idx", "kind", "value") if key not in raw]
    if missing:
        raise ValueError(
            "SAE clamp entry is missing required field(s): "
            f"{', '.join(repr(key) for key in missing)}."
        )
    return SAEClampEntry(
        feature_idx=raw["feature_idx"],
        kind=raw["kind"],
        value=raw["value"],
        only_if_active=raw.get("only_if_active", False),
    )


def hash_sae_clamp_specs(specs: tuple[SAEClampSpec, ...] | None) -> int:
    """Deterministic SHA-256 hash of a tuple of SAE clamp specs.

    Returns 0 for ``None`` or empty input; otherwise returns a positive
    63-bit integer.  Within the digest:

    * Specs are sorted by ``module_name`` then ``phase``.
    * Hook entries are sorted by hook-point name.
    * Layer entries are sorted by layer index.
    * Clamp entries within a (hook, layer) are sorted by ``feature_idx``
      so callers may pass entries in arbitrary order without affecting
      the hash.

    The dedicated ``b"\\x00sae_clamps\\x00"`` domain separator at the
    front of the SAE block ensures the hash cannot collide with any
    additive-vector hash regardless of name overlap.

    Designed to compose with :func:`hash_steering_config` so that a
    single combined config hash uniquely identifies the (additive +
    SAE) state for a request, used for prefix-cache keying.
    """
    if not specs:
        return 0
    return _hash_sae_clamp_specs_with_phase(specs)


def _hash_sae_clamp_specs_with_phase(
    specs: tuple[SAEClampSpec, ...],
    *,
    phase_override: Literal["prefill", "decode"] | None = None,
) -> int:
    """Hash *specs*, optionally replacing each spec phase in the digest."""
    h = hashlib.sha256()
    h.update(b"\x00sae_clamps\x00")
    sort_key = (
        (lambda s: _sae_spec_sort_key(s, phase_override=phase_override))
        if phase_override is not None
        else _sae_spec_sort_key
    )
    for spec in sorted(specs, key=sort_key):
        h.update(b"\x01module\x01")
        h.update(spec.module_name.encode("utf-8"))
        h.update((phase_override or spec.phase).encode("utf-8"))
        for hook_name in sorted(spec.clamps.keys()):
            h.update(b"\x02hook\x02")
            h.update(hook_name.encode("utf-8"))
            layer_map = spec.clamps[hook_name]
            for layer_idx in sorted(layer_map.keys()):
                h.update(b"\x03layer\x03")
                h.update(
                    validate_steering_index(layer_idx, "SAEClampSpec layer_idx")
                    .to_bytes(4, "little", signed=True)
                )
                entries = sorted(layer_map[layer_idx], key=lambda e: e.feature_idx)
                for entry in entries:
                    h.update(b"\x04entry\x04")
                    h.update(
                        validate_steering_index(
                            entry.feature_idx, "SAEClampEntry.feature_idx"
                        ).to_bytes(4, "little", signed=True)
                    )
                    # 1 byte discriminator for the kind so additive vs
                    # absolute can never collide on the same numerical
                    # value.
                    h.update(b"a" if entry.kind == "absolute" else b"r")
                    # Promote to fp64 for hash-stable bit pattern.
                    h.update(struct.pack("<d", float(entry.value)))
                    h.update(b"\x01" if entry.only_if_active else b"\x00")
    return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF


def hash_sae_clamp_specs_for_phase(
    specs: tuple[SAEClampSpec, ...] | None,
    phase: Literal["prefill", "decode"],
) -> int:
    """Hash SAE clamp table content for one worker phase.

    Request/APC hashes use :func:`hash_sae_clamp_specs` and include each
    spec's declared phase, because ``phase="both"`` and ``phase="prefill"``
    differ in decode semantics.  Worker SAE table rows, however, are already
    keyed by the active worker phase.  Within a prefill row, a ``both`` spec
    and an otherwise-identical ``prefill`` spec produce identical clamp-table
    content, so this helper normalizes applicable specs to *phase* before
    hashing.  That lets the scheduler and worker share rows by actual table
    content without weakening prefix-cache isolation.
    """
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}.")
    if not specs:
        return 0
    phase_specs = tuple(spec for spec in specs if spec.phase in ("both", phase))
    if not phase_specs:
        return 0
    return _hash_sae_clamp_specs_with_phase(phase_specs, phase_override=phase)


def _sae_spec_sort_key(
    spec: SAEClampSpec,
    *,
    phase_override: Literal["prefill", "decode"] | None = None,
) -> tuple[str, str, tuple[Any, ...]]:
    """Canonical ordering key for SAE specs.

    Requests may carry multiple non-overlapping specs for the same
    module and phase.  Sorting only by ``(module_name, phase)`` would
    leave those equal-key specs in caller order and make the digest
    depend on request-list ordering.  Include the canonical clamp
    content in the sort key so semantically identical sets hash the
    same regardless of outer tuple order.
    """
    hook_items = []
    for hook_name in sorted(spec.clamps):
        layer_items = []
        for layer_idx in sorted(spec.clamps[hook_name]):
            entries = tuple(
                sorted(
                    (
                        (
                            entry.feature_idx,
                            entry.kind,
                            float(entry.value),
                            entry.only_if_active,
                        )
                        for entry in spec.clamps[hook_name][layer_idx]
                    ),
                    key=lambda item: item[0],
                )
            )
            layer_items.append((layer_idx, entries))
        hook_items.append((hook_name, tuple(layer_items)))
    return (spec.module_name, phase_override or spec.phase, tuple(hook_items))
