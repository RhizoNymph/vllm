# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types and pure helpers for per-request MLP neuron clamps.

A neuron clamp overrides one coordinate of the SiLU-gated MLP intermediate
activation (the tensor fed into ``down_proj``) at one decoder layer::

    act = silu(gate) * up  # (tokens, intermediate)
    set: act[:, neuron] = value
    add: act[:, neuron] = act[:, neuron] + value
    scale: act[:, neuron] = act[:, neuron] * value

Zero ablation is ``set 0``; mean ablation is ``set <mean>``.

Clamps are part of a request's *phase-effective steering config*: the
clamps active in a phase (``both`` + that phase) are folded into
:func:`vllm.config.steering_types.hash_steering_config` for that phase, so a
request with steering vectors + clamps (or clamps alone) owns exactly one
``(hash, phase)`` steering-table row. Scheduler admission, row allocation,
prefix-cache keying and phase transitions are reused unchanged.

Everything in this module is a pure function of its inputs (no torch, no
engine state) so it is identical on every TP/PP rank — the steering
determinism contract.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from typing import Any, Literal, get_args

from vllm.exceptions import VLLMValidationError

NeuronClampMode = Literal["set", "add", "scale"]
NeuronClampPhase = Literal["both", "prefill", "decode"]
SteeringPhase = Literal["prefill", "decode"]

VALID_NEURON_CLAMP_MODES: frozenset[str] = frozenset(get_args(NeuronClampMode))
VALID_NEURON_CLAMP_PHASES: frozenset[str] = frozenset(get_args(NeuronClampPhase))

# Stable integer codes for hashing (never reorder: they feed APC keys).
_MODE_CODE: dict[str, int] = {"set": 0, "add": 1, "scale": 2}

# Architectures whose MLP forward routes through the neuron-clamp op.
# Requests carrying clamps for any other architecture are rejected at the
# input processor instead of being silently ignored.
# Native ``qwen3.py`` decoders only; ``Qwen2ForCausalLM`` now runs on the
# Transformers backend, whose MLPs carry no clamp op.
NEURON_CLAMP_SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset({"Qwen3ForCausalLM"})

_PARAM = "neuron_clamps"


class NeuronClampValidationError(VLLMValidationError, ValueError):
    """A ``neuron_clamps`` entry is malformed or out of range (HTTP 400).

    Also a ``ValueError`` so pydantic validators surface it as a field error.
    """

    def __init__(self, message: str, *, value: Any = None) -> None:
        super().__init__(message, parameter=_PARAM, value=value)


class NeuronClamp:
    """One validated clamp. Immutable; construct via :func:`make_neuron_clamp`
    or :func:`coerce_neuron_clamps`.

    Plain ``__slots__`` class (not a pydantic/msgspec type) so it can never
    hold an invalid state: ``__init__`` validates every field. It
    round-trips the engine IPC boundary through ``SamplingParams`` as a
    plain dict (see :meth:`to_dict`), because ``SamplingParams`` stores the
    wire form.
    """

    __slots__ = ("layer", "neuron", "mode", "value", "phase")

    layer: int
    neuron: int
    mode: NeuronClampMode
    value: float
    phase: NeuronClampPhase

    def __init__(
        self,
        layer: int,
        neuron: int,
        mode: NeuronClampMode,
        value: float,
        phase: NeuronClampPhase = "both",
    ) -> None:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise NeuronClampValidationError(
                f"neuron clamp 'layer' must be a non-negative int, got {layer!r}",
                value=layer,
            )
        if isinstance(neuron, bool) or not isinstance(neuron, int) or neuron < 0:
            raise NeuronClampValidationError(
                f"neuron clamp 'neuron' must be a non-negative int, got {neuron!r}",
                value=neuron,
            )
        if mode not in VALID_NEURON_CLAMP_MODES:
            raise NeuronClampValidationError(
                f"neuron clamp 'mode' must be one of "
                f"{sorted(VALID_NEURON_CLAMP_MODES)}, got {mode!r}",
                value=mode,
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise NeuronClampValidationError(
                f"neuron clamp 'value' must be a finite number, got {value!r}",
                value=value,
            )
        if phase not in VALID_NEURON_CLAMP_PHASES:
            raise NeuronClampValidationError(
                f"neuron clamp 'phase' must be one of "
                f"{sorted(VALID_NEURON_CLAMP_PHASES)}, got {phase!r}",
                value=phase,
            )
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "neuron", neuron)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "value", float(value))
        object.__setattr__(self, "phase", phase)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("NeuronClamp is immutable")

    def _key(self) -> tuple:
        return (self.layer, self.neuron, self.mode, self.value, self.phase)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeuronClamp) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"NeuronClamp(layer={self.layer}, neuron={self.neuron}, "
            f"mode={self.mode!r}, value={self.value!r}, phase={self.phase!r})"
        )

    def __reduce__(self):
        return (NeuronClamp, self._key())

    def applies_to(self, phase: SteeringPhase) -> bool:
        return self.phase == "both" or self.phase == phase

    def affine(self) -> tuple[float, float]:
        """Return ``(mul, add)`` such that the clamp is ``x*mul + add``.

        ``set`` is ``(0, value)``; kernels special-case ``mul == 0`` to
        write ``add`` exactly (so a non-finite activation cannot leak
        through ``0 * x``).
        """
        if self.mode == "set":
            return 0.0, self.value
        if self.mode == "add":
            return 1.0, self.value
        return self.value, 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "neuron": self.neuron,
            "mode": self.mode,
            "value": self.value,
            "phase": self.phase,
        }


_ALLOWED_KEYS = frozenset({"layer", "neuron", "mode", "value", "phase"})
_REQUIRED_KEYS = frozenset({"layer", "neuron", "mode", "value"})


def make_neuron_clamp(raw: NeuronClamp | dict[str, Any]) -> NeuronClamp:
    """Build a :class:`NeuronClamp` from a dict (wire / JSON form)."""
    if isinstance(raw, NeuronClamp):
        return raw
    if not isinstance(raw, dict):
        raise NeuronClampValidationError(
            f"each neuron clamp must be an object, got {type(raw).__name__}",
            value=raw,
        )
    keys = set(raw.keys())
    extra = keys - _ALLOWED_KEYS
    if extra:
        raise NeuronClampValidationError(
            f"neuron clamp has unexpected keys {sorted(extra)}; "
            f"allowed: {sorted(_ALLOWED_KEYS)}",
            value=raw,
        )
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise NeuronClampValidationError(
            f"neuron clamp missing required keys {sorted(missing)}", value=raw
        )
    return NeuronClamp(
        layer=raw["layer"],
        neuron=raw["neuron"],
        mode=raw["mode"],
        value=raw["value"],
        phase=raw.get("phase", "both"),
    )


def _phases_overlap(a: NeuronClampPhase, b: NeuronClampPhase) -> bool:
    return a == "both" or b == "both" or a == b


def coerce_neuron_clamps(
    raw: Sequence[NeuronClamp | dict[str, Any]] | None,
) -> tuple[NeuronClamp, ...] | None:
    """Validate a raw clamp list structurally; ``None``/empty -> ``None``.

    Rejects (with :class:`NeuronClampValidationError`): non-list input,
    malformed entries, and two clamps on the same ``(layer, neuron)`` whose
    phases overlap (``both`` overlaps everything; ``prefill`` and
    ``decode`` do not overlap each other). Model-dependent range checks
    live in :func:`validate_neuron_clamps_for_model`.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise NeuronClampValidationError(
            f"neuron_clamps must be a list, got {type(raw).__name__}", value=raw
        )
    clamps = tuple(make_neuron_clamp(entry) for entry in raw)
    seen: dict[tuple[int, int], list[NeuronClampPhase]] = {}
    for c in clamps:
        phases = seen.setdefault((c.layer, c.neuron), [])
        for p in phases:
            if _phases_overlap(p, c.phase):
                raise NeuronClampValidationError(
                    f"duplicate neuron clamp for layer {c.layer} neuron "
                    f"{c.neuron}: phases {p!r} and {c.phase!r} overlap",
                    value=c.to_dict(),
                )
        phases.append(c.phase)
    return clamps or None


def effective_clamps_for_phase(
    clamps: Iterable[NeuronClamp] | None,
    phase: SteeringPhase,
) -> tuple[NeuronClamp, ...] | None:
    """Clamps active in *phase*, canonically sorted by ``(layer, neuron)``.

    The result has at most one clamp per ``(layer, neuron)`` (guaranteed by
    :func:`coerce_neuron_clamps`). ``None`` when nothing applies, so a
    phase without clamps hashes exactly as before clamps existed.
    """
    if not clamps:
        return None
    active = sorted(
        (c for c in clamps if c.applies_to(phase)),
        key=lambda c: (c.layer, c.neuron),
    )
    return tuple(active) or None


def encode_clamps_for_hash(clamps: Sequence[NeuronClamp]) -> bytes:
    """Deterministic byte encoding of phase-effective clamps.

    Phase is deliberately excluded: the caller hashes one phase's
    effective set, so ``both`` and the matching single-phase clamp with
    identical numbers produce the same phase identity (they are the same
    computation). Order-independent because input is canonically sorted.
    """
    out = bytearray()
    for c in sorted(clamps, key=lambda c: (c.layer, c.neuron)):
        out += struct.pack("<iiBd", c.layer, c.neuron, _MODE_CODE[c.mode], c.value)
    return bytes(out)


def validate_neuron_clamps_for_model(
    clamps: Sequence[NeuronClamp],
    *,
    num_hidden_layers: int,
    intermediate_size: int,
    max_clamps_per_layer: int,
) -> None:
    """Range-check clamps against the served model and engine limits.

    ``max_clamps_per_layer`` bounds the clamps one request may place on a
    single layer *per phase* (the per-layer clamp table width).
    """
    if max_clamps_per_layer <= 0:
        raise NeuronClampValidationError(
            "neuron clamps are disabled on this server "
            "(--max-neuron-clamps-per-layer=0)"
        )
    for c in clamps:
        if c.layer >= num_hidden_layers:
            raise NeuronClampValidationError(
                f"neuron clamp layer {c.layer} out of range [0, {num_hidden_layers})",
                value=c.to_dict(),
            )
        if c.neuron >= intermediate_size:
            raise NeuronClampValidationError(
                f"neuron clamp neuron {c.neuron} out of range "
                f"[0, {intermediate_size}) for layer {c.layer}",
                value=c.to_dict(),
            )
    for phase in ("prefill", "decode"):
        per_layer: dict[int, int] = {}
        for c in effective_clamps_for_phase(clamps, phase) or ():
            per_layer[c.layer] = per_layer.get(c.layer, 0) + 1
        for layer, count in per_layer.items():
            if count > max_clamps_per_layer:
                raise NeuronClampValidationError(
                    f"{count} neuron clamps on layer {layer} in the {phase} "
                    f"phase exceed --max-neuron-clamps-per-layer="
                    f"{max_clamps_per_layer}"
                )


def global_to_local_neuron(neuron: int, *, tp_rank: int, local_size: int) -> int | None:
    """Map a global intermediate index to this TP rank's local index.

    ``down_proj`` is row-parallel, so rank ``r`` owns the contiguous slice
    ``[r * local_size, (r + 1) * local_size)`` of the intermediate dim.
    Returns ``None`` when another rank owns the neuron (that rank clamps
    it; this one leaves the slot empty).
    """
    start = tp_rank * local_size
    if start <= neuron < start + local_size:
        return neuron - start
    return None
