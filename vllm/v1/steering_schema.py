# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative per-request steering gate schema.

A request can declare its own conditional steering in the request payload
(no server-registered consumer). A gate is ``when × scope × apply``:

- ``when`` (trigger): ``always`` | ``probe``
  (``sigmoid(sharpness·(residual@probe − threshold))``).
- ``scope`` (extent): ``this_token`` | ``next_step`` | ``rest_of_request`` |
  ``rest_of_conversation``.
- ``apply`` (action): ``add`` (vector × strength, composed *on top* of the
  request's static decode steering) | ``attenuate`` (damp existing steering).

The wire schema (:class:`SteeringGate` and friends) are msgspec tagged
unions so the nested list rides on :class:`vllm.v1.request_metadata.
RequestMetadata` through the ``EngineCoreRequest`` msgpack channel, exactly
like ``conversation_id``. A vector source is either a server-registered
``{"kind":"name","name":...}`` (resolved to packed bytes at the frontend,
see :func:`build_steering_gates`) or an inline
``{"kind":"inline","packed":{hook: SteeringHookPacked}}`` escape hatch.

Worker-side, the packed bytes are unpacked to numpy once at admission
(:func:`resolve_gates` → :class:`ResolvedGate`) and surfaced on
:class:`vllm.v1.capture.step_view.StepRequestView.steering`; the built-in
declarative consumer reads those and drives the steering substrate.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import numpy as np

from vllm.config.steering_types import SteeringHookPacked, unpack_steering_vectors
from vllm.logger import init_logger

logger = init_logger(__name__)

# Rate-limit the admission-side warning: a broken producer would otherwise
# emit one line per request. Latched on first graceful skip (see
# ``resolve_gates_safe``); mirrors ``_missing_site_warned`` in the
# declarative consumer.
_resolve_failure_warned = False

if TYPE_CHECKING:
    from typing import Protocol

    class _VectorRegistry(Protocol):
        """Minimal frontend registry surface used for name resolution."""

        def get_packed(
            self, name: str, kind: str
        ) -> dict[str, SteeringHookPacked] | None: ...


# --------------------------------------------------------------------------
# Wire schema (msgspec tagged unions). Discriminator field is ``kind``.
# --------------------------------------------------------------------------


class NamedVec(
    msgspec.Struct, tag="name", tag_field="kind", frozen=True, omit_defaults=True
):
    """Reference to a server-registered probe/steer vector by name.

    Resolved to :class:`InlineVec` at the frontend
    (:func:`build_steering_gates`); the worker never sees a name.
    """

    name: str


class InlineVec(
    msgspec.Struct, tag="inline", tag_field="kind", frozen=True, omit_defaults=True
):
    """Inline packed vector(s): a ``{hook: SteeringHookPacked}`` spec."""

    packed: dict[str, SteeringHookPacked]


VecSource = NamedVec | InlineVec


class AlwaysWhen(
    msgspec.Struct, tag="always", tag_field="kind", frozen=True, omit_defaults=True
):
    """Trigger unconditionally."""


class ProbeWhen(
    msgspec.Struct, tag="probe", tag_field="kind", frozen=True, omit_defaults=True
):
    """Trigger on ``sigmoid(sharpness·(residual@probe − threshold))``.

    ``probe`` must resolve to exactly one ``(hook, layer)`` vector; that
    site is where the residual is read.
    """

    probe: VecSource
    threshold: float
    sharpness: float = 1.0


When = AlwaysWhen | ProbeWhen


class GateScope(str, enum.Enum):
    THIS_TOKEN = "this_token"
    NEXT_STEP = "next_step"
    REST_OF_REQUEST = "rest_of_request"
    REST_OF_CONVERSATION = "rest_of_conversation"


class AddApply(
    msgspec.Struct, tag="add", tag_field="kind", frozen=True, omit_defaults=True
):
    """Add ``steer × strength`` on top of the request's static steering."""

    steer: VecSource
    strength: float = 1.0


class AttenuateApply(
    msgspec.Struct, tag="attenuate", tag_field="kind", frozen=True, omit_defaults=True
):
    """Damp existing steering by ``strength`` (a multiplicative factor)."""

    strength: float


Apply = AddApply | AttenuateApply


class SteeringGate(msgspec.Struct, frozen=True, omit_defaults=True):
    """One declarative gate: ``when × scope × apply``."""

    when: When
    scope: GateScope
    apply: Apply


# --------------------------------------------------------------------------
# Host-side resolved form (packed bytes unpacked to numpy once at admission).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedGate:
    """A gate with its vector sources unpacked to numpy, ready for the
    consumer. Built once per request at admission by :func:`resolve_gates`;
    never re-unpacked in the per-step hot loop."""

    scope: str  # GateScope value
    # when
    when_kind: str  # "always" | "probe"
    probe_site: tuple[int, str] | None  # (layer, hook) for probe
    probe_vec: np.ndarray | None  # 1-D probe vector
    threshold: float | None
    sharpness: float | None
    # apply
    apply_kind: str  # "add" | "attenuate"
    steer_vectors: dict[str, dict[int, np.ndarray]] | None  # add override (×strength)
    strength: float


def _single_site(
    unpacked: dict[str, dict[int, np.ndarray]] | None,
) -> tuple[tuple[int, str], np.ndarray]:
    """Require *unpacked* to hold exactly one ``(hook, layer)`` vector and
    return ``((layer, hook), vector)``. Used for probe sources."""
    if not unpacked:
        raise ValueError("probe vector source resolved to no vectors")
    sites = [
        (layer, hook)
        for hook, layers in unpacked.items()
        for layer in layers
    ]
    if len(sites) != 1:
        raise ValueError(
            f"probe vector source must name exactly one (hook, layer); "
            f"got {sorted(sites)}"
        )
    layer, hook = sites[0]
    return (layer, hook), unpacked[hook][layer]


def _scale_vectors(
    unpacked: dict[str, dict[int, np.ndarray]] | None, strength: float
) -> dict[str, dict[int, np.ndarray]] | None:
    """Multiply every row of *unpacked* by *strength* (dtype-preserving)."""
    if not unpacked:
        return None
    if strength == 1.0:
        return unpacked
    return {
        hook: {
            layer: (arr if strength == 1.0 else arr * arr.dtype.type(strength))
            for layer, arr in layers.items()
        }
        for hook, layers in unpacked.items()
    }


def resolve_gates(
    gates: list[SteeringGate] | None,
) -> list[ResolvedGate] | None:
    """Unpack every gate's inline-packed vector sources to numpy, once.

    Assumes all ``VecSource`` are :class:`InlineVec` (names were resolved to
    packed at the frontend by :func:`build_steering_gates`). Pure host-side:
    no GPU work / D2H, so it runs identically on the v1 and v2 runners.
    """
    if not gates:
        return None
    resolved: list[ResolvedGate] = []
    for gate in gates:
        # when
        if isinstance(gate.when, ProbeWhen):
            src = gate.when.probe
            if not isinstance(src, InlineVec):
                raise ValueError("unresolved named probe reached the worker")
            site, vec = _single_site(unpack_steering_vectors(src.packed))
            when_kind = "probe"
            probe_site: tuple[int, str] | None = site
            probe_vec: np.ndarray | None = np.ascontiguousarray(vec)
            threshold: float | None = float(gate.when.threshold)
            sharpness: float | None = float(gate.when.sharpness)
        else:
            when_kind = "always"
            probe_site = probe_vec = threshold = sharpness = None
        # apply
        if isinstance(gate.apply, AddApply):
            src = gate.apply.steer
            if not isinstance(src, InlineVec):
                raise ValueError("unresolved named steer reached the worker")
            steer = _scale_vectors(
                unpack_steering_vectors(src.packed), float(gate.apply.strength)
            )
            apply_kind = "add"
            strength = float(gate.apply.strength)
        else:
            apply_kind = "attenuate"
            steer = None
            strength = float(gate.apply.strength)
        resolved.append(
            ResolvedGate(
                scope=gate.scope.value,
                when_kind=when_kind,
                probe_site=probe_site,
                probe_vec=probe_vec,
                threshold=threshold,
                sharpness=sharpness,
                apply_kind=apply_kind,
                steer_vectors=steer,
                strength=strength,
            )
        )
    return resolved


def resolve_gates_safe(
    gates: list[SteeringGate] | None,
    req_id: str | None = None,
) -> list[ResolvedGate] | None:
    """Fail-safe :func:`resolve_gates` for the model-runner admission path.

    The frontend dry-runs :func:`build_steering_gates` so a malformed spec is
    rejected as HTTP 400, but other producers of ``RequestMetadata.steering``
    (offline ``LLM``, the Rust frontend, msgpack version skew) bypass that
    check. A raw ``resolve_gates`` there would raise ``ValueError`` /
    ``TypeError`` / ``KeyError`` straight through ``_update_states`` and abort
    the engine core — a full-server DoS from one bad request. Here we instead
    log once and drop the request's declarative gates, letting generation
    proceed without them.

    Determinism: the gates arrive as identical serialized bytes on every TP
    rank, so a malformed spec fails identically on all ranks; the graceful
    path returns ``None`` everywhere and cannot desync them.

    Args:
        gates: Per-request gates to resolve, or ``None``.
        req_id: Request id, included in the warning for triage.

    Returns:
        Resolved gates, or ``None`` if the input was empty or malformed.
    """
    if not gates:
        return None
    try:
        return resolve_gates(gates)
    except Exception:  # noqa: BLE001 - fail-safe: never abort the engine core
        global _resolve_failure_warned
        if not _resolve_failure_warned:
            _resolve_failure_warned = True
            logger.warning(
                "declarative steering: failed to resolve gates for request %s; "
                "gates skipped (request proceeds without declarative steering). "
                "This suppresses further identical warnings.",
                req_id,
                exc_info=True,
            )
        return None


# --------------------------------------------------------------------------
# Frontend: validate raw JSON gates + resolve names → inline packed.
# --------------------------------------------------------------------------


def _validate_hooks(packed: dict[str, SteeringHookPacked]) -> None:
    from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

    for hook in packed:
        if hook not in VALID_HOOK_POINT_NAMES:
            raise ValueError(
                f"unknown steering hook {hook!r}; valid: "
                f"{sorted(VALID_HOOK_POINT_NAMES)}"
            )


def _resolve_source_dict(
    src: dict, kind: str, registry: _VectorRegistry | None
) -> dict:
    """Resolve one raw vec-source dict to an inline packed dict.

    ``{"kind":"name","name":X}`` → registry lookup → inline packed.
    ``{"kind":"inline","packed":...}`` passes through (hooks validated).
    """
    if not isinstance(src, dict):
        raise ValueError(f"vector source must be an object, got {type(src).__name__}")
    skind = src.get("kind")
    if skind == "name":
        name = src.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("named vector source requires a non-empty 'name'")
        if registry is None:
            raise ValueError(
                f"named {kind} vector {name!r} requested but no steering vector "
                f"registry is configured (start server with --enable-steering)"
            )
        packed = registry.get_packed(name, kind)
        if packed is None:
            raise ValueError(f"unknown {kind} vector name {name!r}")
        _validate_hooks(packed)
        return {"kind": "inline", "packed": packed}
    if skind == "inline":
        packed = src.get("packed")
        if not isinstance(packed, dict) or not packed:
            raise ValueError("inline vector source requires a non-empty 'packed'")
        _validate_hooks(packed)
        return {"kind": "inline", "packed": packed}
    raise ValueError(f"vector source 'kind' must be 'name' or 'inline', got {skind!r}")


def _validate_gate_semantics(gates: list[SteeringGate]) -> None:
    """Reject gates the steering substrate cannot honor.

    Structural validity (``resolve_gates``) is necessary but not
    sufficient; these are the semantic constraints a well-formed request
    must also satisfy so it never fails mid-flight on the worker:

    - ``attenuate`` with ``when=probe`` and ``scope=this_token`` is
      unsupported. Same-token conditional damping would need a per-row
      gate of the form ``scale = 1 - (1 - strength)*gate``, but the
      in-graph per-row monitor can only multiply a row's contribution
      *toward zero* when the probe is LOW — the wrong shape for "damp when
      the probe fires this token". Use ``next_step`` or
      ``rest_of_request`` (host-evaluated) instead.
    - a probe trigger's ``threshold`` and ``sharpness`` must be finite and
      ``sharpness`` non-negative, matching the worker-side monitor
      validation (:func:`validate_steering_monitor`) so a request that
      passes the frontend can never have its paired row monitor rejected
      on the step thread (which would strand the override applying every
      token unconditionally).
    """
    for gate in gates:
        if (
            isinstance(gate.apply, AttenuateApply)
            and isinstance(gate.when, ProbeWhen)
            and gate.scope == GateScope.THIS_TOKEN
        ):
            raise ValueError(
                "attenuate with when=probe and scope=this_token is not "
                "supported: the in-graph per-row monitor cannot damp a row "
                "only when the probe fires. Use scope=next_step or "
                "rest_of_request (host-evaluated) instead."
            )
        if isinstance(gate.when, ProbeWhen):
            if not math.isfinite(gate.when.threshold):
                raise ValueError("probe threshold must be finite")
            if not math.isfinite(gate.when.sharpness):
                raise ValueError("probe sharpness must be finite")
            if gate.when.sharpness < 0:
                raise ValueError(
                    f"probe sharpness must be non-negative, got "
                    f"{gate.when.sharpness!r}"
                )


def build_steering_gates(
    raw: list[dict] | None,
    registry: _VectorRegistry | None,
) -> list[SteeringGate] | None:
    """Validate raw JSON gates and resolve named vectors to inline packed.

    Called at the frontend (``to_request_metadata``). Every ``{name}`` vector
    source is replaced by its registered packed bytes so the worker only ever
    sees :class:`InlineVec`. Raises ``ValueError`` on malformed gates,
    unknown names, or invalid hooks — callers surface this as HTTP 400.
    """
    if not raw:
        return None
    if not isinstance(raw, list):
        raise ValueError("'steering' must be a list of gate objects")
    prepared: list[dict] = []
    for gate in raw:
        if not isinstance(gate, dict):
            raise ValueError("each steering gate must be an object")
        g = dict(gate)
        when = g.get("when")
        if isinstance(when, dict) and when.get("kind") == "probe":
            w = dict(when)
            w["probe"] = _resolve_source_dict(w.get("probe", {}), "probe", registry)
            g["when"] = w
        apply = g.get("apply")
        if isinstance(apply, dict) and apply.get("kind") == "add":
            a = dict(apply)
            a["steer"] = _resolve_source_dict(a.get("steer", {}), "steer", registry)
            g["apply"] = a
        prepared.append(g)
    try:
        gates = msgspec.convert(prepared, type=list[SteeringGate])
    except msgspec.ValidationError as exc:
        raise ValueError(f"invalid steering gate spec: {exc}") from exc
    # Fail fast on semantic problems the substrate cannot honor
    # (unsupported combos, non-finite/negative probe params).
    _validate_gate_semantics(gates)
    # Fail fast on structural problems the consumer would otherwise hit
    # (probe must name exactly one site; add must carry vectors). Normalize
    # non-ValueError unpack failures (e.g. ``np.dtype("garbage")`` raises
    # TypeError, missing keys raise KeyError) into ValueError so callers
    # return HTTP 400 rather than 500.
    try:
        resolve_gates(gates)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"invalid steering gate spec: {exc}") from exc
    return gates


__all__ = [
    "NamedVec",
    "InlineVec",
    "VecSource",
    "AlwaysWhen",
    "ProbeWhen",
    "When",
    "GateScope",
    "AddApply",
    "AttenuateApply",
    "Apply",
    "SteeringGate",
    "ResolvedGate",
    "resolve_gates",
    "resolve_gates_safe",
    "build_steering_gates",
]
