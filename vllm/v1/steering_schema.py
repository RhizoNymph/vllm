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
  request's static decode steering) | ``attenuate`` (damp existing steering) |
  ``clamp`` (FORWARD-COMPAT, currently rejected at validation — see
  :class:`ClampApply`; clamp gating is server-side via the global
  cross-layer monitor today).

The wire schema (:class:`SteeringGate` and friends) are msgspec tagged
unions so the nested list rides on :class:`vllm.v1.request_metadata.
RequestMetadata` through the ``EngineCoreRequest`` msgpack channel, exactly
like ``conversation_id``. A vector source is either a server-registered
``{"kind":"name","name":...}`` (a reference the frontend validates for
existence but leaves un-inflated — the name rides the wire and is resolved
against the *worker* registry at admission) or an inline
``{"kind":"inline","packed":{hook: SteeringHookPacked}}`` escape hatch.

Worker-side, every source is resolved to numpy once at admission
(:func:`resolve_gates` → :class:`ResolvedGate`) — ``NamedVec`` via the
worker-resident registry
(:mod:`vllm.v1.worker.steering_vector_registry`), ``InlineVec`` by
unpacking its bytes — and surfaced on
:class:`vllm.v1.capture.step_view.StepRequestView.steering`; the built-in
declarative consumer reads those and drives the steering substrate. A
``rest_of_conversation`` ``add`` gate must use a ``NamedVec`` so its
cross-turn latch persists a reference (name + content digest), not the
client's bytes (see :func:`build_steering_gates` and
docs/design/dynamic_steering.md §8.2/§8.3).
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
        """Registry surface used to resolve ``NamedVec`` sources.

        Implemented by both the frontend
        (:class:`~vllm.entrypoints.openai.steering.vector_registry.\
SteeringVectorRegistry`, unpacking on demand) and the worker
        (:class:`~vllm.v1.worker.steering_vector_registry.\
WorkerSteeringVectorRegistry`, precomputed at registration), so the same
        :func:`resolve_gates` code path resolves names on either side.
        """

        def resolve_vectors(
            self, name: str, kind: str
        ) -> tuple[dict[str, dict[int, np.ndarray]], str] | None: ...


# --------------------------------------------------------------------------
# Wire schema (msgspec tagged unions). Discriminator field is ``kind``.
# --------------------------------------------------------------------------


class NamedVec(
    msgspec.Struct, tag="name", tag_field="kind", frozen=True, omit_defaults=True
):
    """Reference to a server-registered probe/steer vector by name.

    The frontend validates only that the name exists (fast HTTP 400 on a
    typo) and passes it through un-inflated; the worker resolves it against
    its own replicated registry at admission (:func:`resolve_gates`). This
    keeps the wire payload small and lets a ``rest_of_conversation`` latch
    persist a reference rather than the client's bytes.
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


class ClampApply(
    msgspec.Struct, tag="clamp", tag_field="kind", frozen=True, omit_defaults=True
):
    """FORWARD-COMPAT: modulate the request's directional CLAMPS by the gate.

    Reserved wire-schema member — **currently rejected at validation**
    (:func:`_validate_gate_semantics` / :func:`validate_clamp_gate_support`)
    because no substrate can honor a *per-request* clamp gate today. The
    intended semantics: when the gate fires, the clamp's effective strength
    is scaled via the shared per-token row gate (``effective =
    clamp_strength * gate``), row-level — all K clamp entries of the
    request's row scaled uniformly, congruent with how the row gate treats
    additive steering. No vector source: the clamps themselves are declared
    statically in ``steering_clamps`` (the gate only modulates them).

    Why rejected: the clamp ops read the shared ``steering_row_gate``
    buffer, which is only *written* by the GLOBAL cross-layer monitor
    (``enable_cross_layer_monitor``). The per-request (per-row) monitor is
    fused into the steering kernel — non-mutating, same-hook — so a
    per-request probe can never materialize gate values the clamp op sees.
    Until a materializing per-row monitor exists, clamp gating is available
    only server-side: enable the cross-layer monitor and install a global
    monitor with ``gate_rows`` — every clamp at layers >= the probe layer is
    then modulated. The member is kept so the tagged union stays stable when
    per-request support lands.
    """

    strength: float = 1.0


Apply = AddApply | AttenuateApply | ClampApply


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
    apply_kind: str  # "add" | "attenuate" | "clamp"
    steer_vectors: dict[str, dict[int, np.ndarray]] | None  # add override (×strength)
    strength: float
    # Source provenance for an ``add`` gate whose steer vector is a
    # ``NamedVec`` (``None`` for inline). Carried so a ``rest_of_conversation``
    # latch can persist a *reference* (name + content digest) and re-resolve
    # it at bridge time, disengaging on digest mismatch. See
    # :class:`vllm.v1.capture.controller.ByRefLatch`.
    steer_name: str | None = None
    steer_digest: str | None = None


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


def _resolve_source(
    src: VecSource,
    kind: str,
    registry: _VectorRegistry | None,
) -> tuple[dict[str, dict[int, np.ndarray]] | None, str | None, str | None]:
    """Resolve one vector source to ``(vectors, name, digest)``.

    ``InlineVec`` unpacks its bytes (``name``/``digest`` ``None``).
    ``NamedVec`` looks up *registry* (``kind`` selects the namespace),
    returning the registered vectors and its content digest. A named source
    with no registry, or an unknown name, raises ``ValueError`` — callers on
    the admission path route that through :func:`resolve_gates_safe`.
    """
    if isinstance(src, InlineVec):
        return unpack_steering_vectors(src.packed), None, None
    # NamedVec
    if registry is None:
        raise ValueError(
            f"unresolved named {kind} vector {src.name!r} reached name "
            "resolution but no vector registry is available"
        )
    resolved = registry.resolve_vectors(src.name, kind)
    if resolved is None:
        raise ValueError(f"unknown {kind} vector name {src.name!r}")
    vectors, digest = resolved
    return vectors, src.name, digest


def resolve_gates(
    gates: list[SteeringGate] | None,
    registry: _VectorRegistry | None = None,
) -> list[ResolvedGate] | None:
    """Resolve every gate's vector sources to numpy, once.

    ``InlineVec`` sources are unpacked from their bytes; ``NamedVec`` sources
    are resolved against *registry* (the frontend registry during the
    :func:`build_steering_gates` dry-run, the worker registry at admission).
    An ``add`` gate additionally carries the resolved steer source's name and
    content digest on the :class:`ResolvedGate` so a ``rest_of_conversation``
    latch can persist a reference. Pure host-side: no GPU work / D2H, so it
    runs identically on the v1 and v2 runners.
    """
    if not gates:
        return None
    resolved: list[ResolvedGate] = []
    for gate in gates:
        # when
        if isinstance(gate.when, ProbeWhen):
            vecs, _, _ = _resolve_source(gate.when.probe, "probe", registry)
            site, vec = _single_site(vecs)
            when_kind = "probe"
            probe_site: tuple[int, str] | None = site
            probe_vec: np.ndarray | None = np.ascontiguousarray(vec)
            threshold: float | None = float(gate.when.threshold)
            sharpness: float | None = float(gate.when.sharpness)
        else:
            when_kind = "always"
            probe_site = probe_vec = threshold = sharpness = None
        # apply
        steer_name: str | None = None
        steer_digest: str | None = None
        if isinstance(gate.apply, AddApply):
            vecs, steer_name, steer_digest = _resolve_source(
                gate.apply.steer, "steer", registry
            )
            steer = _scale_vectors(vecs, float(gate.apply.strength))
            apply_kind = "add"
            strength = float(gate.apply.strength)
        elif isinstance(gate.apply, ClampApply):
            # Modulates the request's static clamps (no vector source).
            apply_kind = "clamp"
            steer = None
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
                steer_name=steer_name,
                steer_digest=steer_digest,
            )
        )
    return resolved


def resolve_gates_safe(
    gates: list[SteeringGate] | None,
    req_id: str | None = None,
    registry: _VectorRegistry | None = None,
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

    An unknown ``NamedVec`` (a name unregistered between the frontend check
    and worker admission — a benign race) is a resolution failure and takes
    the same graceful path: the request proceeds without its declarative
    gates rather than aborting the engine core.

    Determinism: the gates arrive as identical serialized bytes on every TP
    rank and the worker registry is rank-replicated, so a malformed or
    unresolvable spec fails identically on all ranks; the graceful path
    returns ``None`` everywhere and cannot desync them.

    Args:
        gates: Per-request gates to resolve, or ``None``.
        req_id: Request id, included in the warning for triage.
        registry: Worker named-vector registry for ``NamedVec`` resolution.

    Returns:
        Resolved gates, or ``None`` if the input was empty or unresolvable.
    """
    if not gates:
        return None
    try:
        return resolve_gates(gates, registry)
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


def _validate_source_dict(
    src: dict, kind: str, registry: _VectorRegistry | None
) -> None:
    """Validate one raw vec-source dict *in place* (no inflation).

    ``{"kind":"name","name":X}`` → existence-checked against *registry*; the
    name is left un-inflated so it rides the wire and resolves worker-side.
    ``{"kind":"inline","packed":...}`` → hooks validated.
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
        if registry.resolve_vectors(name, kind) is None:
            raise ValueError(f"unknown {kind} vector name {name!r}")
        return
    if skind == "inline":
        packed = src.get("packed")
        if not isinstance(packed, dict) or not packed:
            raise ValueError("inline vector source requires a non-empty 'packed'")
        _validate_hooks(packed)
        return
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
    - ``add`` with ``scope=rest_of_conversation`` must use a ``NamedVec``
      steer source. Such a gate is latched server-side and bridged across
      later turns; persisting the client's inline bytes would pin them in
      server memory indefinitely (see docs/design/dynamic_steering.md §8.3).
      Latching by reference to a registered name requires the source to be a
      name — inline is rejected here. Ephemeral scopes keep inline support.
    - ``clamp`` (:class:`ClampApply`) is rejected outright: no substrate can
      honor a *per-request* clamp gate today (the clamp op reads the shared
      row-gate buffer, written only by the GLOBAL cross-layer monitor; the
      per-row monitor is fused/non-mutating and cannot materialize gate
      values). Accepting it would either silently do nothing (no global
      monitor) or silently follow the global monitor's probe instead of the
      declared one. Server-side clamp gating remains available.
    """
    for gate in gates:
        if isinstance(gate.apply, ClampApply):
            raise ValueError(
                "per-request clamp gates (apply.kind='clamp') are not "
                "supported yet: clamp gating currently follows the server's "
                "GLOBAL cross-layer monitor, not a per-request probe. "
                "Configure it server-side instead: set "
                "steering_config.enable_cross_layer_monitor=true and install "
                "a global steering monitor with gate_rows — all clamps at "
                "layers >= the probe layer are then modulated."
            )
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
        if (
            isinstance(gate.apply, AddApply)
            and gate.scope == GateScope.REST_OF_CONVERSATION
            and isinstance(gate.apply.steer, InlineVec)
        ):
            raise ValueError(
                "rest_of_conversation steering persists server-side and "
                "requires a registered vector name; register via "
                "/v1/steering/vectors/register or use rest_of_request."
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


def validate_clamp_gate_support(
    gates: list[SteeringGate] | None,
    *,
    monitor_writes_gates: bool,
) -> None:
    """Reject clamp-target gates — no engine mode can honor them today.

    A gate with ``apply.kind == "clamp"`` would modulate the request's
    directional clamps through the SHARED ``steering_row_gate`` buffer. That
    buffer is only *written* by the GLOBAL cross-layer monitor
    (``steering_config.enable_cross_layer_monitor``, "monitor writes gates");
    the per-request (per-row) monitor is fused into the steering kernel —
    non-mutating, same-hook — so a per-request probe can never materialize
    gate values the clamp op reads. Consequently:

    - **fused mode** (``monitor_writes_gates=False``): the gate never reaches
      the clamp op at all — reject, naming the missing flag AND the
      per-request limitation.
    - **materialized mode** (``True``): the clamp op does read the shared
      gate, but the values come from the server's GLOBAL monitor probe —
      the request's declared probe/threshold/scope would be silently
      ignored (or, with no global monitor installed, the gate would
      silently do nothing). Reject rather than mislead; point at the
      server-side global-monitor flow that actually works.

    Kept as a separate seam (rather than only ``_validate_gate_semantics``)
    so that when a materializing per-row monitor lands, the materialized-mode
    branch can start accepting without touching the semantics validator's
    contract.

    Args:
        gates: The parsed gates, or ``None``.
        monitor_writes_gates: Whether the engine materializes gates into the
            shared row-gate buffer (``enable_cross_layer_monitor``).

    Raises:
        ValueError: A clamp-target gate is present. The message is
            actionable for the given mode.
    """
    if not gates:
        return
    if not any(isinstance(gate.apply, ClampApply) for gate in gates):
        return
    if not monitor_writes_gates:
        raise ValueError(
            "declarative gate targets clamps (apply.kind='clamp') but this "
            "engine does not materialize gates: directional clamps read the "
            "shared row-gate buffer, which is only written when the "
            "cross-layer monitor is enabled "
            "(steering_config.enable_cross_layer_monitor=true, "
            "monitor_writes_gates). Note that per-request clamp gates are "
            "not supported in any mode yet — clamp gating follows the "
            "server's GLOBAL monitor (install one with gate_rows); drop the "
            "clamp gate from the request."
        )
    raise ValueError(
        "per-request clamp gates (apply.kind='clamp') are not supported "
        "yet: this engine materializes gates, but the values written to the "
        "shared row-gate buffer come from the server's GLOBAL cross-layer "
        "monitor — the request's declared probe/threshold/scope would be "
        "ignored. Install a global steering monitor with gate_rows to "
        "modulate all clamps at layers >= the probe layer, and drop the "
        "clamp gate from the request."
    )


def build_steering_gates(
    raw: list[dict] | None,
    registry: _VectorRegistry | None,
    *,
    monitor_writes_gates: bool | None = None,
) -> list[SteeringGate] | None:
    """Validate raw JSON gates for the wire (no name inflation).

    Called at the frontend (``to_request_metadata``). ``NamedVec`` sources are
    validated for existence against *registry* but left un-inflated so the
    short name — not the full vector blob — rides the msgpack channel and
    resolves against the worker registry at admission. Raises ``ValueError``
    on malformed gates, unknown names, invalid hooks, or an inline steer on a
    ``rest_of_conversation`` gate (which must latch by name) — callers surface
    this as HTTP 400.

    Clamp-target gates (``apply.kind == "clamp"``) are rejected
    unconditionally by ``_validate_gate_semantics`` — no engine mode honors
    per-request clamp gates today (see :class:`ClampApply`).
    ``monitor_writes_gates`` (when supplied by the caller) is the engine's
    ``enable_cross_layer_monitor`` setting, threaded to
    :func:`validate_clamp_gate_support` so its error can be mode-tailored
    and so materialized-mode acceptance can be enabled there later without
    changing this signature. ``None`` ⇒ mode unknown; the unconditional
    semantics rejection still applies.
    """
    if not raw:
        return None
    if not isinstance(raw, list):
        raise ValueError("'steering' must be a list of gate objects")
    for gate in raw:
        if not isinstance(gate, dict):
            raise ValueError("each steering gate must be an object")
        when = gate.get("when")
        if isinstance(when, dict) and when.get("kind") == "probe":
            _validate_source_dict(when.get("probe", {}), "probe", registry)
        apply = gate.get("apply")
        if isinstance(apply, dict) and apply.get("kind") == "add":
            _validate_source_dict(apply.get("steer", {}), "steer", registry)
    try:
        gates = msgspec.convert(raw, type=list[SteeringGate])
    except msgspec.ValidationError as exc:
        raise ValueError(f"invalid steering gate spec: {exc}") from exc
    # Fail fast on semantic problems the substrate cannot honor
    # (unsupported combos, non-finite/negative probe params, an inline steer
    # on a persisted rest_of_conversation gate).
    _validate_gate_semantics(gates)
    # Reject clamp-target gates the engine cannot materialize (HTTP 400 here
    # rather than silently ungated clamps at the worker). Skipped when the
    # caller did not supply the mode.
    if monitor_writes_gates is not None:
        validate_clamp_gate_support(gates, monitor_writes_gates=monitor_writes_gates)
    # Fail fast on structural problems the consumer would otherwise hit
    # (probe must name exactly one site; add must carry vectors). Resolve
    # against the frontend registry so named sources are validated too;
    # normalize non-ValueError unpack failures (e.g. ``np.dtype("garbage")``
    # raises TypeError, missing keys raise KeyError) into ValueError so
    # callers return HTTP 400 rather than 500.
    try:
        resolve_gates(gates, registry)
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
    "ClampApply",
    "Apply",
    "SteeringGate",
    "ResolvedGate",
    "resolve_gates",
    "resolve_gates_safe",
    "build_steering_gates",
    "validate_clamp_gate_support",
]
