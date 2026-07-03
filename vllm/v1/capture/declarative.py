# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in declarative per-request steering consumer.

Where :class:`~vllm.v1.capture.controller.SteeringController` runs one
operator-authored policy (:meth:`decide`), this consumer runs the *client's*
own gates: each request may carry a nested list of ``when × scope × apply``
gates in :attr:`~vllm.v1.capture.step_view.StepRequestView.steering` (from
``RequestMetadata.steering``; see :mod:`vllm.v1.steering_schema`). This
consumer reads them off the per-step view and drives the steering substrate —
no server-registered consumer needed.

It subclasses :class:`SteeringController` to reuse the bounded conversation
latch/bridge (:meth:`_latch`/:meth:`_bridge_override`) and the ``_armed``
per-request lifecycle, but overrides :meth:`on_step` because the declarative
model is multi-gate and multi-scope (the base dispatches a single site to a
single :meth:`decide`).

Gate → substrate mapping:

- ``apply=add`` installs a per-request dynamic override
  (:class:`RequestSteeringOverride`, ``compose_admitted=True`` so the add is
  composed *on top of* the request's static decode steering). ``scope`` sets
  the override's lifetime; ``when=probe`` with ``scope=this_token`` also
  attaches a per-request row monitor (:class:`SteeringMonitorUpdate` keyed by
  ``req_id``) so the gate is re-evaluated **in-graph every decode token**
  (free, cudagraph-safe, requires ``enable_row_monitor``). Host-evaluated
  scopes (``next_step`` / ``rest_of_request`` / ``rest_of_conversation``)
  evaluate the probe once on the CPU against the captured residual.
- ``apply=attenuate`` damps the request's steering by ``strength`` via a
  per-request :class:`SteeringScaleUpdate` (installing an admitted-only
  override first when the request has no add gate, so the damp is per-request
  rather than shared across a config row).

Precedence (operator wins) is enforced runner-side: every action is stamped
``source="declarative"``, and the runner rejects a declarative action for a
request already owned by another source (see
``steering_model_runner_mixin._apply_request_override``).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config.steering_types import merge_steering_specs
from vllm.logger import init_logger
from vllm.v1.capture.controller import SteeringController
from vllm.v1.worker.steering_action_queue import (
    DECLARATIVE_SOURCE,
    RequestSteeringOverride,
    SteeringAction,
    SteeringMonitorUpdate,
    SteeringScaleUpdate,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.capture.step_view import StepCaptureView, StepRequestView
    from vllm.v1.capture.types import CaptureSpec
    from vllm.v1.steering_schema import ResolvedGate

logger = init_logger(__name__)

# Max entries kept in the per-consumer host-probe tensor cache.
_PROBE_CACHE_MAX = 64

# Override-lifetime ranking used to pick the widest scope among a request's
# firing add gates (they share one override row).
_SCOPE_RANK = {
    "this_token": 0,
    "next_step": 1,
    "rest_of_request": 2,
    "rest_of_conversation": 3,
}


def _parse_site(spec: str) -> tuple[int, str]:
    """Parse a ``"layer:hook"`` probe-site string to ``(layer, hook)``."""
    from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

    layer_str, sep, hook = spec.partition(":")
    if not sep or not hook:
        raise ValueError(
            f"declarative probe site {spec!r} must be 'layer:hook' "
            f"(e.g. '12:post_block')"
        )
    try:
        layer = int(layer_str)
    except ValueError as exc:
        raise ValueError(
            f"declarative probe site {spec!r} has a non-integer layer"
        ) from exc
    if hook not in VALID_HOOK_POINT_NAMES:
        raise ValueError(
            f"declarative probe site {spec!r} has unknown hook {hook!r}; "
            f"valid: {sorted(VALID_HOOK_POINT_NAMES)}"
        )
    return layer, hook


class DeclarativeSteeringConsumer(SteeringController):
    """Applies each request's declared ``when × scope × apply`` gates.

    Params:

    - ``probe_sites`` (list[str], default ``["0:post_block"]``): the
      ``layer:hook`` sites whose residual is captured so host-evaluated probes
      (non-``this_token`` scopes with ``when=probe``) can read it. A gate whose
      host probe names a site outside this list never fires (logged once).
      ``this_token`` probes are in-graph and need no capture.
    - ``max_conversations`` (int): bound on the ``rest_of_conversation`` latch
      map (see :class:`SteeringController`).
    """

    def __init__(self, vllm_config: VllmConfig, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        sites = params.get("probe_sites") or ["0:post_block"]
        # Accept both space-separated (list) and comma-separated (one string)
        # forms, since CLI list args commonly arrive as a single "a,b" token.
        self._probe_sites: list[tuple[int, str]] = [
            _parse_site(s.strip())
            for entry in sites
            for s in str(entry).split(",")
            if s.strip()
        ]
        # req_ids with a one-step (``next_step``) override to clear next step.
        self._next_step_pending: set[str] = set()
        # Bounded LRU cache: numpy probe → torch tensor (host-probe GEMM).
        # Keyed by a content digest (not ``id(probe)``): the consumer runs
        # independently on every TP rank and must decide bit-identically, so
        # the key must be a pure function of the array's value. ``id()`` keying
        # both leaked (one tensor per distinct array, never evicted) and, once
        # a gate's array was GC'd, could alias a new probe's reused address to
        # a stale tensor — diverging per rank and silently desyncing tables.
        self._probe_tensor_cache: OrderedDict[
            tuple[tuple[int, ...], str, bytes], torch.Tensor
        ] = OrderedDict()
        self._missing_site_warned = False
        self._unsupported_combo_warned = False
        self._gate_errors = 0

    # -- SyncCaptureConsumer interface -------------------------------------

    def global_capture_spec(self) -> CaptureSpec:
        from vllm.v1.capture.types import CaptureSpec

        hooks: dict[str, list[int]] = {}
        for layer, hook in self._probe_sites:
            hooks.setdefault(hook, []).append(layer)
        return CaptureSpec(hooks=hooks, positions="all_generated")

    def decide(self, request_view, residual):  # noqa: D102 - unused
        # SteeringController requires ``decide``; this consumer overrides
        # ``on_step`` entirely and never calls it.
        return None

    # -- helpers -----------------------------------------------------------

    def _probe_tensor(self, probe: np.ndarray, device: torch.device) -> torch.Tensor:
        """Return ``probe`` as an fp32 device tensor, content-cached (bounded).

        Args:
            probe: The per-request probe vector.
            device: Target device for the GEMV.

        Returns:
            The uploaded fp32 tensor, reused across value-equal probes.
        """
        arr = np.ascontiguousarray(probe, dtype=np.float32)
        key = (arr.shape, arr.dtype.str, hashlib.sha1(arr.tobytes()).digest())
        cache = self._probe_tensor_cache
        t = cache.get(key)
        if t is None or t.device != device:
            t = torch.from_numpy(arr).to(device)
            cache[key] = t
            cache.move_to_end(key)
            while len(cache) > _PROBE_CACHE_MAX:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return t

    def _host_probe_fires(
        self, gate: ResolvedGate, req: StepRequestView, view: StepCaptureView
    ) -> bool:
        """Evaluate a host-side probe against the captured residual window."""
        assert gate.probe_site is not None and gate.probe_vec is not None
        key = gate.probe_site  # (layer, hook)
        tensor = view.tensors.get(key)
        if tensor is None:
            if not self._missing_site_warned:
                self._missing_site_warned = True
                logger.warning(
                    "declarative steering: host-probe site %s is not captured "
                    "(add it to the consumer's probe_sites); gate skipped.",
                    key,
                )
            return False
        residual = tensor[req.start : req.end]
        probe = self._probe_tensor(gate.probe_vec, residual.device)
        score = (residual.to(torch.float32) @ probe).mean()
        thr = float(gate.threshold or 0.0)
        sharp = float(gate.sharpness or 1.0)
        gate_val = torch.sigmoid(sharp * (score - thr))
        return bool(gate_val.item() > 0.5)

    def _row_monitor_action(
        self, gate: ResolvedGate, req_id: str
    ) -> SteeringMonitorUpdate:
        layer, hook = gate.probe_site  # type: ignore[misc]
        return SteeringMonitorUpdate(
            hook=hook,
            layer=layer,
            probe=np.ascontiguousarray(gate.probe_vec, dtype=np.float32),
            threshold=float(gate.threshold or 0.0),
            sharpness=float(gate.sharpness or 1.0),
            req_id=req_id,
            source=DECLARATIVE_SOURCE,
        )

    # -- main loop ---------------------------------------------------------

    def on_step(self, view: StepCaptureView) -> list[SteeringAction] | None:
        actions: list[SteeringAction] = []
        live: set[str] = set()
        for req in view.requests:
            live.add(req.req_id)
            rid = req.req_id
            # Expire one-step (next_step) overrides installed last step.
            if rid in self._next_step_pending:
                self._next_step_pending.discard(rid)
                actions.append(
                    RequestSteeringOverride(
                        req_id=rid, vectors=None, source=DECLARATIVE_SOURCE
                    )
                )
            if req.phase != "decode":
                continue
            cid = req.conversation_id
            # Bridge a new request of an already-latched conversation FIRST —
            # a later turn is bridged even when it declares no gates of its own
            # (that is the point of rest_of_conversation).
            if (
                rid not in self._armed
                and cid is not None
                and cid in self._latched
            ):
                actions.append(self._bridge_override(self._latched[cid], rid))
                self._armed.add(rid)
                self._bridges += 1
                continue
            if rid in self._armed:
                continue
            if not req.steering:
                continue
            try:
                self._process_request(req, view, actions)
            except Exception as exc:  # never break the step for one bad gate
                self._gate_errors += 1
                logger.warning(
                    "declarative steering: gate processing failed for %s: %s",
                    rid,
                    exc,
                )
        self._armed &= live
        self._next_step_pending &= live
        return actions or None

    def _process_request(
        self,
        req: StepRequestView,
        view: StepCaptureView,
        actions: list[SteeringAction],
    ) -> None:
        rid = req.req_id
        cid = req.conversation_id
        gates: list[ResolvedGate] = req.steering  # type: ignore[assignment]

        merged_add: dict | None = None
        add_scope_rank = -1
        latch_add = False
        row_monitor: SteeringMonitorUpdate | None = None
        attenuate_strength: float | None = None

        for gate in gates:
            # Trigger evaluation.
            if gate.when_kind == "probe" and gate.scope != "this_token":
                # Host-evaluated probe (sticky/pulse/latched scopes).
                if not self._host_probe_fires(gate, req, view):
                    continue
            # ``always`` and ``this_token+probe`` always install; the latter's
            # per-token conditionality is handled in-graph by the row monitor.

            if gate.apply_kind == "add":
                if not gate.steer_vectors:
                    continue
                merged_add = (
                    dict(gate.steer_vectors)
                    if merged_add is None
                    else merge_steering_specs(merged_add, gate.steer_vectors)
                )
                add_scope_rank = max(add_scope_rank, _SCOPE_RANK[gate.scope])
                if gate.scope == "rest_of_conversation":
                    latch_add = True
                if (
                    gate.when_kind == "probe"
                    and gate.scope == "this_token"
                    and row_monitor is None
                ):
                    row_monitor = self._row_monitor_action(gate, rid)
            else:  # attenuate: last one wins (they all damp the same row)
                if gate.when_kind == "probe" and gate.scope == "this_token":
                    # Unsupported combo: the substrate cannot damp a row only
                    # when the probe fires this token (the per-row monitor
                    # gates a row toward zero when the probe is LOW — the wrong
                    # shape). The frontend rejects this
                    # (steering_schema._validate_gate_semantics), but
                    # non-frontend producers (offline, the Rust frontend) can
                    # still emit it: skip-and-warn rather than silently
                    # attenuating unconditionally.
                    if not self._unsupported_combo_warned:
                        self._unsupported_combo_warned = True
                        logger.warning(
                            "declarative steering: attenuate with when=probe "
                            "and scope=this_token is unsupported; gate skipped. "
                            "Use scope=next_step or rest_of_request."
                        )
                    continue
                attenuate_strength = float(gate.strength)

        armed = False
        pulse = False

        if merged_add is not None:
            override = RequestSteeringOverride(
                req_id=rid,
                vectors=merged_add,
                compose_admitted=True,
                source=DECLARATIVE_SOURCE,
            )
            actions.append(override)
            if row_monitor is not None:
                actions.append(row_monitor)
            if latch_add and cid is not None:
                self._latch(cid, override)
            # Lifetime: next_step alone → one-step pulse; anything sticky
            # (this_token/rest_of_request/rest_of_conversation) → keep for the
            # request. rest_of_conversation additionally latched above.
            if add_scope_rank == _SCOPE_RANK["next_step"]:
                pulse = True
            armed = True
            self._triggers += 1
        elif attenuate_strength is not None:
            # Attenuate with no add gate: put the request's admitted decode
            # steering on its own dynamic row (compose_admitted with an empty
            # add), then scale that row per-request. A request with no static
            # steering has nothing to damp — the runner rejects the empty
            # override and logs.
            actions.append(
                RequestSteeringOverride(
                    req_id=rid,
                    vectors={},
                    compose_admitted=True,
                    source=DECLARATIVE_SOURCE,
                )
            )
            armed = True

        if attenuate_strength is not None:
            actions.append(
                SteeringScaleUpdate(
                    scale=max(0.0, attenuate_strength),
                    req_id=rid,
                    source=DECLARATIVE_SOURCE,
                )
            )

        if armed:
            self._armed.add(rid)
        if pulse:
            self._next_step_pending.add(rid)

    def status(self) -> dict[str, Any]:
        st = super().status()
        st.update(
            {
                "next_step_pending": len(self._next_step_pending),
                "probe_sites": [f"{layer}:{hook}" for layer, hook in self._probe_sites],
                "gate_errors": self._gate_errors,
            }
        )
        return st


__all__ = ["DeclarativeSteeringConsumer"]
