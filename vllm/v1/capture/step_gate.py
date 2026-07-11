# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-replicated force-eager predicate for activation capture.

The **client-spec** activation gather (``CaptureManager.on_hook``'s
``index_select`` path) allocates a fresh, variable-size output each step,
so it cannot execute inside a replayed CUDA graph: any forward step that
gathers for a client spec must run eager.  (Global specs take a separate,
CUDA-graph-safe path — a fixed-shape full-residual ``copy_`` into a
persistent buffer baked into the graph at warmup — so they do **not**
force eager; see :meth:`CaptureManager.on_hook` and
``CaptureManager._global_buffers``.)

Under tensor/pipeline parallelism every rank must reach the *same*
eager-vs-cudagraph decision: a divergent ``num_tokens_padded`` would
misalign the TP all-reduce / PP send-recv and deadlock.  A per-step
collective to agree is impossible — it would be a synchronous barrier
inside PP's asynchronous pipeline.

:class:`CaptureStepGate` resolves this without any collective.  It is built
identically on every rank and fed only data that is byte-identical across
ranks: the broadcast ``scheduler_output`` (projected into a
:class:`CaptureBatchView`) and each request's client capture spec (which
rides in ``SamplingParams`` on every rank).  Because the inputs are
identical and the predicate is a pure function, every rank computes the
same boolean by construction.

The gate is a deliberate **superset** of what the capturer rank actually
gathers: it ignores pipeline-stage layer filtering and admission-time
validation (both of which only *reduce* the capturer's captures).  So it
never returns ``False`` on a step the capturer would gather for a client
spec — it can only force an occasional harmless extra eager step.
Under-forcing would be a correctness bug (the dynamic gather would
silently no-op inside a CUDA graph); over-forcing only costs a little
cudagraph speed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.v1.capture.manager import selector_hits_window

if TYPE_CHECKING:
    from vllm.v1.capture.plan import CaptureBatchView

# A position selector as it appears in a client capture spec: either an
# explicit list of logical indices or a symbolic selector string
# (``last_prompt`` / ``all_prompt`` / ``all_generated`` / ``all``).
PositionSelector = list[int] | str

# Selector used when a request carries a capture spec we cannot parse a
# position out of.  ``"all"`` hits every step with scheduled tokens, so the
# gate conservatively forces eager whenever that request runs — never
# under-forcing.
_CONSERVATIVE_SELECTOR: PositionSelector = "all"

# Sentinel ``(layer, hook)`` set meaning "this consumer's tapped keys could
# not be parsed". A consumer whose keys are unknown is conservatively treated
# as tapping an uncovered key, so it forces eager whenever it captures —
# never under-forcing into a CUDA graph the dynamic gather can't replay.
_UNKNOWN_KEYS: frozenset[tuple[int, str]] | None = None


def _extract_consumer_taps(
    raw_capture: Mapping[str, Any] | None,
) -> list[tuple[PositionSelector, frozenset[tuple[int, str]] | None]]:
    """Pull per-consumer ``(position selector, tapped keys)`` from a spec.

    ``raw_capture`` is ``SamplingParams.capture`` — a mapping of consumer
    name to that consumer's raw client spec (a dict like the filesystem
    request, or an already-parsed object). Returns one entry per consumer:
    its position selector and the set of ``(layer, hook)`` keys it taps.

    The tapped keys decide whether this consumer's capture can ride the
    graph-safe persistent-buffer path: a consumer tapping only allowlisted
    keys never forces eager. A consumer with no recognizable ``positions``
    contributes the conservative ``"all"`` selector, and a consumer whose
    ``hooks`` cannot be parsed contributes ``_UNKNOWN_KEYS`` (treated as
    tapping an uncovered key) — both so the request still forces eager.
    """
    if not raw_capture:
        return []
    taps: list[tuple[PositionSelector, frozenset[tuple[int, str]] | None]] = []
    for raw_spec in raw_capture.values():
        if isinstance(raw_spec, Mapping):
            positions = raw_spec.get("positions")
            hooks = raw_spec.get("hooks")
        else:
            positions = getattr(raw_spec, "positions", None)
            hooks = getattr(raw_spec, "hooks", None)
        selector: PositionSelector
        if isinstance(positions, (str, list)):
            selector = positions
        else:
            selector = _CONSERVATIVE_SELECTOR
        taps.append((selector, _parse_hook_keys(hooks)))
    return taps


def _parse_hook_keys(hooks: Any) -> frozenset[tuple[int, str]] | None:
    """Parse a spec's ``hooks`` mapping into a set of ``(layer, hook)`` keys.

    Returns ``_UNKNOWN_KEYS`` (``None``) when the structure is not the
    expected ``{hook_name: [layer, ...]}`` mapping, so the gate treats the
    consumer conservatively (forces eager when it captures).
    """
    if not isinstance(hooks, Mapping):
        return _UNKNOWN_KEYS
    keys: set[tuple[int, str]] = set()
    for hook_name, layers in hooks.items():
        if not isinstance(hook_name, str) or not isinstance(layers, (list, tuple)):
            return _UNKNOWN_KEYS
        for layer in layers:
            if not isinstance(layer, int) or isinstance(layer, bool):
                return _UNKNOWN_KEYS
            keys.add((layer, hook_name))
    return frozenset(keys)


def _extract_selectors(raw_capture: Mapping[str, Any] | None) -> list[PositionSelector]:
    """Position selectors only (kept for existing call sites/tests)."""
    return [selector for selector, _keys in _extract_consumer_taps(raw_capture)]


class CaptureStepGate:
    """Decides force-eager per step from rank-identical inputs only.

    Construct one per model runner (on *every* rank when the capture
    feature is enabled).  Populate it from the new-request stream with
    :meth:`register` and drop finished requests with :meth:`drop`, both
    driven off the same ``scheduler_output`` every rank sees.
    """

    def __init__(
        self,
        *,
        force_all: bool = False,
        graphsafe_keys: frozenset[tuple[int, str]] | None = None,
    ) -> None:
        # ``force_all`` is a manual always-eager escape hatch (debugging,
        # or a hypothetical consumer whose capture genuinely cannot use the
        # graph-safe global-buffer path).  It is **off** by default: global
        # specs no longer force eager — they ride the persistent-buffer
        # path baked into the CUDA graph at warmup — so the gate forces
        # eager only when a *client* spec actually captures this step.  It
        # must still be set identically on every rank when used, since the
        # eager decision is rank-replicated.
        #
        # ``graphsafe_keys`` is the startup-configured per-request capture
        # allowlist (the global, unfiltered ``(layer, hook)`` set). A client
        # spec tapping only allowlisted keys rides the persistent-buffer path
        # and does **not** force eager; tapping any other key still does. It
        # must be byte-identical across ranks (it comes from config), which
        # keeps the rank-replicated decision in lockstep.
        self._force_all = force_all
        self._graphsafe_keys = graphsafe_keys or frozenset()
        # req_id -> list of (position selector, tapped keys | _UNKNOWN_KEYS).
        self._taps: dict[
            str, list[tuple[PositionSelector, frozenset[tuple[int, str]] | None]]
        ] = {}

    @property
    def force_all(self) -> bool:
        return self._force_all

    def register(self, req_id: str, raw_capture: Mapping[str, Any] | None) -> None:
        """Track a request's capture taps (no-op if it captures nothing)."""
        if self._force_all:
            return
        taps = _extract_consumer_taps(raw_capture)
        if taps:
            self._taps[req_id] = taps

    def drop(self, req_id: str) -> None:
        """Forget a finished/aborted request."""
        self._taps.pop(req_id, None)

    def tracked_requests(self) -> int:
        """Number of capture requests currently tracked (for tests/debug)."""
        return len(self._taps)

    def step_captures(self, view: CaptureBatchView) -> bool:
        """True iff some *client*-spec request must run eager this step.

        Pure function of *view* (rank-identical) and the registered client
        taps (parsed from rank-identical ``SamplingParams.capture``). A
        consumer forces eager only when it both (a) captures a position in
        this step's window and (b) taps a ``(layer, hook)`` outside the
        startup graph-safe allowlist. A consumer tapping only allowlisted
        keys rides the persistent-buffer path and never forces eager; global
        specs (absent from client specs) are likewise excluded.
        """
        if self._force_all:
            # Manual always-eager escape hatch (off by default).
            return True
        if not self._taps:
            return False
        graphsafe = self._graphsafe_keys
        for i, req_id in enumerate(view.req_ids):
            taps = self._taps.get(req_id)
            if not taps:
                continue
            num_scheduled = view.num_scheduled_tokens[i]
            if num_scheduled <= 0:
                continue
            num_computed = view.num_computed_tokens[i]
            num_prompt = view.num_prompt_tokens[i]
            for selector, keys in taps:
                # Only consumers whose keys are entirely covered by the
                # graph-safe allowlist avoid eager. ``keys is None``
                # (``_UNKNOWN_KEYS``) or any uncovered key forces eager.
                if keys is not None and keys <= graphsafe:
                    continue
                if selector_hits_window(
                    selector, num_prompt, num_computed, num_scheduled
                ):
                    return True
        return False
