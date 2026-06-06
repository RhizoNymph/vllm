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


def _extract_selectors(raw_capture: Mapping[str, Any] | None) -> list[PositionSelector]:
    """Pull the per-consumer position selectors out of a client spec.

    ``raw_capture`` is ``SamplingParams.capture`` — a mapping of consumer
    name to that consumer's raw client spec (a dict like the filesystem
    request, or an already-parsed object).  We only need the union of
    position selectors across the request's consumers; layers and hooks
    are irrelevant to *whether* the step captures (if any layer captures,
    some pipeline stage gathers and all ranks must agree to run eager).

    A consumer entry with no recognizable ``positions`` field contributes
    the conservative ``"all"`` selector so the request still forces eager.
    """
    if not raw_capture:
        return []
    selectors: list[PositionSelector] = []
    for raw_spec in raw_capture.values():
        if isinstance(raw_spec, Mapping):
            positions = raw_spec.get("positions")
        else:
            positions = getattr(raw_spec, "positions", None)
        if isinstance(positions, (str, list)):
            selectors.append(positions)
        else:
            selectors.append(_CONSERVATIVE_SELECTOR)
    return selectors


class CaptureStepGate:
    """Decides force-eager per step from rank-identical inputs only.

    Construct one per model runner (on *every* rank when the capture
    feature is enabled).  Populate it from the new-request stream with
    :meth:`register` and drop finished requests with :meth:`drop`, both
    driven off the same ``scheduler_output`` every rank sees.
    """

    def __init__(self, *, force_all: bool = False) -> None:
        # ``force_all`` is a manual always-eager escape hatch (debugging,
        # or a hypothetical consumer whose capture genuinely cannot use the
        # graph-safe global-buffer path).  It is **off** by default: global
        # specs no longer force eager — they ride the persistent-buffer
        # path baked into the CUDA graph at warmup — so the gate forces
        # eager only when a *client* spec actually captures this step.  It
        # must still be set identically on every rank when used, since the
        # eager decision is rank-replicated.
        self._force_all = force_all
        self._selectors: dict[str, list[PositionSelector]] = {}

    @property
    def force_all(self) -> bool:
        return self._force_all

    def register(self, req_id: str, raw_capture: Mapping[str, Any] | None) -> None:
        """Track a request's capture position selectors (no-op if none)."""
        if self._force_all:
            return
        selectors = _extract_selectors(raw_capture)
        if selectors:
            self._selectors[req_id] = selectors

    def drop(self, req_id: str) -> None:
        """Forget a finished/aborted request."""
        self._selectors.pop(req_id, None)

    def tracked_requests(self) -> int:
        """Number of capture requests currently tracked (for tests/debug)."""
        return len(self._selectors)

    def step_captures(self, view: CaptureBatchView) -> bool:
        """True iff some *client*-spec request captures a position this step.

        Pure function of *view* (rank-identical) and the registered
        client-spec selectors (parsed from rank-identical
        ``SamplingParams.capture``).  Global specs are excluded — they are
        served by the CUDA-graph-safe persistent-buffer path and never
        force eager.
        """
        if self._force_all:
            # Manual always-eager escape hatch (off by default).
            return True
        if not self._selectors:
            return False
        for i, req_id in enumerate(view.req_ids):
            selectors = self._selectors.get(req_id)
            if not selectors:
                continue
            num_computed = view.num_computed_tokens[i]
            num_scheduled = view.num_scheduled_tokens[i]
            if num_scheduled <= 0:
                continue
            num_prompt = view.num_prompt_tokens[i]
            for selector in selectors:
                if selector_hits_window(
                    selector, num_prompt, num_computed, num_scheduled
                ):
                    return True
        return False
