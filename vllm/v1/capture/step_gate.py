# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-replicated force-eager predicate for activation capture.

The per-step activation gather (``CaptureManager.on_hook``) runs Python
that cannot execute inside a replayed CUDA graph, so any forward step
that actually captures must run eager.  Under tensor/pipeline parallelism
every rank must reach the *same* eager-vs-cudagraph decision: a divergent
``num_tokens_padded`` would misalign the TP all-reduce / PP send-recv and
deadlock.  A per-step collective to agree is impossible — it would be a
synchronous barrier inside PP's asynchronous pipeline.

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
never returns ``False`` on a step the capturer would capture — it can only
force an occasional harmless extra eager step.  Under-forcing would be a
correctness bug (the gather would silently no-op inside a CUDA graph);
over-forcing only costs a little cudagraph speed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.v1.capture.manager import selector_hits_window

if TYPE_CHECKING:
    from vllm.config import VllmConfig
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
        # ``force_all`` mirrors the old always-eager "hammer": set when any
        # configured consumer *may* define a global capture spec (one that
        # captures for every request, independent of the client spec).  In
        # that mode essentially every request captures, so there is no
        # plain-request speedup to recover and the gate degrades to the
        # safe always-eager behavior.  Detected from config identically on
        # every rank (see :func:`force_all_from_config`).
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
        """True iff some request in *view* captures a position this step.

        Pure function of *view* (rank-identical) and the registered
        selectors (parsed from rank-identical client specs).
        """
        if self._force_all:
            # Global-spec mode: every request captures — reproduce the
            # always-eager hammer exactly.
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


def force_all_from_config(vllm_config: VllmConfig) -> bool:
    """Detect whether any configured consumer defines a global spec.

    A global capture spec captures for *every* request regardless of the
    client spec, so its presence means the gate must fall back to
    always-eager (there are no plain requests left to speed up).

    ``global_capture_spec()`` can return ``None`` even when overridden
    (e.g. :class:`FilesystemConsumer` defines it to return ``None``), so a
    class-level "is it overridden" check is too coarse — it would wrongly
    flag filesystem-only deployments and defeat the whole optimization.
    We instead **probe-construct** each configured consumer once, call
    ``global_capture_spec()``, and shut it down.  This is run identically
    on every rank against the rank-identical config, so the result agrees
    by construction with no collective.  The probe builds the consumer
    class directly (not via ``build_consumers``), so it installs no driver
    bridge; any writer threads it spins are joined immediately by the
    ``shutdown`` in the ``finally``.  Construction or spec-evaluation
    failure is treated conservatively as "has a global spec" (safe
    over-forcing), keeping ranks identical.

    This runs once at model-runner construction; the cost is a transient
    consumer build, never on the per-step hot path.
    """
    config = getattr(vllm_config, "capture_consumers_config", None)
    if config is None:
        return False

    if hasattr(config, "consumers"):
        config_entries: list[Any] = list(config.consumers)
    elif isinstance(config, list):
        config_entries = list(config)
    else:
        config_entries = []

    for entry in config_entries:
        if hasattr(entry, "name"):
            name = entry.name
            params = getattr(entry, "params", {}) or {}
        else:
            name = entry["name"]
            params = entry.get("params", {}) or {}
        if _probe_has_global_spec(name, vllm_config, params):
            return True

    for instance in getattr(config, "instances", None) or []:
        try:
            if instance.global_capture_spec() is not None:
                return True
        except Exception:
            return True

    return False


def _probe_has_global_spec(
    name: str, vllm_config: VllmConfig, params: dict[str, Any]
) -> bool:
    """Construct consumer *name*, ask for its global spec, then tear down."""
    from vllm.v1.capture.registry import load_consumer_class

    try:
        cls = load_consumer_class(name)
    except Exception:
        # Unknown consumer name → conservative.
        return True

    consumer = None
    try:
        consumer = cls(vllm_config, params)
        return consumer.global_capture_spec() is not None
    except Exception:
        return True
    finally:
        if consumer is not None:
            shutdown = getattr(consumer, "shutdown", None)
            if callable(shutdown):
                with contextlib.suppress(Exception):
                    shutdown()
