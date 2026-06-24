# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-consumer capture manager.

``CaptureManager`` is the per-runner object that coordinates activation
capture across an ordered tuple of ``CaptureSink`` instances.  Each sink
corresponds to one registered capture consumer (e.g., filesystem writer,
reward trainer, dashboard).

Key design properties:

- **Union gather:** When multiple consumers want rows from the same
  ``(layer, hook)`` pair, the gather happens once.  Each entry's
  ``consumer_mask`` bitset records which consumers want it so the
  dispatch path can fan-out without redundant GPU reads.

- **Consumer isolation:** A failing ``submit_chunk`` or
  ``submit_finalize`` on one sink never prevents delivery to the others.
  Errors are captured per ``(consumer, request)`` and surfaced through
  ``CaptureResult``.

- **Position expansion:** The manager resolves the five selector
  modes (``last_prompt``, ``all_prompt``, ``all_generated``, ``all``,
  explicit ``list[int]``) and intersects the result with the step's
  ``[num_computed, num_computed + num_scheduled)`` window.

See ``docs/design/capture_consumers.md`` for the full spec.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import pathlib
import queue
import tempfile
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from vllm.v1.capture.activation_store import (
    activation_key,
    get_active_activation_store,
)
from vllm.v1.capture.plan import (
    CaptureBatchView,
    CapturePositionEntry,
    StepCapturePlan,
)
from vllm.v1.capture.sink import CaptureSink
from vllm.v1.capture.types import (
    CaptureChunk,
    CaptureFinalize,
    CaptureKey,
    CaptureResult,
    CaptureSpec,
    CaptureStatus,
    HookName,
    VllmInternalRequestId,
    captured_prompt_positions,
)

logger = logging.getLogger(__name__)

_CAPTURE_RESULT_SEVERITY: dict[CaptureStatus, int] = {
    "pending": 0,
    "ok": 1,
    "not_requested": 2,
    "partial_error": 3,
    "error": 4,
}


# ---------------------------------------------------------------------------
# Per-request internal state
# ---------------------------------------------------------------------------


@dataclass
class _RequestCaptureState:
    """Bookkeeping for one registered capture request.

    ``consumer_specs`` maps consumer index to the merged spec for this
    request.  ``position_kind`` and ``static_positions`` are per-consumer
    because each consumer may use a different position selector.
    """

    req_id: str
    consumer_specs: dict[int, CaptureSpec]
    position_kind: dict[int, str]
    static_positions: dict[int, list[int] | None]
    num_prompt_tokens: int
    # Prompt block hashes + the granularity they were computed at, for
    # activation-store write-through. ``None`` disables it for this request.
    block_hashes: list[bytes] | None = None
    hash_block_size: int = 0
    steps_seen: int = 0
    error: str | None = None
    sidecar_fields: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dispatch packet — main thread → dispatch thread queue item
# ---------------------------------------------------------------------------


@dataclass
class _DispatchPacket:
    """One step's worth of capture work, handed to the dispatch thread.

    Built on the main thread by ``dispatch_step_captures`` after issuing
    H2D copies into pinned host buffers.  Once the packet is queued, the
    main thread is free to return to the runner; the dispatch thread
    waits on ``cuda_event`` (signalled when the H2Ds complete), then
    walks ``entries`` to fan out CPU-resident slices to each consumer's
    sink.

    ``scratch_pinned`` maps ``(layer, hook)`` to ``(owner_buffer, view)``.
    ``view`` is the row-bounded slice consumed by the dispatch loop;
    ``owner_buffer`` (if not ``None``) is the pinned-pool tensor that
    must be returned to the pool after the packet is processed.  CPU
    scratches (``owner_buffer is None``) need no recycling.
    """

    entries: list[CapturePositionEntry]
    scratch_pinned: dict[tuple[int, str], tuple[torch.Tensor | None, torch.Tensor]]
    cuda_event: torch.cuda.Event | None


@dataclass
class _FinalizeJob:
    """One request's finalize work, handed to the finalize thread.

    The request's capture state is popped from ``_requests`` on the
    *caller* (model-runner step) thread at enqueue time, so the finalize
    thread never touches ``_requests`` and the step thread never blocks on
    ``submit_finalize`` / ``wait_for_result`` (the ~per-layer NFS waits).
    ``on_complete`` is invoked on the finalize thread with the aggregated
    ``{consumer_index: CaptureResult}`` once every key is terminal.
    """

    req_id: str
    state: _RequestCaptureState
    on_complete: Callable[[dict[int, CaptureResult]], None]


# ---------------------------------------------------------------------------
# Position expansion helpers
# ---------------------------------------------------------------------------


def _resolve_positions(
    positions: list[int] | str,
    num_prompt_tokens: int,
    num_computed: int,
    num_scheduled: int,
) -> list[int]:
    """Expand a position selector against the current step bounds.

    Returns absolute logical indices.  For symbolic selectors the upper
    bound is ``num_computed + num_scheduled`` (the highest token the
    forward pass will touch).
    """
    upper = num_computed + num_scheduled

    if isinstance(positions, list):
        return list(positions)

    if positions == "last_prompt":
        return [num_prompt_tokens - 1]

    if positions == "all_prompt":
        return list(range(num_prompt_tokens))

    if positions == "all_generated":
        start = num_prompt_tokens
        return list(range(start, upper))

    if positions == "all":
        return list(range(upper))

    msg = f"Unknown position selector: {positions!r}"
    raise ValueError(msg)


def selector_hits_window(
    positions: list[int] | str,
    num_prompt_tokens: int,
    num_computed: int,
    num_scheduled: int,
) -> bool:
    """True if a position selector captures anything in this step window.

    This is the boolean reduction of :func:`_resolve_positions` against
    the step window ``[num_computed, num_computed + num_scheduled)`` — the
    exact intersection :meth:`CaptureManager.build_step_plan` performs per
    consumer (``in_step = [p for p in all_positions if step_start <= p <
    step_end]``).  It is factored out so the rank-replicated
    ``CaptureStepGate`` can decide force-eager identically on every rank
    using the same logic the capturer rank uses to gather, without
    building a step plan.  Reusing :func:`_resolve_positions` keeps the
    two in lockstep by construction.

    An unknown selector raises (matching :func:`_resolve_positions`); the
    gate treats that as "capture" so a malformed spec never silently
    skips the eager step it needs.
    """
    step_start = num_computed
    step_end = num_computed + num_scheduled
    if step_end <= step_start:
        return False
    resolved = _resolve_positions(
        positions, num_prompt_tokens, num_computed, num_scheduled
    )
    return any(step_start <= p < step_end for p in resolved)


def _classify_positions(
    spec: CaptureSpec,
    num_prompt_tokens: int,
) -> tuple[str, list[int] | None]:
    """Return ``(kind, static_positions | None)`` for a spec.

    Static kinds (``last_prompt``, ``all_prompt``, explicit list) can be
    fully resolved once at registration time.  Dynamic kinds
    (``all_generated``, ``all``) must be re-expanded each step.
    """
    positions = spec.positions

    if isinstance(positions, list):
        return "explicit", list(positions)

    if positions == "last_prompt":
        return "last_prompt", [num_prompt_tokens - 1]

    if positions == "all_prompt":
        return "all_prompt", list(range(num_prompt_tokens))

    # Dynamic: will be expanded per-step.
    return positions, None


def _filter_specs_to_layer_range(
    specs: dict[int, CaptureSpec],
    start: int,
    end: int,
) -> dict[int, CaptureSpec]:
    """Restrict every consumer spec to layers in the global ``[start, end)``.

    Returns a new mapping containing only consumers that retain at least
    one hook layer in range. Hook entries that become empty are dropped,
    and a consumer whose every hook empties out is removed entirely. The
    layer indices are global (as produced by ``make_layers``), so the
    comparison is directly against this stage's owned slice.
    """
    filtered: dict[int, CaptureSpec] = {}
    for consumer_idx, spec in specs.items():
        kept_hooks: dict[HookName, list[int]] = {}
        for hook_name, layers in spec.hooks.items():
            in_range = [layer for layer in layers if start <= layer < end]
            if in_range:
                kept_hooks[hook_name] = in_range
        if kept_hooks:
            filtered[consumer_idx] = CaptureSpec(
                hooks=kept_hooks,
                positions=spec.positions,
            )
    return filtered


# ---------------------------------------------------------------------------
# Capture manager
# ---------------------------------------------------------------------------


class CaptureManager:
    """Per-runner multi-consumer capture coordinator.

    Instantiated once per engine worker with an ordered tuple of sinks
    and their (possibly ``None``) global specs.  The manager's lifetime
    matches the runner's.
    """

    def __init__(
        self,
        consumers: tuple[CaptureSink, ...],
        consumer_specs: tuple[CaptureSpec | None, ...],
        num_hidden_layers: int,
        hidden_size: int,
        model_dtype: torch.dtype,
        device: torch.device | str = "cpu",
        max_num_tokens: int = 0,
        finalize_timeout_s: float = 5.0,
        dispatch_queue_size: int = 0,
        overload_policy: str = "block",
        spill_dir: str | None = None,
        spill_max_bytes: int = 4 << 30,
        local_layer_range: tuple[int, int] | None = None,
        graphsafe_keys: Sequence[tuple[int, str]] | None = None,
    ) -> None:
        if len(consumers) != len(consumer_specs):
            msg = (
                f"consumers length ({len(consumers)}) must match "
                f"consumer_specs length ({len(consumer_specs)})"
            )
            raise ValueError(msg)
        if overload_policy not in ("block", "drop", "spill"):
            raise ValueError(
                f"overload_policy must be 'block', 'drop', or 'spill', "
                f"got {overload_policy!r}"
            )
        self._consumers = consumers
        self._consumer_specs = consumer_specs
        # ``num_hidden_layers`` is the GLOBAL layer count (across all
        # pipeline stages); client/global specs reference global layer
        # indices and are validated against it.
        self._num_hidden_layers = num_hidden_layers
        # The global ``[start, end)`` layer slice this manager's worker
        # actually computes (its pipeline stage). Specs are filtered to
        # this range at registration so each stage captures and finalizes
        # only its own layers; ``None`` means the whole model (no PP).
        if local_layer_range is None:
            self._local_layer_range = (0, num_hidden_layers)
        else:
            start, end = local_layer_range
            if not (0 <= start <= end <= num_hidden_layers):
                raise ValueError(
                    f"local_layer_range {local_layer_range!r} must satisfy "
                    f"0 <= start <= end <= num_hidden_layers "
                    f"({num_hidden_layers})"
                )
            self._local_layer_range = (start, end)
        self._hidden_size = hidden_size
        self._model_dtype = model_dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._finalize_timeout = finalize_timeout_s
        self._requests: dict[str, _RequestCaptureState] = {}

        # ---- Global-spec persistent buffers (CUDA-graph-safe path) ----
        #
        # A *global* spec captures for every request, so the set of
        # ``(layer, hook)`` keys it touches is fixed at construction and
        # independent of any per-request client spec.  That lets the
        # gather run inside a replayed CUDA graph: instead of the dynamic
        # ``index_select`` (which allocates a fresh, variable-size output
        # each step and therefore only works eager), :meth:`on_hook`
        # issues a fixed-shape full-residual ``copy_`` of the hook's
        # ``[num_tokens, hidden]`` tensor into a persistent buffer.  That
        # copy is recorded into every cudagraph descriptor at warmup (the
        # manager is installed before ``capture_model`` runs) and replays
        # against the persistent address, so a global-only step keeps full
        # cudagraph speed.  After the forward the host slices the wanted
        # rows out of the buffer (eager, off-graph) — see
        # :meth:`_materialize_global_keys`.
        #
        # The candidate global keys are restricted to the layers this
        # pipeline stage owns (``local_layer_range``); other stages never
        # see those hooks fire.
        #
        # The graph-safe buffer path is enabled only when ``max_num_tokens
        # > 0`` (always true in the real runner, where it is the scheduler
        # token budget).  When it is unset — direct ``CaptureManager``
        # construction in CPU unit tests, where there are no CUDA graphs
        # and every step runs eager — ``_global_keys`` stays empty and
        # global specs fall through the dynamic ``index_select`` path like
        # client specs.  That keeps the eager-only behavior unchanged while
        # the runner gets the recorded full-residual copy.
        # Per-request (client) specs that tap only graph-safe-allowlisted keys
        # take the same persistent-buffer path as global specs (no force-eager).
        # The allowlist is fixed at startup so its buffers and full-residual
        # copies are baked into the CUDA graph at warmup — independent of which
        # request, if any, currently taps them. Filtered to this stage's owned
        # layers like the global candidate keys.
        start, end = self._local_layer_range
        candidate_keys: set[tuple[int, str]] = set()
        for spec in self._consumer_specs:
            if spec is None:
                continue
            for hook_name, layers in spec.hooks.items():
                for layer_idx in layers:
                    if start <= layer_idx < end:
                        candidate_keys.add((layer_idx, hook_name))

        graphsafe_in_range: set[tuple[int, str]] = set()
        for layer_idx, hook_name in graphsafe_keys or ():
            if start <= layer_idx < end:
                graphsafe_in_range.add((layer_idx, hook_name))
        candidate_keys |= graphsafe_in_range

        self._global_buffers: dict[tuple[int, str], torch.Tensor] = {}
        if candidate_keys and max_num_tokens > 0:
            for key in candidate_keys:
                self._global_buffers[key] = torch.empty(
                    (max_num_tokens, hidden_size),
                    dtype=model_dtype,
                    device=self._device,
                )
            total_bytes = sum(
                b.numel() * b.element_size() for b in self._global_buffers.values()
            )
            logger.info(
                "capture: allocated %d persistent global-capture buffer(s) "
                "(%.1f MiB total) for CUDA-graph-safe capture",
                len(self._global_buffers),
                total_bytes / (1 << 20),
                extra={
                    "num_buffers": len(self._global_buffers),
                    "total_bytes": total_bytes,
                    "max_num_tokens": max_num_tokens,
                    "hidden_size": hidden_size,
                    "global_keys": sorted(candidate_keys),
                },
            )
        elif candidate_keys:
            logger.debug(
                "capture: %d global capture key(s) present but max_num_tokens "
                "is unset; using the eager dynamic-gather path (no persistent "
                "buffers)",
                len(candidate_keys),
            )
        # Keys actually served by the buffer path. Empty unless buffers were
        # allocated, so :meth:`build_step_plan` routing and ``on_hook`` agree.
        # Includes both global-spec keys and graph-safe-allowlisted keys: both
        # are served by the persistent buffer post-forward.
        self._global_keys: frozenset[tuple[int, str]] = frozenset(self._global_buffers)
        # Subset of buffered keys that came from the graph-safe per-request
        # allowlist (may overlap with global-spec keys). Exposed so the
        # rank-replicated step gate can decide which client specs avoid eager.
        # Only populated when buffers were actually allocated.
        self._graphsafe_keys: frozenset[tuple[int, str]] = frozenset(
            graphsafe_in_range & set(self._global_buffers)
        )
        # Active plan buffered between ``build_step_plan`` (called by the
        # runner pre-forward) and ``on_hook`` fires from inside the
        # compiled forward graph.  Cleared by ``consume_step_plan`` once
        # the runner's finalize path has copied the scratch tensors out.
        self._step_plan: StepCapturePlan | None = None

        # Async dispatch path.  ``dispatch_step_captures`` issues H2D
        # copies into pinned host buffers, records a CUDA event, and
        # queues a ``_DispatchPacket``; the main runner thread returns
        # immediately.  A single dispatch thread waits on each event,
        # then fans the captured rows out to consumer sinks.  This pulls
        # the previous in-line ``cuda.synchronize()`` and per-chunk
        # construction off the model-runner critical path so they can
        # overlap with the next forward step.
        # Bounded dispatch queue is the single GPU-facing backpressure
        # point. ``dispatch_queue_size <= 0`` keeps the legacy unbounded
        # behaviour (no backpressure). ``overload_policy`` decides what
        # happens when a bounded queue is full: ``block`` stalls the
        # forward (no loss, bounded memory), ``drop`` discards the step's
        # captures (counted), ``spill`` parks them on local disk to be
        # replayed when the queue drains (implemented in the spill path).
        maxsize = dispatch_queue_size if dispatch_queue_size > 0 else 0
        self._dispatch_queue: queue.Queue[_DispatchPacket | None] = queue.Queue(
            maxsize=maxsize
        )
        self._overload_policy = overload_policy
        self._spill_max_bytes = spill_max_bytes
        self._dropped_packets = 0
        self._spilled_packets = 0
        # Spill state (``spill`` policy): overflow packets are serialized to
        # numbered files in ``_spill_dir`` and replayed by the dispatch
        # thread, in order, when the in-memory queue drains. ``_spill_pending``
        # is the FIFO of (path, nbytes) awaiting replay; ``_spill_bytes``
        # tracks on-disk usage against ``spill_max_bytes``. All under
        # ``_spill_lock``. Strict ordering invariant: while ``_spill_pending``
        # is non-empty, new packets route to spill (never the in-memory queue),
        # so the queue (older) always drains before spill (newer).
        self._spill_lock = threading.Lock()
        # Throttle overload-warning spam: log at most once per interval.
        self._last_overload_log = 0.0
        self._overload_log_interval = 5.0
        self._spill_pending: deque[tuple[pathlib.Path, int]] = deque()
        self._spill_seq = 0
        self._spill_bytes = 0
        self._spill_dir: pathlib.Path | None = None
        if overload_policy == "spill":
            base = spill_dir or os.path.join(
                tempfile.gettempdir(), "vllm-capture-spill"
            )
            self._spill_dir = pathlib.Path(base) / f"mgr-{id(self):x}"
            self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._pinned_pool: dict[tuple[int, str], list[torch.Tensor]] = {}
        self._pinned_lock = threading.Lock()
        self._pending_dispatches = 0
        self._pending_cond = threading.Condition()
        # Dedicated CUDA stream for the H2D copies in
        # ``dispatch_step_captures``.  Issuing the ``copy_`` on the
        # compute stream serialises against the next forward step;
        # placing it on a side stream lets the device pipe the transfer
        # over PCIe while the next step's kernels run.  Lazily allocated
        # on first use so non-CUDA managers (CPU tests) skip the cost.
        self._capture_stream: torch.cuda.Stream | None = None
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="vllm-capture-dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()
        # Finalize runs on its own thread so the model-runner step never
        # blocks on the per-layer ``submit_finalize`` / ``wait_for_result``
        # NFS round-trips. The step thread pops the request state and
        # enqueues a ``_FinalizeJob``; this thread drains the dispatch
        # queue (so all of the request's chunks have landed), issues the
        # finalizes, waits for results, and invokes the job callback.
        # Unbounded queue: finalize jobs are tiny control messages and must
        # never be dropped (they carry the request's terminal result).
        self._finalize_queue: queue.Queue[_FinalizeJob | None] = queue.Queue()
        self._finalize_thread = threading.Thread(
            target=self._finalize_loop,
            name="vllm-capture-finalize",
            daemon=True,
        )
        self._finalize_thread.start()

    # ------------------------------------------------------------------ props

    @property
    def num_consumers(self) -> int:
        return len(self._consumers)

    @property
    def graphsafe_keys(self) -> frozenset[tuple[int, str]]:
        """``(layer, hook)`` keys served by the graph-safe per-request path.

        These are the startup-allowlisted keys for which a persistent buffer
        was allocated, so a per-request client spec tapping only these keys
        avoids the force-eager gate. Empty when the allowlist is unset or
        ``max_num_tokens`` was 0 (CPU tests). Layer indices are global
        (model-wide), matching client-spec layer references.
        """
        return self._graphsafe_keys

    # ---------------------------------------------------------- registration

    def register_request(
        self,
        req_id: str,
        client_specs: dict[int, CaptureSpec] | None,
        num_prompt_tokens: int,
        sidecar_fields: dict[str, Any] | None = None,
        block_hashes: list[bytes] | None = None,
        hash_block_size: int = 0,
    ) -> None:
        """Register a request for capture.

        ``client_specs`` maps consumer index to a per-request spec.
        These are merged with the global specs: a client spec overrides
        the global spec for that consumer.  A consumer is active for this
        request if it has either a global spec or a client spec.

        ``block_hashes`` (the request's prompt block hashes) and
        ``hash_block_size`` (their granularity) enable activation-store
        write-through for this request; omitting them disables it.
        """
        if req_id in self._requests:
            msg = f"capture request {req_id!r} is already registered"
            raise ValueError(msg)
        if num_prompt_tokens <= 0:
            msg = (
                f"capture request {req_id!r} has non-positive "
                f"num_prompt_tokens={num_prompt_tokens}"
            )
            raise ValueError(msg)

        merged: dict[int, CaptureSpec] = {}
        for i, global_spec in enumerate(self._consumer_specs):
            if global_spec is not None:
                merged[i] = global_spec

        if client_specs:
            for i, spec in client_specs.items():
                if i < 0 or i >= len(self._consumers):
                    msg = (
                        f"client_specs key {i} out of range [0, {len(self._consumers)})"
                    )
                    raise ValueError(msg)
                merged[i] = spec

        if not merged:
            # No consumer has a spec for this request — nothing to do.
            return

        # Validate hook layers against the GLOBAL layer space (a client
        # genuinely out of range, e.g. layer 999 on a 64-layer model, is
        # rejected regardless of which pipeline stage admits it).
        for consumer_idx, spec in merged.items():
            for hook_name, layers in spec.hooks.items():
                for layer_idx in layers:
                    if layer_idx < 0 or layer_idx >= self._num_hidden_layers:
                        msg = (
                            f"capture request {req_id!r} consumer "
                            f"{consumer_idx} hook {hook_name!r} layer "
                            f"{layer_idx} is out of range "
                            f"[0, {self._num_hidden_layers})"
                        )
                        raise ValueError(msg)

        # Filter each spec to the layers this pipeline stage owns. Under
        # PP a request's layers are split across stages; each stage's
        # manager captures and finalizes only its slice, and the engine
        # merges per-stage results. Consumers left with no in-range layer
        # are inactive for this request on this rank.
        start, end = self._local_layer_range
        merged = _filter_specs_to_layer_range(merged, start, end)
        if not merged:
            # None of the requested layers live on this stage.
            return

        position_kind: dict[int, str] = {}
        static_positions: dict[int, list[int] | None] = {}
        for consumer_idx, spec in merged.items():
            kind, static = _classify_positions(spec, num_prompt_tokens)
            position_kind[consumer_idx] = kind
            static_positions[consumer_idx] = static

        state = _RequestCaptureState(
            req_id=req_id,
            consumer_specs=merged,
            position_kind=position_kind,
            static_positions=static_positions,
            num_prompt_tokens=num_prompt_tokens,
            block_hashes=block_hashes,
            hash_block_size=hash_block_size,
            sidecar_fields=dict(sidecar_fields) if sidecar_fields else {},
        )
        self._requests[req_id] = state

    def unregister_request(self, req_id: str) -> None:
        """Remove all state for ``req_id``.  Silent no-op if unknown."""
        self._requests.pop(req_id, None)

    # ------------------------------------------------------- plan building

    def build_step_plan(
        self,
        batch_view: CaptureBatchView,
    ) -> StepCapturePlan:
        """Build a :class:`StepCapturePlan` for the current batch.

        For each request in the batch that has registered consumers, each
        consumer's position selector is expanded and intersected with the
        step window.  Gather indices reflect the **union** across all
        consumers; each entry's ``consumer_mask`` records which consumers
        want it.
        """
        num_requests = len(batch_view.req_ids)
        if (
            len(batch_view.num_prompt_tokens) != num_requests
            or len(batch_view.num_computed_tokens) != num_requests
            or len(batch_view.num_scheduled_tokens) != num_requests
            or len(batch_view.token_offsets) != num_requests
        ):
            msg = (
                "CaptureBatchView list lengths must match req_ids length "
                f"(got {num_requests})"
            )
            raise ValueError(msg)

        # (layer, hook) -> list of (abs_row, entry_partial)
        # We'll build gather rows and entries together.
        gather_rows: dict[tuple[int, str], list[int]] = {}
        entries: list[CapturePositionEntry] = []
        request_errors: dict[str, str] = {}

        for i in range(num_requests):
            req_id = batch_view.req_ids[i]
            state = self._requests.get(req_id)
            if state is None:
                continue

            if state.error is not None:
                request_errors[req_id] = state.error
                continue

            num_scheduled = batch_view.num_scheduled_tokens[i]
            if num_scheduled <= 0:
                continue

            num_computed = batch_view.num_computed_tokens[i]
            token_offset = batch_view.token_offsets[i]
            step_start = num_computed
            step_end = num_computed + num_scheduled

            # Collect the union of (hook, layers, positions) across all
            # consumers for this request, tracking which consumer wants
            # each (layer, hook, logical_pos).
            #
            # consumer_positions: (layer, hook) -> {logical_pos -> mask}
            consumer_positions: dict[tuple[int, str], dict[int, int]] = defaultdict(
                dict
            )

            has_any = False
            for consumer_idx, spec in state.consumer_specs.items():
                # Resolve positions for this consumer.
                static = state.static_positions[consumer_idx]
                if static is not None:
                    all_positions = static
                else:
                    try:
                        all_positions = _resolve_positions(
                            state.position_kind[consumer_idx],
                            state.num_prompt_tokens,
                            num_computed,
                            num_scheduled,
                        )
                    except ValueError as exc:
                        err_msg = str(exc)
                        state.error = err_msg
                        request_errors[req_id] = err_msg
                        break

                # Intersect with step window.
                in_step = [p for p in all_positions if step_start <= p < step_end]
                if not in_step:
                    continue

                has_any = True
                bit = 1 << consumer_idx

                for hook_name, layers in spec.hooks.items():
                    for layer_idx in layers:
                        key: tuple[int, str] = (layer_idx, hook_name)
                        pos_map = consumer_positions[key]
                        for pos in in_step:
                            pos_map[pos] = pos_map.get(pos, 0) | bit

            # If error was set in the inner loop, skip this request.
            if state.error is not None:
                continue

            if not has_any:
                continue

            # Bump step counter.
            step_index = state.steps_seen
            state.steps_seen += 1

            # Now build gather rows and entries from the union.
            for key in sorted(consumer_positions.keys()):
                pos_map = consumer_positions[key]
                entry_layer, entry_hook = key
                rows_list = gather_rows.setdefault(key, [])
                for logical_pos in sorted(pos_map.keys()):
                    mask = pos_map[logical_pos]
                    abs_row = token_offset + (logical_pos - step_start)
                    scratch_row = len(rows_list)
                    rows_list.append(abs_row)
                    entries.append(
                        CapturePositionEntry(
                            request_id=req_id,
                            layer=entry_layer,
                            hook=cast(HookName, entry_hook),
                            logical_pos=logical_pos,
                            scratch_row=scratch_row,
                            step_index=step_index,
                            consumer_mask=mask,
                        )
                    )

        # Materialize index tensors.  ``gather_indices`` lives on the
        # model's device so ``hidden_states.index_select`` during
        # :meth:`on_hook` is a device-local op.  ``scratch_gpu`` starts
        # empty — :meth:`on_hook` populates it by storing the gathered
        # tensor directly, so there is no point pre-allocating here.
        # Split the per-key gather rows by capture path.  Global keys are
        # served from the persistent buffer post-forward
        # (``global_gather_indices``); client-only keys take the in-hook
        # dynamic ``index_select`` (``gather_indices``).  A key requested
        # by *both* a global and a client consumer this step routes to the
        # global path — the buffer holds the full residual, so any
        # consumer's rows can be sliced from it, and the per-entry
        # ``consumer_mask`` still fans the rows out to both.
        gather_indices: dict[tuple[int, str], torch.Tensor] = {}
        global_gather_indices: dict[tuple[int, str], torch.Tensor] = {}
        scratch_gpu: dict[tuple[int, str], torch.Tensor] = {}
        scratch_dtype: dict[tuple[int, str], torch.dtype] = {}
        for key, rows in gather_rows.items():
            idx = torch.tensor(rows, dtype=torch.int64, device=self._device)
            scratch_dtype[key] = self._model_dtype
            if key in self._global_keys:
                global_gather_indices[key] = idx
            else:
                gather_indices[key] = idx

        plan = StepCapturePlan(
            gather_indices=gather_indices,
            scratch_gpu=scratch_gpu,
            scratch_dtype=scratch_dtype,
            entries=entries,
            global_gather_indices=global_gather_indices,
            request_errors=request_errors,
        )
        self._step_plan = plan
        return plan

    # ---------------------------------------- runner/custom-op glue helpers

    def set_step_plan(self, plan: StepCapturePlan | None) -> None:
        """Install *plan* as the active plan without re-running the builder.

        Used by unit tests that exercise :meth:`on_hook` in isolation.
        """
        self._step_plan = plan

    def consume_step_plan(self) -> StepCapturePlan | None:
        """Return and clear the active plan.

        The runner's finalize path calls this once per forward step to
        take ownership of scratch tensors before copying them out.
        Returning ``None`` guards the next forward pass against stale
        plans.
        """
        plan = self._step_plan
        self._step_plan = None
        return plan

    def has_pending_capture(self) -> bool:
        """True if the current step's plan actually gathers any rows.

        The model runner consults this *before* the cudagraph dispatch
        decision: :meth:`on_hook` does per-step Python gathering that
        cannot run inside a replayed CUDA graph, so any step that really
        captures must execute eagerly. Returns ``False`` when no plan is
        built or the plan has no gather targets (so non-capturing steps
        keep full cudagraph speed).
        """
        plan = self._step_plan
        return plan is not None and bool(plan.gather_indices)

    def on_hook(
        self,
        layer_idx: int,
        hook_name: str,
        hidden_states: torch.Tensor,
    ) -> None:
        """Custom-op callback fired from inside the compiled forward graph.

        For any ``(layer, hook)`` key the active plan wants, gather the
        rows out of ``hidden_states`` into ``plan.scratch_gpu``.  Keys
        absent from the plan are silently skipped — the op is a no-op
        on any forward step that isn't capturing.

        The tensor passed in is the pristine residual (spec invariant
        1); we must not mutate it.
        """
        key: tuple[int, str] = (layer_idx, hook_name)

        # Global-spec path: a fixed-shape full-residual copy into a
        # persistent buffer.  Runs whenever this is a global key —
        # *independent of any per-step plan* — so it fires during warmup
        # graph capture (recording the copy into every cudagraph
        # descriptor) and on eager steps alike.  At replay no Python runs
        # but the recorded ``copy_`` reproduces against the persistent
        # buffer address; the host slices the wanted rows out afterwards
        # in :meth:`_materialize_global_keys`.  The copy is collective-free
        # (a local D2D), so adding it to only the capturer rank's graph
        # keeps the divergent per-rank graphs safe.
        buf = self._global_buffers.get(key)
        if buf is not None:
            n = hidden_states.shape[0]
            buf[:n].copy_(hidden_states)

        # Client-spec path: dynamic per-step gather.  ``index_select``
        # allocates a fresh, variable-size output each step, so it cannot
        # be replayed from a CUDA graph — it only runs eager, and the step
        # gate forces eager whenever a client-spec request captures.
        plan = self._step_plan
        if plan is None:
            return
        idx = plan.gather_indices.get(key)
        if idx is None:
            return
        gathered = hidden_states.index_select(0, idx)
        target_dtype = plan.scratch_dtype[key]
        if gathered.dtype != target_dtype:
            gathered = gathered.to(target_dtype)
        plan.scratch_gpu[key] = gathered

    # ----------------------------------------------------- dispatch

    def _materialize_global_keys(self, plan: StepCapturePlan) -> None:
        """Slice global-spec rows out of the persistent buffers to scratch.

        Runs after the forward pass (eager, off the CUDA graph).  The
        graph's recorded ``copy_`` — or the eager ``on_hook`` copy on a
        forced-eager step — has already filled each global key's
        persistent buffer with this step's full ``[num_tokens, hidden]``
        residual.  Here we gather the wanted rows in the same
        ``scratch_row`` order the plan's ``entries`` assume (the order
        ``build_step_plan`` appended them), so from this point on the
        dispatch path handles global keys exactly like client keys.

        The ``index_select`` allocates a fresh output (independent of the
        buffer), so the next step's forward may overwrite the buffer
        without racing this step's already-copied-out rows.  It is issued
        on the current (compute) stream after the forward, so it observes
        the buffer the forward wrote.
        """
        if not plan.global_gather_indices:
            return
        for key, idx in plan.global_gather_indices.items():
            buf = self._global_buffers.get(key)
            if buf is None:
                # Defensive: a global gather index with no buffer would
                # mean a key escaped the construction-time global set.
                # Skip rather than crash; entries for this key yield no
                # rows downstream.
                continue
            plan.scratch_gpu[key] = buf.index_select(0, idx)

    def dispatch_step_captures(self, plan: StepCapturePlan) -> None:
        """Hand a finished step's scratch tensors to the dispatch thread.

        Issues a non-blocking H2D copy of every ``scratch_gpu`` tensor
        into a pinned host buffer, records a CUDA event that fires when
        all copies complete, and queues a ``_DispatchPacket`` for the
        dispatch thread to consume.  Returns immediately — the
        ``cuda.synchronize`` and per-chunk fan-out happen off the main
        runner thread so they overlap with the next forward step.

        Pinned destinations are required for ``copy_(non_blocking=True)``
        to deliver an asynchronous transfer; without pinning CUDA falls
        back to a synchronous staged copy.

        Each consumer's dispatch is wrapped in try/except inside the
        dispatch loop, so a failure in one sink never blocks delivery
        to the others.
        """
        # Pull global-spec rows out of the persistent buffers into
        # ``scratch_gpu`` first, so the rest of this method treats global
        # and client keys uniformly. No-op when no global key captured.
        self._materialize_global_keys(plan)

        if not plan.entries:
            return

        scratch_pinned: dict[
            tuple[int, str], tuple[torch.Tensor | None, torch.Tensor]
        ] = {}
        cuda_event: torch.cuda.Event | None = None

        has_cuda = any(s.is_cuda for s in plan.scratch_gpu.values())
        if has_cuda:
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=self._device)
            compute_stream = torch.cuda.current_stream()
            # Make the side stream wait for the union-gather kernels on
            # the compute stream to finish writing ``scratch`` before we
            # start copying.  Without this we'd race the producer.
            self._capture_stream.wait_stream(compute_stream)
            with torch.cuda.stream(self._capture_stream):
                for key, scratch in plan.scratch_gpu.items():
                    if scratch.is_cuda:
                        rows, hidden = scratch.shape
                        pinned = self._acquire_pinned(key, rows, hidden, scratch.dtype)
                        view = pinned.narrow(0, 0, rows)
                        # ``record_stream`` keeps the caching allocator
                        # from recycling ``scratch`` until this stream
                        # finishes the copy.  Cheap; without it we risk
                        # use-after-free when the runner immediately
                        # reuses the same scratch slot next step.
                        scratch.record_stream(self._capture_stream)
                        view.copy_(scratch, non_blocking=True)
                        scratch_pinned[key] = (pinned, view)
                    else:
                        scratch_pinned[key] = (None, scratch)
                cuda_event = torch.cuda.Event()
                cuda_event.record(stream=self._capture_stream)
        else:
            for key, scratch in plan.scratch_gpu.items():
                scratch_pinned[key] = (None, scratch)

        packet = _DispatchPacket(
            entries=list(plan.entries),
            scratch_pinned=scratch_pinned,
            cuda_event=cuda_event,
        )
        self._enqueue_packet(packet)

    def _release_packet_buffers(self, packet: _DispatchPacket) -> None:
        """Return a packet's pinned host buffers to the pool.

        Called when a packet is discarded (``drop`` policy) so dropping a
        step's captures does not leak pinned memory.
        """
        for key, (pinned, _view) in packet.scratch_pinned.items():
            if pinned is not None:
                self._release_pinned(key, pinned)

    def _enqueue_packet(self, packet: _DispatchPacket) -> None:
        """Hand a packet to the dispatch thread under the overload policy.

        ``block`` (or an unbounded queue): ``put`` blocks until there is
        room, propagating backpressure to the forward pass. ``drop``: try
        non-blocking, and on a full queue discard the packet (counted) so
        serving never stalls. ``spill``: park the packet on local disk when
        the queue is full and replay it when the queue drains; falls back to
        ``block`` if the spill area is exhausted.
        """
        policy = self._overload_policy
        if policy == "block" or self._dispatch_queue.maxsize == 0:
            with self._pending_cond:
                self._pending_dispatches += 1
            self._dispatch_queue.put(packet)
            return

        # Non-blocking policies: count the in-flight packet first, then try
        # to enqueue; undo the count if we end up discarding it.
        with self._pending_cond:
            self._pending_dispatches += 1

        # Ordering invariant for ``spill``: once any packet has spilled,
        # route every new packet to spill (even if the in-memory queue has
        # since drained) until spill is fully empty. Otherwise a new packet
        # could ``put_nowait`` onto the queue and jump ahead of older spilled
        # packets, which the dispatch loop drains only after the queue.
        if policy == "spill":
            with self._spill_lock:
                spill_active = bool(self._spill_pending)
            if spill_active:
                self._spill_packet(packet)
                return

        try:
            self._dispatch_queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        if policy == "drop":
            self._release_packet_buffers(packet)
            with self._pending_cond:
                self._pending_dispatches -= 1
                self._dropped_packets += 1
                if self._pending_dispatches == 0:
                    self._pending_cond.notify_all()
            self._log_overload("drop")
            return

        # policy == "spill": the queue is full -> spill this packet to disk.
        self._spill_packet(packet)

    def _spill_packet(self, packet: _DispatchPacket) -> None:
        """Serialize an overflow packet to the spill area (``spill`` policy).

        ``_pending_dispatches`` is already incremented by the caller, and
        stays counted until the dispatch loop replays this packet — so
        ``_drain_dispatch_queue`` (and thus finalize) waits for spilled data
        to reach consumers, never losing it. If the spill area is at its cap,
        blocks until the dispatch loop frees room (degrades to ``block``).
        """
        data = self._serialize_packet(packet)
        # Bytes are captured; release the pinned host buffers now.
        self._release_packet_buffers(packet)
        n = len(data)
        while True:
            with self._spill_lock:
                if (
                    self._spill_bytes + n <= self._spill_max_bytes
                    or not self._spill_pending
                ):
                    # Accept when under cap, or when spill is empty (a single
                    # packet larger than the cap still goes through rather
                    # than deadlocking).
                    path = self._spill_dir / f"spill-{self._spill_seq:012d}.pkt"
                    self._spill_seq += 1
                    self._spill_bytes += n
                    self._spill_pending.append((path, n))
                    self._spilled_packets += 1
                    break
            # Spill area full: wait for the dispatch loop to drain some.
            time.sleep(0.01)
        path.write_bytes(data)
        self._log_overload("spill")

    def _log_overload(self, kind: str) -> None:
        """Throttled warning so the policy is visible on a live server."""
        now = time.monotonic()
        if now - self._last_overload_log < self._overload_log_interval:
            return
        self._last_overload_log = now
        logger.warning(
            "capture overload: dispatch queue full, policy=%s "
            "(dropped=%d, spilled=%d, spill_backlog=%.1f MiB)",
            kind,
            self._dropped_packets,
            self._spilled_packets,
            self._spill_bytes / (1 << 20),
        )

    def _serialize_packet(self, packet: _DispatchPacket) -> bytes:
        """Serialize a packet's entries + CPU scratch tensors to bytes."""
        scratch = {key: view for key, (_owner, view) in packet.scratch_pinned.items()}
        buf = io.BytesIO()
        torch.save({"entries": packet.entries, "scratch": scratch}, buf)
        return buf.getvalue()

    def _deserialize_packet(self, data: bytes) -> _DispatchPacket:
        """Rebuild a packet from spill bytes (CPU scratch, no pinned owner)."""
        obj = torch.load(io.BytesIO(data), weights_only=False)
        scratch_pinned = {key: (None, view) for key, view in obj["scratch"].items()}
        return _DispatchPacket(
            entries=obj["entries"],
            scratch_pinned=scratch_pinned,
            cuda_event=None,
        )

    def _next_spilled_packet(self) -> _DispatchPacket | None:
        """Pop and reload the oldest spilled packet, or ``None`` if empty."""
        with self._spill_lock:
            if not self._spill_pending:
                return None
            path, n = self._spill_pending.popleft()
            self._spill_bytes -= n
        try:
            data = path.read_bytes()
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
        return self._deserialize_packet(data)

    @property
    def dropped_packets(self) -> int:
        """Number of capture steps discarded under the ``drop`` policy."""
        return self._dropped_packets

    @property
    def spilled_packets(self) -> int:
        """Number of capture steps spilled to disk under the ``spill`` policy."""
        return self._spilled_packets

    # --------------------------------------------------- pinned-pool helpers

    def _acquire_pinned(
        self,
        key: tuple[int, str],
        rows: int,
        hidden: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Lease a pinned host buffer of at least ``rows × hidden``.

        Reuses a buffer from the per-key pool if one is large enough and
        has the right dtype/width; allocates fresh otherwise.  Allocations
        round capacity up to a 16-row boundary so a request whose row
        count nudges by one doesn't trigger a fresh allocation.
        """
        with self._pinned_lock:
            free = self._pinned_pool.setdefault(key, [])
            for i, buf in enumerate(free):
                if (
                    buf.shape[0] >= rows
                    and buf.shape[1] == hidden
                    and buf.dtype == dtype
                ):
                    return free.pop(i)
        capacity = ((rows + 15) // 16) * 16
        return torch.empty((capacity, hidden), dtype=dtype, pin_memory=True)

    def _release_pinned(
        self,
        key: tuple[int, str],
        buf: torch.Tensor,
    ) -> None:
        """Return a pinned buffer to its per-key pool.

        Pool size is capped per key so a long-lived manager doesn't
        pin unbounded host memory if the workload thrashes between
        large and small requests.
        """
        with self._pinned_lock:
            free = self._pinned_pool.setdefault(key, [])
            if len(free) < 4:
                free.append(buf)

    # ------------------------------------------------------- dispatch thread

    def _dispatch_loop(self) -> None:
        """Background thread that drains ``_dispatch_queue``.

        Started once per manager.  Each iteration:

        1. Pop a packet (blocks).  ``None`` is the shutdown sentinel.
        2. Wait on the packet's CUDA event so the H2D copies are
           guaranteed visible on the host.
        3. Fan out the now-CPU-resident rows to every consumer that
           wants them.
        4. Return the pinned buffers to their pools and decrement the
           pending-dispatches counter, notifying any caller waiting in
           :meth:`_drain_dispatch_queue`.
        """
        while True:
            try:
                # Short timeout so that, when the in-memory queue drains while
                # spilled packets are still pending, we pick them up promptly
                # (the producer can't wake a blocked get() — the queue is full
                # precisely when it spills).
                packet = self._dispatch_queue.get(timeout=0.1)
            except queue.Empty:
                spilled = self._next_spilled_packet()
                if spilled is not None:
                    self._process_packet(spilled)
                continue
            if packet is None:
                # Shutdown sentinel: drain any remaining spill first so no
                # spilled captures are lost on shutdown.
                while True:
                    s = self._next_spilled_packet()
                    if s is None:
                        break
                    self._process_packet(s)
                return
            # Live queue items are always older than spilled items (the
            # ordering invariant routes new packets to spill while spill is
            # active), so process the queue before touching spill.
            self._process_packet(packet)

    def _process_packet(self, packet: _DispatchPacket) -> None:
        """Sync the packet's H2D copies, fan out to sinks, recycle buffers."""
        try:
            if packet.cuda_event is not None:
                packet.cuda_event.synchronize()
            self._fan_out_to_consumers(packet)
            self._write_through_to_store(packet)
        except Exception:
            logger.exception("capture dispatch loop error")
        finally:
            self._release_packet_buffers(packet)
            with self._pending_cond:
                self._pending_dispatches -= 1
                if self._pending_dispatches == 0:
                    self._pending_cond.notify_all()

    def serve_from_store(
        self, req_id: str, payload: dict[tuple[int, str, int], torch.Tensor]
    ) -> None:
        """Inject store-served prompt residuals as capture chunks.

        Called once at registration when the scheduler reserved a
        whole-prefix store serve: the prompt positions were reused from the
        KV cache and not forwarded, so for each consumer active for ``req_id``
        we assemble its requested ``(layer, hook)`` rows from ``payload``
        (keyed by ``(layer, hook, position)``) and submit them to its sink
        exactly like a forward-path chunk. Generated positions still flow
        through the normal forward path. Runs on the registration (main)
        thread; sinks lock ``submit_chunk``, and a request's serve precedes
        its own forward dispatch, so there is no same-request race.
        """
        state = self._requests.get(req_id)
        if state is None:
            return
        rid = VllmInternalRequestId(req_id)
        for consumer_idx, spec in state.consumer_specs.items():
            positions = captured_prompt_positions(spec, state.num_prompt_tokens)
            if not positions:
                continue
            sink = self._consumers[consumer_idx]
            try:
                for hook, layers in spec.hooks.items():
                    for layer in layers:
                        rows: list[torch.Tensor] = []
                        kept: list[int] = []
                        for pos in positions:
                            row = payload.get((layer, hook, pos))
                            if row is not None:
                                rows.append(row)
                                kept.append(pos)
                        if not rows:
                            continue
                        tensor = torch.stack(rows, dim=0)
                        sink.submit_chunk(
                            CaptureChunk(
                                key=(rid, layer, hook),
                                tensor=tensor,
                                dtype=tensor.dtype,
                                row_offset=0,
                                step_index=0,
                                metadata={
                                    "consumer_index": consumer_idx,
                                    "positions": kept,
                                    "served_from_store": True,
                                },
                            )
                        )
            except Exception:
                logger.exception(
                    "Consumer %d raised during store serve; other consumers "
                    "are unaffected.",
                    consumer_idx,
                )
                if state.error is None:
                    state.error = f"consumer {consumer_idx} store serve failed"

    def _write_through_to_store(self, packet: _DispatchPacket) -> None:
        """Populate the activation store with freshly-captured prompt rows.

        Runs once per packet on the dispatch thread, after fan-out and
        before the pinned buffers are recycled. The pristine residual at a
        prompt position is content-addressable (a pure function of the
        prefix), so storing it lets a later request sharing the prefix serve
        it without re-forwarding. Only prompt positions are stored —
        generated positions are not shared across requests — and each
        ``(request, layer, hook, position)`` is stored once regardless of
        how many consumers wanted it, since the residual is
        consumer-independent. Rows are cloned off the recycled pinned
        buffer. A no-op when no store is installed.
        """
        store = get_active_activation_store()
        if store is None:
            return
        seen: set[tuple[str, int, str, int]] = set()
        for entry in packet.entries:
            pos = entry.logical_pos
            dedup = (entry.request_id, entry.layer, entry.hook, pos)
            if dedup in seen:
                continue
            seen.add(dedup)
            state = self._requests.get(entry.request_id)
            if state is None or state.block_hashes is None:
                continue
            if pos >= state.num_prompt_tokens:
                continue
            key = activation_key(
                state.block_hashes,
                state.hash_block_size,
                pos,
                entry.layer,
                entry.hook,
            )
            if key is None:
                continue
            pinned_view = packet.scratch_pinned.get((entry.layer, entry.hook))
            if pinned_view is None:
                continue
            _pinned, view = pinned_view
            store.put(key, view[entry.scratch_row].clone())

    def _fan_out_to_consumers(self, packet: _DispatchPacket) -> None:
        """Walk consumers and submit chunks for ``packet`` (dispatch thread).

        Same per-(consumer × request × layer × hook) shape the inline
        path used to have, but reading from the packet's pinned host
        views instead of touching GPU memory.  Each consumer's submit
        loop is isolated by try/except so one failing sink doesn't
        block delivery to the others.
        """
        for consumer_idx, sink in enumerate(self._consumers):
            bit = 1 << consumer_idx

            grouped: dict[tuple[str, int, str], list[CapturePositionEntry]] = (
                defaultdict(list)
            )
            for entry in packet.entries:
                if entry.consumer_mask & bit:
                    grouped_key = (entry.request_id, entry.layer, entry.hook)
                    grouped[grouped_key].append(entry)

            if not grouped:
                continue

            try:
                # Build every chunk for this consumer's slice of the step,
                # then hand them over in one batch call. Batching lets sinks
                # amortize per-chunk overhead (locking, write-task creation,
                # payload concatenation) across the whole step instead of
                # paying it per (layer, hook) — the dominant cost when a
                # request captures many layers per step.
                chunks: list[CaptureChunk] = []
                for (req_id, layer, hook), chunk_entries in grouped.items():
                    scratch_key = (layer, hook)
                    if scratch_key not in packet.scratch_pinned:
                        continue
                    _pinned, view = packet.scratch_pinned[scratch_key]

                    row_indices = [e.scratch_row for e in chunk_entries]
                    idx_tensor = torch.tensor(row_indices, dtype=torch.long)
                    chunk_tensor = view.index_select(0, idx_tensor)

                    step_index = chunk_entries[0].step_index
                    capture_key = (
                        VllmInternalRequestId(req_id),
                        layer,
                        hook,
                    )
                    chunks.append(
                        CaptureChunk(
                            key=capture_key,
                            tensor=chunk_tensor,
                            dtype=chunk_tensor.dtype,
                            row_offset=0,
                            step_index=step_index,
                            metadata={
                                "consumer_index": consumer_idx,
                                "positions": [e.logical_pos for e in chunk_entries],
                            },
                        )
                    )

                if chunks:
                    batch_submit = getattr(sink, "submit_chunk_batch", None)
                    if batch_submit is not None:
                        batch_submit(chunks)
                    else:
                        for chunk in chunks:
                            sink.submit_chunk(chunk)
            except Exception:
                logger.exception(
                    "Consumer %d raised during dispatch; "
                    "other consumers are unaffected.",
                    consumer_idx,
                )
                # Record error for each request this consumer was handling.
                # ``_requests`` is touched by the main thread (register) and
                # the dispatch thread here; in-place ``error`` assignment
                # is safe under the GIL given the simple read-modify-write
                # pattern, and we only set the field if it's still ``None``.
                for req_id_key in {k[0] for k in grouped}:
                    s = self._requests.get(req_id_key)
                    if s is not None and s.error is None:
                        s.error = f"consumer {consumer_idx} dispatch failed"

    def _drain_dispatch_queue(self) -> None:
        """Block the calling thread until the dispatch queue is empty.

        Called from :meth:`finalize_request` so that all per-step chunks
        for a request have been submitted to the consumer's sink before
        we issue ``submit_finalize`` and start waiting on results.
        Without this barrier a finalize could race ahead of late chunks
        still queued for the dispatch thread.
        """
        with self._pending_cond:
            while self._pending_dispatches > 0:
                self._pending_cond.wait()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain the dispatch queue and stop the dispatch thread.

        Idempotent: safe to call multiple times.  Held back from being
        a destructor because ``__del__`` ordering during interpreter
        shutdown is not reliable for thread joins.
        """
        if not self._dispatch_thread.is_alive():
            return
        # Stop the finalize thread first: pending finalize jobs drain the
        # dispatch queue, so the dispatch thread must still be alive while
        # they run. The sentinel is processed only after all queued jobs.
        if self._finalize_thread.is_alive():
            self._finalize_queue.put(None)
            self._finalize_thread.join(timeout=timeout)
        self._drain_dispatch_queue()
        self._dispatch_queue.put(None)
        self._dispatch_thread.join(timeout=timeout)
        if self._dropped_packets or self._spilled_packets:
            logger.info(
                "capture overload summary: %d dropped, %d spilled (policy=%s)",
                self._dropped_packets,
                self._spilled_packets,
                self._overload_policy,
            )
        # Best-effort cleanup of the spill scratch directory.
        if self._spill_dir is not None:
            import shutil

            with contextlib.suppress(OSError):
                shutil.rmtree(self._spill_dir, ignore_errors=True)

    # ----------------------------------------------------- finalization

    def finalize_request(self, req_id: str) -> dict[int, CaptureResult]:
        """Synchronously finalize capture for a request across consumers.

        Drains in-flight dispatches, pops the request, then issues
        ``submit_finalize`` and waits for every key's terminal result on
        the calling thread.  Retained for tests and any caller that wants
        the result inline; the model runner uses
        :meth:`finalize_request_async` to keep finalize off the step
        thread.  Returns a dict mapping consumer index to ``CaptureResult``
        (empty if the request was never registered).
        """
        self._drain_dispatch_queue()
        state = self._requests.pop(req_id, None)
        if state is None:
            return {}
        return self._run_finalize(req_id, state)

    def finalize_request_async(
        self,
        req_id: str,
        on_complete: Callable[[dict[int, CaptureResult]], None],
    ) -> bool:
        """Finalize *req_id* off the model-runner step thread.

        Pops the request's capture state on the **caller's** thread (so
        ``_requests`` stays single-threaded) and hands the blocking
        finalize work — draining the dispatch queue, ``submit_finalize``,
        and the per-key ``wait_for_result`` NFS round-trips — to the
        dedicated finalize thread.  ``on_complete`` is invoked there with
        the aggregated ``{consumer_index: CaptureResult}`` once finalize
        finishes.  Returns ``False`` if the request was never registered
        (caller should skip), ``True`` once the job is enqueued.

        Best-effort by design: the caller is free to proceed (and the
        request's text output to be emitted) before ``on_complete`` runs,
        so the result may attach to a later step's output or not reach the
        client at all.  The captured activations on disk are unaffected.
        """
        state = self._requests.pop(req_id, None)
        if state is None:
            return False
        self._finalize_queue.put(
            _FinalizeJob(req_id=req_id, state=state, on_complete=on_complete)
        )
        return True

    def _finalize_loop(self) -> None:
        """Background thread draining ``_finalize_queue``.

        Each job: drain the dispatch queue so all of the request's chunks
        have been submitted to the sinks, run the (blocking) finalize, then
        invoke the job callback.  ``None`` is the shutdown sentinel.
        """
        while True:
            job = self._finalize_queue.get()
            if job is None:
                return
            try:
                self._drain_dispatch_queue()
                results = self._run_finalize(job.req_id, job.state)
            except Exception:
                logger.exception("capture finalize failed for req=%s", job.req_id)
                results = {}
            try:
                job.on_complete(results)
            except Exception:
                logger.exception(
                    "capture finalize callback failed for req=%s", job.req_id
                )

    def _run_finalize(
        self, req_id: str, state: _RequestCaptureState
    ) -> dict[int, CaptureResult]:
        """Issue ``submit_finalize`` + ``wait_for_result`` for every key.

        Pure work over an already-popped *state* — no dispatch-queue drain
        and no ``_requests`` access — so it runs identically on the calling
        thread (sync path) or the finalize thread (async path).
        """
        results: dict[int, CaptureResult] = {}

        for consumer_idx, spec in state.consumer_specs.items():
            sink = self._consumers[consumer_idx]

            # Build per-consumer sidecar from request-level sidecar
            # fields plus consumer index.
            sidecar = dict(state.sidecar_fields)
            sidecar["consumer_index"] = consumer_idx

            capture_keys: list[CaptureKey] = []
            for hook_name, layers in spec.hooks.items():
                for layer_idx in layers:
                    capture_key: CaptureKey = (
                        VllmInternalRequestId(req_id),
                        layer_idx,
                        hook_name,
                    )
                    capture_keys.append(capture_key)
                    finalize = CaptureFinalize(
                        key=capture_key,
                        sidecar=sidecar,
                    )
                    try:
                        sink.submit_finalize(finalize)
                    except Exception:
                        logger.exception(
                            "Consumer %d submit_finalize failed for %s",
                            consumer_idx,
                            capture_key,
                        )

            if capture_keys:
                per_key_results: list[CaptureResult] = []
                for capture_key in capture_keys:
                    try:
                        result = sink.wait_for_result(
                            capture_key,
                            timeout=self._finalize_timeout,
                        )
                    except Exception:
                        logger.exception(
                            "Consumer %d wait_for_result failed for %s",
                            consumer_idx,
                            capture_key,
                        )
                        result = None

                    if result is None:
                        result = CaptureResult(
                            key=capture_key,
                            status="error",
                            error=f"finalize timed out for {capture_key}",
                        )
                    per_key_results.append(result)

                results[consumer_idx] = _aggregate_capture_results(per_key_results)
            else:
                # No hooks in spec — unusual but not impossible.
                dummy_key = (
                    VllmInternalRequestId(req_id),
                    0,
                    "post_block",
                )
                results[consumer_idx] = CaptureResult(
                    key=dummy_key,
                    status="not_requested",
                )

        return results

    # ----------------------------------------------------- error recording

    def record_request_error(self, req_id: str, message: str) -> None:
        """Record a terminal error for ``req_id``."""
        state = self._requests.get(req_id)
        if state is not None:
            state.error = message

    # ----------------------------------------------------- queries

    def is_active(self) -> bool:
        """True if any requests are registered."""
        return bool(self._requests)

    def has_request(self, req_id: str) -> bool:
        """True if ``req_id`` is registered."""
        return req_id in self._requests


def _aggregate_capture_results(results: list[CaptureResult]) -> CaptureResult:
    """Reduce per-key capture results into one per-consumer result."""
    if not results:
        raise ValueError("results must not be empty")

    worst_severity = max(_CAPTURE_RESULT_SEVERITY[r.status] for r in results)
    representative = next(
        result
        for result in results
        if _CAPTURE_RESULT_SEVERITY[result.status] == worst_severity
    )
    errors = [result.error for result in results if result.error]

    if len(results) == 1:
        payload: Any = results[0].payload
    else:
        payload = {result.key: result.payload for result in results}

    return CaptureResult(
        key=representative.key,
        status=representative.status,
        error="; ".join(errors) if errors else None,
        payload=payload,
    )


def merge_capture_results(
    outputs: Sequence[Any | None],
    output_rank: int,
) -> None:
    """Union every rank's ``capture_results`` into ``outputs[output_rank]``.

    Mutates ``outputs[output_rank].capture_results`` in place. Under
    pipeline parallelism a request's captured layers are split across
    stages, so each stage's TP-rank-0 worker produces capture results for
    the layers it owns. Only ``outputs[output_rank]`` (TP rank 0 of the
    last stage) is returned to the engine, so the other ranks' results are
    folded in here: grouped by ``(request_id, consumer_name)`` and reduced
    with the same worst-of-status precedence ``finalize`` uses within a
    rank. ``outputs`` is duck-typed on ``.capture_results`` to avoid a
    hard import of ``ModelRunnerOutput`` into this module.

    A request whose layers all live on one rank contributes exactly one
    result for that ``(request, consumer)`` and passes through unchanged.
    """
    target = outputs[output_rank] if 0 <= output_rank < len(outputs) else None
    if target is None:
        return

    # request_id -> consumer_name -> list[CaptureResult]
    collected: dict[str, dict[str, list[CaptureResult]]] = {}
    for out in outputs:
        if out is None:
            continue
        for req_id, per_consumer in out.capture_results.items():
            req_bucket = collected.setdefault(req_id, {})
            for name, result in per_consumer.items():
                req_bucket.setdefault(name, []).append(result)

    merged: dict[str, dict[str, CaptureResult]] = {}
    for req_id, per_consumer in collected.items():
        merged[req_id] = {
            name: (
                results[0] if len(results) == 1 else _aggregate_capture_results(results)
            )
            for name, results in per_consumer.items()
        }
    target.capture_results = merged


__all__ = [
    "CaptureManager",
    "_aggregate_capture_results",
    "merge_capture_results",
]
