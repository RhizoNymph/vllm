# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation-capture manager, step plan, and custom op.

This module is the Phase 3 core of the per-request activation-storing
feature documented in ``docs/features/activation_storing.md``.
Responsibilities:

- Hold per-request capture state (resolved hooks, resolved positions,
  registration-time errors).
- Build a :class:`StepCapturePlan` against a lightweight
  :class:`CaptureBatchView` on every forward step, translating absolute
  batch positions into per-``(layer, hook)`` ``gather_indices`` tensors.
- Expose ``torch.ops.vllm.capture_residual`` as a custom op registered
  via :func:`vllm.utils.torch_utils.direct_register_custom_op` plus an
  FX-friendly fake impl.
- Expose :func:`maybe_capture_residual`, a ``None``-check gate that
  ``vllm.model_executor.layers.steering.apply_layer_steering`` calls
  **before** adding the steering vector. Under ``torch.compile`` with no
  active manager, the gate constant-folds and the custom op never
  enters the compiled graph (spec invariant 3).

The manager is a pure in-process component: Phase 4 will instantiate it
on the model runner and wire it to the writer thread pool. Phase 3 only
ships the plumbing and exercises it via direct CPU unit tests.
"""

from __future__ import annotations

import datetime
import pathlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.config.activation_storing_types import CaptureResult, resolve_positions
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.entrypoints.openai.activation_storing_validation import (
        ResolvedActivationStoringSpec,
    )
    from vllm.v1.worker.activation_writer import (
        ActivationWriter,
        CaptureKey,
        WriteResult,
    )


# ---------------------------------------------------------------------------
# Hook-name / hook-id encoding
# ---------------------------------------------------------------------------
#
# The custom op takes an ``int hook_id`` rather than a string because
# ``torch.library`` cannot serialize Python strings across the compiled
# boundary. The manager internally stores capture state keyed by
# ``(layer_idx, hook_name)`` so the lookup tables below translate between
# the two representations.

_HOOK_NAME_TO_ID: dict[str, int] = {
    "pre_attn": 0,
    "post_attn": 1,
    "post_mlp": 2,
    "mlp_in": 3,
    "mlp_out": 4,
}
_HOOK_ID_TO_NAME: dict[int, str] = {v: k for k, v in _HOOK_NAME_TO_ID.items()}


# ---------------------------------------------------------------------------
# Module-global active manager
# ---------------------------------------------------------------------------
#
# Phase 4 will call :func:`set_active_capture_manager` from the gpu model
# runner once per worker. Phase 3 uses the same setter from tests to
# install fake/real managers. ``None`` means the cold path is active and
# the custom op is constant-folded out of compiled graphs.

_ACTIVE_CAPTURE_MANAGER: ActivationCaptureManager | None = None


def set_active_capture_manager(mgr: ActivationCaptureManager | None) -> None:
    """Install ``mgr`` as the process-global active capture manager.

    Passing ``None`` disables capture entirely and restores the cold
    path. Subsequent calls to :func:`maybe_capture_residual` become
    no-ops that ``torch.compile`` constant-folds away.
    """
    global _ACTIVE_CAPTURE_MANAGER
    _ACTIVE_CAPTURE_MANAGER = mgr


def get_active_capture_manager() -> ActivationCaptureManager | None:
    """Return the currently installed capture manager, if any."""
    return _ACTIVE_CAPTURE_MANAGER


# ---------------------------------------------------------------------------
# Dataclasses describing the per-step plan and batch view
# ---------------------------------------------------------------------------


@dataclass
class CapturePositionEntry:
    """One scratch row's worth of capture metadata.

    Phase 4 will walk ``StepCapturePlan.entries`` after the forward pass
    and materialize a :class:`vllm.v1.worker.activation_writer.WriteTask`
    per chunk of rows sharing the same ``(request_id, layer, hook)``.
    """

    request_id: str
    layer: int
    hook: str  # "pre_attn" | "post_attn" | "post_mlp"
    logical_pos: int
    scratch_row: int
    step_index: int


@dataclass
class StepCapturePlan:
    """Snapshot of everything the custom op needs for one forward step.

    ``gather_indices`` holds absolute batch-row indices that
    :meth:`ActivationCaptureManager.on_hook` will ``index_select`` out of
    the hidden-state tensor. One entry per ``(layer, hook)`` pair that at
    least one active request wants this step. ``(layer, hook)`` pairs
    nobody wants have no entry and no scratch allocation (spec § Forward
    pass).

    ``scratch_gpu`` is written in-place by ``on_hook``. Phase 4 reads it
    on the finalize path to stage a pinned-CPU copy.

    ``scratch_dtype`` is baked in at plan-build time so the capture path
    never has to look at the model config during the forward pass.

    ``entries`` is a flat list of per-row metadata. The order is stable:
    entries for one ``(layer, hook)`` key are contiguous and their
    ``scratch_row`` values are ``[0, 1, 2, ...]`` within that key.

    ``request_errors`` surfaces admission-time or registration-time
    failures (e.g., resolution of ``"all_generated"`` positions that no
    longer make sense) so Phase 4 can bubble them onto the owning
    request's capture status.
    """

    gather_indices: dict[tuple[int, str], torch.Tensor]
    scratch_gpu: dict[tuple[int, str], torch.Tensor]
    scratch_dtype: dict[tuple[int, str], torch.dtype]
    entries: list[CapturePositionEntry]
    request_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class CaptureBatchView:
    """Minimal view of an ``InputBatch`` that the manager understands.

    Phase 3 explicitly does NOT import from
    ``vllm.v1.worker.gpu_input_batch``. Keeping the view decoupled lets
    us unit-test the manager entirely on CPU without any v1 scheduler
    plumbing, and lets Phase 4 populate it from whatever the runner has
    handy.

    Semantics (all lists are in the same order, indexed by absolute
    batch row 0..N-1):

    - ``req_ids``: one entry per active request.
    - ``num_prompt_tokens[i]``: the prompt length of request ``i``.
    - ``num_computed_tokens[i]``: number of tokens for request ``i`` that
      have already been forwarded before this step. ``0`` means this
      step is the prefill.
    - ``num_scheduled_tokens[i]``: number of tokens for request ``i``
      being forwarded in this step.
    - ``token_offsets[i]``: absolute row index into the flat batched
      hidden-state tensor where request ``i``'s scheduled tokens begin.
    """

    req_ids: list[str]
    num_prompt_tokens: list[int]
    num_computed_tokens: list[int]
    num_scheduled_tokens: list[int]
    token_offsets: list[int]


# ---------------------------------------------------------------------------
# Per-request state held inside the manager
# ---------------------------------------------------------------------------


@dataclass
class _RequestCaptureState:
    """Internal bookkeeping for one registered capture request.

    ``resolved_hooks`` is a copy of the resolved hook → layers mapping
    from :class:`ResolvedActivationStoringSpec`. ``position_kind`` mirrors
    the spec field. ``static_positions`` holds the fully resolved absolute
    indices for ``last_prompt`` / ``all_prompt`` / explicit-list selectors;
    it is ``None`` for ``all_generated`` / ``all`` which grow per step.

    ``error`` holds any registration-time or step-time error; when set,
    :meth:`ActivationCaptureManager.build_step_plan` surfaces it through
    ``StepCapturePlan.request_errors`` and skips planning for the
    request until it is cleared by ``unregister_request``.

    ``steps_seen`` counts how many ``build_step_plan`` calls have
    produced at least one entry for this request. It is used for the
    ``step_index`` field on :class:`CapturePositionEntry`, which Phase 4
    uses to preserve append ordering across decode steps.

    Sidecar provenance fields (``request_id_slug`` / ``tag_slug`` /
    ``model_name`` / ``model_dtype_str`` / ``element_size_bytes`` /
    ``vllm_internal_request_id`` / ``prompt_token_ids`` / ``created_at``)
    are captured at admission time so :meth:`finalize_request` can
    assemble the sidecar without reaching back into runner state.
    ``captured_positions`` accumulates the absolute logical indices that
    have actually been routed into scratch (one list per
    ``(layer, hook)`` key) so the sidecar reflects the real rows that
    were written — not whatever the spec asked for, which may diverge in
    ``"all_generated"``/``"all"`` modes if the request is cut short.
    """

    request_id: str
    resolved_hooks: dict[str, list[int]]
    position_kind: str
    static_positions: list[int] | None
    num_prompt_tokens: int
    error: str | None = None
    steps_seen: int = 0

    # Phase 4 additive fields ------------------------------------------------
    request_id_slug: str = ""
    tag_slug: str = ""
    model_name: str = ""
    model_dtype_str: str = ""
    element_size_bytes: int = 0
    vllm_internal_request_id: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    created_at: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    # (layer, hook) -> absolute positions captured so far, in row order.
    captured_positions: dict[tuple[int, str], list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capture manager
# ---------------------------------------------------------------------------


class ActivationCaptureManager:
    """Per-runner capture state + plan builder.

    The manager is a process-local object installed via
    :func:`set_active_capture_manager`. It stores one
    :class:`_RequestCaptureState` per active capture request, and each
    forward step walks the batch view to emit a :class:`StepCapturePlan`
    tailored to that step's token slice.

    Invariants the tests pin down:

    - ``(layer, hook)`` pairs that no active request wants in this step
      are **not** present in ``gather_indices``, so ``on_hook`` skips
      the matching decoder-layer hook point entirely.
    - ``gather_indices`` rows are in the same order the runner fed
      tokens into the model (spec invariant 4).
    - The manager never mutates ``hidden_states`` passed to
      ``on_hook``; it only reads via ``index_select`` and writes a
      newly allocated scratch tensor.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        model_dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, got {num_hidden_layers}"
            )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self._num_hidden_layers = num_hidden_layers
        self._hidden_size = hidden_size
        self._model_dtype = model_dtype
        self._device = torch.device(device) if isinstance(device, str) else device
        self._requests: dict[str, _RequestCaptureState] = {}
        self._step_plan: StepCapturePlan | None = None
        # Terminal per-request errors populated by the runner at
        # admission time (e.g., ``ActivationStoringValidationError``).
        # When set, ``finalize_request`` returns a ``CaptureResult`` with
        # ``status="error"`` and the stored message, without touching
        # the writer. Keys are request ids; values are human-readable
        # strings.
        self._request_errors: dict[str, str] = {}

    # ------------------------------------------------------------------ props

    @property
    def num_hidden_layers(self) -> int:
        return self._num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def model_dtype(self) -> torch.dtype:
        return self._model_dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def is_active(self) -> bool:
        """Return ``True`` if at least one request is currently registered."""
        return bool(self._requests)

    # ---------------------------------------------------------- registration

    def register_request(
        self,
        req_id: str,
        resolved_spec: ResolvedActivationStoringSpec,
        num_prompt_tokens: int,
        *,
        model_name: str = "",
        model_dtype_str: str = "",
        element_size_bytes: int = 0,
        vllm_internal_request_id: str = "",
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        """Admit a new request for capture.

        Called by Phase 4 once per newly-scheduled request whose
        ``SamplingParams.activation_storing`` is set. The admission
        validator has already resolved hooks + positions, so this path
        is pure bookkeeping. Idempotent re-registration is rejected to
        catch runner bugs early.

        Phase 4 passes the sidecar-provenance fields (``model_name``,
        ``model_dtype_str``, ``element_size_bytes``,
        ``vllm_internal_request_id``, ``prompt_token_ids``) so the
        manager can assemble the ``.json`` sidecar on
        :meth:`finalize_request` without reaching back into runner state.
        """
        if req_id in self._requests:
            raise ValueError(
                f"activation capture request {req_id!r} is already registered"
            )
        if num_prompt_tokens <= 0:
            raise ValueError(
                f"activation capture request {req_id!r} has non-positive "
                f"num_prompt_tokens={num_prompt_tokens}"
            )

        # Validate that hook layer indices stay in-range for the model the
        # manager was built for. The admission validator already enforces
        # this against num_hidden_layers, but we double-check here because
        # the manager and the validator were constructed independently.
        for hook_name, layers in resolved_spec.hooks.items():
            if hook_name not in _HOOK_NAME_TO_ID:
                raise ValueError(
                    f"activation capture request {req_id!r} has unknown "
                    f"hook name {hook_name!r}"
                )
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= self._num_hidden_layers:
                    raise ValueError(
                        f"activation capture request {req_id!r} hook "
                        f"{hook_name!r} layer {layer_idx} is out of range "
                        f"[0, {self._num_hidden_layers})"
                    )

        # ``static_positions`` is a list of absolute logical indices
        # whose target tokens may appear in this step or any later step.
        # "all_generated" / "all" stay ``None`` because the upper bound
        # is unknown; we instead expand them lazily in build_step_plan.
        #
        # list form: already resolved at admission time (last_prompt /
        # all_prompt / explicit list); copy to freeze against later
        # mutation. str form: symbolic kind name (all_generated / all).
        positions = resolved_spec.positions
        static_positions: list[int] | None = (
            list(positions) if isinstance(positions, list) else None
        )

        state = _RequestCaptureState(
            request_id=req_id,
            resolved_hooks={
                hook: list(layers) for hook, layers in resolved_spec.hooks.items()
            },
            position_kind=resolved_spec.position_kind,
            static_positions=static_positions,
            num_prompt_tokens=num_prompt_tokens,
            request_id_slug=resolved_spec.request_id_slug,
            tag_slug=resolved_spec.tag_slug,
            model_name=model_name,
            model_dtype_str=model_dtype_str,
            element_size_bytes=element_size_bytes,
            vllm_internal_request_id=vllm_internal_request_id or req_id,
            prompt_token_ids=list(prompt_token_ids or []),
            created_at=_utc_now_iso(),
        )
        self._requests[req_id] = state

    def unregister_request(self, req_id: str) -> None:
        """Remove all capture state for ``req_id``.

        Silent no-op if the request was never registered. Called by
        Phase 4 when a request finishes (for any finish reason).
        """
        self._requests.pop(req_id, None)

    # ------------------------------------------------------- plan building

    def build_step_plan(self, batch_view: CaptureBatchView) -> StepCapturePlan:
        """Produce a :class:`StepCapturePlan` for the given batch view.

        Walks ``batch_view.req_ids`` in order, intersects each
        registered request's target positions with the step's token
        slice, and builds one ``gather_indices`` entry per
        ``(layer, hook)`` the current step touches.

        The returned plan's tensors live on ``self.device``. Phase 3
        drives everything on CPU; Phase 4 will pass a CUDA device.

        Errors accumulated against a request during this step are
        surfaced in ``request_errors`` and the request is skipped.
        """
        # Per-request row lists keyed by (layer, hook). Built as Python
        # lists of ints for easy concatenation; converted to tensors at
        # the end.
        gather_rows: dict[tuple[int, str], list[int]] = {}
        # Entries are appended in the order we discover (layer, hook,
        # request, position) so the output is stable and deterministic.
        entries: list[CapturePositionEntry] = []
        request_errors: dict[str, str] = {}

        num_requests = len(batch_view.req_ids)
        if (
            len(batch_view.num_prompt_tokens) != num_requests
            or len(batch_view.num_computed_tokens) != num_requests
            or len(batch_view.num_scheduled_tokens) != num_requests
            or len(batch_view.token_offsets) != num_requests
        ):
            raise ValueError(
                "CaptureBatchView list lengths must match req_ids length "
                f"(got req_ids={num_requests}, "
                f"num_prompt_tokens={len(batch_view.num_prompt_tokens)}, "
                f"num_computed_tokens={len(batch_view.num_computed_tokens)}, "
                f"num_scheduled_tokens={len(batch_view.num_scheduled_tokens)}, "
                f"token_offsets={len(batch_view.token_offsets)})"
            )

        for i in range(num_requests):
            req_id = batch_view.req_ids[i]
            state = self._requests.get(req_id)
            if state is None:
                # Request in the batch view that the manager doesn't
                # know about is not our problem: it simply isn't a
                # capture request, so skip silently.
                continue

            if state.error is not None:
                # Already failed earlier; surface once and move on
                # without planning.
                request_errors[req_id] = state.error
                continue

            num_scheduled = batch_view.num_scheduled_tokens[i]
            if num_scheduled <= 0:
                continue

            num_computed = batch_view.num_computed_tokens[i]
            token_offset = batch_view.token_offsets[i]

            # Resolve the target positions for this request into a flat
            # list of absolute logical indices. "all_generated" / "all"
            # expand against the current known upper bound (computed +
            # scheduled), which is the only bound the runner has at this
            # point.
            try:
                positions = self._resolve_positions_for_step(
                    state, num_computed, num_scheduled
                )
            except ValueError as exc:
                msg = str(exc)
                state.error = msg
                request_errors[req_id] = msg
                continue

            # Intersect positions with this step's [num_computed,
            # num_computed + num_scheduled) window.
            step_start = num_computed
            step_end = num_computed + num_scheduled

            in_step = [p for p in positions if step_start <= p < step_end]
            if not in_step:
                continue

            # Bump steps_seen only when we actually produce entries for
            # this request this step — so ``step_index`` counts
            # capture-relevant steps rather than every forward pass.
            step_index = state.steps_seen
            state.steps_seen += 1

            for hook_name, layers in state.resolved_hooks.items():
                for layer_idx in layers:
                    key = (layer_idx, hook_name)
                    rows_list = gather_rows.setdefault(key, [])
                    for logical_pos in in_step:
                        abs_row = token_offset + (logical_pos - step_start)
                        scratch_row = len(rows_list)
                        rows_list.append(abs_row)
                        entries.append(
                            CapturePositionEntry(
                                request_id=req_id,
                                layer=layer_idx,
                                hook=hook_name,
                                logical_pos=logical_pos,
                                scratch_row=scratch_row,
                                step_index=step_index,
                            )
                        )

        # Finalize: materialize gather_indices as int64 tensors on the
        # target device and pre-allocate empty scratch tensors so
        # on_hook can fill them in place.
        gather_indices: dict[tuple[int, str], torch.Tensor] = {}
        scratch_gpu: dict[tuple[int, str], torch.Tensor] = {}
        scratch_dtype: dict[tuple[int, str], torch.dtype] = {}
        for key, rows in gather_rows.items():
            idx_tensor = torch.tensor(rows, dtype=torch.int64, device=self._device)
            gather_indices[key] = idx_tensor
            scratch_gpu[key] = torch.empty(
                (len(rows), self._hidden_size),
                dtype=self._model_dtype,
                device=self._device,
            )
            scratch_dtype[key] = self._model_dtype

        plan = StepCapturePlan(
            gather_indices=gather_indices,
            scratch_gpu=scratch_gpu,
            scratch_dtype=scratch_dtype,
            entries=entries,
            request_errors=request_errors,
        )
        self._step_plan = plan
        return plan

    def _resolve_positions_for_step(
        self,
        state: _RequestCaptureState,
        num_computed: int,
        num_scheduled: int,
    ) -> list[int]:
        """Expand a request's position selector against current progress.

        For static kinds (``last_prompt`` / ``all_prompt`` / explicit
        list) this just echoes the stored list. For symbolic kinds
        (``all_generated`` / ``all``) the upper bound is
        ``num_computed + num_scheduled`` so each step only sees positions
        whose underlying tokens exist in the forward pass.
        """
        if state.static_positions is not None:
            return state.static_positions

        # Symbolic: expand against the current upper bound.
        current_generated = max(
            0, (num_computed + num_scheduled) - state.num_prompt_tokens
        )
        return resolve_positions(
            state.position_kind,
            state.num_prompt_tokens,
            current_generated,
            where=f"activation_capture[{state.request_id}].positions",
        )

    # ------------------------------------------------------- plan lifecycle

    def set_step_plan(self, plan: StepCapturePlan | None) -> None:
        """Install ``plan`` as the active plan for the next forward pass.

        Phase 4 calls this in ``_prepare_activation_storing_step`` right
        before the forward. Tests use it to pin a hand-crafted plan
        without going through :meth:`build_step_plan`.
        """
        self._step_plan = plan

    def consume_step_plan(self) -> StepCapturePlan | None:
        """Return and clear the active plan.

        Phase 4 calls this in ``_finalize_activation_storing_step`` to
        take ownership of scratch tensors before copying them to pinned
        CPU. Returning ``None`` afterwards protects the next forward
        pass from re-using stale plans.
        """
        plan = self._step_plan
        self._step_plan = None
        return plan

    # ------------------------------------------------------- hook callback

    def on_hook(
        self,
        layer_idx: int,
        hook_name: str,
        hidden_states: torch.Tensor,
    ) -> None:
        """Callback fired from the custom op inside ``apply_layer_steering``.

        Reads ``hidden_states`` for this ``(layer, hook)`` key through
        ``index_select`` into pre-allocated scratch. Keys not present in
        the current plan are skipped entirely — no allocation, no copy.

        The tensor passed in must be the *pristine* residual (spec
        invariant 2); the steering edit in
        :mod:`vllm.model_executor.layers.steering` guarantees that by
        calling :func:`maybe_capture_residual` **before** the steering
        custom op fires.
        """
        plan = self._step_plan
        if plan is None:
            return
        key = (layer_idx, hook_name)
        idx = plan.gather_indices.get(key)
        if idx is None:
            return
        gathered = hidden_states.index_select(0, idx)
        target_dtype = plan.scratch_dtype[key]
        if gathered.dtype != target_dtype:
            gathered = gathered.to(target_dtype)
        plan.scratch_gpu[key] = gathered

    # -------------------------------------------------- admission/step errors

    def has_request(self, req_id: str) -> bool:
        """Return ``True`` if the manager knows about ``req_id`` at all.

        Accounts for both normal registration and admission-time errors
        recorded via :meth:`record_request_error`. Used by Phase 4's
        finalize step to decide whether to emit a terminal
        :class:`CaptureResult` or silently skip the request.
        """
        return req_id in self._requests or req_id in self._request_errors

    def record_request_error(self, req_id: str, message: str) -> None:
        """Record a terminal per-request error surfaced outside planning.

        Phase 4 calls this when
        :class:`vllm.entrypoints.openai.activation_storing_validation.ActivationStoringValidationError`
        (or any other admission failure) fires for a request whose
        ``SamplingParams.activation_storing`` was set. The request is
        left unregistered so ``build_step_plan`` skips it entirely;
        :meth:`finalize_request` materializes the error as
        ``CaptureResult.status == "error"`` with this message.
        """
        if req_id in self._requests:
            # Already registered: surface the error via the request
            # state so it also flows through ``build_step_plan``'s
            # request_errors channel.
            self._requests[req_id].error = message
        self._request_errors[req_id] = message

    def record_generated_token_ids(
        self, req_id: str, generated_token_ids: list[int]
    ) -> None:
        """Snapshot the request's final generated token list for the sidecar.

        Called by Phase 4 during finalization (when the scheduler reports
        a finished request) so the sidecar can echo the real generated
        stream rather than reconstruct it from partial state. Silent
        no-op for requests the manager never saw.
        """
        state = self._requests.get(req_id)
        if state is not None:
            state.generated_token_ids = list(generated_token_ids)

    def get_request_path_info(
        self, req_id: str
    ) -> tuple[str, str, str] | None:
        """Return ``(tag_slug, request_id_slug, model_dtype_str)`` or ``None``.

        Used by Phase 4's finalize-step drain to compute per-chunk
        ``.bin`` paths without reaching into the manager's private
        state. ``None`` means the request was never registered (e.g.,
        admission error; its chunks should be dropped not written).
        """
        state = self._requests.get(req_id)
        if state is None:
            return None
        return (state.tag_slug, state.request_id_slug, state.model_dtype_str)

    def record_captured_rows(
        self,
        req_id: str,
        layer_idx: int,
        hook_name: str,
        positions: list[int],
    ) -> None:
        """Append newly-written rows' logical positions to the sidecar log.

        Phase 4's finalize step walks ``StepCapturePlan.entries`` to
        compute per-``(req_id, layer, hook)`` row chunks and calls this
        method once per chunk in step / row order. The manager stores
        them so :meth:`finalize_request` can bake the real set of
        positions into the sidecar ``positions`` field.
        """
        state = self._requests.get(req_id)
        if state is None or not positions:
            return
        key = (layer_idx, hook_name)
        state.captured_positions.setdefault(key, []).extend(positions)

    # ---------------------------------------------------------- finalization

    def finalize_request(
        self,
        req_id: str,
        *,
        writer: "ActivationWriter | None" = None,
        root: pathlib.Path | None = None,
    ) -> CaptureResult:
        """Assemble the sidecar payloads, submit finalize tasks, return status.

        Phase 4 calls this from ``_update_states`` once the scheduler
        reports that ``req_id`` has finished. The sequence is:

        1. If the request was never registered and has an error recorded
           via :meth:`record_request_error` (admission-time failure),
           return a terminal ``CaptureResult.status == "error"`` and
           clear any stored state. No writer traffic.
        2. If the request has no recorded error but was never registered,
           return ``CaptureResult.status == "error"`` with a generic
           message (``"request was never registered"``) — this is a
           runner bug, not normal flow.
        3. Otherwise, for each ``(hook, layer)`` in ``resolved_hooks``,
           build a :class:`~vllm.v1.worker.activation_writer.FinalizeTask`
           with the sidecar dict and submit it. Writer failures during
           submit are recorded against the result and converted to
           ``status == "partial_error"``.
        4. Drain the request state from the manager's internal dict
           before returning so the next forward step doesn't see it.

        The method does **not** synchronously wait for writer threads to
        finalize. Spec invariant 6 is maintained by the writer's own
        atomic rename logic: the returned ``CaptureResult`` tells the
        caller what paths *will* exist (or did, for the
        ``partial_error`` case), not that the bytes are visible yet.
        """
        state = self._requests.pop(req_id, None)
        stored_error = self._request_errors.pop(req_id, None)

        if state is None:
            # Admission failure or spurious finalize: no scratch to
            # flush, just report.
            msg = stored_error or (
                f"activation capture request {req_id!r} was never registered"
            )
            return CaptureResult(status="error", error=msg, paths=[])

        # A registration that later hit a runtime error (step_plan
        # surfaced something unusual) shows up on ``state.error``.
        # Treat it the same as an admission failure: no writer traffic.
        if state.error is not None and not state.captured_positions:
            return CaptureResult(status="error", error=state.error, paths=[])

        paths: list[str] = []
        submit_errors: list[str] = []

        if writer is None or root is None:
            # Runner is configured without a writer (cold path or
            # test fixture). Nothing to do beyond returning a clean
            # result. Spec invariant 6 is vacuous here.
            return CaptureResult(status="ok", error=None, paths=[])

        # Import the writer types lazily so importing this module
        # without a running Phase 2 writer stays cheap. The runner
        # always has them available; tests may not.
        from vllm.v1.worker.activation_writer import FinalizeTask, WriteError

        # Build one FinalizeTask per (layer, hook). We walk
        # resolved_hooks (rather than captured_positions) so that even
        # zero-row captures emit a terminal sidecar — this matches the
        # spec's "empty file is a valid outcome" behavior when a request
        # finishes before any target position was touched (e.g.,
        # "all_generated" with zero generated tokens).
        finalize_tasks: list[tuple["CaptureKey", pathlib.Path, pathlib.Path]] = []
        for hook_name, layers in state.resolved_hooks.items():
            for layer_idx in layers:
                bin_path, sidecar_path = _compute_paths(
                    root=root,
                    model_slug=_resolve_model_slug(state.model_name),
                    model_dtype=state.model_dtype_str,
                    tag_slug=state.tag_slug,
                    layer=layer_idx,
                    hook=hook_name,
                    request_id_slug=state.request_id_slug,
                )
                sidecar_payload = _build_sidecar_payload(
                    state=state,
                    layer=layer_idx,
                    hook=hook_name,
                    hidden_size=self._hidden_size,
                )
                key: "CaptureKey" = (req_id, layer_idx, hook_name)
                try:
                    writer.submit(
                        FinalizeTask(
                            bin_path=bin_path,
                            sidecar_path=sidecar_path,
                            sidecar_payload=sidecar_payload,
                            key=key,
                        )
                    )
                except WriteError as exc:
                    submit_errors.append(f"{bin_path}: {exc.message}")
                    continue
                finalize_tasks.append((key, bin_path, sidecar_path))
                paths.append(str(bin_path))
                paths.append(str(sidecar_path))

        if submit_errors and finalize_tasks:
            return CaptureResult(
                status="partial_error",
                error="; ".join(submit_errors),
                paths=paths,
            )
        if submit_errors:
            return CaptureResult(
                status="error",
                error="; ".join(submit_errors),
                paths=paths,
            )

        return CaptureResult(status="ok", error=None, paths=paths)


# ---------------------------------------------------------------------------
# Hook helper + custom op
# ---------------------------------------------------------------------------


def maybe_capture_residual(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_name: str,
) -> None:
    """Cold-path-free gate around the capture custom op.

    Called from :func:`vllm.model_executor.layers.steering.apply_layer_steering`
    **before** the steering op is applied so captures always read the
    pristine residual stream.

    When no manager is installed (cold path / server started without
    ``--activation-storing``) this is a pure Python ``None``-check and
    a dict lookup and returns early. Under ``torch.compile`` the early
    return constant-folds away and the compiled graph contains no
    ``capture_residual`` ops at all — spec invariant 3.
    """
    mgr = _ACTIVE_CAPTURE_MANAGER
    if mgr is None:
        return
    hook_id = _HOOK_NAME_TO_ID[hook_name]
    torch.ops.vllm.capture_residual(hidden_states, layer_idx, hook_id)


def _capture_residual_impl(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_id: int,
) -> torch.Tensor:
    """Real impl of ``torch.ops.vllm.capture_residual``.

    Looks up the process-global active manager and forwards to
    :meth:`ActivationCaptureManager.on_hook`. When the manager is
    ``None`` the op degenerates to returning ``hidden_states``
    unchanged; in practice :func:`maybe_capture_residual` gates the
    call so this branch is only reached in compiled graphs that baked
    in a previous manager reference.

    We mark the op as ``mutates_args=["hidden_states"]`` (see
    registration below) so ``torch.compile`` does not dead-code
    eliminate the call when its return value is discarded by the
    caller. The implementation itself never writes to
    ``hidden_states`` — we only need the side-effect annotation to
    preserve the op in the compiled graph.
    """
    mgr = _ACTIVE_CAPTURE_MANAGER
    if mgr is None:
        return hidden_states
    hook_name = _HOOK_ID_TO_NAME[hook_id]
    mgr.on_hook(layer_idx, hook_name, hidden_states)
    return hidden_states


def _capture_residual_fake(
    hidden_states: torch.Tensor,
    layer_idx: int,
    hook_id: int,
) -> torch.Tensor:
    """FX-tracing fake impl for the capture custom op."""
    return torch.empty_like(hidden_states)


# ``mutates_args=["hidden_states"]`` is a deliberate white lie: the
# real impl does not mutate its argument, but the annotation tells
# ``torch.compile`` the op has observable side effects and therefore
# must not be DCE'd even though the return value is discarded by
# ``apply_layer_steering``. Without this annotation some compile
# passes elide the op when its return equals its input and the
# return value is unused.
direct_register_custom_op(
    op_name="capture_residual",
    op_func=_capture_residual_impl,
    fake_impl=_capture_residual_fake,
    mutates_args=["hidden_states"],
)


# ---------------------------------------------------------------------------
# Path + sidecar helpers
# ---------------------------------------------------------------------------

# Slugging for path segments other than ``tag`` / ``request_id`` (which are
# slugged at admission time by ``vllm.config.activation_storing_types.slug``).
# Used on ``model_name`` fallback and ``model_dtype_str``.
_PATH_SLUG_REGEX = re.compile(r"[^a-zA-Z0-9._-]")


def _slug_segment(value: str) -> str:
    return _PATH_SLUG_REGEX.sub("_", value) if value else "unknown"


def _resolve_model_slug(model_name: str) -> str:
    """Translate ``model_name`` into a filesystem-safe ``{model_slug}`` segment.

    Mirrors the rules documented in the spec's "Model slug resolution"
    section:

    1. Exact ``org/name`` form (two path-like segments, no traversal) is
       preserved as-is so the org becomes a real directory level.
    2. Anything else gets the regex slug.

    Phase 5 may eventually plug in ``served_model_name`` ahead of the
    model config lookup; that's a call-site concern — by the time the
    runner calls :meth:`register_request` it has already decided what
    ``model_name`` to record.
    """
    if not model_name:
        return "unknown"
    parts = model_name.split("/")
    if (
        len(parts) == 2
        and all(p for p in parts)
        and ".." not in model_name
        and not model_name.startswith("/")
    ):
        return "/".join(_slug_segment(p) for p in parts)
    return _slug_segment(model_name)


def _compute_paths(
    *,
    root: pathlib.Path,
    model_slug: str,
    model_dtype: str,
    tag_slug: str,
    layer: int,
    hook: str,
    request_id_slug: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Return ``(bin_path, sidecar_path)`` per the spec filesystem layout.

    Layout: ``{root}/{model_slug}/{model_dtype}/{tag_slug}/{layer}/{hook}/{id}.{ext}``.
    """
    base = (
        root
        / model_slug
        / _slug_segment(model_dtype or "unknown")
        / tag_slug
        / str(layer)
        / hook
    )
    bin_path = base / f"{request_id_slug}.bin"
    sidecar_path = base / f"{request_id_slug}.json"
    return bin_path, sidecar_path


def _utc_now_iso() -> str:
    """Return an RFC-3339 UTC timestamp with millisecond precision."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _build_sidecar_payload(
    *,
    state: _RequestCaptureState,
    layer: int,
    hook: str,
    hidden_size: int,
) -> dict[str, object]:
    """Assemble the ``.json`` sidecar payload for a single ``(layer, hook)``.

    Captures the exact set of keys the spec requires (see
    ``docs/features/activation_storing.md`` § File format). Keep this in
    lockstep with the ``test_sidecar_payload_keys`` test.
    """
    positions = state.captured_positions.get((layer, hook), [])
    num_rows = len(positions)
    last_prompt_idx = state.num_prompt_tokens - 1

    if state.error is not None:
        capture_status = "error"
        capture_error: str | None = state.error
    else:
        capture_status = "ok"
        capture_error = None

    return {
        "request_id": state.request_id_slug,
        "tag": state.tag_slug,
        "model": state.model_name,
        "model_dtype": state.model_dtype_str,
        "layer": layer,
        "hook": hook,
        "shape": [num_rows, hidden_size],
        "dtype": state.model_dtype_str,
        "element_size": state.element_size_bytes,
        "positions": list(positions),
        "position_kind": state.position_kind,
        "last_prompt_token_index": last_prompt_idx,
        "prompt_token_ids": list(state.prompt_token_ids),
        "generated_token_ids": list(state.generated_token_ids),
        "created_at": state.created_at,
        "finalized_at": _utc_now_iso(),
        "vllm_internal_request_id": state.vllm_internal_request_id,
        "capture_status": capture_status,
        "capture_error": capture_error,
    }


__all__ = [
    "ActivationCaptureManager",
    "CaptureBatchView",
    "CapturePositionEntry",
    "StepCapturePlan",
    "get_active_capture_manager",
    "maybe_capture_residual",
    "set_active_capture_manager",
]
