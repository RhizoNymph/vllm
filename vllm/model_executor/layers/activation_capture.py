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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.config.activation_storing_types import resolve_positions
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.entrypoints.openai.activation_storing_validation import (
        ResolvedActivationStoringSpec,
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
    """

    request_id: str
    resolved_hooks: dict[str, list[int]]
    position_kind: str
    static_positions: list[int] | None
    num_prompt_tokens: int
    error: str | None = None
    steps_seen: int = 0


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
    ) -> None:
        """Admit a new request for capture.

        Called by Phase 4 once per newly-scheduled request whose
        ``SamplingParams.activation_storing`` is set. The admission
        validator has already resolved hooks + positions, so this path
        is pure bookkeeping. Idempotent re-registration is rejected to
        catch runner bugs early.
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


__all__ = [
    "ActivationCaptureManager",
    "CaptureBatchView",
    "CapturePositionEntry",
    "StepCapturePlan",
    "get_active_capture_manager",
    "maybe_capture_residual",
    "set_active_capture_manager",
]
