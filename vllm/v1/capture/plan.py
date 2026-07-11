# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan types for the multi-consumer capture manager.

``StepCapturePlan`` is the per-step data structure that the
:class:`~vllm.v1.capture.manager.CaptureManager` builds before the
forward pass and that the dispatch path reads after scratch tensors are
populated.  ``CaptureBatchView`` is a lightweight, scheduler-agnostic
snapshot of the current batch that decouples the manager from v1
scheduler internals.

These types are deliberately kept in their own module so that downstream
code can depend on them without pulling in the manager's implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Batch view
# ---------------------------------------------------------------------------


@dataclass
class CaptureBatchView:
    """Minimal view of the active batch for capture planning.

    All lists are parallel — indexed by batch-row ``0..N-1``.

    Attributes:
        req_ids: One entry per active request in the batch.
        num_prompt_tokens: Prompt length of each request.
        num_computed_tokens: Tokens already forwarded before this step.
        num_scheduled_tokens: Tokens being forwarded in this step.
        token_offsets: Absolute row index into the flat hidden-state
            tensor where each request's scheduled tokens begin.
    """

    req_ids: list[str]
    num_prompt_tokens: list[int]
    num_computed_tokens: list[int]
    num_scheduled_tokens: list[int]
    token_offsets: list[int]


# ---------------------------------------------------------------------------
# Per-position metadata
# ---------------------------------------------------------------------------


@dataclass
class CapturePositionEntry:
    """One scratch row's worth of capture metadata.

    After the forward pass the dispatch path walks the plan's ``entries``
    list and uses ``consumer_mask`` to decide which sinks receive a
    ``CaptureChunk`` containing this row.

    Attributes:
        request_id: The owning request's id.
        layer: Decoder-layer index.
        hook: Hook-point name (e.g. ``"post_block"``).
        logical_pos: Absolute position in the request's token sequence.
        scratch_row: Index within the ``(layer, hook)``'s scratch tensor.
        step_index: Capture-step ordinal for this request.
        consumer_mask: Bitset — bit *i* is set when consumer *i* wants
            this row.
    """

    request_id: str
    layer: int
    hook: str
    logical_pos: int
    scratch_row: int
    step_index: int
    consumer_mask: int


# ---------------------------------------------------------------------------
# Step capture plan
# ---------------------------------------------------------------------------


@dataclass
class StepCapturePlan:
    """Snapshot of everything the capture path needs for one forward step.

    ``gather_indices`` maps each **client-spec** ``(layer, hook)`` pair to
    an int64 tensor of absolute batch-row indices.  The forward-pass hook
    (:meth:`CaptureManager.on_hook`) uses ``index_select`` to copy those
    rows into ``scratch_gpu``.  This dynamic gather allocates a fresh
    output each step and therefore only runs on **eager** steps — the
    :class:`~vllm.v1.capture.step_gate.CaptureStepGate` forces eager
    whenever a client-spec request captures.

    ``global_gather_indices`` maps each **global-spec** ``(layer, hook)``
    pair to the rows to slice *after* the forward pass.  Global keys are
    captured by a fixed-shape full-residual ``copy_`` into a persistent
    buffer that is baked into the CUDA graph at warmup (see
    :meth:`CaptureManager.on_hook` and the ``_global_buffers`` design), so
    a global-only step keeps full cudagraph speed.  The host slices these
    rows out of the persistent buffer into ``scratch_gpu`` post-forward
    (eager, off-graph) via :meth:`CaptureManager._materialize_global_keys`.

    ``entries`` is a flat list whose order matches the row order inside
    each ``(layer, hook)`` scratch tensor — for both client and global
    keys — so the dispatch path groups entries by
    ``(request_id, layer, hook)`` to build per-consumer ``CaptureChunk``
    objects regardless of which path populated the scratch row.

    ``request_errors`` surfaces registration-time or step-time failures
    keyed by request id.
    """

    gather_indices: dict[tuple[int, str], torch.Tensor]
    scratch_gpu: dict[tuple[int, str], torch.Tensor]
    scratch_dtype: dict[tuple[int, str], torch.dtype]
    entries: list[CapturePositionEntry]
    global_gather_indices: dict[tuple[int, str], torch.Tensor] = field(
        default_factory=dict
    )
    request_errors: dict[str, str] = field(default_factory=dict)


__all__ = [
    "CaptureBatchView",
    "CapturePositionEntry",
    "StepCapturePlan",
]
