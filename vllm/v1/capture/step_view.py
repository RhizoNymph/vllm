# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step view handed to sync capture consumers.

Sync consumers (``CaptureConsumer.execution == "sync"``) run on the
model-runner step thread immediately after the forward pass and receive
a :class:`StepCaptureView` describing the step that just executed. See
``docs/design/dynamic_steering.md`` §5.1 for the execution-axis design.

Determinism contract: every field is derived from rank-identical inputs
(the broadcast ``scheduler_output`` plus ``InputBatch`` state that every
TP rank maintains), and the monitored residual is read after the
tensor-parallel all-reduce, so the view — and therefore a pure
consumer's decisions — is identical on every TP rank with no
communication.

Validity window: ``tensors`` are **zero-copy views** into the capture
manager's persistent global buffers. The next forward pass overwrites
them in place (the copy is baked into the CUDA graph). A sync consumer
must finish reading (e.g. its probe GEMM and any D2H) inside
``on_step``; stashing a view for later use reads garbage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

__all__ = ["StepCaptureView", "StepRequestView"]


@dataclass(frozen=True)
class StepRequestView:
    """One request's slice of the step.

    ``start``/``end`` index rows of every tensor in
    :attr:`StepCaptureView.tensors` (the step's tokens are laid out in
    input-batch request order; this request's scheduled tokens occupy
    ``tensor[start:end]``).
    """

    req_id: str
    start: int
    end: int
    # Phase of the tokens in this step's window. A request whose prompt
    # completes mid-step still counts as "prefill" for the whole window
    # (matching the steering phase used to build steering_index).
    phase: Literal["prefill", "decode"]
    # The window's input token ids (length ``end - start``), copied out
    # of the batch's CPU token table — safe to retain across steps.
    token_ids: np.ndarray
    # Optional client-supplied conversation grouping id
    # (``RequestMetadata.conversation_id``). Pure host-side string metadata
    # (no GPU work / D2H), so it is populated identically on the v1 and v2
    # runners — unlike ``token_ids``, which is empty on v2. Lets a sync
    # consumer correlate successive requests of the same conversation (e.g.
    # latch a steering decision after a trigger and re-apply it to later
    # turns). ``None`` when the request did not set it.
    conversation_id: str | None = None


@dataclass(frozen=True)
class StepCaptureView:
    """Everything a sync consumer sees for one forward step."""

    # Runner-local monotonic step counter (identical across ranks).
    step: int
    # ``(layer_idx, hook_name)`` -> GPU tensor ``[num_tokens, hidden]``,
    # sliced to the step's unpadded token count. Zero-copy views of the
    # capture manager's persistent buffers — valid only until the next
    # forward pass begins (see module docstring).
    tensors: dict[tuple[int, str], torch.Tensor]
    # Per-request spans in row order; requests with zero scheduled
    # tokens this step are absent.
    requests: list[StepRequestView]
