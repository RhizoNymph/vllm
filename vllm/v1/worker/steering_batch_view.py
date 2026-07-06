# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-agnostic batch view for the shared per-step steering hot path.

The v1 and v2 GPU runners hold per-request batch state differently: v1 keeps
batch-ordered columns on its ``input_batch`` (``num_computed_tokens_cpu`` /
``num_prompt_tokens`` indexed directly by batch position), while v2 keeps the
counts in an ``idx_mapping``-indirected ``RequestState``. :class:`SteeringBatchView`
is the thin seam that lets the single ``_update_steering_buffers`` body read
either layout without a fork.

Each runner keeps one reusable instance (``self._steering_bview``) and mutates
its fields in place once per step, so the hot path adds no per-step allocation
and every array is borrowed (never copied).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SteeringBatchView:
    """Per-step batch state the unified steering hot path reads.

    Position ``i`` (``0 <= i < num_reqs``) is a batch slot. ``idx_np[i]`` maps
    that slot to the row of ``num_computed_np`` / ``num_prompt_np`` holding the
    request's token counts (the identity map for v1, where the columns are
    already batch-ordered).

    Attributes:
        num_reqs: Number of active request slots this step.
        req_ids: Request ids in batch order (may contain ``None`` padding).
        idx_np: Batch slot ``i`` -> state-row index.
        num_computed_np: Computed-token counts, indexed by ``idx_np[i]``.
        num_prompt_np: Prompt-token counts, indexed by ``idx_np[i]``.
    """

    num_reqs: int
    req_ids: Sequence[str | None]
    idx_np: np.ndarray
    num_computed_np: np.ndarray
    num_prompt_np: np.ndarray
