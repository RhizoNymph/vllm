# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-patching control plane for the v2 GPU model runner.

The patching data plane and the runner-agnostic control plane (state, kernel
warmup, the pure step planner, buffer writes, request lifecycle) live in
:class:`PatchModelRunnerMixin`. This subclass adds only the v2-coupled half:
projecting v2's ``InputBatch`` + ``req_states`` into the per-step batch view and
resolving a newly admitted request's ``patch`` spec into source-vector entries.

Like steering, patching needs no force-eager seam: the per-(layer, hook)
buffers are written in place before the forward, so a FULL cudagraph replay
reads this step's values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.patch_runner_mixin import (
    PatchModelRunnerMixin,
    _PatchBatchView,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class PatchRunnerMixin(PatchModelRunnerMixin):
    """v2 activation-patching control plane.

    ``_patch_add_request`` (spec resolution) and the request lifecycle are
    runner-agnostic and live in :class:`PatchModelRunnerMixin`; this subclass
    adds only the v2-coupled per-step batch-view projection.
    """

    if TYPE_CHECKING:
        # Provided by the concrete v2 runner this mixin is mixed into.
        req_states: RequestState

    def _update_patch_buffers_v2(
        self, scheduler_output: SchedulerOutput, input_batch: InputBatch
    ) -> None:
        """Write this step's patch buffers from per-request specs.

        Projects ``InputBatch`` (decode-first sorted) into the patch view using
        ``query_start_loc_np`` for token offsets and ``idx_mapping_np`` to read
        prompt/computed lengths from ``req_states`` — identical to the capture
        batch-view build.
        """
        if not self._patchable_layers or self._patch_max_slots <= 0:
            return
        # Short-circuit cheaply when no request in the engine has a patch spec
        # (still honor the dirty cleanup path inside _write_patch_step).
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return

        idx = input_batch.idx_mapping_np[:num_reqs]
        req_states = self.req_states
        view = _PatchBatchView(
            req_ids=list(input_batch.req_ids),
            num_computed=req_states.num_computed_tokens_np[idx].tolist(),
            num_scheduled=input_batch.num_scheduled_tokens[:num_reqs].tolist(),
            token_offsets=input_batch.query_start_loc_np[:num_reqs].tolist(),
        )
        self._write_patch_step(view)
