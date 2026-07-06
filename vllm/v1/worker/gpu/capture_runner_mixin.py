# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-capture control plane for the v2 GPU model runner.

Most of the control plane is shared with the v1 runner and lives in
``vllm/v1/worker/capture_runner_mixin.py`` (managers/gate/store construction,
request registration/finalize, the sync-consumer step loop, drains). This
module holds only the genuinely-v2 pieces layered on top of that shared mixin:

* the force-eager gate view and gather-plan view, which v2 must build *before*
  its ``InputBatch`` exists (so they cannot be shared with v1), and
* the two per-step hooks the shared ``_build_step_capture_view`` calls —
  ``_iter_step_capture_rows`` (offsets from ``query_start_loc_np`` + counts from
  ``req_states``) and ``_step_view_token_ids`` (empty; v2's token ids are
  GPU-resident, so mirroring them would force a per-step D2H).

See ``docs/design/v2_runner_steering_capture.md`` for the full contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.logger import init_logger
from vllm.v1.worker.capture_runner_mixin import (
    CaptureRunnerMixin as _SharedCaptureMixin,
)

if TYPE_CHECKING:
    from vllm.v1.capture.plan import CaptureBatchView
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Shared read-only empty token-id array. v2's per-request token windows are
# GPU-resident, so ``StepRequestView.token_ids`` is always empty; one immutable
# array is reused across every request/step rather than allocating per row.
_EMPTY_STEP_TOKEN_IDS = np.empty(0, dtype=np.int64)


class CaptureRunnerMixin(_SharedCaptureMixin):
    """v2-runner activation-capture control plane.

    Inherits the runner-agnostic control plane from
    :class:`vllm.v1.worker.capture_runner_mixin.CaptureRunnerMixin` and adds the
    v2-only gate/plan views plus the per-step hooks the shared view builder
    calls. Assumes the concrete runner also provides ``req_states``.
    """

    # ---- per-step gate / gather plan (v2-only) -----------------------------

    def _build_capture_gate_view(
        self, scheduler_output: SchedulerOutput
    ) -> CaptureBatchView:
        """Build an (unordered) view for the force-eager gate decision.

        v2 resolves the cudagraph-vs-eager batch descriptor *before*
        ``prepare_inputs`` builds the ``InputBatch``, so the gate view is built
        here from ``scheduler_output`` + ``req_states``. The gate only inspects
        per-request prompt/computed/scheduled token counts (``token_offsets`` is
        unused for the boolean), so request ordering does not matter.
        """
        from vllm.v1.capture.plan import CaptureBatchView

        req_states = self.req_states
        req_ids: list[str] = []
        num_prompt_tokens: list[int] = []
        num_computed_tokens: list[int] = []
        num_scheduled_tokens: list[int] = []
        token_offsets: list[int] = []

        offset = 0
        for req_id, n_tokens in scheduler_output.num_scheduled_tokens.items():
            req_ids.append(req_id)
            req_idx = req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                num_prompt_tokens.append(0)
                num_computed_tokens.append(0)
            else:
                num_prompt_tokens.append(int(req_states.prompt_len.np[req_idx]))
                num_computed_tokens.append(
                    int(req_states.num_computed_tokens_np[req_idx])
                )
            num_scheduled_tokens.append(int(n_tokens))
            token_offsets.append(offset)
            offset += int(n_tokens)

        return CaptureBatchView(
            req_ids=req_ids,
            num_prompt_tokens=num_prompt_tokens,
            num_computed_tokens=num_computed_tokens,
            num_scheduled_tokens=num_scheduled_tokens,
            token_offsets=token_offsets,
        )

    def _build_capture_batch_view(self, input_batch: InputBatch) -> CaptureBatchView:
        """Project v2's ``InputBatch`` into a :class:`CaptureBatchView`.

        Used for the gather plan, which needs token offsets that match the
        actual forward batch layout. v2's batch is sorted decode-first;
        ``query_start_loc_np`` gives the per-request token offset and
        ``idx_mapping_np`` maps batch index to the request-state slot holding
        prompt/computed lengths.
        """
        from vllm.v1.capture.plan import CaptureBatchView

        num_reqs = input_batch.num_reqs
        idx = input_batch.idx_mapping_np[:num_reqs]
        req_states = self.req_states
        return CaptureBatchView(
            req_ids=list(input_batch.req_ids),
            num_prompt_tokens=req_states.prompt_len.np[idx].tolist(),
            num_computed_tokens=req_states.num_computed_tokens_np[idx].tolist(),
            num_scheduled_tokens=input_batch.num_scheduled_tokens[:num_reqs].tolist(),
            token_offsets=input_batch.query_start_loc_np[:num_reqs].tolist(),
        )

    def _capture_gate_decision(self, scheduler_output: SchedulerOutput) -> bool:
        """Rank-replicated force-eager decision for this step.

        Returns ``True`` iff a per-request client spec captures this step, so the
        runner must run eager (the dynamic ``index_select`` gather cannot be
        recorded into a CUDA graph). Global specs ride the persistent-buffer path
        and never force eager.
        """
        if self._capture_step_gate is None:
            return False
        view = self._build_capture_gate_view(scheduler_output)
        return self._capture_step_gate.step_captures(view)

    def _capture_build_plan(self, input_batch: InputBatch) -> None:
        """Build the per-step gather plan on the capturer rank (pre-forward)."""
        if self._capture_manager is not None and self._capture_manager.is_active():
            view = self._build_capture_batch_view(input_batch)
            self._capture_manager.build_step_plan(view)

    # ---- shared-view hooks -------------------------------------------------

    def _iter_step_capture_rows(
        self, scheduler_output: SchedulerOutput, input_batch: Any
    ):
        """Enumerate this step's batch rows for the shared step-capture view.

        Token spans come from ``query_start_loc_np`` (the forward batch layout
        the capture op wrote into the global buffers); per-request phase counts
        come from the ``req_states`` arrays via ``idx_mapping_np``. The residual
        is read after the tensor-parallel all-reduce, so ``start``/``end`` index
        the same buffer rows on every rank.
        """
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_np = input_batch.idx_mapping_np
        qsl_np = input_batch.query_start_loc_np
        num_computed_np = self.req_states.num_computed_tokens_np
        prompt_len_np = self.req_states.prompt_len.np

        for i in range(num_reqs):
            start = int(qsl_np[i])
            end = int(qsl_np[i + 1])
            if end <= start:
                continue
            req_idx = int(idx_np[i])
            yield (
                req_ids[i],
                start,
                end,
                int(num_computed_np[req_idx]),
                int(prompt_len_np[req_idx]),
                req_idx,
            )

    def _step_view_token_ids(
        self, row_index: int, num_computed: int, n_tokens: int
    ) -> np.ndarray:
        """v2 token ids are GPU-resident: expose an empty CPU window.

        Materializing the input-token window would force a per-step D2H sync the
        v2 runner is built to avoid; sync consumers read activations from
        ``tensors`` and per-request spans/phase from ``requests`` instead.
        """
        return _EMPTY_STEP_TOKEN_IDS
