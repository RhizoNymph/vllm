# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-steering control plane for the v2 GPU model runner.

The steering *data plane* (the ``apply_steering`` custom op, per-layer table
buffers, the fused Triton kernel, ``SteeringManager``) is shared with the v1
runner unchanged. So is the entire runner-agnostic half of
:class:`SteeringModelRunnerMixin`: state init, steerable-layer discovery, spec
validation, the public RPC API (``set_steering_vectors`` etc., which
``gpu_worker`` already forwards to ``self.model_runner``), and
``_resolve_request_steering``.

The per-request steering lifecycle (admission, streaming re-add, transition,
finish/preempt release, resume) is shared: both runners drive the canonical
``self._steering_reqs`` store on :class:`SteeringModelRunnerMixin`
(``_steering_add_request`` / ``_steering_finish_requests`` / ...), populated
identically from the broadcast ``NewRequestData``. The per-step hot path
(``_update_steering_buffers`` + the effective-decode-signature deltas) and the
dynamic-override apply are shared too (de-fork step E). This subclass now
overrides only two thin batch-state accessors that read v2's ``InputBatch`` +
``RequestState`` instead of v1's batch-ordered columns: ``_steering_batch_view``
(the per-step hot path's view) and ``_steering_req_position`` (the override
apply's decode-only phase guard).

Steering needs no force-eager seam: the per-layer tables and ``steering_index``
are persistent buffers written in place before the forward, so a FULL cudagraph
replay reads the current step's values.

See ``docs/design/v2_runner_steering_capture.md`` for the full contract.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.worker.steering_batch_view import SteeringBatchView
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
    _SteeringReqState,
)

# Re-exported for callers (and tests) that import the per-request steering
# state dataclass from the v2 module. It now lives in the shared mixin so both
# runners populate one canonical ``self._steering_reqs`` store.
__all__ = ["SteeringRunnerMixin", "_SteeringReqState"]

logger = init_logger(__name__)


class SteeringRunnerMixin(SteeringModelRunnerMixin):
    """v2 steering control plane. Mixes the shared steering logic in via the
    v1 mixin and overrides only the v2-runner-coupled paths.

    The per-request steering lifecycle (add / finish / transition / release /
    resume), the per-step buffer build, the effective-decode-signature deltas,
    and the dynamic-override apply all live on the shared
    :class:`SteeringModelRunnerMixin`; both runners drive them identically. This
    subclass only overrides the two batch-state accessors that read v2's
    ``req_states`` (v2 ``RequestState``) instead of v1's batch-ordered columns:
    ``_steering_batch_view`` (per-step hot path) and ``_steering_req_position``
    (override-apply decode-only phase guard).
    """

    # ---- per-step batch view (the only v2-coupled steering seam left) ------

    def _steering_batch_view(self) -> SteeringBatchView:
        """v2 batch view for the shared per-step steering hot path.

        Overrides :meth:`SteeringModelRunnerMixin._steering_batch_view` to read
        v2's layout: batch order + slot->row mapping from ``input_batch``
        (``req_ids`` / ``idx_mapping_np``) and the per-request token counts from
        ``req_states`` (``num_computed_tokens_np`` / ``prompt_len.np``), instead
        of v1's batch-ordered ``input_batch`` columns. Returns one reusable
        instance mutated in place — no per-step allocation.

        The unified body reaches ``input_batch`` through ``self.input_batch``,
        set by the runner just before the call (v2 keeps no persistent
        ``self.input_batch``).
        """
        ib = self.input_batch
        rs = self.req_states
        bv = self._steering_bview
        if bv is None:
            bv = SteeringBatchView(
                num_reqs=ib.num_reqs,
                req_ids=ib.req_ids,
                idx_np=ib.idx_mapping_np,
                num_computed_np=rs.num_computed_tokens_np,
                num_prompt_np=rs.prompt_len.np,
            )
            self._steering_bview = bv
        else:
            bv.num_reqs = ib.num_reqs
            bv.req_ids = ib.req_ids
            bv.idx_np = ib.idx_mapping_np
            bv.num_computed_np = rs.num_computed_tokens_np
            bv.num_prompt_np = rs.prompt_len.np
        return bv

    def _steering_req_position(self, req_id: str) -> tuple[int, int] | None:
        """v2 batch-position accessor for the shared override apply.

        Overrides :meth:`SteeringModelRunnerMixin._steering_req_position` to
        read v2's ``req_states`` (``req_id_to_index`` + ``num_computed_tokens_np``
        / ``prompt_len.np``) instead of v1's ``self.input_batch``. Returns
        ``None`` when the request is not in the current batch.
        """
        req_idx = self.req_states.req_id_to_index.get(req_id)
        if req_idx is None:
            return None
        num_computed = int(self.req_states.num_computed_tokens_np[req_idx])
        num_prompt = int(self.req_states.prompt_len.np[req_idx])
        return num_computed, num_prompt
