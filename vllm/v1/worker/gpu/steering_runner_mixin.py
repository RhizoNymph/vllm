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

Only three v1 methods read v1-runner state (``self.input_batch`` /
``self.requests``): the per-step ``_update_steering_buffers`` hot path, the
prefill->decode transition, and finished-config release. v2 retains no
``CachedRequestState`` dict, so this subclass keeps its own per-request steering
state (populated from ``NewRequestData`` in ``add_requests``) and reimplements
those three against v2's ``InputBatch`` + ``RequestState``.

Steering needs no force-eager seam: the per-layer tables and ``steering_index``
are persistent buffers written in place before the forward, so a FULL cudagraph
replay reads the current step's values.

See ``docs/design/v2_runner_steering_capture.md`` for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    SteeringHookPoint,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass
class _SteeringReqState:
    """Per-request steering state the v2 runner must retain itself.

    v2 does not keep ``sampling_params`` or the steering hashes past
    ``add_requests``, so we capture what the transition / release / resolve
    paths need: the params (for re-resolving the decode tier lazily), both
    config hashes, the prompt length (for the prefill->decode boundary), and
    the currently-registered phase.
    """

    sampling_params: SamplingParams
    prefill_hash: int
    decode_hash: int
    num_prompt_tokens: int
    phase: str  # "prefill" | "decode"


class SteeringRunnerMixin(SteeringModelRunnerMixin):
    """v2 steering control plane. Mixes the shared steering logic in via the
    v1 mixin and overrides only the v2-runner-coupled paths.

    Assumes the concrete runner provides ``req_states`` (v2 ``RequestState``)
    in addition to everything :class:`SteeringModelRunnerMixin` needs.
    """

    _steering_reqs: dict[str, _SteeringReqState]

    def _init_steering_state(self) -> None:
        super()._init_steering_state()
        self._steering_reqs = {}

    # ---- request lifecycle -------------------------------------------------

    def _steering_add_request(self, new_req_data: NewRequestData) -> None:
        """Track a newly admitted request and register its initial config.

        Also covers streaming re-adds: ``add_requests`` removes the prior
        instance first, so any state we already held for this id is released
        before the fresh prefill config is registered.
        """
        mgr = self._steering_manager
        if mgr is None:
            return

        # Streaming re-add / stale state: release whatever we held before.
        old = self._steering_reqs.pop(new_req_data.req_id, None)
        if old is not None:
            self._steering_release_state(old)

        sp = new_req_data.sampling_params
        prefill_hash = new_req_data.prefill_steering_config_hash
        decode_hash = new_req_data.decode_steering_config_hash
        if sp is None or (prefill_hash == 0 and decode_hash == 0):
            return

        num_prompt = length_from_prompt_token_ids_or_embeds(
            new_req_data.prompt_token_ids,
            new_req_data.prompt_embeds,
        )
        rs = _SteeringReqState(
            sampling_params=sp,
            prefill_hash=prefill_hash,
            decode_hash=decode_hash,
            num_prompt_tokens=num_prompt,
            phase="prefill",
        )
        self._steering_reqs[new_req_data.req_id] = rs

        # A full prefix-cache hit admits the request directly into decode; the
        # scheduler reserves the matching row, so register_config is expected to
        # succeed (a RuntimeError indicates a scheduler accounting bug).
        if new_req_data.num_computed_tokens >= num_prompt:
            effective_decode = self._resolve_request_steering(sp, "decode")
            if decode_hash != 0 and effective_decode:
                mgr.register_config(
                    decode_hash,
                    effective_decode,
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                )
            rs.phase = "decode"
        else:
            # Normal: start in prefill; the decode config is registered lazily
            # at the prefill->decode boundary in _update_steering_buffers_v2.
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            if prefill_hash != 0 and effective_prefill:
                mgr.register_config(
                    prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                )
            rs.phase = "prefill"

    def _steering_finish_requests(self, req_ids: set[str] | list[str]) -> None:
        """Release configs for finished (or preempted) requests.

        Preempted requests are released too: they re-enter through
        ``add_requests`` on resume, which re-registers a fresh prefill config.
        """
        if self._steering_manager is None:
            return
        for req_id in req_ids:
            rs = self._steering_reqs.pop(req_id, None)
            if rs is not None:
                self._steering_release_state(rs)

    def _steering_release_state(self, rs: _SteeringReqState) -> None:
        """Release the config for whichever phase ``rs`` is currently in."""
        mgr = self._steering_manager
        if mgr is None:
            return
        if rs.phase == "prefill" and rs.prefill_hash != 0:
            mgr.release_config(rs.prefill_hash, "prefill")
        elif rs.phase == "decode" and rs.decode_hash != 0:
            mgr.release_config(rs.decode_hash, "decode")

    def _steering_transition(self, rs: _SteeringReqState) -> None:
        """Handle a request crossing the prefill->decode boundary this step.

        Releases the prefill config and registers the decode config so it is
        ready for the next step's table population. The scheduler reserves the
        decode row at the step prefill completes, so register_config succeeds.
        """
        mgr = self._steering_manager
        assert mgr is not None
        if rs.prefill_hash != 0:
            mgr.release_config(rs.prefill_hash, "prefill")
        if rs.decode_hash != 0:
            effective_decode = self._resolve_request_steering(
                rs.sampling_params, "decode"
            )
            if effective_decode:
                mgr.register_config(
                    rs.decode_hash,
                    effective_decode,
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                )
        rs.phase = "decode"

    # ---- per-step buffer / index maintenance -------------------------------

    def _update_steering_buffers_v2(
        self, scheduler_output: SchedulerOutput, input_batch: InputBatch
    ) -> None:
        """Populate per-layer steering tables and the shared steering index.

        v2 port of ``SteeringModelRunnerMixin._update_steering_buffers``: the
        per-request hashes come from ``self._steering_reqs`` (not the input
        batch), and token counts / phase come from ``input_batch`` +
        ``req_states``.
        """
        mgr = self._steering_manager
        if mgr is None or not self._steerable_layers_cache:
            return

        reqs = self._steering_reqs
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_np = input_batch.idx_mapping_np

        # Short-circuit when nothing is active (no per-request config in this
        # batch and no globals). A decode-only request (prefill_hash == 0,
        # decode_hash != 0) registers its config lazily at the transition below,
        # so the batch scan must not let the short-circuit swallow it.
        batch_has_per_request_steering = any(
            (rs := reqs.get(req_ids[i])) is not None
            and (rs.prefill_hash != 0 or rs.decode_hash != 0)
            for i in range(num_reqs)
        )
        if (
            not batch_has_per_request_steering
            and not mgr.config_to_row
            and not mgr.global_base_vectors
            and not mgr.global_prefill_vectors
            and not mgr.global_decode_vectors
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                steering_index = cast(torch.Tensor, any_layer.steering_index)
                steering_index.zero_()
                for mod in self._steerable_layers_cache.values():
                    for hp in SteeringHookPoint:
                        flag_buf = getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp], None)
                        if flag_buf is not None:
                            flag_buf.zero_()
                self._steering_index_dirty = False
            return

        # 1. Populate tables only when state changed since the last populate.
        if mgr._tables_dirty:
            mgr.populate_steering_tables(self._steerable_layers_cache)

        # 2. Build the per-token steering index.
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = cast(torch.Tensor, any_layer.steering_index)

        rows_scratch = self._steering_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._steering_index_pinned
        assert rows_scratch is not None
        assert n_tokens_scratch is not None
        assert index_pinned is not None
        if rows_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            self._steering_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch

        num_computed_np = self.req_states.num_computed_tokens_np
        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            rs = reqs.get(req_id)
            if rs is None:
                # No steering for this request — row 0 is the no-steer sentinel.
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                active_count += 1
                continue

            num_computed = int(num_computed_np[int(idx_np[i])])
            num_prompt = rs.num_prompt_tokens
            if num_computed < num_prompt:
                row = mgr.get_row_for_config(rs.prefill_hash, is_prefill=True)
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens
                if num_computed + n_tokens >= num_prompt:
                    self._steering_transition(rs)
            else:
                row = mgr.get_row_for_config(rs.decode_hash, is_prefill=False)
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens
            active_count += 1

        if active_count > 0:
            expanded = np.repeat(
                rows_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_expanded = int(expanded.shape[0])
            n_expanded = min(n_expanded, index_pinned.shape[0], steering_index.shape[0])
            index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
            steering_index[:n_expanded].copy_(
                index_pinned[:n_expanded], non_blocking=True
            )
        else:
            n_expanded = 0

        if n_expanded < steering_index.shape[0]:
            steering_index[n_expanded:].zero_()

        self._steering_index_dirty = True
