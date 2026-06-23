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

from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    SteeringHookPoint,
)
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.worker.steering_action_queue import (
    get_steering_action_queue,
    validate_steering_vectors,
)
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.steering_action_queue import RequestSteeringOverride

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
        # A re-add goes back through prefill, so any live dynamic decode
        # override belongs to the finished decode run and is dropped (the
        # driving policy re-engages on the continuation's decode).
        self._drop_request_dynamic_override(new_req_data.req_id)
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
            # Drop any live dynamic decode override (routing state local to the
            # finished/preempted decode run); preempted requests re-register a
            # fresh prefill config on resume.
            self._drop_request_dynamic_override(req_id)
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
        """Populate per-layer steering tables and the shared steering buffers.

        v2 port of ``SteeringModelRunnerMixin._update_steering_buffers``: the
        per-request hashes come from ``self._steering_reqs`` (not the input
        batch), and token counts / phase come from ``input_batch`` +
        ``req_states``. Brings the v2 runner to parity with v1 — in addition
        to the per-token ``steering_index`` it builds the §5.4 dynamic-tier
        gate (``steering_token_scales``), the Phase 2 row gate
        (``steering_row_gate`` / ``steering_decode_mask``), drains the async
        action queue, routes live dynamic-decode overrides, and emits the APC
        effective-decode-signature deltas.
        """
        mgr = self._steering_manager
        if mgr is None or not self._steerable_layers_cache:
            self._pending_decode_sigs = {}
            return

        # Fresh each step; populated at the exits below so the model runner
        # attaches this step's effective-decode-signature deltas (APC).
        self._pending_decode_sigs = {}

        # Dynamic steering, async transport: drain the in-process action queue
        # before anything else so updates submitted during step N (by a capture
        # consumer on the dispatch thread) are visible to the tables built for
        # step N+1. Must run before the nothing-active short-circuit — a drained
        # update may be exactly what activates steering. Application sets
        # ``_tables_dirty``, so the existing populate path uploads the new state.
        action_queue = get_steering_action_queue()
        if action_queue is not None and action_queue:
            self._apply_steering_actions(
                action_queue.drain(),
                source="action_queue",
                queue=action_queue,
            )

        reqs = self._steering_reqs
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_np = input_batch.idx_mapping_np
        num_computed_np = self.req_states.num_computed_tokens_np
        prompt_len_np = self.req_states.prompt_len.np

        # Short-circuit when nothing is active (no per-request config in this
        # batch, no override pool, no dynamic tier, no monitor, no globals). A
        # decode-only request (prefill_hash == 0, decode_hash != 0) registers
        # its config lazily at the transition below, so the batch scan must not
        # let the short-circuit swallow it.
        batch_has_per_request_steering = any(
            (rs := reqs.get(req_ids[i])) is not None
            and (rs.prefill_hash != 0 or rs.decode_hash != 0)
            for i in range(num_reqs)
        )
        if (
            not batch_has_per_request_steering
            and not mgr.config_to_row
            and not mgr.has_dynamic
            and not mgr.has_dynamic_tier
            and not mgr.has_monitor
            and not mgr.global_base_vectors
            and not mgr.global_prefill_vectors
            and not mgr.global_decode_vectors
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                steering_index = cast(torch.Tensor, any_layer.steering_index)
                steering_index.zero_()
                # Clear the per-token dynamic-tier gate so a stale gate doesn't
                # apply a now-removed tier.
                tscales = getattr(any_layer, "steering_token_scales", None)
                if tscales is not None:
                    tscales.zero_()
                # Reset the Phase 2 row gate to 1.0 and clear the decode mask so
                # a stale monitor reduction doesn't gate now-removed rows.
                rgate = getattr(any_layer, "steering_row_gate", None)
                if rgate is not None:
                    rgate.fill_(1.0)
                dmask = getattr(any_layer, "steering_decode_mask", None)
                if dmask is not None:
                    dmask.zero_()
                for mod in self._steerable_layers_cache.values():
                    for hp in SteeringHookPoint:
                        flag_buf = getattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp], None)
                        if flag_buf is not None:
                            flag_buf.zero_()
                        mon_buf = getattr(mod, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp], None)
                        if mon_buf is not None:
                            mon_buf.zero_()
                self._steering_index_dirty = False
            # Nothing dynamic is active; revert any request still reported as
            # dynamically steered back to its admitted decode key.
            self._pending_decode_sigs = self._compute_decode_signature_deltas_v2(
                scheduler_output, input_batch
            )
            return

        # 1. Populate tables only when state changed since the last populate.
        # The cheap scales-only path (§5.3) rewrites per-row scale buffers
        # without a table recompose.
        if mgr._tables_dirty:
            mgr.populate_steering_tables(self._steerable_layers_cache)
        elif mgr._scales_dirty:
            mgr.populate_steering_scales(self._steerable_layers_cache)

        # 2. Build the per-token steering index + tier gate + decode mask.
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = cast(torch.Tensor, any_layer.steering_index)

        rows_scratch = self._steering_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._steering_index_pinned
        tier_gain_scratch = self._steering_tier_gain_scratch
        token_scales_pinned = self._steering_token_scales_pinned
        decode_mask_scratch = self._steering_decode_mask_scratch
        decode_mask_pinned = self._steering_decode_mask_pinned
        assert rows_scratch is not None
        assert n_tokens_scratch is not None
        assert index_pinned is not None
        assert tier_gain_scratch is not None
        assert token_scales_pinned is not None
        assert decode_mask_scratch is not None
        assert decode_mask_pinned is not None
        if rows_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            tier_gain_scratch = np.zeros(num_reqs, dtype=np.float32)
            decode_mask_scratch = np.zeros(num_reqs, dtype=np.float32)
            self._steering_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch
            self._steering_tier_gain_scratch = tier_gain_scratch
            self._steering_decode_mask_scratch = decode_mask_scratch

        # Per-token dynamic-tier gate (§5.4): the gain for decode tokens of a
        # tier-active state, 0 otherwise (so the tier stays decode-only).
        tier_gain = mgr.dynamic_tier_gain if mgr.has_dynamic_tier else 0.0

        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_idx = int(idx_np[i])
            num_computed = int(num_computed_np[req_idx])
            # Untracked requests (no per-request config) still pass through
            # get_row_for_config with hash 0 so any global vectors apply — the
            # manager maps hash 0 to the global prefill/decode row (or to the
            # row-0 no-steer sentinel when no globals are set).
            rs = reqs.get(req_id)
            if rs is not None:
                num_prompt = rs.num_prompt_tokens
                prefill_hash = rs.prefill_hash
                decode_hash = rs.decode_hash
            else:
                num_prompt = int(prompt_len_np[req_idx])
                prefill_hash = 0
                decode_hash = 0

            if num_computed < num_prompt:
                row = mgr.get_row_for_config(prefill_hash, is_prefill=True)
                # Prefill tokens never get the dynamic tier or row gate (§7
                # cache safety).
                tier_gain_scratch[active_count] = 0.0
                decode_mask_scratch[active_count] = 0.0
                if rs is not None and num_computed + n_tokens >= num_prompt:
                    self._steering_transition(rs)
            else:
                # Decode: a live dynamic override routes this request's tokens
                # to its dynamic-pool row INSTEAD of the admitted config's row.
                # Pure routing — the admitted config stays registered.
                dyn_id = self._req_dynamic_decode.get(req_id)
                if dyn_id is not None:
                    row = mgr.get_dynamic_row(dyn_id)
                else:
                    row = mgr.get_row_for_config(decode_hash, is_prefill=False)
                # Decode tokens carry the dynamic-tier gate and are eligible
                # for in-graph row gating.
                tier_gain_scratch[active_count] = tier_gain
                decode_mask_scratch[active_count] = 1.0
            rows_scratch[active_count] = row
            n_tokens_scratch[active_count] = n_tokens
            active_count += 1

        # Single non-blocking H2D copy: expand per-request rows into the
        # per-token row array, then copy that prefix to the GPU in one shot.
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

        # Per-token dynamic-tier gate (§5.4): same expand + H2D as the index.
        token_scales = cast(torch.Tensor, any_layer.steering_token_scales)
        if active_count > 0:
            gate_expanded = np.repeat(
                tier_gain_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_gate = min(
                int(gate_expanded.shape[0]),
                token_scales_pinned.shape[0],
                token_scales.shape[0],
            )
            token_scales_pinned[:n_gate].copy_(
                torch.from_numpy(gate_expanded[:n_gate])
            )
            token_scales[:n_gate].copy_(
                token_scales_pinned[:n_gate], non_blocking=True
            )
        else:
            n_gate = 0
        if n_gate < token_scales.shape[0]:
            token_scales[n_gate:].zero_()

        # Phase 2 row gating: reset the per-token row gate to 1.0 (full
        # strength) so any monitor reduction from the previous step is cleared,
        # and write the decode mask (1.0 decode / 0.0 prefill) so the monitor
        # only gates decode rows. Both are no-ops downstream unless a row-gating
        # monitor is active. Mirrors the token_scales expand + H2D.
        row_gate = cast(torch.Tensor, any_layer.steering_row_gate)
        decode_mask = cast(torch.Tensor, any_layer.steering_decode_mask)
        row_gate.fill_(1.0)
        if active_count > 0:
            mask_expanded = np.repeat(
                decode_mask_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_mask = min(
                int(mask_expanded.shape[0]),
                decode_mask_pinned.shape[0],
                decode_mask.shape[0],
            )
            decode_mask_pinned[:n_mask].copy_(
                torch.from_numpy(mask_expanded[:n_mask])
            )
            decode_mask[:n_mask].copy_(
                decode_mask_pinned[:n_mask], non_blocking=True
            )
        else:
            n_mask = 0
        if n_mask < decode_mask.shape[0]:
            decode_mask[n_mask:].zero_()

        self._steering_index_dirty = True

        # Effective-decode-signature deltas for APC (computed from the steering
        # state as applied THIS step — before any sync consumer mutates it for
        # the next step).
        self._pending_decode_sigs = self._compute_decode_signature_deltas_v2(
            scheduler_output, input_batch
        )

    def _compute_decode_signature_deltas_v2(
        self, scheduler_output: SchedulerOutput, input_batch: InputBatch
    ) -> dict[str, int]:
        """v2 port of ``_compute_decode_signature_deltas`` (APC §; see
        docs/design/dynamic_steering_apc_notification.md).

        For each decode request, fold its admitted decode config with any live
        override / tier / monitor into an effective signature and report only
        the requests whose signature changed since the last report. The decode
        hash comes from ``_steering_reqs`` (v2 keeps no input-batch hash
        columns); phase comes from ``req_states``.
        """
        mgr = self._steering_manager
        if mgr is None:
            return {}
        reqs = self._steering_reqs
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_np = input_batch.idx_mapping_np
        num_computed_np = self.req_states.num_computed_tokens_np
        prompt_len_np = self.req_states.prompt_len.np
        deltas: dict[str, int] = {}
        seen: set[str] = set()
        for i in range(num_reqs):
            req_id = req_ids[i]
            if req_id is None:
                continue
            if scheduler_output.num_scheduled_tokens.get(req_id, 0) == 0:
                continue
            req_idx = int(idx_np[i])
            num_computed = int(num_computed_np[req_idx])
            rs = reqs.get(req_id)
            num_prompt = (
                rs.num_prompt_tokens if rs is not None else int(prompt_len_np[req_idx])
            )
            if num_computed < num_prompt:
                # Prefill: decode steering (and its signature) does not apply.
                continue
            seen.add(req_id)
            base = rs.decode_hash if rs is not None else 0
            dyn_id = self._req_dynamic_decode.get(req_id)
            sig = mgr.effective_decode_signature(dyn_id, base)
            report_val = base if sig is None else sig
            if self._req_decode_sig_reported.get(req_id) != report_val:
                deltas[req_id] = report_val
                self._req_decode_sig_reported[req_id] = report_val

        # Drop reported state for requests no longer in the decode batch.
        if self._req_decode_sig_reported:
            for rid in [
                r for r in self._req_decode_sig_reported if r not in seen
            ]:
                self._req_decode_sig_reported.pop(rid, None)
        return deltas

    def _apply_request_override(
        self,
        action: RequestSteeringOverride,
        *,
        source: str,
    ) -> bool:
        """v2 port of the per-request dynamic decode override apply.

        Identical routing semantics to v1 (allocate/update/release a
        dynamic-pool row recorded in ``_req_dynamic_decode``; admitted config
        lifecycle untouched), but the decode-only phase guard reads v2's
        ``req_states`` (``req_id_to_index`` + ``num_computed_tokens_np`` /
        ``prompt_len.np``) instead of the v1 input batch.
        """
        mgr = self._steering_manager
        req_id = action.req_id

        def _reject(reason: str) -> bool:
            logger.warning(
                "rejected dynamic steering override (source=%s, req=%s): %s",
                source,
                req_id,
                reason,
            )
            return False

        if mgr is None or not self._steerable_layers_cache:
            return _reject("steering is not initialized on this worker")
        if mgr.max_dynamic_steering_configs <= 0:
            return _reject(
                "dynamic override pool is disabled "
                "(max_dynamic_steering_configs=0)"
            )

        existing_dyn_id = self._req_dynamic_decode.get(req_id)

        # Clear: revert to admitted routing. Idempotent disengage.
        if action.vectors is None:
            if existing_dyn_id is not None:
                self._req_dynamic_decode.pop(req_id, None)
                mgr.release_dynamic_config(existing_dyn_id)
            return True

        req_idx = self.req_states.req_id_to_index.get(req_id)
        if req_idx is None:
            return _reject("request is not in the batch")
        num_computed = int(self.req_states.num_computed_tokens_np[req_idx])
        num_prompt = int(self.req_states.prompt_len.np[req_idx])
        if num_computed < num_prompt:
            return _reject(
                "request is still prefilling (overrides are decode-only; "
                "prefill steering feeds prefix-cache keys)"
            )
        try:
            validate_steering_vectors(action.vectors, self._steerable_layers_cache)
        except SteeringVectorError as exc:
            return _reject(str(exc))

        if existing_dyn_id is not None:
            mgr.update_dynamic_config(
                existing_dyn_id,
                action.vectors,
                locally_owned_layers=self._locally_owned_layers,
            )
            return True
        try:
            dyn_id, _row = mgr.register_dynamic_config(
                action.vectors,
                locally_owned_layers=self._locally_owned_layers,
            )
        except RuntimeError as exc:
            return _reject(str(exc))
        self._req_dynamic_decode[req_id] = dyn_id
        return True
