# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-capture control plane for the v2 GPU model runner.

The capture *data plane* (the ``capture_residual`` custom op, per-layer taps,
persistent global buffers, kernels) lives in ``vllm/model_executor`` and
``vllm/v1/capture`` and is shared with the v1 runner unchanged. This mixin is
only the v2 runner-side glue: it builds the managers/gate/store, registers and
finalizes requests, drives the per-step force-eager decision and gather plan,
and drains finalized results onto ``ModelRunnerOutput``.

It deliberately keeps its own per-request bookkeeping rather than piggybacking
on runner state, because v2's ``RequestState`` does not retain ``sampling_params``
or capture specs past ``add_requests``.

See ``docs/design/v2_runner_steering_capture.md`` for the full contract.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    from vllm.v1.capture.plan import CaptureBatchView
    from vllm.v1.capture.types import CaptureResult
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


class CaptureRunnerMixin:
    """Mixin adding the activation-capture control plane to the v2 runner.

    Assumes the concrete runner provides: ``vllm_config``, ``model_config``,
    ``parallel_config``, ``device``, ``max_num_tokens`` and ``req_states``.
    """

    # ---- state (set by _init_capture_state) -------------------------------
    _capture_feature_enabled: bool = False
    _capture_piecewise_fallback_enabled: bool = False
    _capture_manager: Any = None
    _capture_step_gate: Any = None
    _capture_validators: list[Any]
    _capture_name_to_index: dict[str, int]
    _capture_index_to_name: dict[int, str]
    _pending_capture_results: dict[str, dict[str, CaptureResult]]
    _pending_capture_results_lock: threading.Lock

    def _init_capture_state(self) -> None:
        """Construct the capture managers/gate/store. Idempotent no-op when
        capture is not configured.

        Mirrors the v1 runner's ``__init__`` capture block: the rank-replicated
        ``CaptureStepGate`` is built on every rank so the eager-vs-cudagraph
        choice agrees across the TP/PP topology without a per-step collective,
        while the ``CaptureManager`` and process-global active manager are
        installed only on TP rank 0 (the residual stream is byte-identical
        across the TP group after the all-reduce, so exactly one rank captures).
        """
        self._capture_manager = None
        self._capture_step_gate = None
        self._capture_validators = []
        self._capture_name_to_index = {}
        self._capture_index_to_name = {}
        self._pending_capture_results = {}
        self._pending_capture_results_lock = threading.Lock()

        cc_config = self.vllm_config.capture_consumers_config
        self._capture_feature_enabled = cc_config is not None
        # Capture-aware piecewise cudagraph fallback (opt-in). When enabled,
        # the capture op is a graph split point (registered at config time) so
        # a per-request capture step replays the piecewise cudagraph instead of
        # forcing the whole step eager. The runtime fallback in the model
        # runner is gated on this flag for soundness — without the split op the
        # dynamic gather would land inside a cudagraphed segment.
        self._capture_piecewise_fallback_enabled = (
            cc_config is not None and cc_config.piecewise_capture_fallback
        )
        if cc_config is None:
            return

        from vllm.model_executor.layers.activation_capture import (
            set_active_capture_manager,
        )
        from vllm.v1.capture.step_gate import CaptureStepGate

        # Global capture specs ride a CUDA-graph-safe persistent-buffer path, so
        # the gate forces eager only when a per-request *client* spec captures.
        # The graph-safe allowlist (a rank-identical config value) further lets
        # a client spec tapping only allowlisted keys skip eager too; the gate
        # is built with the global, unfiltered key set so every rank agrees.
        graphsafe_keys = frozenset(getattr(cc_config, "graphsafe_keys", None) or ())
        self._capture_step_gate = CaptureStepGate(graphsafe_keys=graphsafe_keys)

        if get_tp_group().rank_in_group != 0:
            # Non-capturer rank: no manager, cold-path custom op.
            set_active_capture_manager(None)
        else:
            from vllm.v1.capture import registry as _capture_registry
            from vllm.v1.capture.manager import CaptureManager

            instances = list(cc_config.instances)
            sinks, validators, name_to_index = _capture_registry.build_consumers(
                self.vllm_config, consumer_instances=instances
            )

            global_specs: list[Any] = []
            for validator in validators:
                spec = None
                try:
                    if hasattr(validator, "global_capture_spec"):
                        spec = validator.global_capture_spec()
                except Exception:
                    spec = None
                global_specs.append(spec)

            self._capture_manager = CaptureManager(
                consumers=sinks,
                consumer_specs=tuple(global_specs),
                num_hidden_layers=self.model_config.get_total_num_hidden_layers(),
                local_layer_range=self.model_config.get_layers_start_end_indices(
                    self.parallel_config
                ),
                hidden_size=self.model_config.get_hidden_size(),
                model_dtype=self.model_config.dtype,
                device=self.device,
                max_num_tokens=self.max_num_tokens,
                dispatch_queue_size=getattr(cc_config, "dispatch_queue_size", 256),
                overload_policy=getattr(cc_config, "overload_policy", "spill"),
                spill_dir=getattr(cc_config, "spill_dir", None),
                spill_max_bytes=getattr(cc_config, "spill_max_bytes", 4 << 30),
                graphsafe_keys=getattr(cc_config, "graphsafe_keys", None),
            )
            self._capture_validators = validators
            self._capture_name_to_index = dict(name_to_index)
            self._capture_index_to_name = {
                idx: name for name, idx in name_to_index.items()
            }
            set_active_capture_manager(self._capture_manager)

        budget = cc_config.activation_cache_bytes
        if budget > 0:
            from vllm.v1.capture.activation_store import (
                ActivationStore,
                set_active_activation_store,
            )

            set_active_activation_store(ActivationStore(max_bytes=budget))
            logger.info(
                "Capture activation store enabled: budget=%.3f GB",
                budget / 1_000_000_000,
            )

    # ---- request lifecycle -------------------------------------------------

    def _capture_add_request(
        self, new_req_data: NewRequestData, was_present: bool
    ) -> None:
        """Hook a newly admitted request into capture tracking.

        ``register`` runs on every rank (the gate is rank-replicated);
        manager registration runs only on the capturer rank.

        ``was_present`` is ``True`` only for a **streaming re-add** — a still
        live request re-admitted with a grown prompt. In that case the prior
        chunk's gate selector and manager registration are stale and must be
        discarded (without finalizing — a partial first-chunk capture is
        dropped, not emitted) before re-registering against the new prompt.
        This mirrors the steering control plane's re-add handling.

        A **preemption resume** reaches this path too (the v2 scheduler folds
        ``scheduled_resumed_reqs`` into ``scheduled_new_reqs``), but with
        ``was_present`` ``False`` because ``finish_requests`` removed the
        request on preemption while intentionally leaving its capture
        registration open. Such a request is re-prefilled (recompute) into the
        existing registration, so we keep it and skip re-registration — both
        avoiding the manager's duplicate-register error and preserving any
        rows already captured before preemption.
        """
        if not self._capture_feature_enabled:
            return
        req_id = new_req_data.req_id
        mgr = self._capture_manager
        if was_present:
            # Streaming re-add: prior chunk's capture state is stale.
            if self._capture_step_gate is not None:
                self._capture_step_gate.drop(req_id)
            if mgr is not None:
                mgr.unregister_request(req_id)
        sp = new_req_data.sampling_params
        if self._capture_step_gate is not None:
            self._capture_step_gate.register(req_id, getattr(sp, "capture", None))
        # Skip re-registration if the request is already registered (a
        # preemption resume whose registration survived); otherwise admit it.
        if mgr is not None and not mgr.has_request(req_id):
            self._register_capture_request(new_req_data)

    def _capture_finish_request(self, req_id: str) -> None:
        """Drop a finished request from the gate and finalize its capture."""
        if not self._capture_feature_enabled:
            return
        if self._capture_step_gate is not None:
            self._capture_step_gate.drop(req_id)
        if self._capture_manager is not None:
            self._finalize_capture_for_request_async(req_id)

    def _register_capture_request(self, new_req_data: NewRequestData) -> None:
        """Admit a request into the capture framework (capturer rank only).

        Resolves ``sampling_params.capture`` consumer names against the registry,
        validates raw specs, and registers with the manager. Admission errors
        never abort generation — they surface as ``CaptureResult(status="error")``
        on finalize.
        """
        assert self._capture_manager is not None
        mgr = self._capture_manager

        sp = new_req_data.sampling_params
        if sp is None:
            return

        prompt_len = length_from_prompt_token_ids_or_embeds(
            new_req_data.prompt_token_ids,
            new_req_data.prompt_embeds,
        )

        try:
            element_size_bytes = get_dtype_size(self.model_config.dtype)
        except Exception:
            element_size_bytes = 2

        from vllm.v1.capture.activation_store import pop_pending_serve
        from vllm.v1.capture.errors import CaptureValidationError
        from vllm.v1.capture.types import (
            CaptureContext,
            CaptureSpec,
            VllmInternalRequestId,
            capture_expert_parallel_size,
        )

        # Step A serve: whole-prefix store serve means the prompt prefix was
        # reused from KV cache; validate against num_computed=0 so the validator
        # accepts those positions, then inject served rows after registration.
        served_rows = pop_pending_serve(new_req_data.req_id)
        ctx_num_computed = (
            0 if served_rows is not None else new_req_data.num_computed_tokens
        )

        parallel_config = self.parallel_config
        ctx = CaptureContext(
            vllm_internal_request_id=VllmInternalRequestId(new_req_data.req_id),
            num_prompt_tokens=prompt_len,
            num_computed_tokens=ctx_num_computed,
            num_hidden_layers=self.model_config.get_total_num_hidden_layers(),
            hidden_size=self.model_config.get_hidden_size(),
            element_size_bytes=element_size_bytes,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            expert_parallel_size=capture_expert_parallel_size(parallel_config),
            data_parallel_size=parallel_config.data_parallel_size,
        )

        raw_client = getattr(sp, "capture", None)
        client_specs: dict[int, CaptureSpec] = {}

        if raw_client:
            if not isinstance(raw_client, dict):
                mgr.record_request_error(
                    new_req_data.req_id,
                    "SamplingParams.capture must be a dict keyed by consumer "
                    f"name, got {type(raw_client).__name__}",
                )
                return

            for name, raw in raw_client.items():
                idx = self._capture_name_to_index.get(name)
                if idx is None:
                    mgr.record_request_error(
                        new_req_data.req_id,
                        f"capture consumer {name!r} is not registered; known "
                        f"consumers: {sorted(self._capture_name_to_index)}",
                    )
                    logger.warning(
                        "capture admission rejected req=%s: unknown consumer %s",
                        new_req_data.req_id,
                        name,
                    )
                    return

                if isinstance(raw, CaptureSpec):
                    client_specs[idx] = raw
                    continue

                validator = self._capture_validators[idx]
                try:
                    resolved = validator.validate_client_spec(raw, ctx)
                except CaptureValidationError as exc:
                    mgr.record_request_error(new_req_data.req_id, str(exc))
                    logger.warning(
                        "capture admission rejected req=%s consumer=%s: %s",
                        new_req_data.req_id,
                        name,
                        exc,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    mgr.record_request_error(
                        new_req_data.req_id,
                        f"consumer {name!r} validator raised: {exc}",
                    )
                    logger.warning(
                        "capture admission rejected req=%s consumer=%s: %s",
                        new_req_data.req_id,
                        name,
                        exc,
                    )
                    return

                client_specs[idx] = resolved

        sidecar_fields: dict[str, Any] = {
            "vllm_internal_request_id": new_req_data.req_id,
            # Client-supplied request id for universal attribution. Falls
            # back to the internal id if randomization was disabled / the
            # client id is unavailable (None-safe).
            "client_request_id": (
                new_req_data.client_request_id
                if new_req_data.client_request_id is not None
                else new_req_data.req_id
            ),
            "prompt_token_ids": (
                list(new_req_data.prompt_token_ids)
                if new_req_data.prompt_token_ids is not None
                else []
            ),
        }

        try:
            mgr.register_request(
                new_req_data.req_id,
                client_specs=client_specs,
                num_prompt_tokens=prompt_len,
                sidecar_fields=sidecar_fields,
                block_hashes=new_req_data.capture_block_hashes,
                hash_block_size=new_req_data.capture_hash_block_size,
            )
            if served_rows is not None:
                mgr.serve_from_store(new_req_data.req_id, served_rows)
        except ValueError as exc:
            mgr.record_request_error(new_req_data.req_id, str(exc))
            logger.warning(
                "capture register rejected req=%s: %s", new_req_data.req_id, exc
            )

    # ---- per-step ----------------------------------------------------------

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

    def _finalize_capture_step(self) -> None:
        """Dispatch captured rows to consumer sinks (after the forward)."""
        if self._capture_manager is None:
            return
        plan = self._capture_manager.consume_step_plan()
        if plan is None:
            return
        self._capture_manager.dispatch_step_captures(plan)

    def _finalize_capture_for_request_async(self, req_id: str) -> None:
        """Finalize *req_id* off the step thread; stash results for draining."""
        mgr = self._capture_manager
        if mgr is None:
            return

        index_to_name = self._capture_index_to_name

        def _on_complete(indexed: dict[int, CaptureResult]) -> None:
            if not indexed:
                return
            named = {
                index_to_name.get(idx, f"consumer_{idx}"): result
                for idx, result in indexed.items()
            }
            # Resolve the stash dict at CALL time, not closure-creation time:
            # ``_drain_capture_results`` / ``drain_pending_capture_results``
            # swap ``self._pending_capture_results`` for a fresh dict, so a
            # reference captured here would be orphaned by the time the
            # finalize thread fires (~seconds later, after writer fsync) and
            # the results would silently vanish -- the request would never
            # report capture results (and ``capture_wait`` would hang).
            with self._pending_capture_results_lock:
                self._pending_capture_results.setdefault(req_id, {}).update(
                    named
                )

        mgr.finalize_request_async(req_id, _on_complete)

    def _drain_capture_results(self) -> dict[str, dict[str, CaptureResult]]:
        """Atomically take the finalized results buffered since the last step."""
        if not self._capture_feature_enabled:
            return {}
        with self._pending_capture_results_lock:
            results = self._pending_capture_results
            self._pending_capture_results = {}
        return results

    def drain_pending_capture_results(
        self,
    ) -> dict[str, dict[str, CaptureResult]]:
        """collective_rpc target for the ``capture_wait`` idle-loop drain.

        The engine core's idle loop calls this (via the worker's
        ``drain_pending_capture_results``) so capture results that finalize
        after a request's final step still reach ``capture_wait`` clients,
        even when no further ``ModelRunnerOutput`` is produced. Mirrors the
        v1 ``GPUModelRunner`` method; shares the per-step swap buffer with
        :meth:`_drain_capture_results`.
        """
        return self._drain_capture_results()
