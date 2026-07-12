# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared activation-capture control plane for the GPU model runners.

The capture *data plane* (the ``capture_residual`` custom op, per-layer taps,
persistent global buffers, kernels) lives in ``vllm/model_executor`` and
``vllm/v1/capture`` and is shared with both runners unchanged. This mixin is
the runner-side glue shared by the v1 (``gpu_model_runner.GPUModelRunner``) and
v2 (``gpu/model_runner.GPUModelRunner``) runners: it builds the
managers/gate/store, registers and finalizes requests, runs the sync-execution
consumers on the step thread, and drains finalized results.

The two runners diverge in only a few spots, expressed here as hooks the
concrete runner overrides:

* ``_iter_step_capture_rows`` — enumerate the current step's batch rows as
  ``(req_id, start, end, num_computed, num_prompt, row_index)`` tuples. v1
  walks its persistent ``input_batch`` accumulating scheduled-token offsets;
  v2 reads ``query_start_loc_np`` off the freshly built ``InputBatch`` and the
  ``req_states`` arrays. The offset source is fetched once per step, not per
  token.
* ``_step_view_token_ids`` — the per-request CPU token window for
  ``StepRequestView.token_ids``. v1 copies the ``token_ids_cpu`` slice; v2
  returns a shared empty array (its token ids are GPU-resident, so mirroring
  them would force a per-step D2H the v2 runner avoids).

The force-eager gate view and gather-plan view are *not* shared: v2 must decide
cudagraph-vs-eager before its ``InputBatch`` exists, so those live on the v2
mixin.

See ``docs/design/v2_runner_steering_capture.md`` for the full contract.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    import numpy as np

    from vllm.v1.capture.types import CaptureResult
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

logger = init_logger(__name__)


class CaptureRunnerMixin:
    """Runner-agnostic activation-capture control plane.

    Assumes the concrete runner provides: ``vllm_config``, ``model_config``,
    ``parallel_config``, ``device``, ``dtype``, ``max_num_tokens``, and the two
    per-step hooks ``_iter_step_capture_rows`` / ``_step_view_token_ids``. The
    steering apply path (``_apply_steering_actions``) is resolved on the
    concrete runner via its steering mixin.
    """

    # ---- state (set by _init_capture_state) -------------------------------
    _capture_feature_enabled: bool = False
    _capture_piecewise_fallback_enabled: bool = False
    # Set True only while the v2 runner's ``warmup_kernels`` drives synthetic
    # forwards through the real ``execute_model`` (a v2-only JIT-compile pass;
    # v1 instead warms up via ``_dummy_run``, which never reaches
    # ``execute_model``). Gates sync consumers off those startup warmup steps
    # so consumer policy never runs on dummy activations. Always False on v1.
    _in_kernel_warmup: bool = False
    _capture_manager: Any = None
    _capture_step_gate: Any = None
    _capture_validators: list[Any]
    _capture_name_to_index: dict[str, int]
    _capture_index_to_name: dict[int, str]
    _pending_capture_results: dict[str, dict[str, CaptureResult]]
    _pending_capture_results_lock: threading.Lock
    # Sync-execution consumers (``execution="sync"``): constructed on EVERY TP
    # rank, run post-forward on the step thread via ``_run_sync_consumers``,
    # reading the persistent global capture buffers. ``_sync_capture_buffers``
    # is whichever manager owns those buffers on this rank — the full manager
    # on TP rank 0, a slim (buffers-only) manager elsewhere.
    _sync_consumers: list[tuple[str, Any]]
    _sync_capture_buffers: Any
    _sync_monitor_keys: list[tuple[int, str]]
    _sync_consumer_stats: dict[str, dict[str, Any]]
    _sync_step_counter: int
    _sync_timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] | None

    def _init_capture_state(self) -> None:
        """Construct the capture managers/gate/store. Idempotent no-op when
        capture is not configured.

        The rank-replicated ``CaptureStepGate`` is built on every rank so the
        eager-vs-cudagraph choice agrees across the TP/PP topology without a
        per-step collective, while the ``CaptureManager`` and process-global
        active manager are installed only on TP rank 0 (the residual stream is
        byte-identical across the TP group after the all-reduce, so exactly one
        rank captures). Sync consumers are constructed on *every* rank.
        """
        self._capture_manager = None
        self._capture_step_gate = None
        self._capture_validators = []
        self._capture_name_to_index = {}
        self._capture_index_to_name = {}
        self._pending_capture_results = {}
        self._pending_capture_results_lock = threading.Lock()
        # Sync-execution consumer state (see class docstring). Populated below
        # on every rank when sync consumers are configured. ``_sync_timing_events``
        # holds per-consumer CUDA event pairs measuring the *added* GPU time of
        # ``on_step`` (None off CUDA, e.g. CPU tests).
        self._sync_consumers = []
        self._sync_capture_buffers = None
        self._sync_monitor_keys = []
        self._sync_consumer_stats = {}
        self._sync_step_counter = 0
        self._sync_timing_events = None
        # req_id -> client conversation id
        # (``RequestMetadata.conversation_id``). Stashed here at admission from
        # the request metadata and read back when building the per-step view.
        self._sync_conversation_ids: dict[str, str | None] = {}
        # req_id -> declarative per-request steering gates
        # (``RequestMetadata.steering``), unpacked to numpy once at admission
        # and surfaced on ``StepRequestView.steering``.
        self._sync_steering_gates: dict[str, list | None] = {}

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
            from vllm.v1.capture import registry as _capture_registry
            from vllm.v1.capture.manager import CaptureManager

            # Sync consumers exist on every rank; async consumers must NOT be
            # constructed here (their constructors have side effects — writer
            # threads, open files).
            self._sync_consumers = _capture_registry.build_sync_consumers(
                self.vllm_config
            )
            if not self._sync_consumers:
                # Non-capturer rank with no sync consumers: cold-path custom op.
                set_active_capture_manager(None)
            else:
                # Buffers-only manager: gives this rank the same graph-baked
                # full-residual ``copy_`` for the sync monitor keys that rank 0
                # gets, with no dispatch pipeline. ``_capture_manager`` stays
                # None so every rank-0-only call site short-circuits.
                slim_mgr = CaptureManager(
                    consumers=(),
                    consumer_specs=(),
                    extra_global_specs=tuple(
                        c.global_capture_spec() for _, c in self._sync_consumers
                    ),
                    num_hidden_layers=(self.model_config.get_total_num_hidden_layers()),
                    local_layer_range=self.model_config.get_layers_start_end_indices(
                        self.parallel_config
                    ),
                    hidden_size=self.model_config.get_hidden_size(),
                    model_dtype=self.model_config.dtype,
                    device=self.device,
                    max_num_tokens=self.max_num_tokens,
                    slim=True,
                )
                self._sync_capture_buffers = slim_mgr
                set_active_capture_manager(slim_mgr)
        else:
            from vllm.v1.capture import registry as _capture_registry
            from vllm.v1.capture.manager import CaptureManager

            instances = list(cc_config.instances)
            (
                sinks,
                validators,
                name_to_index,
                sync_consumers,
            ) = _capture_registry.build_consumers(
                self.vllm_config, consumer_instances=instances
            )
            self._sync_consumers = sync_consumers

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
                # Sync consumers' monitor keys get persistent buffers without
                # sink slots; they read the buffers directly on the step thread.
                extra_global_specs=tuple(
                    c.global_capture_spec() for _, c in sync_consumers
                ),
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
            # Sync consumers on rank 0 read the full manager's buffers.
            self._sync_capture_buffers = self._capture_manager

        # Sorted union of every sync consumer's monitored (layer, hook) keys —
        # deterministic iteration order for the per-step view build on every
        # rank — plus the CUDA-event timers for the honest per-step added cost.
        if self._sync_consumers:
            monitor_keys: set[tuple[int, str]] = set()
            for _name, sync_consumer in self._sync_consumers:
                sync_spec = sync_consumer.global_capture_spec()
                for hook_name, layers in sync_spec.hooks.items():
                    monitor_keys.update((int(layer), hook_name) for layer in layers)
            self._sync_monitor_keys = sorted(monitor_keys)
            from vllm.v1.capture.config import graphsafe_buffer_bytes

            footprint = graphsafe_buffer_bytes(
                num_keys=len(self._sync_monitor_keys),
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.model_config.get_hidden_size(),
                dtype_bytes=self.model_config.dtype.itemsize,
            )
            logger.info(
                "sync capture consumers active: %s (monitor keys: %s; "
                "persistent capture buffers: %d sites x %d tokens x %d hidden "
                "= %.1f MiB VRAM)",
                [name for name, _ in self._sync_consumers],
                self._sync_monitor_keys,
                len(self._sync_monitor_keys),
                self.max_num_tokens,
                self.model_config.get_hidden_size(),
                footprint / (1024 * 1024),
            )
            if getattr(self.device, "type", None) == "cuda":
                self._sync_timing_events = {
                    name: (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for name, _ in self._sync_consumers
                }

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
        if self._in_kernel_warmup:
            # ``warmup_kernels`` drives synthetic requests through the real
            # execute_model path (v2). They must never enter capture tracking:
            # an admitted warmup request would finalize on finish and deliver
            # a (empty) ``on_capture`` to consumers before any real request.
            return
        req_id = new_req_data.req_id
        # Stash the conversation id for the per-step view (host-side metadata,
        # survives the streaming re-add / preemption-resume branches below).
        rmeta = new_req_data.request_metadata
        self._sync_conversation_ids[req_id] = (
            rmeta.conversation_id if rmeta is not None else None
        )
        if rmeta is not None and rmeta.steering is not None:
            from vllm.v1.steering_schema import resolve_gates_safe
            from vllm.v1.worker.steering_vector_registry import (
                get_worker_steering_vector_registry,
            )

            self._sync_steering_gates[req_id] = resolve_gates_safe(
                rmeta.steering,
                req_id,
                get_worker_steering_vector_registry(),
            )
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
        self._sync_conversation_ids.pop(req_id, None)
        self._sync_steering_gates.pop(req_id, None)
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
            # Global layer count: client specs reference global layer indices,
            # so admission must validate against the full layer space even on a
            # pipeline stage that owns only a slice.
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
                    # A buggy validator should never take down the request —
                    # surface it as an admission error.
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

        # Sidecar fields the manager echoes to each consumer on finalize.
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

    def _finalize_capture_step(self) -> None:
        """Dispatch captured rows to every consumer's sink (after the forward).

        The manager's ``dispatch_step_captures`` fan-outs chunks to every sink
        whose bit is set in the plan's per-entry ``consumer_mask``; consumer
        errors are isolated so a failing sink never stops delivery to others.
        """
        if self._capture_manager is None:
            return
        plan = self._capture_manager.consume_step_plan()
        if plan is None:
            return
        self._capture_manager.dispatch_step_captures(plan)

    # ---- sync-execution consumers ------------------------------------------

    def _iter_step_capture_rows(
        self, scheduler_output: SchedulerOutput, input_batch: Any
    ):
        """Yield ``(req_id, start, end, num_computed, num_prompt, row_index)``
        for each request contributing tokens to this step's forward batch.

        ``start``/``end`` are contiguous offsets into the global capture
        buffers (the forward batch layout the capture op wrote), and
        ``row_index`` is the concrete runner's per-request slot passed back to
        :meth:`_step_view_token_ids`. Runner-specific; overridden by each
        concrete runner.
        """
        raise NotImplementedError

    def _step_view_token_ids(
        self, row_index: int, num_computed: int, n_tokens: int
    ) -> np.ndarray:
        """Return the request's CPU input-token window for this step.

        Runner-specific: v1 copies the ``token_ids_cpu`` slice; v2 returns a
        shared empty array (its token ids are GPU-resident).
        """
        raise NotImplementedError

    def _build_step_capture_view(
        self, scheduler_output: SchedulerOutput, input_batch: Any = None
    ):
        """Build the :class:`StepCaptureView` for sync consumers.

        Every input is rank-identical — ``scheduler_output`` is broadcast and
        the batch arrays read through the runner hooks are maintained on every
        rank — so the view (and any pure consumer decision derived from it)
        agrees across the TP group without communication. Token spans index the
        global buffers the capture op wrote after the tensor-parallel
        all-reduce, so request ``start``/``end`` index the same buffer rows on
        every rank.
        """
        from vllm.v1.capture.step_view import StepCaptureView, StepRequestView

        # Real scheduled token count, NOT the cudagraph-padded forward size:
        # the global buffer's leading rows hold the real tokens, so slicing to
        # the padded size would expose padding rows.
        num_tokens = int(scheduler_output.total_num_scheduled_tokens)
        tensors: dict[tuple[int, str], torch.Tensor] = {}
        for key in self._sync_monitor_keys:
            buf = self._sync_capture_buffers.global_buffer(key)
            if buf is not None:
                tensors[key] = buf[:num_tokens]

        requests: list[StepRequestView] = []
        for row in self._iter_step_capture_rows(scheduler_output, input_batch):
            req_id, start, end, num_computed, num_prompt, row_index = row
            requests.append(
                StepRequestView(
                    req_id=req_id,
                    start=start,
                    end=end,
                    phase="prefill" if num_computed < num_prompt else "decode",
                    token_ids=self._step_view_token_ids(
                        row_index, num_computed, end - start
                    ),
                    conversation_id=self._sync_conversation_ids.get(req_id),
                    steering=self._sync_steering_gates.get(req_id),
                )
            )

        return StepCaptureView(
            step=self._sync_step_counter,
            tensors=tensors,
            requests=requests,
        )

    def _run_sync_consumers(
        self, scheduler_output: SchedulerOutput, input_batch: Any = None
    ) -> None:
        """Run every sync consumer's ``on_step`` on the step thread.

        Called post-forward (after capture dispatch) on **every** TP rank.
        Consumer exceptions are isolated; returned steering actions apply inline
        through ``_apply_steering_actions`` (the steering mixin, resolved on the
        concrete runner) so they are visible to the next step's steering-buffer
        update.

        Two timings are kept per consumer in ``_sync_consumer_stats``: a
        wall-clock ``perf_counter`` span (a diagnostic that absorbs the forward
        drain via the consumer's D2H sync) and the CUDA-event GPU time of only
        the consumer's own enqueued work (the honest added cost, read one step
        late so it never blocks). The budget check (consumer attr
        ``sync_budget_ms``, default 5.0) charges the GPU time when available and
        is metric + warning only, never an automatic disable.
        """
        self._sync_step_counter += 1
        view = self._build_step_capture_view(scheduler_output, input_batch)
        events_by_name = getattr(self, "_sync_timing_events", None)
        for name, consumer in self._sync_consumers:
            stats = self._sync_consumer_stats.setdefault(
                name,
                {
                    "steps": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "gpu_steps": 0,
                    "gpu_total_ms": 0.0,
                    "gpu_max_ms": 0.0,
                    "gpu_last_ms": None,
                    "over_budget_steps": 0,
                    "ring": deque(maxlen=256),
                    "_gpu_armed": False,
                    "_last_budget_warn": 0.0,
                },
            )
            events = events_by_name.get(name) if events_by_name is not None else None

            # Deferred read: events recorded on this consumer's PREVIOUS step
            # are usually complete now, so ``elapsed_time`` normally neither
            # blocks nor forces a sync. A consumer whose ``on_step`` does no D2H
            # may leave the end event incomplete, so guard with ``query()`` and
            # drop the sample when not ready (best-effort metric).
            gpu_ms: float | None = None
            if events is not None and stats["_gpu_armed"] and events[1].query():
                gpu_ms = events[0].elapsed_time(events[1])
                stats["gpu_steps"] += 1
                stats["gpu_total_ms"] += gpu_ms
                stats["gpu_max_ms"] = max(stats["gpu_max_ms"], gpu_ms)
                stats["gpu_last_ms"] = round(gpu_ms, 3)

            if events is not None:
                events[0].record()
            t0 = time.perf_counter()
            try:
                actions = consumer.on_step(view)
            except Exception:
                logger.exception(
                    "sync capture consumer %s failed in on_step; "
                    "continuing without its actions",
                    name,
                )
                actions = None
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if events is not None:
                events[1].record()
                stats["_gpu_armed"] = True

            stats["steps"] += 1
            stats["total_ms"] += wall_ms
            stats["max_ms"] = max(stats["max_ms"], wall_ms)
            n_actions = len(actions) if actions else 0
            stats["ring"].append(
                (self._sync_step_counter, round(wall_ms, 3), n_actions)
            )

            # Charge the honest added GPU cost when we have a CUDA-event reading;
            # off CUDA (CPU tests) fall back to wall time.
            charge_ms = gpu_ms if events is not None else wall_ms
            budget_ms = float(getattr(consumer, "sync_budget_ms", 5.0))
            if charge_ms is not None and charge_ms > budget_ms:
                stats["over_budget_steps"] += 1
                now = time.monotonic()
                if now - stats["_last_budget_warn"] >= 5.0:
                    stats["_last_budget_warn"] = now
                    logger.warning(
                        "sync capture consumer %s on_step used %.2f ms of GPU "
                        "time (budget %.2f ms, %d over-budget steps so far); "
                        "this runs on the model-runner critical path",
                        name,
                        charge_ms,
                        budget_ms,
                        stats["over_budget_steps"],
                    )
            if actions:
                self._apply_steering_actions(actions, source=name)

    def _warmup_sync_consumers(self) -> None:
        """Let each sync consumer warm device-side compute before the first real
        ``on_step``.

        A consumer's first ``on_step`` GEMV pays a one-time init cost (cuBLAS
        handle creation, lazy kernel JIT) that would otherwise land on the first
        served step (on the critical path) and skew the ``gpu_*`` timing.
        Running it during graph capture moves it off the hot path. Optional and
        duck-typed (``warmup(device, dtype)``); isolated so a raising consumer
        cannot abort model setup. Skipped under ``enforce_eager`` (no capture).
        """
        for name, consumer in self._sync_consumers:
            warmup = getattr(consumer, "warmup", None)
            if not callable(warmup):
                continue
            try:
                warmup(self.device, self.dtype)
            except Exception:
                logger.exception(
                    "sync capture consumer %s warmup failed; continuing", name
                )

    # ---- finalize / drains -------------------------------------------------

    def _finalize_capture_for_request_async(self, req_id: str) -> None:
        """Finalize *req_id* off the step thread; stash results for draining.

        The manager pops the request's capture state here (on the step thread)
        and runs the blocking finalize on its finalize thread. The callback maps
        consumer indices to names and stashes the ``{consumer_name:
        CaptureResult}`` dict on ``_pending_capture_results`` (under the lock,
        since it fires on the finalize thread) for a later ``ModelRunnerOutput``
        to drain. A no-op when the manager never saw the request.
        """
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
            # reference captured here would be orphaned by the time the finalize
            # thread fires (~seconds later, after writer fsync) and the results
            # would silently vanish -- the request would never report capture
            # results (and ``capture_wait`` would hang).
            with self._pending_capture_results_lock:
                self._pending_capture_results.setdefault(req_id, {}).update(named)

        mgr.finalize_request_async(req_id, _on_complete)

    def _drain_capture_results(self) -> dict[str, dict[str, CaptureResult]]:
        """Atomically take the finalized results buffered since the last drain.

        Called on the step thread to flush results onto ``ModelRunnerOutput``
        and via ``collective_rpc`` from the engine core's idle loop
        (:meth:`drain_pending_capture_results`). The lock guards against the
        finalize thread writing while we swap the dict out.
        """
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
        after a request's final step still reach ``capture_wait`` clients, even
        when no further ``ModelRunnerOutput`` is produced. Shares the per-step
        swap buffer with :meth:`_drain_capture_results`.
        """
        return self._drain_capture_results()
