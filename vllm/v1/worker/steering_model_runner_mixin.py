# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR
from vllm.v1.worker.steering_manager import SteeringManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class SteeringModelRunnerMixin:
    """Consolidates all activation-steering state and logic on the model runner.

    Mirrors the ``LoRAModelRunnerMixin`` pattern: the mixin owns every
    piece of per-request steering state and lifecycle hook wired into
    ``GPUModelRunner._update_states()``.
    """

    # --- class-level attribute declarations --------------------------------
    # These back the mixin's stateful methods.  Attributes that are
    # unconditionally read (possibly before lazy init) have
    # class-level defaults so the mixin can use plain attribute
    # access without ``hasattr``/``getattr`` guards.
    #
    # ``_steering_manager`` intentionally does NOT have a default:
    # ``_update_steering_buffers`` uses ``hasattr(self, "_steering_manager")``
    # as the lazy-init trigger, so assigning a class-level default would
    # make initialisation skip permanently.
    _steerable_layers_cache: dict[int, nn.Module] | None = None
    # The attributes below are populated by the lazy init in
    # ``_update_steering_buffers`` and are only read after that path
    # has run.  Test fixtures that exercise the mixin in isolation
    # must set them explicitly.
    _steering_manager: SteeringManager | None
    _req_steering_phase: dict[str, str]
    _steering_index_dirty: bool
    # Set of layer indices physically owned by this worker.  Populated
    # during lazy init and threaded into ``SteeringManager`` calls so
    # non-local tensors are never materialized on this rank.  Under TP/
    # single-worker execution this equals the full model's layer set.
    _locally_owned_layers: frozenset[int]

    # Attributes provided by the concrete model runner that mixes this
    # class in.  Declared here purely so static type checking can see
    # them — there is no runtime assignment.
    if TYPE_CHECKING:
        vllm_config: VllmConfig
        input_batch: InputBatch
        requests: dict[str, CachedRequestState]

        def get_model(self) -> nn.Module: ...

    # -----------------------------------------------------------------------
    # Per-step buffer / index maintenance
    # -----------------------------------------------------------------------

    def _update_steering_buffers(self, scheduler_output: "SchedulerOutput") -> None:
        """Update per-layer steering tables and the shared steering index.

        Lazily initializes the SteeringManager on first call.  Each step:
        1. Populate each layer's per-hook steering_table from the manager
        2. Build the steering_index mapping tokens to table rows
        3. Detect prefill->decode phase transitions and swap configs
        """
        # Short-circuit when steering is disabled.  Steerable models
        # (e.g. Gemma3) unconditionally register per-layer steering_table
        # buffers so the forward path can stay branch-free, but when
        # --enable-steering is off there is no SteeringConfig and no work
        # to do — populating tables and building the index every step is
        # pure overhead.
        if getattr(self.vllm_config, "steering_config", None) is None:
            if not hasattr(self, "_steering_manager"):
                self._steering_manager = None
                self._steerable_layers_cache = {}
            return

        # Lazy init
        if not hasattr(self, "_steering_manager"):
            steerable: dict = {}
            model = self.get_model()
            for mod in model.modules():
                if not hasattr(mod, "layer_idx"):
                    continue
                has_any_table = any(
                    hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
                )
                if has_any_table:
                    steerable[mod.layer_idx] = mod
            self._steerable_layers_cache = steerable
            # Snapshot the set of layer indices this worker physically
            # owns.  Used to skip tensor materialization for non-local
            # layers when passing vectors into the SteeringManager.
            self._locally_owned_layers = frozenset(steerable.keys())

            if steerable:
                steering_config = getattr(self.vllm_config, "steering_config", None)
                max_configs = (
                    steering_config.max_steering_configs if steering_config else 0
                )

                # Resolve device from the first steerable layer's table
                # buffer so per-request vectors are allocated on the same
                # device, avoiding CPU->GPU copies each step.
                table_device: torch.device | None = None
                for mod in steerable.values():
                    for attr in HOOK_POINT_TABLE_ATTR.values():
                        if hasattr(mod, attr):
                            table_device = getattr(mod, attr).device
                            break
                    if table_device is not None:
                        break

                self._steering_manager = SteeringManager(
                    max_configs, device=table_device
                )
                self._req_steering_phase: dict[str, str] = {}
                # Tracks whether steering_index has been written with non-zero
                # row references. Used by the no-active-state short-circuit
                # to know if it needs to zero the index on transition.
                self._steering_index_dirty: bool = False

                # Register any configs that were added to the batch
                # before the manager existed (first-step race).
                for i in range(self.input_batch.num_reqs):
                    rid = self.input_batch.req_ids[i]
                    rs = self.requests.get(rid)
                    if rs is None or rs.sampling_params is None:
                        continue
                    ri = self.input_batch.req_id_to_index.get(rid)
                    if ri is None:
                        continue

                    num_computed = int(self.input_batch.num_computed_tokens_cpu[ri])
                    num_prompt = int(self.input_batch.num_prompt_tokens[ri])

                    if num_computed < num_prompt:
                        # In prefill — register prefill config
                        ph = int(self.input_batch.request_prefill_steering_hash[ri])
                        if ph != 0:
                            eff = rs.sampling_params.effective_prefill_steering
                            if eff:
                                self._steering_manager.register_config(
                                    ph,
                                    eff,
                                    phase="prefill",
                                    locally_owned_layers=(self._locally_owned_layers),
                                )
                        self._req_steering_phase[rid] = "prefill"
                    else:
                        # In decode (full prefix-cache hit) — register
                        # decode config
                        dh = int(self.input_batch.request_decode_steering_hash[ri])
                        if dh != 0:
                            eff = rs.sampling_params.effective_decode_steering
                            if eff:
                                self._steering_manager.register_config(
                                    dh,
                                    eff,
                                    phase="decode",
                                    locally_owned_layers=(self._locally_owned_layers),
                                )
                        self._req_steering_phase[rid] = "decode"
            else:
                self._steering_manager = None
                self._steerable_layers_cache = {}

        if self._steering_manager is None or not self._steerable_layers_cache:
            return

        # Short-circuit when no steering state is actually active. The model
        # runner allocates per-layer steering buffers (zero-initialized) and
        # the forward path always calls apply_steering, but if no per-request
        # configs are registered and no global vectors have been set, every
        # gather hits the zero sentinel and adds nothing. There is nothing
        # to populate.
        #
        # Correctness: when we previously had active steering and now don't
        # (e.g., the last steered request just finished), the steering_index
        # may still contain non-zero row references from the previous step.
        # We must zero it before returning to ensure all gathers point to
        # row 0. We only do this on the transition; in the steady "nothing
        # ever active" case the index is already zero from initialization.
        if (
            not self._steering_manager.config_to_row
            and not self._steering_manager.global_base_vectors
            and not self._steering_manager.global_prefill_vectors
            and not self._steering_manager.global_decode_vectors
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                any_layer.steering_index.zero_()
                self._steering_index_dirty = False
            return

        # 1. Populate steering tables — but only if state has changed since
        # the last populate. populate_steering_tables() clears the flag at
        # the end, and every state mutator (register_config new-row,
        # release_config refcount->0, update_global_vectors,
        # clear_global_vectors) sets it. In steady-state decode steps
        # where no config churn happens, this skips ~102 kernel launches
        # per step.
        if self._steering_manager._tables_dirty:
            self._steering_manager.populate_steering_tables(
                self._steerable_layers_cache
            )

        # 2. Build steering index
        # Get the shared steering_index buffer (all layers share one tensor)
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = any_layer.steering_index

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids

        # Walk requests in batch order, assigning each token's table row
        token_offset = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # Request not in batch yet (shouldn't happen but guard)
                steering_index[token_offset : token_offset + n_tokens] = 0
                token_offset += n_tokens
                continue

            # Determine phase from num_computed vs num_prompt
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt

            if is_prefilling:
                # Prefill: use prefill steering hash
                prefill_hash = int(
                    self.input_batch.request_prefill_steering_hash[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    prefill_hash, is_prefill=True
                )
                steering_index[token_offset : token_offset + n_tokens] = row

                # Check if this request will transition to decode after
                # this step's tokens are processed.
                num_computed_after = num_computed + n_tokens
                if num_computed_after >= num_prompt:
                    self._handle_steering_transition(req_id, req_index, prefill_hash)
            else:
                # Decode: use decode steering hash
                decode_hash = int(
                    self.input_batch.request_decode_steering_hash[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    decode_hash, is_prefill=False
                )
                steering_index[token_offset : token_offset + n_tokens] = row

            token_offset += n_tokens

        # Zero out remaining positions
        if token_offset < steering_index.shape[0]:
            steering_index[token_offset:].zero_()

        # Mark the index as having non-zero row references this step. The
        # no-active-state short-circuit on a future step will zero the index
        # if needed when transitioning back to "nothing active".
        self._steering_index_dirty = True

    def _handle_steering_transition(
        self,
        req_id: str,
        req_index: int,
        prefill_hash: int,
    ) -> None:
        """Handle prefill->decode steering config transition.

        Called when a request will complete prefill after this step.
        Releases the prefill config and registers the decode config
        so it is ready for the next step's table population.

        The scheduler guarantees capacity for the decode row, so
        registration always succeeds.
        """
        mgr = self._steering_manager
        assert mgr is not None, (
            "_handle_steering_transition called without an initialised manager"
        )
        if prefill_hash != 0:
            mgr.release_config(prefill_hash, "prefill")

        decode_hash = int(self.input_batch.request_decode_steering_hash[req_index])
        if decode_hash != 0:
            req_state = self.requests.get(req_id)
            if req_state is not None and req_state.sampling_params is not None:
                sp = req_state.sampling_params
                if sp.effective_decode_steering:
                    mgr.register_config(
                        decode_hash,
                        sp.effective_decode_steering,
                        phase="decode",
                        locally_owned_layers=self._locally_owned_layers,
                    )

        self._req_steering_phase[req_id] = "decode"

    def _reset_steering_for_resumption(
        self,
        req_id: str,
        req_state: "CachedRequestState",
        new_num_computed_tokens: int,
    ) -> None:
        """Reset steering config registration when a request re-enters prefill.

        Called when a preempted request is resumed with num_computed_tokens
        reset. If the request had transitioned to decode before preemption,
        its decode config is still registered and its phase is stale.
        This helper releases the stale decode config and re-registers the
        prefill config.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return
        prev_phase = self._req_steering_phase.get(req_id)
        if prev_phase != "decode":
            return
        if new_num_computed_tokens >= req_state.num_prompt_tokens:
            return  # still in decode, nothing to reset

        # Release the stale decode config.
        if req_state.decode_steering_config_hash != 0:
            mgr.release_config(req_state.decode_steering_config_hash, "decode")

        self._req_steering_phase[req_id] = "prefill"

        sp = req_state.sampling_params
        prefill_hash = req_state.prefill_steering_config_hash
        if prefill_hash == 0 or sp is None or not sp.effective_prefill_steering:
            return
        mgr.register_config(
            prefill_hash,
            sp.effective_prefill_steering,
            phase="prefill",
            locally_owned_layers=self._locally_owned_layers,
        )

    # -----------------------------------------------------------------------
    # Hooks called from _update_states() / _update_streaming_request()
    # -----------------------------------------------------------------------

    def _release_finished_steering_configs(
        self, finished_req_ids: "set[str] | list[str]"
    ) -> None:
        """Release the currently-active steering config for finished requests.

        Called before finished request state is popped so
        ``prefill_steering_config_hash`` /
        ``decode_steering_config_hash`` are still accessible.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        for req_id in finished_req_ids:
            phase = self._req_steering_phase.pop(req_id, None)
            if phase is not None:
                req_state = self.requests.get(req_id)
                if req_state is not None:
                    if phase == "prefill":
                        h = req_state.prefill_steering_config_hash
                    else:
                        h = req_state.decode_steering_config_hash
                    if h != 0:
                        mgr.release_config(h, phase)

    def _register_initial_steering_config(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        req_state: "CachedRequestState",
    ) -> None:
        """Register the initial-phase steering config for a new request.

        Normally requests start in prefill, but a full prefix-cache hit
        (``num_computed >= num_prompt``) puts a request directly into
        decode.  The scheduler guarantees capacity so registration
        always succeeds.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None or new_req_data.sampling_params is None:
            return

        sp = new_req_data.sampling_params
        if new_req_data.num_computed_tokens >= req_state.num_prompt_tokens:
            # Already past prefill — register decode config.
            if (
                new_req_data.decode_steering_config_hash != 0
                and sp.effective_decode_steering
            ):
                mgr.register_config(
                    new_req_data.decode_steering_config_hash,
                    sp.effective_decode_steering,
                    phase="decode",
                    locally_owned_layers=self._locally_owned_layers,
                )
            self._req_steering_phase[req_id] = "decode"
        else:
            # Normal: start in prefill; decode registered
            # on transition in _update_steering_buffers.
            if (
                new_req_data.prefill_steering_config_hash != 0
                and sp.effective_prefill_steering
            ):
                mgr.register_config(
                    new_req_data.prefill_steering_config_hash,
                    sp.effective_prefill_steering,
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                )
            self._req_steering_phase[req_id] = "prefill"

    def _refresh_streaming_steering(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        old_prefill_hash: int,
        old_decode_hash: int,
        new_prefill_hash: int,
        new_decode_hash: int,
    ) -> None:
        """Refresh steering state for a streaming re-added request.

        Streaming re-adds go back through prefill, so we must:
        1. Release the old config (whatever phase we were tracking)
        2. Register the new prefill config
        3. Update phase tracking
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        # Release the old phase config.
        old_phase = self._req_steering_phase.get(req_id)
        if old_phase is not None:
            if old_phase == "prefill" and old_prefill_hash != 0:
                mgr.release_config(old_prefill_hash, "prefill")
            elif old_phase == "decode" and old_decode_hash != 0:
                mgr.release_config(old_decode_hash, "decode")

        # Register new prefill config (streaming re-adds start
        # in prefill).
        sp = new_req_data.sampling_params
        if new_prefill_hash != 0 and sp is not None and sp.effective_prefill_steering:
            mgr.register_config(
                new_prefill_hash,
                sp.effective_prefill_steering,
                phase="prefill",
                locally_owned_layers=self._locally_owned_layers,
            )
            self._req_steering_phase[req_id] = "prefill"
        elif new_prefill_hash == 0 and new_decode_hash == 0:
            # No steering for this request anymore.
            self._req_steering_phase.pop(req_id, None)
        else:
            # Has hashes but no effective prefill vectors (e.g.,
            # decode-only steering).  Mark as prefill since the
            # request re-enters prefill; transition to decode
            # will handle decode registration.
            self._req_steering_phase[req_id] = "prefill"
