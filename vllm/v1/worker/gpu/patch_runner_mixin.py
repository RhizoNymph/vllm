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

import torch

from vllm.logger import init_logger
from vllm.v1.worker.patch_runner_mixin import (
    PatchModelRunnerMixin,
    _PatchBatchView,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Hook whose captured activation is the merged residual stream entering the
# next layer (== the entry residual for 2a): entry@L = post_block[L-1].
_TRUNK_HOOK = "post_block"


class PatchRunnerMixin(PatchModelRunnerMixin):
    """v2 activation-patching control plane."""

    def _patch_add_request(self, new_req_data: NewRequestData) -> None:
        """Resolve a newly admitted request's patch spec into source entries.

        Resolution (sampling_params.patch + the per-worker source store) is
        wired in the source-store / admission phase. Streaming re-adds drop any
        prior spec first, matching the steering/capture re-add discipline.
        """
        if not self._patchable_layers:
            return
        self._patch_specs.pop(new_req_data.req_id, None)

        from vllm.v1.worker.gpu.patch_resolve import resolve_patch_entries

        entries = resolve_patch_entries(
            new_req_data,
            local_layers=self._locally_owned_patch_layers,
        )
        if entries:
            self._patch_specs[new_req_data.req_id] = entries

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

        view = self._patch_batch_view(input_batch)
        self._write_patch_step(view)

    def _patch_batch_view(self, input_batch: InputBatch) -> _PatchBatchView:
        """Project ``InputBatch`` into the per-step patch/entry token view."""
        num_reqs = input_batch.num_reqs
        idx = input_batch.idx_mapping_np[:num_reqs]
        req_states = self.req_states
        return _PatchBatchView(
            req_ids=list(input_batch.req_ids),
            num_computed=req_states.num_computed_tokens_np[idx].tolist(),
            num_scheduled=input_batch.num_scheduled_tokens[:num_reqs].tolist(),
            token_offsets=input_batch.query_start_loc_np[:num_reqs].tolist(),
        )

    # ---- Level-2 (2a) trunk re-entry --------------------------------------

    def _patch_2a_add_request(self, new_req_data: NewRequestData) -> None:
        """Record a newly admitted request's ``patch_2a`` re-entry spec."""
        self._patch_2a_specs.pop(new_req_data.req_id, None)
        sp = getattr(new_req_data, "sampling_params", None)
        spec = getattr(sp, "patch_2a", None) if sp is not None else None
        if not spec:
            return
        self._patch_2a_specs[new_req_data.req_id] = (
            int(spec["entry_layer"]),
            str(spec["trunk_run"]),
        )

    def _build_2a_entry(
        self, input_batch: InputBatch
    ) -> tuple[int, torch.Tensor] | None:
        """Build this step's mid-stack entry: ``(start_layer, entry_embeds)``.

        Returns ``None`` (→ normal full forward, i.e. Level-1 fallback) unless
        **every** scheduled request carries a ``patch_2a`` spec and they share a
        single ``entry_layer``. The entry residual for each scheduled token is
        the trunk run's ``post_block[entry_layer - 1]`` at that position; a
        missing row also degrades to the (correct) full forward.
        """
        if not self._patch_2a_specs:
            return None
        req_ids = list(input_batch.req_ids)
        specs = [self._patch_2a_specs.get(r) for r in req_ids]
        if any(s is None for s in specs):
            return None  # mixed batch → cannot skip layers; run Level-1
        entry_layers = {s[0] for s in specs}
        if len(entry_layers) != 1:
            logger.warning(
                "2a batch not homogeneous in entry_layer (%s); running full "
                "forward", entry_layers
            )
            return None
        entry_layer = next(iter(entry_layers))

        from vllm.v1.capture.source_store import get_active_patch_source_store

        store = get_active_patch_source_store()
        if store is None:
            return None

        view = self._patch_batch_view(input_batch)
        n_tokens = int(input_batch.positions.shape[0])
        # Gather the trunk rows on CPU, then move to device in ONE transfer and
        # scatter by row index — a per-token .to(device) loop is ~100x slower
        # and swamps the layer-skip win.
        rows: list[torch.Tensor] = []
        idxs: list[int] = []
        for i in range(len(req_ids)):
            _, trunk_run = specs[i]
            start = view.num_computed[i]
            offset = view.token_offsets[i]
            for k in range(view.num_scheduled[i]):
                row = store.get_row(
                    trunk_run, entry_layer - 1, _TRUNK_HOOK, start + k
                )
                if row is None:
                    return None  # missing trunk → correct full-forward fallback
                rows.append(row)
                idxs.append(offset + k)
        entry = torch.zeros(
            (n_tokens, self._patch_2a_hidden),
            dtype=self._patch_2a_dtype,
            device=self._patch_2a_device,
        )
        staged = torch.stack(rows).to(
            device=entry.device, dtype=entry.dtype, non_blocking=True
        )
        index = torch.tensor(idxs, dtype=torch.long, device=entry.device)
        entry[index] = staged
        return entry_layer, entry
