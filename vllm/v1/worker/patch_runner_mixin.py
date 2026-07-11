# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-patching control plane (runner-agnostic half).

The patching *data plane* (the ``apply_patch`` / ``apply_patch_block`` custom
ops, per-(layer, hook) buffers, the Triton kernels) lives in
:mod:`vllm.model_executor.layers.patch`. This module owns the per-step control
plane that, before each forward, writes the source vectors and per-token slot
indices into those persistent buffers — mirroring
:class:`SteeringModelRunnerMixin` but with three differences the design calls
out:

* The index is **per-(layer, hook)** (a request patches different positions at
  different layers), built per-token from the capture-style address map
  ``abs_row = token_offset + (dest_pos - num_computed)``.
* Slots are **ephemeral per step** (assigned ``1..n`` each step, reset next
  step) — no refcount, no register/release/transition state machine.
* Pool overflow is **strict / fail-fast**: the scheduler reserves capacity so a
  step can never exceed ``max_patch_slots``; a breach is a loud error, never a
  silent drop.

Like steering, patching needs no force-eager seam: the buffers are written in
place before the forward, so a FULL cudagraph replay reads this step's values.

This module is intentionally free of v1/v2 runner coupling: the v2 wiring lives
in :mod:`vllm.v1.worker.gpu.patch_runner_mixin`. The step planner
(:func:`build_patch_step_plan`) is a pure function so it can be unit-tested
without a runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.patch import (
    PATCH_ALPHA_ATTR,
    PATCH_ANY_ACTIVE_ATTR,
    PATCH_INDEX_ATTR,
    PATCH_TABLE_ATTR,
)
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR, SteeringHookPoint

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class PatchPoolOverflow(RuntimeError):
    """Raised when a step would need more patch slots than the pool holds.

    Under strict policy the scheduler reserves per-site capacity so this never
    fires in normal operation; if it does, it indicates a scheduler accounting
    bug or a request admitted without capacity reservation.
    """


@dataclass
class PatchEntry:
    """One resolved patch site for a request: overwrite/lerp ``hook`` at
    ``layer`` for the request's logical position ``dest_pos`` with ``source``.

    ``source`` is a CPU 1-D tensor of shape ``(hidden,)`` (the resolved source
    activation, from a capture run, a named module, ``zeros``, or an inline
    row). ``alpha_row`` is a CPU 1-D fp32 tensor of shape ``(hidden,)``: the
    per-dimension interpolation weight ``alpha * mask`` (all-``alpha`` when
    unmasked, ``1.0`` = full replace). A per-dim weight folds the optional mask
    into the same lerp with no separate kernel path.
    """

    layer: int
    hook: SteeringHookPoint
    dest_pos: int
    source: torch.Tensor
    alpha_row: torch.Tensor


@dataclass
class SitePlan:
    """Per-(layer, hook) work for one step. Slot ``k`` (1-based) corresponds to
    ``abs_rows[k-1]`` / ``sources[k-1]`` / ``alpha_rows[k-1]``; slot 0 is the
    untouched passthrough sentinel.
    """

    abs_rows: list[int] = field(default_factory=list)
    sources: list[torch.Tensor] = field(default_factory=list)
    alpha_rows: list[torch.Tensor] = field(default_factory=list)


def build_patch_step_plan(
    *,
    req_ids: Sequence[str],
    num_computed: Sequence[int],
    num_scheduled: Sequence[int],
    token_offsets: Sequence[int],
    specs: Mapping[str, list[PatchEntry]],
    local_layers: frozenset[int],
    max_patch_slots: int,
) -> dict[tuple[int, SteeringHookPoint], SitePlan]:
    """Compute, for this step, which batch rows to patch at each site.

    Pure function over the batch view (lists indexed by batch position) and the
    per-request specs. A patch entry fires this step iff its ``dest_pos`` is
    computed in this step's window ``[num_computed, num_computed + num_sched)``;
    the row in the flat hidden-state tensor is
    ``token_offset + (dest_pos - num_computed)``. Only locally-owned layers
    (PP) are considered.

    Raises :class:`PatchPoolOverflow` (strict policy) if any site needs more
    than ``max_patch_slots - 1`` slots.
    """
    plan: dict[tuple[int, SteeringHookPoint], SitePlan] = {}
    for i, req_id in enumerate(req_ids):
        spec = specs.get(req_id)
        if not spec:
            continue
        n_sched = int(num_scheduled[i])
        if n_sched <= 0:
            continue
        start = int(num_computed[i])
        end = start + n_sched
        offset = int(token_offsets[i])
        for entry in spec:
            if entry.layer not in local_layers:
                continue
            if not (start <= entry.dest_pos < end):
                continue
            site = plan.setdefault((entry.layer, entry.hook), SitePlan())
            if len(site.abs_rows) >= max_patch_slots - 1:
                raise PatchPoolOverflow(
                    f"patch pool exhausted at layer={entry.layer} "
                    f"hook={entry.hook.value}: need > {max_patch_slots - 1} "
                    f"slots in one step (req={req_id})"
                )
            site.abs_rows.append(offset + (entry.dest_pos - start))
            site.sources.append(entry.source)
            site.alpha_rows.append(entry.alpha_row)
    return plan


@dataclass
class _PatchBatchView:
    """Minimal per-step batch view the patch update consumes."""

    req_ids: list[str]
    num_computed: list[int]
    num_scheduled: list[int]
    token_offsets: list[int]

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled)


class PatchModelRunnerMixin:
    """Runner-agnostic activation-patching control plane.

    The concrete runner must expose ``get_model()`` and ``vllm_config``. The v2
    runner mixes this in via :class:`PatchRunnerMixin`, which adds the
    ``InputBatch`` / ``req_states`` projection and the request lifecycle.
    """

    # Populated by _init_patch_state.
    _patchable_layers: dict[int, nn.Module]
    _patch_specs: dict[str, list[PatchEntry]]
    _patch_max_slots: int
    _patch_touched_sites: set[tuple[int, SteeringHookPoint]]
    _patch_index_dirty: bool

    if TYPE_CHECKING:
        vllm_config: VllmConfig

        def get_model(self) -> nn.Module: ...

    # ---- init --------------------------------------------------------------

    def _init_patch_state(self) -> None:
        """Discover patchable layers and warm the patch kernels.

        Patchable layers are those with ``layer_idx`` and a ``patch_table_*``
        buffer (attached by :func:`register_steering_buffers` when patching is
        enabled process-globally). When no layer has patch buffers, all the
        per-step / lifecycle methods short-circuit cheaply.
        """
        self._patch_specs = {}
        self._patch_touched_sites = set()
        self._patch_index_dirty = False
        patchable: dict[int, nn.Module] = {}
        if hasattr(self, "get_model"):
            for mod in self.get_model().modules():
                if not hasattr(mod, "layer_idx"):
                    continue
                if any(hasattr(mod, a) for a in PATCH_TABLE_ATTR.values()):
                    patchable[mod.layer_idx] = mod
        self._patchable_layers = patchable
        self._locally_owned_patch_layers = frozenset(patchable.keys())

        if not patchable:
            self._patch_max_slots = 0
            return

        # Patch-table geometry (also feeds the source-store auto-sizing below).
        sample = next(iter(patchable.values()))
        a_table = getattr(sample, PATCH_TABLE_ATTR[SteeringHookPoint.POST_BLOCK])
        self._patch_max_slots = int(a_table.shape[0])
        table_device = a_table.device
        table_dtype = a_table.dtype
        hidden_size = int(a_table.shape[1])

        # Install the per-worker patch source store (clean-run activations).
        # Lives wherever this rank's capture manager writes (TP rank 0 /
        # each PP rank); resolution reads it, broadcasting to TP peers. The
        # budget auto-sizes from the model (``-1``, the default) so enabling
        # patching provisions the store; ``0`` disables it.
        from vllm.model_executor.layers.patch import (
            get_patch_source_cache_bytes,
            resolve_patch_source_cache_bytes,
        )
        from vllm.v1.capture.source_store import (
            PatchSourceStore,
            get_active_patch_source_store,
            set_active_patch_source_store,
        )

        source_budget = resolve_patch_source_cache_bytes(
            self.vllm_config,
            hidden_size=hidden_size,
            num_patch_layers=len(patchable),
            num_hooks=len(PATCH_TABLE_ATTR),
            dtype=table_dtype,
        )
        if source_budget > 0 and get_active_patch_source_store() is None:
            set_active_patch_source_store(PatchSourceStore(max_bytes=source_budget))
            auto = get_patch_source_cache_bytes(self.vllm_config) < 0
            logger.info(
                "Patch source store enabled: budget=%.3f GB%s",
                source_budget / 1_000_000_000,
                " (auto-sized from model)" if auto else "",
            )

        if table_device.type == "cuda":
            from vllm.model_executor.layers.patch_kernel import (
                warmup_apply_patch_kernel,
            )

            compute_dtype = getattr(self.vllm_config.model_config, "dtype", table_dtype)
            compilation_config = getattr(self.vllm_config, "compilation_config", None)
            capture_sizes = (
                getattr(compilation_config, "cudagraph_capture_sizes", None)
                if compilation_config is not None
                else None
            )
            warmup_apply_patch_kernel(
                hidden_size=hidden_size,
                table_slots=self._patch_max_slots,
                table_dtype=table_dtype,
                compute_dtype=compute_dtype,
                device=table_device,
                capture_sizes=list(capture_sizes) if capture_sizes else None,
            )

    # ---- request lifecycle -------------------------------------------------

    def _patch_add_request(self, new_req_data) -> None:
        """Resolve a newly admitted request's patch spec into source entries.

        Runner-agnostic: resolution reads ``sampling_params.patch`` + the
        per-worker source store, keeping only locally-owned layers (PP) and
        broadcasting across the TP group. Streaming re-adds drop any prior spec
        first, matching the steering/capture re-add discipline.
        """
        if not self._patchable_layers:
            return
        self._patch_specs.pop(new_req_data.req_id, None)

        from vllm.v1.worker.gpu.patch_resolve import resolve_patch_entries

        # The named-module source kind reuses the steering module registry the
        # runner also carries (both mixins live on the same runner); ``zeros``
        # and inline sources need only the hook width (== hidden_size).
        module_registry = getattr(self, "_steering_module_registry", None)
        hidden_size = self._patch_hidden_size()
        entries = resolve_patch_entries(
            new_req_data,
            local_layers=self._locally_owned_patch_layers,
            module_registry=module_registry,
            hidden_size=hidden_size,
        )
        if entries:
            self._patch_specs[new_req_data.req_id] = entries

    def _patch_hidden_size(self) -> int:
        """Hook width for ``zeros`` / inline sources (all three hooks are
        residual-width == hidden_size). Read from a patch table so it matches
        the buffers exactly, falling back to the model config."""
        for mod in self._patchable_layers.values():
            table = getattr(mod, PATCH_TABLE_ATTR[SteeringHookPoint.POST_BLOCK], None)
            if table is not None:
                return int(table.shape[1])
        return int(self.vllm_config.model_config.get_hidden_size())

    def _patch_finish_requests(self, req_ids: set[str] | list[str]) -> None:
        """Drop specs for finished/preempted requests.

        Preempted requests re-enter via the add path on resume, which
        re-resolves their spec; recomputation re-fires the patch automatically.
        """
        if not self._patch_specs:
            return
        for req_id in req_ids:
            self._patch_specs.pop(req_id, None)

    # ---- per-step buffer maintenance ---------------------------------------

    def _write_patch_step(self, view: _PatchBatchView) -> None:
        """Build and write this step's patch buffers from ``_patch_specs``.

        Mirrors steering's ``_update_steering_buffers`` short-circuit + dirty
        tracking: when nothing is patched this step, the previously-touched
        sites are cleared once and the method returns.
        """
        if not self._patchable_layers or self._patch_max_slots <= 0:
            return

        n_tokens = view.total_tokens
        batch_has_patch = any(req in self._patch_specs for req in view.req_ids)
        if not batch_has_patch:
            if self._patch_index_dirty:
                self._clear_patch_sites(self._patch_touched_sites, n_tokens)
                self._patch_touched_sites = set()
                self._patch_index_dirty = False
            return

        plan = build_patch_step_plan(
            req_ids=view.req_ids,
            num_computed=view.num_computed,
            num_scheduled=view.num_scheduled,
            token_offsets=view.token_offsets,
            specs=self._patch_specs,
            local_layers=self._locally_owned_patch_layers,
            max_patch_slots=self._patch_max_slots,
        )

        touched_now: set[tuple[int, SteeringHookPoint]] = set()
        for (layer, hook), site in plan.items():
            self._write_site(layer, hook, site, n_tokens)
            touched_now.add((layer, hook))

        # Clear sites patched last step but not this step.
        self._clear_patch_sites(self._patch_touched_sites - touched_now, n_tokens)
        self._patch_touched_sites = touched_now
        self._patch_index_dirty = bool(touched_now)

    def _write_site(
        self,
        layer: int,
        hook: SteeringHookPoint,
        site: SitePlan,
        n_tokens: int,
    ) -> None:
        module = self._patchable_layers[layer]
        table = getattr(module, PATCH_TABLE_ATTR[hook])
        alpha_buf = getattr(module, PATCH_ALPHA_ATTR[hook])
        index_buf = getattr(module, PATCH_INDEX_ATTR[hook])
        flag_buf = getattr(module, PATCH_ANY_ACTIVE_ATTR[hook])

        n = len(site.abs_rows)
        # Stage source rows into slots 1..n (slot 0 stays passthrough).
        src = torch.stack([s.reshape(-1) for s in site.sources]).to(
            device=table.device, dtype=table.dtype
        )
        table[1 : n + 1].copy_(src, non_blocking=True)
        # Per-dim alpha rows (alpha * mask) into the matching slots. Row 0 stays
        # all-zeros (passthrough); staging here never touches it.
        alpha_host = torch.stack([a.reshape(-1) for a in site.alpha_rows]).to(
            torch.float32
        )
        alpha_buf[1 : n + 1].copy_(alpha_host.to(alpha_buf.device), non_blocking=True)

        # Rebuild the per-token index: zero the live window, scatter the slots.
        index_buf[:n_tokens].zero_()
        rows = torch.tensor(site.abs_rows, dtype=torch.long, device=index_buf.device)
        slots = torch.arange(1, n + 1, dtype=index_buf.dtype, device=index_buf.device)
        index_buf[rows] = slots
        flag_buf.fill_(True)

    def _clear_patch_sites(
        self,
        sites: set[tuple[int, SteeringHookPoint]],
        n_tokens: int,
    ) -> None:
        for layer, hook in sites:
            module = self._patchable_layers.get(layer)
            if module is None:
                continue
            getattr(module, PATCH_INDEX_ATTR[hook])[:n_tokens].zero_()
            getattr(module, PATCH_ANY_ACTIVE_ATTR[hook]).zero_()


def get_patchable_hook(name: str) -> SteeringHookPoint:
    """Validate and resolve an injection hook name to its enum.

    Only the three steering hook points have patch buffers, so only they are
    injection-capable (a source run may capture more, but injection reuses the
    steering apply sites).
    """
    try:
        hp = SteeringHookPoint(name)
    except ValueError as exc:
        raise ValueError(
            f"hook {name!r} is not patchable; valid: "
            f"{sorted(h.value for h in HOOK_POINT_TABLE_ATTR)}"
        ) from exc
    return hp
