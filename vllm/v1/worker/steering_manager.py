# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request steering state manager (shared-nothing, deterministic replay).

Determinism contract
--------------------
Steering state is shared-nothing with deterministic replay. Every worker
executes identical ``set_steering_vectors`` / ``clear_steering_vectors``
calls (via ``collective_rpc``) and sees an identical ``SchedulerOutput``
stream, so every worker's ``SteeringManager`` derives identical
``config_to_row`` assignments, identical ``free_rows`` state, and an
identical ``steering_index`` tensor each step -- even though each worker
stores vectors only for layers it physically owns. No cross-rank
collectives are needed in the hot path.

Concrete implications:

* Row allocation is fully rank-local. ``register_config`` runs on every
  rank for every config, regardless of whether that rank owns an
  affected layer. This is a correctness requirement, not an optimization:
  row IDs flow through ``steering_index`` into the ``apply_steering``
  gather on every rank, so they MUST match.
* Global-vector broadcast lives at the API layer (one ``collective_rpc``),
  not at the worker layer. There is no NCCL all-reduce of steering tables.
* ``SamplingParams.prefill_steering_config_hash`` and
  ``decode_steering_config_hash`` are pure functions of the request
  payload, identical on every rank.

See ``docs/design/steering_runtime.md`` section "Distributed execution"
for the full mental model.

Class responsibilities (unchanged by the contract):

Tracks registered steering configs, assigns table rows, handles
reference counting, and populates per-layer steering_table buffers
with the correct combined (global + per_request) vectors. Supports
multiple hook points per layer; each hook point has its own steering
table buffer (e.g. ``steering_table_pre_attn``) and global vector cache.
Supports phase-aware (prefill vs decode) steering with separate global
effective vectors for each phase.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config.steering_types import hash_steering_config
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    _ROW_MONITOR_DEFAULT_PARAMS,
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_DYNVEC_ATTR,
    HOOK_POINT_MONITOR_ACTIVE_ATTR,
    HOOK_POINT_MONITOR_PARAMS_ATTR,
    HOOK_POINT_MONITOR_PROBE_ATTR,
    HOOK_POINT_ROW_ACTIVE_ATTR,
    HOOK_POINT_ROW_PARAMS_ATTR,
    HOOK_POINT_ROW_PROBE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.steering_owner import OwnerStore, RowOwner

logger = init_logger(__name__)


@dataclass
class _DirtyState:
    """Populate-scheduling flags with the implication rules encoded in code.

    Three orthogonal reasons a populate is owed, with the invariants that used
    to live in scattered comments made explicit:

    * ``content`` (was ``_tables_dirty``): the per-layer table buffers need a
      full recompose + H2D.
    * ``membership`` (was ``_indices_dirty``): the row set changed, so the
      cached ``indices`` / ordered-config scratch must be rebuilt. Membership
      always implies content (a new/dropped row must be written), so
      :meth:`mark_membership` sets both.
    * ``scales`` (was ``_scales_dirty``): only the cheap per-row scale buffers
      need rewriting.

    A full table populate clears all three (it writes scales alongside the
    tables); the cheap scales-only path is eligible only when scales are dirty
    and neither content nor membership is (:attr:`scales_only_eligible`).
    """

    content: bool = True
    membership: bool = True
    scales: bool = True

    def mark_content(self) -> None:
        """A content mutator ran: a full table populate is owed."""
        self.content = True

    def mark_membership(self) -> None:
        """The row set changed: rebuild indices scratch (implies content)."""
        self.membership = True
        self.content = True

    def mark_scales(self) -> None:
        """A scale mutator ran: the cheap scale-buffer write is owed."""
        self.scales = True

    def clear_after_full_populate(self) -> None:
        """A full ``populate_steering_tables`` brought every buffer in sync."""
        self.content = False
        self.membership = False
        self.scales = False

    @property
    def scales_only_eligible(self) -> bool:
        """True when only scales are dirty (the cheap populate path applies)."""
        return self.scales and not (self.content or self.membership)


class SteeringManager:
    """Per-request steering config manager.

    Maintains a mapping from config hashes to steering table rows,
    handles reference counting for shared configs, and writes
    combined vectors into each layer's per-hook-point steering_table
    buffers.

    Table layout (per hook point):
        Row 0: zeros sentinel (no steering)
        Row 1: global prefill effective (global_base + global_prefill)
        Row 2: global decode effective
            (global_base + global_decode + dynamic_tier; the dynamic
            additive tier is folded in here only — decode-derived rows
            see it, prefill rows do not. See §5.4.)
        Rows 3..max_steering_configs+2: phase-appropriate global
            + per_request combined
        Rows max_steering_configs+3..+2+max_dynamic_steering_configs:
            dynamic-override pool — runtime-registered decode rows
            (global decode effective + override vectors), allocated by
            dynamic steering and never by request admission. The two
            pools share nothing: dynamic registrations can never
            exhaust rows the scheduler reserved for admitted requests.
            See docs/design/dynamic_steering.md §5.2.
    """

    def __init__(
        self,
        max_steering_configs: int,
        device: torch.device | None = None,
        max_dynamic_steering_configs: int = 0,
    ):
        self.max_steering_configs = max_steering_configs
        self.max_dynamic_steering_configs = max_dynamic_steering_configs
        self.device = device
        # (config_hash, phase) -> assigned table row index (3-based)
        self.config_to_row: dict[tuple[int, str], int] = {}
        # (config_hash, phase) -> {hook_point_str: {layer_idx: tensor}}
        # (per-request vectors only, not combined)
        self.config_vectors: dict[
            tuple[int, str], dict[str, dict[int, torch.Tensor]]
        ] = {}
        # (config_hash, phase) -> number of active requests using this config
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Available row indices (rows 3 through max_steering_configs + 2)
        # Reversed so pop() gives lowest
        self.free_rows: list[int] = list(range(max_steering_configs + 2, 2, -1))

        # Dynamic-override pool (rows above the static pool). Allocated
        # and released only by the dynamic steering path; deterministic
        # monotonically increasing ids keep ranks in lock-step because
        # the register/release call sequence is identical on every rank
        # (rank-replicated sync consumers). Ids are never reused.
        self._dynamic_free_rows: list[int] = list(
            range(
                max_steering_configs + 2 + max_dynamic_steering_configs,
                max_steering_configs + 2,
                -1,
            )
        )
        # dyn_id -> {hook_point_str: {layer_idx: tensor}} (override
        # vectors only, not combined). Insertion-ordered.
        self._dynamic_vectors: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
        self._dynamic_to_row: dict[int, int] = {}
        self._next_dynamic_id: int = 1

        # Global vectors split into three tiers:
        #   base:    both-phases vectors (from global API)
        #   prefill: prefill-specific global vectors
        #   decode:  decode-specific global vectors
        self.global_base_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_prefill_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_decode_vectors: dict[str, dict[int, torch.Tensor]] = {}

        # Dynamic additive tier (decode-only): a global steering
        # contribution owned by dynamic consumers, folded ADDITIVELY into
        # the decode-effective vector at populate time rather than
        # overwriting ``global_decode_vectors``. This is what lets dynamic
        # global steering compose with operator-set (``/v1/steering/set``)
        # decode steering instead of clobbering it. Decode-only by
        # construction (§7): folded into ``global_decode`` only, so it
        # reaches row 2, decode per-request rows, and dynamic-override
        # rows, never prefill rows, and never feeds prefix-cache keys.
        # See docs/design/dynamic_steering.md §5.4.
        self.dynamic_tier_vectors: dict[str, dict[int, torch.Tensor]] = {}
        # Scalar strength for the dedicated dynamic tier (§5.4). The runner
        # folds it into the per-token gate (``steering_token_scales``) each
        # step, so changing it is free (no buffer rewrite of its own).
        self.dynamic_tier_gain: float = 1.0

        # In-graph monitor configs (Phase 2, §8), keyed hook -> layer ->
        # {"probe": fp32 (hidden,) tensor, "threshold": float,
        # "sharpness": float}. At a probe site the runner's flat decode
        # gain in ``steering_token_scales`` is modulated per token by
        # ``sigmoid(sharpness*(residual@probe - threshold))``, conditioning
        # the §5.4 dynamic tier at this and later hooks/layers within the
        # same forward. Written into the per-layer monitor buffers at
        # populate time (gated by ``_tables_dirty``); policy params then
        # live in the persistent buffers, host-tunable without recapture.
        self.monitor_configs: dict[str, dict[int, dict]] = {}

        # PER-ROW (per-request) in-graph monitor configs. Keyed
        # hook -> layer -> RowOwner -> {"probe", "threshold", "sharpness"},
        # where the :class:`RowOwner` is the LOGICAL row owner so configs
        # survive row reassignment (like ``_row_scales``):
        #   RowOwner.global_("decode")       -> row 2 (global decode)
        #   RowOwner.config(hash, "decode")  -> a static decode config row
        #   RowOwner.dyn(dyn_id)             -> a dynamic-override row
        # Unlike ``monitor_configs`` (one global probe per site), each row is
        # gated by ITS OWN probe, so concurrent requests at a site can carry
        # different probes. Written into the per-row probe-table buffers at
        # populate; gates the row term only, decode-only. Opt-in via
        # ``enable_row_monitor`` (else the buffers stay dummy and this is
        # never activated). See docs/design/dynamic_steering.md.
        self._row_monitor: dict[str, dict[int, dict[RowOwner, dict]]] = {}
        # Lazy per-owner signature cache for APC (None ⇒ stale, rebuild).
        self._row_monitor_sig_cache: dict[RowOwner, int] | None = None

        # APC steering-signature caches (see
        # docs/design/dynamic_steering_apc_notification.md). The worker
        # reports a per-request *effective decode steering signature* to the
        # scheduler so steered decode KV blocks are keyed by the steering
        # that produced them (not the admitted config). Component hashes are
        # cached and only recomputed when their source state mutates:
        # per-dyn_id override-vector hash, and lazily-recomputed global
        # tier / monitor hashes (``None`` ⇒ stale, recompute on next read).
        self._dynamic_sig: dict[int, int] = {}
        self._tier_sig_cache: int | None = None
        self._monitor_sig_cache: int | None = None

        # Per-row strength scales (the §5.3 "how much" knob), keyed by the
        # LOGICAL :class:`RowOwner` so they survive row reassignment:
        # ``RowOwner.global_(phase)`` for rows 1/2, ``RowOwner.config(hash,
        # phase)`` for static per-request rows, ``RowOwner.dyn(dyn_id)`` for
        # dynamic-override rows. A missing key means the default 1.0
        # (unscaled). Decode-only by policy (prefill rows are forced to 1.0 at
        # populate time, §7); the scale never enters config hashes (runtime
        # state, not identity). See docs/design/dynamic_steering.md §5.3.
        self._row_scales: dict[RowOwner, float] = {}

        # Populate-scheduling flags (``content`` / ``membership`` / ``scales``)
        # with the implication rules encoded in :class:`_DirtyState`. Every
        # state mutator marks the appropriate flag; a full populate clears all
        # three, the cheap scales-only path clears just ``scales``. Initialized
        # all-dirty so the first populate call always runs.
        self._dirty = _DirtyState()

        # Cached scratch tensors for populate_steering_tables. ``indices``
        # is the GPU int64 tensor of target row positions
        # ``[0, 1, 2, *config_rows]`` and ``zero_row`` is a hidden-size
        # fp32 zeros tensor used as the row-0 / no-vector fallback. Their
        # contents only depend on ``config_to_row`` (and the per-layer
        # table device/hidden_size, which is fixed). They DO NOT depend
        # on global-vector updates, so we cache them and only invalidate
        # in register_config / release_config (the two paths that mutate
        # ``config_to_row``).
        #
        # ``_dirty.membership`` is independent of ``_dirty.content``: every
        # global-vector update marks content (forcing a populate) but does NOT
        # need to rebuild the scratch tensors.
        self._cached_indices: torch.Tensor | None = None
        self._cached_zero_row: torch.Tensor | None = None
        self._cached_ordered_configs: list[tuple[tuple[int, str], int]] | None = None
        self._cached_ordered_dynamic: list[tuple[int, int]] | None = None

        # Reusable pinned-CPU staging ring for ``_stack_vectors_to_device``.
        #
        # Allocating a fresh pinned tensor per call is a measurable wall-time
        # cost — ``torch.Tensor.pin_memory()`` does a synchronous host-side
        # page-locking copy, which dominates the ``register_config`` path
        # for multi-MB stacks and defeats the purpose of ``non_blocking=True``
        # on the H2D itself.
        #
        # We can't just reuse a single pinned buffer though: the H2D
        # ``cudaMemcpyAsync`` reads from the host pointer when the GPU
        # runs the copy (not at submission), so overwriting the buffer
        # before the previous DMA has drained corrupts the in-flight
        # transfer. To avoid that we maintain a small ring of pinned
        # slots; each slot is paired with a CUDA event recorded right
        # after its H2D is issued, and we wait on that event before
        # reusing the slot.
        #
        # The ring size needs to cover the longest plausible burst of
        # back-to-back ``_stack_vectors_to_device`` calls inside one
        # ``register_config``: one per hook point. With
        # ``HOOK_POINT_TABLE_ATTR`` at 5 entries plus a small safety
        # margin, 6 slots is enough that under steady state every reuse
        # finds the H2D already complete (event wait is a no-op).
        self._stack_pinned_ring: list[torch.Tensor | None] = [None] * 6
        self._stack_pinned_events: list[torch.cuda.Event | None] = [None] * 6
        self._stack_pinned_numel: list[int] = [0] * 6
        self._stack_pinned_next: int = 0

    # ------------------------------------------------------------------
    # Dirty-flag adapters (thin views over ``self._dirty``) — kept so the
    # runner-mixin dispatch ladder and existing tests can read/reset the
    # three historical flag names unchanged.
    # ------------------------------------------------------------------

    @property
    def _tables_dirty(self) -> bool:
        return self._dirty.content

    @_tables_dirty.setter
    def _tables_dirty(self, value: bool) -> None:
        self._dirty.content = bool(value)

    @property
    def _scales_dirty(self) -> bool:
        return self._dirty.scales

    @_scales_dirty.setter
    def _scales_dirty(self, value: bool) -> None:
        self._dirty.scales = bool(value)

    @property
    def _indices_dirty(self) -> bool:
        return self._dirty.membership

    @_indices_dirty.setter
    def _indices_dirty(self, value: bool) -> None:
        self._dirty.membership = bool(value)

    # ------------------------------------------------------------------
    # Per-owner scale adapters — legacy-shaped read views over
    # ``self._row_scales`` for the ``/v1/steering/dynamic`` status payload.
    # ------------------------------------------------------------------

    @property
    def _global_scales(self) -> dict[str, float]:
        return {o.phase: s for o, s in self._row_scales.items() if o.kind == "global"}

    @property
    def _config_scales(self) -> dict[tuple[int, str], float]:
        return {
            (o.config_hash, o.phase): s
            for o, s in self._row_scales.items()
            if o.kind == "config"
        }

    @property
    def _dynamic_scales(self) -> dict[int, float]:
        return {o.dyn_id: s for o, s in self._row_scales.items() if o.kind == "dyn"}

    # ------------------------------------------------------------------
    # Owner-keyed store registry + single purge path
    # ------------------------------------------------------------------

    def _owner_stores(self) -> list[OwnerStore]:
        """Every owner-keyed runtime store, described uniformly.

        Central registry so :meth:`_purge_owner` drops all of an owner's
        state in one place and a parametrized test asserts each store is
        purged (a future owner-keyed store added here without a working
        ``purge`` fails that test). ``install_dummy`` exists only so the
        test can populate each store generically.
        """
        return [
            OwnerStore(
                "row_scales",
                contains=lambda o: o in self._row_scales,
                purge=lambda o: self._row_scales.pop(o, None) is not None,
                install_dummy=lambda o: self._row_scales.__setitem__(o, 0.5),
            ),
            OwnerStore(
                "row_monitor",
                contains=self._row_monitor_has_owner,
                purge=self._row_monitor_purge_owner,
                install_dummy=self._row_monitor_install_dummy,
            ),
        ]

    def _row_monitor_has_owner(self, owner: RowOwner) -> bool:
        return any(
            owner in owners
            for layers in self._row_monitor.values()
            for owners in layers.values()
        )

    def _row_monitor_purge_owner(self, owner: RowOwner) -> bool:
        removed = False
        for layers in self._row_monitor.values():
            for owners in layers.values():
                if owners.pop(owner, None) is not None:
                    removed = True
        return removed

    def _row_monitor_install_dummy(self, owner: RowOwner) -> None:
        """Attach a throwaway per-row monitor for ``owner`` (test helper)."""
        self._row_monitor.setdefault("post_block", {}).setdefault(0, {})[owner] = {
            "probe": torch.zeros(1, dtype=torch.float32),
            "threshold": 0.0,
            "sharpness": 1.0,
        }

    def _purge_owner(self, owner: RowOwner) -> None:
        """Drop every owner-keyed runtime store entry for ``owner``.

        The single cleanup path shared by both release routes
        (:meth:`release_dynamic_config` and the refcount-0 branch of
        :meth:`release_config`). Purges per-row scales and per-row monitors
        and invalidates the affected signature caches / dirty flags. Freeing
        the owner's table row (and the resulting membership/content dirty) is
        the caller's responsibility.
        """
        purged = {store.name: store.purge(owner) for store in self._owner_stores()}
        if purged.get("row_scales"):
            self._dirty.mark_scales()
        if purged.get("row_monitor"):
            self._row_monitor_sig_cache = None

    def register_config(
        self,
        config_hash: int,
        vectors: dict[str, dict[int, list[float] | np.ndarray]],
        phase: str = "prefill",
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> int:
        """Register a steering config, return its table row index.

        Args:
            config_hash: Deterministic hash identifying the config.
            vectors: ``{hook_point_str: {layer_idx: vec}}`` where ``vec`` is
                either a ``list[float]`` (legacy) or a 1-D ``np.ndarray``
                (the float64 arrays produced by
                :func:`resolve_effective_vectors`).
            phase: ``"prefill"`` or ``"decode"``
            locally_owned_layers: If provided, only layers in this set
                have tensors materialized on this worker.  Layers
                outside the set are skipped at tensor-construction time
                but row allocation still proceeds, so row IDs remain
                identical across ranks (distributed-steering
                determinism contract).  When ``None`` (default), no
                filtering — all layers in ``vectors`` get tensors.

        If the ``(config_hash, phase)`` pair is already registered,
        increments refcount and returns the existing row. Otherwise
        assigns a new row. The same ``config_hash`` with a different
        phase gets its own independent row.

        Raises RuntimeError if no free rows are available.
        """
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            return self.config_to_row[key]

        if not self.free_rows:
            raise RuntimeError(
                f"No free steering table rows. max_steering_configs="
                f"{self.max_steering_configs}, active configs="
                f"{len(self.config_to_row)}"
            )

        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_refcounts[key] = 1
        # Store per-request vectors as tensors, keyed by hook point.
        # Under PP, each rank only owns a subset of decoder layers, so
        # materializing tensors for non-local layers is pure waste.
        # Row allocation above is unconditional — the filter only
        # affects what tensors get constructed, not which row is
        # assigned.
        self.config_vectors[key] = self._store_vectors(vectors, locally_owned_layers)
        # New row content + a changed row set: rebuild indices scratch and
        # recompose the tables on the next populate (membership implies
        # content). (Refcount-hit path doesn't mark dirty because the row's
        # contents are already in the table.)
        self._dirty.mark_membership()
        return row

    def _store_vectors(
        self,
        vectors: dict[str, dict[int, list[float] | np.ndarray]],
        locally_owned_layers: frozenset[int] | None,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Materialize per-layer vectors as device tensors.

        Per-layer vectors are batched into ONE stacked H2D copy per hook
        point. Building each row as its own ``torch.tensor(list,
        device=cuda)`` triggers a synchronous ``cudaMemcpy`` per layer,
        which dominates the phase-transition cost when many configs are
        registered at the start of a decode step. Stacking up front and
        transferring once amortizes the sync to a single cost per hook.
        """
        stored: dict[str, dict[int, torch.Tensor]] = {}
        for hook_point, layer_vecs in vectors.items():
            items = [
                (layer_idx, vec)
                for layer_idx, vec in layer_vecs.items()
                if locally_owned_layers is None or layer_idx in locally_owned_layers
            ]
            if not items:
                stored[hook_point] = {}
                continue
            layer_idxs = [layer_idx for layer_idx, _ in items]
            raw_vecs = [vec for _, vec in items]
            stacked = self._stack_vectors_to_device(raw_vecs)
            # ``stacked[i:i+1]`` is a (1, hidden) view, matching the
            # per-layer ``.unsqueeze(0)`` shape that ``_populate_one_table``
            # expects to ``.squeeze(0)``. No extra copy.
            stored[hook_point] = {
                layer_idx: stacked[i : i + 1] for i, layer_idx in enumerate(layer_idxs)
            }
        return stored

    # ------------------------------------------------------------------
    # Dynamic-override pool (docs/design/dynamic_steering.md §5.2)
    # ------------------------------------------------------------------

    @property
    def has_dynamic(self) -> bool:
        """True if any dynamic-override row is live."""
        return bool(self._dynamic_to_row)

    @property
    def num_active_dynamic_configs(self) -> int:
        return len(self._dynamic_to_row)

    def register_dynamic_config(
        self,
        vectors: dict[str, dict[int, list[float] | np.ndarray]],
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> tuple[int, int]:
        """Allocate a dynamic-override row; returns ``(dyn_id, row)``.

        Rows come from the dedicated dynamic pool, never the
        scheduler-reserved static pool. ``dyn_id`` is a monotonically
        increasing id, never reused — identical across ranks because the
        dynamic register/release sequence is rank-replicated. Raises
        ``RuntimeError`` when the pool is exhausted (callers reject the
        triggering action and keep previous state).
        """
        if not self._dynamic_free_rows:
            raise RuntimeError(
                f"No free dynamic steering rows. "
                f"max_dynamic_steering_configs="
                f"{self.max_dynamic_steering_configs}, active="
                f"{len(self._dynamic_to_row)}"
            )
        dyn_id = self._next_dynamic_id
        self._next_dynamic_id += 1
        row = self._dynamic_free_rows.pop()
        self._dynamic_to_row[dyn_id] = row
        self._dynamic_vectors[dyn_id] = self._store_vectors(
            vectors, locally_owned_layers
        )
        # Cache the override-vector hash for the APC decode signature. Hash
        # the raw input vectors (np/list) — no device sync, rank-identical.
        self._dynamic_sig[dyn_id] = hash_steering_config(vectors)
        self._dirty.mark_membership()
        return dyn_id, row

    def update_dynamic_config(
        self,
        dyn_id: int,
        vectors: dict[str, dict[int, list[float] | np.ndarray]],
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Replace a live dynamic config's vectors in place (same row).

        The common re-emit path: gain/vector changes for an existing
        override rewrite the row's content without free-list churn, so
        the cached populate indices stay valid (``_tables_dirty`` only).
        """
        if dyn_id not in self._dynamic_to_row:
            raise KeyError(f"dynamic steering config {dyn_id} is not registered")
        self._dynamic_vectors[dyn_id] = self._store_vectors(
            vectors, locally_owned_layers
        )
        self._dynamic_sig[dyn_id] = hash_steering_config(vectors)
        self._dirty.mark_content()

    def release_dynamic_config(self, dyn_id: int) -> None:
        """Free a dynamic-override row. No-op for unknown ids.

        Also drops every owner-keyed runtime store for this row's owner
        (``RowOwner.dyn(dyn_id)``) via :meth:`_purge_owner` — the per-row
        monitor and strength scale. Without this, per-request monitors
        installed via ``SteeringMonitorUpdate(req_id=...)`` (and scales) would
        accumulate for the lifetime of the process, since dyn_ids are
        monotonic and never reused.
        """
        row = self._dynamic_to_row.pop(dyn_id, None)
        if row is None:
            return
        self._dynamic_vectors.pop(dyn_id, None)
        self._dynamic_sig.pop(dyn_id, None)
        self._dynamic_free_rows.append(row)
        # Purge the row's owner-keyed runtime state (scale + per-row monitors).
        self._purge_owner(RowOwner.dyn(dyn_id))
        self._dirty.mark_membership()

    def get_dynamic_row(self, dyn_id: int) -> int:
        """Return the table row for a live dynamic config."""
        row = self._dynamic_to_row.get(dyn_id)
        if row is None:
            raise RuntimeError(
                f"dynamic steering config {dyn_id} is not registered; "
                f"the mixin's override bookkeeping must release stale "
                f"ids before routing to them."
            )
        return row

    def release_config(self, config_hash: int, phase: str) -> None:
        """Decrement refcount for ``(config_hash, phase)``.

        Free the row when it reaches 0. On that live->0 transition, also purge
        every owner-keyed runtime store for this config's owner
        (``RowOwner.config(config_hash, phase)``) via :meth:`_purge_owner` —
        its per-config strength scale and any per-row monitors. Without this,
        a scale or monitor set for content hash H would silently re-apply to a
        *future* request that re-registers H (content hashes collide by
        design). A scale pre-armed for a not-yet-registered hash is untouched:
        purge fires only on this live->0 transition.
        """
        key = (config_hash, phase)
        if key not in self.config_to_row:
            return
        self.config_refcounts[key] -= 1
        if self.config_refcounts[key] <= 0:
            row = self.config_to_row.pop(key)
            self.config_vectors.pop(key, None)
            del self.config_refcounts[key]
            self.free_rows.append(row)
            # Last live registration released: drop its owner-keyed runtime
            # state so a re-registration of the same hash starts clean.
            self._purge_owner(RowOwner.config(config_hash, phase))
            # config_to_row shrunk; rebuild indices scratch and recompose the
            # tables on the next populate (membership implies content) so a
            # config later assigned to this row overwrites the stale content.
            self._dirty.mark_membership()

    def get_row_for_config(self, config_hash: int, is_prefill: bool = False) -> int:
        """Return table row for a config.

        For hash == 0 (no per-request steering):
            is_prefill=True  -> row 1 (global prefill effective)
            is_prefill=False -> row 2 (global decode effective)

        For registered per-request configs:
            Returns the assigned row (3+), looked up by
            ``(config_hash, "prefill"/"decode")``.

        Raises ``RuntimeError`` for unregistered nonzero hashes.  The
        scheduler reserves a row for every per-request hash before the
        request is dispatched, so reaching this branch indicates a
        scheduler accounting bug.  Crashing loudly is preferable to
        silently substituting global rows, which would corrupt the
        output of requests that asked for per-request steering.
        """
        if config_hash == 0:
            return 1 if is_prefill else 2
        phase = "prefill" if is_prefill else "decode"
        row = self.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        raise RuntimeError(
            f"Steering config (hash={config_hash}, phase={phase}) is "
            "not registered. The scheduler must guarantee capacity "
            "before dispatching a request that uses per-request "
            "steering; reaching this branch is a scheduler bug."
        )

    def update_global_vectors(
        self,
        hook_point: str,
        layer_idx: int,
        vector: torch.Tensor,
        phase: str = "base",
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Update cached global vector for a hook point and layer.

        Args:
            hook_point: Hook point string (e.g. ``"post_block"``).
            layer_idx: Layer index.
            vector: The global vector tensor.
            phase: ``"base"``, ``"prefill"``, or ``"decode"``.
            locally_owned_layers: If provided and ``layer_idx`` is not
                in the set, this call is a no-op.  Defense-in-depth
                for the distributed-steering determinism contract:
                callers in the mixin already filter by locally-present
                layers, but self-defending the manager means its
                invariants do not depend on the caller.
        """
        if locally_owned_layers is not None and layer_idx not in locally_owned_layers:
            return
        target = self._global_dict_for_phase(phase)
        if hook_point not in target:
            target[hook_point] = {}
        stored = vector.clone()
        # Global vectors are produced on the CONTROL-PLANE thread (the HTTP
        # ``set_steering_vectors`` RPC) but consumed by
        # ``populate_steering_tables`` on the STEP thread. Under the classic
        # Ray executor those are different threads with distinct default CUDA
        # streams (the compiled DAG runs the model on its own background
        # thread), and CUDA only orders work within a stream. The producing
        # H2D + this ``clone()`` are enqueued on the RPC thread's stream; the
        # step thread's ``index_copy_`` into the layer table can otherwise run
        # before that copy drains, baking a stale/garbage row and silently
        # no-op'ing base-tier steering. Per-request configs never hit this
        # because they are both produced and consumed on the step thread (see
        # ``_stack_vectors_to_device``'s same-stream invariant). Synchronize
        # here so the tensor's memory is fully materialized before the manager
        # exposes it for cross-thread reads. This is a rare control-plane op,
        # so the sync cost is irrelevant; the single-rank / non-CUDA paths are
        # unaffected.
        if stored.is_cuda:
            torch.cuda.synchronize(stored.device)
        target[hook_point][layer_idx] = stored
        # Global rows 1, 2 and all per-request rows depend on this state.
        self._dirty.mark_content()

    def clear_global_vectors(self) -> None:
        """Clear all cached global vectors across all phases and hook points."""
        self.global_base_vectors.clear()
        self.global_prefill_vectors.clear()
        self.global_decode_vectors.clear()
        self._dirty.mark_content()

    def update_dynamic_tier(
        self,
        hook_point: str,
        layer_idx: int,
        vector: torch.Tensor,
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Set the dynamic additive-tier vector for a hook point and layer.

        The tier is folded ADDITIVELY into the decode-effective vector at
        populate time (so it reaches row 2, decode per-request rows, and
        dynamic-override rows), letting dynamic global steering compose
        with operator-set decode steering rather than overwriting
        ``global_decode_vectors``. Decode-only (§7): it never touches
        prefill rows and never feeds prefix-cache keys.

        Args:
            hook_point: Hook point string (e.g. ``"post_mlp"``).
            layer_idx: Layer index.
            vector: The (already gain-scaled) tier vector.
            locally_owned_layers: If provided and ``layer_idx`` is not in
                the set, this call is a no-op — the same
                distributed-determinism guard as
                :meth:`update_global_vectors`.
        """
        if locally_owned_layers is not None and layer_idx not in locally_owned_layers:
            return
        if hook_point not in self.dynamic_tier_vectors:
            self.dynamic_tier_vectors[hook_point] = {}
        self.dynamic_tier_vectors[hook_point][layer_idx] = vector.clone()
        self._tier_sig_cache = None  # tier changed → APC signature stale
        self._dirty.mark_content()

    def clear_dynamic_tier(self) -> None:
        """Clear all dynamic additive-tier vectors."""
        if self.dynamic_tier_vectors:
            self.dynamic_tier_vectors.clear()
            self._tier_sig_cache = None
            self._dirty.mark_content()

    @property
    def has_dynamic_tier(self) -> bool:
        """True if any dynamic additive-tier vector is set."""
        return bool(self.dynamic_tier_vectors)

    def set_dynamic_tier_gain(self, gain: float) -> None:
        """Set the scalar strength of the dedicated dynamic tier (§5.4).

        Cheap: the runner reads this when it rebuilds the per-token gate
        each step, so no buffer write happens here.
        """
        self.dynamic_tier_gain = float(gain)

    # ------------------------------------------------------------------
    # In-graph monitor (Phase 2, §8)
    # ------------------------------------------------------------------

    def set_monitor(
        self,
        hook_point: str,
        layer_idx: int,
        probe: torch.Tensor,
        threshold: float,
        sharpness: float,
        gate_rows: bool = False,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Configure the in-graph monitor at ``(hook_point, layer_idx)``.

        The probe is a 1-D detector vector; ``threshold``/``sharpness``
        parameterize the fixed elementwise gate
        ``sigmoid(sharpness*(residual@probe - threshold))`` the monitor op
        writes into ``steering_token_scales``. Stored here and written to
        the per-layer monitor buffers at the next populate; the policy
        params then live host-tunable in the buffers (no recapture).

        ``locally_owned_layers`` (TP/PP): if provided and ``layer_idx`` is
        not owned by this worker, the call is a no-op so rank-replicated
        callers stay in lock-step.
        """
        if locally_owned_layers is not None and layer_idx not in locally_owned_layers:
            return
        self.monitor_configs.setdefault(hook_point, {})[layer_idx] = {
            "probe": probe.detach().to(torch.float32).clone().reshape(-1),
            "threshold": float(threshold),
            "sharpness": float(sharpness),
            "gate_rows": bool(gate_rows),
        }
        self._monitor_sig_cache = None  # monitor changed → APC signature stale
        self._dirty.mark_content()

    def clear_monitor(
        self,
        hook_point: str | None = None,
        layer_idx: int | None = None,
    ) -> None:
        """Remove monitor configs.

        No arguments clears every site; a ``hook_point`` clears that
        hook's sites; both clear a single ``(hook, layer)`` site. Marks
        tables dirty so the next populate deactivates the cleared buffers.
        """
        if not self.monitor_configs:
            return
        if hook_point is None:
            self.monitor_configs.clear()
            self._monitor_sig_cache = None
            self._dirty.mark_content()
            return
        layers = self.monitor_configs.get(hook_point)
        if layers is None:
            return
        if layer_idx is None:
            del self.monitor_configs[hook_point]
            self._monitor_sig_cache = None
            self._dirty.mark_content()
            return
        if layer_idx in layers:
            del layers[layer_idx]
            if not layers:
                del self.monitor_configs[hook_point]
            self._monitor_sig_cache = None
            self._dirty.mark_content()

    @property
    def has_monitor(self) -> bool:
        """True if any in-graph monitor site is configured."""
        return any(layers for layers in self.monitor_configs.values())

    # ------------------------------------------------------------------
    # Per-row (per-request) in-graph monitor
    # ------------------------------------------------------------------

    def set_row_monitor(
        self,
        hook_point: str,
        layer_idx: int,
        owner: RowOwner,
        probe: torch.Tensor,
        threshold: float,
        sharpness: float,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Configure the per-row monitor for one logical row owner at a site.

        ``owner`` is a :class:`RowOwner` — ``RowOwner.global_("decode")``,
        ``RowOwner.config(config_hash, "decode")`` or ``RowOwner.dyn(dyn_id)``
        — the gate ``sigmoid(sharpness*(residual@probe - threshold))`` is
        applied to that owner's row term only, decode-only. Stored keyed by
        owner so it survives row reassignment; written into the per-row probe
        table at the next populate.

        ``locally_owned_layers`` (TP/PP): a no-op when ``layer_idx`` is not
        owned by this worker (rank-replicated callers stay in lock-step).
        """
        if locally_owned_layers is not None and layer_idx not in locally_owned_layers:
            return
        self._row_monitor.setdefault(hook_point, {}).setdefault(layer_idx, {})[
            owner
        ] = {
            "probe": probe.detach().to(torch.float32).clone().reshape(-1),
            "threshold": float(threshold),
            "sharpness": float(sharpness),
        }
        self._row_monitor_sig_cache = None
        self._dirty.mark_content()

    def clear_row_monitor(
        self,
        hook_point: str | None = None,
        layer_idx: int | None = None,
        owner: RowOwner | None = None,
    ) -> None:
        """Remove per-row monitor configs.

        No args clears everything; progressively narrower args clear a hook,
        a ``(hook, layer)`` site, or a single ``(hook, layer, owner)`` entry.
        """
        if not self._row_monitor:
            return
        if hook_point is None:
            self._row_monitor.clear()
            self._row_monitor_sig_cache = None
            self._dirty.mark_content()
            return
        layers = self._row_monitor.get(hook_point)
        if layers is None:
            return
        if layer_idx is None:
            del self._row_monitor[hook_point]
        elif owner is None:
            layers.pop(layer_idx, None)
            if not layers:
                del self._row_monitor[hook_point]
        else:
            owners = layers.get(layer_idx)
            if owners is None:
                return
            owners.pop(owner, None)
            if not owners:
                layers.pop(layer_idx, None)
            if not layers:
                del self._row_monitor[hook_point]
        self._row_monitor_sig_cache = None
        self._dirty.mark_content()

    @property
    def has_row_monitor(self) -> bool:
        """True if any per-row monitor entry is configured."""
        return any(
            owners
            for layers in self._row_monitor.values()
            for owners in layers.values()
        )

    def _build_row_probe_and_params(
        self,
        hp_str: str,
        layer_idx: int,
        device: torch.device,
        hidden_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Build the per-row probe table + ``[threshold, sharpness]`` params in
        populate (row-position) order ``[0, 1, 2, *config_rows,
        *dynamic_rows]`` — matching ``_cached_indices`` — for one ``(hook,
        layer)`` site.

        Unconfigured rows (and rows 0/1, and all prefill rows) get a zero
        probe + the default ``[-1e30, 1.0]`` params ⇒ ``sigmoid → 1.0`` ⇒
        ungated pass-through. Returns ``(probe_mat, params_mat,
        any_configured)``.
        """
        site = self._row_monitor.get(hp_str, {}).get(layer_idx, {})
        ordered_configs = self._cached_ordered_configs or list(
            self.config_to_row.items()
        )
        ordered_dynamic = self._cached_ordered_dynamic or list(
            self._dynamic_to_row.items()
        )
        num_rows = 3 + len(ordered_configs) + len(ordered_dynamic)
        thr0, sharp0 = _ROW_MONITOR_DEFAULT_PARAMS
        probe_mat = torch.zeros(
            num_rows, hidden_size, dtype=torch.float32, device=device
        )
        params_mat = (
            torch.tensor([thr0, sharp0], dtype=torch.float32, device=device)
            .expand(num_rows, 2)
            .clone()
        )
        any_configured = False

        def _apply(pos: int, owner: RowOwner) -> None:
            nonlocal any_configured
            cfg = site.get(owner)
            if cfg is None:
                return
            probe = cfg["probe"]
            if probe.numel() != hidden_size:
                return
            probe_mat[pos].copy_(probe.to(device))
            params_mat[pos, 0] = cfg["threshold"]
            params_mat[pos, 1] = cfg["sharpness"]
            any_configured = True

        # Row 2 = global decode; rows 0/1 (sentinel/prefill) stay default.
        _apply(2, RowOwner.global_("decode"))
        pos = 3
        for (config_hash, phase), _row in ordered_configs:
            if phase == "decode":
                _apply(pos, RowOwner.config(config_hash, "decode"))
            pos += 1
        for dyn_id, _row in ordered_dynamic:
            _apply(pos, RowOwner.dyn(dyn_id))
            pos += 1
        return probe_mat, params_mat, any_configured

    def _row_monitor_signature_for(
        self, owner_keys: tuple[RowOwner, ...]
    ) -> int | None:
        """Cached hash of every per-row monitor entry matching any of
        ``owner_keys`` (probe + threshold + sharpness + site), or ``None`` when
        none match. Used to fold a request's effective row monitor into its APC
        decode signature."""
        if self._row_monitor_sig_cache is None:
            cache: dict[RowOwner, int] = {}
            for hook in sorted(self._row_monitor.keys()):
                layers = self._row_monitor[hook]
                for layer_idx in sorted(layers.keys()):
                    # Sort by the RowOwner total order so the fold is
                    # insertion-order independent (free determinism insurance).
                    for owner_key in sorted(layers[layer_idx].keys()):
                        cfg = layers[layer_idx][owner_key]
                        h = cache.get(owner_key)
                        acc = hashlib.sha256(b"dynsteer-rowmon")
                        if h is not None:
                            acc.update(int(h).to_bytes(8, "little"))
                        acc.update(hook.encode())
                        acc.update(int(layer_idx).to_bytes(4, "little", signed=True))
                        acc.update(
                            cfg["probe"]
                            .detach()
                            .cpu()
                            .to(torch.float32)
                            .numpy()
                            .tobytes()
                        )
                        acc.update(np.float64(cfg["threshold"]).tobytes())
                        acc.update(np.float64(cfg["sharpness"]).tobytes())
                        cache[owner_key] = (
                            int(acc.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
                        )
            self._row_monitor_sig_cache = cache
        out: int | None = None
        for key in owner_keys:
            sig = self._row_monitor_sig_cache.get(key)
            if sig is not None:
                out = sig if out is None else (out ^ sig)
        return out

    # ------------------------------------------------------------------
    # APC effective-decode-steering signature (see
    # docs/design/dynamic_steering_apc_notification.md)
    # ------------------------------------------------------------------

    def _tier_signature(self) -> int:
        """Cached hash of the global dynamic-tier vectors (gain excluded —
        the gain is folded per-call since it is a cheap scalar that the
        caller reads fresh)."""
        if self._tier_sig_cache is None:
            self._tier_sig_cache = self._hash_tensor_vectors(self.dynamic_tier_vectors)
        return self._tier_sig_cache

    def _monitor_signature(self) -> int:
        """Cached hash of the global monitor configs (probe + params)."""
        if self._monitor_sig_cache is None:
            h = hashlib.sha256(b"dynsteer-monitor")
            for hook in sorted(self.monitor_configs.keys()):
                layers = self.monitor_configs[hook]
                for layer_idx in sorted(layers.keys()):
                    cfg = layers[layer_idx]
                    h.update(hook.encode())
                    h.update(int(layer_idx).to_bytes(4, "little", signed=True))
                    h.update(
                        cfg["probe"].detach().cpu().to(torch.float32).numpy().tobytes()
                    )
                    h.update(np.float64(cfg["threshold"]).tobytes())
                    h.update(np.float64(cfg["sharpness"]).tobytes())
                    h.update(b"\x01" if cfg.get("gate_rows") else b"\x00")
            self._monitor_sig_cache = int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF
        return self._monitor_sig_cache

    @staticmethod
    def _hash_tensor_vectors(
        vectors: dict[str, dict[int, torch.Tensor]],
    ) -> int:
        """Deterministic, rank-identical hash of a hook→layer→tensor dict,
        routed through :func:`hash_steering_config` (fp32 ``tobytes``)."""
        if not vectors:
            return 0
        converted = {
            hook: {
                layer: t.detach().cpu().to(torch.float32).numpy()
                for layer, t in layers.items()
            }
            for hook, layers in vectors.items()
        }
        return hash_steering_config(converted)

    def effective_decode_signature(
        self, dyn_id: int | None, base_decode_hash: int
    ) -> int | None:
        """Per-request effective decode steering signature, or ``None``.

        Returns ``None`` when no dynamic decode steering applies to the
        request (admitted ``base_decode_hash`` already identifies the KV).
        Otherwise returns a deterministic hash folding the admitted decode
        config with whatever dynamic steering shaped the decode KV — a
        per-request override, the global dynamic tier (+ gain), and/or the
        global in-graph monitor — so steered decode blocks are keyed by the
        steering that produced them (and only reused by requests under the
        identical effective steering). Rank-identical: all inputs are
        rank-replicated, hashed with the same ``hash_steering_config``.
        """
        has_override = dyn_id is not None and dyn_id in self._dynamic_sig
        has_tier = self.has_dynamic_tier
        has_monitor = self.has_monitor
        # Per-row monitor on THIS request's decode row: keyed by its dyn
        # override (if any) else its static decode config, plus the global
        # decode row (row 2) a no-config decode request routes through. The
        # probe/params are runtime state not in ``base_decode_hash``, so fold
        # them in — else a temporal probe change reuses stale steered KV.
        if dyn_id is not None:
            row_owner_keys: tuple[RowOwner, ...] = (RowOwner.dyn(dyn_id),)
        else:
            row_owner_keys = (
                RowOwner.config(base_decode_hash, "decode"),
                RowOwner.global_("decode"),
            )
        row_mon_sig = self._row_monitor_signature_for(row_owner_keys)
        has_row_mon = row_mon_sig is not None
        if not (has_override or has_tier or has_monitor or has_row_mon):
            return None
        h = hashlib.sha256(b"dynsteer-decode-sig")
        h.update(int(base_decode_hash).to_bytes(8, "little", signed=False))
        if has_override:
            h.update(b"\x01ovr")
            h.update(int(self._dynamic_sig[dyn_id]).to_bytes(8, "little"))
        if has_tier:
            h.update(b"\x02tier")
            h.update(int(self._tier_signature()).to_bytes(8, "little"))
            # No quantization (decision locked): any gain change ⇒ new key.
            h.update(np.float64(self.dynamic_tier_gain).tobytes())
        if has_monitor:
            h.update(b"\x03mon")
            h.update(int(self._monitor_signature()).to_bytes(8, "little"))
        if has_row_mon:
            h.update(b"\x04rowmon")
            h.update(int(row_mon_sig).to_bytes(8, "little"))
        return int(h.hexdigest()[:16], 16) & 0x7FFFFFFFFFFFFFFF

    # ------------------------------------------------------------------
    # Per-row strength scales (§5.3) — cheap "how much" knob
    # ------------------------------------------------------------------

    def set_global_scale(self, phase: str, scale: float) -> None:
        """Set the strength scale for the global prefill/decode row.

        ``phase`` is ``"prefill"`` or ``"decode"`` (rows 1 / 2). Decode is
        the cache-safe knob; a prefill scale is accepted and stored but
        forced to 1.0 at populate time (§7) so it never takes effect —
        callers should use decode.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(
                f"global scale phase must be prefill/decode, got {phase!r}"
            )
        self._row_scales[RowOwner.global_(phase)] = float(scale)
        self._dirty.mark_scales()

    def set_row_scale(self, config_hash: int, phase: str, scale: float) -> None:
        """Set the strength scale for a static per-request config row."""
        if phase not in ("prefill", "decode"):
            raise ValueError(f"row scale phase must be prefill/decode, got {phase!r}")
        self._row_scales[RowOwner.config(config_hash, phase)] = float(scale)
        self._dirty.mark_scales()

    def set_dynamic_scale(self, dyn_id: int, scale: float) -> None:
        """Set the strength scale for a dynamic-override row (by dyn_id)."""
        self._row_scales[RowOwner.dyn(dyn_id)] = float(scale)
        self._dirty.mark_scales()

    def clear_scales(self) -> None:
        """Reset every row's scale to the 1.0 default."""
        if self._row_scales:
            self._row_scales.clear()
            self._dirty.mark_scales()

    def _build_scales_vector(self, device: torch.device) -> torch.Tensor:
        """Assemble the per-row scale vector in populate (row-position)
        order ``[0, 1, 2, *config_rows, *dynamic_rows]`` — matching
        ``_cached_indices`` — so it can be scattered with the same index
        tensor the table write uses.

        Row 0 (sentinel) and any prefill row are pinned to 1.0: scaling
        them is meaningless (row 0) or cache-unsafe (prefill, §7).
        """
        ordered_configs = self._cached_ordered_configs or list(
            self.config_to_row.items()
        )
        ordered_dynamic = self._cached_ordered_dynamic or list(
            self._dynamic_to_row.items()
        )
        scales: list[float] = [
            1.0,  # row 0 sentinel
            1.0,  # row 1 global prefill — never scaled (cache safety)
            self._row_scales.get(RowOwner.global_("decode"), 1.0),  # row 2 decode
        ]
        for (config_hash, phase), _row in ordered_configs:
            # Prefill rows pinned to 1.0; decode rows take their scale.
            scale = (
                self._row_scales.get(RowOwner.config(config_hash, phase), 1.0)
                if phase == "decode"
                else 1.0
            )
            scales.append(scale)
        for dyn_id, _row in ordered_dynamic:
            scales.append(self._row_scales.get(RowOwner.dyn(dyn_id), 1.0))
        return torch.tensor(scales, dtype=torch.float32, device=device)

    def populate_steering_scales(
        self, steerable_layers: dict[int, "torch.nn.Module"]
    ) -> None:
        """Cheap path: write ONLY the per-row scale buffers, no table
        recompose. Called when ``_scales_dirty`` but not ``_tables_dirty``.

        Writes the same per-row scale vector into every layer's
        ``steering_scales`` buffer (each is tiny — ``num_rows`` floats).
        """
        if self._indices_dirty or self._cached_indices is None:
            # Indices stale (config/dynamic membership changed) — fall back
            # to a full populate which rebuilds indices and writes scales.
            self._dirty.mark_content()
            self.populate_steering_tables(steerable_layers)
            return
        indices = self._cached_indices
        written: set[int] = set()
        for layer_idx, mod in steerable_layers.items():
            scales_buf = getattr(mod, "steering_scales", None)
            if scales_buf is None or id(scales_buf) in written:
                continue
            scales_vec = self._build_scales_vector(scales_buf.device)
            scales_buf.index_copy_(0, indices, scales_vec)
            written.add(id(scales_buf))
        self._dirty.scales = False

    def _global_dict_for_phase(self, phase: str) -> dict[str, dict[int, torch.Tensor]]:
        """Return the global vector dict for the given phase."""
        if phase == "base":
            return self.global_base_vectors
        elif phase == "prefill":
            return self.global_prefill_vectors
        elif phase == "decode":
            return self.global_decode_vectors
        else:
            raise ValueError(
                f"Invalid global vector phase: {phase!r}. "
                f"Must be 'base', 'prefill', or 'decode'."
            )

    def _get_global_vec(
        self,
        hp_str: str,
        layer_idx: int,
        source: dict[str, dict[int, torch.Tensor]],
    ) -> torch.Tensor | None:
        """Look up a global vector, returning None if absent."""
        return source.get(hp_str, {}).get(layer_idx)

    def _add_vecs(
        self,
        *vecs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Additively combine non-None tensors. Returns None if all None."""
        result: torch.Tensor | None = None
        for v in vecs:
            if v is None:
                continue
            squeezed = v.squeeze(0)
            result = squeezed.clone() if result is None else result + squeezed
        return result

    def _stack_vectors_to_device(
        self, vecs: list[list[float] | np.ndarray]
    ) -> torch.Tensor:
        """Stack a list of equal-length float vectors into a (N, hidden)
        tensor on ``self.device``.

        For CUDA targets this returns the device tensor IMMEDIATELY; the
        underlying H2D ``cudaMemcpyAsync`` is queued on the current CUDA
        stream with ``non_blocking=True`` and has not necessarily completed
        by the time we return. This is safe because ``populate_steering_tables``
        — the only consumer that reads from the returned tensor — runs on
        the same default stream, and CUDA preserves in-stream ordering: any
        op that touches the destination on this stream observes the copy
        as already finished.

        Implementation notes:

        * The copy uses a reusable pinned-CPU staging buffer
          (``self._stack_pinned_cpu``). ``torch.Tensor.pin_memory()`` does
          a synchronous host-side copy into a freshly page-locked region,
          which is the dominant cost for multi-MB stacks. Reusing one
          allocation amortizes that to one-time work.
        * Inputs whose total byte size exceeds ``_STACK_PINNED_CAP_BYTES``
          fall back to a non-pinned ``torch.from_numpy`` + ``non_blocking=True``
          copy. Pinned host memory is a finite resource (locked, not
          swappable); unbounded growth is not an option. The cap is
          generous enough to cover Gemma-3-4B-class workloads
          (~6 MB / hook).
        * The signature still returns a device tensor, matching what
          ``register_config`` expects to slice with ``stacked[i:i+1]``.
        """
        try:
            arr = np.asarray(vecs, dtype=np.float32)
        except (ValueError, TypeError) as exc:  # ragged / non-numeric input
            raise ValueError(
                "register_config received steering vectors of inconsistent "
                "shape or non-numeric dtype; expected a list of equal-length "
                f"float vectors. Underlying error: {exc}"
            ) from exc
        if arr.ndim != 2:
            raise ValueError(
                "register_config expected a 2D stack of steering vectors "
                f"(N, hidden); got array with shape {arr.shape}."
            )

        if self.device is None:
            # CPU-only path: no copy needed at all, the numpy buffer is
            # the storage.
            return torch.from_numpy(arr)

        if self.device.type != "cuda":
            # Non-CUDA accelerator (e.g. xpu, hpu) — there's no pinned-host
            # concept that helps us, so do the simple copy.
            return torch.from_numpy(arr).to(self.device)

        # CUDA path: pinned-ring-backed async copy.
        numel = arr.size
        nbytes = arr.nbytes
        if nbytes > self._STACK_PINNED_CAP_BYTES:
            # Outlier: don't lock that much host memory just for this call.
            # Still ``non_blocking=True`` so the copy enqueues without
            # blocking the host on driver-side queue submission, but the
            # source is pageable so the runtime does an internal staging
            # copy that's effectively synchronous w.r.t. the host. That's
            # acceptable for the rare-large case.
            cpu_t = torch.from_numpy(arr)
            return cpu_t.to(self.device, non_blocking=True)

        slot = self._stack_pinned_next
        ring_size = len(self._stack_pinned_ring)
        self._stack_pinned_next = (slot + 1) % ring_size

        # If a previous H2D from this slot is still in flight, wait for it
        # to drain on the current stream before we reuse the host buffer.
        # In steady state this event has long since completed and the wait
        # is a no-op (microseconds). The wait happens on the CUDA stream,
        # not the host — host-side ``copy_`` below could still race the
        # DMA, so we follow this with an explicit ``event.synchronize()``
        # to make the host wait too. With a 4-slot ring this only gates
        # on the H2D from 4 calls ago, which is essentially always done.
        prev_event = self._stack_pinned_events[slot]
        if prev_event is not None:
            prev_event.synchronize()

        # Grow the pinned slot if it's too small. Slots grow monotonically
        # so a steady-state workload pays the pin cost once per slot.
        if (
            self._stack_pinned_ring[slot] is None
            or self._stack_pinned_numel[slot] < numel
        ):
            try:
                self._stack_pinned_ring[slot] = torch.empty(
                    numel, dtype=torch.float32, pin_memory=True
                )
                self._stack_pinned_numel[slot] = numel
            except RuntimeError:
                # Pinned allocation failed (e.g. CPU-only test env, or
                # pinned-memory exhausted). Fall back to non-pinned copy
                # without poisoning the slot — a future call may succeed.
                self._stack_pinned_ring[slot] = None
                self._stack_pinned_numel[slot] = 0
                self._stack_pinned_events[slot] = None
                cpu_t = torch.from_numpy(arr)
                return cpu_t.to(self.device, non_blocking=True)

        pinned = self._stack_pinned_ring[slot]
        assert pinned is not None
        flat_view = pinned[:numel]
        # ``copy_`` from a numpy-backed tensor of identical dtype is a
        # plain host memcpy into the pinned buffer — no extra pin, no
        # tensor allocation beyond the temporary view.
        flat_view.copy_(torch.from_numpy(arr.reshape(-1)))
        cpu_view = flat_view.view(arr.shape)
        # ``non_blocking=True`` from a pinned source is a true async H2D
        # on the current stream. The returned device tensor is a fresh
        # allocation owned by the caller; the pinned source can be
        # overwritten safely once the recorded event below has fired.
        device_t = cpu_view.to(self.device, non_blocking=True)

        # Record an event on the current stream right after the H2D was
        # enqueued, so the next user of this slot knows when the DMA has
        # drained. Re-using a stale ``cuda.Event`` would race the previous
        # ``record()``, so we always allocate a fresh one — Event objects
        # are cheap (just a CUDA event handle).
        ev = torch.cuda.Event()
        ev.record()
        self._stack_pinned_events[slot] = ev
        return device_t

    # Soft cap on pinned-CPU staging-buffer size. Sized to hold one Gemma-3
    # hook's worth of vectors comfortably (~5.8 MB) with headroom for larger
    # models. Inputs above this fall back to a non-pinned copy rather than
    # locking unbounded host memory.
    _STACK_PINNED_CAP_BYTES: int = 32 * 1024 * 1024

    def populate_steering_tables(
        self, steerable_layers: dict[int, "torch.nn.Module"]
    ) -> None:
        """Write current state into each layer's per-hook steering_table
        buffers.

        For each hook point that has a table buffer on a layer
        (``global_decode`` below = global_base + global_decode +
        dynamic_tier, the decode-effective vector):
            Row 0 = zeros (always)
            Row 1 = global_base + global_prefill (or zeros)
            Row 2 = global_decode effective (or zeros)
            Rows 3+ = phase-appropriate global + per_request
            Dynamic-override rows = global_decode effective + override

        Optimizations vs. the naive per-(hook, layer) loop:

        1.  ``indices`` (GPU int64) and ``zero_row`` (GPU fp32) scratch
            tensors are cached on the manager and only rebuilt when
            ``config_to_row`` mutates (register/release). Global-vector
            updates do NOT invalidate them.
        2.  Row assembly across all active (hook, layer) tables produces
            a single ``(num_active_tables, num_rows, hidden)`` fp32
            tensor that is dtype-cast in one kernel launch, then written
            to each table via ``index_copy_``. This consolidates ~84
            independent ``stacked.to(dtype=...)`` casts on Gemma-3-4B
            (3 hooks * 28 layers) into one.
        """
        # Build a flat list of (table_buffer, hp_str, layer_idx, mod) for
        # every (hook, layer) pair that actually has a table buffer
        # registered.  ``mod`` is carried through so the per-hook
        # ``_any_active`` flag buffer can be written alongside the table
        # body once the rows are assembled below.  Layers may register a
        # SUBSET of hook tables (the ``hasattr(mod, table_attr)`` check),
        # so this drives the batched scatter on the active set rather
        # than assuming a dense layout.
        active_tables: list[tuple[torch.Tensor, str, int, torch.nn.Module]] = []
        for hook_point, table_attr in HOOK_POINT_TABLE_ATTR.items():
            hp_str = hook_point.value
            for layer_idx, mod in steerable_layers.items():
                if not hasattr(mod, table_attr):
                    continue
                table = getattr(mod, table_attr)
                active_tables.append((table, hp_str, layer_idx, mod))

        if not active_tables:
            self._dirty.content = False
            return

        # Derive device and hidden_size from the first active table.
        # All tables registered through ``register_steering_buffers`` share
        # the same device and hidden_size by construction.
        first_table = active_tables[0][0]
        device = first_table.device
        hidden_size = first_table.shape[1]

        # Per-(hook, layer) "any non-zero row" tracking.  Filled in during
        # row assembly below and written into each layer's ``_any_active``
        # flag tensor at the end.  A layer's flag is True iff any row
        # >= 1 carries a non-zero contribution — i.e. at least one of the
        # global prefill / global decode / per-request rows is not the
        # zero sentinel.  When the flag is False, the apply_steering
        # kernel skips the gather + add and just emits hidden_states.
        per_table_any_active: list[bool] = []
        # Per active-table dynamic-tier vector (or None), captured during
        # row assembly and written into each layer's ``dynamic_vec`` buffer
        # at the end (dedicated-gather, §5.4).
        per_table_tier_vec: list[torch.Tensor | None] = []

        # Snapshot config_to_row ordering. This is ALWAYS needed for the
        # row-assembly loop below, but ``indices`` only needs rebuilding
        # when this ordering changed (register/release).
        if (
            self._dirty.membership
            or self._cached_indices is None
            or self._cached_zero_row is None
            or self._cached_ordered_configs is None
            or self._cached_ordered_dynamic is None
        ):
            new_ordered_configs: list[tuple[tuple[int, str], int]] = list(
                self.config_to_row.items()
            )
            new_ordered_dynamic: list[tuple[int, int]] = list(
                self._dynamic_to_row.items()
            )
            target_indices_list = (
                [0, 1, 2]
                + [row for _, row in new_ordered_configs]
                + [row for _, row in new_ordered_dynamic]
            )
            self._cached_indices = torch.tensor(
                target_indices_list, dtype=torch.long, device=device
            )
            self._cached_zero_row = torch.zeros(
                hidden_size, dtype=torch.float32, device=device
            )
            self._cached_ordered_configs = new_ordered_configs
            self._cached_ordered_dynamic = new_ordered_dynamic
            self._dirty.membership = False
        indices: torch.Tensor = self._cached_indices
        zero_row: torch.Tensor = self._cached_zero_row
        ordered_configs: list[tuple[tuple[int, str], int]] = (
            self._cached_ordered_configs
        )
        ordered_dynamic: list[tuple[int, int]] = self._cached_ordered_dynamic

        # Build all rows for all active tables in fp32. ``all_rows`` ends
        # up shape ``(num_active_tables, num_rows, hidden)``. We do ONE
        # ``.to(dtype=table.dtype)`` cast on the whole stack instead of
        # per-(hook, layer), then index_copy_ each layer's slice.
        num_rows = 3 + len(ordered_configs) + len(ordered_dynamic)
        per_table_rows: list[list[torch.Tensor]] = []
        for _table, hp_str, layer_idx, _mod in active_tables:
            base_vec = self._get_global_vec(hp_str, layer_idx, self.global_base_vectors)
            prefill_vec = self._get_global_vec(
                hp_str, layer_idx, self.global_prefill_vectors
            )
            decode_vec = self._get_global_vec(
                hp_str, layer_idx, self.global_decode_vectors
            )
            # Dynamic additive tier (§5.4, dedicated-gather): NOT folded
            # into the rows. It lives in a per-(layer, hook) ``dynamic_vec``
            # buffer the kernel adds on top of the row gather, gated
            # per-token (decode-only). Captured here to write that buffer
            # below and to flag ``any_active``.
            tier_vec = self._get_global_vec(
                hp_str, layer_idx, self.dynamic_tier_vectors
            )
            per_table_tier_vec.append(tier_vec)

            global_prefill = self._add_vecs(base_vec, prefill_vec)
            global_decode = self._add_vecs(base_vec, decode_vec)

            # ``any_active`` is True iff at least one row >= 1 carries a
            # non-zero contribution — equivalent to "not every row >= 1 is
            # the ``zero_row`` sentinel".  Tracked as we append rows. The
            # dedicated tier also counts: the kernel must run to apply it
            # even when no table row is active.
            any_active = tier_vec is not None

            rows: list[torch.Tensor] = [zero_row]  # row 0: always zero
            if global_prefill is not None:
                rows.append(global_prefill)
                any_active = True
            else:
                rows.append(zero_row)
            if global_decode is not None:
                rows.append(global_decode)
                any_active = True
            else:
                rows.append(zero_row)

            for (config_hash, phase), _row_idx in ordered_configs:
                per_req = (
                    self.config_vectors.get((config_hash, phase), {})
                    .get(hp_str, {})
                    .get(layer_idx)
                )
                if phase == "prefill":
                    phase_global = global_prefill
                elif phase == "decode":
                    phase_global = global_decode
                else:
                    raise ValueError(
                        f"Invalid phase: {phase!r}. Must be 'prefill' or 'decode'."
                    )

                if phase_global is not None and per_req is not None:
                    # Per-request vectors are registered from raw Python lists
                    # and default to CPU; global vectors inherit the model's
                    # device. Align to the global's device before adding so a
                    # CPU/CUDA mix doesn't raise.
                    per_req_aligned = per_req.squeeze(0).to(phase_global.device)
                    row_content = phase_global + per_req_aligned
                    any_active = True
                elif phase_global is not None:
                    row_content = phase_global
                    any_active = True
                elif per_req is not None:
                    row_content = per_req.squeeze(0)
                    any_active = True
                else:
                    row_content = zero_row
                rows.append(row_content)

            # Dynamic-override rows: composed exactly like a decode
            # per-request row — global decode effective + override
            # vectors. (Dynamic overrides are decode-only by design;
            # see docs/design/dynamic_steering.md §7.)
            for dyn_id, _row_idx in ordered_dynamic:
                dyn_vec = (
                    self._dynamic_vectors.get(dyn_id, {}).get(hp_str, {}).get(layer_idx)
                )
                if global_decode is not None and dyn_vec is not None:
                    dyn_aligned = dyn_vec.squeeze(0).to(global_decode.device)
                    row_content = global_decode + dyn_aligned
                    any_active = True
                elif global_decode is not None:
                    row_content = global_decode
                    any_active = True
                elif dyn_vec is not None:
                    row_content = dyn_vec.squeeze(0)
                    any_active = True
                else:
                    row_content = zero_row
                rows.append(row_content)

            assert len(rows) == num_rows
            per_table_rows.append(rows)
            per_table_any_active.append(any_active)

        # Stack all rows into one fp32 tensor of shape
        # ``(num_active_tables, num_rows, hidden)`` and split by dtype.
        # The vast majority of deployments use a single dtype across all
        # tables, so the dtype loop is one iteration in the common case.
        flat_rows: list[torch.Tensor] = [
            r for table_rows in per_table_rows for r in table_rows
        ]
        stacked_fp32 = torch.stack(flat_rows).reshape(
            len(active_tables), num_rows, hidden_size
        )

        # Group by dtype so we can do one cast per dtype.
        dtype_to_indices: dict[torch.dtype, list[int]] = defaultdict(list)
        for i, (table, _hp, _layer, _mod) in enumerate(active_tables):
            dtype_to_indices[table.dtype].append(i)

        for dtype, table_indices_in_active in dtype_to_indices.items():
            # One batched cast covering every table that uses this dtype.
            casted = stacked_fp32[table_indices_in_active].to(dtype=dtype)
            for casted_pos, active_pos in enumerate(table_indices_in_active):
                table = active_tables[active_pos][0]
                table.index_copy_(0, indices, casted[casted_pos])

        # Write the per-(hook, layer) any-active flags into each layer's
        # bool buffer so the apply_steering kernel can short-circuit when
        # its hook point has no non-zero rows for the current state.
        # Layers built outside ``register_steering_buffers`` (e.g. unit-
        # test fakes) may not register the flag attribute — skip them
        # gracefully so the manager remains decoupled from the buffer
        # registration pathway.
        for active_pos, (_table, hp_str, _layer_idx, mod) in enumerate(active_tables):
            try:
                hp_enum = SteeringHookPoint(hp_str)
            except ValueError:
                continue
            flag_attr = HOOK_POINT_ANY_ACTIVE_ATTR[hp_enum]
            flag_buf = getattr(mod, flag_attr, None)
            if flag_buf is None:
                continue
            flag_buf.fill_(per_table_any_active[active_pos])

        # Write each (hook, layer)'s dedicated dynamic-tier vector (§5.4)
        # into its ``dynamic_vec`` buffer (or zero it when no tier is set).
        for active_pos, (_table, hp_str, _layer_idx, mod) in enumerate(active_tables):
            try:
                hp_enum = SteeringHookPoint(hp_str)
            except ValueError:
                continue
            dvec_buf = getattr(mod, HOOK_POINT_DYNVEC_ATTR[hp_enum], None)
            if dvec_buf is None:
                continue
            tier_vec = per_table_tier_vec[active_pos]
            if tier_vec is None:
                dvec_buf.zero_()
            else:
                dvec_buf.copy_(tier_vec.squeeze(0).to(dvec_buf.device))

        # Write each (hook, layer)'s in-graph monitor config (Phase 2, §8)
        # into its probe / params / active buffers. A configured site sets
        # the probe + [threshold, sharpness] and flips ``active`` True; an
        # unconfigured site is deactivated (``active`` False ⇒ the monitor
        # op is a no-op there, leaving the runner's flat gate intact). The
        # site can move at runtime without recapture because the op is
        # emitted at every hook and gated by this tensor flag.
        for active_pos, (_table, hp_str, layer_idx, mod) in enumerate(active_tables):
            try:
                hp_enum = SteeringHookPoint(hp_str)
            except ValueError:
                continue
            active_buf = getattr(mod, HOOK_POINT_MONITOR_ACTIVE_ATTR[hp_enum], None)
            if active_buf is None:
                continue
            cfg = self.monitor_configs.get(hp_str, {}).get(layer_idx)
            if cfg is None:
                active_buf.fill_(False)
                continue
            probe_buf = getattr(mod, HOOK_POINT_MONITOR_PROBE_ATTR[hp_enum], None)
            params_buf = getattr(mod, HOOK_POINT_MONITOR_PARAMS_ATTR[hp_enum], None)
            if probe_buf is None or params_buf is None:
                active_buf.fill_(False)
                continue
            probe_buf.copy_(cfg["probe"].to(probe_buf.device))
            params_buf.copy_(
                torch.tensor(
                    [
                        cfg["threshold"],
                        cfg["sharpness"],
                        1.0 if cfg.get("gate_rows") else 0.0,
                    ],
                    dtype=torch.float32,
                    device=params_buf.device,
                )
            )
            active_buf.fill_(True)

        # Write each (hook, layer)'s PER-ROW monitor probe table + params
        # (per-request in-graph monitor). Built in row-position order and
        # scattered with the same ``indices`` as the table write; the
        # ``row_active`` flag is set iff some row at this site carries a probe.
        # Layers whose row-monitor buffers are still the ``(1, 1)`` dummies
        # (engine did not enable the row monitor, or a test fake) are skipped.
        for active_pos, (_table, hp_str, layer_idx, mod) in enumerate(active_tables):
            try:
                hp_enum = SteeringHookPoint(hp_str)
            except ValueError:
                continue
            row_active_buf = getattr(mod, HOOK_POINT_ROW_ACTIVE_ATTR[hp_enum], None)
            if row_active_buf is None:
                continue
            probe_tbl = getattr(mod, HOOK_POINT_ROW_PROBE_ATTR[hp_enum], None)
            row_params_buf = getattr(mod, HOOK_POINT_ROW_PARAMS_ATTR[hp_enum], None)
            if (
                probe_tbl is None
                or row_params_buf is None
                or probe_tbl.shape[0] != _table.shape[0]
            ):
                # Disabled (dummy buffers) — never activate the per-row path.
                row_active_buf.fill_(False)
                continue
            probe_mat, params_mat, any_cfg = self._build_row_probe_and_params(
                hp_str, layer_idx, probe_tbl.device, int(_table.shape[1])
            )
            if not any_cfg:
                row_active_buf.fill_(False)
                continue
            probe_tbl.index_copy_(0, indices, probe_mat.to(probe_tbl.dtype))
            row_params_buf.index_copy_(0, indices, params_mat.to(row_params_buf.dtype))
            row_active_buf.fill_(True)

        # Write the per-row strength scales (§5.3) alongside the tables.
        # The scale vector is hook/layer-independent, so build it once and
        # scatter into each distinct layer's ``steering_scales`` buffer with
        # the same ``indices`` used for the table write.
        scales_written: set[int] = set()
        for _table, _hp_str, _layer_idx, mod in active_tables:
            scales_buf = getattr(mod, "steering_scales", None)
            if scales_buf is None or id(scales_buf) in scales_written:
                continue
            scales_vec = self._build_scales_vector(scales_buf.device)
            scales_buf.index_copy_(0, indices, scales_vec)
            scales_written.add(id(scales_buf))

        # All per-layer table buffers now reflect current state. Subsequent
        # calls can be skipped by the caller until a mutator sets dirty again.
        # A full populate writes scales alongside the tables, so it clears
        # every dirty flag (membership was cleared in the indices-rebuild block
        # above; clearing it again is a no-op).
        self._dirty.clear_after_full_populate()

    @property
    def num_active_configs(self) -> int:
        """Number of currently active per-request steering configs."""
        return len(self.config_to_row)
