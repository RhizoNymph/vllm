# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SAE-clamp state manager (shared-nothing, deterministic replay).

Parallel to :class:`vllm.v1.worker.steering_manager.SteeringManager`, but
for the SAE feature-surgery path: where the additive manager allocates
rows for precomputed steering vectors, this manager allocates rows for
:class:`SAEClampSpec` directives and stores the spec so the per-layer
buffer populator can re-derive each row's clamp content for whichever
``(layer, hook)`` site the spec covers.

The contract mirrors the additive manager's table layout:

* Row 0 is the no-op sentinel (all clamp_kind = NONE; encoder pass
  produces zero delta).
* Row 1 is the global prefill tier.
* Row 2 is the global decode tier.
* Rows ``3..max_sae_configs + 2`` are per-(spec_hash, phase)
  configurations.

Determinism contract is identical to the additive manager: every rank
executes identical register/release calls (via ``collective_rpc``)
and sees identical scheduler output, so every rank derives identical
``config_to_row`` mappings.  Row IDs flow through ``sae_index`` into
the per-token gather on every rank, so they MUST match.

This module owns row allocation only.  Buffer materialization (writing
clamp tables and ``sae_index`` tensors) lives alongside the per-layer
buffer module so the manager stays pure-Python and trivially testable
without GPU buffers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from vllm.config.sae_steering_types import (
    SAEClampSpec,
    hash_sae_clamp_specs_for_phase,
    validate_sae_clamp_specs_no_overlap,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SAEClampManager:
    """Per-request SAE clamp config manager.

    Maintains a mapping from (spec_hash, phase) to a row index in the
    per-(layer, hook) clamp tables, with refcounting so multiple
    requests carrying the same clamp spec share a row.

    Args:
        max_sae_configs: maximum number of distinct ``(hash, phase)``
            configs admitted concurrently.  The scheduler must reserve
            a row before dispatching a request that uses SAE clamps;
            registering past capacity raises ``RuntimeError``.

    Layout:
        Row 0: no-op sentinel.
        Row 1: global prefill tier.
        Row 2: global decode tier.
        Rows 3..max_sae_configs + 2: per-request configurations.  The
            populator merges global clamps into each per-request row
            so a request that opts into per-request clamps still gets
            the globals stacked on top (any feature_idx collision
            between a global clamp and a per-request clamp is rejected
            at admission time by
            :func:`validate_sae_clamp_specs_no_overlap`).
    """

    def __init__(self, max_sae_configs: int) -> None:
        if max_sae_configs < 0:
            raise ValueError(
                f"max_sae_configs must be non-negative; got {max_sae_configs!r}."
            )
        self.max_sae_configs = max_sae_configs
        # (sae_config_hash, phase) -> assigned row in
        # [3, max_sae_configs + 2].
        # Worker code passes the phase-specific SAE-only content hash, so
        # identical clamp tables share capacity even when additive steering
        # differs.
        self.config_to_row: dict[tuple[int, str], int] = {}
        # (sae_config_hash, phase) -> tuple of specs for diagnostics/tests.
        self.config_specs: dict[tuple[int, str], tuple[SAEClampSpec, ...]] = {}
        # (sae_config_hash, phase) -> active reference count.
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Alias from external config hash to content hash.  In production
        # these hashes are the same; tests may pass arbitrary config hashes
        # and still exercise content-based row sharing.
        self._config_to_content: dict[tuple[int, str], tuple[int, str]] = {}
        self._content_to_row: dict[tuple[int, str], int] = {}
        self._content_specs: dict[tuple[int, str], tuple[SAEClampSpec, ...]] = {}
        self._content_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Reversed so pop() returns the lowest free per-request row —
        # symmetric and deterministic across ranks.
        self.free_rows: list[int] = list(range(max_sae_configs + 2, 2, -1))

        # Global SAE clamp tier.  Per-phase tuples of
        # :class:`SAEClampSpec`; the populator materializes these into
        # rows 1 and 2 for no-per-request tokens, and stacks the
        # matching phase's globals into per-request rows.
        #
        # Phase-keyed (rather than a single "base" tier) because a
        # global clamp can legitimately want decode-only or prefill-
        # only semantics — analogous to the additive path's
        # ``global_prefill_vectors`` / ``global_decode_vectors``.
        self.global_prefill_specs: tuple[SAEClampSpec, ...] = ()
        self.global_decode_specs: tuple[SAEClampSpec, ...] = ()

        # Set by every state mutator that changes row content (register
        # new-row path, release refcount->0 path, global-clamp update);
        # cleared by the populator via :meth:`mark_tables_clean`.
        # Initialized True so the first populate call always runs.
        self._tables_dirty: bool = True

    def _global_specs_for_phase_unchecked(
        self, phase: str
    ) -> tuple[SAEClampSpec, ...]:
        if phase == "prefill":
            return self.global_prefill_specs
        if phase == "decode":
            return self.global_decode_specs
        raise ValueError(f"phase must be 'prefill' or 'decode'; got {phase!r}.")

    def _validate_specs_against_globals(
        self,
        specs: tuple[SAEClampSpec, ...],
        phase: str,
        *,
        global_specs: tuple[SAEClampSpec, ...] | None = None,
    ) -> None:
        global_specs = (
            self._global_specs_for_phase_unchecked(phase)
            if global_specs is None
            else global_specs
        )
        if global_specs:
            validate_sae_clamp_specs_no_overlap(global_specs + specs)

    def _validate_globals_against_active_rows(
        self,
        global_specs: tuple[SAEClampSpec, ...],
        phase: str,
    ) -> None:
        if not global_specs:
            return
        for (content_hash, content_phase), specs in self._content_specs.items():
            if content_phase != phase:
                continue
            validate_sae_clamp_specs_no_overlap(global_specs + specs)

    def register_clamp_spec(
        self,
        config_hash: int,
        specs: tuple[SAEClampSpec, ...],
        phase: str,
    ) -> int:
        """Register a request's full SAE clamp tuple, return its row index.

        If ``(config_hash, phase)`` is already registered, increments
        the refcount and returns the existing row.  Otherwise assigns
        a new row.  The same hash with a different phase gets its own
        independent row, mirroring the additive manager.

        Args:
            config_hash: deterministic SAE-only hash for the request in
                this phase.  Hash 0 is reserved for "no SAE clamps in
                this phase" and must not be passed here.
            specs: the request's ``sae_clamp_specs`` tuple — one entry
                per referenced SAE module.  Stored verbatim so the
                buffer populator can resolve (feature_idx → position)
                per module against each module's
                ``clampable_features`` and write clamp values into
                the row.  An empty tuple is rejected; callers must
                short-circuit before calling.
            phase: ``"prefill"`` or ``"decode"`` — the worker's phase,
                not the specs' ``phase`` field (which gates whether a
                given spec applies in a given worker phase).

        Returns:
            Row index in ``[3, max_sae_configs + 2]``.

        Raises:
            ValueError: if ``config_hash == 0``, ``phase`` is invalid,
                or ``specs`` is empty.
            RuntimeError: if no free rows are available.  The scheduler
                is expected to reserve capacity before dispatch; this
                branch indicates a scheduler bug.
        """
        if config_hash == 0:
            raise ValueError(
                "config_hash 0 is reserved for the no-op sentinel and must "
                "not be registered.  Callers should short-circuit on hash "
                "0 before calling register_clamp_spec."
            )
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode'; got {phase!r}.")
        if not specs:
            raise ValueError(
                "register_clamp_spec called with empty specs tuple; the "
                "caller must short-circuit on the no-clamps case."
            )
        specs = tuple(specs)
        validate_sae_clamp_specs_no_overlap(specs)
        self._validate_specs_against_globals(specs, phase)
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            content_key = self._config_to_content[key]
            self._content_refcounts[content_key] += 1
            return self.config_to_row[key]

        content_hash = hash_sae_clamp_specs_for_phase(specs, phase)  # type: ignore[arg-type]
        if content_hash == 0:
            raise ValueError(
                "register_clamp_spec called with specs that do not apply "
                f"to phase {phase!r}."
            )
        content_key = (content_hash, phase)
        if content_key in self._content_to_row:
            row = self._content_to_row[content_key]
            self.config_to_row[key] = row
            self.config_specs[key] = specs
            self.config_refcounts[key] = 1
            self._config_to_content[key] = content_key
            self._content_refcounts[content_key] += 1
            return row

        if not self.free_rows:
            raise RuntimeError(
                "No free SAE clamp table rows.  max_sae_configs="
                f"{self.max_sae_configs}, active configs="
                f"{len(self.config_to_row)}.  The scheduler must reserve "
                "capacity before dispatching SAE-clamp requests."
            )

        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_specs[key] = specs
        self.config_refcounts[key] = 1
        self._config_to_content[key] = content_key
        self._content_to_row[content_key] = row
        self._content_specs[content_key] = specs
        self._content_refcounts[content_key] = 1
        self._tables_dirty = True
        return row

    def release_clamp_spec(self, config_hash: int, phase: str) -> None:
        """Decrement refcount for ``(config_hash, phase)``.

        Frees the row when it reaches zero.  Releasing an unregistered
        ``(hash, phase)`` is a silent no-op so the worker mixin's
        request-completion path can call it unconditionally without
        first checking whether the request actually used SAE clamps.
        """
        key = (config_hash, phase)
        if key not in self.config_to_row:
            return
        self.config_refcounts[key] -= 1
        content_key = self._config_to_content[key]
        self._content_refcounts[content_key] -= 1
        if self.config_refcounts[key] <= 0:
            self.config_to_row.pop(key)
            self.config_specs.pop(key, None)
            del self.config_refcounts[key]
            self._config_to_content.pop(key, None)
        if self._content_refcounts[content_key] <= 0:
            row = self._content_to_row.pop(content_key)
            self._content_specs.pop(content_key, None)
            del self._content_refcounts[content_key]
            self.free_rows.append(row)
            # Stale row content; mark dirty so the next populate
            # overwrites it before any subsequent request lands here.
            self._tables_dirty = True

    def get_row_for_config(self, config_hash: int, is_prefill: bool) -> int:
        """Return the row index for a config in the given worker phase.

        ``config_hash == 0`` returns the phase-specific global row:
        row 1 for prefill and row 2 for decode.  When no globals are
        configured for that phase, the row remains zero and is a
        true no-op for tokens with no per-request SAE state.

        For registered per-request configs, returns the assigned row
        (``3..max_sae_configs + 2``), looked up by
        ``(config_hash, "prefill"/"decode")``.  Raises
        ``RuntimeError`` for unregistered nonzero hashes — a
        scheduler-bug signal, identical fail-loud behaviour to the
        additive manager.
        """
        if config_hash == 0:
            return 1 if is_prefill else 2
        phase = "prefill" if is_prefill else "decode"
        row = self.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        raise RuntimeError(
            f"SAE clamp config (hash={config_hash}, phase={phase}) is "
            "not registered.  The scheduler must guarantee capacity "
            "before dispatching a request that uses SAE clamps; "
            "reaching this branch is a scheduler bug."
        )

    def set_global_clamps(
        self,
        prefill_specs: tuple[SAEClampSpec, ...] | None = None,
        decode_specs: tuple[SAEClampSpec, ...] | None = None,
        *,
        replace: bool = False,
    ) -> None:
        """Configure global SAE clamps applied to every token in a phase.

        Mirrors :meth:`SteeringManager.update_global_vectors` for the
        SAE delta path.  When ``replace`` is True, existing state is
        cleared before applying — used by the API server's
        ``POST /v1/steering/sae/set`` to atomically install a new
        global configuration.  When False, the new specs are appended
        to the existing tier (subject to the standard no-overlap
        check on ``validate_sae_clamp_specs_no_overlap`` per phase).

        Empty tuples are tolerated: passing both ``None`` and
        ``replace=False`` is a no-op; passing an empty tuple with
        ``replace=True`` clears that phase.  This lets the API surface
        the natural "set decode-only globals, leave prefill globals
        alone" pattern.

        Args:
            prefill_specs: tuple of :class:`SAEClampSpec` to apply to
                the global prefill tier.  ``None`` leaves the existing
                state untouched unless ``replace`` is True.
            decode_specs: tuple of :class:`SAEClampSpec` to apply to
                the global decode tier.  Same semantics as
                ``prefill_specs``.
            replace: when True, clear the existing global state before
                applying.  Atomic with the new content — partial state
                during the swap is never observable.
        """
        base_prefill = () if replace else tuple(self.global_prefill_specs)
        base_decode = () if replace else tuple(self.global_decode_specs)
        new_prefill = base_prefill
        new_decode = base_decode
        if prefill_specs is not None:
            new_prefill = base_prefill + tuple(prefill_specs)
            validate_sae_clamp_specs_no_overlap(new_prefill)
        if decode_specs is not None:
            new_decode = base_decode + tuple(decode_specs)
            validate_sae_clamp_specs_no_overlap(new_decode)
        self._validate_globals_against_active_rows(new_prefill, "prefill")
        self._validate_globals_against_active_rows(new_decode, "decode")
        self.global_prefill_specs = new_prefill
        self.global_decode_specs = new_decode
        self._tables_dirty = True

    def clear_global_clamps(self) -> None:
        """Drop all configured global SAE clamps.

        Symmetric to :meth:`SteeringManager.clear_global_vectors`.
        After this call, rows 1 and 2 will be re-zeroed on the next
        populate, restoring no-op semantics for tokens whose request
        does not carry per-request SAE clamps.
        """
        self.global_prefill_specs = ()
        self.global_decode_specs = ()
        self._tables_dirty = True

    def has_global_clamps(self) -> bool:
        """Return True iff any global SAE clamps are configured."""
        return bool(self.global_prefill_specs) or bool(self.global_decode_specs)

    def global_specs_for_phase(
        self, phase: str
    ) -> tuple[SAEClampSpec, ...]:
        """Return the global spec tuple for ``phase``.

        Used by the per-layer populator to materialise rows 1 and 2.
        """
        if phase == "prefill":
            return self.global_prefill_specs
        if phase == "decode":
            return self.global_decode_specs
        raise ValueError(f"phase must be 'prefill' or 'decode'; got {phase!r}.")

    def active_rows(
        self,
    ) -> Iterator[tuple[int, int, str, tuple[SAEClampSpec, ...]]]:
        """Yield ``(row, config_hash, phase, specs)`` for every active row.

        Ordered by row index so the populator writes rows in a
        deterministic order.  Used by the per-layer buffer populator
        to re-derive clamp content per-(layer, hook) site.
        """
        ordered = sorted(self._content_to_row.items(), key=lambda kv: kv[1])
        for (config_hash, phase), row in ordered:
            yield row, config_hash, phase, self._content_specs[(config_hash, phase)]

    def mark_tables_clean(self) -> None:
        """Clear the dirty flag.  Called by the populator after flushing."""
        self._tables_dirty = False
