# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request SAE-clamp state manager (shared-nothing, deterministic replay).

Parallel to :class:`vllm.v1.worker.steering_manager.SteeringManager`, but
for the SAE feature-surgery path: where the additive manager allocates
rows for precomputed steering vectors, this manager allocates rows for
:class:`SAEClampSpec` directives and stores the spec so the per-layer
buffer populator can re-derive each row's clamp content for whichever
``(layer, hook)`` site the spec covers.

The contract mirrors the additive manager's, with one structural
difference: there is **no global tier**.  The SAE design has no
"global SAE clamp" — every clamp is per-request.  Concretely:

* Row 0 is the no-op sentinel (all clamp_kind = NONE; encoder pass
  produces zero delta).  Used for tokens that don't carry a clamp.
* Rows ``1..max_sae_configs`` are per-(spec_hash, phase) configurations.

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

from vllm.config.sae_steering_types import SAEClampSpec
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
        Row 0: no-op sentinel (filled with ``CLAMP_KIND_NONE``)
        Rows 1..max_sae_configs: per-request configurations
    """

    def __init__(self, max_sae_configs: int) -> None:
        if max_sae_configs < 0:
            raise ValueError(
                f"max_sae_configs must be non-negative; got {max_sae_configs!r}."
            )
        self.max_sae_configs = max_sae_configs
        # (config_hash, phase) -> assigned row in [1, max_sae_configs].
        self.config_to_row: dict[tuple[int, str], int] = {}
        # (config_hash, phase) -> tuple of specs, retained so the buffer
        # populator can re-derive row content per-(layer, hook) site.
        # A request's ``SamplingParams.sae_clamp_specs`` is a tuple
        # (one entry per referenced SAE module); the whole tuple shares
        # one row because the admission hash already covers all of them.
        self.config_specs: dict[tuple[int, str], tuple[SAEClampSpec, ...]] = {}
        # (config_hash, phase) -> active reference count.
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Reversed so pop() returns the lowest free row — symmetric and
        # deterministic across ranks.
        self.free_rows: list[int] = list(range(max_sae_configs, 0, -1))

        # Set by every state mutator that changes row content (register
        # new-row path, release refcount->0 path); cleared by the
        # populator via :meth:`mark_tables_clean`.  Initialized True so
        # the first populate call always runs.
        self._tables_dirty: bool = True

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
            config_hash: deterministic hash from
                :func:`hash_sae_clamp_specs` for the SAE state of a
                request in this phase.  Hash 0 is reserved for "no SAE
                clamps in this phase" and must not be passed here.
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
            Row index in ``[1, max_sae_configs]``.

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
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            return self.config_to_row[key]

        if not self.free_rows:
            raise RuntimeError(
                "No free SAE clamp table rows.  max_sae_configs="
                f"{self.max_sae_configs}, active configs="
                f"{len(self.config_to_row)}.  The scheduler must reserve "
                "capacity before dispatching SAE-clamp requests."
            )

        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_specs[key] = tuple(specs)
        self.config_refcounts[key] = 1
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
        if self.config_refcounts[key] <= 0:
            row = self.config_to_row.pop(key)
            self.config_specs.pop(key, None)
            del self.config_refcounts[key]
            self.free_rows.append(row)
            # Stale row content; mark dirty so the next populate
            # overwrites it before any subsequent request lands here.
            self._tables_dirty = True

    def get_row_for_config(self, config_hash: int, is_prefill: bool) -> int:
        """Return the row index for a config in the given worker phase.

        ``config_hash == 0`` always returns row 0 (the no-op sentinel).
        Otherwise looks up by ``(hash, phase)``; raises
        ``RuntimeError`` if the entry is missing — a scheduler-bug
        signal, identical fail-loud behaviour to the additive manager.
        """
        if config_hash == 0:
            return 0
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

    def active_rows(
        self,
    ) -> Iterator[tuple[int, int, str, tuple[SAEClampSpec, ...]]]:
        """Yield ``(row, config_hash, phase, specs)`` for every active row.

        Ordered by row index so the populator writes rows in a
        deterministic order.  Used by the per-layer buffer populator
        to re-derive clamp content per-(layer, hook) site.
        """
        ordered = sorted(self.config_to_row.items(), key=lambda kv: kv[1])
        for (config_hash, phase), row in ordered:
            yield row, config_hash, phase, self.config_specs[(config_hash, phase)]

    def mark_tables_clean(self) -> None:
        """Clear the dirty flag.  Called by the populator after flushing."""
        self._tables_dirty = False
