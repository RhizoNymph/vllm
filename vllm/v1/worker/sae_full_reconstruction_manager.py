# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request manager for SAE full-reconstruction state.

Sibling of :class:`vllm.v1.worker.sae_clamp_manager.SAEClampManager`,
specialised for the *replacement* (Phase-4) variant.  The contract is
structurally identical to the delta manager — strict capacity, refcount,
deterministic-replay across ranks, no global tier — but it stores
:class:`SAEFullReconstructionSpec` tuples whose ``clamps`` field may be
empty (pure reconstruction is a meaningful op for this kind).

Layout invariants:

* Row 0 is the no-reconstruction sentinel.  A token whose
  ``sae_recon_index`` selects row 0 passes through unchanged because
  :func:`apply_layer_sae_full_reconstruction` derives
  ``recon_mask = (recon_index != 0)``.
* Rows ``1..max_recon_configs`` are per-(spec_hash, phase) configurations.

Determinism: every rank executes identical register / release calls
through ``collective_rpc`` and observes identical scheduler output, so
every rank derives identical ``config_to_row`` mappings.  Row IDs flow
through ``sae_recon_index`` into the per-token gather on every rank, so
they MUST match.

This module owns row allocation only.  Buffer materialisation (writing
clamp tables and the ``sae_recon_index`` tensor) lives alongside the
per-layer buffer module so the manager stays pure-Python and trivially
testable without GPU buffers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

from vllm.config.sae_steering_types import SAEFullReconstructionSpec
from vllm.logger import init_logger

logger = init_logger(__name__)


class SAEFullReconstructionManager:
    """Per-request SAE full-reconstruction config manager.

    Maintains a mapping from ``(spec_hash, phase)`` to a row index in
    the per-(layer, hook) clamp tables, with refcounting so multiple
    requests carrying the same full-reconstruction spec share a row.

    Args:
        max_recon_configs: maximum number of distinct ``(hash, phase)``
            configs admitted concurrently.  The scheduler reserves a
            row before dispatching a request; registering past
            capacity raises ``RuntimeError``.

    Layout:
        Row 0: no-reconstruction sentinel (clamp tables are zeros and
            ``recon_index == 0`` short-circuits the layer dispatch).
        Rows 1..max_recon_configs: per-request configurations.
    """

    def __init__(self, max_recon_configs: int) -> None:
        if max_recon_configs < 0:
            raise ValueError(
                f"max_recon_configs must be non-negative; got {max_recon_configs!r}."
            )
        self.max_recon_configs = max_recon_configs
        self.config_to_row: dict[tuple[int, str], int] = {}
        self.config_specs: dict[
            tuple[int, str], tuple[SAEFullReconstructionSpec, ...]
        ] = {}
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Reversed so pop() returns the lowest free row — symmetric and
        # deterministic across ranks.
        self.free_rows: list[int] = list(range(max_recon_configs, 0, -1))
        self._tables_dirty: bool = True

    def register_recon_spec(
        self,
        config_hash: int,
        specs: tuple[SAEFullReconstructionSpec, ...],
        phase: str,
    ) -> int:
        """Register a request's full-reconstruction spec tuple, return its row.

        If ``(config_hash, phase)`` is already registered, increments
        the refcount and returns the existing row.  Otherwise allocates
        a new row.  The same hash with a different phase gets its own
        independent row, mirroring the delta manager.

        Args:
            config_hash: deterministic hash from
                :func:`hash_sae_full_reconstruction_specs` for the
                full-recon state of a request in this phase.  Hash 0
                is reserved for "no full-reconstruction in this phase"
                and must not be passed here.
            specs: the request's ``sae_full_reconstruction_specs``
                tuple — one entry per referenced full-reconstruction
                module.  Stored verbatim so the buffer populator can
                resolve ``feature_idx → position`` per module against
                each module's ``clampable_features``.  Empty tuple is
                rejected.
            phase: ``"prefill"`` or ``"decode"`` — the worker's phase.

        Returns:
            Row index in ``[1, max_recon_configs]``.

        Raises:
            ValueError: if ``config_hash == 0``, ``phase`` is invalid,
                or ``specs`` is empty.
            RuntimeError: if no free rows are available.  The scheduler
                must reserve capacity before dispatch; reaching this
                branch is a scheduler bug.
        """
        if config_hash == 0:
            raise ValueError(
                "config_hash 0 is reserved for the no-reconstruction "
                "sentinel and must not be registered.  Callers should "
                "short-circuit on hash 0 before calling register_recon_spec."
            )
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode'; got {phase!r}.")
        if not specs:
            raise ValueError(
                "register_recon_spec called with empty specs tuple; the "
                "caller must short-circuit on the no-recon case."
            )
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            return self.config_to_row[key]
        if not self.free_rows:
            raise RuntimeError(
                "No free SAE full-reconstruction table rows.  "
                f"max_recon_configs={self.max_recon_configs}, active "
                f"configs={len(self.config_to_row)}.  The scheduler "
                "must reserve capacity before dispatching full-"
                "reconstruction requests."
            )
        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_specs[key] = tuple(specs)
        self.config_refcounts[key] = 1
        self._tables_dirty = True
        return row

    def release_recon_spec(self, config_hash: int, phase: str) -> None:
        """Decrement refcount; free the row when it reaches zero.

        Releasing an unregistered ``(hash, phase)`` is a silent no-op
        so the worker mixin's request-completion path can call it
        unconditionally.
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
            self._tables_dirty = True

    def get_row_for_config(self, config_hash: int, is_prefill: bool) -> int:
        """Return the row index for a config in the given worker phase.

        ``config_hash == 0`` always returns row 0.  Otherwise looks up
        by ``(hash, phase)``; raises ``RuntimeError`` if the entry is
        missing — a scheduler-bug signal, identical fail-loud
        behaviour to the delta manager.
        """
        if config_hash == 0:
            return 0
        phase = "prefill" if is_prefill else "decode"
        row = self.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        raise RuntimeError(
            f"SAE full-reconstruction config (hash={config_hash}, "
            f"phase={phase}) is not registered.  The scheduler must "
            "guarantee capacity before dispatching a request that uses "
            "full-reconstruction; reaching this branch is a scheduler bug."
        )

    def active_rows(
        self,
    ) -> Iterator[tuple[int, int, str, tuple[SAEFullReconstructionSpec, ...]]]:
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
