# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process run-id-keyed store of clean-run activations for patching.

Activation patching overwrites a destination run's activation at a site with a
vector captured from a prior *clean* run. Those clean vectors are produced once
(via the :class:`PatchSourceConsumer`, which rides the normal capture pipeline)
and parked here under a client-chosen **run handle**, then referenced cheaply
across an entire coarse->fine sweep.

This is deliberately NOT the content-addressed :class:`ActivationStore`:

- **Key is identity, not content.** ``(run_id, layer, hook, position)`` — a
  patch request names the exact clean run + site it wants.
- **Authoritative, not a cache.** A miss is a user-visible error (the sweep
  references a run that was never captured or has been evicted), never a silent
  fall-back to re-forwarding.
- **Whole-run eviction.** Evicting individual rows of a live run would silently
  corrupt a sweep, so the LRU unit is the whole run.

Like :class:`ActivationStore` it is CPU-resident, bounded, thread-safe (the
capture dispatch thread writes; the runner resolution path reads), and
invalidated wholesale on a weight update. Under PP each pipeline rank owns a
store for its local layers; under TP only rank 0 runs the capture manager, so
only rank 0 populates a store (the runner broadcasts resolved vectors to TP
peers at resolution time, since the residual is replicated across the group).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ``(layer_idx, hook_name, position)`` within a run.
SourceRowKey = tuple[int, str, int]


@dataclass(frozen=True)
class PatchSourceManifest:
    """Lightweight description of a stored run, safe to ship to the entrypoint
    for existence/shape validation without moving any vectors."""

    run_id: str
    num_prompt_tokens: int
    hidden_size: int
    dtype: str
    hook_layers: list[tuple[str, int]]
    positions: list[int]
    resident_bytes: int


@dataclass(frozen=True)
class PatchSourceStoreStats:
    """Snapshot of store counters (lifetime monotonic + point-in-time gauges)."""

    run_hits: int = 0
    run_misses: int = 0
    rows_put: int = 0
    runs_evicted: int = 0
    invalidations: int = 0
    runs: int = 0
    resident_bytes: int = 0
    max_bytes: int = 0


@dataclass
class _SourceRun:
    run_id: str
    rows: dict[SourceRowKey, torch.Tensor] = field(default_factory=dict)
    resident_bytes: int = 0
    num_prompt_tokens: int = 0
    hidden_size: int = 0
    dtype: torch.dtype | None = None


class PatchSourceStore:
    """Bounded, whole-run-LRU store of clean-run CPU activation rows.

    Thread-safe via a single internal mutex so the capture dispatch thread
    (writes) and the runner resolution path (reads) can share one instance.
    """

    def __init__(self, max_bytes: int, max_runs: int | None = None) -> None:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        if max_runs is not None and max_runs <= 0:
            raise ValueError(f"max_runs must be positive or None, got {max_runs}")
        self._max_bytes = max_bytes
        self._max_runs = max_runs
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, _SourceRun] = OrderedDict()
        self._resident_bytes = 0
        self._run_hits = 0
        self._run_misses = 0
        self._rows_put = 0
        self._runs_evicted = 0
        self._invalidations = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @staticmethod
    def _row_bytes(row: torch.Tensor) -> int:
        return row.numel() * row.element_size()

    def put_row(
        self,
        run_id: str,
        layer: int,
        hook: str,
        position: int,
        row: torch.Tensor,
        *,
        num_prompt_tokens: int,
    ) -> None:
        """Store one clean-run activation row under ``run_id``.

        ``row`` is a 1-D ``(hidden,)`` tensor; it is detached, moved to CPU, and
        cloned so the caller's buffer can be recycled. Writing into a run moves
        it to most-recently-used; whole runs are then evicted from the LRU front
        until under budget.
        """
        row = row.detach().to("cpu").reshape(-1).clone()
        key: SourceRowKey = (int(layer), str(hook), int(position))
        nbytes = self._row_bytes(row)
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                run = _SourceRun(run_id=run_id)
                self._runs[run_id] = run
            else:
                self._runs.move_to_end(run_id)
            prev = run.rows.get(key)
            delta = nbytes - (self._row_bytes(prev) if prev is not None else 0)
            run.rows[key] = row
            run.resident_bytes += delta
            run.num_prompt_tokens = max(run.num_prompt_tokens, int(num_prompt_tokens))
            run.hidden_size = row.shape[0]
            run.dtype = row.dtype
            self._resident_bytes += delta
            self._rows_put += 1
            self._evict_to_budget_locked(protect=run_id)

    def _evict_to_budget_locked(self, *, protect: str | None = None) -> None:
        over_bytes = self._max_bytes > 0 and self._resident_bytes > self._max_bytes
        over_runs = self._max_runs is not None and len(self._runs) > self._max_runs
        while (over_bytes or over_runs) and len(self._runs) > 0:
            run_id, run = next(iter(self._runs.items()))
            if run_id == protect:
                # Never evict the run being written; if it alone exceeds the
                # budget we keep it (and log) rather than corrupt it.
                if len(self._runs) == 1:
                    logger.warning(
                        "patch source run %s (%.3f GB) exceeds budget %.3f GB; "
                        "kept (single run)",
                        run_id,
                        run.resident_bytes / 1e9,
                        self._max_bytes / 1e9,
                    )
                    return
                # Rotate the protected run to the back and evict the next LRU.
                self._runs.move_to_end(run_id)
                run_id, run = next(iter(self._runs.items()))
            self._runs.pop(run_id, None)
            self._resident_bytes -= run.resident_bytes
            self._runs_evicted += 1
            logger.info(
                "evicted patch source run %s (%.3f GB)",
                run_id,
                run.resident_bytes / 1e9,
            )
            over_bytes = self._max_bytes > 0 and self._resident_bytes > self._max_bytes
            over_runs = self._max_runs is not None and len(self._runs) > self._max_runs

    def get_row(
        self, run_id: str, layer: int, hook: str, position: int
    ) -> torch.Tensor | None:
        """Return a cloned row for one site, or ``None`` on miss (LRU-touch)."""
        key: SourceRowKey = (int(layer), str(hook), int(position))
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                self._run_misses += 1
                return None
            row = run.rows.get(key)
            if row is None:
                self._run_misses += 1
                return None
            self._runs.move_to_end(run_id)
            self._run_hits += 1
            return row.clone()

    def get_rows(
        self, run_id: str, requests: list[SourceRowKey]
    ) -> list[torch.Tensor] | None:
        """All-or-nothing batch read; returns cloned rows in request order.

        Returns ``None`` if the run is unknown or any requested site is missing
        (an authoritative miss the caller should surface as an error).
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                self._run_misses += 1
                return None
            out: list[torch.Tensor] = []
            for key in requests:
                row = run.rows.get(
                    (int(key[0]), str(key[1]), int(key[2]))
                )
                if row is None:
                    self._run_misses += 1
                    return None
                out.append(row)
            self._runs.move_to_end(run_id)
            self._run_hits += 1
            return [r.clone() for r in out]

    def has_run(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._runs

    def manifest(self, run_id: str) -> PatchSourceManifest | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return self._manifest_locked(run)

    def manifests(self) -> list[PatchSourceManifest]:
        with self._lock:
            return [self._manifest_locked(run) for run in self._runs.values()]

    @staticmethod
    def _manifest_locked(run: _SourceRun) -> PatchSourceManifest:
        hook_layers = sorted({(hook, layer) for (layer, hook, _pos) in run.rows})
        positions = sorted({pos for (_layer, _hook, pos) in run.rows})
        return PatchSourceManifest(
            run_id=run.run_id,
            num_prompt_tokens=run.num_prompt_tokens,
            hidden_size=run.hidden_size,
            dtype=str(run.dtype) if run.dtype is not None else "",
            hook_layers=hook_layers,
            positions=positions,
            resident_bytes=run.resident_bytes,
        )

    def drop_run(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.pop(run_id, None)
            if run is None:
                return False
            self._resident_bytes -= run.resident_bytes
            return True

    def invalidate_all(self) -> None:
        """Drop every run (e.g. on a weight update — clean vectors are stale)."""
        with self._lock:
            self._runs.clear()
            self._resident_bytes = 0
            self._invalidations += 1

    def stats(self) -> PatchSourceStoreStats:
        with self._lock:
            return PatchSourceStoreStats(
                run_hits=self._run_hits,
                run_misses=self._run_misses,
                rows_put=self._rows_put,
                runs_evicted=self._runs_evicted,
                invalidations=self._invalidations,
                runs=len(self._runs),
                resident_bytes=self._resident_bytes,
                max_bytes=self._max_bytes,
            )


# Process-global active store, mirroring the activation-store accessor pair.
# Installed on the capture-manager rank in the runner's capture init.
_ACTIVE_PATCH_SOURCE_STORE: PatchSourceStore | None = None


def set_active_patch_source_store(store: PatchSourceStore | None) -> None:
    global _ACTIVE_PATCH_SOURCE_STORE
    _ACTIVE_PATCH_SOURCE_STORE = store


def get_active_patch_source_store() -> PatchSourceStore | None:
    return _ACTIVE_PATCH_SOURCE_STORE
