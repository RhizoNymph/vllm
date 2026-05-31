# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process content-addressed store of captured pristine residuals.

Step A of the prefix-cache/capture roadmap (see
``docs/design/capture_consumers.md`` § "Prefix-Cache Interaction"). A
prompt-touching capture normally re-forwards from its lowest captured
prompt position (step C) so that position's residual is produced. When an
earlier request sharing the same prefix already captured that residual,
this store lets the position be served from RAM instead of re-forwarded —
the only mechanism that helps repeated ``all_prompt`` capture over a fixed
corpus, which C alone cannot.

Validity rests on one fact: a *pristine* residual (read pre-steering) at a
token position is a pure function of the prefix token ids and the model
weights. Identical prefix -> identical residual. The store is therefore
content-addressed by the block-hash chain (the same hash the KV cache
uses), so a hit is exact, not approximate.

Scope and guarantees:

- **Pure cache.** A miss or eviction simply falls back to re-forwarding;
  the store is never a source of truth. Durability of captured data is the
  consumer's output, not this store.
- **CPU-RAM, bounded, LRU, drop-on-eviction.** Total resident bytes are
  capped; the least-recently-used rows are dropped to stay under budget.
- **Single process.** Capture is TP1/PP1 only, so the model runner and the
  scheduler share one process (``UniProcExecutor``); one ``ActivationStore``
  instance is shared between the scheduler (membership queries) and the
  capture manager (write-through / serve). Access is guarded by a lock
  because write-through runs on the capture dispatch thread while the
  scheduler reads on the main thread.
- **Invalidated wholesale on weight update.** Weights changing breaks the
  pure-function premise, so :meth:`invalidate_all` must be called from the
  same path that resets the prefix cache.
- **Steering.** Captures read pre-steering residuals, but steering applied
  at an earlier layer poisons a later tap. Activation steering is not yet
  implemented in this tree; when it lands, store participation must be
  gated on steering being off before the read/serve path may use it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash

# ``(block_hash, offset_in_block, layer_idx, hook_name)``. ``block_hash``
# chains the full prefix, so together with the within-block offset it
# uniquely identifies the token positions ``0..p`` that determine the
# residual. Model dtype and weight version are intentionally absent: a
# server runs one model at one dtype, and weight updates invalidate the
# whole store.
ActivationKey = tuple["BlockHash", int, int, str]


@dataclass(frozen=True)
class ActivationStoreStats:
    """Snapshot of store counters.

    ``hits`` .. ``invalidations`` are monotonic lifetime counters;
    ``entries`` / ``resident_bytes`` / ``max_bytes`` are point-in-time
    gauges.
    """

    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    skipped_too_large: int = 0
    invalidations: int = 0
    entries: int = 0
    resident_bytes: int = 0
    max_bytes: int = 0


class ActivationStore:
    """Bounded LRU cache of captured CPU residual rows, keyed by content.

    Thread-safe: every operation takes a single internal mutex, so the
    scheduler thread (membership/read) and the capture dispatch thread
    (write-through) may use one shared instance.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[ActivationKey, torch.Tensor] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._skipped_too_large = 0
        self._invalidations = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @staticmethod
    def _row_bytes(row: torch.Tensor) -> int:
        return row.numel() * row.element_size()

    def get(self, key: ActivationKey) -> torch.Tensor | None:
        """Return the stored row for ``key`` (LRU-touched), or ``None``.

        The returned tensor is the stored reference; callers must treat it
        as read-only and must not mutate it in place.
        """
        with self._lock:
            row = self._entries.get(key)
            if row is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return row

    def __contains__(self, key: ActivationKey) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def put(self, key: ActivationKey, row: torch.Tensor) -> None:
        """Insert/replace ``key`` -> ``row`` (a CPU tensor), evicting the
        least-recently-used rows to stay within the byte budget.

        A row that alone exceeds the whole budget is skipped (counted in
        ``skipped_too_large``) rather than thrashing the store empty; any
        stale entry under ``key`` is dropped so a later read cannot return
        a wrong residual.
        """
        if row.is_cuda:
            raise ValueError("ActivationStore rows must be CPU tensors")
        row_bytes = self._row_bytes(row)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._resident_bytes -= self._row_bytes(old)
            if row_bytes > self._max_bytes:
                # Even on its own it would not fit; refuse rather than evict
                # everything else to make room for a row that cannot stay.
                self._skipped_too_large += 1
                return
            self._entries[key] = row
            self._resident_bytes += row_bytes
            self._puts += 1
            self._evict_to_budget_locked()

    def _evict_to_budget_locked(self) -> None:
        while self._resident_bytes > self._max_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._resident_bytes -= self._row_bytes(evicted)
            self._evictions += 1

    def invalidate_all(self) -> None:
        """Drop every entry. Must be called when model weights change."""
        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0
            self._invalidations += 1

    def stats(self) -> ActivationStoreStats:
        with self._lock:
            return ActivationStoreStats(
                hits=self._hits,
                misses=self._misses,
                puts=self._puts,
                evictions=self._evictions,
                skipped_too_large=self._skipped_too_large,
                invalidations=self._invalidations,
                entries=len(self._entries),
                resident_bytes=self._resident_bytes,
                max_bytes=self._max_bytes,
            )


# ---------------------------------------------------------------------------
# Process-global active store
# ---------------------------------------------------------------------------
#
# Set once per process when capture is enabled with a non-zero activation
# cache budget. ``None`` means the store is disabled and the read/serve
# path must fall back to re-forwarding (step C).

_ACTIVE_ACTIVATION_STORE: ActivationStore | None = None


def set_active_activation_store(store: ActivationStore | None) -> None:
    """Install ``store`` as the process-global active activation store."""
    global _ACTIVE_ACTIVATION_STORE
    _ACTIVE_ACTIVATION_STORE = store


def get_active_activation_store() -> ActivationStore | None:
    """Return the currently installed activation store, if any."""
    return _ACTIVE_ACTIVATION_STORE


__all__ = [
    "ActivationKey",
    "ActivationStore",
    "ActivationStoreStats",
    "get_active_activation_store",
    "set_active_activation_store",
]
