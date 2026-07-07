# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a request's ``patch`` spec into source-vector :class:`PatchEntry`s.

A patch spec entry references a clean run's stored activation by
``(source_run, layer, hook, source_position)``; resolution looks each up in the
per-worker :class:`PatchSourceStore` and pairs it with the destination
``(layer, hook, dest_position, alpha)``.

Cross-rank behavior:

- **TP1 / PP**: the store lives on this rank (PP: with its local layers), so
  resolution is purely local. Only locally-owned layers are resolved.
- **TP>1**: the capture manager (hence the store) lives only on TP rank 0, but
  injection must be byte-identical on every TP rank (the residual is replicated
  across the group). Rank 0 resolves and **broadcasts** the resolved entries to
  its TP peers, who apply the identical patch. Resolution runs in lockstep on
  all ranks (``add_requests`` is driven by the broadcast scheduler output), so
  the collective is safe.

Strictness: source existence is validated at admission (the entrypoint checks
the source manifest), so a miss here means an eviction race or bug — it is
logged and the entry skipped rather than crashing the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.capture.source_store import get_active_patch_source_store
from vllm.v1.worker.patch_runner_mixin import PatchEntry, get_patchable_hook

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData

logger = init_logger(__name__)

# Process-global registry of per-request resolution failures. A miss here
# means an admitted patch could not resolve its source (eviction race or bug):
# the request runs UNPATCHED, so its output must not be trusted as a patched
# result. Leasing (PatchSourceStore.lease_runs) makes this near-impossible;
# this registry is the loud backstop — the sweep endpoint drains it after each
# sweep (collective_rpc) and voids the affected cells.
_RESOLUTION_FAILURES: dict[str, list[str]] = {}
_FAILURES_MAX = 1024  # bound the registry; oldest dropped with a log


def record_resolution_failure(req_id: str, detail: str) -> None:
    if req_id not in _RESOLUTION_FAILURES and (
        len(_RESOLUTION_FAILURES) >= _FAILURES_MAX
    ):
        dropped = next(iter(_RESOLUTION_FAILURES))
        _RESOLUTION_FAILURES.pop(dropped, None)
        logger.warning("patch resolution-failure registry full; dropped %s", dropped)
    _RESOLUTION_FAILURES.setdefault(req_id, []).append(detail)


def pop_resolution_failures() -> dict[str, list[str]]:
    """Drain the registry (worker RPC target)."""
    global _RESOLUTION_FAILURES
    failures, _RESOLUTION_FAILURES = _RESOLUTION_FAILURES, {}
    return failures


def _tp_group():
    """Return the TP GroupCoordinator, or ``None`` if unavailable (CPU/tests)."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        return get_tp_group()
    except Exception:
        return None


def _resolve_local(candidates: list[dict], req_id: str) -> list[PatchEntry]:
    """Resolve candidate spec entries against the local source store.

    Missing sources are logged, recorded in the resolution-failure registry
    (so the request's output can be voided), and skipped — admission is the
    strict gate; a miss here is an eviction race or a bug, and the engine
    must not crash for it.
    """
    store = get_active_patch_source_store()
    if store is None:
        logger.warning(
            "patch resolution: no active source store on this rank; "
            "dropping %d patch entries",
            len(candidates),
        )
        record_resolution_failure(req_id, "no active source store on rank")
        return []
    entries: list[PatchEntry] = []
    for e in candidates:
        layer = int(e["layer"])
        hook = str(e["hook"])
        source_run = str(e["source_run"])
        source_pos = int(e["source_position"])
        row = store.get_row(source_run, layer, hook, source_pos)
        if row is None:
            detail = (
                f"source missing: run={source_run} layer={layer} "
                f"hook={hook} pos={source_pos} (evicted or never captured)"
            )
            logger.error("patch resolution for %s: %s; skipping entry", req_id, detail)
            record_resolution_failure(req_id, detail)
            continue
        entries.append(
            PatchEntry(
                layer=layer,
                hook=get_patchable_hook(hook),
                dest_pos=int(e["dest_position"]),
                source=row,
                alpha=float(e.get("alpha", 1.0)),
            )
        )
    return entries


def resolve_patch_entries(
    new_req_data: NewRequestData,
    *,
    local_layers: frozenset[int],
) -> list[PatchEntry]:
    """Resolve ``new_req_data``'s patch spec into entries for local layers.

    Runs in lockstep on every TP rank; rank 0 owns the store and broadcasts the
    resolved entries to peers (no-op at TP1).
    """
    sampling_params = getattr(new_req_data, "sampling_params", None)
    if sampling_params is None:
        return []
    spec = getattr(sampling_params, "patch", None)
    if not spec:
        return []

    # Candidate entries for locally-owned layers, in deterministic spec order —
    # identical on every rank (all ranks see the same broadcast spec).
    candidates = [e for e in spec if int(e["layer"]) in local_layers]
    if not candidates:
        return []

    tp = _tp_group()
    world_size = getattr(tp, "world_size", 1) if tp is not None else 1

    if world_size <= 1:
        return _resolve_local(candidates, new_req_data.req_id)

    # TP>1: rank 0 resolves from its store and broadcasts to peers so all ranks
    # apply the identical patch (residual is replicated across the TP group).
    if tp.rank_in_group == 0:
        resolved = _resolve_local(candidates, new_req_data.req_id)
        tp.broadcast_object(resolved, src=0)
        return resolved
    return tp.broadcast_object(None, src=0) or []
