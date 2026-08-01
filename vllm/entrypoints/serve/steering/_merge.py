# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for merging per-worker steering RPC results in the router.

Under the steering determinism contract (see
``vllm/v1/worker/steering_manager.py``), TP ranks within the same PP
stage own identical layer sets and report identical state; PP ranks
report disjoint layer sets. The helpers here exploit that invariant
and raise ``RuntimeError`` on divergence so the router can surface a
server-side invariant violation (HTTP 500) rather than silently
dropping conflicting entries.
"""

import regex as re

_RANK_PREFIX = re.compile(r"^Rank\s+\d+:\s*")


def deep_merge_status(
    per_worker: list[dict],
) -> dict:
    """Merge ``get_steering_status`` payloads across workers.

    Input shape per worker:
    ``{layer_idx: {hook_point: {norm_key: float}}}``.

    Policy: for each ``(layer_idx, hook_point, norm_key)`` triple we
    keep the first non-empty value seen; if a later worker reports
    the same triple with a different value we raise ``RuntimeError``.
    Under the determinism contract duplicates are expected only
    across TP ranks and must be identical, so a mismatch indicates a
    server-side bug (e.g. a rank loaded the wrong weights).
    """
    merged: dict[int, dict[str, dict[str, float]]] = {}
    for worker_result in per_worker:
        if not worker_result:
            continue
        for layer_idx, hooks in worker_result.items():
            dest_hooks = merged.setdefault(layer_idx, {})
            for hook_point, norms in hooks.items():
                dest_norms = dest_hooks.setdefault(hook_point, {})
                for norm_key, value in norms.items():
                    if norm_key in dest_norms:
                        if dest_norms[norm_key] != value:
                            raise RuntimeError(
                                "Steering status divergence at "
                                f"layer={layer_idx} hook={hook_point} "
                                f"key={norm_key}: "
                                f"{dest_norms[norm_key]} != {value}"
                            )
                    else:
                        dest_norms[norm_key] = value
    return merged


def check_action_determinism(per_worker: list[dict]) -> dict:
    """Cross-rank determinism check of applied dynamic steering actions.

    Each worker reports an ``action_checksum`` (rolling u64 hex of every
    applied action, folded in order) and an ``action_count``. Under the
    determinism contract, sync capture consumers run identically and
    independently on every rank in lock-step; the checksum makes a silent
    per-rank desync (a swallowed consumer fault that drifts one rank's
    dyn-id allocation forever) observable at poll time.

    Topology scoping (mirrors :func:`deep_merge_status`): TP ranks within
    a PP stage own identical layers and apply an identical action stream,
    so their checksums MUST match. PP stages own disjoint layers, so a
    rank rejects actions it cannot apply (layer not owned) and stages may
    legitimately differ — comparison is therefore scoped WITHIN each PP
    stage (grouped by ``pp_rank``). Sync-consumer-originated actions only
    exist at ``pipeline_parallel_size == 1`` anyway (see
    ``vllm/v1/capture/registry.py``), where every worker is one stage and
    this reduces to an all-workers comparison.

    Returns a picklable block. On agreement:
    ``{"consistent": True, "action_count": N}``. On divergence:
    ``{"consistent": False, "checksums": {worker_key: checksum}}`` where a
    worker key is ``"tp{t}/pp{p}"`` (plus per-stage detail).
    """
    stages: dict[int, list[tuple[int, str | None, int]]] = {}
    for worker in per_worker:
        if not worker or worker.get("action_checksum") is None:
            continue
        pp = int(worker.get("pp_rank", 0))
        tp = int(worker.get("tp_rank", 0))
        stages.setdefault(pp, []).append(
            (tp, worker.get("action_checksum"), int(worker.get("action_count", 0)))
        )

    if not stages:
        return {"consistent": True, "action_count": 0}

    consistent = True
    per_worker_checksums: dict[str, str | None] = {}
    representative_count = 0
    for pp, members in sorted(stages.items()):
        distinct = {chk for _tp, chk, _cnt in members}
        if len(distinct) > 1:
            consistent = False
        for tp, chk, cnt in members:
            per_worker_checksums[f"tp{tp}/pp{pp}"] = chk
            representative_count = max(representative_count, cnt)

    if consistent:
        return {"consistent": True, "action_count": representative_count}
    return {"consistent": False, "checksums": per_worker_checksums}


def normalize_worker_err(err_str: str) -> str:
    """Strip a leading ``"Rank N: "`` prefix from a worker error message.

    ``collective_rpc`` wraps worker-side exceptions with a rank prefix
    so the router can see which rank failed. For user-facing errors we
    strip it so the user sees a clean message (all TP ranks produce
    the same error for validation failures anyway).
    """
    return _RANK_PREFIX.sub("", err_str, count=1)
