# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a request's ``patch`` spec into source-vector :class:`PatchEntry`s.

A patch spec entry sources its value one of four ways (exactly one per entry):

- ``source_run`` + ``source_position``: a clean run's stored activation, looked
  up in the per-worker :class:`PatchSourceStore`.
- ``source_module``: the BASE ``vectors`` tier of a named steering module at the
  same ``[hook][layer]`` the entry patches (``{"vector", "scale"}`` entries
  resolve to ``scale * vector``). Reuses the runner's steering module registry.
- ``source_module = "zeros"``: a reserved built-in resolving to a zero row of
  the hook width — no registry needed (works offline).
- ``source_inline``: a row index into the request-level packed ``patch_vectors``
  table (base64 binary wire form, decoded once per request).

An optional per-entry ``mask`` (``{"indices": [...]}`` or ``{"inline": row}``)
composes with any source kind: it restricts the patch to a subset of dims via
``out_d = hs_d + alpha * m_d * (src_d - hs_d)``. Since ``alpha * mask`` is just a
per-dimension alpha, resolution folds it into ``PatchEntry.alpha_row`` — no
separate kernel path.

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

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.capture.source_store import get_active_patch_source_store
from vllm.v1.worker.patch_runner_mixin import PatchEntry, get_patchable_hook

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData

# Reserved built-in module name: a zero source row of the hook width. Needs no
# registry and no store, so it works everywhere including offline.
ZEROS_MODULE = "zeros"

_PATCH_VECTOR_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

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


def _decode_patch_vectors(patch_vectors: Any) -> torch.Tensor:
    """Decode the request-level packed ``patch_vectors`` table to a CPU tensor.

    Same binary wire form as ``SteeringHookPacked`` minus layer_indices/scales:
    ``{"dtype", "shape": [n_rows, width], "data": <base64 bytes>}``. Returns an
    owned ``(n_rows, width)`` fp32 tensor (rows are cast once here; the final
    device/table-dtype cast happens at step staging). Raises on a malformed
    payload — the caller records a per-entry resolution failure.
    """
    import pybase64 as base64

    dtype = _PATCH_VECTOR_DTYPES[str(patch_vectors["dtype"])]
    shape = tuple(int(s) for s in patch_vectors["shape"])
    raw = base64.b64decode(patch_vectors["data"])
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    return flat.reshape(shape).to(torch.float32).clone()


def _module_source_row(
    module_registry: dict | None,
    name: str,
    hook: str,
    layer: int,
    hidden_size: int | None,
) -> torch.Tensor | None:
    """Resolve a named-module source row: its BASE ``vectors`` tier at
    ``[hook][layer]`` (``{"vector", "scale"}`` -> ``scale * vector``). The
    reserved ``zeros`` name needs no registry."""
    if name == ZEROS_MODULE:
        if hidden_size is None:
            return None
        return torch.zeros(int(hidden_size), dtype=torch.float32)
    if module_registry is None:
        return None
    specs = module_registry.get(name)
    if not specs:
        return None
    base = specs[0]  # (vectors, prefill_vectors, decode_vectors)
    if not base:
        return None
    layer_dict = base.get(hook)
    if not layer_dict:
        return None
    entry = layer_dict.get(layer, layer_dict.get(int(layer)))
    if entry is None:
        return None
    from vllm.config.steering_types import normalize_layer_entry

    vec, scale = normalize_layer_entry(entry)
    row = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
    if scale != 1.0:
        row = row * float(scale)
    return row


def _resolve_source_row(
    e: dict,
    *,
    store,
    module_registry: dict | None,
    hidden_size: int | None,
    decoded: torch.Tensor | None,
) -> tuple[torch.Tensor | None, str | None]:
    """Resolve one entry's source vector by its kind; ``(row, None)`` on hit,
    ``(None, detail)`` on a resolvable-but-missing source."""
    layer = int(e["layer"])
    hook = str(e["hook"])
    if e.get("source_run") is not None:
        if store is None:
            return None, "no active source store on rank"
        run = str(e["source_run"])
        pos = int(e["source_position"])
        row = store.get_row(run, layer, hook, pos)
        if row is None:
            return None, (
                f"source missing: run={run} layer={layer} hook={hook} "
                f"pos={pos} (evicted or never captured)"
            )
        return row, None
    if e.get("source_module") is not None:
        name = str(e["source_module"])
        row = _module_source_row(module_registry, name, hook, layer, hidden_size)
        if row is None:
            return None, (
                f"source_module {name!r} has no vectors row at hook={hook} "
                f"layer={layer} (unregistered or missing site)"
            )
        return row, None
    if e.get("source_inline") is not None:
        if decoded is None:
            return None, "source_inline set but patch_vectors missing/undecodable"
        idx = int(e["source_inline"])
        if not (0 <= idx < decoded.shape[0]):
            return None, (
                f"source_inline index {idx} out of range "
                f"[0, {decoded.shape[0]})"
            )
        return decoded[idx].clone(), None
    return None, "entry has no source kind (need source_run/module/inline)"


def _resolve_alpha_row(
    e: dict, *, width: int, decoded: torch.Tensor | None
) -> tuple[torch.Tensor | None, str | None]:
    """Fold ``alpha * mask`` into a per-dim weight row of ``width`` dims.

    No mask -> a constant ``alpha`` fill. ``{"indices": [...]}`` -> a 0/1 mask.
    ``{"inline": row}`` -> a graded mask row from ``patch_vectors``.
    """
    alpha = float(e.get("alpha", 1.0))
    mask = e.get("mask")
    if mask is None:
        return torch.full((width,), alpha, dtype=torch.float32), None
    if mask.get("indices") is not None:
        m = torch.zeros(width, dtype=torch.float32)
        idxs = [int(i) for i in mask["indices"]]
        for i in idxs:
            if not (0 <= i < width):
                return None, f"mask index {i} out of range [0, {width})"
        if idxs:
            m[idxs] = 1.0
        return alpha * m, None
    if mask.get("inline") is not None:
        if decoded is None:
            return None, "mask inline set but patch_vectors missing/undecodable"
        idx = int(mask["inline"])
        if not (0 <= idx < decoded.shape[0]):
            return None, f"mask inline index {idx} out of range [0, {decoded.shape[0]})"
        row = decoded[idx]
        if int(row.shape[0]) != width:
            return None, (
                f"mask inline row width {int(row.shape[0])} != source width {width}"
            )
        return alpha * row.clone(), None
    return None, "mask has neither indices nor inline"


def _resolve_local(
    candidates: list[dict],
    req_id: str,
    *,
    module_registry: dict | None,
    hidden_size: int | None,
    patch_vectors: Any,
) -> list[PatchEntry]:
    """Resolve candidate spec entries against every source kind.

    Missing sources (evicted store row, unknown module, out-of-range inline
    index, bad mask) are logged, recorded in the resolution-failure registry
    (so the request's output can be voided), and skipped — admission is the
    strict gate; a miss here is a race or a bug, and the engine must not crash.
    """
    store = get_active_patch_source_store()
    decoded: torch.Tensor | None = None
    if patch_vectors:
        try:
            decoded = _decode_patch_vectors(patch_vectors)
        except Exception as exc:  # noqa: BLE001 - loud-not-fatal
            detail = f"patch_vectors decode failed: {exc}"
            logger.error("patch resolution for %s: %s", req_id, detail)
            record_resolution_failure(req_id, detail)

    entries: list[PatchEntry] = []
    for e in candidates:
        source, detail = _resolve_source_row(
            e,
            store=store,
            module_registry=module_registry,
            hidden_size=hidden_size,
            decoded=decoded,
        )
        if source is None:
            logger.error("patch resolution for %s: %s; skipping entry", req_id, detail)
            record_resolution_failure(req_id, detail or "unresolved source")
            continue
        alpha_row, mdetail = _resolve_alpha_row(
            e, width=int(source.reshape(-1).shape[0]), decoded=decoded
        )
        if alpha_row is None:
            logger.error("patch resolution for %s: %s; skipping entry", req_id, mdetail)
            record_resolution_failure(req_id, mdetail or "unresolved mask")
            continue
        entries.append(
            PatchEntry(
                layer=int(e["layer"]),
                hook=get_patchable_hook(str(e["hook"])),
                dest_pos=int(e["dest_position"]),
                source=source.reshape(-1),
                alpha_row=alpha_row,
            )
        )
    return entries


def resolve_patch_entries(
    new_req_data: NewRequestData,
    *,
    local_layers: frozenset[int],
    module_registry: dict | None = None,
    hidden_size: int | None = None,
) -> list[PatchEntry]:
    """Resolve ``new_req_data``'s patch spec into entries for local layers.

    Runs in lockstep on every TP rank; rank 0 owns the store + registry and
    broadcasts the resolved entries (source + ``alpha_row`` tensors ride the
    same ``broadcast_object``) to peers (no-op at TP1).
    """
    sampling_params = getattr(new_req_data, "sampling_params", None)
    if sampling_params is None:
        return []
    spec = getattr(sampling_params, "patch", None)
    if not spec:
        return []
    patch_vectors = getattr(sampling_params, "patch_vectors", None)

    # Candidate entries for locally-owned layers, in deterministic spec order —
    # identical on every rank (all ranks see the same broadcast spec).
    candidates = [e for e in spec if int(e["layer"]) in local_layers]
    if not candidates:
        return []

    tp = _tp_group()
    world_size = getattr(tp, "world_size", 1) if tp is not None else 1

    if world_size <= 1:
        return _resolve_local(
            candidates,
            new_req_data.req_id,
            module_registry=module_registry,
            hidden_size=hidden_size,
            patch_vectors=patch_vectors,
        )

    # TP>1: rank 0 resolves (store + registry live there) and broadcasts to
    # peers so all ranks apply the identical patch (residual is replicated).
    if tp.rank_in_group == 0:
        resolved = _resolve_local(
            candidates,
            new_req_data.req_id,
            module_registry=module_registry,
            hidden_size=hidden_size,
            patch_vectors=patch_vectors,
        )
        tp.broadcast_object(resolved, src=0)
        return resolved
    return tp.broadcast_object(None, src=0) or []
