# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-time validation + prefix-cache classification for patch specs.

Runs at the OpenAI entrypoint (``_admit_patch``). Unlike capture, patch entries
carry explicit ``dest_position`` ints, so no consumer validator is needed — the
prefix floor is read straight off the spec. Validates hook/layer/position and
the strict single-request pool capacity, then stamps ``patch_touches_prompt`` /
``patch_min_prompt_position`` (mirrors ``capture_*``) so prompt-range patches
re-forward from their lowest patched position and the injection hook fires.

Source-run existence is not checked here (it would need an engine RPC to the
per-worker source store); a missing source surfaces at worker resolution as a
logged, skipped entry.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.sampling_params import SamplingParams
    from vllm.v1.capture.types import CaptureContext

logger = init_logger(__name__)

_INJECTABLE_HOOKS = frozenset(h.value for h in HOOK_POINT_TABLE_ATTR)


class PatchValidationError(ValueError):
    """Raised when a patch spec fails admission validation (-> HTTP 400)."""


def resolve_patch_prefix_flags(
    sampling_params: SamplingParams,
    ctx: CaptureContext,
    *,
    max_patch_slots: int,
) -> None:
    """Validate ``sampling_params.patch`` and stamp prefix-cache flags in place.

    Raises :class:`PatchValidationError` on an invalid hook, out-of-range
    layer, negative position, or a single-request per-site demand exceeding the
    pool (strict policy).
    """
    spec = sampling_params.patch
    if not spec:
        return

    prompt_floors: list[int] = []
    site_counts: dict[tuple[int, str], int] = {}
    for i, entry in enumerate(spec):
        hook = entry["hook"]
        if hook not in _INJECTABLE_HOOKS:
            raise PatchValidationError(
                f"patch[{i}]: hook {hook!r} is not injectable; valid: "
                f"{sorted(_INJECTABLE_HOOKS)}"
            )
        layer = int(entry["layer"])
        if not (0 <= layer < ctx.num_hidden_layers):
            raise PatchValidationError(
                f"patch[{i}]: layer {layer} out of range "
                f"[0, {ctx.num_hidden_layers})"
            )
        dest = int(entry["dest_position"])
        if dest < 0:
            raise PatchValidationError(
                f"patch[{i}]: dest_position {dest} must be >= 0"
            )
        if int(entry["source_position"]) < 0:
            raise PatchValidationError(
                f"patch[{i}]: source_position must be >= 0"
            )
        key = (layer, hook)
        site_counts[key] = site_counts.get(key, 0) + 1
        if max_patch_slots and site_counts[key] > max_patch_slots - 1:
            raise PatchValidationError(
                f"patch needs more than {max_patch_slots - 1} slots at "
                f"layer={layer} hook={hook}; raise --max-patch-slots"
            )
        if dest < ctx.num_prompt_tokens:
            prompt_floors.append(dest)

    if prompt_floors:
        sampling_params.patch_touches_prompt = True
        sampling_params.patch_min_prompt_position = min(prompt_floors)
    else:
        sampling_params.patch_touches_prompt = False
        sampling_params.patch_min_prompt_position = None


class _PatchSourceCache:
    """Entrypoint-side cache of available patch source runs, refreshed from the
    worker(s) via ``collective_rpc("get_patch_source_manifests")``.

    Refresh-on-miss: a referenced run/site that is not cached triggers one
    refresh before rejecting, so a just-captured clean run is never
    false-rejected. A present run is fetched once and then served from cache,
    so a sweep that reuses one run issues ~one RPC total.

    Validated runs are also LEASED against store eviction (the positive cache
    can go stale on whole-run eviction, and even a fresh check leaves an
    admission→resolution window). Leases are renewed lazily — one lease RPC per
    run per ~half-TTL — so the per-cell cost stays zero.
    """

    LEASE_TTL_S = 300.0

    def __init__(self) -> None:
        # run_id -> {"sites": set[(hook, layer)], "positions": set[int]}
        self._runs: dict[str, dict[str, set]] = {}
        self._run_prompt_tokens: dict[str, int] = {}
        self._lock = asyncio.Lock()
        # run_id -> monotonic time we last leased it on the workers.
        self._leased_at: dict[str, float] = {}

    async def _refresh(self, engine_client: EngineClient) -> None:
        results = await engine_client.collective_rpc("get_patch_source_manifests")
        agg: dict[str, dict[str, set]] = {}
        num_prompt_tokens: dict[str, int] = {}
        for rank_manifests in results or []:
            for m in rank_manifests or []:
                entry = agg.setdefault(
                    m["run_id"], {"sites": set(), "positions": set()}
                )
                entry["sites"].update(
                    (hook, int(layer)) for hook, layer in m["hook_layers"]
                )
                entry["positions"].update(int(p) for p in m["positions"])
                num_prompt_tokens[m["run_id"]] = max(
                    num_prompt_tokens.get(m["run_id"], 0),
                    int(m.get("num_prompt_tokens", 0)),
                )
        self._runs = agg
        self._run_prompt_tokens = num_prompt_tokens

    async def run_prompt_tokens(
        self, run_id: str, engine_client: EngineClient
    ) -> int | None:
        """Prompt-token count of ``run_id``'s captured clean prompt, or None.

        Used to detect clean/corrupt tokenization-length mismatch before a
        sweep assumes ``source == dest`` positions. Refresh-on-miss like
        ``validate``; best-effort (None) on RPC failure."""
        if run_id not in getattr(self, "_run_prompt_tokens", {}):
            async with self._lock:
                try:
                    await self._refresh(engine_client)
                except Exception:  # noqa: BLE001 — best-effort
                    return None
        return getattr(self, "_run_prompt_tokens", {}).get(run_id)

    def _missing(
        self, refs: list[tuple[str, str, int, int]]
    ) -> tuple[str, str, int, int] | None:
        for run, hook, layer, pos in refs:
            entry = self._runs.get(run)
            if (
                entry is None
                or (hook, layer) not in entry["sites"]
                or pos not in entry["positions"]
            ):
                return (run, hook, layer, pos)
        return None

    async def validate(
        self, sampling_params: SamplingParams, engine_client: EngineClient
    ) -> None:
        """Raise :class:`PatchValidationError` if any referenced source site is
        absent. Best-effort: if the RPC fails, admission proceeds (the worker
        log-and-skips a missing source)."""
        spec = sampling_params.patch
        if not spec:
            return
        refs = [
            (
                str(e["source_run"]),
                str(e["hook"]),
                int(e["layer"]),
                int(e["source_position"]),
            )
            for e in spec
        ]
        if self._missing(refs) is None:
            await self._maybe_lease(engine_client, {r[0] for r in refs})
            return  # all cached-present
        async with self._lock:
            try:
                await self._refresh(engine_client)
            except Exception as exc:  # noqa: BLE001 — best-effort admission check
                logger.warning(
                    "patch source manifest RPC failed (%s); skipping admission "
                    "existence check (worker records a resolution failure on "
                    "a missing source)",
                    exc,
                )
                return
        miss = self._missing(refs)
        if miss is not None:
            run, hook, layer, pos = miss
            raise PatchValidationError(
                f"patch source not found: run={run!r} site=(layer={layer}, "
                f"hook={hook}, position={pos}). Capture the clean run "
                f"(capture={{'patch_source': ...}}, capture_wait=True) first."
            )
        await self._maybe_lease(engine_client, {r[0] for r in refs})

    async def _maybe_lease(
        self, engine_client: EngineClient, runs: set[str]
    ) -> None:
        """Lease ``runs`` against eviction, renewing at most once per half-TTL.

        Best-effort: a failed lease RPC degrades to the resolution-failure
        backstop rather than blocking admission.
        """
        now = time.monotonic()
        stale = [
            run for run in runs
            if now - self._leased_at.get(run, 0.0) > self.LEASE_TTL_S / 2
        ]
        if not stale:
            return
        try:
            await engine_client.collective_rpc(
                "lease_patch_source_runs", args=(stale, self.LEASE_TTL_S)
            )
            for run in stale:
                self._leased_at[run] = now
        except Exception as exc:  # noqa: BLE001 — lease is best-effort
            logger.warning("patch source lease RPC failed (%s)", exc)


# Process-global cache shared across the chat + completion serving instances
# (one engine per API-server process).
_PATCH_SOURCE_CACHE = _PatchSourceCache()


async def validate_patch_sources(
    engine_client: EngineClient, sampling_params: SamplingParams
) -> None:
    """Validate that a request's patch sources exist (raises on missing)."""
    await _PATCH_SOURCE_CACHE.validate(sampling_params, engine_client)


async def get_run_prompt_tokens(
    engine_client: EngineClient, run_id: str
) -> int | None:
    """Prompt-token count of a captured source run (None if unknown)."""
    return await _PATCH_SOURCE_CACHE.run_prompt_tokens(run_id, engine_client)
