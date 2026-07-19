# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-side activation-patching sweep endpoint.

``POST /v1/patch_sweep`` expands a ``(layers x positions)`` grid into one
patched variant per cell, runs them through the continuously-batched engine in
a single fan-out (the shared corrupt-prompt prefix is reused via prefix
caching; each variant is patched at its own site via the per-row patch gating),
and returns the assembled metric grid — replacing the client's per-cell HTTP
round-trips with one call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import AsyncGenerator, Callable
from http import HTTPStatus
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.patch.alignment import align_token_positions
from vllm.entrypoints.serve.patch.protocol import (
    HookGrid,
    LayerRange,
    PatchSweepRequest,
    PatchSweepResponse,
    SpanPosition,
)
from vllm.entrypoints.serve.patch.spans import (
    dedup_positions,
    prompt_char_offsets,
    resolve_span_positions,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

router = APIRouter()

_INJECTABLE_HOOKS = frozenset(h.value for h in HOOK_POINT_TABLE_ATTR)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


# ---- pure helpers (unit-tested without an engine) -------------------------


def resolve_layers(layers: list[int] | LayerRange) -> list[int]:
    if isinstance(layers, LayerRange):
        return list(range(layers.start, layers.stop, layers.step))
    return list(layers)


def resolve_span_body_positions(
    tokenizer,
    prompt: str,
    positions: list[int | SpanPosition],
) -> list[int]:
    """Resolve ``SpanPosition`` markers against ``prompt`` to token indices.

    Plain ints pass through; each span expands to its covering token positions
    (computed with the same tokenization the sweep uses). Expansion is
    order-preserving with dedup across the whole list. Tokenizes only when a
    span is present.

    Raises:
        ValueError: an empty span, a substring not found, or an out-of-range
            occurrence (surfaced by the endpoint as a 400).
    """
    if not any(isinstance(p, SpanPosition) for p in positions):
        return [int(p) for p in positions if isinstance(p, int)]
    text, offsets = prompt_char_offsets(tokenizer, prompt)
    return dedup_positions(
        resolve_span_positions(offsets, text, p.span, p.occurrence)
        if isinstance(p, SpanPosition)
        else [int(p)]
        for p in positions
    )


def answer_logprob(
    first_token_logprobs: dict[int, Any],
    answer_token_id: int | None,
    answer_token: str | None,
) -> float | None:
    """Logprob of the answer in a ``{token_id: Logprob}`` dict, or ``None``.

    Prefers ``answer_token_id`` (exact); else matches ``answer_token`` against
    each entry's ``decoded_token`` (whitespace-tolerant)."""
    if answer_token_id is not None:
        entry = first_token_logprobs.get(answer_token_id)
        return float(entry.logprob) if entry is not None else None
    if answer_token is None:
        return None
    target = answer_token.strip()
    for entry in first_token_logprobs.values():
        decoded = getattr(entry, "decoded_token", None)
        if decoded is not None and decoded.strip() == target:
            return float(entry.logprob)
    return None


def cell_metric(
    first_token_logprobs: dict[int, Any], req: PatchSweepRequest
) -> float | None:
    """``logprob`` or ``logit_diff`` (``recovered`` is normalized later)."""
    ans = answer_logprob(first_token_logprobs, req.answer_token_id, req.answer_token)
    if req.metric == "logit_diff":
        foil = answer_logprob(first_token_logprobs, req.foil_token_id, req.foil_token)
        if ans is None or foil is None:
            return None
        return ans - foil
    return ans


def argmax_cell(
    grid: list[list[float | None]], layers: list[int], positions: list[int]
) -> dict | None:
    best = None
    best_val = -math.inf
    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            v = grid[i][j]
            if v is not None and v > best_val:
                best_val = v
                best = {"layer": layer, "position": pos, "value": v}
    return best


def _err(msg: str, code: int = HTTPStatus.BAD_REQUEST.value) -> JSONResponse:
    return JSONResponse(content={"error": msg}, status_code=code)


def _make_capture_cell_patch(
    body: PatchSweepRequest, source_for: Callable[[int], int | None]
) -> Callable[[str, int, int], list[dict]]:
    """Capture-sourced per-cell patch: the run's stored activation at the
    aligned source position, plus any shared per-dim mask."""

    def cell_patch(hook: str, layer: int, pos: int) -> list[dict]:
        entry: dict[str, Any] = {
            "layer": layer,
            "hook": hook,
            "dest_position": pos,
            "source_run": body.source_run,
            "source_position": source_for(pos),
            "alpha": body.alpha,
        }
        if body.mask is not None:
            entry["mask"] = body.mask
        return [entry]

    return cell_patch


def _make_vector_cell_patch(
    body: PatchSweepRequest,
) -> Callable[[str, int, int], list[dict]]:
    """Vector-sourced per-cell patch: the same client source (``source_module``
    or ``source_inline``) + shared mask patched at every cell's site."""

    def cell_patch(hook: str, layer: int, pos: int) -> list[dict]:
        entry: dict[str, Any] = {
            "layer": layer,
            "hook": hook,
            "dest_position": pos,
            "alpha": body.alpha,
        }
        if body.source_module is not None:
            entry["source_module"] = body.source_module
        else:
            entry["source_inline"] = body.source_inline
        if body.mask is not None:
            entry["mask"] = body.mask
        return [entry]

    return cell_patch


def _validate_sweep_vectors(body: PatchSweepRequest, vllm_config) -> str | None:
    """Reuse the SamplingParams structural validator on a representative cell
    (exactly-one-of source, mask shape, packed ``patch_vectors``, inline
    index-in-range), then the hidden-size width check for inline rows."""
    entry: dict[str, Any] = {
        "layer": 0,
        "hook": "post_block",
        "dest_position": 0,
        "alpha": body.alpha,
    }
    if body.source_module is not None:
        entry["source_module"] = body.source_module
    elif body.source_inline is not None:
        entry["source_inline"] = body.source_inline
    else:
        entry["source_run"] = body.source_run
        entry["source_position"] = 0
    if body.mask is not None:
        entry["mask"] = body.mask
    try:
        SamplingParams(patch=[entry], patch_vectors=body.patch_vectors)
    except ValueError as exc:
        return str(exc)
    if body.patch_vectors is not None:
        uses_inline = body.source_inline is not None or (
            body.mask is not None and body.mask.get("inline") is not None
        )
        if uses_inline:
            hidden = vllm_config.model_config.get_hidden_size()
            width = int(body.patch_vectors["shape"][1])
            if width != hidden:
                return (
                    f"patch_vectors width {width} != hook width {hidden} "
                    f"(injectable hooks are hidden_size-wide)"
                )
    return None


def _validate_vector_source(
    body: PatchSweepRequest, raw_request: Request, vllm_config, vector_sourced: bool
) -> str | None:
    """Validate the sweep's source mode. Returns an error message or ``None``.

    Capture-sourced sweeps need ``source_run``; vector-sourced sweeps need
    exactly one of ``source_module`` / ``source_inline`` (and reject
    capture-only knobs + the recovered metric)."""
    if not vector_sourced:
        if body.source_run is None:
            return (
                "source_run is required for a capture-sourced sweep (or set "
                "source_module / source_inline for a vector-sourced sweep)"
            )
        return _validate_sweep_vectors(body, vllm_config)
    if body.source_module is not None and body.source_inline is not None:
        return "set exactly one of source_module / source_inline"
    if body.source_run is not None:
        return "source_run cannot be combined with a vector source"
    if body.clean_prompt is not None:
        return "clean_prompt is capture-sourced; omit it for vector-sourced sweeps"
    if body.metric == "recovered":
        return (
            "recovered metric needs a clean baseline, which is unavailable for "
            "vector-sourced sweeps"
        )
    if body.source_module is not None and body.source_module != "zeros":
        registry = getattr(raw_request.app.state, "steering_module_registry", None)
        if registry is None or registry.get(body.source_module) is None:
            avail = registry.list_modules() if registry is not None else []
            return (
                f"unknown source_module {body.source_module!r}; "
                f"available: {avail or 'none'}"
            )
    if body.source_inline is not None and body.patch_vectors is None:
        return "source_inline requires patch_vectors"
    return _validate_sweep_vectors(body, vllm_config)


# ---- endpoint -------------------------------------------------------------


async def _first_token_logprobs(
    eng: EngineClient,
    prompt: str,
    patch: list[dict] | None,
    logprobs: int,
    tag: str,
    grade_token_ids: list[int] | None = None,
    request_id: str | None = None,
    patch_vectors: dict | None = None,
) -> tuple[dict[int, Any] | None, int]:
    """Run one short greedy generation; return (first-token logprobs, n_prompt).

    ``grade_token_ids`` (the answer/foil ids) are scored exactly via
    ``logprob_token_ids`` — their logprobs are always present in the returned
    dict, independent of top-k rank. Without it, an answer outside the top-k
    silently graded as ``None`` (top-k boundary flicker). The engine requires
    ``logprobs == len(logprob_token_ids)`` when ids are given (ids replace
    top-k). ``patch_vectors`` (the request-level packed table) rides along for
    ``source_inline`` / mask ``inline`` cells."""
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(grade_token_ids) if grade_token_ids else logprobs,
        patch=patch,
        patch_vectors=patch_vectors,
        logprob_token_ids=grade_token_ids,
    )
    request_id = request_id or f"patchsweep-{tag}-{uuid4().hex[:8]}"
    final = None
    async for out in eng.generate({"prompt": prompt}, sp, request_id):
        final = out
    if final is None or not final.outputs or not final.outputs[0].logprobs:
        return None, len(final.prompt_token_ids or []) if final else 0
    return final.outputs[0].logprobs[0], len(final.prompt_token_ids or [])


async def _auto_capture_clean(
    eng: EngineClient,
    clean_prompt: str,
    source_run: str,
    layers: list[int],
    logprobs: int,
    grade_token_ids: list[int] | None,
) -> dict[int, Any] | None:
    """Capture the clean run server-side, blocking until it is durable.

    Taps *every* injectable hook (``pre_attn``, ``post_attn``, ``post_block``,
    ``mlp_in``, ``mlp_out``) at every swept ``layer`` over ``all_prompt``
    positions into ``source_run``
    via the ``patch_source`` capture consumer, then waits for the capture to
    finalize (``capture_wait`` semantics) so the per-worker source store is
    populated before any cell resolves against it. Tapping all hooks (one
    forward, only extra source-store bytes) makes a kept run reusable for
    hook-comparison follow-up sweeps at a different hook.

    Args:
        layers: The swept layer set (the capture covers exactly these sites).
        grade_token_ids: Answer/foil ids scored exactly via ``logprob_token_ids``
            so the returned first-token logprobs grade the clean baseline the
            same way the corrupt baseline is graded.

    Returns:
        The clean run's first-token ``{token_id: Logprob}`` dict, or ``None``
        if the generation produced no logprobs.
    """
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(grade_token_ids) if grade_token_ids else logprobs,
        logprob_token_ids=grade_token_ids,
        capture={
            "patch_source": {
                "run": source_run,
                "hooks": {h: list(layers) for h in sorted(_INJECTABLE_HOOKS)},
                "positions": "all_prompt",
            }
        },
    )
    # ``all_prompt`` taps every prompt position: mark the prefix capture-
    # touching (min position 0) so it is re-forwarded and its residual is
    # captured — mirrors what ``_admit_capture`` stamps for ``all_prompt``.
    sp.capture_touches_prompt = True
    sp.capture_min_prompt_position = 0
    request_id = f"patchsweep-capture-{source_run}-{uuid4().hex[:8]}"
    final = None
    async for out in eng.generate({"prompt": clean_prompt}, sp, request_id):
        final = out
    # capture_wait: hold until the source-store writes finalize (async), unless
    # results already arrived inline (waiting then would block to timeout).
    if not getattr(final, "capture_results", None) and hasattr(
        eng, "wait_for_capture_results"
    ):
        await eng.wait_for_capture_results(request_id)
    if final is None or not final.outputs or not final.outputs[0].logprobs:
        return None
    return final.outputs[0].logprobs[0]


async def _resolve_grade_token(
    eng: EngineClient, token: str | None, token_id: int | None, what: str
) -> int | None:
    """Resolve an answer/foil to a single token id (id wins; str must be
    exactly one token). Raises ValueError with a client-facing message."""
    if token_id is not None:
        return int(token_id)
    if token is None:
        return None
    # get_tokenizer is on the concrete engine, not the EngineClient ABC.
    tokenizer = eng.get_tokenizer()  # type: ignore[attr-defined]
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{what} {token!r} tokenizes to {len(ids)} tokens {ids}; grading "
            f"needs a single token — pass {what}_id explicitly"
        )
    return int(ids[0])


async def _drop_patch_source_run(eng: EngineClient, run_id: str) -> bool:
    """Drop ``run_id`` from every worker's source store; return if any dropped.

    Also invalidates the admission-side manifest cache so a subsequent
    existence check no longer reports the run present (otherwise a follow-up
    sweep would skip auto-capture and 400 on the now-absent run).
    """
    from vllm.v1.capture.patch_admission import invalidate_patch_source_run

    results = await eng.collective_rpc("drop_patch_source_run", args=(run_id,))
    invalidate_patch_source_run(run_id)
    return any(bool(r) for r in (results or []))


async def _drop_source_run_quiet(eng: EngineClient, run_id: str) -> None:
    """Best-effort auto-drop: a failure is log-warn, not a request failure."""
    try:
        await _drop_patch_source_run(eng, run_id)
    except Exception as exc:  # noqa: BLE001 — drop is best-effort
        logger.warning(
            "patch_sweep auto-drop of source run %r failed (%s)", run_id, exc
        )


async def _sweep_events(
    eng: EngineClient,
    body: PatchSweepRequest,
    effective_hooks: list[str],
    multi_hook: bool,
    layers: list[int],
    positions: list[int],
    cell_patch: Callable[[str, int, int], list[dict]],
    patch_vectors: dict | None,
    grade_ids: list[int],
    corrupt_val: float | None,
    auto_captured: bool,
    captured_source_run: str | None,
    auto_clean_val: float | None,
    skipped: list[dict],
    alignment_summary: dict | None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Fan out one patched cell per grid site and assemble the response.

    Fans out ``effective_hooks x layers x positions`` cells concurrently (one
    grid per hook, sharing the corrupt baseline and noise floor). Yields
    ``("cell", event)`` as each cell completes (completion order, no ordering
    promise) with the cell's own ``hook``, then a terminal
    ``("summary", PatchSweepResponse)``. Both endpoint paths consume this
    generator: the streaming path serializes each event to SSE, the
    non-streaming path drains it and returns the summary — so the assembled
    response is a single code path.

    Cells run concurrently; if the consumer stops early (client disconnect
    closes the SSE generator) the outstanding cell tasks are cancelled, which
    propagates into their engine requests to abort them best-effort. An
    auto-captured run is dropped (unless ``keep_source``) just before the
    summary yields, and again best-effort on early close.
    """
    grids: dict[str, list[list[float | None]]] = {
        hook: [[None] * len(positions) for _ in layers] for hook in effective_hooks
    }
    cell_req_ids: dict[str, tuple[str, int, int, int, int]] = {}

    # When we auto-captured, the clean baseline is graded from the same
    # internal clean generation (exactly like the corrupt baseline) so the
    # caller needn't supply it; an explicit clean_baseline is used otherwise.
    # The recovered normalization is resolved up front so cell events and the
    # assembled grid are on the same scale (shared across all hooks).
    clean_val = auto_clean_val if auto_captured else body.clean_baseline
    denom: float | None = None
    if body.metric == "recovered":
        if (
            clean_val is None
            or corrupt_val is None
            or abs(clean_val - corrupt_val) < 1e-9
        ):
            logger.warning(
                "recovered metric needs clean_baseline + corrupt baseline "
                "with clean != corrupt; returning raw logprob grid"
            )
        else:
            denom = clean_val - corrupt_val

    def to_metric(value: float | None) -> float | None:
        if value is None or denom is None:
            return value
        # denom is only set when corrupt_val is non-None (see above).
        assert corrupt_val is not None
        return (value - corrupt_val) / denom

    async def run_cell(
        hook: str, i: int, layer: int, j: int, pos: int
    ) -> tuple[str, int, int, int, int, float | None]:
        patch = cell_patch(hook, layer, pos)
        request_id = f"patchsweep-{hook}-{layer}-{pos}-{uuid4().hex[:8]}"
        cell_req_ids[request_id] = (hook, i, layer, j, pos)
        lp, _ = await _first_token_logprobs(
            eng,
            body.prompt,
            patch,
            body.logprobs,
            f"{hook}-{layer}-{pos}",
            grade_ids,
            request_id,
            patch_vectors=patch_vectors,
        )
        return hook, i, layer, j, pos, (cell_metric(lp, body) if lp else None)

    async def rerun_baseline() -> float | None:
        # Same unpatched request as the solo baseline, but batched with the
        # cells: the metric delta between the two IS the batch-nondeterminism
        # noise floor for this sweep (vLLM is not batch-invariant by default).
        # Computed once, shared across hooks.
        lp, _ = await _first_token_logprobs(
            eng, body.prompt, None, body.logprobs, "noisefloor", grade_ids
        )
        return cell_metric(lp, body) if lp else None

    # Auto-drop the fresh uuid run we captured (unless kept) after the grid is
    # assembled: recent uuid runs sit at the MRU end of the source-store LRU
    # and would otherwise evict a user's older deliberate captures first. Never
    # drop a pre-existing run. Runs once — before the summary yields, and again
    # (best-effort) in the finally if the consumer closed the generator early.
    drop_done = False

    async def _maybe_drop() -> None:
        nonlocal drop_done
        if drop_done:
            return
        drop_done = True
        if auto_captured and captured_source_run is not None and not body.keep_source:
            await _drop_source_run_quiet(eng, captured_source_run)

    cell_tasks = [
        asyncio.create_task(run_cell(hook, i, layer, j, pos))
        for hook in effective_hooks
        for i, layer in enumerate(layers)
        for j, pos in enumerate(positions)
    ]
    baseline_task = asyncio.create_task(rerun_baseline())
    try:
        try:
            for fut in asyncio.as_completed(cell_tasks):
                hook, i, layer, j, pos, value = await fut
                grids[hook][i][j] = to_metric(value)
                yield (
                    "cell",
                    {
                        "type": "cell",
                        "hook": hook,
                        "layer": layer,
                        "position": pos,
                        "value": grids[hook][i][j],
                    },
                )
            corrupt_val_batched = await baseline_task
        finally:
            for task in (*cell_tasks, baseline_task):
                if not task.done():
                    task.cancel()

        # In recovered mode the floor is scaled into recovered units so it stays
        # comparable to the grid it qualifies.
        noise_floor = (
            abs(corrupt_val_batched - corrupt_val)
            / (abs(denom) if denom is not None else 1.0)
            if corrupt_val_batched is not None and corrupt_val is not None
            else None
        )

        # Void any cell whose patch failed to resolve on the workers (source
        # evicted between admission and resolution — near-impossible with
        # leasing, but a silently-unpatched cell reported as a patched result is
        # wrong science, so drain the failure registry and null those cells
        # loudly. (Appends to `skipped`, which may already carry unaligned
        # positions.) Each voided cell also re-emits as a null-valued cell event
        # (streaming), carrying its own hook.
        try:
            failure_maps = await eng.collective_rpc("pop_patch_resolution_failures")
        except Exception as exc:  # noqa: BLE001 — backstop is best-effort
            logger.warning("patch resolution-failure drain RPC failed (%s)", exc)
            failure_maps = None
        for rank_failures in failure_maps or []:
            for req_id, details in (rank_failures or {}).items():
                cell = cell_req_ids.get(req_id)
                if cell is None:
                    continue  # not one of this sweep's cells
                hook, i, layer, j, pos = cell
                grids[hook][i][j] = None
                reason = "; ".join(details)
                skipped.append(
                    {"hook": hook, "layer": layer, "position": pos, "reason": reason}
                )
                logger.error(
                    "patch_sweep cell (hook=%s, layer=%d, pos=%d) ran unpatched: %s",
                    hook,
                    layer,
                    pos,
                    details,
                )
                yield (
                    "cell",
                    {
                        "type": "cell",
                        "hook": hook,
                        "layer": layer,
                        "position": pos,
                        "value": None,
                        "error": reason,
                    },
                )

        # Top-level grid/hook/argmax mirror the first hook (single-hook
        # contract); hook_grids carries every hook when `hooks` was requested.
        primary = effective_hooks[0]
        top_grid = grids[primary]
        hook_grids = None
        if multi_hook:
            hook_grids = [
                HookGrid(
                    hook=hook,
                    grid=grids[hook],
                    argmax=argmax_cell(grids[hook], layers, positions),
                )
                for hook in effective_hooks
            ]

        # Drop before the summary yields so both consumption paths (streaming
        # SSE + drained JSON) free the auto-captured run: code after a final
        # yield in an async generator is not reliably executed.
        await _maybe_drop()

        yield (
            "summary",
            PatchSweepResponse(
                layers=layers,
                positions=positions,
                hook=primary,
                metric=body.metric,
                grid=top_grid,
                clean=clean_val,
                corrupt=corrupt_val,
                argmax=argmax_cell(top_grid, layers, positions),
                skipped=skipped,
                alignment=alignment_summary,
                noise_floor=noise_floor,
                auto_captured=auto_captured,
                captured_source_run=captured_source_run,
                hook_grids=hook_grids,
            ),
        )
    finally:
        # Disconnect guard: if the consumer closed the generator early
        # (GeneratorExit before the summary yield), still drop the auto-captured
        # run best-effort. Awaiting in an async generator's finally is allowed
        # as long as we do not yield here.
        await _maybe_drop()


async def _sweep_sse(start_event: dict, **gen_kwargs) -> AsyncGenerator[str, None]:
    """Serialize the sweep event stream as SSE (``text/event-stream``).

    Emits a ``start`` event (grid shape / resolved axes) so consumers can size a
    live heatmap, one ``cell`` event per completed cell (carrying its own
    ``hook``), a terminal ``summary`` event carrying the full
    ``PatchSweepResponse`` payload, then ``[DONE]``.
    """
    yield f"data: {json.dumps(start_event)}\n\n"
    async for kind, payload in _sweep_events(**gen_kwargs):
        if kind == "summary":
            event = {"type": "summary", **payload.model_dump()}
            yield f"data: {json.dumps(event)}\n\n"
        else:
            yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/patch_sweep")
async def patch_sweep(body: PatchSweepRequest, raw_request: Request):
    eng = engine_client(raw_request)
    vllm_config = eng.vllm_config

    if getattr(vllm_config, "patch_config", None) is None:
        return _err("server was not started with --enable-patching")
    if body.answer_token is None and body.answer_token_id is None:
        return _err("one of answer_token / answer_token_id is required")
    if body.metric == "logit_diff" and (
        body.foil_token is None and body.foil_token_id is None
    ):
        return _err("logit_diff metric requires foil_token / foil_token_id")
    # `hooks` (multi-hook) wins over the single `hook` in spirit; validate the
    # effective hook set and dedup order-preserving.
    if body.hooks is not None:
        if not body.hooks:
            return _err("hooks must be non-empty when provided")
        bad = [h for h in body.hooks if h not in _INJECTABLE_HOOKS]
        if bad:
            return _err(
                f"hooks {bad} not injectable; valid: {sorted(_INJECTABLE_HOOKS)}"
            )
        effective_hooks = list(dict.fromkeys(body.hooks))
    else:
        if body.hook not in _INJECTABLE_HOOKS:
            return _err(
                f"hook {body.hook!r} not injectable; valid: {sorted(_INJECTABLE_HOOKS)}"
            )
        effective_hooks = [body.hook]
    multi_hook = body.hooks is not None

    num_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layers = resolve_layers(body.layers)
    if any(not (0 <= layer < num_layers) for layer in layers):
        return _err(f"layer out of range [0, {num_layers})")

    # Source mode: a sweep is either capture-sourced (source_run, existing
    # behavior) or vector-sourced (a client-provided source_module / inline
    # value patched identically into every cell — no capture, no clean run).
    vector_sourced = body.source_module is not None or body.source_inline is not None
    err = _validate_vector_source(body, raw_request, vllm_config, vector_sourced)
    if err is not None:
        return _err(err)

    # Resolve answer/foil to token ids once; every generation then scores them
    # exactly via logprob_token_ids (no top-k dependence, no None flicker).
    try:
        body.answer_token_id = await _resolve_grade_token(
            eng, body.answer_token, body.answer_token_id, "answer_token"
        )
        body.foil_token_id = await _resolve_grade_token(
            eng, body.foil_token, body.foil_token_id, "foil_token"
        )
    except ValueError as exc:
        return _err(str(exc))
    grade_ids = [t for t in (body.answer_token_id, body.foil_token_id) if t is not None]

    # Corrupt baseline (no patch) — also fixes the prompt length for
    # "all_prompt" position resolution.
    try:
        corrupt_lp, n_prompt = await _first_token_logprobs(
            eng, body.prompt, None, body.logprobs, "baseline", grade_ids
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("patch_sweep baseline failed")
        return _err(
            f"baseline generation failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR.value
        )
    if body.positions == "all_prompt":
        positions = list(range(n_prompt))
    else:
        # Substring spans resolve against the corrupt prompt (the destination
        # run), tokenized exactly as the sweep tokenizes it.
        # get_tokenizer is on the concrete engine, not the EngineClient ABC.
        tokenizer = eng.get_tokenizer()  # type: ignore[attr-defined]
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        try:
            positions = resolve_span_body_positions(
                tokenizer, body.prompt, body.positions
            )
        except ValueError as exc:
            return _err(str(exc))
    if not layers or not positions:
        return _err("empty sweep grid (no layers or positions)")
    corrupt_val = cell_metric(corrupt_lp, body) if corrupt_lp else None

    # Vector-sourced sweeps carry their value with the request: no auto-capture,
    # no clean run, no source-manifest existence check, no alignment (positions
    # are the dest run's own). Every cell patches its site from the same client
    # source + mask.
    if vector_sourced:
        skipped: list[dict] = []
        return await _dispatch_sweep(
            eng=eng,
            body=body,
            effective_hooks=effective_hooks,
            multi_hook=multi_hook,
            layers=layers,
            positions=positions,
            cell_patch=_make_vector_cell_patch(body),
            patch_vectors=body.patch_vectors,
            grade_ids=grade_ids,
            corrupt_val=corrupt_val,
            auto_captured=False,
            captured_source_run=None,
            auto_clean_val=None,
            skipped=skipped,
            alignment_summary=None,
        )

    # One-call auto-capture: if the referenced run is confirmed missing and a
    # clean_prompt was given, capture the clean run ourselves (all injectable
    # hooks + swept layers, all_prompt) with capture-wait durability, then
    # proceed as if it had been captured explicitly. A missing run with no
    # clean_prompt (or an existing run) falls through unchanged — the former
    # 400s at validate_patch_sources below, the latter is reused as-is. On an
    # unknown existence check (RPC failure -> None) we also fall through to the
    # best-effort resolution path rather than force-capturing.
    from vllm.v1.capture.patch_admission import (
        PatchValidationError,
        get_run_prompt_tokens,
        patch_source_run_exists,
        validate_patch_sources,
    )

    auto_captured = False
    captured_source_run: str | None = None
    auto_clean_val: float | None = None
    if body.clean_prompt is not None:
        run_exists = await patch_source_run_exists(eng, body.source_run)
        if run_exists is False:
            try:
                clean_lp = await _auto_capture_clean(
                    eng,
                    body.clean_prompt,
                    body.source_run,
                    layers,
                    body.logprobs,
                    grade_ids,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("patch_sweep auto-capture failed")
                return _err(
                    f"auto-capture of clean run {body.source_run!r} failed "
                    f"({exc}); is the patch_source capture consumer enabled "
                    f"(--capture-consumers patch_source)?",
                    HTTPStatus.INTERNAL_SERVER_ERROR.value,
                )
            auto_captured = True
            captured_source_run = body.source_run
            auto_clean_val = cell_metric(clean_lp, body) if clean_lp else None

    # Position alignment: map each dest position to its clean source position.
    # With clean_prompt given, the alignment handles length mismatches (common
    # prefix identity, common suffix shifted, differing middle skipped loudly).
    # Without it, refuse a length mismatch outright — assuming source == dest
    # across a divergence would sweep silently shifted positions.
    skipped: list[dict] = []
    alignment_summary: dict | None = None
    if body.clean_prompt is not None:
        # get_tokenizer is on the concrete engine, not the EngineClient ABC.
        tokenizer = eng.get_tokenizer()  # type: ignore[attr-defined]
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        align = align_token_positions(
            tokenizer.encode(body.clean_prompt),
            tokenizer.encode(body.prompt),
        )
        alignment_summary = align.summary()
        aligned: list[int] = []
        dropped: list[int] = []
        for pos in positions:
            (aligned if align.source_for(pos) is not None else dropped).append(pos)
        for pos in dropped:
            skipped.append(
                {
                    "position": pos,
                    "reason": "unaligned: clean/corrupt token spans differ "
                    "here (no positional correspondence)",
                }
            )
        positions = aligned
        if not positions:
            return _err(
                "no alignable positions: the prompts share no common token "
                "prefix/suffix at the requested positions"
            )
        source_for: Callable[[int], int | None] = align.source_for
    else:
        run_len = await get_run_prompt_tokens(eng, body.source_run)
        if run_len is not None and run_len != n_prompt:
            return _err(
                f"source run {body.source_run!r} was captured from a "
                f"{run_len}-token prompt but this prompt has {n_prompt} "
                f"tokens; positions would misalign. Pass clean_prompt for "
                f"automatic alignment (or explicit aligned positions)."
            )
        source_for = lambda pos: pos  # noqa: E731 — identity alignment

    # Validate every referenced source site exists (one combined spec, all
    # hooks x layers x positions).
    probe = SamplingParams(
        patch=[
            {
                "layer": layer,
                "hook": hook,
                "dest_position": pos,
                "source_run": body.source_run,
                "source_position": source_for(pos),
                "alpha": body.alpha,
            }
            for hook in effective_hooks
            for layer in layers
            for pos in positions
        ]
    )
    try:
        await validate_patch_sources(eng, probe)
    except PatchValidationError as exc:
        return _err(str(exc))

    # Fan out one patched variant per (hook, layer, position) cell; the engine
    # batches them. Streaming and non-streaming share one assembly generator
    # (see _dispatch_sweep). The auto-drop lifecycle lives inside the generator
    # so both paths (and early disconnects) free the auto-captured run.
    return await _dispatch_sweep(
        eng=eng,
        body=body,
        effective_hooks=effective_hooks,
        multi_hook=multi_hook,
        layers=layers,
        positions=positions,
        cell_patch=_make_capture_cell_patch(body, source_for),
        patch_vectors=body.patch_vectors,
        grade_ids=grade_ids,
        corrupt_val=corrupt_val,
        auto_captured=auto_captured,
        captured_source_run=captured_source_run,
        auto_clean_val=auto_clean_val,
        skipped=skipped,
        alignment_summary=alignment_summary,
    )


async def _dispatch_sweep(**gen_kwargs):
    """Route an assembled sweep to SSE (``stream``) or a single JSON response.

    Both consume the shared :func:`_sweep_events` generator; the streaming path
    serializes each event to SSE, the non-streaming path drains the summary.
    """
    body = gen_kwargs["body"]
    effective_hooks = gen_kwargs["effective_hooks"]
    if body.stream:
        start_event = {
            "type": "start",
            "layers": gen_kwargs["layers"],
            "positions": gen_kwargs["positions"],
            "hook": effective_hooks[0],
            "metric": body.metric,
            "auto_captured": gen_kwargs["auto_captured"],
            "captured_source_run": gen_kwargs["captured_source_run"],
        }
        if gen_kwargs["multi_hook"]:
            start_event["hooks"] = effective_hooks
        return StreamingResponse(
            _sweep_sse(start_event, **gen_kwargs),
            media_type="text/event-stream",
        )
    async for kind, payload in _sweep_events(**gen_kwargs):
        if kind == "summary":
            return payload


@router.delete("/v1/patch_source/{run_id}")
async def drop_patch_source(run_id: str, raw_request: Request):
    """Free a captured clean run from the per-worker source stores.

    One-call sweeps auto-drop their fresh uuid runs, but a run kept with
    ``keep_source`` (or captured explicitly) is freed here. Aggregates across
    ranks: 200 ``{"dropped": true}`` if any rank held it, else 404.
    """
    eng = engine_client(raw_request)
    try:
        dropped = await _drop_patch_source_run(eng, run_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("patch_source drop failed")
        return _err(
            f"drop of source run {run_id!r} failed: {exc}",
            HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    if not dropped:
        return _err(
            f"patch source run {run_id!r} not found", HTTPStatus.NOT_FOUND.value
        )
    return JSONResponse(content={"dropped": True})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
