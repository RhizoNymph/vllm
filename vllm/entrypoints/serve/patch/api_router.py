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
import math
from http import HTTPStatus
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.patch.alignment import align_token_positions
from vllm.entrypoints.serve.patch.protocol import (
    LayerRange,
    PatchSweepRequest,
    PatchSweepResponse,
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


def resolve_positions(positions: list[int] | str, num_prompt_tokens: int) -> list[int]:
    if positions == "all_prompt":
        return list(range(num_prompt_tokens))
    return list(positions)


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
    ans = answer_logprob(
        first_token_logprobs, req.answer_token_id, req.answer_token
    )
    if req.metric == "logit_diff":
        foil = answer_logprob(
            first_token_logprobs, req.foil_token_id, req.foil_token
        )
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


# ---- endpoint -------------------------------------------------------------


async def _first_token_logprobs(
    eng: EngineClient,
    prompt: str,
    patch: list[dict] | None,
    logprobs: int,
    tag: str,
    grade_token_ids: list[int] | None = None,
    request_id: str | None = None,
) -> tuple[dict[int, Any] | None, int]:
    """Run one short greedy generation; return (first-token logprobs, n_prompt).

    ``grade_token_ids`` (the answer/foil ids) are scored exactly via
    ``logprob_token_ids`` — their logprobs are always present in the returned
    dict, independent of top-k rank. Without it, an answer outside the top-k
    silently graded as ``None`` (top-k boundary flicker). The engine requires
    ``logprobs == len(logprob_token_ids)`` when ids are given (ids replace
    top-k)."""
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=len(grade_token_ids) if grade_token_ids else logprobs,
        patch=patch,
        logprob_token_ids=grade_token_ids,
    )
    request_id = request_id or f"patchsweep-{tag}-{uuid4().hex[:8]}"
    final = None
    async for out in eng.generate({"prompt": prompt}, sp, request_id):
        final = out
    if final is None or not final.outputs or not final.outputs[0].logprobs:
        return None, len(final.prompt_token_ids) if final else 0
    return final.outputs[0].logprobs[0], len(final.prompt_token_ids)


async def _auto_capture_clean(
    eng: EngineClient,
    clean_prompt: str,
    source_run: str,
    hook: str,
    layers: list[int],
    logprobs: int,
    grade_token_ids: list[int] | None,
) -> dict[int, Any] | None:
    """Capture the clean run server-side, blocking until it is durable.

    Mirrors the client's ``capture_clean``: taps ``hook`` at every swept
    ``layer`` over ``all_prompt`` positions into ``source_run`` via the
    ``patch_source`` capture consumer, then waits for the capture to finalize
    (``capture_wait`` semantics) so the per-worker source store is populated
    before any cell resolves against it.

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
                "hooks": {hook: list(layers)},
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
    import inspect

    tokenizer = eng.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{what} {token!r} tokenizes to {len(ids)} tokens {ids}; grading "
            f"needs a single token — pass {what}_id explicitly"
        )
    return int(ids[0])


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
    if body.hook not in _INJECTABLE_HOOKS:
        return _err(f"hook {body.hook!r} not injectable; valid: "
                    f"{sorted(_INJECTABLE_HOOKS)}")

    num_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layers = resolve_layers(body.layers)
    if any(not (0 <= layer < num_layers) for layer in layers):
        return _err(f"layer out of range [0, {num_layers})")

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
    grade_ids = [t for t in (body.answer_token_id, body.foil_token_id)
                 if t is not None]

    # Corrupt baseline (no patch) — also fixes the prompt length for
    # "all_prompt" position resolution.
    try:
        corrupt_lp, n_prompt = await _first_token_logprobs(
            eng, body.prompt, None, body.logprobs, "baseline", grade_ids
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("patch_sweep baseline failed")
        return _err(f"baseline generation failed: {exc}",
                    HTTPStatus.INTERNAL_SERVER_ERROR.value)
    positions = resolve_positions(body.positions, n_prompt)
    if not layers or not positions:
        return _err("empty sweep grid (no layers or positions)")
    corrupt_val = cell_metric(corrupt_lp, body) if corrupt_lp else None

    # One-call auto-capture: if the referenced run is confirmed missing and a
    # clean_prompt was given, capture the clean run ourselves (hook + swept
    # layers, all_prompt) with capture-wait durability, then proceed as if it
    # had been captured explicitly. A missing run with no clean_prompt (or an
    # existing run) falls through unchanged — the former 400s at
    # validate_patch_sources below, the latter is reused as-is. On an unknown
    # existence check (RPC failure -> None) we also fall through to the
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
                    eng, body.clean_prompt, body.source_run, body.hook,
                    layers, body.logprobs, grade_ids,
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
        import inspect

        tokenizer = eng.get_tokenizer()
        if inspect.isawaitable(tokenizer):
            tokenizer = await tokenizer
        align = align_token_positions(
            tokenizer.encode(body.clean_prompt),
            tokenizer.encode(body.prompt),
        )
        alignment_summary = align.summary()
        aligned, dropped = [], []
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
        source_for = align.source_for
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

    # Validate every referenced source site exists (one combined spec).
    probe = SamplingParams(
        patch=[
            {
                "layer": layer,
                "hook": body.hook,
                "dest_position": pos,
                "source_run": body.source_run,
                "source_position": source_for(pos),
                "alpha": body.alpha,
            }
            for layer in layers
            for pos in positions
        ]
    )
    try:
        await validate_patch_sources(eng, probe)
    except PatchValidationError as exc:
        return _err(str(exc))

    # Fan out one patched variant per cell; the engine batches them.
    grid: list[list[float | None]] = [
        [None] * len(positions) for _ in layers
    ]
    cell_req_ids: dict[str, tuple[int, int, int, int]] = {}

    async def run_cell(i: int, layer: int, j: int, pos: int) -> None:
        patch = [
            {
                "layer": layer,
                "hook": body.hook,
                "dest_position": pos,
                "source_run": body.source_run,
                "source_position": source_for(pos),
                "alpha": body.alpha,
            }
        ]
        request_id = f"patchsweep-{layer}-{pos}-{uuid4().hex[:8]}"
        cell_req_ids[request_id] = (i, layer, j, pos)
        lp, _ = await _first_token_logprobs(
            eng, body.prompt, patch, body.logprobs, f"{layer}-{pos}",
            grade_ids, request_id,
        )
        grid[i][j] = cell_metric(lp, body) if lp else None

    async def rerun_baseline() -> float | None:
        # Same unpatched request as the solo baseline, but batched with the
        # cells: the metric delta between the two IS the batch-nondeterminism
        # noise floor for this sweep (vLLM is not batch-invariant by default).
        lp, _ = await _first_token_logprobs(
            eng, body.prompt, None, body.logprobs, "noisefloor", grade_ids
        )
        return cell_metric(lp, body) if lp else None

    _, corrupt_val_batched = await asyncio.gather(
        asyncio.gather(
            *(
                run_cell(i, layer, j, pos)
                for i, layer in enumerate(layers)
                for j, pos in enumerate(positions)
            )
        ),
        rerun_baseline(),
    )
    noise_floor = (
        abs(corrupt_val_batched - corrupt_val)
        if corrupt_val_batched is not None and corrupt_val is not None
        else None
    )

    # Void any cell whose patch failed to resolve on the workers (source
    # evicted between admission and resolution — near-impossible with leasing,
    # but a silently-unpatched cell reported as a patched result is wrong
    # science, so drain the failure registry and null those cells loudly.
    # (Appends to `skipped`, which may already carry unaligned positions.)
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
            i, layer, j, pos = cell
            grid[i][j] = None
            skipped.append(
                {"layer": layer, "position": pos, "reason": "; ".join(details)}
            )
            logger.error(
                "patch_sweep cell (layer=%d, pos=%d) ran unpatched: %s",
                layer, pos, details,
            )

    # When we auto-captured, the clean baseline is graded from the same
    # internal clean generation (exactly like the corrupt baseline) so the
    # caller needn't supply it; an explicit clean_baseline is used otherwise.
    clean_val = auto_clean_val if auto_captured else body.clean_baseline
    if body.metric == "recovered":
        if clean_val is None or corrupt_val is None or abs(
            clean_val - corrupt_val
        ) < 1e-9:
            logger.warning(
                "recovered metric needs clean_baseline + corrupt baseline "
                "with clean != corrupt; returning raw logprob grid"
            )
        else:
            denom = clean_val - corrupt_val
            grid = [
                [None if v is None else (v - corrupt_val) / denom for v in row]
                for row in grid
            ]

    return PatchSweepResponse(
        layers=layers,
        positions=positions,
        hook=body.hook,
        metric=body.metric,
        grid=grid,
        clean=clean_val,
        corrupt=corrupt_val,
        argmax=argmax_cell(grid, layers, positions),
        skipped=skipped,
        alignment=alignment_summary,
        noise_floor=noise_floor,
        auto_captured=auto_captured,
        captured_source_run=captured_source_run,
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
