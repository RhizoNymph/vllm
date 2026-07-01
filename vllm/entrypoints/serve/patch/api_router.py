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
    patch_2a: dict | None = None,
) -> tuple[dict[int, Any] | None, int]:
    """Run one short greedy generation; return (first-token logprobs, n_prompt)."""
    sp = SamplingParams(
        temperature=0.0, max_tokens=1, logprobs=logprobs, patch=patch,
        patch_2a=patch_2a,
    )
    request_id = f"patchsweep-{tag}-{uuid4().hex[:8]}"
    final = None
    async for out in eng.generate({"prompt": prompt}, sp, request_id):
        final = out
    if final is None or not final.outputs or not final.outputs[0].logprobs:
        return None, len(final.prompt_token_ids) if final else 0
    return final.outputs[0].logprobs[0], len(final.prompt_token_ids)


async def _capture_trunk(
    eng: EngineClient, prompt: str, trunk_layers: list[int]
) -> str:
    """Capture the corrupt baseline's ``post_block`` trunk (the 2a entry rows).

    Only the layers actually needed as entry residuals are captured (``L - 1``
    for each swept entry-layer ``L``); capturing all layers is dominated by the
    per-row store writes and needlessly slow. Returns the trunk run handle, held
    until the capture writes are durable.
    """
    trunk_run = f"trunk-{uuid4().hex[:8]}"
    capture = {
        "patch_source": {
            "run": trunk_run,
            "hooks": {"post_block": trunk_layers},
            "positions": "all_prompt",
        }
    }
    sp = SamplingParams(temperature=0.0, max_tokens=1, capture=capture)
    request_id = f"patchsweep-trunk-{uuid4().hex[:8]}"
    async for _ in eng.generate({"prompt": prompt}, sp, request_id):
        pass
    if hasattr(eng, "wait_for_capture_results"):
        await eng.wait_for_capture_results(request_id)
    return trunk_run


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
    if body.mode == "2a":
        # 2a re-enters at layer L and must recompute every position (layers >= L
        # build their KV from the injected trunk); prefix reuse would serve stale
        # KV. Fail loud until the per-request floor-0 wiring lands.
        cache_config = getattr(vllm_config, "cache_config", None)
        if getattr(cache_config, "enable_prefix_caching", False):
            return _err("mode=2a requires the server to run with "
                        "--no-enable-prefix-caching")

    num_layers = vllm_config.model_config.get_total_num_hidden_layers()
    layers = resolve_layers(body.layers)
    if any(not (0 <= layer < num_layers) for layer in layers):
        return _err(f"layer out of range [0, {num_layers})")

    # Corrupt baseline (no patch) — also fixes the prompt length for
    # "all_prompt" position resolution.
    try:
        corrupt_lp, n_prompt = await _first_token_logprobs(
            eng, body.prompt, None, body.logprobs, "baseline"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("patch_sweep baseline failed")
        return _err(f"baseline generation failed: {exc}",
                    HTTPStatus.INTERNAL_SERVER_ERROR.value)
    positions = resolve_positions(body.positions, n_prompt)
    if not layers or not positions:
        return _err("empty sweep grid (no layers or positions)")
    corrupt_val = cell_metric(corrupt_lp, body) if corrupt_lp else None

    # Validate every referenced source site exists (one combined spec).
    from vllm.v1.capture.patch_admission import (
        PatchValidationError,
        validate_patch_sources,
    )

    probe = SamplingParams(
        patch=[
            {
                "layer": layer,
                "hook": body.hook,
                "dest_position": pos,
                "source_run": body.source_run,
                "source_position": pos,
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

    # For 2a, capture the corrupt trunk once so cells can re-enter mid-stack.
    trunk_run: str | None = body.trunk_run if body.mode == "2a" else None
    if body.mode == "2a" and trunk_run is None:
        # No pre-captured trunk: capture it in-line (only the needed layers,
        # post_block[L-1]). Reusing a pre-captured trunk across sweeps avoids
        # paying this each time.
        trunk_layers = sorted({layer - 1 for layer in layers if layer >= 1})
        try:
            trunk_run = await _capture_trunk(eng, body.prompt, trunk_layers)
        except Exception as exc:  # noqa: BLE001
            logger.exception("patch_sweep 2a trunk capture failed")
            return _err(f"2a trunk capture failed: {exc}",
                        HTTPStatus.INTERNAL_SERVER_ERROR.value)

    async def run_cell(i: int, layer: int, j: int, pos: int) -> None:
        patch = [
            {
                "layer": layer,
                "hook": body.hook,
                "dest_position": pos,
                "source_run": body.source_run,
                "source_position": pos,
                "alpha": body.alpha,
            }
        ]
        # Re-enter at ``layer`` from the cached trunk (entry_layer >= 1 only;
        # layer 0 has nothing below to skip, so it runs as Level-1).
        patch_2a = (
            {"entry_layer": layer, "trunk_run": trunk_run}
            if trunk_run is not None and layer >= 1
            else None
        )
        lp, _ = await _first_token_logprobs(
            eng, body.prompt, patch, body.logprobs, f"{layer}-{pos}", patch_2a
        )
        grid[i][j] = cell_metric(lp, body) if lp else None

    if trunk_run is not None:
        # Serialize per entry-layer so each in-flight batch is homogeneous in
        # entry_layer (the runner requires a single start_layer per forward).
        for i, layer in enumerate(layers):
            await asyncio.gather(
                *(run_cell(i, layer, j, pos) for j, pos in enumerate(positions))
            )
    else:
        await asyncio.gather(
            *(
                run_cell(i, layer, j, pos)
                for i, layer in enumerate(layers)
                for j, pos in enumerate(positions)
            )
        )

    clean_val = body.clean_baseline
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
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
