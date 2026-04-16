# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import hashlib
import secrets
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.config.steering_types import (
    SteeringVectorSpec,
    normalize_layer_entry,
)
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.steering.protocol import SetSteeringRequest
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES

logger = init_logger(__name__)

router = APIRouter()

# Serializes steering mutations (set / clear) so the two-phase
# validate-then-apply flow in /set cannot be interleaved with
# another /set or /clear request.
_steering_lock = asyncio.Lock()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _authorize_steering_mutation(request: Request) -> JSONResponse | None:
    """Gate mutating steering endpoints behind the steering API key."""
    tokens: list[str] = getattr(request.app.state, "steering_api_tokens", None) or []
    if not tokens:
        return None

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse(
            content={"error": "Unauthorized: steering API key required."},
            status_code=HTTPStatus.UNAUTHORIZED.value,
        )

    scheme, _, param = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not param:
        return JSONResponse(
            content={"error": "Unauthorized: steering API key required."},
            status_code=HTTPStatus.UNAUTHORIZED.value,
        )

    presented = hashlib.sha256(param.encode("utf-8")).digest()
    matched = False
    for token in tokens:
        expected = hashlib.sha256(token.encode("utf-8")).digest()
        matched |= secrets.compare_digest(presented, expected)
    if not matched:
        return JSONResponse(
            content={"error": "Unauthorized: invalid steering API key."},
            status_code=HTTPStatus.UNAUTHORIZED.value,
        )
    return None


def _normalize_spec(
    spec: SteeringVectorSpec,
) -> dict[str, dict[int, list[float]]]:
    """Convert a SteeringVectorSpec with co-located scales to pre-scaled
    flat vectors.

    Each layer entry may be a bare ``list[float]`` (scale=1.0) or
    ``{"vector": [...], "scale": float}``.  This function applies the
    scale and returns plain ``list[float]`` values.

    Hook entries whose inner layer dict is empty are dropped from the
    result: a hook with no layers is functionally equivalent to
    omitting the hook entirely, and keeping an empty entry would
    produce a truthy-but-empty spec that confuses downstream checks.
    """
    result: dict[str, dict[int, list[float]]] = {}
    for hook_name, layer_vecs in spec.items():
        normalized_layers: dict[int, list[float]] = {}
        for layer_idx, entry in layer_vecs.items():
            vec, scale = normalize_layer_entry(entry)
            if scale != 1.0:
                normalized_layers[layer_idx] = [v * scale for v in vec]
            else:
                normalized_layers[layer_idx] = vec
        if normalized_layers:
            result[hook_name] = normalized_layers
    return result


def _validate_hook_points(
    spec: SteeringVectorSpec,
) -> set[str] | None:
    """Return invalid hook point names from *spec*, or ``None`` if all
    are valid."""
    invalid = set(spec.keys()) - VALID_HOOK_POINT_NAMES
    return invalid if invalid else None


@router.post("/v1/steering/set")
async def set_steering(
    request: SetSteeringRequest,
    raw_request: Request,
) -> JSONResponse:
    """Set activation steering vectors on decoder layers.

    Supports three-tier steering:
    - ``vectors``: base vectors applied to both prefill and decode
    - ``prefill_vectors``: added to base during prefill only
    - ``decode_vectors``: added to base during decode only

    Each layer entry is either a bare ``list[float]`` (scale=1.0) or
    ``{"vector": [...], "scale": float}``.

    When ``replace`` is ``True``, all existing vectors across all tiers
    are cleared atomically before the new ones are applied.
    """
    if (unauthorized := _authorize_steering_mutation(raw_request)) is not None:
        return unauthorized

    engine = engine_client(raw_request)

    # Collect all tiers that have data.
    tiers: dict[str, SteeringVectorSpec] = {}
    if request.vectors:
        tiers["vectors"] = request.vectors
    if request.prefill_vectors:
        tiers["prefill_vectors"] = request.prefill_vectors
    if request.decode_vectors:
        tiers["decode_vectors"] = request.decode_vectors

    if not tiers:
        return JSONResponse(
            content={
                "error": (
                    "No vectors provided. Include at least one of "
                    "vectors, prefill_vectors, or decode_vectors "
                    "with hook point/layer data."
                ),
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    # Validate hook point names across all tiers.
    all_invalid: set[str] = set()
    for spec in tiers.values():
        invalid = _validate_hook_points(spec)
        if invalid:
            all_invalid.update(invalid)
    if all_invalid:
        return JSONResponse(
            content={
                "error": (
                    f"Invalid hook point name(s): "
                    f"{sorted(all_invalid)}. "
                    f"Valid values: "
                    f"{sorted(VALID_HOOK_POINT_NAMES)}"
                ),
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    # Normalize co-located scales to pre-scaled flat vectors.
    normalized_base: dict[str, dict[int, list[float]]] | None = None
    normalized_prefill: dict[str, dict[int, list[float]]] | None = None
    normalized_decode: dict[str, dict[int, list[float]]] | None = None

    try:
        if "vectors" in tiers:
            normalized_base = _normalize_spec(tiers["vectors"])
        if "prefill_vectors" in tiers:
            normalized_prefill = _normalize_spec(tiers["prefill_vectors"])
        if "decode_vectors" in tiers:
            normalized_decode = _normalize_spec(tiers["decode_vectors"])
    except (KeyError, TypeError, ValueError) as err:
        return JSONResponse(
            content={"error": f"Malformed steering vector entry: {err}"},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    try:
        async with _steering_lock:
            # Phase 1 -- validate on every worker without mutating
            # buffers.  We validate all tiers together.
            results = await engine.collective_rpc(
                "set_steering_vectors",
                args=(),
                kwargs=dict(
                    vectors=normalized_base,
                    prefill_vectors=normalized_prefill,
                    decode_vectors=normalized_decode,
                    validate_only=True,
                ),
            )
            validated_layers: set[int] = set()
            for per_worker in results:
                validated_layers.update(per_worker)

            # Collect all requested layers across all tiers.
            requested_layers: set[int] = set()
            for spec in [normalized_base, normalized_prefill, normalized_decode]:
                if spec:
                    for layer_vecs in spec.values():
                        requested_layers.update(layer_vecs.keys())

            missing = requested_layers - validated_layers
            if missing:
                return JSONResponse(
                    content={
                        "error": (
                            f"Layer(s) {sorted(missing)} not found in "
                            f"model. Steerable layers that matched: "
                            f"{sorted(validated_layers) or 'none'}"
                        ),
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )
            if not validated_layers:
                return JSONResponse(
                    content={
                        "error": (
                            "No steerable layers found.  The loaded "
                            "model may not support activation steering, "
                            "or the requested layer indices do not "
                            "exist."
                        ),
                    },
                    status_code=HTTPStatus.BAD_REQUEST.value,
                )

            # Phase 2 -- apply.
            await engine.collective_rpc(
                "set_steering_vectors",
                args=(),
                kwargs=dict(
                    vectors=normalized_base,
                    prefill_vectors=normalized_prefill,
                    decode_vectors=normalized_decode,
                    replace=request.replace,
                    validate_only=False,
                ),
            )

            # Invalidate prefix cache when any steering vectors change.
            # Base and prefill vectors affect prefill-phase KV blocks
            # directly.  Decode vectors affect decode-phase KV blocks
            # which can also be cached and reused via automatic prefix
            # caching (APC) — stale decode-phase blocks would produce
            # outputs inconsistent with the new steering state.
            affects_cache = (
                normalized_base is not None
                or normalized_prefill is not None
                or normalized_decode is not None
                or request.replace  # replace clears all tiers
            )
            if affects_cache:
                success = await engine.reset_prefix_cache(reset_running_requests=True)
                if success:
                    logger.info("Prefix cache invalidated after steering change.")
                else:
                    # Steering vectors are already applied on the
                    # workers but stale KV blocks remain in the prefix
                    # cache.  Returning success here would let future
                    # requests silently reuse blocks computed under the
                    # old steering state.
                    logger.error(
                        "Prefix cache reset failed after steering "
                        "change — some blocks were still in use. "
                        "Steering vectors have been applied but "
                        "cached KV blocks may be stale."
                    )
                    return JSONResponse(
                        content={
                            "error": (
                                "Steering vectors were applied but "
                                "prefix cache could not be fully "
                                "invalidated. Retry the request or "
                                "call /v1/steering/clear and re-apply."
                            )
                        },
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                    )

        # Build response with all hook points across tiers.
        all_hooks: set[str] = set()
        for spec in [normalized_base, normalized_prefill, normalized_decode]:
            if spec:
                all_hooks.update(spec.keys())

        return JSONResponse(
            content={
                "status": "ok",
                "hook_points": sorted(all_hooks),
                "layers_updated": sorted(validated_layers),
            },
        )
    except SteeringVectorError as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:
        err_str = str(err)
        if "expected vector of size" in err_str or "non-finite" in err_str:
            return JSONResponse(
                content={"error": err_str},
                status_code=HTTPStatus.BAD_REQUEST.value,
            )
        logger.exception("Failed to set steering vectors")
        return JSONResponse(
            content={"error": f"Failed to set steering vectors: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/v1/steering/clear")
async def clear_steering(raw_request: Request) -> JSONResponse:
    """Reset all steering vectors to zero (no-op steering)."""
    if (unauthorized := _authorize_steering_mutation(raw_request)) is not None:
        return unauthorized

    engine = engine_client(raw_request)

    try:
        async with _steering_lock:
            await engine.collective_rpc("clear_steering_vectors")
            # Clearing removes all steering vectors, so invalidate
            # prefix cache to prevent reuse of stale KV blocks.
            success = await engine.reset_prefix_cache(reset_running_requests=True)
            if not success:
                logger.error(
                    "Prefix cache reset failed after clearing "
                    "steering vectors — some blocks still in use. "
                    "Cached KV blocks may be stale."
                )
                return JSONResponse(
                    content={
                        "error": (
                            "Steering vectors were cleared but "
                            "prefix cache could not be fully "
                            "invalidated. Retry the request."
                        )
                    },
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                )
        return JSONResponse(content={"status": "ok"})
    except Exception as err:
        logger.exception("Failed to clear steering vectors")
        return JSONResponse(
            content={"error": f"Failed to clear steering vectors: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.get("/v1/steering")
async def get_steering(raw_request: Request) -> JSONResponse:
    """Return which layers currently have non-zero steering vectors."""
    engine = engine_client(raw_request)

    try:
        results = await engine.collective_rpc("get_steering_status")
        active: dict = {}
        for worker_result in results:
            active.update(worker_result)
        return JSONResponse(
            content={
                "active_layers": {str(k): v for k, v in sorted(active.items())},
            },
        )
    except Exception as err:
        logger.exception("Failed to get steering status")
        return JSONResponse(
            content={"error": f"Failed to get steering status: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
