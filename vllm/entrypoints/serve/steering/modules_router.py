# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.config.steering_types import coerce_steering_spec
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.steering.api_router import (
    _authorize_steering_mutation,
)
from vllm.entrypoints.serve.steering.modules_protocol import (
    RegisterSteeringModuleRequest,
    UnregisterSteeringModuleRequest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_registry(request: Request):
    """Get the steering module registry from app state."""
    registry = getattr(request.app.state, "steering_module_registry", None)
    if registry is None:
        return None
    return registry


def _engine_client(request: Request) -> EngineClient | None:
    """Return the engine client from app state if available."""
    return getattr(request.app.state, "engine_client", None)


async def _broadcast_module_to_workers(
    engine: EngineClient | None,
    name: str,
    payload: dict | None,
) -> None:
    """Push a single module entry (or removal) to every worker.

    Mirrors the per-process worker-side ``_steering_module_registry``
    so requests carrying ``SamplingParams.steering_module_ref`` can
    resolve the name without crossing the multiprocessing boundary
    with the full vector spec.

    *payload* of ``None`` removes the module on workers; the
    matching pinned refcount taken at register time (see
    :func:`_pre_materialize_module_on_workers`) is dropped first so the
    manager's row table can GC the row once the last in-flight request
    finishes.

    On the register path, this only mirrors the registry update.  The
    eager row materialization is a separate RPC issued by
    :func:`_pre_materialize_module_on_workers` so the registry state
    is consistent across workers before any per-row materialization
    (which depends on it) runs.
    """
    if engine is None:
        return
    if payload is None:
        await engine.collective_rpc(
            "release_pre_materialized_steering_module",
            kwargs=dict(name=name),
        )
        await engine.collective_rpc(
            "unregister_steering_modules",
            kwargs=dict(names=[name]),
        )
    else:
        await engine.collective_rpc(
            "register_steering_modules",
            kwargs=dict(modules={name: payload}, replace=False),
        )


async def _pre_materialize_module_on_workers(
    engine: EngineClient | None,
    name: str,
) -> None:
    """Tell every worker to materialize the named module's rows now.

    Issued after the registry-update RPC so the worker has the resolved
    spec available.  The pre-materialize call adds ``+1`` to the
    manager's refcount for each ``(hash, phase)`` it materializes,
    pinning the row until ``unregister_steering_modules`` releases
    it.  Per-request register_config calls subsequently bump the
    refcount further, so the request hot path becomes a refcount-hit
    (~5 µs) instead of paying the cold-path materialize cost
    (~15 ms on gemma-3-4b-it/3090 in named_shared mode).
    """
    if engine is None:
        return
    await engine.collective_rpc(
        "pre_materialize_steering_module",
        kwargs=dict(name=name),
    )


@router.post("/v1/steering/modules/register")
async def register_steering_module(
    request: RegisterSteeringModuleRequest,
    raw_request: Request,
) -> JSONResponse:
    """Register a named steering vector configuration."""
    if (unauthorized := _authorize_steering_mutation(raw_request)) is not None:
        return unauthorized
    registry = _get_registry(raw_request)
    if registry is None:
        return JSONResponse(
            content={
                "error": "Steering module registry not initialized. "
                "Ensure --enable-steering is set."
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    # Each tier may arrive as either the legacy SteeringVectorSpec or the
    # binary-wire SteeringVectorSpecPacked shape; normalize before handing
    # off so the registry, the broadcast payload, and the pre-materialize
    # path all see the same legacy-shaped dict.
    try:
        vectors = coerce_steering_spec(request.vectors)
        prefill_vectors = coerce_steering_spec(request.prefill_vectors)
        decode_vectors = coerce_steering_spec(request.decode_vectors)
    except (KeyError, ValueError, TypeError) as err:
        return JSONResponse(
            content={"error": f"Malformed steering payload: {err}"},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    try:
        await registry.register(
            name=request.name,
            vectors=vectors,
            prefill_vectors=prefill_vectors,
            decode_vectors=decode_vectors,
            clamps=request.clamps,
            prefill_clamps=request.prefill_clamps,
            decode_clamps=request.decode_clamps,
        )
        # Push the freshly-registered module to every worker so requests
        # carrying ``SamplingParams.steering_module_ref`` resolve it
        # locally instead of forcing the API server to materialize the
        # full vector spec into the multiprocessing payload. Clamp tiers
        # are read back from the registry so the broadcast carries the
        # coerced (int layer keys) form it validated.
        registered = registry.get(request.name)
        engine = _engine_client(raw_request)
        await _broadcast_module_to_workers(
            engine,
            request.name,
            {
                "vectors": vectors,
                "prefill_vectors": prefill_vectors,
                "decode_vectors": decode_vectors,
                "clamps": registered.clamps if registered else None,
                "prefill_clamps": registered.prefill_clamps if registered else None,
                "decode_clamps": registered.decode_clamps if registered else None,
            },
        )
        # Eagerly upload the module's vectors to the manager so the
        # first request resolving to this name finds the (hash, phase)
        # row already in the refcount table — turning a ~15 ms
        # cold-path materialize (synchronous bf16 H2D for every layer)
        # into a ~5 µs refcount bump on its TTFT.  Strictly ordered
        # after the registry-update broadcast: pre-materialize reads
        # the resolved cache populated by ``register_steering_modules``.
        await _pre_materialize_module_on_workers(engine, request.name)
        return JSONResponse(
            content={
                "status": "ok",
                "name": request.name,
                "modules": registry.list_modules(),
            },
        )
    except (ValueError, TypeError) as err:
        return JSONResponse(
            content={"error": str(err)},
            status_code=HTTPStatus.BAD_REQUEST.value,
        )
    except Exception as err:
        logger.exception("Failed to register steering module '%s'", request.name)
        return JSONResponse(
            content={
                "error": f"Failed to register steering module: {err}",
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )


@router.post("/v1/steering/modules/unregister")
async def unregister_steering_module(
    request: UnregisterSteeringModuleRequest,
    raw_request: Request,
) -> JSONResponse:
    """Remove a named steering vector configuration."""
    if (unauthorized := _authorize_steering_mutation(raw_request)) is not None:
        return unauthorized
    registry = _get_registry(raw_request)
    if registry is None:
        return JSONResponse(
            content={
                "error": "Steering module registry not initialized.",
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    existed = await registry.unregister(request.name)
    if not existed:
        return JSONResponse(
            content={
                "error": (
                    f"Steering module '{request.name}' not found. "
                    f"Available: {registry.list_modules() or 'none'}"
                ),
            },
            status_code=HTTPStatus.NOT_FOUND.value,
        )
    # Drop the module on every worker to keep the broadcast registry
    # in lock-step with the server-side registry.  Workers will raise
    # on subsequent requests that reference this name.
    await _broadcast_module_to_workers(
        _engine_client(raw_request),
        request.name,
        None,
    )
    return JSONResponse(
        content={
            "status": "ok",
            "name": request.name,
            "modules": registry.list_modules(),
        },
    )


@router.get("/v1/steering/modules")
async def list_steering_modules(raw_request: Request) -> JSONResponse:
    """List all registered named steering modules."""
    registry = _get_registry(raw_request)
    if registry is None:
        return JSONResponse(
            content={
                "error": "Steering module registry not initialized.",
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    modules = registry.list_modules()
    return JSONResponse(
        content={
            "modules": modules,
            "count": len(modules),
        },
    )


def attach_router(app: FastAPI):
    app.include_router(router)
