# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admin endpoints for the named probe/steer vector registry.

The whole router is gated behind ``VLLM_SERVER_DEV_MODE`` (like the steering
module router). Unlike the module registry / ``/v1/steering/set``, the mutating
endpoints here are NOT additionally gated behind the steering API key:
registering a named vector grants no capability a client doesn't already have
(the identical probe/steer vectors can be passed inline, unauthenticated, in a
per-request declarative gate) — a name is pure server-side sugar over that open
path and is inert until a request references it. The registry is frontend-only
— no worker broadcast — so these handlers only touch
``app.state.steering_vector_registry``.
"""

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.entrypoints.serve.steering.vectors_protocol import (
    RegisterVectorRequest,
    UnregisterVectorRequest,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _get_registry(request: Request):
    return getattr(request.app.state, "steering_vector_registry", None)


def _registry_unavailable() -> JSONResponse:
    return JSONResponse(
        content={
            "error": "Steering vector registry not initialized. "
            "Ensure the server was started with --enable-steering."
        },
        status_code=HTTPStatus.BAD_REQUEST.value,
    )


@router.post("/v1/steering/vectors/register")
async def register_vector(
    request: RegisterVectorRequest, raw_request: Request
) -> JSONResponse:
    registry = _get_registry(raw_request)
    if registry is None:
        return _registry_unavailable()
    try:
        await registry.register(request.name, request.kind, request.packed)
    except (ValueError, TypeError) as err:
        return JSONResponse(
            content={"error": str(err)}, status_code=HTTPStatus.BAD_REQUEST.value
        )
    except Exception as err:  # noqa: BLE001
        logger.exception("Failed to register steering vector '%s'", request.name)
        return JSONResponse(
            content={"error": f"Failed to register steering vector: {err}"},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    return JSONResponse(
        content={
            "status": "ok",
            "name": request.name,
            "kind": request.kind,
            "vectors": registry.list_vectors(),
        }
    )


@router.post("/v1/steering/vectors/unregister")
async def unregister_vector(
    request: UnregisterVectorRequest, raw_request: Request
) -> JSONResponse:
    registry = _get_registry(raw_request)
    if registry is None:
        return _registry_unavailable()
    existed = await registry.unregister(request.name, request.kind)
    if not existed:
        return JSONResponse(
            content={
                "error": f"{request.kind} vector '{request.name}' not found.",
                "vectors": registry.list_vectors(),
            },
            status_code=HTTPStatus.NOT_FOUND.value,
        )
    return JSONResponse(
        content={
            "status": "ok",
            "name": request.name,
            "kind": request.kind,
            "vectors": registry.list_vectors(),
        }
    )


@router.get("/v1/steering/vectors")
async def list_vectors(raw_request: Request) -> JSONResponse:
    registry = _get_registry(raw_request)
    if registry is None:
        return _registry_unavailable()
    vectors = registry.list_vectors()
    return JSONResponse(content={"vectors": vectors, "count": registry.count()})


def attach_router(app: FastAPI) -> None:
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
