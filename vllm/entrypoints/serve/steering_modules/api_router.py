# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.entrypoints.serve.steering_modules.protocol import (
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


@router.post("/v1/steering/modules/register")
async def register_steering_module(
    request: RegisterSteeringModuleRequest,
    raw_request: Request,
) -> JSONResponse:
    """Register a named steering vector configuration."""
    registry = _get_registry(raw_request)
    if registry is None:
        return JSONResponse(
            content={
                "error": "Steering module registry not initialized. "
                "Ensure --enable-steering is set."
            },
            status_code=HTTPStatus.BAD_REQUEST.value,
        )

    try:
        await registry.register(
            name=request.name,
            vectors=request.vectors,
            prefill_vectors=request.prefill_vectors,
            decode_vectors=request.decode_vectors,
        )
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
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
