# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

import vllm.envs as envs
from vllm.config.sae_steering_types import SAEActivation, SteeringModuleKind
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.steering.registry import SAEModuleManifest
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

    *payload* of ``None`` removes the module on workers.
    """
    if engine is None:
        return
    if payload is None:
        await engine.collective_rpc(
            "unregister_steering_modules",
            kwargs=dict(names=[name]),
        )
    else:
        await engine.collective_rpc(
            "register_steering_modules",
            kwargs=dict(modules={name: payload}, replace=False),
        )


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
        kind = SteeringModuleKind(request.kind)
        if kind is SteeringModuleKind.ADDITIVE:
            await registry.register(
                name=request.name,
                kind=kind,
                vectors=request.vectors,
                prefill_vectors=request.prefill_vectors,
                decode_vectors=request.decode_vectors,
            )
            payload: dict = {
                "kind": kind.value,
                "vectors": request.vectors,
                "prefill_vectors": request.prefill_vectors,
                "decode_vectors": request.decode_vectors,
            }
        else:  # SAE_DELTA
            if request.sae_manifest is None:
                raise ValueError("kind='sae_delta' requires a 'sae_manifest' payload.")
            manifest = SAEModuleManifest(
                d_model=request.sae_manifest.d_model,
                d_sae=request.sae_manifest.d_sae,
                activation=SAEActivation(request.sae_manifest.activation),
                layers=tuple((li, hp) for li, hp in request.sae_manifest.layers),
                clampable_features=tuple(
                    sorted(set(request.sae_manifest.clampable_features))
                ),
                activation_params=dict(request.sae_manifest.activation_params),
                weights_uri=request.sae_manifest.weights_uri,
            )
            # Forward the additive vector fields too, even though the
            # SAE branch must reject them: ``SteeringModuleRegistry.register``
            # raises a 400-shaped ValueError when an SAE registration
            # carries any additive payload, so a mixed-kind request
            # surfaces as a clear client error instead of being
            # silently discarded at the router.
            await registry.register(
                name=request.name,
                kind=kind,
                vectors=request.vectors,
                prefill_vectors=request.prefill_vectors,
                decode_vectors=request.decode_vectors,
                sae_manifest=manifest,
            )
            payload = {
                "kind": kind.value,
                "sae_manifest": {
                    "d_model": manifest.d_model,
                    "d_sae": manifest.d_sae,
                    "activation": manifest.activation.value,
                    "layers": [list(p) for p in manifest.layers],
                    "clampable_features": list(manifest.clampable_features),
                    "activation_params": dict(manifest.activation_params),
                    "weights_uri": manifest.weights_uri,
                },
            }
        # Push the freshly-registered module to every worker so requests
        # carrying ``SamplingParams.steering_module_ref`` (additive) or
        # ``SamplingParams.sae_clamp_specs`` (sae_delta) resolve names
        # locally without crossing the multiprocessing boundary with
        # the full payload.
        await _broadcast_module_to_workers(
            _engine_client(raw_request),
            request.name,
            payload,
        )
        return JSONResponse(
            content={
                "status": "ok",
                "name": request.name,
                "kind": kind.value,
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
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
