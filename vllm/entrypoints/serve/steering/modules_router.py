# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from http import HTTPStatus
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.config.sae_steering_types import SAEActivation, SteeringModuleKind
from vllm.config.steering_types import coerce_steering_spec
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.steering.registry import (
    SAEModuleManifest,
    SteeringModule,
    pack_sae_weights_for_broadcast,
)
from vllm.entrypoints.openai.steering.sae_loader import (
    _load_weights_for_manifest,
)
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


def _build_broadcast_payload_for_module(module: SteeringModule) -> dict:
    """Reconstruct the broadcast payload for an in-registry module.

    Mirrors the inline payload built by :func:`register_steering_module`
    so the compensating-broadcast path can re-install a previously-
    working module after a failed replacement.  For SAE modules, the
    encoder/decoder tensors are re-loaded from ``manifest.weights_uri``
    — the registry stores only the manifest, not the loaded weights.
    """
    if module.kind is SteeringModuleKind.ADDITIVE:
        return {
            "kind": module.kind.value,
            "vectors": module.vectors,
            "prefill_vectors": module.prefill_vectors,
            "decode_vectors": module.decode_vectors,
        }
    assert module.sae_manifest is not None
    manifest = module.sae_manifest
    if not manifest.weights_uri:
        raise ValueError(
            f"Cannot rebuild compensating payload for SAE module "
            f"{module.name!r}: manifest has no 'weights_uri' to re-load "
            "weights from."
        )
    weights = _load_weights_for_manifest(manifest, Path(manifest.weights_uri))
    return {
        "kind": module.kind.value,
        "sae_manifest": {
            "d_model": manifest.d_model,
            "d_sae": manifest.d_sae,
            "activation": manifest.activation.value,
            "layers": [list(p) for p in manifest.layers],
            "clampable_features": list(manifest.clampable_features),
            "activation_params": dict(manifest.activation_params),
            "weights_uri": manifest.weights_uri,
        },
        "sae_weights": pack_sae_weights_for_broadcast(weights),
    }


async def _compensating_broadcast_after_failure(
    engine: EngineClient | None,
    name: str,
    prev_module: SteeringModule | None,
) -> None:
    """Best-effort cluster repair after a failed register broadcast.

    ``collective_rpc`` is not transactional — when a multi-worker
    broadcast raises, some ranks may have already accepted the new
    module while others rolled back to their prior state.  After the
    server-side registry has been rolled back, this helper attempts to
    realign every worker by sending a second broadcast that re-installs
    *prev_module* (or unregisters the name when there was no prior
    entry).

    This is strictly best-effort: if the compensating broadcast itself
    fails, the cluster may remain inconsistent and we log a warning.
    A proper fix would be a worker-side prepare/commit protocol; that
    is tracked separately.  We never raise from this helper so the
    caller's original exception always reaches the client.
    """
    if engine is None:
        return
    try:
        if prev_module is None:
            await _broadcast_module_to_workers(engine, name, None)
        else:
            restore_payload = _build_broadcast_payload_for_module(prev_module)
            await _broadcast_module_to_workers(engine, name, restore_payload)
    except Exception:
        logger.warning(
            "Compensating broadcast for steering module %r failed; "
            "cluster state may be inconsistent across worker ranks.  "
            "Manually unregister and re-register the module to "
            "resynchronise.",
            name,
            exc_info=True,
        )


async def _reset_prefix_cache_after_module_change(
    engine: EngineClient | None,
    *,
    action: str,
) -> bool:
    """Invalidate KV blocks whose steering hash only names a module.

    Request hashes include named-module references by name/scale, not by
    the current vector payload or SAE weights.  Replacing a module under
    the same name can therefore make old prefix-cache blocks appear
    reusable unless we invalidate the cache after the worker registry
    changes.
    """
    if engine is None:
        return True
    success = await engine.reset_prefix_cache(reset_running_requests=True)
    if not success:
        logger.error(
            "Prefix cache reset failed after steering module %s; cached "
            "KV blocks may still reflect the previous module payload.",
            action,
        )
        return False
    return True


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

    # Snapshot the pre-call entry so a failed broadcast can either
    # restore it (when re-registering an existing name) or remove the
    # newly-created one — never destroy a previously-working module
    # because its replacement failed.
    prev_module = registry.get(request.name)
    try:
        kind = SteeringModuleKind(request.kind)
        if kind is SteeringModuleKind.ADDITIVE:
            if request.sae_manifest is not None:
                raise ValueError(
                    f"Steering module {request.name!r}: sae_manifest is "
                    "not valid for kind='additive'."
                )
            # Each tier may arrive as either the legacy SteeringVectorSpec or
            # the binary-wire SteeringVectorSpecPacked shape; normalize before
            # handing off so the registry, the broadcast payload, and the
            # pre-materialize path all see the same legacy-shaped dict.
            try:
                vectors = coerce_steering_spec(request.vectors)
                prefill_vectors = coerce_steering_spec(request.prefill_vectors)
                decode_vectors = coerce_steering_spec(request.decode_vectors)
            except ValueError as err:
                raise ValueError(f"Malformed steering payload: {err}") from err
            await registry.register(
                name=request.name,
                kind=kind,
                vectors=vectors,
                prefill_vectors=prefill_vectors,
                decode_vectors=decode_vectors,
            )
            payload: dict = {
                "kind": kind.value,
                "vectors": vectors,
                "prefill_vectors": prefill_vectors,
                "decode_vectors": decode_vectors,
            }
        else:  # SAE_DELTA
            if request.sae_manifest is None:
                raise ValueError("kind='sae_delta' requires a 'sae_manifest' payload.")
            if (
                request.vectors is not None
                or request.prefill_vectors is not None
                or request.decode_vectors is not None
            ):
                raise ValueError(
                    f"Steering module {request.name!r}: additive vector fields "
                    "are not valid for kind='sae_delta'."
                )
            # ``clampable_features`` order is significant — the safetensors
            # loader aligns each weight row to ``manifest.clampable_features[i]``,
            # so reordering here would relabel decoder directions and silently
            # clamp the wrong features.  Reject duplicates without reordering.
            clampable = tuple(request.sae_manifest.clampable_features)
            if len(set(clampable)) != len(clampable):
                raise ValueError(
                    f"Steering module {request.name!r}: "
                    "sae_manifest.clampable_features must not contain "
                    f"duplicates; got {list(clampable)}."
                )
            manifest = SAEModuleManifest(
                d_model=request.sae_manifest.d_model,
                d_sae=request.sae_manifest.d_sae,
                activation=SAEActivation(request.sae_manifest.activation),
                layers=tuple((li, hp) for li, hp in request.sae_manifest.layers),
                clampable_features=clampable,
                activation_params=dict(request.sae_manifest.activation_params),
                weights_uri=request.sae_manifest.weights_uri,
            )
            # Validate shape/site invariants before touching checkpoint
            # files.  Registry.register repeats this check when it commits,
            # but doing it here keeps malformed manifests from triggering
            # expensive SAE weight I/O.
            registry._validate_sae_manifest(name=request.name, manifest=manifest)
            if not manifest.weights_uri:
                # Without a weights_uri the worker would attach
                # zero-filled encoder/decoder buffers and every clamp
                # would silently no-op.  Fail fast at the API boundary
                # so callers see a 400 instead of an opaque runtime
                # mis-behaviour.
                raise ValueError(
                    f"Steering module {request.name!r}: kind='sae_delta' "
                    "requires 'sae_manifest.weights_uri' to point to a "
                    "local SAE checkpoint directory containing per-(layer, "
                    "hook) safetensors files.  Without weights the worker "
                    "buffers would stay zero-filled and every clamp would "
                    "no-op."
                )
            # Load weights synchronously off the event loop so a large
            # SAE checkpoint doesn't block other API traffic.  Use the
            # caller-provided manifest as the source of truth for shapes;
            # this lets the loader read only the weight files (the disk
            # ``manifest.json`` is irrelevant on this path).
            try:
                weights = await asyncio.to_thread(
                    _load_weights_for_manifest,
                    manifest,
                    Path(manifest.weights_uri),
                )
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"Steering module {request.name!r}: failed to load "
                    f"SAE weights from {manifest.weights_uri!r}: {exc}"
                ) from exc
            await registry.register(
                name=request.name,
                kind=kind,
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
                # Weights ride along with the manifest so the worker
                # registers the module and attaches its encoder/decoder
                # tensors in one indivisible RPC.  Without this, a
                # successful manifest broadcast followed by an attach
                # failure would leave the worker with a registered SAE
                # module whose buffers are still zero-filled — the
                # silent-no-op failure mode this endpoint exists to
                # prevent.  Packed to the wire-safe form: raw tensors
                # do not survive the collective_rpc msgpack hop.
                "sae_weights": pack_sae_weights_for_broadcast(weights),
            }
        # Push the freshly-registered module to every worker so requests
        # carrying ``SamplingParams.steering_module_ref`` (additive) or
        # ``SamplingParams.sae_clamp_specs`` (sae_delta) resolve names
        # locally without crossing the multiprocessing boundary with
        # the full payload.  If the broadcast raises, restore the
        # pre-call entry so a failed replacement does not destroy the
        # previously-working module (and a failed first-time
        # registration removes the name entirely).  ``collective_rpc``
        # is not transactional — some workers may have committed the
        # new state before the failing rank raised — so we follow up
        # with a compensating broadcast that re-installs the prior
        # state (or unregisters the name) on every rank.
        engine = _engine_client(raw_request)
        try:
            await _broadcast_module_to_workers(
                engine,
                request.name,
                payload,
            )
        except Exception:
            await registry.restore_or_remove(request.name, prev_module)
            await _compensating_broadcast_after_failure(
                engine, request.name, prev_module
            )
            raise
        # Eagerly upload the module's vectors to the manager so the
        # first request resolving to this name finds the (hash, phase)
        # row already in the refcount table — turning a ~15 ms cold-path
        # materialize into a ~5 µs refcount bump on its TTFT.  Only
        # additive modules have precomputed rows to pre-materialize;
        # SAE modules attach their encoder/decoder buffers as part of
        # the register broadcast above.  Strictly ordered after the
        # registry-update broadcast: pre-materialize reads the resolved
        # cache populated by ``register_steering_modules``.
        if kind is SteeringModuleKind.ADDITIVE:
            await _pre_materialize_module_on_workers(engine, request.name)
        if not await _reset_prefix_cache_after_module_change(
            engine,
            action=f"registration for {request.name!r}",
        ):
            return JSONResponse(
                content={
                    "error": (
                        "Steering module was registered but prefix cache "
                        "could not be fully invalidated. Retry the request "
                        "or reset the prefix cache before generating with "
                        "this module."
                    )
                },
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
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

    prev_module = registry.get(request.name)
    if prev_module is None:
        return JSONResponse(
            content={
                "error": (
                    f"Steering module '{request.name}' not found. "
                    f"Available: {registry.list_modules() or 'none'}"
                ),
            },
            status_code=HTTPStatus.NOT_FOUND.value,
        )

    await registry.unregister(request.name)
    # Drop the module on every worker to keep the broadcast registry
    # in lock-step with the server-side registry.  If the worker RPC
    # fails, restore the server entry and best-effort re-register it
    # on every rank; otherwise a failed unregister could leave the
    # API server believing the name is gone while some workers still
    # retain it (or vice versa after a partial collective failure).
    engine = _engine_client(raw_request)
    try:
        await _broadcast_module_to_workers(
            engine,
            request.name,
            None,
        )
    except Exception as err:
        await registry.restore_or_remove(request.name, prev_module)
        await _compensating_broadcast_after_failure(
            engine, request.name, prev_module
        )
        logger.exception("Failed to unregister steering module '%s'", request.name)
        return JSONResponse(
            content={
                "error": f"Failed to unregister steering module: {err}",
            },
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        )
    if not await _reset_prefix_cache_after_module_change(
        engine,
        action=f"unregister for {request.name!r}",
    ):
        return JSONResponse(
            content={
                "error": (
                    "Steering module was unregistered but prefix cache "
                    "could not be fully invalidated. Retry the request "
                    "or reset the prefix cache before continuing."
                )
            },
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
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
