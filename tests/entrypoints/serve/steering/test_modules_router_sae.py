# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SAE branch of the steering modules router.

Focuses on the SAE-specific behavior added to
:func:`vllm.entrypoints.serve.steering.modules_router.register_steering_module`:
rejecting registrations without ``weights_uri`` (since the worker would
attach zero-filled buffers) and loading + broadcasting weights when the
URI points to a valid on-disk layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import torch
from fastapi import FastAPI
from safetensors.torch import save_file

import vllm.entrypoints.serve.steering.modules_router as modules_router
from vllm.entrypoints.openai.steering.registry import SteeringModuleRegistry
from vllm.entrypoints.serve.steering.modules_router import router


@pytest.fixture(autouse=True)
def inline_to_thread(monkeypatch):
    """Keep the sync ASGI harness deterministic.

    The production endpoint offloads disk reads with ``asyncio.to_thread``.
    These unit tests assert router state/RPC behavior, not threadpool
    scheduling, so run the callable inline to avoid per-request event-loop
    threadpool lifecycle noise in the synchronous test client.
    """

    async def _run_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(modules_router.asyncio, "to_thread", _run_inline)


def _make_app(engine_mock, registry) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = engine_mock
    app.state.steering_module_registry = registry
    app.include_router(router)
    return app


class _SyncASGIClient:
    def __init__(self, app: FastAPI):
        self._app = app

    def post(self, *args, **kwargs):
        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(*args, **kwargs)

        return pytest.importorskip("anyio").run(_run)


@pytest.fixture
def engine():
    mock = AsyncMock()
    mock.reset_prefix_cache.return_value = True
    return mock


@pytest.fixture
def registry():
    return SteeringModuleRegistry()


@pytest.fixture
def client(engine, registry):
    return _SyncASGIClient(_make_app(engine, registry))


def _sae_dir(
    tmp_path: Path,
    *,
    d_model: int = 4,
    clampable: tuple[int, ...] = (0, 1),
    layers: tuple[tuple[int, str], ...] = ((0, "post_block"),),
) -> Path:
    """Write the per-(layer, hook) safetensors files an SAE manifest needs."""
    n_clamp = len(clampable)
    for layer_idx, hook_str in layers:
        save_file(
            {
                "encoder_weight": torch.ones(n_clamp, d_model),
                "encoder_bias": torch.zeros(n_clamp),
                "decoder_weight": torch.ones(n_clamp, d_model),
            },
            str(tmp_path / f"layer_{layer_idx}_{hook_str}.safetensors"),
        )
    # The router uses the request manifest as authoritative, but write a
    # manifest.json anyway so the directory matches the canonical layout.
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "d_model": d_model,
                "d_sae": 8,
                "activation": "relu",
                "layers": [list(p) for p in layers],
                "clampable_features": list(clampable),
                "activation_params": {},
                "weights_uri": None,
            }
        )
    )
    return tmp_path


def _sae_request_body(
    *,
    name: str = "g",
    weights_uri: str | None,
    d_model: int = 4,
    clampable: tuple[int, ...] = (0, 1),
    layers: tuple[tuple[int, str], ...] = ((0, "post_block"),),
) -> dict:
    return {
        "name": name,
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": 8,
            "activation": "relu",
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable),
            "activation_params": {},
            "weights_uri": weights_uri,
        },
    }


class TestSaeRegistrationRejectsMissingWeightsUri:
    """Without ``weights_uri`` the worker would attach zero-filled
    buffers and every clamp would silently no-op — the router must
    fail loudly at the API boundary."""

    def test_missing_weights_uri_returns_400(self, client, engine, registry):
        body = _sae_request_body(weights_uri=None)
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 400
        assert "weights_uri" in resp.json()["error"]
        # No broadcast attempted.
        engine.collective_rpc.assert_not_called()
        # Registry untouched.
        assert registry.list_modules() == []


class TestSaeManifestRequestValidation:
    def test_bool_integer_fields_are_rejected(self, client, engine, tmp_path):
        sae_dir = _sae_dir(tmp_path)
        cases = [
            ("d_model", True),
            ("d_sae", True),
            ("layers", [[True, "post_block"]]),
            ("clampable_features", [True]),
            ("activation_params", {"threshold": True}),
        ]
        for field_name, value in cases:
            body = _sae_request_body(weights_uri=str(sae_dir))
            body["sae_manifest"][field_name] = value
            resp = client.post("/v1/steering/modules/register", json=body)
            assert resp.status_code == 422, resp.text

        engine.collective_rpc.assert_not_called()


class TestSaeRegistrationLoadsAndBroadcastsWeights:
    """``weights_uri`` set to a valid SAE directory triggers an on-disk
    weight load.  Workers receive the manifest *and* the weights in a
    single ``register_steering_modules`` RPC so registration is atomic
    on the worker — there is no window where the buffers are zero but
    the module is registered."""

    def test_register_emits_one_atomic_rpc_with_weights_inline(
        self, client, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        body = _sae_request_body(weights_uri=str(sae_dir))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 200, resp.text
        assert registry.list_modules() == ["g"]

        calls = engine.collective_rpc.await_args_list
        assert len(calls) == 1
        assert calls[0].args[0] == "register_steering_modules"
        modules = calls[0].kwargs["kwargs"]["modules"]
        assert set(modules) == {"g"}
        payload = modules["g"]
        assert payload["kind"] == "sae_delta"
        # Weights ride along with the manifest so the worker register-
        # and-attach happens in one indivisible step, in the packed wire
        # form (raw tensors do not survive the collective_rpc hop).
        weights = payload["sae_weights"]
        assert set(weights.keys()) == {"0:post_block"}
        for tensors in weights.values():
            assert set(tensors.keys()) == {
                "encoder_weight",
                "encoder_bias",
                "decoder_weight",
            }
            for packed in tensors.values():
                assert set(packed.keys()) == {"dtype", "shape", "data"}
                assert isinstance(packed["data"], bytes)
        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )

    def test_bad_weights_uri_returns_400(self, client, engine, registry, tmp_path):
        body = _sae_request_body(weights_uri=str(tmp_path / "does-not-exist"))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 400
        # Failure happens before any RPC.
        engine.collective_rpc.assert_not_called()
        engine.reset_prefix_cache.assert_not_called()

    def test_prefix_cache_reset_failure_returns_503(
        self, client, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        engine.reset_prefix_cache.return_value = False
        body = _sae_request_body(weights_uri=str(sae_dir))

        resp = client.post("/v1/steering/modules/register", json=body)

        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"]
        assert registry.list_modules() == ["g"]
        engine.collective_rpc.assert_awaited_once()
        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )


class TestSaeRegistrationRollsBackRegistryOnBroadcastFailure:
    """If the single worker RPC fails, the server-side registry must
    not be left holding a phantom entry that has no presence on the
    workers — that would let a follow-up request resolve the name and
    then crash (or silently no-op)."""

    def test_first_time_registration_removes_name_on_rpc_failure(
        self, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        engine.collective_rpc.side_effect = RuntimeError("worker exploded")
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 500
        # No prior entry, so the rollback removes the name entirely.
        assert registry.list_modules() == []

    def test_replacement_preserves_previous_module_on_rpc_failure(
        self, engine, registry, tmp_path
    ):
        """The headline bug: a failed re-registration must not destroy
        the existing valid module."""
        sae_dir = _sae_dir(tmp_path)
        client = _SyncASGIClient(_make_app(engine, registry))
        # First call succeeds and seeds the registry.
        first_body = _sae_request_body(weights_uri=str(sae_dir))
        ok = client.post("/v1/steering/modules/register", json=first_body)
        assert ok.status_code == 200, ok.text
        assert registry.list_modules() == ["g"]
        good_module = registry.get("g")
        assert good_module is not None

        # Second call (re-registering the same name) fails at broadcast.
        engine.collective_rpc.side_effect = RuntimeError("worker exploded")
        resp = client.post("/v1/steering/modules/register", json=first_body)
        assert resp.status_code == 500
        # Prior module must still be present and unchanged.
        assert registry.list_modules() == ["g"]
        assert registry.get("g") is good_module


class TestClampableFeaturesOrderPreserved:
    """The safetensors loader aligns each weight row to
    ``manifest.clampable_features[i]``, so reordering the caller's
    list at the router would silently relabel decoder directions and
    apply clamps to the wrong features."""

    def test_non_sorted_order_round_trips_into_manifest(
        self, client, engine, registry, tmp_path
    ):
        clampable = (5, 2)  # deliberately non-sorted
        sae_dir = _sae_dir(tmp_path, clampable=clampable)
        body = _sae_request_body(weights_uri=str(sae_dir), clampable=clampable)
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 200, resp.text
        # Manifest in registry keeps the caller's order verbatim.
        module = registry.get("g")
        assert module is not None
        assert module.sae_manifest is not None
        assert module.sae_manifest.clampable_features == (5, 2)
        # Broadcast payload mirrors the same order — the worker relies on
        # this to align each loaded tensor row to the right feature.
        calls = engine.collective_rpc.await_args_list
        payload = calls[0].kwargs["kwargs"]["modules"]["g"]
        assert payload["sae_manifest"]["clampable_features"] == [5, 2]

    def test_duplicate_features_rejected(self, client, engine, registry, tmp_path):
        sae_dir = _sae_dir(tmp_path)
        body = _sae_request_body(weights_uri=str(sae_dir), clampable=(0, 1, 0))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 400
        assert "duplicate" in resp.json()["error"].lower()
        engine.collective_rpc.assert_not_called()

    def test_sae_registration_with_additive_fields_rejects_before_loading(
        self,
        client,
        engine,
        registry,
        tmp_path,
        monkeypatch,
    ):
        sae_dir = _sae_dir(tmp_path)

        def fail_load(*args, **kwargs):
            raise AssertionError("SAE weights should not be loaded")

        monkeypatch.setattr(modules_router, "_load_weights_for_manifest", fail_load)
        body = _sae_request_body(weights_uri=str(sae_dir))
        body["vectors"] = {"post_block": {"0": [0.1, 0.2, 0.3, 0.4]}}

        resp = client.post("/v1/steering/modules/register", json=body)

        assert resp.status_code == 400
        assert "additive vector fields" in resp.json()["error"]
        assert registry.get("g") is None
        engine.collective_rpc.assert_not_called()

    def test_sae_registration_with_empty_additive_fields_rejects_before_loading(
        self,
        client,
        engine,
        registry,
        tmp_path,
        monkeypatch,
    ):
        sae_dir = _sae_dir(tmp_path)

        def fail_load(*args, **kwargs):
            raise AssertionError("SAE weights should not be loaded")

        monkeypatch.setattr(modules_router, "_load_weights_for_manifest", fail_load)
        body = _sae_request_body(weights_uri=str(sae_dir))
        body["vectors"] = {}

        resp = client.post("/v1/steering/modules/register", json=body)

        assert resp.status_code == 400
        assert "additive vector fields" in resp.json()["error"]
        assert registry.get("g") is None
        engine.collective_rpc.assert_not_called()

    def test_additive_registration_with_sae_manifest_rejects(
        self,
        client,
        engine,
        registry,
        tmp_path,
    ):
        sae_dir = _sae_dir(tmp_path)
        body = _sae_request_body(weights_uri=str(sae_dir))
        body["kind"] = "additive"
        body["vectors"] = {"post_block": {"0": [0.1, 0.2, 0.3, 0.4]}}

        resp = client.post("/v1/steering/modules/register", json=body)

        assert resp.status_code == 400
        assert "sae_manifest" in resp.json()["error"]
        assert registry.get("g") is None
        engine.collective_rpc.assert_not_called()

    def test_invalid_sae_manifest_rejects_before_loading(
        self,
        client,
        engine,
        registry,
        tmp_path,
        monkeypatch,
    ):
        sae_dir = _sae_dir(tmp_path)

        def fail_load(*args, **kwargs):
            raise AssertionError("SAE weights should not be loaded")

        monkeypatch.setattr(modules_router, "_load_weights_for_manifest", fail_load)
        body = _sae_request_body(weights_uri=str(sae_dir))
        body["sae_manifest"]["layers"] = [[0, "post_block"], [0, "post_block"]]

        resp = client.post("/v1/steering/modules/register", json=body)

        assert resp.status_code == 400
        assert "duplicate" in resp.json()["error"].lower()
        assert registry.get("g") is None
        engine.collective_rpc.assert_not_called()


class TestCompensatingBroadcastOnPartialFailure:
    """``collective_rpc`` is not transactional — when the failing-rank
    raise reaches the router, other ranks may already hold the new
    module.  The router must fire a compensating broadcast that
    re-installs the prior state on every worker, or unregisters the
    name when no prior state existed."""

    def test_first_time_failure_emits_compensating_unregister(
        self, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        # Fail the first RPC (the register attempt) and let the
        # compensating broadcast (pin release + unregister) succeed.
        engine.collective_rpc.side_effect = [
            RuntimeError("worker exploded"),
            None,
            None,
        ]
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 500

        calls = engine.collective_rpc.await_args_list
        assert len(calls) == 3
        assert calls[0].args[0] == "register_steering_modules"
        # Compensating: no prior entry, so drop any pre-materialize pin
        # (a no-op for SAE modules) and unregister on every rank.
        assert calls[1].args[0] == "release_pre_materialized_steering_module"
        assert calls[1].kwargs["kwargs"] == {"name": "g"}
        assert calls[2].args[0] == "unregister_steering_modules"
        assert calls[2].kwargs["kwargs"] == {"names": ["g"]}
        assert registry.list_modules() == []

    def test_replacement_failure_emits_compensating_reregister(
        self, engine, registry, tmp_path
    ):
        """After a failed replacement, the compensating broadcast must
        re-send the prior module's payload — including its SAE weights —
        so partially-committed workers fall back to the previous good
        state."""
        sae_dir = _sae_dir(tmp_path)
        client = _SyncASGIClient(_make_app(engine, registry))
        # Seed the registry with a working SAE.
        body = _sae_request_body(weights_uri=str(sae_dir))
        ok = client.post("/v1/steering/modules/register", json=body)
        assert ok.status_code == 200

        # Re-register the same name; this time the register-broadcast
        # raises but the compensating broadcast succeeds.
        engine.collective_rpc.side_effect = [
            RuntimeError("worker exploded"),
            None,
        ]
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 500

        # First call after seeding was the failing register; second was
        # the compensating broadcast that re-pushes the prior payload.
        calls = engine.collective_rpc.await_args_list
        # First seed call + 2 calls from the failing attempt.
        assert len(calls) == 3
        assert calls[1].args[0] == "register_steering_modules"  # failing
        assert calls[2].args[0] == "register_steering_modules"  # compensating
        comp_modules = calls[2].kwargs["kwargs"]["modules"]
        assert set(comp_modules) == {"g"}
        comp_payload = comp_modules["g"]
        assert comp_payload["kind"] == "sae_delta"
        # Compensating payload reloads the prior weights from disk —
        # without this, partially-committed workers could be left on
        # a half-attached new module.
        assert "sae_weights" in comp_payload
        assert set(comp_payload["sae_weights"].keys()) == {"0:post_block"}

    def test_compensating_broadcast_failure_is_swallowed(
        self, engine, registry, tmp_path
    ):
        """If the compensating broadcast itself fails, the original
        500 must still reach the caller — we don't want the second
        failure to mask the first."""
        sae_dir = _sae_dir(tmp_path)
        engine.collective_rpc.side_effect = RuntimeError("everything explodes")
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        resp = client.post("/v1/steering/modules/register", json=body)
        assert resp.status_code == 500
        # Two attempted RPCs (register + compensating); both raised on
        # the mock.  Registry rollback still happened.
        assert len(engine.collective_rpc.await_args_list) == 2
        assert registry.list_modules() == []


class TestUnregisterRollbackOnPartialFailure:
    """Unregister must be transactional with the worker broadcast too.

    The register path already rolls back and emits a compensating RPC
    on broadcast failure.  Unregister needs the same treatment because
    ``collective_rpc`` can fail after some ranks have removed the
    module while others have not.
    """

    def test_unregister_failure_restores_registry_and_compensates(
        self, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        ok = client.post("/v1/steering/modules/register", json=body)
        assert ok.status_code == 200, ok.text
        prior_module = registry.get("g")
        assert prior_module is not None

        # Removal broadcasts drop the pre-materialize pin first (a no-op
        # for SAE modules), then unregister; fail the unregister itself.
        engine.collective_rpc.side_effect = [
            None,
            RuntimeError("worker exploded"),
            None,
        ]
        resp = client.post("/v1/steering/modules/unregister", json={"name": "g"})
        assert resp.status_code == 500
        assert registry.get("g") is prior_module

        calls = engine.collective_rpc.await_args_list
        # First call is the successful seed registration.  The failing
        # unregister (after its pin release) is followed by a
        # compensating re-register.
        assert len(calls) == 4
        assert calls[1].args[0] == "release_pre_materialized_steering_module"
        assert calls[1].kwargs["kwargs"] == {"name": "g"}
        assert calls[2].args[0] == "unregister_steering_modules"
        assert calls[2].kwargs["kwargs"] == {"names": ["g"]}
        assert calls[3].args[0] == "register_steering_modules"
        comp_modules = calls[3].kwargs["kwargs"]["modules"]
        assert set(comp_modules) == {"g"}
        comp_payload = comp_modules["g"]
        assert comp_payload["kind"] == "sae_delta"
        assert "sae_weights" in comp_payload
        assert set(comp_payload["sae_weights"].keys()) == {"0:post_block"}

    def test_successful_unregister_resets_prefix_cache(
        self, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        ok = client.post("/v1/steering/modules/register", json=body)
        assert ok.status_code == 200, ok.text
        engine.reset_prefix_cache.reset_mock()

        resp = client.post("/v1/steering/modules/unregister", json={"name": "g"})

        assert resp.status_code == 200, resp.text
        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )

    def test_unregister_prefix_cache_reset_failure_returns_503(
        self, engine, registry, tmp_path
    ):
        sae_dir = _sae_dir(tmp_path)
        client = _SyncASGIClient(_make_app(engine, registry))
        body = _sae_request_body(weights_uri=str(sae_dir))
        ok = client.post("/v1/steering/modules/register", json=body)
        assert ok.status_code == 200, ok.text
        engine.reset_prefix_cache.reset_mock()
        engine.reset_prefix_cache.return_value = False

        resp = client.post("/v1/steering/modules/unregister", json={"name": "g"})

        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"]
        assert registry.get("g") is None
        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )
