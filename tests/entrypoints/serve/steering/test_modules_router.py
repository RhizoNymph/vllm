# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the steering modules router using a mock engine and a
real :class:`SteeringModuleRegistry`.

Focused on the binary-wire packed input path on
``POST /v1/steering/modules/register``: the legacy JSON shape is covered
by ``tests/entrypoints/openai/test_steering_modules.py`` (which exercises
the registry directly).  The packed-shape coverage here proves the
``coerce_steering_spec`` seam in the router decodes the wire format
before the registry sees it, so the stored module is uniformly
legacy-shaped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from tests.entrypoints.serve.steering._packed_helpers import pack_hook
from vllm.entrypoints.openai.steering.registry import SteeringModuleRegistry
from vllm.entrypoints.serve.steering.modules_router import router
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT

_HP = DEFAULT_HOOK_POINT.value


def _make_app(engine_mock, registry: SteeringModuleRegistry) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = engine_mock
    app.state.steering_module_registry = registry
    app.include_router(router)
    return app


class _SyncASGIClient:
    def __init__(self, app: FastAPI):
        self._app = app

    def post(self, *args, **kwargs):
        return self._request("post", *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._request("get", *args, **kwargs)

    def _request(self, method: str, *args, **kwargs):
        async def _run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await getattr(client, method)(*args, **kwargs)

        return pytest.importorskip("anyio").run(_run)


@pytest.fixture
def engine():
    eng = AsyncMock()
    eng.collective_rpc.return_value = []
    return eng


@pytest.fixture
def registry():
    return SteeringModuleRegistry()


@pytest.fixture
def client(engine, registry):
    return _SyncASGIClient(_make_app(engine, registry))


class TestRegisterPacked:
    """``/v1/steering/modules/register`` accepts the packed wire shape."""

    def test_register_legacy_baseline(self, client, registry):
        """Legacy form keeps inner string layer keys as ints once routed
        through ``coerce_steering_spec`` — exact list contents preserved."""
        resp = client.post(
            "/v1/steering/modules/register",
            json={
                "name": "calm",
                "vectors": {_HP: {"5": [0.1, 0.2]}},
            },
        )
        assert resp.status_code == 200, resp.json()
        stored = registry.get("calm")
        assert stored is not None
        assert stored.vectors[_HP][5] == [0.1, 0.2]

    def test_register_packed_single_tier(self, client, registry):
        resp = client.post(
            "/v1/steering/modules/register",
            json={
                "name": "creativity",
                "vectors": {_HP: pack_hook({5: [0.1, 0.2], 6: [0.3, 0.4]})},
            },
        )
        assert resp.status_code == 200, resp.json()
        stored = registry.get("creativity")
        assert stored is not None
        # Coercion must yield plain-list form so dump_for_broadcast stays
        # JSON/pickle-friendly and _validate_layer_entry (list-only) passes.
        assert stored.vectors[_HP][5] == pytest.approx([0.1, 0.2], rel=1e-6)
        assert stored.vectors[_HP][6] == pytest.approx([0.3, 0.4], rel=1e-6)
        assert isinstance(stored.vectors[_HP][5], list)

    def test_register_packed_with_scales(self, client, registry):
        resp = client.post(
            "/v1/steering/modules/register",
            json={
                "name": "scaled",
                "vectors": {_HP: pack_hook({5: [1.0, 2.0]}, scales=[3.0])},
            },
        )
        assert resp.status_code == 200, resp.json()
        stored = registry.get("scaled")
        assert stored is not None
        assert stored.vectors[_HP][5] == pytest.approx([3.0, 6.0], rel=1e-6)

    def test_register_packed_mixed_tiers(self, client, registry):
        """Base packed + prefill legacy + decode None — each tier
        is normalized independently."""
        resp = client.post(
            "/v1/steering/modules/register",
            json={
                "name": "mixed",
                "vectors": {_HP: pack_hook({5: [0.1, 0.2]})},
                "prefill_vectors": {_HP: {"5": [0.3, 0.4]}},
            },
        )
        assert resp.status_code == 200, resp.json()
        stored = registry.get("mixed")
        assert stored is not None
        assert stored.vectors[_HP][5] == pytest.approx([0.1, 0.2], rel=1e-6)
        # Legacy list values: pass through unchanged so exact compare works.
        assert stored.prefill_vectors[_HP][5] == [0.3, 0.4]
        assert stored.decode_vectors is None

    def test_register_packed_malformed_returns_400(self, client, registry):
        """Wrong ``data`` length → 400; registry stays empty."""
        import pybase64 as base64

        bad = {
            "dtype": "float32",
            "shape": [1, 4],
            "layer_indices": [5],
            "data": base64.b64encode(b"\x00" * 8).decode("ascii"),
        }
        resp = client.post(
            "/v1/steering/modules/register",
            json={"name": "bad", "vectors": {_HP: bad}},
        )
        assert resp.status_code == 400
        assert "Malformed steering payload" in resp.json()["error"]
        assert registry.get("bad") is None

    def test_register_packed_broadcast_payload_uses_legacy_shape(
        self, client, engine, registry
    ):
        """The broadcast RPC must carry the normalized (list-shaped)
        spec, not the raw packed blob — workers serialize via cloudpickle
        and expect ``dict[int, list[float]]`` entries."""
        resp = client.post(
            "/v1/steering/modules/register",
            json={
                "name": "broadcast",
                "vectors": {_HP: pack_hook({5: [0.1, 0.2]})},
            },
        )
        assert resp.status_code == 200, resp.json()

        register_calls = [
            call
            for call in engine.collective_rpc.call_args_list
            if call.args and call.args[0] == "register_steering_modules"
        ]
        assert register_calls, "expected register_steering_modules RPC"
        kwargs = register_calls[0].kwargs.get("kwargs", {})
        modules = kwargs["modules"]
        broadcast_vec = modules["broadcast"]["vectors"][_HP][5]
        assert isinstance(broadcast_vec, list)
        assert broadcast_vec == pytest.approx([0.1, 0.2], rel=1e-6)


class TestModuleMutationAuth:
    """register/unregister are steering mutations: when steering API tokens
    are configured they require the bearer key, mirroring /v1/steering/set."""

    @pytest.fixture
    def keyed_client(self, engine, registry):
        app = _make_app(engine, registry)
        app.state.steering_api_tokens = ["sekrit"]
        return _SyncASGIClient(app)

    def _body(self):
        return {"name": "m", "vectors": {_HP: {"5": [0.1, 0.2]}}}

    def test_register_without_key_401(self, keyed_client):
        resp = keyed_client.post("/v1/steering/modules/register", json=self._body())
        assert resp.status_code == 401

    def test_register_wrong_key_401(self, keyed_client):
        resp = keyed_client.post(
            "/v1/steering/modules/register",
            json=self._body(),
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_register_with_key_ok(self, keyed_client, registry):
        resp = keyed_client.post(
            "/v1/steering/modules/register",
            json=self._body(),
            headers={"Authorization": "Bearer sekrit"},
        )
        assert resp.status_code == 200
        assert registry.get("m") is not None

    def test_unregister_without_key_401(self, keyed_client):
        resp = keyed_client.post(
            "/v1/steering/modules/unregister", json={"name": "m"}
        )
        assert resp.status_code == 401

    def test_no_tokens_configured_is_open(self, client, registry):
        resp = client.post("/v1/steering/modules/register", json=self._body())
        assert resp.status_code == 200
        assert registry.get("m") is not None
