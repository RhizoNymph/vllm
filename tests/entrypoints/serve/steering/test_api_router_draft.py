# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``target`` field on steering set/clear endpoints.

PR 4 introduces optional ``target: "main" | "draft" | null`` on
``/v1/steering/set`` and ``/v1/steering/clear``. Draft-model steering
is not yet wired up on the worker side; the router returns HTTP 501
when ``target="draft"``. Main (and omitted ``target``) behave as
before.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.serve.steering.api_router import router
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT

_HP = DEFAULT_HOOK_POINT.value


def _make_app(engine_mock) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = engine_mock
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
    return AsyncMock()


@pytest.fixture
def client(engine):
    return _SyncASGIClient(_make_app(engine))


def _vecs(layer_vecs: dict, hook: str = _HP) -> dict:
    return {"vectors": {hook: layer_vecs}}


class TestSetTarget:
    def test_target_main_is_accepted(self, client, engine):
        engine.collective_rpc.side_effect = [
            [(0, 0, [0])],
            [(0, 0, [0])],
        ]
        engine.reset_prefix_cache.return_value = True
        body = {**_vecs({0: [1.0]}), "target": "main"}
        resp = client.post("/v1/steering/set", json=body)
        assert resp.status_code == 200
        # Both validate and apply RPCs receive target="main".
        for call in engine.collective_rpc.await_args_list:
            assert call.kwargs["kwargs"]["target"] == "main"

    def test_target_null_passes_through_as_none(self, client, engine):
        engine.collective_rpc.side_effect = [
            [(0, 0, [0])],
            [(0, 0, [0])],
        ]
        engine.reset_prefix_cache.return_value = True
        # No ``target`` in body -> None server-side -> today's behavior.
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 200
        for call in engine.collective_rpc.await_args_list:
            assert call.kwargs["kwargs"]["target"] is None

    def test_target_draft_returns_501(self, client, engine):
        body = {**_vecs({0: [1.0]}), "target": "draft"}
        resp = client.post("/v1/steering/set", json=body)
        assert resp.status_code == 501
        body_json = resp.json()
        assert "not yet implemented" in body_json["error"]
        # No RPC fired.
        engine.collective_rpc.assert_not_called()

    def test_target_invalid_returns_422(self, client, engine):
        """Pydantic rejects values outside the Literal at parse time."""
        body = {**_vecs({0: [1.0]}), "target": "nonsense"}
        resp = client.post("/v1/steering/set", json=body)
        assert resp.status_code == 422


class TestClearTarget:
    def test_clear_no_body_is_main(self, client, engine):
        engine.collective_rpc.return_value = None
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 200
        engine.collective_rpc.assert_awaited_once_with(
            "clear_steering_vectors",
            args=(),
            kwargs={"target": None},
        )

    def test_clear_target_main_is_accepted(self, client, engine):
        engine.collective_rpc.return_value = None
        resp = client.post("/v1/steering/clear", json={"target": "main"})
        assert resp.status_code == 200
        engine.collective_rpc.assert_awaited_once_with(
            "clear_steering_vectors",
            args=(),
            kwargs={"target": "main"},
        )

    def test_clear_target_draft_returns_501(self, client, engine):
        resp = client.post("/v1/steering/clear", json={"target": "draft"})
        assert resp.status_code == 501
        assert "not yet implemented" in resp.json()["error"]
        engine.collective_rpc.assert_not_called()
