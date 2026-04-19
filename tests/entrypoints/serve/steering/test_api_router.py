# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the steering API router using a mock engine.

Tests cover three-tier steering (base, prefill, decode) with co-located
scale format support.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

import vllm.envs as envs
from vllm.entrypoints.serve.steering.api_router import (
    _normalize_spec,
    attach_router,
    router,
)
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT

_HP = DEFAULT_HOOK_POINT.value


def _make_app(engine_mock) -> FastAPI:
    """Build a FastAPI app with the steering router and a mock engine."""
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
    """Shorthand: wrap layer vectors in base-tier format."""
    return {"vectors": {hook: layer_vecs}}


# --- POST /v1/steering/set: base vectors ---


class TestSetSteeringBase:
    """Tests for setting base-tier vectors."""

    def test_set_basic(self, client, engine):
        """Set vectors on one layer."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0, 2.0, 3.0]}),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["layers_updated"] == [0]

    def test_set_multiple_layers(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0, [0, 3])], [(0, 0, [0, 3])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0], "3": [2.0]}),
        )
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [0, 3]

    def test_set_with_co_located_scale(self, client, engine):
        """Co-located scale factors are pre-multiplied before sending."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {_HP: {"0": {"vector": [1.0, 2.0], "scale": 3.0}}}},
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        kwargs = validate_call.kwargs.get("kwargs", validate_call[1].get("kwargs", {}))
        vectors = kwargs.get("vectors", {})
        assert vectors[_HP][0] == [3.0, 6.0]

    def test_set_bare_list_scale_one(self, client, engine):
        """Bare list vectors pass through with scale=1.0."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [5.0, 10.0]}),
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        kwargs = validate_call.kwargs.get("kwargs", validate_call[1].get("kwargs", {}))
        vectors = kwargs.get("vectors", {})
        assert vectors[_HP][0] == [5.0, 10.0]

    def test_set_missing_layer_returns_400(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0, [])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"999": [1.0]}),
        )
        assert resp.status_code == 400
        assert "999" in resp.json()["error"]

    def test_set_partial_missing_layers(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0], "999": [1.0]}),
        )
        assert resp.status_code == 400
        assert "999" in resp.json()["error"]

    def test_set_no_steerable_layers(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0, [])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["error"]

    def test_set_vector_size_mismatch_returns_400(self, client, engine):
        engine.collective_rpc.side_effect = SteeringVectorError(
            "Layer 0: expected vector of size 8, got 3"
        )
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0, 2.0, 3.0]}),
        )
        assert resp.status_code == 400
        assert "expected vector of size" in resp.json()["error"]

    def test_set_nonfinite_returns_400(self, client, engine):
        engine.collective_rpc.side_effect = SteeringVectorError(
            "Layer 0: steering vector contains non-finite values"
        )
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 400
        assert "non-finite" in resp.json()["error"]

    def test_set_runtime_error_returns_500(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("GPU exploded")
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 500
        assert "GPU exploded" in resp.json()["error"]

    def test_set_with_replace(self, client, engine):
        """replace=True passes replace flag to set_steering_vectors."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={**_vecs({"0": [1.0]}), "replace": True},
        )
        assert resp.status_code == 200
        assert engine.collective_rpc.call_count == 2
        apply_call = engine.collective_rpc.call_args_list[1]
        kwargs = apply_call.kwargs.get("kwargs", apply_call[1].get("kwargs", {}))
        assert kwargs["replace"] is True

    def test_set_without_replace_no_clear(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 200
        assert engine.collective_rpc.call_count == 2

    def test_set_without_replace_passes_false(self, client, engine):
        """Default replace=False is passed through to workers."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 200
        apply_call = engine.collective_rpc.call_args_list[1]
        kwargs = apply_call.kwargs.get("kwargs", apply_call[1].get("kwargs", {}))
        assert kwargs["replace"] is False

    def test_set_empty_all_tiers(self, client, engine):
        """No tiers provided -> immediate 400."""
        resp = client.post("/v1/steering/set", json={})
        assert resp.status_code == 400
        assert "No vectors provided" in resp.json()["error"]
        engine.collective_rpc.assert_not_called()

    def test_set_empty_vectors(self, client, engine):
        """Empty vectors dict -> immediate 400 (no data in any tier)."""
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {}},
        )
        assert resp.status_code == 400
        assert "No vectors provided" in resp.json()["error"]
        engine.collective_rpc.assert_not_called()

    def test_set_invalid_hook_point(self, client, engine):
        resp = client.post(
            "/v1/steering/set",
            json={"vectors": {"invalid_hook": {"0": [1.0]}}},
        )
        assert resp.status_code == 400
        assert "Invalid hook point" in resp.json()["error"]


# --- POST /v1/steering/set: three-tier vectors ---


class TestSetSteeringThreeTier:
    """Tests for three-tier steering (base, prefill, decode)."""

    def test_set_prefill_only(self, client, engine):
        """Set prefill-specific vectors without base."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={"prefill_vectors": {_HP: {"0": [1.0, 2.0]}}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["layers_updated"] == [0]

    def test_set_decode_only(self, client, engine):
        """Set decode-specific vectors without base."""
        engine.collective_rpc.side_effect = [[(0, 0, [1])], [(0, 0, [1])]]
        resp = client.post(
            "/v1/steering/set",
            json={"decode_vectors": {_HP: {"1": [3.0, 4.0]}}},
        )
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [1]

    def test_set_all_three_tiers(self, client, engine):
        """Set all three tiers simultaneously."""
        engine.collective_rpc.side_effect = [[(0, 0, [0, 1, 2])], [(0, 0, [0, 1, 2])]]
        resp = client.post(
            "/v1/steering/set",
            json={
                "vectors": {_HP: {"0": [1.0]}},
                "prefill_vectors": {_HP: {"1": [2.0]}},
                "decode_vectors": {_HP: {"2": [3.0]}},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert sorted(body["layers_updated"]) == [0, 1, 2]

    def test_prefill_with_co_located_scale(self, client, engine):
        """Prefill vectors support co-located scale format."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={
                "prefill_vectors": {_HP: {"0": {"vector": [1.0, 2.0], "scale": 2.0}}}
            },
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        kwargs = validate_call.kwargs.get("kwargs", validate_call[1].get("kwargs", {}))
        prefill_vecs = kwargs.get("prefill_vectors", {})
        assert prefill_vecs[_HP][0] == [2.0, 4.0]

    def test_decode_with_co_located_scale(self, client, engine):
        """Decode vectors support co-located scale format."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={"decode_vectors": {_HP: {"0": {"vector": [3.0, 6.0], "scale": 0.5}}}},
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        kwargs = validate_call.kwargs.get("kwargs", validate_call[1].get("kwargs", {}))
        decode_vecs = kwargs.get("decode_vectors", {})
        assert decode_vecs[_HP][0] == [1.5, 3.0]

    def test_replace_clears_all_tiers(self, client, engine):
        """replace=True passes replace flag to set_steering_vectors."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={
                "decode_vectors": {_HP: {"0": [1.0]}},
                "replace": True,
            },
        )
        assert resp.status_code == 200
        assert engine.collective_rpc.call_count == 2
        apply_call = engine.collective_rpc.call_args_list[1]
        kwargs = apply_call.kwargs.get("kwargs", apply_call[1].get("kwargs", {}))
        assert kwargs["replace"] is True

    def test_invalid_hook_in_prefill_tier(self, client, engine):
        """Invalid hook in any tier returns 400."""
        resp = client.post(
            "/v1/steering/set",
            json={"prefill_vectors": {"bad_hook": {"0": [1.0]}}},
        )
        assert resp.status_code == 400
        assert "Invalid hook point" in resp.json()["error"]

    def test_invalid_hook_in_decode_tier(self, client, engine):
        resp = client.post(
            "/v1/steering/set",
            json={"decode_vectors": {"bad_hook": {"0": [1.0]}}},
        )
        assert resp.status_code == 400
        assert "Invalid hook point" in resp.json()["error"]

    def test_hook_points_response_includes_all_tiers(self, client, engine):
        """Response includes hook points from all tiers."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        resp = client.post(
            "/v1/steering/set",
            json={
                "vectors": {"pre_attn": {"0": [1.0]}},
                "decode_vectors": {_HP: {"0": [1.0]}},
            },
        )
        assert resp.status_code == 200
        hooks = resp.json()["hook_points"]
        assert "pre_attn" in hooks
        assert _HP in hooks


# --- POST /v1/steering/clear ---


class TestClearSteering:
    def test_clear(self, client, engine):
        engine.collective_rpc.return_value = None
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        engine.collective_rpc.assert_called_once_with(
            "clear_steering_vectors",
            args=(),
            kwargs={"target": None},
        )

    def test_clear_engine_error(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]

    def test_clear_cache_reset_failure_returns_503(self, client, engine):
        """When prefix cache reset fails after clear, return 503."""
        engine.collective_rpc.return_value = None
        engine.reset_prefix_cache.return_value = False
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"].lower()


# --- Cache invalidation on set ---


class TestSetSteeringCacheInvalidation:
    """Cache invalidation must succeed for set to return 200."""

    def test_base_vectors_cache_failure_returns_503(self, client, engine):
        """Base vectors affect cache; failure returns 503."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        engine.reset_prefix_cache.return_value = False
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"].lower()

    def test_decode_only_cache_failure_returns_503(self, client, engine):
        """Decode-only vectors also require cache invalidation."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        engine.reset_prefix_cache.return_value = False
        resp = client.post(
            "/v1/steering/set",
            json={"decode_vectors": {_HP: {"0": [1.0]}}},
        )
        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"].lower()

    def test_cache_success_returns_200(self, client, engine):
        """When cache reset succeeds, set returns 200."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        engine.reset_prefix_cache.return_value = True
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
        )
        assert resp.status_code == 200


# --- GET /v1/steering ---


class TestGetSteering:
    def test_get_no_active(self, client, engine):
        engine.collective_rpc.return_value = [{}]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        assert resp.json()["active_layers"] == {}

    def test_get_active_layers(self, client, engine):
        engine.collective_rpc.return_value = [
            {
                0: {_HP: {"norm": 1.5}},
                3: {_HP: {"norm": 2.0}},
            },
        ]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        active = resp.json()["active_layers"]
        assert "0" in active
        assert "3" in active
        assert active["0"][_HP]["norm"] == 1.5

    def test_get_merges_multiple_workers(self, client, engine):
        """PP workers own disjoint layers — results are merged."""
        engine.collective_rpc.return_value = [
            {0: {_HP: {"norm": 1.0}}},
            {5: {_HP: {"norm": 2.0}}},
        ]
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
        active = resp.json()["active_layers"]
        assert "0" in active
        assert "5" in active

    def test_get_engine_error(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.get("/v1/steering")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]


# --- _normalize_spec ---


class TestNormalizeSpec:
    """Tests for the ``_normalize_spec`` helper."""

    def test_normalize_spec_drops_empty_hook(self):
        """Hooks whose layer dict is empty are dropped from the result.

        An input like ``{"post_mlp": {}}`` is functionally
        equivalent to omitting the hook entirely: no layers and no
        vectors would be applied. Keeping the empty hook in the
        normalized spec would produce a truthy-but-empty entry that
        confuses downstream checks (affects_cache, response payload).
        """
        spec: dict = {_HP: {}}
        result = _normalize_spec(spec)
        assert _HP not in result
        assert result == {}

    def test_normalize_spec_drops_only_empty_hook(self):
        """Empty hooks are dropped while populated hooks are preserved."""
        spec: dict = {_HP: {0: [1.0, 2.0]}, "pre_attn": {}}
        result = _normalize_spec(spec)
        assert "pre_attn" not in result
        assert _HP in result
        assert result[_HP] == {0: [1.0, 2.0]}

    def test_normalize_spec_keeps_populated_hook(self):
        """Hooks with at least one layer entry survive normalization."""
        spec: dict = {_HP: {0: [1.0, 2.0]}}
        result = _normalize_spec(spec)
        assert result == {_HP: {0: [1.0, 2.0]}}


# --- attach_router ---


class TestAttachRouter:
    def test_attached_in_dev_mode(self):
        app = FastAPI()
        with patch.object(envs, "VLLM_SERVER_DEV_MODE", True):
            attach_router(app)
        paths = {r.path for r in app.routes}
        assert "/v1/steering/set" in paths
        assert "/v1/steering/clear" in paths
        assert "/v1/steering" in paths

    def test_not_attached_without_dev_mode(self):
        app = FastAPI()
        with patch.object(envs, "VLLM_SERVER_DEV_MODE", False):
            attach_router(app)
        paths = {r.path for r in app.routes}
        assert "/v1/steering/set" not in paths


# --- steering API key auth ---


def _make_app_with_tokens(engine_mock, tokens: list[str] | None) -> FastAPI:
    """Build a FastAPI app with steering_api_tokens on app.state."""
    app = FastAPI()
    app.state.engine_client = engine_mock
    app.state.steering_api_tokens = tokens
    app.include_router(router)
    return app


class TestSteeringAPIKeyAuth:
    """Tests that /v1/steering/set and /clear honor the steering API key."""

    def test_set_no_tokens_configured_is_open(self, engine):
        """When no tokens configured, /set requires no Authorization."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        client = _SyncASGIClient(_make_app_with_tokens(engine, None))
        resp = client.post("/v1/steering/set", json=_vecs({"0": [1.0]}))
        assert resp.status_code == 200

    def test_set_empty_tokens_configured_is_open(self, engine):
        """An empty token list behaves the same as unconfigured."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        client = _SyncASGIClient(_make_app_with_tokens(engine, []))
        resp = client.post("/v1/steering/set", json=_vecs({"0": [1.0]}))
        assert resp.status_code == 200

    def test_set_missing_bearer_returns_401(self, engine):
        """Tokens configured but no Authorization header -> 401."""
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post("/v1/steering/set", json=_vecs({"0": [1.0]}))
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_set_wrong_bearer_returns_401(self, engine):
        """Wrong bearer token -> 401, no RPC dispatched."""
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_set_wrong_scheme_returns_401(self, engine):
        """Non-Bearer scheme rejected even if the token value matches."""
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
            headers={"Authorization": "Basic secret"},
        )
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_set_correct_bearer_is_allowed(self, engine):
        """Correct bearer token lets the request through."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_set_accepts_any_configured_token(self, engine):
        """Any token in the configured list is accepted."""
        engine.collective_rpc.side_effect = [[(0, 0, [0])], [(0, 0, [0])]]
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["primary", "backup"]))
        resp = client.post(
            "/v1/steering/set",
            json=_vecs({"0": [1.0]}),
            headers={"Authorization": "Bearer backup"},
        )
        assert resp.status_code == 200

    def test_clear_missing_bearer_returns_401(self, engine):
        """Clear endpoint is gated the same as set."""
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post("/v1/steering/clear")
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_clear_correct_bearer_is_allowed(self, engine):
        """Clear with a valid bearer token succeeds."""
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.post(
            "/v1/steering/clear",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_get_status_is_not_gated(self, engine):
        """Read-only GET /v1/steering is not gated by the steering key."""
        engine.collective_rpc.return_value = [{}]
        client = _SyncASGIClient(_make_app_with_tokens(engine, ["secret"]))
        resp = client.get("/v1/steering")
        assert resp.status_code == 200
