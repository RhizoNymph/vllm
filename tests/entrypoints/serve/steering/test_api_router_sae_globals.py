# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the global-SAE-clamp endpoints of the steering API
router, using a mock engine.

Covers ``POST /v1/steering/sae/set`` (two-phase validate-then-apply,
replace passthrough, error mapping, mandatory prefix-cache reset),
``POST /v1/steering/sae/clear``, ``GET /v1/steering/sae`` (identical-
worker merge + divergence detection), and the steering API key gate.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.serve.steering.api_router import router
from vllm.exceptions import SteeringVectorError


def _make_app(engine_mock, tokens: list[str] | None = None) -> FastAPI:
    """Build a FastAPI app with the steering router and a mock engine."""
    app = FastAPI()
    app.state.engine_client = engine_mock
    if tokens is not None:
        app.state.steering_api_tokens = tokens
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


def _clamp_specs(
    *,
    module_name: str = "g",
    layer: int = 20,
    hook: str = "post_block",
    feature_idx: int = 34,
    value: float = 5.0,
) -> list[dict]:
    """One JSON-shape clamp spec, as a caller would send it."""
    return [
        {
            "module_name": module_name,
            "clamps": {
                hook: {
                    str(layer): [
                        {
                            "feature_idx": feature_idx,
                            "kind": "absolute",
                            "value": value,
                        }
                    ]
                }
            },
        }
    ]


def _rpc_kwargs(call) -> dict:
    return call.kwargs.get("kwargs", {})


# --- POST /v1/steering/sae/set ---


class TestSetSAEGlobalClamps:
    def test_set_happy_path_two_phase(self, client, engine):
        """Set fans out validate_only first, then apply, then cache reset."""
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        assert engine.collective_rpc.call_count == 2
        validate_call, apply_call = engine.collective_rpc.call_args_list
        assert validate_call.args[0] == "set_sae_global_clamps"
        assert _rpc_kwargs(validate_call)["validate_only"] is True
        assert _rpc_kwargs(validate_call)["prefill_specs_raw"] == _clamp_specs()
        assert _rpc_kwargs(validate_call)["decode_specs_raw"] is None
        assert apply_call.args[0] == "set_sae_global_clamps"
        assert _rpc_kwargs(apply_call)["validate_only"] is False
        assert _rpc_kwargs(apply_call)["prefill_specs_raw"] == _clamp_specs()

        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )

    def test_set_decode_specs_passthrough(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"decode_specs": _clamp_specs(feature_idx=7)},
        )
        assert resp.status_code == 200
        validate_call = engine.collective_rpc.call_args_list[0]
        assert _rpc_kwargs(validate_call)["prefill_specs_raw"] is None
        assert _rpc_kwargs(validate_call)["decode_specs_raw"] == _clamp_specs(
            feature_idx=7
        )

    def test_set_replace_flag_passthrough(self, client, engine):
        """replace=True reaches the apply-phase RPC kwargs."""
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs(), "replace": True},
        )
        assert resp.status_code == 200
        apply_call = engine.collective_rpc.call_args_list[1]
        assert _rpc_kwargs(apply_call)["replace"] is True

    def test_set_default_replace_is_false(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 200
        apply_call = engine.collective_rpc.call_args_list[1]
        assert _rpc_kwargs(apply_call)["replace"] is False

    def test_set_validation_error_returns_400_without_apply(self, client, engine):
        """A worker-side validation failure in phase 1 means no apply RPC."""
        engine.collective_rpc.side_effect = SteeringVectorError(
            "SAE clamp spec references unknown module 'nope'."
        )
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs(module_name="nope")},
        )
        assert resp.status_code == 400
        assert "unknown module" in resp.json()["error"]
        assert engine.collective_rpc.call_count == 1
        engine.reset_prefix_cache.assert_not_awaited()

    def test_set_coercion_error_returns_400(self, client, engine):
        """Worker-side coercion ValueErrors map to 400."""
        engine.collective_rpc.side_effect = ValueError(
            "sae_clamp_specs must be a list of clamp-spec objects, got str."
        )
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": ["bogus"]},
        )
        assert resp.status_code == 400
        assert "clamp-spec objects" in resp.json()["error"]

    def test_set_worker_error_rank_prefix_stripped(self, client, engine):
        """The 'Rank N: ' collective_rpc prefix is stripped from 400s."""
        engine.collective_rpc.side_effect = SteeringVectorError(
            "Rank 0: SAE clamp spec references unknown module 'nope'."
        )
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs(module_name="nope")},
        )
        assert resp.status_code == 400
        assert not resp.json()["error"].startswith("Rank 0:")

    def test_set_runtime_error_returns_500(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("GPU exploded")
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 500
        assert "GPU exploded" in resp.json()["error"]

    def test_set_empty_request_returns_400(self, client, engine):
        """No specs and replace=False is a no-op — reject it up front."""
        resp = client.post("/v1/steering/sae/set", json={})
        assert resp.status_code == 400
        assert "No clamp specs provided" in resp.json()["error"]
        engine.collective_rpc.assert_not_called()

    def test_set_replace_only_clears_atomically(self, client, engine):
        """replace=True with no specs is a legal atomic clear-via-swap."""
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"replace": True},
        )
        assert resp.status_code == 200
        assert engine.collective_rpc.call_count == 2
        engine.reset_prefix_cache.assert_awaited_once()

    def test_set_unknown_field_rejected(self, client, engine):
        """extra='forbid' on the request model rejects unknown fields."""
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs(), "bogus_field": 1},
        )
        assert resp.status_code == 422
        engine.collective_rpc.assert_not_called()

    def test_set_duplicate_rank_coordinate_returns_500(self, client, engine):
        """Two workers claiming the same (tp, pp) is an invariant violation."""
        engine.collective_rpc.side_effect = [[(0, 0), (0, 0)]]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 500
        assert "invariant violation" in resp.json()["error"]
        # Divergence detected in phase 1: no apply RPC, no cache reset.
        assert engine.collective_rpc.call_count == 1
        engine.reset_prefix_cache.assert_not_awaited()

    def test_set_multi_rank_happy_path(self, client, engine):
        """Distinct (tp, pp) coordinates across workers are accepted."""
        ranks = [(0, 0), (1, 0), (0, 1), (1, 1)]
        engine.collective_rpc.side_effect = [ranks, ranks]
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 200


# --- Cache invalidation ---


class TestSAEGlobalClampsCacheInvalidation:
    def test_set_cache_failure_returns_503(self, client, engine):
        """Clamps applied but cache reset failed -> 503, not 200."""
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        engine.reset_prefix_cache.return_value = False
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"].lower()

    def test_set_cache_success_returns_200(self, client, engine):
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        engine.reset_prefix_cache.return_value = True
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 200

    def test_clear_cache_failure_returns_503(self, client, engine):
        engine.collective_rpc.return_value = None
        engine.reset_prefix_cache.return_value = False
        resp = client.post("/v1/steering/sae/clear")
        assert resp.status_code == 503
        assert "prefix cache" in resp.json()["error"].lower()


# --- POST /v1/steering/sae/clear ---


class TestClearSAEGlobalClamps:
    def test_clear(self, client, engine):
        engine.collective_rpc.return_value = None
        resp = client.post("/v1/steering/sae/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        engine.collective_rpc.assert_called_once_with("clear_sae_global_clamps")
        engine.reset_prefix_cache.assert_awaited_once_with(
            reset_running_requests=True
        )

    def test_clear_engine_error_returns_500(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.post("/v1/steering/sae/clear")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]


# --- GET /v1/steering/sae ---


class TestGetSAEGlobalClamps:
    _STATUS = {
        "prefill": [
            {
                "module_name": "g",
                "phase": "both",
                "clamps": {
                    "post_block": {
                        "20": [
                            {
                                "feature_idx": 34,
                                "kind": "absolute",
                                "value": 5.0,
                                "only_if_active": False,
                            }
                        ]
                    }
                },
            }
        ],
        "decode": [],
    }

    def test_get_empty(self, client, engine):
        engine.collective_rpc.return_value = [{"prefill": [], "decode": []}]
        resp = client.get("/v1/steering/sae")
        assert resp.status_code == 200
        assert resp.json() == {"prefill": [], "decode": []}

    def test_get_identical_workers_returns_one(self, client, engine):
        """Workers are deterministic-identical: return one worker's view."""
        engine.collective_rpc.return_value = [self._STATUS, dict(self._STATUS)]
        resp = client.get("/v1/steering/sae")
        assert resp.status_code == 200
        assert resp.json() == self._STATUS
        engine.collective_rpc.assert_called_once_with(
            "get_sae_global_clamps_status"
        )

    def test_get_divergent_workers_returns_500(self, client, engine):
        engine.collective_rpc.return_value = [
            self._STATUS,
            {"prefill": [], "decode": []},
        ]
        resp = client.get("/v1/steering/sae")
        assert resp.status_code == 500
        assert "invariant violation" in resp.json()["error"]

    def test_get_engine_error_returns_500(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("fail")
        resp = client.get("/v1/steering/sae")
        assert resp.status_code == 500
        assert "fail" in resp.json()["error"]


# --- steering API key auth ---


class TestSAEGlobalClampsAuth:
    """The SAE mutating endpoints honor the steering API key exactly
    like the additive /set and /clear."""

    def test_set_no_tokens_configured_is_open(self, engine):
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        client = _SyncASGIClient(_make_app(engine, tokens=None))
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 200

    def test_set_missing_bearer_returns_401(self, engine):
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
        )
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_set_wrong_bearer_returns_401(self, engine):
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_set_correct_bearer_is_allowed(self, engine):
        engine.collective_rpc.side_effect = [[(0, 0)], [(0, 0)]]
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.post(
            "/v1/steering/sae/set",
            json={"prefill_specs": _clamp_specs()},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_clear_missing_bearer_returns_401(self, engine):
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.post("/v1/steering/sae/clear")
        assert resp.status_code == 401
        engine.collective_rpc.assert_not_called()

    def test_clear_correct_bearer_is_allowed(self, engine):
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.post(
            "/v1/steering/sae/clear",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200

    def test_get_status_is_not_gated(self, engine):
        engine.collective_rpc.return_value = [{"prefill": [], "decode": []}]
        client = _SyncASGIClient(_make_app(engine, tokens=["secret"]))
        resp = client.get("/v1/steering/sae")
        assert resp.status_code == 200
