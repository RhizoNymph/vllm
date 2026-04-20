# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed-execution tests for the steering API router.

Exercises the multi-rank aspects of the determinism contract: TP
divergence detection, PP-disjoint layer unioning, status deep-merge,
and the `/v1/steering/layers` debug endpoint. Uses a mocked engine
so no GPUs / workers are required.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from vllm.entrypoints.serve.steering._merge import (
    deep_merge_status,
    normalize_worker_err,
)
from vllm.entrypoints.serve.steering.api_router import router
from vllm.exceptions import SteeringVectorError
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


# --- TP/PP agreement and divergence ---


class TestValidatePhaseAgreement:
    def test_all_tp_ranks_agree(self, client, engine):
        """TP=2, PP=1: both ranks return identical layer sets → 200."""
        engine.collective_rpc.side_effect = [
            [(0, 0, [0, 1, 2]), (1, 0, [0, 1, 2])],
            [(0, 0, [0, 1, 2]), (1, 0, [0, 1, 2])],
        ]
        engine.reset_prefix_cache.return_value = True
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [0, 1, 2]

    def test_pp_disjoint_unions_correctly(self, client, engine):
        """PP=2, TP=1: each pp_rank reports disjoint layers → unioned."""
        engine.collective_rpc.side_effect = [
            [(0, 0, [0, 1]), (0, 1, [2, 3])],
            [(0, 0, [0, 1]), (0, 1, [2, 3])],
        ]
        engine.reset_prefix_cache.return_value = True
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [0, 1, 2, 3]

    def test_tp_pp_mixed_unions_correctly(self, client, engine):
        """TP=2 × PP=2: two TP ranks per PP stage, disjoint across PP."""
        entries = [
            (0, 0, [0, 1]),
            (1, 0, [0, 1]),
            (0, 1, [2, 3]),
            (1, 1, [2, 3]),
        ]
        engine.collective_rpc.side_effect = [entries, entries]
        engine.reset_prefix_cache.return_value = True
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 200
        assert resp.json()["layers_updated"] == [0, 1, 2, 3]


class TestTPDivergence:
    def test_tp_ranks_disagree_returns_500(self, client, engine):
        """TP ranks in same PP stage returning different layer sets → 500."""
        engine.collective_rpc.side_effect = [
            [(0, 0, [0, 1, 2]), (1, 0, [0, 1])],
        ]
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 500
        body = resp.json()
        assert "invariant violation" in body["error"].lower()
        assert "pp_rank=0" in body["error"]

    def test_tp_divergence_across_pp_detected_per_pp(self, client, engine):
        """Divergence in one PP stage surfaces, regardless of other PP."""
        engine.collective_rpc.side_effect = [
            [
                (0, 0, [0, 1]),
                (1, 0, [0, 1]),
                (0, 1, [2, 3]),
                (1, 1, [2]),
            ],
        ]
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0]}))
        assert resp.status_code == 500
        assert "pp_rank=1" in resp.json()["error"]


# --- Error consolidation ---


class TestErrorConsolidation:
    def test_size_mismatch_single_400(self, client, engine):
        """SteeringVectorError from any rank → single 400 with clean message."""
        engine.collective_rpc.side_effect = SteeringVectorError(
            "Rank 1: Layer 0 (post_mlp): expected vector of size 128, got 2"
        )
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0, 2.0]}))
        assert resp.status_code == 400
        # Rank prefix stripped.
        assert not resp.json()["error"].startswith("Rank 1:")
        assert "expected vector of size 128" in resp.json()["error"]

    def test_non_finite_single_400(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError(
            "Rank 0: Layer 0 (post_mlp): steering vector contains "
            "non-finite values (NaN or Infinity)"
        )
        resp = client.post("/v1/steering/set", json=_vecs({0: [1.0, 2.0]}))
        assert resp.status_code == 400
        assert not resp.json()["error"].startswith("Rank 0:")
        assert "non-finite" in resp.json()["error"]


# --- Status deep-merge ---


class TestDeepMergeStatus:
    def test_merges_disjoint(self):
        result = deep_merge_status(
            [
                {0: {"post_mlp": {"norm": 1.0}}},
                {5: {"post_mlp": {"norm": 2.5}}},
            ]
        )
        assert result == {
            0: {"post_mlp": {"norm": 1.0}},
            5: {"post_mlp": {"norm": 2.5}},
        }

    def test_merges_identical_tp_duplicates(self):
        """TP ranks report identical state — merge must not raise."""
        result = deep_merge_status(
            [
                {0: {"post_mlp": {"norm": 1.0}}},
                {0: {"post_mlp": {"norm": 1.0}}},
            ]
        )
        assert result == {0: {"post_mlp": {"norm": 1.0}}}

    def test_raises_on_divergence(self):
        with pytest.raises(RuntimeError, match="divergence"):
            deep_merge_status(
                [
                    {0: {"post_mlp": {"norm": 1.0}}},
                    {0: {"post_mlp": {"norm": 2.0}}},
                ]
            )

    def test_handles_empty_inputs(self):
        assert deep_merge_status([]) == {}
        assert deep_merge_status([{}, {}]) == {}
        assert deep_merge_status([None, {}]) == {}


class TestGetSteeringDivergence:
    def test_divergence_surfaces_as_500(self, client, engine):
        engine.collective_rpc.return_value = [
            {0: {"post_mlp": {"norm": 1.0}}},
            {0: {"post_mlp": {"norm": 2.0}}},
        ]
        resp = client.get("/v1/steering")
        assert resp.status_code == 500
        assert "invariant violation" in resp.json()["error"].lower()


# --- /v1/steering/layers ---


class TestGetSteeringLayers:
    def test_merges_hook_points_across_workers(self, client, engine):
        """PP-disjoint layers + TP-identical hooks are merged correctly."""
        engine.collective_rpc.return_value = [
            {0: ["post_mlp"], 1: ["post_mlp", "pre_attn"]},
            {0: ["post_mlp"], 1: ["post_mlp", "pre_attn"]},
            {2: ["post_mlp"], 3: ["post_mlp"]},
            {2: ["post_mlp"], 3: ["post_mlp"]},
        ]
        resp = client.get("/v1/steering/layers")
        assert resp.status_code == 200
        layers = resp.json()["layers"]
        assert set(layers.keys()) == {"0", "1", "2", "3"}
        assert layers["1"]["hook_points"] == ["post_mlp", "pre_attn"]
        assert layers["2"]["hook_points"] == ["post_mlp"]

    def test_empty_worker_results(self, client, engine):
        engine.collective_rpc.return_value = [{}, {}]
        resp = client.get("/v1/steering/layers")
        assert resp.status_code == 200
        assert resp.json()["layers"] == {}

    def test_engine_error_returns_500(self, client, engine):
        engine.collective_rpc.side_effect = RuntimeError("rpc exploded")
        resp = client.get("/v1/steering/layers")
        assert resp.status_code == 500


# --- normalize_worker_err ---


class TestNormalizeWorkerErr:
    def test_strips_rank_prefix(self):
        assert normalize_worker_err("Rank 3: boom") == "boom"

    def test_strips_single_digit_rank(self):
        assert normalize_worker_err("Rank 0: error") == "error"

    def test_no_prefix_unchanged(self):
        assert normalize_worker_err("plain message") == "plain message"

    def test_embedded_rank_not_stripped(self):
        # The stripper only touches a leading Rank N: prefix.
        assert (
            normalize_worker_err("warning: see Rank 1: earlier log")
            == "warning: see Rank 1: earlier log"
        )
