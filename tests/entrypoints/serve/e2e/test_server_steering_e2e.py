# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-server e2e for additive steering and directional clamps.

Ports the HTTP checks of ``tests/gpu_clamp_validate.py`` into a
collected pytest suite: per-request packed vectors, clamp exactness
edges and semantics, the global tier via ``/v1/steering/set``, named
modules, and the request-level 400 paths — all through a real server
and real workers rather than the mocked-engine router harness.
"""

from __future__ import annotations

import pytest

from tests.entrypoints.serve.e2e.helpers import (
    CLEAN,
    HOOK,
    LAYER,
    MODEL,
    answer_lp,
    clamp_entries,
    complete,
    pack_hook_vectors,
    requires_cuda,
    unit,
)

pytestmark = [requires_cuda]


@pytest.fixture(scope="module")
def baseline_lp(http) -> float:
    return answer_lp(complete(http))


def test_packed_vectors_change_output_without_contamination(http, baseline_lp):
    """A packed per-request steering vector shifts the answer logprob and
    leaves the next unsteered request untouched."""
    steered = answer_lp(
        complete(http, steering_vectors=pack_hook_vectors(LAYER, 80.0 * unit(1)))
    )
    after = answer_lp(complete(http))
    assert abs(steered - baseline_lp) > 1.0
    assert after == pytest.approx(baseline_lp, abs=1e-6)


def test_clamp_wide_bounds_and_strength_zero_are_exact_noops(http, baseline_lp):
    """In-bounds and strength-0 clamps must reproduce the baseline exactly
    (the delta is exactly zero, so hidden states are bitwise unchanged)."""
    v = unit(2).tolist()
    wide = answer_lp(
        complete(
            http,
            steering_clamps=clamp_entries([{"vector": v, "min": -1e6, "max": 1e6}]),
        )
    )
    s0 = answer_lp(
        complete(
            http,
            steering_clamps=clamp_entries(
                [{"vector": v, "value": 40.0, "strength": 0.0}]
            ),
        )
    )
    assert wide == pytest.approx(baseline_lp, abs=1e-6)
    assert s0 == pytest.approx(baseline_lp, abs=1e-6)


def test_aggressive_pin_degrades_answer(http, baseline_lp):
    pin = clamp_entries([{"vector": unit(2).tolist(), "value": 40.0}])
    pinned = answer_lp(complete(http, steering_clamps=pin))
    assert pinned < baseline_lp - 1.0


def test_clamp_after_additive_erases_component(http):
    """Clamp-last semantics: (add 60·v̂) then (pin v̂ to c) matches the pin
    alone, while the addition alone lands far away."""
    v = unit(3)
    pin = clamp_entries([{"vector": v.tolist(), "value": 1.0}])
    p_pin = answer_lp(complete(http, steering_clamps=pin))
    p_add = answer_lp(
        complete(http, steering_vectors=pack_hook_vectors(LAYER, 60.0 * v))
    )
    p_both = answer_lp(
        complete(
            http,
            steering_vectors=pack_hook_vectors(LAYER, 60.0 * v),
            steering_clamps=pin,
        )
    )
    assert p_both == pytest.approx(p_pin, abs=0.15)
    assert abs(p_add - p_pin) > 1.0


def test_decode_only_clamp_leaves_prefill_token_intact(http):
    """A decode-tier clamp must not change the first sampled token (prefill
    is untouched) while the continuation diverges under an aggressive pin."""
    base = complete(http, max_tokens=24)
    clamped = complete(
        http,
        max_tokens=24,
        decode_steering_clamps=clamp_entries(
            [{"vector": unit(2).tolist(), "value": 40.0}]
        ),
    )
    assert base["logprobs"]["tokens"][0] == clamped["logprobs"]["tokens"][0]
    assert base["text"] != clamped["text"]


def test_global_set_shifts_and_clear_restores(http, baseline_lp):
    """Global vectors via ``/v1/steering/set`` steer an otherwise plain
    request; ``/clear`` restores the baseline exactly; the read endpoints
    report the configured layer."""
    r_set = http.post(
        "/v1/steering/set",
        json={"vectors": {HOOK: {str(LAYER): (80.0 * unit(4)).tolist()}}},
    )
    try:
        assert r_set.status_code == 200, r_set.text
        shifted = answer_lp(complete(http))
        r_status = http.get("/v1/steering")
        r_layers = http.get("/v1/steering/layers")
    finally:
        r_clear = http.post("/v1/steering/clear")
    restored = answer_lp(complete(http))
    assert abs(shifted - baseline_lp) > 1.0
    assert r_clear.status_code == 200
    assert restored == pytest.approx(baseline_lp, abs=1e-6)
    assert r_status.status_code == 200
    assert r_layers.status_code == 200
    assert LAYER in r_layers.json()["layers"]


def test_named_module_matches_inline_and_unregister_rejects(http):
    """A registered vectors module referenced by ``steering_name`` matches
    the packed inline equivalent; after unregistering, the name is a 400."""
    v = unit(5)
    r_reg = http.post(
        "/v1/steering/modules/register",
        json={
            "name": "e2e_vecmod",
            "vectors": {HOOK: {str(LAYER): (80.0 * v).tolist()}},
        },
    )
    assert r_reg.status_code == 200, r_reg.text
    try:
        named = answer_lp(complete(http, steering_name="e2e_vecmod"))
        inline = answer_lp(
            complete(http, steering_vectors=pack_hook_vectors(LAYER, 80.0 * v))
        )
    finally:
        r_unreg = http.post(
            "/v1/steering/modules/unregister", json={"name": "e2e_vecmod"}
        )
    assert r_unreg.status_code == 200
    assert named == pytest.approx(inline, abs=0.15)
    r_gone = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "steering_name": "e2e_vecmod",
        },
    )
    assert r_gone.status_code == 400


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(
            {
                "steering_clamps": {
                    HOOK: {
                        str(LAYER): [
                            {"vector": unit(300 + j).tolist(), "value": 0.0}
                            for j in range(5)
                        ]
                    }
                }
            },
            id="over-k-cap",
        ),
        pytest.param(
            {
                "steering_clamps": {
                    HOOK: {str(LAYER): [{"vector": [1.0, 0.0], "value": 0.0}]}
                }
            },
            id="wrong-width-clamp",
        ),
        pytest.param(
            {
                "steering": [
                    {
                        "when": {"kind": "always"},
                        "scope": "rest_of_request",
                        "apply": {"kind": "clamp", "strength": 1.0},
                    }
                ]
            },
            id="per-request-clamp-gate",
        ),
    ],
)
def test_invalid_steering_requests_return_400(http, extra):
    r = http.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": CLEAN, "max_tokens": 1, **extra},
    )
    assert r.status_code == 400, r.text
