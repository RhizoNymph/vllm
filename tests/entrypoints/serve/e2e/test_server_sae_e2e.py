# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-server e2e for SAE delta steering.

First coverage of the full HTTP flow the real-weights tests bypass via
``collective_rpc``: register a synthetic ``kind=sae_delta`` module from
an on-disk safetensors layout, steer a request with
``sae_clamp_specs``, read the global tier back, and drive the
``/v1/steering/sae/{set,clear}`` global endpoints. The shared server
runs compiled with frozen SAE topology, so registration lands in a
spare delta slot reserved via ``--sae-spare-slot-sites`` — this is also
the first live coverage of hot registration on a compiled engine.

The SAE is synthetic (random unit encoder/decoder rows at the model's
hidden size), so assertions are magnitude-based — a high absolute clamp
must move the answer logprob — never semantic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.entrypoints.serve.e2e.helpers import (
    CLEAN,
    HIDDEN,
    HOOK,
    LAYER,
    MODEL,
    answer_lp,
    complete,
    requires_cuda,
)

pytestmark = [requires_cuda]

_MODULE = "e2e_sae"
_N_FEATURES = 4


def _write_sae_dir(root: Path, d_model: int) -> Path:
    """A minimal single-site SAE: unit-norm encoder/decoder rows."""
    generator = torch.Generator().manual_seed(7)
    encoder = torch.randn(_N_FEATURES, d_model, generator=generator)
    decoder = torch.randn(_N_FEATURES, d_model, generator=generator)
    encoder = encoder / encoder.norm(dim=-1, keepdim=True)
    decoder = decoder / decoder.norm(dim=-1, keepdim=True)
    save_file(
        {
            "encoder_weight": encoder,
            "encoder_bias": torch.zeros(_N_FEATURES),
            "decoder_weight": decoder,
        },
        str(root / f"layer_{LAYER}_{HOOK}.safetensors"),
    )
    return root


def _manifest(weights_uri: str, d_model: int = HIDDEN) -> dict:
    return {
        "d_model": d_model,
        "d_sae": 16,
        "activation": "relu",
        "layers": [[LAYER, HOOK]],
        "clampable_features": list(range(_N_FEATURES)),
        "activation_params": {},
        "weights_uri": weights_uri,
    }


def _clamp_spec(value: float, phase: str = "both") -> dict:
    return {
        "module_name": _MODULE,
        "phase": phase,
        "clamps": {
            HOOK: {str(LAYER): [{"feature_idx": 0, "kind": "absolute", "value": value}]}
        },
    }


@pytest.fixture(scope="module")
def sae_module(http, tmp_path_factory):
    sae_dir = _write_sae_dir(tmp_path_factory.mktemp("sae"), HIDDEN)
    r = http.post(
        "/v1/steering/modules/register",
        json={
            "name": _MODULE,
            "kind": "sae_delta",
            "sae_manifest": _manifest(str(sae_dir)),
        },
        timeout=120.0,
    )
    assert r.status_code == 200, r.text
    yield _MODULE
    http.post("/v1/steering/modules/unregister", json={"name": _MODULE})


@pytest.fixture(scope="module")
def baseline_lp(http) -> float:
    return answer_lp(complete(http))


def test_sae_clamp_spec_moves_output_without_contamination(
    http, sae_module, baseline_lp
):
    """A high absolute clamp through ``sae_clamp_specs`` shifts the answer
    logprob; the next plain request reproduces the baseline exactly."""
    clamped = answer_lp(complete(http, sae_clamp_specs=[_clamp_spec(50.0)]))
    after = answer_lp(complete(http))
    assert abs(clamped - baseline_lp) > 1.0
    assert after == pytest.approx(baseline_lp, abs=1e-6)


def test_sae_clamp_magnitude_scales(http, sae_module, baseline_lp):
    small = answer_lp(complete(http, sae_clamp_specs=[_clamp_spec(5.0)]))
    large = answer_lp(complete(http, sae_clamp_specs=[_clamp_spec(50.0)]))
    assert abs(large - baseline_lp) >= abs(small - baseline_lp) - 1e-3


def test_sae_global_tier_set_and_clear(http, sae_module, baseline_lp):
    """Global decode+prefill SAE clamps steer a plain request; clearing
    restores the baseline exactly and the GET endpoint reflects state."""
    r_set = http.post(
        "/v1/steering/sae/set",
        json={
            "prefill_specs": [_clamp_spec(50.0)],
            "decode_specs": [_clamp_spec(50.0)],
        },
    )
    try:
        assert r_set.status_code == 200, r_set.text
        shifted = answer_lp(complete(http))
        r_get = http.get("/v1/steering/sae")
    finally:
        r_clear = http.post("/v1/steering/sae/clear")
    restored = answer_lp(complete(http))
    assert abs(shifted - baseline_lp) > 1.0
    assert r_get.status_code == 200
    assert r_clear.status_code == 200
    assert restored == pytest.approx(baseline_lp, abs=1e-6)


def test_unknown_sae_module_in_spec_rejected(http, sae_module):
    spec = dict(_clamp_spec(50.0), module_name="no_such_sae")
    r = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "sae_clamp_specs": [spec],
        },
    )
    assert r.status_code == 400, r.text


def test_mismatched_weights_registration_rejected(http, tmp_path_factory):
    """Weight tensors whose width disagrees with the manifest d_model are
    rejected by the frontend loader with a 400, before any broadcast.

    (A self-consistent wrong d_model would only fail worker-side during
    attach, surfacing as a 500 — the frontend never compares d_model to
    the model's hidden size — so the deterministic 400 contract to pin
    is the manifest/weights shape check.)"""
    sae_dir = _write_sae_dir(tmp_path_factory.mktemp("sae_bad"), 8)
    r = http.post(
        "/v1/steering/modules/register",
        json={
            "name": "e2e_sae_bad",
            "kind": "sae_delta",
            "sae_manifest": _manifest(str(sae_dir)),
        },
    )
    assert r.status_code == 400, r.text


def test_full_reconstruction_kind_over_http_is_rejected_today(http):
    """``kind=sae_full_reconstruction`` is not accepted by the register
    endpoint yet (pydantic literal) — pinned here so extending the
    endpoint (feat/sae-fr-http) consciously flips this test."""
    r = http.post(
        "/v1/steering/modules/register",
        json={
            "name": "e2e_sae_fr",
            "kind": "sae_full_reconstruction",
            "sae_manifest": _manifest("/nonexistent"),
        },
    )
    assert r.status_code == 422, r.text
