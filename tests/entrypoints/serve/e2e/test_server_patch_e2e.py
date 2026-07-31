# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-server e2e for activation patching and the sweep endpoint.

Covers the request-level patch path (capture-sourced denoise, zeros
ablation, missing-source 400s) and, for the first time in a collected
test, the ``/v1/patch_sweep`` auto-capture + ``recovered``-metric flow,
SSE streaming parity, and the ``DELETE /v1/patch_source/{run}``
lifecycle against real weights.
"""

from __future__ import annotations

import json

import pytest

from tests.entrypoints.serve.e2e.helpers import (
    ANSWER,
    CLEAN,
    HOOK,
    LAYER,
    MODEL,
    answer_lp,
    complete,
    requires_cuda,
)

pytestmark = [requires_cuda]

CORRUPT = "The capital of Germany is"


def _last_prompt_position(http, prompt: str) -> int:
    r = http.post("/tokenize", json={"model": MODEL, "prompt": prompt})
    r.raise_for_status()
    return len(r.json()["tokens"]) - 1


@pytest.fixture(scope="module")
def site(http) -> dict:
    return {
        "layer": LAYER,
        "hook": HOOK,
        "dest_position": _last_prompt_position(http, CLEAN),
    }


@pytest.fixture(scope="module")
def baseline_lp(http) -> float:
    return answer_lp(complete(http))


@pytest.fixture(scope="module")
def clean_run(http, site) -> str:
    """Capture CLEAN's activations once for the capture-sourced tests."""
    run = "e2e-clean"
    r = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "temperature": 0.0,
            "capture": {
                "patch_source": {
                    "run": run,
                    "hooks": {HOOK: [LAYER]},
                    "positions": "all_prompt",
                }
            },
            "capture_wait": True,
        },
        timeout=300.0,
    )
    assert r.status_code == 200, r.text
    return run


def test_capture_sourced_self_patch_stays_near_baseline(
    http, site, baseline_lp, clean_run
):
    patched = answer_lp(
        complete(
            http,
            patch=[
                {
                    **site,
                    "source_run": clean_run,
                    "source_position": site["dest_position"],
                }
            ],
        )
    )
    assert patched == pytest.approx(baseline_lp, abs=0.2)


def test_zeros_ablation_degrades_answer(http, site, baseline_lp):
    ablated = answer_lp(complete(http, patch=[{**site, "source_module": "zeros"}]))
    assert ablated < baseline_lp - 1.0


@pytest.mark.parametrize(
    "entry_extra",
    [
        pytest.param({"source_module": "no_such_module"}, id="unknown-module"),
        pytest.param(
            {"source_run": "never-captured", "source_position": 0},
            id="missing-source-run",
        ),
    ],
)
def test_bad_patch_sources_rejected_with_400(http, site, entry_extra):
    r = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "patch": [{**site, **entry_extra}],
        },
    )
    assert r.status_code == 400, r.text


@pytest.fixture(scope="module")
def recovered_sweep(http) -> dict:
    """One-call auto-capture sweep graded by the ``recovered`` metric.

    The clean run is captured server-side from ``clean_prompt`` (kept
    alive with ``keep_source`` for the streaming and lifecycle tests),
    then every (layer, position) cell patches the clean activations into
    the corrupt prompt.
    """
    body = {
        "model": MODEL,
        "prompt": CORRUPT,
        "clean_prompt": CLEAN,
        "source_run": "e2e-sweep-clean",
        "hook": HOOK,
        "layers": {"start": LAYER - 2, "stop": LAYER + 2},
        "positions": "all_prompt",
        "metric": "recovered",
        "answer_token": ANSWER,
        "keep_source": True,
    }
    r = http.post("/v1/patch_sweep", json=body, timeout=600.0)
    assert r.status_code == 200, r.text
    return r.json()


def test_sweep_auto_captures_and_recovers_somewhere(recovered_sweep):
    """The server captured the clean prompt itself, filled the grid, and at
    least one cell restores most of the clean answer (recovered ≈ 1)."""
    data = recovered_sweep
    assert data["auto_captured"] is True
    assert data["captured_source_run"] == "e2e-sweep-clean"
    cells = [v for row in data["grid"] for v in row if v is not None]
    assert len(cells) == len(data["layers"]) * len(data["positions"])
    assert max(cells) > 0.5, f"best recovered={max(cells):.3f}"


def test_streaming_sweep_matches_summary(http, recovered_sweep):
    """SSE cells cover the full grid and agree with the same execution's
    summary event."""
    body = {
        "model": MODEL,
        "prompt": CORRUPT,
        "clean_prompt": CLEAN,
        "source_run": "e2e-sweep-clean",
        "hook": HOOK,
        "layers": {"start": LAYER - 2, "stop": LAYER + 2},
        "positions": "all_prompt",
        "metric": "logprob",
        "answer_token": ANSWER,
        "keep_source": True,
        "stream": True,
    }
    cells: dict[tuple[int, int], float] = {}
    summary = None
    with http.stream("POST", "/v1/patch_sweep", json=body, timeout=600.0) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("type") == "cell" and event.get("value") is not None:
                cells[(event["layer"], event["position"])] = event["value"]
            elif event.get("type") == "summary":
                summary = event
    assert summary is not None
    n_expected = len(summary["layers"]) * len(summary["positions"])
    assert len(cells) == n_expected
    for i, layer in enumerate(summary["layers"]):
        for j, pos in enumerate(summary["positions"]):
            value = summary["grid"][i][j]
            if value is not None:
                assert cells[(layer, pos)] == pytest.approx(value, abs=1e-6)


def test_drop_source_run_lifecycle(http, recovered_sweep):
    """The kept auto-captured run deletes with a 200, then 404s."""
    r_first = http.delete("/v1/patch_source/e2e-sweep-clean")
    r_second = http.delete("/v1/patch_source/e2e-sweep-clean")
    assert r_first.status_code == 200, r_first.text
    assert r_second.status_code == 404


def test_recovered_without_clean_baseline_rejected(http):
    r = http.post(
        "/v1/patch_sweep",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "source_module": "zeros",
            "hook": HOOK,
            "layers": {"start": LAYER, "stop": LAYER + 1},
            "positions": "all_prompt",
            "metric": "recovered",
            "answer_token": ANSWER,
        },
    )
    assert r.status_code == 400, r.text
