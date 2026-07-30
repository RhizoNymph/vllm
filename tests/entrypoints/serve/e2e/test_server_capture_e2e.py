# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-server e2e for per-request activation capture.

First real-HTTP coverage of the ``capture`` request field: a request
routed to the filesystem consumer must come back with ``capture_results``
(``capture_wait`` durability) and leave non-empty activation files under
the consumer root; malformed specs must be rejected with a 400 before
admission.
"""

from __future__ import annotations

from pathlib import Path

from tests.entrypoints.serve.e2e.helpers import (
    CLEAN,
    HOOK,
    LAYER,
    MODEL,
    requires_cuda,
)

pytestmark = [requires_cuda]


def test_capture_wait_returns_results_and_writes_files(http, capture_root):
    """``capture_wait=true`` blocks until the filesystem consumer finalized
    the request's captures: the response carries an ok ``capture_results``
    entry and the files exist on disk with real bytes in them."""
    tag = "cap-e2e"
    body = {
        "model": MODEL,
        "prompt": CLEAN,
        "max_tokens": 1,
        "temperature": 0.0,
        "capture": {
            "filesystem": {
                "request_id": "cap-e2e-req-1",
                "tag": tag,
                "hooks": {HOOK: [LAYER]},
                "positions": "all_prompt",
            }
        },
        "capture_wait": True,
    }
    r = http.post("/v1/completions", json=body, timeout=300.0)
    assert r.status_code == 200, r.text
    results = r.json().get("capture_results")
    assert results is not None
    assert results["filesystem"]["status"] == "ok", results
    bins = list((Path(capture_root) / tag).rglob("*.bin"))
    assert bins, f"no capture files under {capture_root}/{tag}"
    assert all(p.stat().st_size > 0 for p in bins)


def test_invalid_capture_hook_rejected_with_400(http):
    r = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "capture": {
                "filesystem": {
                    "request_id": "cap-e2e-bad",
                    "tag": "cap-e2e-bad",
                    "hooks": {"not_a_hook": [LAYER]},
                    "positions": "all_prompt",
                }
            },
        },
    )
    assert r.status_code == 400, r.text


def test_unknown_consumer_rejected_with_400(http):
    r = http.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "capture": {"no_such_consumer": {"anything": True}},
        },
    )
    assert r.status_code == 400, r.text
