# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live-server validation of client-provided patch vector sources (PR #253).

Run against a ``--enable-patching --enable-steering`` server:

    python tests/gpu_patch_vector_validate.py --base-url http://localhost:8399

Checks, all via HTTP (greedy, max_tokens=1, answer graded by logprob):

1.  ``zeros`` ablation at a causal site visibly degrades the answer logprob.
2.  A masked subset (64 of hidden dims) degrades strictly less than full-row.
3.  Cross-kind consistency: zeros-via-module == zeros-via-inline (bitwise
    logprob), and mask over ALL dims == no mask.
4.  A registered module row with ``scale: 0.0`` == ``zeros`` (scale applied).
5.  A registered random-direction module row (alpha=1) changes the output.
6.  Capture-sourced entry + all-dims mask == plain capture-sourced denoising.
7.  Vector-sourced ablation sweep endpoint: no clean_prompt, grid fully
    populated, streaming parity, ``recovered`` metric rejected with 400.

Prints one PASS/FAIL line per check and exits non-zero on any FAIL.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys

import httpx
import numpy as np

CLEAN = "The capital of France is"
ANSWER = " Paris"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def answer_logprob(client: httpx.Client, base: str, prompt: str, **extra) -> float:
    """Greedy 1-token completion; return ANSWER's logprob at the next position."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
        **extra,
    }
    r = client.post(f"{base}/v1/completions", json=body, timeout=120.0)
    r.raise_for_status()
    lp = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    return lp.get(ANSWER, -50.0)


def pack(rows: np.ndarray) -> dict:
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": list(rows.shape),
        "data": base64.b64encode(rows.tobytes()).decode(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8399")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--hidden", type=int, default=1024)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    layer, hidden = args.layer, args.hidden
    client = httpx.Client()

    global MODEL
    MODEL = client.get(f"{base}/v1/models").json()["data"][0]["id"]

    # Patch the last prompt position (the position feeding the answer logit)
    # at a mid/late layer: the strongest single-cell causal site for this task.
    toks_probe = client.post(
        f"{base}/tokenize", json={"model": MODEL, "prompt": CLEAN}
    )
    n_pos = len(toks_probe.json()["tokens"]) if toks_probe.status_code == 200 else None
    dest = (n_pos - 1) if n_pos else 5
    site = {"layer": layer, "hook": "post_block", "dest_position": dest}

    print(f"model={MODEL} layer={layer} dest_position={dest} hidden={hidden}")
    p_base = answer_logprob(client, base, CLEAN)
    print(f"clean baseline logprob({ANSWER!r}) = {p_base:.4f}")

    # -- 1. zeros ablation degrades ------------------------------------------
    p_zero = answer_logprob(
        client, base, CLEAN, patch=[{**site, "source_module": "zeros"}]
    )
    check(
        "zeros ablation degrades answer",
        p_zero < p_base - 1.0,
        f"{p_base:.3f} -> {p_zero:.3f}",
    )

    # -- 2. masked subset degrades less than full row ------------------------
    idx64 = list(range(0, hidden, hidden // 64))[:64]
    p_mask = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**site, "source_module": "zeros", "mask": {"indices": idx64}}],
    )
    check(
        "64-dim masked ablation degrades less than full",
        p_zero < p_mask <= p_base + 0.05,
        f"full={p_zero:.3f} masked={p_mask:.3f} base={p_base:.3f}",
    )

    # Consistency checks use the 64-dim masked form (finite, in-top-20 values;
    # the full-row form saturates the top-logprobs window and would compare
    # sentinel-to-sentinel).
    # -- 3a. zeros-via-module == zeros-via-inline (same 64-dim mask) ---------
    pv = pack(np.zeros((2, hidden)))
    p_inline = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**site, "source_inline": 0, "mask": {"indices": idx64}}],
        patch_vectors=pv,
    )
    check(
        "zeros module == zeros inline (masked)",
        math.isclose(p_mask, p_inline, abs_tol=1e-6),
        f"{p_mask:.6f} vs {p_inline:.6f}",
    )

    # -- 3b. indices mask == equivalent inline graded mask row ----------------
    mask_row = np.zeros(hidden, dtype=np.float32)
    mask_row[idx64] = 1.0
    pv2 = pack(np.stack([np.zeros(hidden, dtype=np.float32), mask_row]))
    p_gmask = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**site, "source_inline": 0, "mask": {"inline": 1}}],
        patch_vectors=pv2,
    )
    check(
        "indices mask == inline mask row",
        math.isclose(p_mask, p_gmask, abs_tol=1e-6),
        f"{p_mask:.6f} vs {p_gmask:.6f}",
    )

    # -- 3c. graded alpha: 0 < alpha=0.5 damage < alpha=1 (full row) ----------
    p_half = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**site, "source_module": "zeros", "alpha": 0.5}],
    )
    check(
        "alpha=0.5 between passthrough and full",
        p_zero <= p_half < p_base + 0.05,
        f"full={p_zero:.3f} half={p_half:.3f} base={p_base:.3f}",
    )

    # -- 4/5. named module: random direction, and scale honored ---------------
    rng = np.random.default_rng(0)
    direction = (rng.standard_normal(hidden) * 5.0).astype(np.float32)
    for name, scale in [("pv_dir", 1.0), ("pv_dir_scale0", 0.0)]:
        r = client.post(
            f"{base}/v1/steering/modules/register",
            json={
                "name": name,
                "vectors": {
                    "post_block": {
                        str(layer): {"vector": direction.tolist(), "scale": scale}
                    }
                },
            },
            timeout=60.0,
        )
        if r.status_code >= 300:
            check(f"register module {name}", False, f"{r.status_code}: {r.text[:200]}")
            return finish()
    p_dir = answer_logprob(
        client, base, CLEAN, patch=[{**site, "source_module": "pv_dir"}]
    )
    check(
        "random-direction module changes output",
        abs(p_dir - p_base) > 0.5,
        f"{p_base:.3f} -> {p_dir:.3f}",
    )
    p_scale0 = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**site, "source_module": "pv_dir_scale0", "mask": {"indices": idx64}}],
    )
    check(
        "module scale=0 row == zeros (masked)",
        math.isclose(p_scale0, p_mask, abs_tol=1e-6),
        f"{p_scale0:.6f} vs {p_mask:.6f}",
    )

    # -- unknown module -> 400 ------------------------------------------------
    r = client.post(
        f"{base}/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "patch": [{**site, "source_module": "no_such_module"}],
        },
        timeout=60.0,
    )
    check("unknown source_module -> 400", r.status_code == 400, f"{r.status_code}")

    # -- 6. capture-sourced + all-dims mask == plain denoise ------------------
    run = "pv-validate-clean"
    r = client.post(
        f"{base}/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "capture": {
                "patch_source": {
                    "run": run,
                    "hooks": {"post_block": [layer]},
                    "positions": "all_prompt",
                },
            },
            "capture_wait": True,
        },
        timeout=120.0,
    )
    if r.status_code >= 300:
        check("clean capture", False, f"{r.status_code}: {r.text[:200]}")
        return finish()
    cap_entry = {**site, "source_run": run, "source_position": dest}
    p_den = answer_logprob(client, base, CLEAN, patch=[cap_entry])
    p_den_mask = answer_logprob(
        client,
        base,
        CLEAN,
        patch=[{**cap_entry, "mask": {"indices": list(range(hidden))}}],
    )
    check(
        "capture-sourced self-patch ~ baseline",
        abs(p_den - p_base) < 0.2,
        f"{p_base:.3f} vs {p_den:.3f}",
    )
    check(
        "capture-sourced + all-dims mask == unmasked",
        math.isclose(p_den, p_den_mask, abs_tol=1e-6),
        f"{p_den:.6f} vs {p_den_mask:.6f}",
    )

    # -- 7. vector-sourced ablation sweep (no clean run) ----------------------
    sweep = {
        "prompt": CLEAN,
        "source_module": "zeros",
        "layers": {"start": max(0, layer - 4), "stop": layer + 4},
        "positions": "all_prompt",
        "metric": "logprob",
        "answer_token": ANSWER,
    }
    r = client.post(f"{base}/v1/patch_sweep", json=sweep, timeout=600.0)
    if r.status_code >= 300:
        check("ablation sweep", False, f"{r.status_code}: {r.text[:300]}")
        return finish()
    data = r.json()
    cells = [v for row in data["grid"] for v in row if v is not None]
    n_expected = len(data["layers"]) * len(data["positions"])
    check(
        "ablation sweep grid fully populated",
        len(cells) == n_expected and not data.get("auto_captured"),
        f"{len(cells)}/{n_expected} cells, auto_captured={data.get('auto_captured')}",
    )
    worst = min(cells)
    check(
        "sweep shows real damage somewhere",
        worst < p_base - 1.0,
        f"worst={worst:.3f} base={p_base:.3f}",
    )

    # recovered metric without a clean baseline -> 400
    r = client.post(
        f"{base}/v1/patch_sweep", json={**sweep, "metric": "recovered"}, timeout=60.0
    )
    check("recovered metric w/o clean -> 400", r.status_code == 400, f"{r.status_code}")

    # streaming parity
    grid_stream: dict[tuple[int, int], float] = {}
    with client.stream(
        "POST", f"{base}/v1/patch_sweep", json={**sweep, "stream": True}, timeout=600.0
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("type") == "cell" and ev.get("value") is not None:
                grid_stream[(ev["layer"], ev["position"])] = ev["value"]
    ok = len(grid_stream) == n_expected and all(
        math.isclose(
            grid_stream[(la, po)], data["grid"][i][j], rel_tol=0, abs_tol=0.15
        )
        for i, la in enumerate(data["layers"])
        for j, po in enumerate(data["positions"])
    )
    check(
        "streaming ablation sweep parity",
        ok,
        f"{len(grid_stream)}/{n_expected} streamed cells",
    )

    return finish()


def finish() -> int:
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
