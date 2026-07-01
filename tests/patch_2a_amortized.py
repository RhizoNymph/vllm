# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure amortized 2a: pre-capture the trunk once, then sweep (long prompt).

This is how 2a is meant to be used — the corrupt trunk is captured once and
reused across sweeps, so its cost is not paid per sweep. Bigger per-layer groups
(more positions) also reduce the fragmentation penalty.

    python tests/patch_2a_amortized.py --base-url http://localhost:8123/v1
"""

from __future__ import annotations

import argparse
import time

import httpx
from openai import OpenAI

_F = ("In the study of large language models, researchers examine how "
      "information flows through the residual stream across many layers and "
      "token positions during inference. ")
CLEAN = _F * 90 + "The capital city of France is the city of"
CORRUPT = _F * 90 + "The capital city of Japan is the city of"
LAYERS = [4, 10, 16, 22]


def capture(oc, model, run, prompt, layers):
    cap = {"patch_source": {"run": run, "hooks": {"post_block": layers},
                            "positions": "all_prompt"}}
    t = time.perf_counter()
    oc.completions.create(model=model, prompt=prompt, max_tokens=1,
                          temperature=0.0,
                          extra_body={"capture": cap, "capture_wait": True})
    return (time.perf_counter() - t) * 1e3


def sweep(hx, base, model, positions, mode, trunk_run=None):
    payload = {"model": model, "prompt": CORRUPT, "source_run": "clean",
               "hook": "post_block", "layers": LAYERS, "positions": positions,
               "answer_token": " Paris", "metric": "logprob", "mode": mode,
               "trunk_run": trunk_run}
    t = time.perf_counter()
    r = hx.post(f"{base}/patch_sweep", json=payload, timeout=600.0)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    oc = OpenAI(base_url=base, api_key="x")
    hx = httpx.Client()

    capture(oc, args.model, "clean", CLEAN, list(range(28)))
    tl = sorted({L - 1 for L in LAYERS})
    t_trunk = capture(oc, args.model, "trunk", CORRUPT, tl)
    nlong = oc.completions.create(model=args.model, prompt=CORRUPT,
                                  max_tokens=1).usage.prompt_tokens
    positions = [int(i * (nlong - 1) / 39) for i in range(40)]
    ncell = len(LAYERS) * len(positions)
    print(f"n={nlong}  grid {len(LAYERS)}x{len(positions)}={ncell} cells  "
          f"trunk-capture(one-time)={t_trunk:.0f}ms")

    g1, t1 = sweep(hx, base, args.model, positions, "level1")
    # warm 2a path once, then measure (trunk pre-captured, reused)
    sweep(hx, base, args.model, positions, "2a", trunk_run="trunk")
    g2, t2 = sweep(hx, base, args.model, positions, "2a", trunk_run="trunk")

    max_d, n = 0.0, 0
    for ra, rb in zip(g1["grid"], g2["grid"]):
        for va, vb in zip(ra, rb):
            if va is not None and vb is not None:
                max_d = max(max_d, abs(va - vb)); n += 1
    print(f"argmax level1={g1['argmax']}")
    print(f"argmax 2a    ={g2['argmax']}")
    print(f"cells compared={n}  max|d|={max_d:.4g}")
    print(f"level1        = {t1:.0f} ms")
    print(f"2a (amortized)= {t2:.0f} ms   ({1 - t2/t1:+.0%} vs level1)")
    print(f"2a (+trunk)   = {t2 + t_trunk:.0f} ms   "
          f"({1 - (t2+t_trunk)/t1:+.0%} incl one-time trunk)")


if __name__ == "__main__":
    main()
