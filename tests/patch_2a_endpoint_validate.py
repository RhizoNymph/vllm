# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the 2a sweep endpoint == Level-1 endpoint (+ timing).

Server must run with --enable-patching --capture-consumers patch_source
--enforce-eager --no-enable-prefix-caching (2a recomputes all positions at
layers >= L, so prefix reuse must be off).

    python tests/patch_2a_endpoint_validate.py --base-url http://localhost:8123/v1
"""

from __future__ import annotations

import argparse
import time

import httpx
from openai import OpenAI

CLEAN = "The capital city of France is the city of"
CORRUPT = "The capital city of Japan is the city of"
_FILLER = ("In the study of large language models, researchers examine how "
           "information flows through the residual stream across many layers "
           "and token positions during inference. ")
CLEAN_LONG = _FILLER * 90 + CLEAN
CORRUPT_LONG = _FILLER * 90 + CORRUPT


def capture_clean(oc, model, run, n_layers, prompt) -> None:
    cap = {"patch_source": {"run": run,
                            "hooks": {"post_block": list(range(n_layers))},
                            "positions": "all_prompt"}}
    oc.completions.create(model=model, prompt=prompt, max_tokens=1,
                          temperature=0.0,
                          extra_body={"capture": cap, "capture_wait": True})


def sweep(hx, base, model, layers, mode, prompt=CORRUPT, positions="all_prompt",
          run="cleanrun"):
    payload = {
        "model": model, "prompt": prompt, "source_run": run,
        "hook": "post_block", "layers": layers, "positions": positions,
        "answer_token": " Paris", "metric": "logprob", "logprobs": 20,
        "mode": mode,
    }
    t0 = time.perf_counter()
    r = hx.post(f"{base}/patch_sweep", json=payload, timeout=600.0)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def cmp_grids(a, b, tol=0.3):
    max_d, n, none_mm = 0.0, 0, 0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if va is None and vb is None:
                continue
            if (va is None) != (vb is None):
                none_mm += 1
                continue
            n += 1
            max_d = max(max_d, abs(va - vb))
    return max_d, n, none_mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    oc = OpenAI(base_url=base, api_key="unused")
    hx = httpx.Client()

    n_layers = 28
    layers = [2, 7, 14, 21, 26]

    # --- short prompt: correctness (2a grid == Level-1 grid) ---
    capture_clean(oc, args.model, "cleanrun", n_layers, CLEAN)
    g1, t1 = sweep(hx, base, args.model, layers, "level1")
    g2, t2 = sweep(hx, base, args.model, layers, "2a")
    max_d, n, none_mm = cmp_grids(g1["grid"], g2["grid"])
    print("=== SHORT (correctness) ===")
    print(f"level1 argmax: {g1['argmax']}")
    print(f"2a     argmax: {g2['argmax']}")
    print(f"cells compared: {n}  max|d|={max_d:.4g}  none-mismatches={none_mm}")
    print(f"level1={t1*1e3:.0f}ms  2a={t2*1e3:.0f}ms  ({1 - t2/t1:+.0%})")
    argmax_ok = (g1["argmax"] or {}).get("layer") == (g2["argmax"] or {}).get("layer") \
        and (g1["argmax"] or {}).get("position") == (g2["argmax"] or {}).get("position")
    corr_ok = argmax_ok and max_d <= 0.3 and n > 0

    # --- long prompt: measured win (timing) ---
    capture_clean(oc, args.model, "cleanlong", n_layers, CLEAN_LONG)
    nlong = oc.completions.create(model=args.model, prompt=CORRUPT_LONG,
                                  max_tokens=1).usage.prompt_tokens
    lpos = [0, nlong // 4, nlong // 2, 3 * nlong // 4, nlong - 1]
    lg1, lt1 = sweep(hx, base, args.model, layers, "level1",
                     prompt=CORRUPT_LONG, positions=lpos, run="cleanlong")
    lg2, lt2 = sweep(hx, base, args.model, layers, "2a",
                     prompt=CORRUPT_LONG, positions=lpos, run="cleanlong")
    lmax_d, ln, lnone = cmp_grids(lg1["grid"], lg2["grid"])
    print(f"\n=== LONG n={nlong} (timing) ===")
    print(f"cells compared: {ln}  max|d|={lmax_d:.4g}  none-mismatches={lnone}")
    print(f"level1={lt1*1e3:.0f}ms  2a={lt2*1e3:.0f}ms  ({1 - lt2/lt1:+.0%} faster)")

    ok = corr_ok and lmax_d <= 0.3
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
