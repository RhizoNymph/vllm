# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live GPU validation of directional projection clamping.

Two modes:

``--mode kernel`` (no server; needs a CUDA device):
    Triton-vs-eager parity for ``apply_clamp`` / ``apply_clamp_block`` over
    random inputs plus the exactness edges (row-0 passthrough, inactive
    flag, zero-padded dirs, in-bounds delta == 0 bitwise).

``--mode http`` (default) against a ``--enable-steering`` server:

    python tests/gpu_clamp_validate.py --base-url http://localhost:8412

    1.  Wide-bounds clamp == baseline logprob EXACTLY (in-bounds delta is
        exactly zero, so hidden states are bitwise unchanged).
    2.  strength=0 pin == baseline exactly (same argument).
    3.  An aggressive pin (value=40, random unit dir) visibly degrades the
        answer logprob — the clamp bites.
    4.  Additive-then-clamp identity: (steer +20·v̂) + (pin c) matches
        (pin c) alone — clamp-last erases the additive component along v̂ —
        while (steer +20·v̂) alone lands far away.
    5.  One-sided max bound below the natural projection == pin to that
        bound (matching logprobs); a max bound far above == baseline.
    6.  Decode-only clamps: first sampled token matches baseline (prefill
        untouched), continuation diverges under an aggressive pin.
    7.  Churn: concurrent requests with distinct clamp configs (and one
        vectors+clamps) all complete across the prefill->decode boundary.
    8.  K-cap (>max_clamp_directions dirs) and wrong-width dirs -> HTTP 400.
    9.  Global clamps via /v1/steering/set on an otherwise steering-free
        server (exercises the has_global_clamps short-circuit fix):
        generation shifts, then /v1/steering/clear restores baseline
        exactly (exercises the active->inactive clamp-flag zeroing).
    10. Named module with a clamps tier: register, steering_name request
        matches the inline-clamp equivalent, unregister restores 400.

Prints one PASS/FAIL line per check and exits non-zero on any FAIL.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import sys

import numpy as np

CLEAN = "The capital of France is"
ANSWER = " Paris"
RESULTS: list[tuple[str, bool, str]] = []
MODEL = ""


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def summarize() -> int:
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# kernel mode
# ---------------------------------------------------------------------------


def run_kernel_mode() -> int:
    import torch

    from vllm.model_executor.layers.clamp import apply_clamp, apply_clamp_block

    assert torch.cuda.is_available(), "kernel mode needs CUDA"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    rows, k, hidden, n = 8, 4, 1024, 33

    def make(dtype):
        dirs = torch.zeros(rows, k, hidden, dtype=dtype)
        # rows 1..5 get 1..3 random unit dirs; leave some K slots zero-padded
        for r in range(1, 6):
            for j in range(1 + r % 3):
                v = torch.randn(hidden)
                dirs[r, j] = (v / v.norm()).to(dtype)
        bounds = torch.empty(rows, k, 2, dtype=torch.float32)
        bounds[..., 0] = -float("inf")
        bounds[..., 1] = float("inf")
        bounds[1, 0] = torch.tensor([0.0, 0.0])  # pin 0
        bounds[2, 0] = torch.tensor([-1.0, 1.0])  # range
        bounds[3, 0, 1] = 2.0  # one-sided max
        bounds[4, 0] = torch.tensor([5.0, 5.0])  # pin 5
        # poison a zero-padded slot's bounds: must stay a no-op
        bounds[5, 3] = torch.tensor([7.0, 7.0])
        strength = torch.ones(rows, k, dtype=torch.float32)
        strength[4, 0] = 0.5
        index = torch.randint(0, 6, (n,), dtype=torch.long)
        index[0] = 0  # sentinel row
        active = torch.ones(1, dtype=torch.bool)
        return dirs, bounds, strength, index, active

    for dtype in (torch.float32, torch.bfloat16):
        dirs, bounds, strength, index, active = make(dtype)
        h = torch.randn(n, hidden, dtype=dtype)
        res = torch.randn(n, hidden, dtype=dtype)

        cpu = apply_clamp(h, dirs, bounds, strength, index, active)
        gpu = apply_clamp(
            h.to(dev),
            dirs.to(dev),
            bounds.to(dev),
            strength.to(dev),
            index.to(dev),
            active.to(dev),
        ).cpu()
        tol = 1e-5 if dtype == torch.float32 else 1e-2
        max_err = (cpu.float() - gpu.float()).abs().max().item()
        check(f"apply_clamp parity ({dtype})", max_err < tol, f"max_err={max_err:.2e}")

        cpu_b = apply_clamp_block(h, res, dirs, bounds, strength, index, active)
        gpu_b = apply_clamp_block(
            h.to(dev),
            res.to(dev),
            dirs.to(dev),
            bounds.to(dev),
            strength.to(dev),
            index.to(dev),
            active.to(dev),
        ).cpu()
        max_err_b = (cpu_b.float() - gpu_b.float()).abs().max().item()
        check(
            f"apply_clamp_block parity ({dtype})",
            max_err_b < tol,
            f"max_err={max_err_b:.2e}",
        )

        # Row-0 sentinel tokens are bitwise passthrough on GPU.
        sent = index.to(dev) == 0
        check(
            f"row-0 bitwise passthrough ({dtype})",
            torch.equal(gpu.to(dev)[sent], h.to(dev)[sent]),
        )

        # Inactive flag: whole batch bitwise passthrough.
        inact = torch.zeros(1, dtype=torch.bool, device=dev)
        gpu_off = apply_clamp(
            h.to(dev),
            dirs.to(dev),
            bounds.to(dev),
            strength.to(dev),
            index.to(dev),
            inact,
        )
        check(
            f"inactive bitwise passthrough ({dtype})",
            torch.equal(gpu_off, h.to(dev)),
        )

        # Semantic: pinned projection lands on the target (GPU, fp32 dirs).
        if dtype == torch.float32:
            row4 = index.to(dev) == 4
            if row4.any():
                proj = gpu.to(dev)[row4].float() @ dirs[4, 0].to(dev).float()
                proj_in = h.to(dev)[row4].float() @ dirs[4, 0].to(dev).float()
                expect = proj_in + 0.5 * (5.0 - proj_in)  # strength 0.5 toward 5
                err = (proj - expect).abs().max().item()
                check("strength-0.5 pin projection", err < 1e-3, f"err={err:.2e}")

    return summarize()


# ---------------------------------------------------------------------------
# http mode
# ---------------------------------------------------------------------------


def _unit(hidden: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden)
    return (v / np.linalg.norm(v)).tolist()


def _pack_vectors(layer: int, row: np.ndarray, hook: str = "post_block") -> dict:
    row = np.ascontiguousarray(row[None, :], dtype=np.float32)
    return {
        hook: {
            "dtype": "float32",
            "shape": list(row.shape),
            "layer_indices": [layer],
            "data": base64.b64encode(row.tobytes()).decode(),
        }
    }


def run_http_mode(base: str, layer: int, hidden: int) -> int:
    import httpx

    client = httpx.Client(timeout=180.0)
    global MODEL
    MODEL = client.get(f"{base}/v1/models").json()["data"][0]["id"]
    hook = "post_block"

    def clamps(entries: list[dict]) -> dict:
        return {hook: {str(layer): entries}}

    def complete(max_tokens: int = 1, **extra) -> dict:
        body = {
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 20,
            **extra,
        }
        r = client.post(f"{base}/v1/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]

    def answer_lp(choice: dict) -> float:
        return choice["logprobs"]["top_logprobs"][0].get(ANSWER, -50.0)

    v_pin = _unit(hidden, seed=1)
    v_add = np.asarray(_unit(hidden, seed=2))

    print(f"model={MODEL} layer={layer} hook={hook} hidden={hidden}")
    base_choice = complete()
    p_base = answer_lp(base_choice)
    print(f"baseline logprob({ANSWER!r}) = {p_base:.4f}")

    # -- 1. wide bounds == baseline exactly ---------------------------------
    p_wide = answer_lp(
        complete(steering_clamps=clamps([{"vector": v_pin, "min": -1e6, "max": 1e6}]))
    )
    check(
        "wide-bounds clamp == baseline (exact)",
        abs(p_wide - p_base) < 1e-6,
        f"{p_base:.6f} vs {p_wide:.6f}",
    )

    # -- 2. strength=0 == baseline exactly -----------------------------------
    p_s0 = answer_lp(
        complete(
            steering_clamps=clamps([{"vector": v_pin, "value": 40.0, "strength": 0.0}])
        )
    )
    check(
        "strength-0 pin == baseline (exact)",
        abs(p_s0 - p_base) < 1e-6,
        f"{p_base:.6f} vs {p_s0:.6f}",
    )

    # -- 3. aggressive pin bites ---------------------------------------------
    p_pin40 = answer_lp(
        complete(steering_clamps=clamps([{"vector": v_pin, "value": 40.0}]))
    )
    check(
        "aggressive pin (value=40) degrades answer",
        p_pin40 < p_base - 1.0,
        f"{p_base:.3f} -> {p_pin40:.3f}",
    )

    # -- 4. additive-then-clamp identity -------------------------------------
    c = 1.0
    p_pin_only = answer_lp(
        complete(
            steering_clamps={
                hook: {str(layer): [{"vector": v_add.tolist(), "value": c}]}
            }
        )
    )
    p_add_only = answer_lp(
        complete(steering_vectors=_pack_vectors(layer, 60.0 * v_add, hook))
    )
    p_add_clamp = answer_lp(
        complete(
            steering_vectors=_pack_vectors(layer, 60.0 * v_add, hook),
            steering_clamps={
                hook: {str(layer): [{"vector": v_add.tolist(), "value": c}]}
            },
        )
    )
    check(
        "clamp-last erases additive component (identity)",
        abs(p_add_clamp - p_pin_only) < 0.15 and abs(p_add_only - p_pin_only) > 1.0,
        f"pin={p_pin_only:.3f} add+pin={p_add_clamp:.3f} add-only={p_add_only:.3f}",
    )

    # -- 5. one-sided bounds --------------------------------------------------
    p_cap_low = answer_lp(
        complete(steering_clamps=clamps([{"vector": v_pin, "max": -30.0}]))
    )
    p_pin_low = answer_lp(
        complete(steering_clamps=clamps([{"vector": v_pin, "value": -30.0}]))
    )
    p_cap_high = answer_lp(
        complete(steering_clamps=clamps([{"vector": v_pin, "max": 1e5}]))
    )
    check(
        "one-sided max below natural proj == pin at bound",
        abs(p_cap_low - p_pin_low) < 1e-6 and abs(p_cap_high - p_base) < 1e-6,
        f"cap={p_cap_low:.3f} pin={p_pin_low:.3f} high-cap={p_cap_high:.6f} "
        f"base={p_base:.6f}",
    )

    # -- 6. decode-only clamps -----------------------------------------------
    # Greedy continuations are sticky: a decode-side perturbation can take
    # several tokens to change the argmax path, so use a 24-token window.
    base_multi = complete(max_tokens=24)
    dec_multi = complete(
        max_tokens=24,
        decode_steering_clamps=clamps([{"vector": v_pin, "value": 40.0}]),
    )
    base_first = base_multi["logprobs"]["tokens"][0]
    dec_first = dec_multi["logprobs"]["tokens"][0]
    check(
        "decode-only clamp: prefill-sampled token unchanged, text diverges",
        base_first == dec_first and base_multi["text"] != dec_multi["text"],
        f"first={dec_first!r} texts_equal={base_multi['text'] == dec_multi['text']}",
    )

    # -- 7. churn across prefill->decode with distinct configs ---------------
    def one_churn(i: int) -> bool:
        body: dict = {
            "model": MODEL,
            "prompt": f"{CLEAN} (variant {i})",
            "max_tokens": 16,
            "temperature": 0.0,
        }
        body["steering_clamps"] = clamps(
            [{"vector": _unit(hidden, seed=100 + i), "value": float(i)}]
        )
        if i % 3 == 0:  # mix in vectors+clamps
            body["steering_vectors"] = _pack_vectors(
                layer, 0.5 * np.asarray(_unit(hidden, seed=200 + i)), hook
            )
        r = client.post(f"{base}/v1/completions", json=body)
        return r.status_code == 200 and bool(r.json()["choices"][0]["text"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        oks = list(ex.map(one_churn, range(6)))
    check("6-way distinct-clamp churn completes", all(oks), f"{sum(oks)}/6 ok")

    # -- 8. rejection: K cap and wrong width ---------------------------------
    r_cap = client.post(
        f"{base}/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "steering_clamps": clamps(
                [
                    {"vector": _unit(hidden, seed=300 + j), "value": 0.0}
                    for j in range(5)
                ]
            ),
        },
    )
    r_width = client.post(
        f"{base}/v1/completions",
        json={
            "model": MODEL,
            "prompt": CLEAN,
            "max_tokens": 1,
            "steering_clamps": clamps([{"vector": [1.0, 0.0], "value": 0.0}]),
        },
    )
    check(
        "over-K and wrong-width clamp specs -> 400",
        r_cap.status_code == 400 and r_width.status_code == 400,
        f"cap={r_cap.status_code} width={r_width.status_code}",
    )

    # -- 9. global clamps (short-circuit fix + clear zeroing) -----------------
    r_set = client.post(
        f"{base}/v1/steering/set",
        json={"clamps": {hook: {str(layer): [{"vector": v_pin, "value": 40.0}]}}},
    )
    p_glob = answer_lp(complete())
    r_clear = client.post(f"{base}/v1/steering/clear")
    p_after = answer_lp(complete())
    check(
        "global-clamps-only set shifts output; clear restores exactly",
        r_set.status_code == 200
        and r_clear.status_code == 200
        and p_glob < p_base - 1.0
        and abs(p_after - p_base) < 1e-6,
        f"set={r_set.status_code} glob={p_glob:.3f} after={p_after:.6f} "
        f"base={p_base:.6f}",
    )

    # -- 10. named module with clamps tier ------------------------------------
    r_reg = client.post(
        f"{base}/v1/steering/modules/register",
        json={
            "name": "clampmod",
            "clamps": {hook: {str(layer): [{"vector": v_pin, "value": 40.0}]}},
        },
    )
    p_mod = answer_lp(complete(steering_name="clampmod"))
    r_unreg = client.post(
        f"{base}/v1/steering/modules/unregister", json={"name": "clampmod"}
    )
    check(
        "named clamp module matches inline pin",
        r_reg.status_code == 200
        and r_unreg.status_code == 200
        and abs(p_mod - p_pin40) < 0.15,
        f"reg={r_reg.status_code} mod={p_mod:.3f} inline={p_pin40:.3f}",
    )

    return summarize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["http", "kernel"], default="http")
    ap.add_argument("--base-url", default="http://localhost:8412")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--hidden", type=int, default=1024)
    args = ap.parse_args()
    if args.mode == "kernel":
        return run_kernel_mode()
    return run_http_mode(args.base_url.rstrip("/"), args.layer, args.hidden)


if __name__ == "__main__":
    sys.exit(main())
