# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU validation: patched KV must not poison the prefix cache.

A patched request re-forwards from its patch floor and registers its computed
blocks in the prefix cache. Without patch-aware block hashes those blocks are
keyed by tokens alone, so a later UNPATCHED request with the same prompt can be
served the patched KV — silently wrong output. The fix folds a patch-spec hash
into the hashes of blocks at/after the patch floor.

Repro shape (order matters):
  Phase A (fresh engine): unpatched corrupt prompt -> ground-truth logprobs.
  Phase B (fresh engine): capture clean source; run PATCHED corrupt FIRST (its
  blocks get registered); run UNPATCHED corrupt -> poisoned iff != ground truth.

Each phase runs in its own process (engine teardown does not reliably release
GPU memory). The prompts diverge at token 0 with >= 2 full KV blocks after the
divergence — full blocks are the poisoning surface (partial blocks are never
cached, which is why short-prompt validation missed this).

    python tests/gpu_patch_poison_validate.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CLEAN = (
    "France is the country under discussion in this geography lesson. "
    "We will talk about its history, its culture, its food, and its people "
    "at length. The capital city of the country under discussion is"
)
CORRUPT = (
    "Japan is the country under discussion in this geography lesson. "
    "We will talk about its history, its culture, its food, and its people "
    "at length. The capital city of the country under discussion is"
)


def _lp(out) -> dict[int, float]:
    return {int(t): v.logprob for t, v in out.outputs[0].logprobs[0].items()}


def _maxdiff(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    if not keys:
        return float("inf")
    return max(abs(a[k] - b[k]) for k in keys)


def _make_llm(model: str):
    from vllm import LLM

    return LLM(
        model=model,
        enable_patching=True,
        max_patch_slots=64,
        patch_source_cache_bytes=2_000_000_000,
        capture_consumers=[{"name": "patch_source"}],
        enable_prefix_caching=True,  # the poisoning vector
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
    )


def phase_a(model: str, out_path: str) -> None:
    from vllm import SamplingParams

    llm = _make_llm(model)
    out = llm.generate(
        [CORRUPT], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    )[0]
    with open(out_path, "w") as f:
        json.dump(_lp(out), f)


def phase_b(model: str, out_path: str) -> None:
    from vllm import SamplingParams

    llm = _make_llm(model)
    n_layers = llm.llm_engine.model_config.get_total_num_hidden_layers()
    tok = llm.get_tokenizer()
    clean_ids, corrupt_ids = tok.encode(CLEAN), tok.encode(CORRUPT)
    assert len(clean_ids) == len(corrupt_ids)
    d0 = next(i for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids)) if a != b)
    assert len(corrupt_ids) - d0 >= 32, "need >= 2 full blocks after divergence"

    cap = {
        "patch_source": {
            "run": "clean",
            "hooks": {"post_block": "all"},
            "positions": "all_prompt",
        }
    }
    llm.generate([CLEAN], SamplingParams(temperature=0.0, max_tokens=1, capture=cap))
    # Offline LLM has no capture_wait; the write-through is async on a dispatch
    # thread. Give it time to land (the sanity check below catches a miss).
    time.sleep(3.0)

    patch = [
        {
            "layer": n_layers // 2,
            "hook": "post_block",
            "dest_position": pos,
            "source_run": "clean",
            "source_position": pos,
            "alpha": 1.0,
        }
        for pos in range(d0, d0 + 4)
    ]
    patched = _lp(
        llm.generate(
            [CORRUPT],
            SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, patch=patch),
        )[0]
    )
    unpatched_after = _lp(
        llm.generate(
            [CORRUPT], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        )[0]
    )
    with open(out_path, "w") as f:
        json.dump({"patched": patched, "unpatched_after": unpatched_after,
                   "d0": d0, "n_prompt": len(corrupt_ids)}, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--phase", choices=["a", "b"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.phase == "a":
        phase_a(args.model, args.out)
        return
    if args.phase == "b":
        phase_b(args.model, args.out)
        return

    # Orchestrate: each phase in its own process so GPU memory fully releases.
    tmp = os.environ.get("TMPDIR", "/tmp")
    a_path = os.path.join(tmp, "poison_ground.json")
    b_path = os.path.join(tmp, "poison_phase_b.json")
    for phase, path in (("a", a_path), ("b", b_path)):
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--model", args.model,
             "--phase", phase, "--out", path],
            env=os.environ.copy(),
        )
        if r.returncode != 0:
            raise SystemExit(f"phase {phase} failed")

    ground = {int(k): v for k, v in json.load(open(a_path)).items()}
    b = json.load(open(b_path))
    patched = {int(k): v for k, v in b["patched"].items()}
    unpatched_after = {int(k): v for k, v in b["unpatched_after"].items()}

    d_patch = _maxdiff(patched, ground)
    d_poison = _maxdiff(unpatched_after, ground)
    print(f"\nn_prompt={b['n_prompt']} first_diff_token_at={b['d0']}")
    print(f"patch changed output (sanity):     max|patched - ground| = {d_patch:.4g}")
    print(f"unpatched-after-patched vs ground: max|d| = {d_poison:.4g}")

    sane = d_patch > 0.05  # the patch must actually do something
    clean_cache = d_poison < 1e-3  # and must not leak into unpatched runs
    print(f"[{'PASS' if sane else 'FAIL'}] patch alters output (guards no-op)")
    print(f"[{'PASS' if clean_cache else 'FAIL'}] no prefix-cache poisoning")
    print("OVERALL:", "PASS" if (sane and clean_cache) else "FAIL")
    if not (sane and clean_cache):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
