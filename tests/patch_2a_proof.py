# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Qwen3 Level-2 (2a) proof: mid-stack trunk re-entry == Level-1 patch.

Runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager) so we can
capture the merged residual stream entering each layer and inject re-entry
directly, without the full continuous-batching / KV-reuse integration.

For a cell (layer L, position p):
  * Level-1 reference: full forward, overwrite the merged residual at (L, p)
    with the clean source (== patching pre_attn[L] at p), run all 28 layers.
  * 2a: enter the stack AT layer L with the cached corrupt trunk residual
    (skip layers 0..L-1), overwrite position p with the clean source, run
    layers L..end.
Both feed layers >= L identical inputs, so the answer logprob must match
bit-for-bit -- that is the correctness proof. Compute saved = L / 28.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python tests/patch_2a_proof.py
"""

from __future__ import annotations

import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.models import qwen2 as _q2

CLEAN = "The capital city of France is the city of"
CORRUPT = "The capital city of Japan is the city of"


def get_model(llm: LLM):
    """Find the top CausalLM module across V1 in-process accessor paths."""
    seen = set()
    stack = [llm.llm_engine]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "model") and hasattr(getattr(obj, "model"), "model"):
            m = obj.model
            if hasattr(m.model, "layers"):
                return m
        for attr in ("engine_core", "model_executor", "driver_worker",
                     "model_runner", "collective_rpc", "worker", "engine"):
            if hasattr(obj, attr):
                try:
                    stack.append(getattr(obj, attr))
                except Exception:  # noqa: BLE001
                    pass
    raise RuntimeError("could not locate model")


def merged_inputs(layers):
    """Attach pre-hooks capturing merged residual (h + r) entering each layer."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(_module, args):
            h, r = args[1], args[2]
            merged = h if r is None else h + r
            captured[i] = merged.detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(mk(i)))
    return captured, handles


def answer_lp(out, tok_id):
    lps = out.outputs[0].logprobs[0]
    e = lps.get(tok_id)
    return float(e.logprob) if e is not None else None


def dist(out) -> dict[int, float]:
    """Full first-token logprob distribution {token_id: logprob}."""
    return {tid: float(e.logprob) for tid, e in out.outputs[0].logprobs[0].items()}


def dist_diff(a: dict[int, float], b: dict[int, float]) -> tuple[float, bool]:
    """Max |logprob| diff over shared tokens; whether top-token sets match."""
    shared = set(a) & set(b)
    md = max((abs(a[t] - b[t]) for t in shared), default=0.0)
    return md, set(a) == set(b)


def main() -> None:
    llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.55)
    model = get_model(llm)
    layers = model.model.layers
    n_layers = len(layers)
    tok = llm.get_tokenizer()
    ans_id = tok.encode(" Paris")[0]
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    # 1. capture merged residual entering each layer, clean and corrupt
    cap, h = merged_inputs(layers)
    clean_out = llm.generate([CLEAN], sp, use_tqdm=False)[0]
    clean_merged = {i: v for i, v in cap.items()}
    for hh in h:
        hh.remove()
    cap, h = merged_inputs(layers)
    corrupt_out = llm.generate([CORRUPT], sp, use_tqdm=False)[0]
    corrupt_merged = {i: v for i, v in cap.items()}
    for hh in h:
        hh.remove()

    n = len(corrupt_out.prompt_token_ids)

    def fmt(x):
        return f"{x:.4f}" if x is not None else "  n/a"

    print(f"prompt_n={n}  clean P(Paris)={fmt(answer_lp(clean_out, ans_id))}  "
          f"corrupt P(Paris)={fmt(answer_lp(corrupt_out, ans_id))}")

    def level1(L, p):
        """Full forward; overwrite merged residual at (L, p) with clean src."""
        src = clean_merged[L][p].clone()

        def hook(_m, args):
            pos, hs, res = args
            hs = hs.clone()
            res = torch.zeros_like(hs) if res is None else res.clone()
            hs[p] = src
            res[p] = 0
            return (pos, hs, res)

        handle = layers[L].register_forward_pre_hook(hook)
        try:
            out = llm.generate([CORRUPT], sp, use_tqdm=False)[0]
        finally:
            handle.remove()
        return dist(out), answer_lp(out, ans_id)

    def twoa(L, p):
        """Enter at layer L with cached corrupt trunk; patch position p."""
        entry = corrupt_merged[L].clone()
        entry[p] = clean_merged[L][p]
        _q2.set_patch_2a_entry((L, entry))
        try:
            out = llm.generate([CORRUPT], sp, use_tqdm=False)[0]
        finally:
            _q2.set_patch_2a_entry(None)
        return dist(out), answer_lp(out, ans_id)

    print(f"\n{'L':>3} {'p':>3} {'l1 P(ans)':>10} {'2a P(ans)':>10} "
          f"{'max|dist d|':>12} {'toks==':>7} {'layers':>7} {'saved':>6}")
    worst = 0.0
    sets_ok = True
    for L in [2, 7, 14, 20, 26]:
        for p in [3, n - 1]:
            d1, a1 = level1(L, p)
            d2, a2 = twoa(L, p)
            md, same = dist_diff(d1, d2)
            worst = max(worst, md)
            sets_ok = sets_ok and same
            fa1 = f"{a1:>10.5f}" if a1 is not None else f"{'n/a':>10}"
            fa2 = f"{a2:>10.5f}" if a2 is not None else f"{'n/a':>10}"
            print(f"{L:>3} {p:>3} {fa1} {fa2} {md:>12.2e} {str(same):>7} "
                  f"{n_layers - L:>7} {L / n_layers:>5.0%}")
    ok = worst < 1e-3 and sets_ok
    print(f"\nmax |level1 - 2a| logprob = {worst:.3e}; token-sets identical: "
          f"{sets_ok}  ({'PASS: re-entry == Level-1' if ok else 'FAIL'})")

    # ---- measured compute win on a long prompt (same harness) --------------
    import statistics
    import time
    filler = ("In the study of large language models, researchers examine how "
              "information flows through the residual stream across many layers "
              "and token positions during inference. ")
    long_prompt = filler * 90 + CORRUPT
    cap, h = merged_inputs(layers)
    lout = llm.generate([long_prompt], sp, use_tqdm=False)[0]
    long_merged = {i: v for i, v in cap.items()}
    for hh in h:
        hh.remove()
    ln = len(lout.prompt_token_ids)

    def timed(entry):
        _q2.set_patch_2a_entry(entry)
        try:
            llm.generate([long_prompt], sp, use_tqdm=False)  # warm
            xs = []
            for _ in range(9):
                t0 = time.perf_counter()
                llm.generate([long_prompt], sp, use_tqdm=False)
                xs.append(time.perf_counter() - t0)
        finally:
            _q2.set_patch_2a_entry(None)
        return statistics.median(xs) * 1e3

    t_full = timed(None)
    print(f"\nlong prompt n={ln}: Level-1 full forward = {t_full:.1f} ms")
    for L in [7, 14, 21]:
        t_2a = timed((L, long_merged[L].clone()))
        print(f"  2a enter@L={L:>2} (run {n_layers - L} layers) = {t_2a:7.1f} ms"
              f"  -> {1 - t_2a / t_full:5.0%} faster")


if __name__ == "__main__":
    main()
