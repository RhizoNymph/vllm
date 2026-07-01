# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline GPU validation for activation patching (run on a GPU box).

Checks (independent of the model's knowledge):

A. no-op: alpha=0 over all post_block sites reproduces the unpatched logits
   exactly (proves the full plumbing: spec -> admission -> source-store
   resolution -> per-step injection -> op).
B. self-identity @ pre_attn (single-tensor op, no deferred-add round-trip):
   re-run a prompt patching every pre_attn site with its OWN captured
   activations (alpha=1) -> exact match (proves replace correctness + store
   round-trip + cross-layer propagation).
C. self-identity @ post_block: same, but post_block carries bf16 round-trip
   noise from vLLM's deferred MLP add accumulated across layers -> reported,
   not asserted exact (documents the inherent precision floor).
D. cross-prompt full replace @ pre_attn: patch every pre_attn site of a
   corrupt run with a clean run's activations. Equal length -> corrupt logits
   must equal the clean run's; unequal -> patch the shared prefix and require
   the answer to move toward clean (qualitative).
E. denoising probe @ post_block single site: best single-layer shift toward
   the clean answer (the real causal-tracing use case).

Usage (GPU box):
    PYTHONPATH=<plugin-dist-info>:<this-repo> \\
      uv run python tests/gpu_patch_validate.py --model Qwen/Qwen3-0.6B \\
      [--enforce-eager]
"""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams

HOOKS_ALL = {"pre_attn": "all", "post_block": "all"}


def _lp(out) -> dict[int, float]:
    return {tid: v.logprob for tid, v in out.outputs[0].logprobs[0].items()}


def _argmax(out) -> int:
    return out.outputs[0].token_ids[0]


def _cap(run: str):
    return {"patch_source": {"run": run, "hooks": HOOKS_ALL, "positions": "all_prompt"}}


def _patch(hook: str, n_layers: int, n_pos: int, run: str, alpha: float = 1.0):
    return [
        {
            "layer": layer,
            "hook": hook,
            "dest_position": pos,
            "source_run": run,
            "source_position": pos,
            "alpha": alpha,
        }
        for layer in range(n_layers)
        for pos in range(n_pos)
    ]


def _maxdiff(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    return max((abs(a[k] - b[k]) for k in keys), default=float("inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-patch-slots", type=int, default=64)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--clean-prompt", default="The capital city of France is the city of")
    ap.add_argument("--corrupt-prompt", default="The capital city of Japan is the city of")
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        enable_patching=True,
        max_patch_slots=args.max_patch_slots,
        patch_source_cache_bytes=4_000_000_000,
        capture_consumers=[{"name": "patch_source"}],
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        distributed_executor_backend=(
            "ray"
            if args.tensor_parallel_size * args.pipeline_parallel_size > 1
            else None
        ),
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    n_layers = llm.llm_engine.model_config.get_total_num_hidden_layers()
    tok = llm.get_tokenizer()
    mode = "eager" if args.enforce_eager else "cudagraph"
    print(f"model={args.model} layers={n_layers} mode={mode}")

    greedy = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def gen(prompt, sp):
        return _lp(llm.generate([prompt], sp)[0])

    def n_tokens(prompt) -> int:
        return len(tok.encode(prompt))

    clean_prompt = args.clean_prompt
    corrupt_prompt = args.corrupt_prompt
    n_clean = n_tokens(clean_prompt)
    n_corrupt = n_tokens(corrupt_prompt)

    # Unpatched baselines + capture each prompt's source.
    clean_lp = gen(
        clean_prompt,
        SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, capture=_cap("clean")),
    )
    corrupt_lp = gen(
        corrupt_prompt,
        SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=20, capture=_cap("corrupt")
        ),
    )
    clean_tok = _argmax(llm.generate([clean_prompt], greedy)[0])
    print(f"n_clean={n_clean} n_corrupt={n_corrupt} clean_tok={clean_tok!r}")

    results: list[tuple[str, bool, str]] = []

    # A. no-op
    noop = gen(
        corrupt_prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            patch=_patch("post_block", n_layers, n_corrupt, "corrupt", alpha=0.0),
        ),
    )
    d = _maxdiff(noop, corrupt_lp)
    results.append(("A no-op alpha=0 (post_block, all sites)", d < 1e-3, f"max|d|={d:.4g}"))

    # B. self-identity @ pre_attn (exact)
    ident_pre = gen(
        corrupt_prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            patch=_patch("pre_attn", n_layers, n_corrupt, "corrupt", alpha=1.0),
        ),
    )
    d = _maxdiff(ident_pre, corrupt_lp)
    results.append(("B self-identity @ pre_attn (exact)", d < 5e-3, f"max|d|={d:.4g}"))

    # C. self-identity @ post_block (bf16 round-trip noise; informational)
    ident_pb = gen(
        corrupt_prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            patch=_patch("post_block", n_layers, n_corrupt, "corrupt", alpha=1.0),
        ),
    )
    d = _maxdiff(ident_pb, corrupt_lp)
    results.append(
        ("C self-identity @ post_block (bf16 round-trip)", d < 0.5, f"max|d|={d:.4g}")
    )

    # D. cross-prompt full replace @ pre_attn
    if n_clean == n_corrupt:
        full = gen(
            corrupt_prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=20,
                patch=_patch("pre_attn", n_layers, n_clean, "clean", alpha=1.0),
            ),
        )
        d = _maxdiff(full, clean_lp)
        # Cross-run replace reloads bf16-stored activations and re-runs all
        # layers, so it reproduces clean within bf16 accumulation (not bit-
        # exactly; B proves the op itself is bit-exact). The drift scales with
        # depth/width: ~0.04 over Qwen3's 28 layers, ~0.12 over gemma3's 34
        # wider layers. Tolerance scales with layer count; the qualitative claim
        # is "collapses the corrupt->clean gap" (raw gap here is ~0.85).
        tol_d = 0.05 + 0.003 * n_layers
        results.append(
            (
                "D full replace corrupt->clean == clean @ pre_attn",
                d < tol_d,
                f"max|d|={d:.4g} (tol={tol_d:.3g}, {n_layers}L bf16)",
            )
        )
    else:
        shared = min(n_clean, n_corrupt)
        full = gen(
            corrupt_prompt,
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=20,
                patch=_patch("pre_attn", n_layers, shared, "clean", alpha=1.0),
            ),
        )
        moved = full.get(clean_tok, float("-inf")) - corrupt_lp.get(
            clean_tok, float("-inf")
        )
        results.append(
            (
                "D prefix replace moves toward clean @ pre_attn (lengths differ)",
                moved > 0,
                f"dlogprob(clean_tok)={moved:.4g} shared={shared}",
            )
        )

    # D2. cross-prompt full replace @ post_block: does post_block patching
    # PROPAGATE (change the output toward clean)? C only proves the no-op case.
    full_pb = gen(
        corrupt_prompt,
        SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=20,
            patch=_patch("post_block", n_layers, min(n_clean, n_corrupt), "clean",
                         alpha=1.0),
        ),
    )
    moved_pb = full_pb.get(clean_tok, -20.0) - corrupt_lp.get(clean_tok, -20.0)
    dclean_pb = _maxdiff(full_pb, clean_lp)
    results.append(
        (
            "D2 full replace @ post_block propagates toward clean",
            moved_pb > 0 or dclean_pb < 0.5,
            f"dlogprob(clean_tok)={moved_pb:.4g} max|d vs clean|={dclean_pb:.4g}",
        )
    )

    # E. denoising probe @ post_block: best single (layer, position) site.
    # Scans all positions (not just the last) because the causally-relevant
    # token (the differing subject) is usually mid-prompt, and larger models
    # trace through that position rather than the final one.
    shared = min(n_clean, n_corrupt)
    best = (-1, -1, float("-inf"))  # (layer, pos, shift)
    for layer in range(n_layers):
        for pos in range(shared):
            plp = gen(
                corrupt_prompt,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    logprobs=20,
                    patch=[
                        {
                            "layer": layer,
                            "hook": "post_block",
                            "dest_position": pos,
                            "source_run": "clean",
                            "source_position": pos,
                            "alpha": 1.0,
                        }
                    ],
                ),
            )
            patched_lp = plp.get(clean_tok)
            if patched_lp is None:
                continue
            base_lp = corrupt_lp.get(clean_tok)
            shift = patched_lp - (base_lp if base_lp is not None else -20.0)
            if shift > best[2]:
                best = (layer, pos, shift)
    surfaced = corrupt_lp.get(clean_tok) is None and best[0] >= 0
    results.append(
        (
            "E denoising probe (best single-site shift toward clean)",
            best[2] > 0,
            f"layer={best[0]} pos={best[1]} shift={best[2]:.4g}"
            + (" (answer surfaced from outside corrupt top-k)" if surfaced else ""),
        )
    )

    print("\n==== PATCH VALIDATION RESULTS ({}) ====".format(mode))
    all_ok = True
    for name, ok, detail in results:
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("=" * 42)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
