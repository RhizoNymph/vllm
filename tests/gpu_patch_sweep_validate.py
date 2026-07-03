# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live validation of the /v1/patch_sweep endpoint vs the per-cell client path.

Run against a vllm server started with --enable-patching + patch_source.
The server-side sweep must produce the same grid as the per-cell fan-out.

    python tests/gpu_patch_sweep_validate.py --base-url http://localhost:8123/v1
"""

from __future__ import annotations

import argparse
import asyncio

import vllm.entrypoints.serve.patch.client as pc


def _cmp(a, b, tol=0.3) -> tuple[bool, float, int]:
    """Compare two grids cell-by-cell; return (ok, max_diff, n_compared).

    Tolerance is loose: the server path batches all cells at once while the
    per-cell path batches ~16 at a time, so identical patches land in different
    batch compositions and vLLM (not batch-invariant by default) returns
    slightly different logprobs. The structural check is the argmax match."""
    max_d = 0.0
    n = 0
    mismatch_none = 0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if va is None and vb is None:
                continue
            if (va is None) != (vb is None):
                # Top-k boundary flicker: the answer is near rank-k and flips
                # in/out of the returned logprobs between batch compositions.
                mismatch_none += 1
                continue
            n += 1
            max_d = max(max_d, abs(va - vb))
    print(f"(top-k boundary None-mismatches: {mismatch_none})")
    return (max_d <= tol), max_d, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--answer", default=" Paris")
    args = ap.parse_args()

    study = pc.PatchStudy(model=args.model, base_url=args.base_url, hook="post_block")
    clean = study.capture_clean(
        "The capital city of France is the city of",
        run="paris",
        answer_token=args.answer,
    )
    print(f"clean: run={clean.run_id} n_prompt={clean.num_prompt_tokens} "
          f"logprob={clean.clean_logprob}")

    corrupt = "The capital city of Japan is the city of"
    layers = list(range(0, 28, 4))
    positions = list(range(0, clean.num_prompt_tokens))

    per_cell = asyncio.run(
        study.sweep_layers_positions(
            corrupt, run="paris", layers=layers, positions=positions,
            answer_token=args.answer, metric="logprob", server_side=False,
        )
    )
    server = asyncio.run(
        study.sweep_layers_positions(
            corrupt, run="paris", layers=layers, positions=positions,
            answer_token=args.answer, metric="logprob", server_side=True,
        )
    )

    ok, max_d, n = _cmp(per_cell.grid, server.grid)
    argmax_match = per_cell.argmax_cell() == server.argmax_cell()
    print(f"per-cell argmax: {per_cell.argmax_cell()}")
    print(f"server   argmax: {server.argmax_cell()}")
    print(f"grids close: {ok}  max|d|={max_d:.4g}  cells_compared={n}")
    print(f"argmax match: {argmax_match}")
    passed = argmax_match and ok and n > 0
    print("OVERALL:", "PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
