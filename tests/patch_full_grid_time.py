# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How long is a FULL small-prompt sweep across every (layer, hook, position)?

Answers: does the ~3x 2a small-prompt regression matter in absolute terms? If a
full combinatorial small-prompt sweep is already sub-second under Level 1, a 3x
regression is still sub-second and the regression is a red herring.

    python tests/patch_full_grid_time.py --base-url http://localhost:8123/v1
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import statistics
import sys
import time
from pathlib import Path

_CLIENT = (Path(__file__).resolve().parent.parent / "examples"
           / "online_serving" / "openai_patch_client.py")
spec = importlib.util.spec_from_file_location("openai_patch_client", _CLIENT)
pc = importlib.util.module_from_spec(spec)
sys.modules["openai_patch_client"] = pc
spec.loader.exec_module(pc)

SHORT = "The capital city of Japan is the city of"
HOOKS = ["pre_attn", "post_attn", "post_block"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    n_layers = 28
    layers = list(range(n_layers))

    # capture the clean run once, all hooks x all layers
    study = pc.PatchStudy(model=args.model, base_url=args.base_url,
                          hook="post_block")
    for hook in HOOKS:
        study.capture_clean(SHORT, run="fg", layers=layers, hook=hook,
                            positions="all_prompt", answer_token=" Tokyo")
    # positions resolved from prompt length
    import openai
    oc = openai.OpenAI(base_url=args.base_url, api_key="unused")
    n_prompt = oc.completions.create(model=args.model, prompt=SHORT,
                                     max_tokens=1).usage.prompt_tokens
    positions = list(range(n_prompt))
    ncell = n_layers * len(HOOKS) * len(positions)

    async def full_sweep():
        for hook in HOOKS:
            await study.sweep_layers_positions(
                SHORT, run="fg", layers=layers, positions=positions,
                hook=hook, answer_token=" Tokyo", metric="logprob",
                server_side=True)

    asyncio.run(full_sweep())  # warm
    ts = sorted(
        (lambda: (t0 := time.perf_counter(), asyncio.run(full_sweep()),
                  time.perf_counter() - t0)[-1])()
        for _ in range(3)
    )
    tot = statistics.median(ts) * 1e3
    print(f"\nFULL small-prompt grid: {n_layers} layers x {len(HOOKS)} hooks "
          f"x {len(positions)} positions = {ncell} cells")
    print(f"  Level-1 total = {tot:.1f} ms  ({tot/ncell:.2f} ms/cell)")
    print(f"  implied 2a (~3x small regression) ~= {tot*3/1000:.2f} s")


if __name__ == "__main__":
    main()
