# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-A2: measure the batch-fragmentation penalty that 2a must pay.

Level 1 runs the whole grid as ONE dense batch (all cells share a full forward,
differing only in patch index). 2a cannot: cells with different entry-layers L
can't share a forward, so it must group by L -> one (smaller) batched forward per
layer. This script measures that grouping penalty by comparing a single all-layer
server-side sweep against the same grid issued as one server-side sweep per layer.

    python tests/patch_fragmentation.py --base-url http://localhost:8123/v1
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

FILLER = ("In the study of large language models, researchers examine how "
          "information flows through the residual stream across many layers "
          "and token positions during inference. ")
SHORT = "The capital city of Japan is the city of"
LONG = FILLER * 90 + "The capital city of Japan is the city of"
LAYERS = list(range(0, 28))


def _time(coro_factory, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        asyncio.run(coro_factory())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def bench(study, label, prompt, positions):
    run = f"frag_{label}"
    clean = study.capture_clean(prompt, run=run, layers=LAYERS,
                                positions="all_prompt", answer_token=" Tokyo")
    ncell = len(LAYERS) * len(positions)
    print(f"\n=== {label}: n={clean.num_prompt_tokens}, "
          f"grid {len(LAYERS)}x{len(positions)}={ncell} cells ===")

    def single():
        return study.sweep_layers_positions(
            prompt, run=run, layers=LAYERS, positions=positions,
            answer_token=" Tokyo", metric="logprob", server_side=True)

    async def grouped():
        for L in LAYERS:  # one batched forward per entry-layer (2a's structure)
            await study.sweep_layers_positions(
                prompt, run=run, layers=[L], positions=positions,
                answer_token=" Tokyo", metric="logprob", server_side=True)

    study.sweep_layers_positions  # noqa: warm below
    asyncio.run(single())  # warm
    t_single = _time(single)
    t_grouped = _time(grouped)
    print(f"  single all-layer sweep (Level 1)   {t_single:8.1f} ms "
          f"({t_single/ncell:.2f} ms/cell)")
    print(f"  per-layer grouped ({len(LAYERS)} groups)      {t_grouped:8.1f} ms "
          f"({t_grouped/ncell:.2f} ms/cell)")
    print(f"  fragmentation penalty = {t_grouped/t_single:.2f}x")
    return t_single, t_grouped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8123/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()
    study = pc.PatchStudy(model=args.model, base_url=args.base_url,
                          hook="post_block")
    bench(study, "short", SHORT, list(range(0, 9)))
    bench(study, "long", LONG, [0, 200, 400, 700, 1000, 1400, 1800, 2200])


if __name__ == "__main__":
    main()
