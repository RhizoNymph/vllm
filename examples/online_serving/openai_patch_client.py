# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runnable activation-patching walkthrough (causal tracing over HTTP).

The client lives in the installed package::

    from vllm.entrypoints.serve.patch.client import PatchStudy, Span

Start a patching-enabled server (``--enable-patching`` implies the clean-run
capture consumer)::

    vllm serve google/gemma-3-4b-it \\
        --enable-patching \\
        --max-patch-slots 64 \\
        --patch-source-cache-bytes 2000000000

Then run this file against it::

    python examples/online_serving/openai_patch_client.py \\
        --model google/gemma-3-4b-it --num-layers 34

The walkthrough is the standard coarse-to-fine causal-tracing "walk", each
step a single HTTP request (the server captures the clean run itself, resolves
the substring to token positions, streams cells as they finish, and cleans up
its capture afterward):

1. Coarse sweep — every 4th layer x every prompt position, live-streamed.
2. Zoom — every layer in the hot band, at the corrupt span only.
3. Hook decomposition — the peak cell at pre_attn / post_attn / post_block
   (is the effect carried by attention or by the MLP?).

See docs/features/activation_patching.md for the full API.
"""

from __future__ import annotations

import argparse
import asyncio

from vllm.entrypoints.serve.patch.client import PatchStudy, Span, SweepResult

CLEAN = "The Eiffel Tower is in the city of"
CORRUPT = "The Colosseum is in the city of"
SPAN = "Colosseum"  # where clean/corrupt diverge
ANSWER = " Paris"  # the clean run's answer; recovered=1 means fully restored

_SHADES = " .:-=+*#%@"


def _ascii_heatmap(result: SweepResult) -> str:
    """Rows = layers (bottom-up), cols = positions; recovered 0..1 shaded."""
    lines = []
    for i, layer in reversed(list(enumerate(result.layers))):
        chars = []
        for j in range(len(result.positions)):
            v = result.grid[i][j]
            if v is None:
                chars.append("?")
            else:
                v = min(max(v, 0.0), 1.0)
                chars.append(_SHADES[int(v * (len(_SHADES) - 1))])
        lines.append(f"  L{layer:>2} |{''.join(chars)}|")
    pos = result.positions
    lines.append(f"       pos {pos[0]}..{pos[-1]}  (recovered: ' '=0 .. '@'=1)")
    return "\n".join(lines)


async def walk(model: str, base_url: str, num_layers: int) -> None:
    study = PatchStudy(model=model, base_url=base_url)

    # 1. Coarse sweep: one call — no pre-capture, no token indices. The
    #    on_cell callback streams cells as the server finishes them.
    done = 0

    def on_cell(event: dict) -> None:
        nonlocal done
        done += 1
        print(f"\r  cells: {done}", end="", flush=True)

    coarse = await study.sweep_layers_positions(
        CORRUPT,
        clean_prompt=CLEAN,  # server auto-captures the clean run
        layers=range(0, num_layers, 4),
        positions="all_prompt",
        answer_token=ANSWER,
        metric="recovered",
        server_side=True,
        on_cell=on_cell,
    )
    print(
        f"\n1. coarse ({len(coarse.layers)}x{len(coarse.positions)} grid, "
        f"auto_captured={coarse.auto_captured}, "
        f"noise_floor={coarse.noise_floor:.3g})"
    )
    print(_ascii_heatmap(coarse))
    print(f"  global peak: layer/position {coarse.argmax_cell()}")

    # Two effects show up in a denoising map: patching late layers at the
    # LAST position trivially writes the answer into the residual the head
    # reads, while the interesting causal path is at the DIVERGING SPAN.
    # Zoom on the span's own peak, not the global argmax.
    span_positions = await study.positions_for(CORRUPT, SPAN)
    span_cols = [j for j, p in enumerate(coarse.positions) if p in span_positions]
    peak_layer = max(
        (r for r in coarse.layers),
        key=lambda layer: max(
            coarse.grid[coarse.layers.index(layer)][j] or 0.0 for j in span_cols
        ),
    )
    print(f"  span peak: layer {peak_layer} @ '{SPAN}' {span_positions}")

    # 2. Zoom: every layer in the hot band, only at the diverging span.
    lo = max(0, peak_layer - 4)
    hi = min(num_layers, peak_layer + 5)
    zoom = await study.sweep_layers_positions(
        CORRUPT,
        clean_prompt=CLEAN,
        layers=range(lo, hi),
        positions=[Span(SPAN)],
        answer_token=ANSWER,
        metric="recovered",
        server_side=True,
    )
    print(f"2. zoom (layers {lo}..{hi - 1} @ '{SPAN}')")
    for i, layer in enumerate(zoom.layers):
        vals = " ".join(f"{v:+.2f}" if v is not None else "  ? " for v in zoom.grid[i])
        print(f"  L{layer:>2}: {vals}")

    # 3. Hook decomposition at the peak: attention stream vs block output.
    best_layer = zoom.argmax_cell()[0]
    hooks = await study.sweep_layers_positions(
        CORRUPT,
        clean_prompt=CLEAN,
        layers=[best_layer],
        positions=[Span(SPAN)],
        hooks=["pre_attn", "post_attn", "post_block"],
        answer_token=ANSWER,
        metric="recovered",
        server_side=True,
    )
    print(f"3. hooks at layer {best_layer} @ '{SPAN}' (one value per span token)")
    for hook, res in hooks.items():
        vals = " ".join(f"{v:+.2f}" if v is not None else "  ? " for v in res.grid[0])
        print(f"  {hook:>10}: {vals}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--num-layers", type=int, default=34)
    args = ap.parse_args()
    asyncio.run(walk(args.model, args.base_url, args.num_layers))


if __name__ == "__main__":
    main()
