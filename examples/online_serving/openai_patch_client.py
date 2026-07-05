# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runnable demo of the activation-patching sweep client.

The client itself lives in the installed package::

    from vllm.entrypoints.serve.patch.client import PatchStudy, Span

This file is just a usage example. Start the server with patching + the
clean-run source consumer enabled::

    vllm serve google/gemma-3-4b-it \\
        --enable-patching \\
        --max-patch-slots 64 \\
        --patch-source-cache-bytes 2000000000 \\
        --capture-consumers '[{"type": "patch_source"}]'

Then run this file, or, e.g.::

    import asyncio
    from vllm.entrypoints.serve.patch.client import PatchStudy

    study = PatchStudy(model="google/gemma-3-4b-it")
    clean = study.capture_clean(
        "The Eiffel Tower is in the city of",
        run="paris", answer_token=" Paris",
    )
    result = asyncio.run(study.sweep_layers_positions(
        "The Colosseum is in the city of",
        run="paris", layers=range(0, 34, 2),
        positions=range(0, clean.num_prompt_tokens),
        answer_token=" Paris",
    ))
    print(result.argmax_cell(), result.top(5))
"""

from __future__ import annotations

import asyncio

from vllm.entrypoints.serve.patch.client import PatchStudy, Span


def _demo() -> None:
    study = PatchStudy(model="google/gemma-3-4b-it")
    clean = study.capture_clean(
        "The Eiffel Tower is in the city of",
        run="paris",
        answer_token=" Paris",
    )
    print("clean:", clean)
    result = asyncio.run(
        study.sweep_layers_positions(
            "The Colosseum is in the city of",
            run="paris",
            layers=range(0, 34, 4),
            positions=range(0, max(clean.num_prompt_tokens, 1)),
            answer_token=" Paris",
            metric="logprob",
        )
    )
    print("peak:", result.argmax_cell())
    print("top 5:", result.top(5))


def _demo_one_call() -> None:
    """One request, no explicit capture and no token indices.

    ``clean_prompt`` + ``server_side=True`` makes the server capture the clean
    run itself before running the grid; ``Span`` positions resolve server-side.
    """
    study = PatchStudy(model="google/gemma-3-4b-it")
    result = asyncio.run(
        study.sweep_layers_positions(
            "The Colosseum is in the city of",
            clean_prompt="The Eiffel Tower is in the city of",
            layers=range(0, 34, 4),
            positions=[Span("Colosseum")],
            answer_token=" Paris",
            metric="recovered",
            server_side=True,
        )
    )
    print("auto_captured:", result.auto_captured, result.captured_source_run)
    print("peak:", result.argmax_cell())


if __name__ == "__main__":
    _demo()
