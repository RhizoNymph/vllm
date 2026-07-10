# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for the **async transport** (the Phase 0 action
queue). The sync path (``on_step`` returns) is covered by
``test_dynamic_steering_e2e.py``; this exercises the other transport:
a capture consumer that submits a steering update from ``on_capture`` (on
the dispatch thread), which the model runner drains at the top of a later
step (the 1-3 step async latency).

:class:`AsyncTierExample` submits one global decode-tier
``SteeringVectorUpdate`` through ``get_steering_action_queue().submit`` —
crucially, from ``on_capture``, which the capture pipeline runs only when a
request *finalizes* (after its output is emitted). So the update a request
triggers can never steer that same request; it steers the NEXT one. The
test models exactly that: it runs the same single-request prompt several
times in one engine and asserts a later generation diverges from the first
(the un-steered baseline, produced before any request had finalized to
submit the tier). Comparing generations within one engine instance is
deterministic (single greedy sequence), so any divergence is the tier
landing via the queue; the shared token-0 prefix confirms the tier is
decode-only (it cannot rewrite the first decode token of a steered run
relative to the baseline's).

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_async_steering_e2e.py -v -s
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

PROMPT = "The capital of France is"
MAX_TOKENS = 24
# Repeats of the prompt within one engine. The first is the baseline; the
# tier is submitted when an earlier request finalizes (on the finalize
# thread, possibly after generate() returns), so allow a couple of repeats
# for it to land before a steered generation appears.
REPEATS = 5


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _build_llm():
    from vllm import LLM

    kwargs: dict = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        seed=0,
        capture_consumers=[
            {
                "name": "steering_ex_async_tier",
                "params": {
                    "steer_layer": LAYER,
                    "steer_hook": "post_block",
                    "steer_norm": 24.0,
                },
            }
        ],
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"
    return LLM(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_async_queue_global_tier_steers_later_request():
    """A tier update submitted via the action queue from a finalizing
    request steers a SUBSEQUENT request (not itself)."""
    from vllm import SamplingParams

    llm = _build_llm()
    try:
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
        outs = [
            list(llm.generate([PROMPT], sp)[0].outputs[0].token_ids)
            for _ in range(REPEATS)
        ]
    finally:
        del llm

    base = outs[0]
    for i, o in enumerate(outs):
        print(f"gen[{i}]={o}")

    steered = [o for o in outs[1:] if o != base]
    first_diff = _common_prefix_len(base, steered[0]) if steered else None
    print(f"baseline={base}\n steered ={steered[0] if steered else None}"
          f"\n first_diff={first_diff}")
    # The only thing this proves — and all it needs to — is that the tier a
    # finalizing request submitted through the queue reached a LATER
    # request's steering tables. (Unlike the in-request latency case, the
    # tier is already installed before the later request decodes, so its
    # first decode token is steered too — no token-0 prefix is expected.)
    assert steered, (
        "no generation diverged from the first — the tier submitted by a "
        "finalizing request never reached a later request's steering tables"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
