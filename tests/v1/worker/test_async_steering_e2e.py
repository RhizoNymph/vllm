# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for the **async transport** (the Phase 0 action
queue). The sync path (``on_step`` returns) is covered by
``test_dynamic_steering_e2e.py``; this exercises the other transport:
a capture consumer that submits a steering update from ``on_capture`` (on
the dispatch thread), which the model runner drains at the top of a later
step (the 1-3 step async latency).

:class:`AsyncTierExample` submits one global decode-tier
``SteeringVectorUpdate`` through ``get_steering_action_queue().submit``.
The global tier steers every request, so this uses a single-request
cross-load comparison (a single greedy sequence is deterministic
run-to-run — the batched-FP nondeterminism that forces the within-run
technique elsewhere only bites multi-sequence batches): the steered run
must diverge from a no-consumer baseline, and — because the queued update
only lands on a later step — token 0 (the first decode token, produced
before any drain) must still match the baseline.

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

# The async update lands a few steps after submission; the global tier then
# steers strongly, so divergence is early. A single greedy sequence is
# deterministic across loads, so an UNsteered run would match the baseline
# for its full length — any divergence is steering. The ceiling just
# corroborates that it is early (real steering), not a late fluke.
ASYNC_FLOOR = 12


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _build_llm(consumers):
    from vllm import LLM

    kwargs: dict = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        seed=0,
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"
    if consumers is not None:
        kwargs["capture_consumers"] = consumers
    return LLM(**kwargs)


def _output(consumers) -> list[int]:
    from vllm import SamplingParams

    llm = _build_llm(consumers)
    try:
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
        return list(llm.generate([PROMPT], sp)[0].outputs[0].token_ids)
    finally:
        del llm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_async_queue_global_tier_steers_subsequent_steps():
    """A tier update submitted via the action queue from ``on_capture``
    changes the output, one or more steps after submission."""
    base = _output(None)
    steered = _output(
        [
            {
                "name": "steering_ex_async_tier",
                "params": {
                    "steer_layer": LAYER,
                    "steer_hook": "post_mlp",
                    "steer_norm": 24.0,
                },
            }
        ]
    )
    first_diff = _common_prefix_len(base, steered)
    print(f"first_diff={first_diff}\n base   ={base}\n steered={steered}")

    assert base != steered, (
        "async queue update never reached the steering tables — output "
        "unchanged from the no-consumer baseline"
    )
    # The queued update lands on a later step, so the first decode token
    # (produced before any drain) must match the baseline.
    assert first_diff >= 1, (
        "output diverged on token 0; an async update submitted during the "
        "first forward can only act on a subsequent step"
    )
    assert first_diff <= ASYNC_FLOOR, (
        f"divergence at token {first_diff} is later than expected for a "
        f"strong global tier (expected <= {ASYNC_FLOOR})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
