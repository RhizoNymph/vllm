# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e test for dynamic steering (Phase 1a).

Drives a real ``LLM`` with a config-driven sync consumer
(:class:`DeterministicOverrideStub`) and asserts the two properties the
unit tests cannot reach through stubs:

1. **Exactly-one-step actuation latency** — an override emitted from
   step N's activations changes the request's output starting at step
   N+1, never the current token. Asserted via a shared token-id prefix
   between the steered and baseline outputs.
2. **Per-request targeting only** — the stub steers exactly one of two
   identical concurrent requests; the other (the in-batch control)
   reproduces the no-consumer baseline byte for byte.

Requires CUDA and a model whose architecture carries the capture taps
*and* steering hooks (only ``gemma4`` today). Skipped unless run
manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_dynamic_steering_e2e.py -v -s

``DYNSTEER_E2E_MODEL`` defaults to the (gated) tiny HF gemma4 with dummy
weights; point it at a local GGUF to run without HF access.
"""

from __future__ import annotations

import os

# The engine core spawns workers; this test touches CUDA in the parent
# (the skipif below), so force spawn to avoid "Cannot re-initialize CUDA
# in forked subprocess". vLLM's conftest normally sets this, but this
# test is run standalone (it has no conftest dependencies).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

PROMPT = "The capital of France is"
MAX_TOKENS = 24


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _build_llm(capture_consumers):
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
        # Tiny dummy-weight gemma4 for CI / no-GGUF environments.
        kwargs["load_format"] = "dummy"
    if capture_consumers is not None:
        kwargs["capture_consumers"] = capture_consumers
    return LLM(**kwargs)


def _token_ids(llm, prompts):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
    outs = llm.generate(prompts, sp)
    return [list(o.outputs[0].token_ids) for o in outs]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dynamic-steering e2e requires CUDA"
)
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_dynamic_override_one_step_latency_and_targeting():
    """One emitted override shifts only the target request, one step late."""
    prompts = [PROMPT, PROMPT]  # two identical concurrent requests

    # --- Baseline: no consumer -> identical greedy outputs. ---
    llm = _build_llm(capture_consumers=None)
    try:
        base_a, base_b = _token_ids(llm, prompts)
    finally:
        del llm

    assert base_a == base_b, (
        "identical greedy prompts must produce identical baseline outputs; "
        f"got {base_a!r} vs {base_b!r}"
    )
    baseline = base_a
    assert len(baseline) >= 3, f"baseline too short to test latency: {baseline!r}"

    # --- Steered: stub steers the first request it sees in decode. ---
    consumers = [
        {
            "name": "dynamic_steering_e2e",
            "params": {
                "steer_layer": LAYER,
                "steer_hook": "post_mlp",
                "steer_norm": 24.0,
                "emit_after_steps": 1,
            },
        }
    ]
    llm = _build_llm(capture_consumers=consumers)
    try:
        steered = _token_ids(llm, prompts)
    finally:
        del llm

    matches = [s == baseline for s in steered]
    # Targeting: exactly one request (the control) reproduces baseline.
    assert sum(matches) == 1, (
        "exactly one of two identical requests must be steered; "
        f"baseline={baseline!r} steered={steered!r}"
    )

    target = steered[matches.index(False)]
    # Latency: the steered request diverges, but not on the first token(s)
    # — the override emitted at step N only takes effect at step N+1.
    assert target != baseline
    prefix = _common_prefix_len(target, baseline)
    assert prefix >= 1, (
        "steered output diverged on the very first token; the override "
        f"should lag by a step. target={target!r} baseline={baseline!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
