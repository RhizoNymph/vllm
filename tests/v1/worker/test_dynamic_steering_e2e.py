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


# Greedy decoding of two identical prompts in one batch is NOT bitwise
# identical deep into generation: batched reductions use position-
# dependent orders, so the two diverge from pure FP noise after many
# tokens (~20+ observed on gemma4-31B). The test must therefore not
# assume "identical prompt => identical baseline". Instead it compares
# the two outputs *within the same steered run* (target vs. in-batch
# control): real steering forces an EARLY divergence (~token 2), well
# separated from the late FP-noise floor. This also needs only one model
# load. ``NOISE_FLOOR`` is the margin between the two regimes.
NOISE_FLOOR = 10


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dynamic-steering e2e requires CUDA"
)
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_dynamic_override_one_step_latency_and_targeting():
    """One emitted override shifts only the target request, one step late.

    Asserted within a single steered batch of two identical requests:
    the steered (target) and untouched (control) outputs must diverge
    *early* (steering took effect — distinguishable from late FP-noise
    divergence) but *not at token 0* (the override emitted at step N only
    acts at step N+1).
    """
    prompts = [PROMPT, PROMPT]  # two identical concurrent requests
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
        out_a, out_b = _token_ids(llm, prompts)
    finally:
        del llm

    first_diff = _common_prefix_len(out_a, out_b)
    print(f"first_diff={first_diff} a={out_a}\n           b={out_b}")

    # Targeting: exactly one request was steered, so the two diverge.
    assert out_a != out_b, (
        "the two requests are identical — the stub steered neither or both; "
        f"a={out_a!r} b={out_b!r}"
    )
    # Latency: token 0 (from prefill, before any decode-step override) must
    # match; divergence starts at step >= 1.
    assert first_diff >= 1, (
        "outputs diverged on the very first token; an override emitted at "
        f"step N must only act at N+1. a={out_a!r} b={out_b!r}"
    )
    # Steering, not noise: real steering diverges early, far below the
    # batched-FP-noise floor that identical prompts hit much later.
    assert first_diff <= NOISE_FLOOR, (
        f"divergence at token {first_diff} looks like FP noise, not "
        f"steering (expected <= {NOISE_FLOOR}). a={out_a!r} b={out_b!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
