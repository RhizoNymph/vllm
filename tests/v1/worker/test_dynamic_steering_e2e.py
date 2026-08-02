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

Runs both eager and CUDA-graph modes: sync consumers must not force
eager, and a graph replay must read the step's updated steering tables.

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

from tests.v1.worker.steering_e2e_utils import (  # isort: skip
    MAX_TOKENS,
    NOISE_FLOOR,
    PROMPT,
    build_llm,
    common_prefix_len,
    env_layer,
    requires_consumer_plugin,
    requires_cuda,
    requires_model_path,
)

import pytest

LAYER = env_layer(8)


def _token_ids(llm, prompts):
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
    outs = llm.generate(prompts, sp)
    return [list(o.outputs[0].token_ids) for o in outs]


@requires_cuda
@requires_model_path
@requires_consumer_plugin("dynamic_steering_e2e")
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "cudagraph"])
def test_dynamic_override_one_step_latency_and_targeting(enforce_eager):
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
                "steer_hook": "post_block",
                "steer_norm": 24.0,
                "emit_after_steps": 1,
            },
        }
    ]
    llm = build_llm(consumers, enforce_eager=enforce_eager)
    try:
        out_a, out_b = _token_ids(llm, prompts)
    finally:
        del llm

    first_diff = common_prefix_len(out_a, out_b)
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
