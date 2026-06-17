# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for the dynamic-steering APC notification.

Proves the worker->scheduler decode-signature fix end-to-end with prefix
caching (docs/design/dynamic_steering_apc_notification.md): a continuation
of a *dynamically steered* request must NOT reuse that request's steered
decode KV blocks (they are keyed by the override signature, not the
admitted config), while a continuation of an *unsteered* request reuses
its decode blocks normally.

The DeterministicOverrideStub steers the first request it sees in decode,
exactly once. So run A (steered) then C (unsteered, stub already fired),
then submit B = A.prompt+A.out and D = C.prompt+C.out as continuations and
compare prefix-cache reuse.

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_apc_steering_e2e.py -v -s
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

PROMPT = (
    "Write a detailed paragraph about the history of the city of Paris, "
    "France, covering its founding, medieval period, and modern era."
)
GEN = 48  # ~3 decode blocks at block_size 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_steered_continuation_does_not_reuse_steered_decode_kv():
    from vllm import LLM, SamplingParams

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
    kwargs = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        max_model_len=2048,
        seed=0,
        capture_consumers=consumers,
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"
    llm = LLM(**kwargs)
    try:
        greedy = SamplingParams(temperature=0.0, max_tokens=GEN)
        one = SamplingParams(temperature=0.0, max_tokens=1)

        out_a = llm.generate([PROMPT], greedy)[0]  # steered (stub targets it)
        out_c = llm.generate([PROMPT], greedy)[0]  # unsteered (stub fired)

        plen = len(out_a.prompt_token_ids)
        b_ids = list(out_a.prompt_token_ids) + list(out_a.outputs[0].token_ids)
        d_ids = list(out_c.prompt_token_ids) + list(out_c.outputs[0].token_ids)
        out_b = llm.generate([{"prompt_token_ids": b_ids}], one)[0]
        out_d = llm.generate([{"prompt_token_ids": d_ids}], one)[0]
        b_cached = out_b.num_cached_tokens
        d_cached = out_d.num_cached_tokens
    finally:
        del llm

    # The override actually changed A (else the test is inconclusive).
    assert out_a.outputs[0].token_ids != out_c.outputs[0].token_ids
    # Unsteered continuation reuses well into its generated region.
    assert d_cached > plen, f"d_cached={d_cached} plen={plen}"
    # Steered continuation reuses far less — the override-keyed decode blocks
    # are not reused (the fix). Without it, B would reuse A's steered KV.
    assert d_cached - b_cached >= 16, f"b_cached={b_cached} d_cached={d_cached}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
