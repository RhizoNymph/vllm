# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level e2e for **dynamic steering under real scheduler
preemption**. A tight KV budget (``num_gpu_blocks_override``) plus several
concurrent long generations forces the scheduler to preempt running
requests (recompute path) while a per-request dynamic override is active.

Preemption is the adversarial case for the dynamic-override pool: a
preempted request re-runs prefill and *drops* its worker-side dynamic
override (see ``Scheduler._preempt_request``), so the pool bookkeeping must
stay consistent — no leaked rows, a clean drain to zero once every request
finishes, and sane applied/rejected counters.

Driven through a real ``LLM`` with the deterministic sync consumer
(:class:`DeterministicOverrideStub`) that steers the first request it sees
in decode, exactly once — the same stub the one-step-latency e2e uses. Two
identical prompts ride in the pressured batch: the stub steers the first
(target), the second is the in-batch control, so steering is shown to remain
in effect *through* the preemption pressure (target diverges early from the
control — the within-run technique of ``test_dynamic_steering_e2e.py``).

Asserts:

1. The whole batch finishes without crashing.
2. Preemptions actually happened (``vllm:num_preemptions`` > 0) — otherwise
   the test is vacuous.
3. Steering was in effect during the pressured run (target != control,
   early divergence).
4. After completion the dynamic-override pool has drained to zero
   (``get_dynamic_steering_status`` on every worker), with a sane
   applied (>= 1) / rejected count.

Requires CUDA + a tapped gemma4 (only gemma4 carries the steering hooks).
Skipped unless run manually against such a model:

    DYNSTEER_E2E_MODEL=/path/to/gemma-4-31B-it-Q4_K_S.gguf \
    DYNSTEER_E2E_LAYER=30 VLLM_USE_FLASHINFER_SAMPLER=0 \
    .venv/bin/python -m pytest tests/v1/worker/test_preemption_steering_e2e.py -v -s
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import torch

MODEL = os.environ.get("DYNSTEER_E2E_MODEL", "google/gemma-4-E2B-it")
LAYER = int(os.environ.get("DYNSTEER_E2E_LAYER", "8"))
IS_LOCAL = MODEL.endswith(".gguf") or os.path.exists(MODEL)

# Small KV budget (in blocks) forces preemption; override via env to tune the
# pressure on a given card.
KV_BLOCKS = int(os.environ.get("DYNSTEER_E2E_KV_BLOCKS", "200"))

TARGET_PROMPT = "The capital of France is"
# Distinct, non-trivial filler prompts so prefix caching (disabled anyway)
# can't collapse them and each holds its own KV blocks.
FILLERS = [
    "Explain in detail how a four-stroke internal combustion engine works,",
    "Write a long story about a lighthouse keeper who discovers",
    "Describe the process of photosynthesis step by step, including",
    "Summarize the entire history of the Roman Empire from its founding",
    "List and explain ten distinct algorithms for sorting an array,",
    "Give a thorough overview of how the TCP/IP networking stack",
]
MAX_TOKENS = 160
NOISE_FLOOR = 10


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    IS_LOCAL and not os.path.exists(MODEL),
    reason=f"DYNSTEER_E2E_MODEL path not found: {MODEL}",
)
def test_steering_survives_preemption_and_pool_drains():
    from vllm import LLM, SamplingParams

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
    kwargs = dict(
        model=MODEL,
        enable_steering=True,
        max_dynamic_steering_configs=4,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        max_model_len=512,
        num_gpu_blocks_override=KV_BLOCKS,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=0,
        capture_consumers=consumers,
    )
    if not IS_LOCAL:
        kwargs["load_format"] = "dummy"

    # Prompt 0 = stub target (steered), prompt 1 = in-batch control; the rest
    # are long fillers that create the KV pressure. All generate MAX_TOKENS to
    # keep many sequences alive at once.
    prompts = [TARGET_PROMPT, TARGET_PROMPT, *FILLERS]

    llm = LLM(**kwargs)
    try:
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=0)
        outs = llm.generate(prompts, sp)
        token_ids = [list(o.outputs[0].token_ids) for o in outs]

        metrics = {m.name: getattr(m, "value", None) for m in llm.get_metrics()}
        preemptions = metrics.get("vllm:num_preemptions")
        status = llm.collective_rpc("get_dynamic_steering_status")
    finally:
        del llm

    # 1) No crash: every request produced tokens.
    assert all(len(t) > 0 for t in token_ids), "a request produced no output"

    # 2) Preemptions actually happened (else the pressure test is vacuous).
    print(f"num_preemptions={preemptions}")
    assert preemptions is not None and preemptions > 0, (
        f"no preemptions occurred (vllm:num_preemptions={preemptions}); the KV "
        f"budget (num_gpu_blocks_override={KV_BLOCKS}) was too large — lower "
        f"DYNSTEER_E2E_KV_BLOCKS until preemption is forced"
    )

    # 3) Steering stayed in effect through the pressure: the stub-targeted
    # request (0) diverges early from the in-batch control (1).
    target, control = token_ids[0], token_ids[1]
    diff = _common_prefix_len(target, control)
    print(
        f"steered first_diff={diff}\n  target ={target[:12]}"
        f"\n  control={control[:12]}"
    )
    assert target != control, "steering had no effect under preemption"
    assert diff <= NOISE_FLOOR, (
        f"divergence at token {diff} looks like FP noise, not steering "
        f"(expected <= {NOISE_FLOOR})"
    )

    # 4) The dynamic-override pool drained to zero on every worker, with sane
    # applied/rejected counters.
    for i, st in enumerate(status):
        pool = st.get("dynamic_pool")
        assert pool is not None, f"worker {i}: steering not initialized"
        print(
            f"worker {i}: pool in_use={pool['in_use']}/{pool['capacity']} "
            f"queue={st.get('action_queue')} apply_stats={st.get('apply_stats')}"
        )
        assert pool["in_use"] == 0, (
            f"worker {i}: dynamic pool did not drain (in_use={pool['in_use']}) "
            f"after all requests finished — a preempted override leaked"
        )
        q = st.get("action_queue")
        applied = 0
        if q is not None:
            assert q["rejected"] >= 0 and q["applied"] >= 0
            applied = q["applied"]
        stats = st.get("apply_stats") or {}
        applied += sum(
            c.get("applied", 0) for c in stats.values() if isinstance(c, dict)
        )
        assert applied >= 1, (
            f"worker {i}: no override was ever applied "
            f"(queue={q} apply_stats={stats}) — the steered run is inconclusive"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
