# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end scheduler tests for steering-config pool capacity (backpressure).

The worker ``SteeringManager`` holds one table row per distinct
``(config_hash, phase)`` and ``register_config`` raises when the pool is full.
The scheduler must therefore never admit more distinct steering rows than
``max_steering_configs`` -- and when at capacity it must apply *backpressure*
(hold the request in the waiting queue until a row frees) rather than running it
unsteered, so a request that asked for steering always gets it.

The historical bug: the admission gate counted only a request's *prefill* row,
not the *decode* row it also needs, so requests sharing a prefill config but
each carrying a distinct decode config were all admitted -- overflowing the pool
at the prefill->decode transition (engine crash).
"""

import os

import pytest

from tests.v1.core.utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get(
    "STEERING_TEST_MODEL",
    "/home/nymph/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
)
MAX_CONFIGS = 4
EOS = 50256


def _set_steering(requests, prefill_hash, decode_base):
    """Give each request a (possibly shared) prefill hash and a DISTINCT
    decode hash. ``prefill_hash=0`` => decode-only."""
    for i, req in enumerate(requests):
        d = decode_base + i
        req.prefill_steering_config_hash = prefill_hash
        req.decode_steering_config_hash = d
        req.block_hash_prefill_steering_config_hash = prefill_hash
        req.block_hash_decode_steering_config_hash = d


def _worker_rows(running):
    """Distinct (hash, phase) rows the worker must hold for the running set:
    the prefill row while a request is prefilling, plus the decode row it holds
    for its whole lifetime. This is exactly the worker SteeringManager pool
    occupancy the scheduler must keep within ``max_steering_configs``."""
    keys = set()
    for r in running:
        if (
            r.num_computed_tokens < r.num_prompt_tokens
            and r.prefill_steering_config_hash != 0
        ):
            keys.add((r.prefill_steering_config_hash, "prefill"))
        if r.decode_steering_config_hash != 0:
            keys.add((r.decode_steering_config_hash, "decode"))
    return keys


def test_shared_prefill_distinct_decode_respects_pool():
    """8 requests sharing one prefill config but each with a distinct decode
    config. The worker needs one decode row per distinct decode config, so the
    scheduler must admit at most ``max`` of them (minus the shared prefill row)
    -- the rest wait. Pre-fix this over-admits (gate ignored the decode row)."""
    # long_prefill_token_threshold keeps requests mid-prefill across steps so
    # many co-exist as running-with-distinct-decode (the worker would need a
    # decode row for each held through their transitions).
    scheduler = create_scheduler(
        model=MODEL, max_steering_configs=MAX_CONFIGS, max_num_seqs=64,
        num_blocks=10000, long_prefill_token_threshold=8,
    )
    reqs = create_requests(num_requests=8, num_tokens=40, max_tokens=8)
    _set_steering(reqs, prefill_hash=7777, decode_base=2000)
    for r in reqs:
        scheduler.add_request(r)

    scheduler.schedule()

    rows = _worker_rows(scheduler.running)
    # Worker holds the shared prefill row + one decode row per admitted request
    # (reserved for life). Pre-fix the gate ignored decode rows and admitted all
    # 8 (=> 1 + 8 = 9 rows demanded against a pool of 4).
    assert len(rows) <= MAX_CONFIGS, (
        f"over-admitted: worker needs {len(rows)} rows for pool {MAX_CONFIGS} "
        f"(running={len(scheduler.running)})"
    )
    # Some requests must be admitted (not a vacuous pass) and some held back.
    assert len(scheduler.running) > 0
    assert scheduler.get_num_unfinished_requests() > len(scheduler.running)


def test_decode_only_respects_pool():
    """Decode-only requests were already gated correctly; guard against
    regression."""
    scheduler = create_scheduler(
        model=MODEL, max_steering_configs=MAX_CONFIGS, max_num_seqs=64,
        num_blocks=10000,
    )
    reqs = create_requests(num_requests=8, num_tokens=8, max_tokens=8)
    _set_steering(reqs, prefill_hash=0, decode_base=3000)
    for r in reqs:
        scheduler.add_request(r)

    scheduler.schedule()
    assert len(_worker_rows(scheduler.running)) <= MAX_CONFIGS


def test_churn_backpressure_never_overflows_and_never_drops():
    """Multi-step churn: every running request's distinct steering rows must
    stay within the pool at every step, and every steered request must
    eventually run STEERED (backpressure, never silently dropped)."""
    scheduler = create_scheduler(
        model=MODEL, max_steering_configs=MAX_CONFIGS, max_num_seqs=64,
        num_blocks=10000,
    )
    total = 20
    reqs = create_requests(num_requests=total, num_tokens=8, max_tokens=6)
    _set_steering(reqs, prefill_hash=7777, decode_base=4000)
    by_id = {r.request_id: r for r in reqs}
    for r in reqs:
        scheduler.add_request(r)

    ran_with_steering: set[str] = set()
    for _ in range(500):
        out = scheduler.schedule()
        # Worker pool demand this step: one row per distinct (hash, phase) over
        # all running requests (prefill row while prefilling, decode row held
        # for life) -- exactly what the scheduler reserves.
        keys = set()
        for r in scheduler.running:
            if (
                r.num_computed_tokens < r.num_prompt_tokens
                and r.prefill_steering_config_hash != 0
            ):
                keys.add((r.prefill_steering_config_hash, "prefill"))
            if r.decode_steering_config_hash != 0:
                keys.add((r.decode_steering_config_hash, "decode"))
        assert len(keys) <= MAX_CONFIGS, f"pool overflow: {len(keys)} > {MAX_CONFIGS}"

        for rid in out.num_scheduled_tokens:
            req = by_id[rid]
            # A scheduled request whose decode block-hash is intact runs steered.
            if req.block_hash_decode_steering_config_hash != 0:
                ran_with_steering.add(rid)

        scheduler.update_from_output(out, _model_output(out))
        if not scheduler.get_num_unfinished_requests():
            break

    assert not scheduler.get_num_unfinished_requests(), "did not drain"
    assert len(ran_with_steering) == total, (
        f"only {len(ran_with_steering)}/{total} ran steered -- steering was "
        f"dropped instead of backpressured"
    )


def _model_output(scheduler_output):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[EOS] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
