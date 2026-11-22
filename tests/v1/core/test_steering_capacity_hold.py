# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``Scheduler.schedule`` holding a waiting request back for steering capacity.

The admission helpers are unit-tested in ``test_steering_scheduler.py``;
these drive a real scheduler through a step where a request does not fit
the steering row pool, so it must be held (skipped this step, kept waiting
at the front of the queue) and admitted once capacity frees up.
"""

import pytest

from vllm.config.steering import SteeringConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK = 16

init_none_hash(sha256)


def _req(rid: str, value: float | None) -> Request:
    vectors = None if value is None else {"post_block": {0: [value, value]}}
    sp = SamplingParams(max_tokens=4, steering_vectors=vectors)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=rid,
        prompt_token_ids=[7] * 40,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _scheduler(max_configs: int):
    sched = create_scheduler(block_size=BLOCK, enable_prefix_caching=True)
    sched.steering_config = SteeringConfig(max_steering_configs=max_configs)
    return sched


def _scheduled_ids(out) -> set[str]:
    return {r.req_id for r in out.scheduled_new_reqs}


def test_over_capacity_request_is_held_not_crashed():
    # A whole-prompt request reserves its prefill and decode rows (2).
    sched = _scheduler(2)
    sched.add_request(_req("a", 1.0))
    sched.add_request(_req("b", 2.0))
    out = sched.schedule()
    assert _scheduled_ids(out) == {"a"}
    assert [r.request_id for r in sched.waiting] == ["b"]


def test_held_request_keeps_its_place_ahead_of_later_arrivals():
    sched = _scheduler(2)
    sched.add_request(_req("a", 1.0))
    sched.add_request(_req("b", 2.0))
    sched.add_request(_req("c", 3.0))
    sched.schedule()
    assert [r.request_id for r in sched.waiting] == ["b", "c"]


def test_unsteered_request_behind_a_held_one_is_admitted():
    sched = _scheduler(2)
    sched.add_request(_req("a", 1.0))
    sched.add_request(_req("b", 2.0))
    sched.add_request(_req("c", None))
    out = sched.schedule()
    assert _scheduled_ids(out) == {"a", "c"}
    assert [r.request_id for r in sched.waiting] == ["b"]
