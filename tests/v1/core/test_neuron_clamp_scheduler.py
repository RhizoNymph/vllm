# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler admission + prefix-cache keying for neuron-clamp requests.

Clamps are folded into the phase-effective steering hash, so the real
scheduler must count clamp-only requests against ``max_steering_configs``
exactly like vector-steered requests, and prefill clamps must change the
prompt block hashes.
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

# NONE_HASH is random per init: initialize once so block hashes of
# different requests in this module are comparable.
init_none_hash(sha256)


def _req(rid, *, clamps=None, vectors=None, prompt=None):
    sp = SamplingParams(max_tokens=4, neuron_clamps=clamps, steering_vectors=vectors)
    sp.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=rid,
        prompt_token_ids=prompt if prompt is not None else [7] * 40,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _clamp(neuron, value=0.0, phase="both"):
    return [
        {"layer": 0, "neuron": neuron, "mode": "set", "value": value, "phase": phase}
    ]


def _scheduler(max_configs):
    sched = create_scheduler(block_size=BLOCK, enable_prefix_caching=True)
    sched.steering_config = SteeringConfig(max_steering_configs=max_configs)
    return sched


def _scheduled_ids(out):
    return {r.req_id for r in out.scheduled_new_reqs}


class TestAdmission:
    """A request whose whole prompt fits in one step reserves both its
    prefill and decode pair in that step (transition reservation), so one
    such request occupies 2 of ``max_steering_configs``."""

    def test_clamp_only_counts_against_capacity(self):
        sched = _scheduler(2)
        a = _req("a", clamps=_clamp(1))
        b = _req("b", clamps=_clamp(2))
        assert a.prefill_steering_config_hash != 0
        sched.add_request(a)
        sched.add_request(b)
        out = sched.schedule()
        assert _scheduled_ids(out) == {"a"}
        assert {r.request_id for r in sched.running} == {"a"}

    def test_identical_clamps_share_capacity(self):
        sched = _scheduler(2)
        sched.add_request(_req("a", clamps=_clamp(1)))
        sched.add_request(_req("b", clamps=_clamp(1)))
        out = sched.schedule()
        assert _scheduled_ids(out) == {"a", "b"}

    def test_unsteered_requests_never_blocked(self):
        sched = _scheduler(2)
        sched.add_request(_req("a", clamps=_clamp(1)))
        sched.add_request(_req("b", clamps=_clamp(2)))
        sched.add_request(_req("c"))
        out = sched.schedule()
        assert _scheduled_ids(out) == {"a", "c"}

    @pytest.mark.parametrize("capacity,expected", [(3, {"a", "b"}), (2, {"a"})])
    def test_vectors_plus_clamps_single_config(self, capacity, expected):
        # vectors+clamps is ONE (hash, phase) per phase: request a reserves
        # exactly its prefill + decode pair (2), leaving room for b at 3.
        sched = _scheduler(capacity)
        vec = {"post_block": {0: [1.0, 2.0]}}
        sched.add_request(_req("a", clamps=_clamp(1), vectors=vec))
        sched.add_request(_req("b", vectors=vec))
        out = sched.schedule()
        assert _scheduled_ids(out) == expected


class TestPrefixCacheKeys:
    def test_prefill_clamp_changes_block_hashes(self):
        plain = _req("p")
        clamped = _req("c", clamps=_clamp(1, phase="prefill"))
        assert plain.block_hashes and clamped.block_hashes
        assert plain.block_hashes[0] != clamped.block_hashes[0]

    def test_decode_only_clamp_keeps_prompt_hashes(self):
        plain = _req("p")
        clamped = _req("c", clamps=_clamp(1, phase="decode"))
        assert plain.block_hashes == clamped.block_hashes

    def test_same_prefill_clamp_shares_hashes(self):
        a = _req("a", clamps=_clamp(1))
        b = _req("b", clamps=_clamp(1))
        assert a.block_hashes == b.block_hashes
        c = _req("c", clamps=_clamp(1, value=0.5))
        assert a.block_hashes[0] != c.block_hashes[0]
