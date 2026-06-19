# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the v2 runner's steering control-plane glue.

Covers the v2-specific lifecycle (register on add, release on finish,
prefill->decode transition) and the per-step index build, using a fake
``SteeringManager`` and CPU tensors. The fused kernel / real manager are
exercised separately in ``tests/v1/worker/test_steering_manager*.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.gpu.steering_runner_mixin import SteeringRunnerMixin


class _FakeManager:
    def __init__(self):
        self.config_to_row: dict = {}
        self.global_base_vectors: dict = {}
        self.global_prefill_vectors: dict = {}
        self.global_decode_vectors: dict = {}
        self._tables_dirty = False
        self.registered: list[tuple[int, str]] = []
        self.released: list[tuple[int, str]] = []
        self.populated = 0

    def register_config(self, h, effective, phase, locally_owned_layers):
        self.registered.append((h, phase))
        self.config_to_row[(h, phase)] = len(self.config_to_row) + 3

    def release_config(self, h, phase):
        self.released.append((h, phase))

    def get_row_for_config(self, h, is_prefill):
        return int(h)  # row == hash keeps the expected index readable

    def populate_steering_tables(self, layers):
        self.populated += 1
        self._tables_dirty = False


def _layer(num_tokens=16):
    layer = SimpleNamespace(steering_index=torch.zeros(num_tokens, dtype=torch.long))
    for hp in SteeringHookPoint:
        setattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hp], torch.ones(1, dtype=torch.bool))
    return layer


def _make_glue(num_computed, max_tokens=16, max_seqs=8):
    glue = SteeringRunnerMixin.__new__(SteeringRunnerMixin)
    glue._steering_manager = _FakeManager()
    glue._steerable_layers_cache = {0: _layer(max_tokens)}
    glue._steering_reqs = {}
    glue._steering_index_dirty = False
    glue._locally_owned_layers = frozenset({0})
    glue._steering_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
    glue._steering_n_tokens_scratch = np.zeros(max_seqs, dtype=np.int64)
    glue._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
    glue.req_states = SimpleNamespace(
        num_computed_tokens_np=np.asarray(num_computed, dtype=np.int32)
    )
    # Avoid building real SamplingParams: the resolve step just needs to be truthy.
    glue._resolve_request_steering = lambda sp, phase: {"pre_attn": {0: [1.0]}}
    return glue


def _new_req(req_id, prefill_hash, decode_hash, prompt_len, num_computed=0):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=object(),
        prefill_steering_config_hash=prefill_hash,
        decode_steering_config_hash=decode_hash,
        prompt_token_ids=list(range(prompt_len)),
        prompt_embeds=None,
        num_computed_tokens=num_computed,
    )


def test_add_request_registers_prefill():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )

    rs = glue._steering_reqs["a"]
    assert rs.phase == "prefill"
    assert rs.num_prompt_tokens == 10
    assert glue._steering_manager.registered == [(7, "prefill")]


def test_add_request_direct_to_decode_on_full_prefix_hit():
    glue = _make_glue(num_computed=[10])
    glue._steering_add_request(
        _new_req("a", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )

    rs = glue._steering_reqs["a"]
    assert rs.phase == "decode"
    assert glue._steering_manager.registered == [(5, "decode")]


def test_add_request_no_hashes_is_untracked():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=0, decode_hash=0, prompt_len=4)
    )

    assert "a" not in glue._steering_reqs
    assert glue._steering_manager.registered == []


def test_finish_request_releases_current_phase():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )

    glue._steering_finish_requests(["a"])

    assert "a" not in glue._steering_reqs
    assert glue._steering_manager.released == [(7, "prefill")]


def test_streaming_readd_releases_old_then_registers_new():
    glue = _make_glue(num_computed=[0])
    glue._steering_add_request(
        _new_req("a", prefill_hash=7, decode_hash=9, prompt_len=10)
    )
    # Re-add same id with a different config (streaming update).
    glue._steering_add_request(
        _new_req("a", prefill_hash=11, decode_hash=9, prompt_len=10)
    )

    assert glue._steering_manager.released == [(7, "prefill")]
    assert glue._steering_manager.registered == [(7, "prefill"), (11, "prefill")]


def test_update_buffers_builds_per_token_index_and_transition():
    # Two requests, batch order [decode "d", prefill "p"].
    glue = _make_glue(num_computed=[10, 8])
    # d: direct-to-decode (computed 10 >= prompt 10), decode_hash 5.
    glue._steering_add_request(
        _new_req("d", prefill_hash=0, decode_hash=5, prompt_len=10, num_computed=10)
    )
    # p: prefilling, computed 8 of 10, prefill_hash 7 / decode_hash 9; this
    # step schedules 3 tokens -> crosses the boundary -> transition fires.
    glue._steering_add_request(
        _new_req("p", prefill_hash=7, decode_hash=9, prompt_len=10)
    )
    glue._steering_reqs["p"].num_prompt_tokens = 10  # ensure boundary at 10

    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["d", "p"],
        idx_mapping_np=np.asarray([0, 1], dtype=np.int32),
    )
    sched = SimpleNamespace(num_scheduled_tokens={"d": 1, "p": 3})

    glue._update_steering_buffers_v2(sched, input_batch)

    steering_index = glue._steerable_layers_cache[0].steering_index
    # d -> row 5 (1 token); p -> row 7 (3 tokens); tail zeroed.
    assert steering_index[:4].tolist() == [5, 7, 7, 7]
    assert steering_index[4:].sum().item() == 0
    # Boundary crossed for p (8 + 3 >= 10): prefill 7 released, decode 9 added.
    assert (7, "prefill") in glue._steering_manager.released
    assert (9, "decode") in glue._steering_manager.registered
    assert glue._steering_reqs["p"].phase == "decode"


def test_update_buffers_short_circuit_zeroes_dirty_index():
    glue = _make_glue(num_computed=[0])
    # No tracked requests and no globals -> nothing active.
    layer = glue._steerable_layers_cache[0]
    layer.steering_index[:3] = torch.tensor([1, 2, 3])
    glue._steering_index_dirty = True

    input_batch = SimpleNamespace(
        num_reqs=0, req_ids=[], idx_mapping_np=np.asarray([], dtype=np.int32)
    )
    sched = SimpleNamespace(num_scheduled_tokens={})

    glue._update_steering_buffers_v2(sched, input_batch)

    assert layer.steering_index.sum().item() == 0
    assert glue._steering_index_dirty is False
    # any_active flags cleared so apply_steering short-circuits.
    for hp in SteeringHookPoint:
        assert getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hp]).item() is False
