# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``_populate_draft_steering_index`` behaviour (PR 5b).

Exercises both ``mode="first"`` (per-token walk mirroring main) and
``mode="loop"`` (one row per active request, decode-phase) without
booting a real speculative-decoding stack: the runner is stubbed and
the draft model is a fake module with ``steering_index`` +
``steering_table_post_mlp`` buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

_HP = DEFAULT_HOOK_POINT.value


class _FakeLayer(nn.Module):
    def __init__(self, layer_idx: int, hidden_size: int, max_configs: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.register_buffer(
            "steering_table_post_mlp",
            torch.zeros(max_configs + 2, hidden_size),
            persistent=False,
        )
        self.register_buffer(
            "steering_index",
            torch.zeros(16, dtype=torch.long),
            persistent=False,
        )


def _make_draft_layers(num: int, hidden_size: int, max_configs: int) -> dict:
    layers = {i: _FakeLayer(i, hidden_size, max_configs) for i in range(num)}
    # Share ``steering_index`` across layers the same way real decoder
    # stacks do.
    shared = layers[0].steering_index
    for mod in list(layers.values())[1:]:
        mod.steering_index = shared
    return layers


@dataclass
class _FakeInputBatch:
    num_reqs: int
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    num_computed_tokens_cpu: np.ndarray
    num_prompt_tokens: np.ndarray
    request_prefill_steering_hash_main: np.ndarray
    request_prefill_steering_hash_draft: np.ndarray
    request_decode_steering_hash_main: np.ndarray
    request_decode_steering_hash_draft: np.ndarray


@dataclass
class _FakeSchedulerOutput:
    num_scheduled_tokens: dict[str, int]


@dataclass
class _FakeRequestState:
    req_id: str
    sampling_params: object = None


class _FakeRunner(SteeringModelRunnerMixin):
    def __init__(self, draft_layers: dict):
        self._draft_steerable_layers_cache = draft_layers
        self._draft_steering_manager = SteeringManager(
            max_steering_configs=4, device=None
        )
        self._draft_locally_owned_layers = frozenset(draft_layers.keys())
        # Main-side state is unused by the draft-index populator but the
        # mixin reads ``_last_scheduler_output`` unconditionally.
        self._last_scheduler_output: _FakeSchedulerOutput | None = None
        self.input_batch: _FakeInputBatch | None = None
        self.requests: dict[str, _FakeRequestState] = {}


def _build_runner(
    draft_layers: dict,
    req_ids: list[str],
    num_scheduled: dict[str, int],
    computed_tokens: list[int],
    prompt_tokens: list[int],
    prefill_draft_hashes: list[int],
    decode_draft_hashes: list[int],
) -> _FakeRunner:
    runner = _FakeRunner(draft_layers)
    n = len(req_ids)
    runner.input_batch = _FakeInputBatch(
        num_reqs=n,
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        num_computed_tokens_cpu=np.array(computed_tokens, dtype=np.int64),
        num_prompt_tokens=np.array(prompt_tokens, dtype=np.int64),
        request_prefill_steering_hash_main=np.zeros(n, dtype=np.int64),
        request_prefill_steering_hash_draft=np.array(
            prefill_draft_hashes, dtype=np.int64
        ),
        request_decode_steering_hash_main=np.zeros(n, dtype=np.int64),
        request_decode_steering_hash_draft=np.array(
            decode_draft_hashes, dtype=np.int64
        ),
    )
    runner._last_scheduler_output = _FakeSchedulerOutput(num_scheduled)
    return runner


def _sampling_params_stub(
    draft_prefill: dict | None = None,
    draft_decode: dict | None = None,
) -> object:
    """Just-enough stub exposing the four effective-steering properties
    the populator reads during lazy draft registration."""
    return SimpleNamespace(
        effective_prefill_steering_draft=draft_prefill,
        effective_decode_steering_draft=draft_decode,
    )


class TestFirstMode:
    def test_writes_zero_when_no_draft_hashes(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        runner = _build_runner(
            draft_layers,
            req_ids=["r0", "r1"],
            num_scheduled={"r0": 3, "r1": 2},
            computed_tokens=[0, 0],
            prompt_tokens=[3, 2],  # both prefilling
            prefill_draft_hashes=[0, 0],  # no per-request draft steering
            decode_draft_hashes=[0, 0],
        )
        runner._populate_draft_steering_index("first", 5)
        idx = draft_layers[0].steering_index
        # Hash 0 means "prefill → row 1" per the manager's sentinel
        # convention.
        assert idx[:3].tolist() == [1, 1, 1]
        assert idx[3:5].tolist() == [1, 1]
        # Remaining positions zeroed.
        assert torch.equal(idx[5:], torch.zeros_like(idx[5:]))

    def test_writes_per_request_rows_on_registered_hashes(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        spec = {_HP: {0: [1.0] * 8}}
        hash_r0 = 42
        runner = _build_runner(
            draft_layers,
            req_ids=["r0"],
            num_scheduled={"r0": 4},
            computed_tokens=[0],
            prompt_tokens=[4],
            prefill_draft_hashes=[hash_r0],
            decode_draft_hashes=[0],
        )
        runner.requests["r0"] = _FakeRequestState(
            req_id="r0",
            sampling_params=_sampling_params_stub(draft_prefill=spec),
        )
        # Pre-register so the populator finds the row in config_to_row.
        row = runner._draft_steering_manager.register_config(
            hash_r0, spec, phase="prefill"
        )
        runner._populate_draft_steering_index("first", 4)
        idx = draft_layers[0].steering_index
        assert idx[:4].tolist() == [row, row, row, row]

    def test_lazy_registers_when_hash_not_in_manager(self):
        """If a draft hash appears without a prior ``register_config``,
        the populator looks it up on the request's SamplingParams and
        registers it on the fly."""
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        spec = {_HP: {0: [1.0] * 8}}
        hash_r0 = 99
        runner = _build_runner(
            draft_layers,
            req_ids=["r0"],
            num_scheduled={"r0": 2},
            computed_tokens=[0],
            prompt_tokens=[2],
            prefill_draft_hashes=[hash_r0],
            decode_draft_hashes=[0],
        )
        runner.requests["r0"] = _FakeRequestState(
            req_id="r0",
            sampling_params=_sampling_params_stub(draft_prefill=spec),
        )
        runner._populate_draft_steering_index("first", 2)
        idx = draft_layers[0].steering_index
        # Lazy registration produced a non-sentinel row.
        row = idx[0].item()
        assert row >= 3  # rows 0, 1, 2 are sentinels
        assert idx[:2].tolist() == [row, row]

    def test_decode_phase_reads_decode_hashes(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        runner = _build_runner(
            draft_layers,
            req_ids=["r0"],
            num_scheduled={"r0": 1},
            computed_tokens=[5],
            prompt_tokens=[5],  # num_computed == num_prompt → decode
            prefill_draft_hashes=[999],  # irrelevant
            decode_draft_hashes=[0],
        )
        runner._populate_draft_steering_index("first", 1)
        idx = draft_layers[0].steering_index
        # Decode with no draft hash → row 2 sentinel.
        assert idx[0].item() == 2


class TestLoopMode:
    def test_single_row_per_active_request(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        runner = _build_runner(
            draft_layers,
            req_ids=["r0", "r1", "r2"],
            num_scheduled={"r0": 1, "r1": 1, "r2": 1},
            computed_tokens=[3, 3, 3],
            prompt_tokens=[3, 3, 3],  # all in decode
            prefill_draft_hashes=[0, 0, 0],
            decode_draft_hashes=[0, 0, 0],
        )
        runner._populate_draft_steering_index("loop", 3)
        idx = draft_layers[0].steering_index
        # Three rows written, all decode sentinel (row 2).
        assert idx[:3].tolist() == [2, 2, 2]
        # Remaining positions zeroed.
        assert torch.equal(idx[3:], torch.zeros_like(idx[3:]))

    def test_loop_caps_at_batch_size(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        runner = _build_runner(
            draft_layers,
            req_ids=["r0"],
            num_scheduled={"r0": 1},
            computed_tokens=[2],
            prompt_tokens=[2],
            prefill_draft_hashes=[0],
            decode_draft_hashes=[0],
        )
        # Claim we have 4 loop tokens but only 1 active request.
        runner._populate_draft_steering_index("loop", 4)
        idx = draft_layers[0].steering_index
        assert idx[0].item() == 2
        # Positions beyond num_reqs are zeroed.
        assert torch.equal(idx[1:], torch.zeros_like(idx[1:]))


class TestNoop:
    def test_bails_when_draft_manager_absent(self):
        """No draft manager → helper is a no-op even if called."""
        runner = SteeringModelRunnerMixin()
        # No state set up.
        runner._populate_draft_steering_index("first", 5)  # must not raise
        runner._populate_draft_steering_index("loop", 5)  # must not raise

    def test_rejects_unknown_mode(self):
        draft_layers = _make_draft_layers(2, 8, max_configs=4)
        runner = _build_runner(
            draft_layers,
            req_ids=["r0"],
            num_scheduled={"r0": 1},
            computed_tokens=[0],
            prompt_tokens=[1],
            prefill_draft_hashes=[0],
            decode_draft_hashes=[0],
        )
        with pytest.raises(ValueError, match="unknown mode"):
            runner._populate_draft_steering_index("oops", 1)
