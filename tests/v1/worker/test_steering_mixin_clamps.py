# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin-level tests for directional-clamp request wiring.

Covers the three integration traps the clamp feature adds to
``SteeringModelRunnerMixin``:

1. A clamp-only request (nonzero config hash, empty effective vectors)
   must still register a manager row — the pre-clamp guards
   (``if hash != 0 and effective_vectors``) would skip registration and
   the later ``get_row_for_config`` would raise a "scheduler bug"
   RuntimeError, killing the engine.
2. A global-clamps-only server must defeat the nothing-active
   short-circuit in ``_update_steering_buffers`` (else the clamp buffers
   are never populated and clamping is silently dropped).
3. The active->inactive transition must zero the per-hook clamp
   ``any_active`` flags alongside the steering/monitor flags.

Plus the request-clamp resolution helper and the named-module clamps tier.
CPU-only, no engine.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.clamp import (
    CLAMP_ANY_ACTIVE_ATTR,
    CLAMP_BOUNDS_ATTR,
    CLAMP_DIRS_ATTR,
    CLAMP_STRENGTH_ATTR,
)
from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

HIDDEN = 8
MAX_CONFIGS = 4
NUM_ROWS = MAX_CONFIGS + 3
K = 4
_HP = DEFAULT_HOOK_POINT.value  # "post_block"
_HP_ENUM = DEFAULT_HOOK_POINT


def _clamp_entry(axis: int, value: float) -> dict:
    vec = [0.0] * HIDDEN
    vec[axis] = 1.0
    return {"vector": vec, "value": value}


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("steering_table_post_block", torch.zeros(NUM_ROWS, HIDDEN))
        self.register_buffer(
            "steering_table_post_block_any_active", torch.zeros(1, dtype=torch.bool)
        )
        self.register_buffer("steering_index", torch.zeros(16, dtype=torch.long))
        self.register_buffer(
            "steering_token_scales", torch.zeros(16, dtype=torch.float32)
        )
        self.register_buffer("steering_row_gate", torch.zeros(16, dtype=torch.float32))
        self.register_buffer(
            "steering_decode_mask", torch.zeros(16, dtype=torch.float32)
        )
        self.register_buffer(
            CLAMP_DIRS_ATTR[_HP_ENUM], torch.zeros(NUM_ROWS, K, HIDDEN)
        )
        bounds = torch.empty(NUM_ROWS, K, 2)
        bounds[..., 0] = -float("inf")
        bounds[..., 1] = float("inf")
        self.register_buffer(CLAMP_BOUNDS_ATTR[_HP_ENUM], bounds)
        self.register_buffer(CLAMP_STRENGTH_ATTR[_HP_ENUM], torch.ones(NUM_ROWS, K))
        self.register_buffer(
            CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM], torch.zeros(1, dtype=torch.bool)
        )


class _FakeInputBatch:
    def __init__(self, reqs: list[dict]):
        self.num_reqs = len(reqs)
        self.req_ids = [r["req_id"] for r in reqs]
        self.req_id_to_index = {r["req_id"]: i for i, r in enumerate(reqs)}
        self.num_computed_tokens_cpu = np.array(
            [r["num_computed"] for r in reqs], dtype=np.int32
        )
        self.num_prompt_tokens = np.array(
            [r["num_prompt"] for r in reqs], dtype=np.int32
        )
        self.request_prefill_steering_hash = np.array(
            [r.get("prefill_hash", 0) for r in reqs], dtype=np.int64
        )
        self.request_decode_steering_hash = np.array(
            [r.get("decode_hash", 0) for r in reqs], dtype=np.int64
        )


class _FakeSchedulerOutput:
    def __init__(self, scheduled: dict[str, int]):
        self.num_scheduled_tokens = dict(scheduled)


class _Host(SteeringModelRunnerMixin):
    def __init__(self, reqs: list[dict]):
        self._steering_manager = SteeringManager(
            max_steering_configs=MAX_CONFIGS,
            device=None,
            max_clamp_directions=K,
        )
        self._max_clamp_directions = K
        self._steerable_layers_cache = {0: _Layer()}
        self._locally_owned_layers = frozenset({0})
        self._steering_reqs = {}
        self._steering_index_dirty = False
        self._steering_module_registry = {}
        self._steering_module_resolved_cache = {}
        self._steering_module_clamps = {}
        self._steering_module_clamps_effective = {}
        self._steering_module_pinned_rows = {}
        self.input_batch = _FakeInputBatch(reqs)
        self.requests = {}
        self._req_dynamic_decode = {}
        self._req_override_source = {}
        self._req_decode_sig_reported = {}
        self._pending_decode_sigs = {}
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(16, dtype=torch.long)
        self._steering_tier_gain_scratch = np.zeros(8, dtype=np.float32)
        self._steering_token_scales_pinned = torch.zeros(16, dtype=torch.float32)
        self._steering_decode_mask_scratch = np.zeros(8, dtype=np.float32)
        self._steering_decode_mask_pinned = torch.zeros(16, dtype=torch.float32)


def _layer_clamp_flag(host: _Host) -> torch.Tensor:
    return getattr(host._steerable_layers_cache[0], CLAMP_ANY_ACTIVE_ATTR[_HP_ENUM])


def _add_request(host: _Host, sp: SamplingParams, req: dict) -> None:
    host.requests[req["req_id"]] = SimpleNamespace(sampling_params=sp)
    host._steering_add_request(
        SimpleNamespace(
            req_id=req["req_id"],
            sampling_params=sp,
            prefill_steering_config_hash=req.get("prefill_hash", 0),
            decode_steering_config_hash=req.get("decode_hash", 0),
            prompt_token_ids=list(range(req["num_prompt"])),
            prompt_embeds=None,
            num_computed_tokens=req["num_computed"],
        )
    )


class TestClampOnlyRequestRegistration:
    """Trap 1: clamp-only requests must register rows (guard fix)."""

    def _sp_decode_clamps(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=4,
            decode_steering_clamps={_HP: {0: [_clamp_entry(0, 2.0)]}},
        )

    def test_clamp_only_decode_registers_at_transition(self):
        sp = self._sp_decode_clamps()
        decode_hash = sp.decode_steering_config_hash
        assert decode_hash != 0
        assert sp.effective_decode_steering is None  # truly vector-free
        req = {
            "req_id": "r1",
            "num_computed": 6,
            "num_prompt": 8,
            "prefill_hash": 0,
            "decode_hash": decode_hash,
        }
        host = _Host([req])
        _add_request(host, sp, req)
        assert not host._steering_manager.config_to_row
        # 6 + 2 >= 8: prefill completes this step -> transition registers
        # the decode config, and the row lookup must NOT raise.
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 2}))
        assert (decode_hash, "decode") in host._steering_manager.config_to_row
        row = host._steering_manager.config_to_row[(decode_hash, "decode")]
        # The registration marked tables dirty; the NEXT step's populate
        # (the first decode step) writes the clamp buffers — same
        # one-step-later semantics as steering vectors. Advance to decode
        # and run the next step.
        host.input_batch.num_computed_tokens_cpu[0] = 8
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        layer = host._steerable_layers_cache[0]
        dirs = getattr(layer, CLAMP_DIRS_ATTR[_HP_ENUM])
        assert bool(_layer_clamp_flag(host).item())
        assert dirs[row, 0, 0].item() == pytest.approx(1.0)
        # The decode token routes to the clamp row via the shared index.
        assert layer.steering_index[0].item() == row

    def test_clamp_only_prefill_registers_at_admission(self):
        sp = SamplingParams(
            max_tokens=4,
            prefill_steering_clamps={_HP: {0: [_clamp_entry(1, 3.0)]}},
        )
        prefill_hash = sp.prefill_steering_config_hash
        assert prefill_hash != 0
        req = {
            "req_id": "r1",
            "num_computed": 0,
            "num_prompt": 8,
            "prefill_hash": prefill_hash,
            "decode_hash": 0,
        }
        host = _Host([req])
        _add_request(host, sp, req)
        assert (prefill_hash, "prefill") in host._steering_manager.config_to_row

    def test_clamp_only_full_cache_hit_registers_decode(self):
        sp = self._sp_decode_clamps()
        decode_hash = sp.decode_steering_config_hash
        req = {
            "req_id": "r1",
            "num_computed": 8,  # full prefix-cache hit: straight to decode
            "num_prompt": 8,
            "prefill_hash": 0,
            "decode_hash": decode_hash,
        }
        host = _Host([req])
        _add_request(host, sp, req)
        assert (decode_hash, "decode") in host._steering_manager.config_to_row

    def test_vectors_plus_clamps_register_both(self):
        sp = SamplingParams(
            max_tokens=4,
            steering_vectors={_HP: {0: [1.0] * HIDDEN}},
            steering_clamps={_HP: {0: [_clamp_entry(0, 2.0)]}},
        )
        prefill_hash = sp.prefill_steering_config_hash
        req = {
            "req_id": "r1",
            "num_computed": 0,
            "num_prompt": 8,
            "prefill_hash": prefill_hash,
            "decode_hash": sp.decode_steering_config_hash,
        }
        host = _Host([req])
        _add_request(host, sp, req)
        key = (prefill_hash, "prefill")
        assert key in host._steering_manager.config_to_row
        assert key in host._steering_manager.config_clamps
        assert host._steering_manager.config_vectors[key][_HP]


class TestGlobalClampShortCircuit:
    """Trap 2: global-clamps-only must defeat the nothing-active
    short-circuit; trap 3: transitions must zero the clamp flags."""

    def test_global_clamps_only_populates(self):
        host = _Host([{"req_id": "r1", "num_computed": 8, "num_prompt": 8}])
        host._steering_manager.update_global_clamps(
            _HP, 0, [_clamp_entry(0, 4.0)], phase="base"
        )
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        assert bool(_layer_clamp_flag(host).item())
        layer = host._steerable_layers_cache[0]
        dirs = getattr(layer, CLAMP_DIRS_ATTR[_HP_ENUM])
        assert dirs[1, 0, 0].item() == pytest.approx(1.0)  # global prefill row
        assert dirs[2, 0, 0].item() == pytest.approx(1.0)  # global decode row

    def test_transition_zeroes_clamp_flag(self):
        host = _Host([{"req_id": "r1", "num_computed": 8, "num_prompt": 8}])
        host._steering_manager.update_global_clamps(
            _HP, 0, [_clamp_entry(0, 4.0)], phase="base"
        )
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        assert bool(_layer_clamp_flag(host).item())
        # Remove the clamps: next update crosses active -> inactive.
        host._steering_manager.clear_global_clamps()
        host._update_steering_buffers(_FakeSchedulerOutput({"r1": 1}))
        assert not bool(_layer_clamp_flag(host).item())


class TestResolveRequestClamps:
    def test_inline_only(self):
        host = _Host([])
        sp = SamplingParams(
            steering_clamps={_HP: {0: [_clamp_entry(0, 1.0)]}},
            decode_steering_clamps={_HP: {0: [_clamp_entry(1, 2.0)]}},
        )
        prefill = host._resolve_request_clamps(sp, "prefill")
        decode = host._resolve_request_clamps(sp, "decode")
        assert len(prefill[_HP][0]) == 1
        assert len(decode[_HP][0]) == 2

    def test_none_when_no_clamps(self):
        host = _Host([])
        sp = SamplingParams()
        assert host._resolve_request_clamps(sp, "prefill") is None

    def test_module_clamps_concat_with_inline(self):
        host = _Host([])
        host.register_steering_modules(
            {
                "mod": {
                    "vectors": {_HP: {0: [1.0] * HIDDEN}},
                    "clamps": {_HP: {"0": [_clamp_entry(0, 1.0)]}},
                    "decode_clamps": {_HP: {"0": [_clamp_entry(1, 2.0)]}},
                }
            }
        )
        sp = SamplingParams(
            steering_module_ref=("mod", 1.0),
            steering_clamps={_HP: {0: [_clamp_entry(2, 3.0)]}},
        )
        decode = host._resolve_request_clamps(sp, "decode")
        # module base + module decode + inline base = 3 entries
        assert len(decode[_HP][0]) == 3

    def test_k_cap_enforced(self):
        host = _Host([])
        sp = SamplingParams(
            steering_clamps={
                _HP: {0: [_clamp_entry(i % HIDDEN, 1.0) for i in range(K)]}
            },
            decode_steering_clamps={_HP: {0: [_clamp_entry(0, 1.0)]}},
        )
        with pytest.raises(ValueError, match="max_clamp_directions"):
            host._resolve_request_clamps(sp, "decode")

    def test_invalid_phase_rejected(self):
        host = _Host([])
        with pytest.raises(ValueError, match="phase"):
            host._resolve_request_clamps(SamplingParams(), "warmup")


class TestModuleClampsRegistry:
    def test_pre_materialize_clamp_only_module(self):
        host = _Host([])
        host.register_steering_modules(
            {"clamps-only": {"clamps": {_HP: {"0": [_clamp_entry(0, 5.0)]}}}}
        )
        pinned = host.pre_materialize_steering_module("clamps-only")
        assert len(pinned) == 2  # both phases pinned
        for config_hash, phase in pinned:
            key = (config_hash, phase)
            assert key in host._steering_manager.config_to_row
            assert key in host._steering_manager.config_clamps

    def test_unregister_drops_module_clamps(self):
        host = _Host([])
        host.register_steering_modules(
            {"m": {"clamps": {_HP: {"0": [_clamp_entry(0, 5.0)]}}}}
        )
        assert "m" in host._steering_module_clamps
        host.unregister_steering_modules(["m"])
        assert "m" not in host._steering_module_clamps
        assert "m" not in host._steering_module_clamps_effective

    def test_replace_clears_module_clamps(self):
        host = _Host([])
        host.register_steering_modules(
            {"m": {"clamps": {_HP: {"0": [_clamp_entry(0, 5.0)]}}}}
        )
        host.register_steering_modules(
            {"n": {"vectors": {_HP: {0: [1.0] * HIDDEN}}}}, replace=True
        )
        assert "m" not in host._steering_module_clamps


class TestSetSteeringClampsAPI:
    def test_set_and_clear_global_clamps(self):
        host = _Host([])
        result = host.set_steering_vectors(
            clamps={_HP: {0: [_clamp_entry(0, 4.0)]}},
            decode_clamps={_HP: {0: [_clamp_entry(1, 1.0)]}},
        )
        assert result[2] == [0]  # layer 0 updated
        mgr = host._steering_manager
        assert mgr.has_global_clamps
        assert 0 in mgr.global_clamp_base[_HP]
        assert 0 in mgr.global_clamp_decode[_HP]
        host.clear_steering_vectors()
        assert not mgr.has_global_clamps

    def test_validate_only_does_not_apply(self):
        host = _Host([])
        result = host.set_steering_vectors(
            clamps={_HP: {0: [_clamp_entry(0, 4.0)]}},
            validate_only=True,
        )
        assert result[2] == [0]
        assert not host._steering_manager.has_global_clamps

    def test_invalid_clamp_hook_rejected(self):
        from vllm.exceptions import SteeringVectorError

        host = _Host([])
        with pytest.raises(SteeringVectorError, match="hook"):
            host.set_steering_vectors(clamps={"nope": {0: [_clamp_entry(0, 4.0)]}})

    def test_wrong_width_clamp_rejected(self):
        from vllm.exceptions import SteeringVectorError

        host = _Host([])
        with pytest.raises(SteeringVectorError, match="size"):
            host.set_steering_vectors(
                clamps={_HP: {0: [{"vector": [1.0, 0.0], "value": 1.0}]}}
            )

    def test_replace_clears_prior_clamps(self):
        host = _Host([])
        host.set_steering_vectors(clamps={_HP: {0: [_clamp_entry(0, 4.0)]}})
        host.set_steering_vectors(vectors={_HP: {0: [1.0] * HIDDEN}}, replace=True)
        assert not host._steering_manager.has_global_clamps
