# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SteeringManager / runner-mixin tests for per-request MLP neuron clamps."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config.neuron_clamp_types import NeuronClamp
from vllm.model_executor.layers.neuron_clamp import (
    NEURON_CLAMP_ADD_ATTR,
    NEURON_CLAMP_ANY_ACTIVE_ATTR,
    NEURON_CLAMP_INDEX_ATTR,
    NEURON_CLAMP_MUL_ATTR,
    register_decoder_mlp_neuron_clamp,
)
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_TABLE_ATTR,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

pytestmark = pytest.mark.cpu_test

HIDDEN = 8
MAX_CONFIGS = 4
K = 3
INTER = 12
_HP = DEFAULT_HOOK_POINT.value
_TABLE_ATTR = HOOK_POINT_TABLE_ATTR[DEFAULT_HOOK_POINT]


class _Down(nn.Module):
    def __init__(self, local, tp_rank):
        super().__init__()
        self.input_size_per_partition = local
        self.tp_rank = tp_rank


class _MLP(nn.Module):
    def __init__(self, local, tp_rank):
        super().__init__()
        self.down_proj = _Down(local, tp_rank)


class _Layer(nn.Module):
    def __init__(self, idx, *, local=INTER, tp_rank=0, clamps=True):
        super().__init__()
        self.layer_idx = idx
        self.register_buffer(_TABLE_ATTR, torch.zeros(MAX_CONFIGS + 3, HIDDEN))
        self.register_buffer("steering_index", torch.zeros(32, dtype=torch.long))
        self.mlp = _MLP(local, tp_rank)
        register_decoder_mlp_neuron_clamp(
            self,
            max_steering_configs=MAX_CONFIGS,
            max_clamps_per_layer=K if clamps else 0,
        )


def _layers(n=4, **kw):
    return {i: _Layer(i, **kw) for i in range(n)}


def _tables(layer):
    m = layer.mlp
    return (
        getattr(m, NEURON_CLAMP_INDEX_ATTR),
        getattr(m, NEURON_CLAMP_MUL_ATTR),
        getattr(m, NEURON_CLAMP_ADD_ATTR),
        bool(getattr(m, NEURON_CLAMP_ANY_ACTIVE_ATTR)),
    )


def _clamps(*specs):
    return tuple(NeuronClamp(*s) for s in specs)


class TestManagerClamps:
    def test_clamp_only_config_gets_row(self):
        mgr = SteeringManager(MAX_CONFIGS)
        row = mgr.register_config(
            123, {}, phase="prefill", neuron_clamps=_clamps((1, 5, "set", 0.0))
        )
        assert row >= 3
        assert mgr.get_row_for_config(123, is_prefill=True) == row
        assert mgr.has_neuron_clamps

    def test_populate_writes_rows(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers()
        r1 = mgr.register_config(
            1,
            {},
            phase="prefill",
            neuron_clamps=_clamps(
                (1, 5, "set", 0.0), (1, 7, "scale", 2.0), (2, 0, "add", -1.5)
            ),
        )
        r2 = mgr.register_config(
            2,
            {_HP: {1: [1.0] * HIDDEN}},
            phase="decode",
            neuron_clamps=_clamps((1, 5, "set", -30.0)),
        )
        mgr.populate_steering_tables(layers)
        idx, mul, add, active = _tables(layers[1])
        assert active
        assert idx[r1].tolist() == [5, 7, -1]
        assert mul[r1].tolist() == [0.0, 2.0, 1.0]
        assert add[r1].tolist() == [0.0, 0.0, 0.0]
        assert idx[r2].tolist() == [5, -1, -1]
        assert add[r2, 0].item() == -30.0
        # Rows 0/1/2 never carry clamps.
        assert torch.all(idx[:3] == -1)
        idx2, mul2, add2, active2 = _tables(layers[2])
        assert active2 and idx2[r1].tolist() == [0, -1, -1]
        assert mul2[r1, 0] == 1.0 and add2[r1, 0] == -1.5
        assert torch.all(idx2[r2] == -1)
        # Untouched layers stay inactive.
        for i in (0, 3):
            idx_i, _, _, act_i = _tables(layers[i])
            assert not act_i and torch.all(idx_i == -1)
        # Vector part of the combined config still lands in the steering row.
        assert torch.all(getattr(layers[1], _TABLE_ATTR)[r2] == 1.0)

    def test_release_clears_layer(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers()
        mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((1, 5, "set", 0.0))
        )
        mgr.register_config(
            2, {}, phase="prefill", neuron_clamps=_clamps((2, 5, "set", 0.0))
        )
        mgr.populate_steering_tables(layers)
        assert _tables(layers[1])[3] and _tables(layers[2])[3]
        mgr.release_config(1, "prefill")
        assert mgr.has_neuron_clamps
        mgr.populate_steering_tables(layers)
        idx, _, _, active = _tables(layers[1])
        assert not active and torch.all(idx == -1)
        assert _tables(layers[2])[3]
        mgr.release_config(2, "prefill")
        assert not mgr.has_neuron_clamps
        mgr.populate_steering_tables(layers)
        assert not _tables(layers[2])[3]

    def test_refcount_shares_row(self):
        mgr = SteeringManager(MAX_CONFIGS)
        c = _clamps((1, 5, "set", 0.0))
        r1 = mgr.register_config(9, {}, phase="decode", neuron_clamps=c)
        r2 = mgr.register_config(9, {}, phase="decode", neuron_clamps=c)
        assert r1 == r2 and mgr.num_active_configs == 1
        mgr.release_config(9, "decode")
        assert mgr.has_neuron_clamps
        mgr.release_config(9, "decode")
        assert not mgr.has_neuron_clamps

    def test_capacity_counts_clamp_only(self):
        mgr = SteeringManager(2)
        mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((0, 0, "set", 0.0))
        )
        mgr.register_config(
            2, {}, phase="prefill", neuron_clamps=_clamps((0, 1, "set", 0.0))
        )
        with pytest.raises(RuntimeError, match="No free steering table rows"):
            mgr.register_config(
                3, {}, phase="prefill", neuron_clamps=_clamps((0, 2, "set", 0.0))
            )

    def test_overflow_raises(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers()
        mgr.register_config(
            1,
            {},
            phase="prefill",
            neuron_clamps=_clamps(*[(1, n, "set", 0.0) for n in range(K + 1)]),
        )
        with pytest.raises(RuntimeError, match="overflow"):
            mgr.populate_steering_tables(layers)

    def test_clear_flags(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers()
        mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((3, 5, "set", 0.0))
        )
        mgr.populate_steering_tables(layers)
        assert _tables(layers[3])[3]
        mgr.clear_neuron_clamp_flags(layers)
        assert not _tables(layers[3])[3]

    def test_pp_missing_layer_skipped(self):
        # This PP stage owns layers 0..1 only; clamps for layer 3 are skipped
        # but the row is still allocated (rank-aligned row ids).
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers(2)
        row = mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((3, 5, "set", 0.0))
        )
        assert row == 3
        mgr.populate_steering_tables(layers)
        assert not any(_tables(lay)[3] for lay in layers.values())

    def test_layer_without_clamp_tables_ignored(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = {0: _Layer(0, clamps=False)}
        mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((0, 5, "set", 0.0))
        )
        mgr.populate_steering_tables(layers)  # must not raise


class TestTensorParallel:
    def test_tp2_each_rank_fills_only_owned_slots(self):
        local = INTER // 2  # 6
        clamps = _clamps((1, 2, "set", 0.0), (1, 6, "add", 1.0), (1, 11, "scale", 3.0))
        ranks = []
        for tp_rank in range(2):
            mgr = SteeringManager(MAX_CONFIGS)
            layers = _layers(2, local=local, tp_rank=tp_rank)
            row = mgr.register_config(77, {}, phase="decode", neuron_clamps=clamps)
            mgr.populate_steering_tables(layers)
            ranks.append((row, _tables(layers[1])))
        (row0, (idx0, mul0, add0, act0)), (row1, (idx1, mul1, add1, act1)) = ranks
        assert row0 == row1  # determinism contract: identical row ids
        assert act0 and act1
        assert idx0[row0].tolist() == [2, -1, -1]
        assert mul0[row0, 0] == 0.0
        assert idx1[row1].tolist() == [0, 5, -1]
        assert (mul1[row1, 0].item(), add1[row1, 0].item()) == (1.0, 1.0)
        assert (mul1[row1, 1].item(), add1[row1, 1].item()) == (3.0, 0.0)

    def test_tp2_rank_without_owned_neuron_stays_inactive(self):
        mgr = SteeringManager(MAX_CONFIGS)
        layers = _layers(2, local=INTER // 2, tp_rank=1)
        mgr.register_config(
            1, {}, phase="prefill", neuron_clamps=_clamps((1, 3, "set", 0.0))
        )
        mgr.populate_steering_tables(layers)
        assert not _tables(layers[1])[3]


# ---------------------------------------------------------------------------
# Runner mixin registration helper
# ---------------------------------------------------------------------------


def _mixin(mgr):
    obj = SteeringModelRunnerMixin.__new__(SteeringModelRunnerMixin)
    obj._steering_manager = mgr
    obj._locally_owned_layers = frozenset(range(4))
    obj._steering_module_registry = {}
    obj._steering_module_resolved_cache = {}
    return obj


class TestMixinRegistration:
    def test_clamp_only_registers(self):
        mgr = SteeringManager(MAX_CONFIGS)
        runner = _mixin(mgr)
        sp = SamplingParams(
            neuron_clamps=[{"layer": 1, "neuron": 5, "mode": "set", "value": 0.0}]
        )
        h = sp.prefill_steering_config_hash
        assert runner._register_request_phase_config(sp, h, "prefill")
        assert mgr.config_neuron_clamps[(h, "prefill")] == sp.effective_prefill_neuron_clamps
        assert mgr.config_vectors[(h, "prefill")] == {}

    def test_vectors_and_clamps_one_row(self):
        mgr = SteeringManager(MAX_CONFIGS)
        runner = _mixin(mgr)
        sp = SamplingParams(
            steering_vectors={_HP: {1: [1.0] * HIDDEN}},
            neuron_clamps=[{"layer": 1, "neuron": 5, "mode": "set", "value": 0.0}],
        )
        h = sp.decode_steering_config_hash
        assert runner._register_request_phase_config(sp, h, "decode")
        assert mgr.num_active_configs == 1
        assert (h, "decode") in mgr.config_neuron_clamps
        assert 1 in mgr.config_vectors[(h, "decode")][_HP]

    def test_phase_without_anything_skips(self):
        mgr = SteeringManager(MAX_CONFIGS)
        runner = _mixin(mgr)
        sp = SamplingParams(
            neuron_clamps=[
                {
                    "layer": 1,
                    "neuron": 5,
                    "mode": "set",
                    "value": 0.0,
                    "phase": "prefill",
                }
            ]
        )
        assert sp.decode_steering_config_hash == 0
        assert not runner._register_request_phase_config(
            sp, sp.decode_steering_config_hash, "decode"
        )
        assert mgr.num_active_configs == 0

    def test_vector_only_unchanged(self):
        mgr = SteeringManager(MAX_CONFIGS)
        runner = _mixin(mgr)
        sp = SamplingParams(steering_vectors={_HP: {1: [1.0] * HIDDEN}})
        h = sp.prefill_steering_config_hash
        assert runner._register_request_phase_config(sp, h, "prefill")
        assert (h, "prefill") not in mgr.config_neuron_clamps

    def test_no_manager(self):
        runner = _mixin(None)
        sp = SamplingParams(
            neuron_clamps=[{"layer": 1, "neuron": 5, "mode": "set", "value": 0.0}]
        )
        assert not runner._register_request_phase_config(
            sp, sp.prefill_steering_config_hash, "prefill"
        )


class TestOverrideRejectsClampedRequest:
    def test_dynamic_override_rejected_for_decode_clamps(self):
        mgr = SteeringManager(MAX_CONFIGS, max_dynamic_steering_configs=2)
        runner = _mixin(mgr)
        runner._steerable_layers_cache = _layers(2)
        runner._req_dynamic_decode = {}
        runner._req_override_source = {}
        sp = SamplingParams(
            neuron_clamps=[{"layer": 1, "neuron": 5, "mode": "set", "value": 0.0}]
        )
        runner._steering_reqs = {"r": SimpleNamespace(sampling_params=sp)}
        runner.input_batch = SimpleNamespace(
            req_id_to_index={"r": 0},
            num_computed_tokens_cpu=[10],
            num_prompt_tokens=[4],
        )
        action = SimpleNamespace(req_id="r", vectors={_HP: {1: [0.0] * HIDDEN}})
        assert not runner._apply_request_override(action, source="t")
        assert runner._req_dynamic_decode == {}
