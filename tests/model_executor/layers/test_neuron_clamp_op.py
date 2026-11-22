# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the ``apply_neuron_clamp`` custom op and its model hook."""

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.neuron_clamp as neuron_clamp_mod
from vllm.model_executor.layers.neuron_clamp import (
    NEURON_CLAMP_ANY_ACTIVE_ATTR,
    NEURON_CLAMP_INDEX_ATTR,
    NEURON_CLAMP_LOCAL_SIZE_ATTR,
    NEURON_CLAMP_TP_RANK_ATTR,
    apply_mlp_neuron_clamp,
    get_neuron_clamp_module,
    neuron_clamp_reference,
    register_decoder_mlp_neuron_clamp,
    register_neuron_clamp_buffers,
)
from vllm.model_executor.layers.steering import share_steering_index_across_layers

pytestmark = pytest.mark.cpu_test

ROWS = 6
K = 4
INTER = 10


def _tables():
    idx = torch.full((ROWS, K), -1, dtype=torch.int32)
    mul = torch.ones(ROWS, K)
    add = torch.zeros(ROWS, K)
    return idx, mul, add


def _set(idx, mul, add, row, slot, neuron, mode, value):
    idx[row, slot] = neuron
    if mode == "set":
        mul[row, slot], add[row, slot] = 0.0, value
    elif mode == "add":
        mul[row, slot], add[row, slot] = 1.0, value
    else:
        mul[row, slot], add[row, slot] = value, 0.0


def _op(act, idx, mul, add, sidx, flag):
    # CPU tensors: call the op's Python implementation (the registered
    # dispatch key is the platform's device key).
    neuron_clamp_mod.apply_neuron_clamp(act, idx, mul, add, sidx, flag)
    return act


@pytest.fixture(autouse=True)
def _cpu_dispatch(monkeypatch):
    """Route ``torch.ops.vllm.apply_neuron_clamp`` to the Python impl so the
    model hook can run on CPU tensors under a CUDA platform build."""
    monkeypatch.setattr(
        torch.ops.vllm, "apply_neuron_clamp", neuron_clamp_mod.apply_neuron_clamp
    )


class TestOpSemantics:
    def setup_method(self):
        torch.manual_seed(0)
        self.act = torch.randn(5, INTER)
        self.idx, self.mul, self.add = _tables()
        self.flag = torch.ones(1, dtype=torch.bool)

    def test_set_add_scale(self):
        _set(self.idx, self.mul, self.add, 3, 0, 2, "set", 7.0)
        _set(self.idx, self.mul, self.add, 3, 1, 5, "add", 1.5)
        _set(self.idx, self.mul, self.add, 3, 2, 9, "scale", -2.0)
        sidx = torch.full((8,), 3, dtype=torch.long)
        before = self.act.clone()
        _op(self.act, self.idx, self.mul, self.add, sidx, self.flag)
        exp = before.clone()
        exp[:, 2] = 7.0
        exp[:, 5] = before[:, 5] + 1.5
        exp[:, 9] = before[:, 9] * -2.0
        torch.testing.assert_close(self.act, exp, rtol=0, atol=0)

    def test_row0_and_unclamped_rows_untouched(self):
        _set(self.idx, self.mul, self.add, 3, 0, 2, "set", 7.0)
        _set(self.idx, self.mul, self.add, 4, 0, 1, "set", -1.0)
        sidx = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], dtype=torch.long)
        before = self.act.clone()
        _op(self.act, self.idx, self.mul, self.add, sidx, self.flag)
        exp = before.clone()
        exp[3, 2] = 7.0
        exp[4, 1] = -1.0
        torch.testing.assert_close(self.act, exp, rtol=0, atol=0)

    def test_flag_clear_is_noop(self):
        _set(self.idx, self.mul, self.add, 3, 0, 2, "set", 7.0)
        sidx = torch.full((8,), 3, dtype=torch.long)
        before = self.act.clone()
        _op(
            self.act,
            self.idx,
            self.mul,
            self.add,
            sidx,
            torch.zeros(1, dtype=torch.bool),
        )
        torch.testing.assert_close(self.act, before, rtol=0, atol=0)

    def test_set_is_exact_even_for_nonfinite(self):
        self.act[:, 2] = float("inf")
        self.act[:, 3] = float("nan")
        _set(self.idx, self.mul, self.add, 3, 0, 2, "set", 0.0)
        _set(self.idx, self.mul, self.add, 3, 1, 3, "set", 0.25)
        sidx = torch.full((8,), 3, dtype=torch.long)
        _op(self.act, self.idx, self.mul, self.add, sidx, self.flag)
        assert torch.all(self.act[:, 2] == 0.0)
        assert torch.all(self.act[:, 3] == 0.25)

    def test_index_longer_than_batch(self):
        _set(self.idx, self.mul, self.add, 5, 0, 0, "set", 3.0)
        sidx = torch.tensor([5, 0, 5, 0, 5, 5, 5, 5, 5], dtype=torch.long)
        before = self.act.clone()
        _op(self.act, self.idx, self.mul, self.add, sidx, self.flag)
        exp = before.clone()
        exp[[0, 2, 4], 0] = 3.0
        torch.testing.assert_close(self.act, exp, rtol=0, atol=0)

    def test_empty_batch(self):
        act = torch.empty(0, INTER)
        _op(
            act,
            self.idx,
            self.mul,
            self.add,
            torch.zeros(4, dtype=torch.long),
            self.flag,
        )

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_low_precision_matches_reference_rounding(self, dtype):
        act = torch.randn(5, INTER).to(dtype)
        _set(self.idx, self.mul, self.add, 3, 0, 4, "add", 0.1)
        _set(self.idx, self.mul, self.add, 3, 1, 6, "set", -30.0)
        _set(self.idx, self.mul, self.add, 3, 2, 7, "scale", 1.7)
        sidx = torch.full((8,), 3, dtype=torch.long)
        exp = act.clone()
        exp[:, 4] = (act[:, 4].float() + 0.1).to(dtype)
        exp[:, 6] = torch.tensor(-30.0).to(dtype)
        exp[:, 7] = (act[:, 7].float() * 1.7).to(dtype)
        _op(act, self.idx, self.mul, self.add, sidx, self.flag)
        torch.testing.assert_close(act, exp, rtol=0, atol=0)

    def test_reference_does_not_mutate(self):
        _set(self.idx, self.mul, self.add, 3, 0, 2, "set", 7.0)
        sidx = torch.full((8,), 3, dtype=torch.long)
        before = self.act.clone()
        out = neuron_clamp_reference(
            self.act, self.idx, self.mul, self.add, sidx, self.flag
        )
        torch.testing.assert_close(self.act, before, rtol=0, atol=0)
        assert torch.all(out[:, 2] == 7.0)


# ---------------------------------------------------------------------------
# buffers + model hook
# ---------------------------------------------------------------------------


class _FakeDown(nn.Module):
    def __init__(self, inter_local, tp_rank=0):
        super().__init__()
        self.input_size_per_partition = inter_local
        self.tp_rank = tp_rank


class _FakeMLP(nn.Module):
    def __init__(self, inter_local, tp_rank=0):
        super().__init__()
        self.down_proj = _FakeDown(inter_local, tp_rank)

    def forward(self, act):
        return apply_mlp_neuron_clamp(self, act)


class _FakeLayer(nn.Module):
    def __init__(self, layer_idx, inter_local=INTER, tp_rank=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.register_buffer("steering_index", torch.zeros(16, dtype=torch.long))
        self.mlp = _FakeMLP(inter_local, tp_rank)


class TestBuffers:
    def test_disabled_registers_nothing(self):
        for max_configs, k in ((0, 16), (4, 0)):
            layer = _FakeLayer(0)
            register_decoder_mlp_neuron_clamp(
                layer, max_steering_configs=max_configs, max_clamps_per_layer=k
            )
            assert get_neuron_clamp_module(layer) is None
            act = torch.randn(3, INTER)
            assert torch.equal(layer.mlp(act.clone()), act)

    def test_register_shapes_and_geometry(self):
        layer = _FakeLayer(0, inter_local=5, tp_rank=1)
        register_decoder_mlp_neuron_clamp(
            layer, max_steering_configs=4, max_clamps_per_layer=3
        )
        mlp = get_neuron_clamp_module(layer)
        assert mlp is layer.mlp
        idx = getattr(mlp, NEURON_CLAMP_INDEX_ATTR)
        assert idx.shape == (4 + 3, 3) and idx.dtype == torch.int32
        assert torch.all(idx == -1)
        assert not bool(getattr(mlp, NEURON_CLAMP_ANY_ACTIVE_ATTR))
        assert getattr(mlp, NEURON_CLAMP_TP_RANK_ATTR) == 1
        assert getattr(mlp, NEURON_CLAMP_LOCAL_SIZE_ATTR) == 5
        assert mlp.steering_index is layer.steering_index
        # Buffers are non-persistent: they never enter the state dict.
        assert NEURON_CLAMP_INDEX_ATTR not in mlp.state_dict()

    def test_no_steering_index_registers_nothing(self):
        mlp = _FakeMLP(INTER)
        register_neuron_clamp_buffers(
            mlp,
            steering_index=None,
            max_steering_configs=4,
            max_clamps_per_layer=4,
            intermediate_size_local=INTER,
            tp_rank=0,
        )
        assert not hasattr(mlp, NEURON_CLAMP_INDEX_ATTR)

    def test_share_steering_index_repoints_mlp(self):
        layers = [_FakeLayer(i) for i in range(3)]
        for layer in layers:
            register_decoder_mlp_neuron_clamp(
                layer, max_steering_configs=2, max_clamps_per_layer=2
            )
        share_steering_index_across_layers(layers)
        shared = layers[0].steering_index
        for layer in layers:
            assert layer.steering_index is shared
            assert layer.mlp.steering_index is shared

    def test_hook_applies_through_shared_index(self):
        layers = [_FakeLayer(i) for i in range(2)]
        for layer in layers:
            register_decoder_mlp_neuron_clamp(
                layer, max_steering_configs=2, max_clamps_per_layer=2
            )
        share_steering_index_across_layers(layers)
        mlp = layers[1].mlp
        getattr(mlp, NEURON_CLAMP_INDEX_ATTR)[3, 0] = 4
        mlp.neuron_clamp_mul[3, 0] = 0.0
        mlp.neuron_clamp_add[3, 0] = 9.0
        getattr(mlp, NEURON_CLAMP_ANY_ACTIVE_ATTR).fill_(True)
        layers[0].steering_index[:2] = torch.tensor([3, 0])
        act = torch.randn(2, INTER)
        out = mlp(act.clone())
        assert out[0, 4] == 9.0
        torch.testing.assert_close(out[1], act[1], rtol=0, atol=0)
        # Layer 0's tables are empty -> untouched.
        torch.testing.assert_close(layers[0].mlp(act.clone()), act, rtol=0, atol=0)


class TestQwen3MLPHook:
    def test_qwen3_mlp_forward_calls_clamp(self, monkeypatch):
        from vllm.model_executor.models import qwen3

        calls = []

        def fake_clamp(mlp, act):
            calls.append(act.shape)
            return act

        monkeypatch.setattr(qwen3, "apply_mlp_neuron_clamp", fake_clamp)

        class _Lin(nn.Module):
            def __init__(self, out):
                super().__init__()
                self.out = out

            def forward(self, x):
                return torch.zeros(x.shape[0], self.out), None

        mlp = qwen3.Qwen3MLP.__new__(qwen3.Qwen3MLP)
        nn.Module.__init__(mlp)
        mlp.gate_up_proj = _Lin(2 * INTER)
        mlp.down_proj = _Lin(4)
        mlp.act_fn = lambda x: x[..., :INTER]
        out = mlp(torch.zeros(3, 4))
        assert out.shape == (3, 4)
        assert calls == [torch.Size([3, INTER])]
