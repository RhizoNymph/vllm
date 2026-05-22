# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-hook ``any_active`` short-circuit."""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    apply_steering,
    register_steering_buffers,
)


class TestApplySteeringCPU:
    def test_inactive_returns_clone_ignores_table(self):
        hidden = torch.randn(4, 8, dtype=torch.float32)
        table = torch.full((6, 8), float("nan"))
        index = torch.full((4,), 5, dtype=torch.long)
        any_active = torch.zeros(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)

        torch.testing.assert_close(result, hidden)
        assert result.data_ptr() != hidden.data_ptr()

    def test_active_matches_indexed_gather(self):
        torch.manual_seed(0)
        hidden = torch.randn(5, 16, dtype=torch.float32)
        table = torch.randn(8, 16, dtype=torch.float32)
        index = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)
        expected = hidden + table[index]

        torch.testing.assert_close(result, expected)

    def test_active_flag_true_with_zero_table_is_identity(self):
        hidden = torch.randn(3, 4, dtype=torch.float32)
        table = torch.zeros(6, 4, dtype=torch.float32)
        index = torch.zeros(3, dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)

        torch.testing.assert_close(result, hidden)

    def test_inactive_with_index_buffer_larger_than_batch(self):
        hidden = torch.ones(3, 4, dtype=torch.float32)
        table = torch.full((6, 4), float("nan"))
        index = torch.zeros(100, dtype=torch.long)
        index[3:] = 999
        any_active = torch.zeros(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)
        torch.testing.assert_close(result, hidden)


class TestRegisterSteeringBuffersFlag:
    def test_each_hook_gets_a_bool_flag(self):
        mod = nn.Module()
        register_steering_buffers(
            mod,
            hidden_size=8,
            max_steering_tokens=16,
            max_steering_configs=4,
        )
        for hp in SteeringHookPoint:
            flag_attr = HOOK_POINT_ANY_ACTIVE_ATTR[hp]
            flag = getattr(mod, flag_attr)
            assert flag.dtype == torch.bool
            assert flag.numel() == 1
            assert not bool(flag.item())

    def test_flag_attr_co_located_with_table(self):
        mod = nn.Module()
        register_steering_buffers(
            mod,
            hidden_size=8,
            max_steering_tokens=16,
            max_steering_configs=4,
        )
        for hp in SteeringHookPoint:
            assert hasattr(mod, HOOK_POINT_TABLE_ATTR[hp])
            assert hasattr(mod, HOOK_POINT_ANY_ACTIVE_ATTR[hp])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestApplySteeringCUDA:
    def test_inactive_kernel_skips_gather(self):
        device = torch.device("cuda")
        hidden = torch.randn(4, 64, dtype=torch.float16, device=device)
        table = torch.full((6, 64), float("nan"), dtype=torch.float16, device=device)
        index = torch.full((4,), 5, dtype=torch.long, device=device)
        any_active = torch.zeros(1, dtype=torch.bool, device=device)

        result = apply_steering(hidden, table, index, any_active)

        torch.testing.assert_close(result, hidden)
        assert result.data_ptr() != hidden.data_ptr()

    def test_active_kernel_matches_eager(self):
        device = torch.device("cuda")
        torch.manual_seed(0)
        hidden = torch.randn(5, 128, dtype=torch.float16, device=device)
        table = torch.randn(8, 128, dtype=torch.float16, device=device)
        index = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device=device)
        any_active = torch.ones(1, dtype=torch.bool, device=device)

        result = apply_steering(hidden, table, index, any_active)
        expected = hidden + table[index]

        torch.testing.assert_close(result, expected)

    def test_kernel_handles_non_power_of_two_hidden(self):
        device = torch.device("cuda")
        hidden = torch.randn(3, 17, dtype=torch.float16, device=device)
        table = torch.full((6, 17), float("nan"), dtype=torch.float16, device=device)
        index = torch.full((3,), 4, dtype=torch.long, device=device)
        any_active = torch.zeros(1, dtype=torch.bool, device=device)

        result = apply_steering(hidden, table, index, any_active)
        torch.testing.assert_close(result, hidden)

        table = torch.randn(6, 17, dtype=torch.float16, device=device)
        any_active.fill_(True)
        result = apply_steering(hidden, table, index, any_active)
        expected = hidden + table[index]
        torch.testing.assert_close(result, expected)
