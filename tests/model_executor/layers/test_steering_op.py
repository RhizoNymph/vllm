# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the indexed-gather steering operation.

The steering math in each decoder layer is:
    result = hidden_states + steering_table[steering_index[:N]]

where ``steering_table`` is a per-layer buffer of shape
``(max_configs + 2, hidden_size)`` and ``steering_index`` is a shared
buffer of shape ``(max_tokens,)`` mapping each token position to its
steering table row.

Row layout:
    row 0  — always zeros (no-steering sentinel)
    row 1  — global-only steering vector
    rows 2+ — global + per-request combined vectors

These tests exercise the tensor math directly with standard PyTorch ops
rather than going through the registered custom op (which requires the
full vllm build).
"""

import builtins

import torch
from torch import nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    SteeringHookPoint,
    apply_layer_steering,
    apply_steering,
    register_steering_buffers,
)


def _apply_steering(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of the indexed-gather steering math.

    ``steering_table`` is expected to already be in ``hidden_states.dtype``
    (the model's compute dtype), so no cast is performed at the gather.
    """
    N = hidden_states.shape[0]
    return hidden_states + steering_table[steering_index[:N]]


class TestIndexedGatherSteering:
    """Tests verify the indexed-gather steering math directly."""

    def test_index_zero_no_steering(self):
        """Index 0 selects row 0 (zeros), so hidden states are unchanged."""
        batch_size, hidden_size = 4, 8
        hidden = torch.randn(batch_size, hidden_size)
        table = torch.randn(6, hidden_size)
        table[0] = 0.0  # row 0 must be zeros

        index = torch.zeros(batch_size, dtype=torch.long)

        result = _apply_steering(hidden, table, index)
        assert torch.allclose(result, hidden), (
            "Index 0 should select the zero row, leaving hidden unchanged."
        )

    def test_index_one_global_steering(self):
        """Index 1 selects the global row, adding it to hidden states."""
        batch_size, hidden_size = 4, 8
        hidden = torch.zeros(batch_size, hidden_size)
        table = torch.zeros(6, hidden_size)
        global_vec = torch.ones(hidden_size) * 3.0
        table[1] = global_vec

        index = torch.ones(batch_size, dtype=torch.long)

        result = _apply_steering(hidden, table, index)
        expected = global_vec.unsqueeze(0).expand(batch_size, -1)
        assert torch.allclose(result, expected), (
            "Index 1 should apply the global steering vector to all tokens."
        )

    def test_mixed_indices_different_vectors(self):
        """Different tokens can get different steering vectors via index."""
        batch_size, hidden_size = 6, 4
        hidden = torch.zeros(batch_size, hidden_size)
        table = torch.zeros(5, hidden_size)
        table[0] = 0.0  # no-steering
        table[1] = torch.ones(hidden_size) * 1.0  # global
        table[2] = torch.ones(hidden_size) * 10.0  # config A
        table[3] = torch.ones(hidden_size) * 100.0  # config B

        # Tokens 0,1 -> no steering; 2,3 -> global; 4 -> A; 5 -> B
        index = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long)

        result = _apply_steering(hidden, table, index)

        assert torch.allclose(result[0], torch.zeros(hidden_size))
        assert torch.allclose(result[1], torch.zeros(hidden_size))
        assert torch.allclose(result[2], torch.ones(hidden_size) * 1.0)
        assert torch.allclose(result[3], torch.ones(hidden_size) * 1.0)
        assert torch.allclose(result[4], torch.ones(hidden_size) * 10.0)
        assert torch.allclose(result[5], torch.ones(hidden_size) * 100.0)

    def test_index_buffer_larger_than_batch(self):
        """Index buffer can be larger than batch; only [:N] is used."""
        batch_size, hidden_size = 3, 4
        hidden = torch.ones(batch_size, hidden_size)
        table = torch.zeros(6, hidden_size)
        table[2] = torch.ones(hidden_size) * 5.0

        # Index buffer is much larger than batch.
        index = torch.zeros(100, dtype=torch.long)
        index[:batch_size] = 2
        # Indices beyond batch_size should be irrelevant.
        index[batch_size:] = 999  # out-of-bounds if ever accessed

        result = _apply_steering(hidden, table, index)
        expected = hidden + torch.ones(hidden_size) * 5.0
        assert torch.allclose(result, expected), (
            "Only index[:N] should be read; extra elements must be ignored."
        )

    def test_output_shape_and_dtype_match_input(self):
        """Output shape and dtype must match the input hidden states."""
        batch_size, hidden_size = 5, 16
        hidden = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        table = torch.randn(6, hidden_size, dtype=torch.float32)
        index = torch.zeros(batch_size, dtype=torch.long)

        result = _apply_steering(hidden, table, index)
        assert result.shape == hidden.shape
        assert result.dtype == hidden.dtype

    def test_zero_table_is_noop(self):
        """An all-zeros table means steering is a noop for any index."""
        batch_size, hidden_size = 4, 8
        hidden = torch.randn(batch_size, hidden_size)
        table = torch.zeros(6, hidden_size)
        index = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        result = _apply_steering(hidden, table, index)
        assert torch.allclose(result, hidden), (
            "All-zeros table should leave hidden states unchanged."
        )

    def test_inplace_table_update_visible_on_next_use(self):
        """In-place updates to the steering table are visible on next call."""
        batch_size, hidden_size = 2, 4
        hidden = torch.zeros(batch_size, hidden_size)
        table = torch.zeros(6, hidden_size)
        index = torch.ones(batch_size, dtype=torch.long)

        # First call: row 1 is zeros.
        result1 = _apply_steering(hidden, table, index)
        assert torch.allclose(result1, torch.zeros(batch_size, hidden_size))

        # Simulate model runner updating the table in-place.
        table[1] = torch.ones(hidden_size) * 7.0

        # Second call: the updated row 1 should now be applied.
        result2 = _apply_steering(hidden, table, index)
        expected = torch.ones(batch_size, hidden_size) * 7.0
        assert torch.allclose(result2, expected), (
            "In-place table update should be visible on the next forward pass."
        )

    def test_inplace_index_update_visible_on_next_use(self):
        """In-place updates to the index buffer are visible on next call."""
        batch_size, hidden_size = 4, 4
        hidden = torch.zeros(batch_size, hidden_size)
        table = torch.zeros(6, hidden_size)
        table[0] = 0.0
        table[1] = torch.ones(hidden_size) * 1.0
        table[2] = torch.ones(hidden_size) * 20.0

        index = torch.zeros(batch_size, dtype=torch.long)

        # First call: all index 0 -> no steering.
        result1 = _apply_steering(hidden, table, index)
        assert torch.allclose(result1, torch.zeros(batch_size, hidden_size))

        # Simulate model runner reassigning tokens to different configs.
        index[0] = 1  # global
        index[1] = 2  # per-request config
        index[2] = 0  # no steering
        index[3] = 2  # per-request config

        result2 = _apply_steering(hidden, table, index)
        assert torch.allclose(result2[0], torch.ones(hidden_size) * 1.0)
        assert torch.allclose(result2[1], torch.ones(hidden_size) * 20.0)
        assert torch.allclose(result2[2], torch.zeros(hidden_size))
        assert torch.allclose(result2[3], torch.ones(hidden_size) * 20.0)

    def test_inactive_flag_skips_gather_and_returns_clone(self):
        hidden = torch.randn(4, 8, dtype=torch.float32)
        table = torch.full((6, 8), float("nan"), dtype=torch.float32)
        index = torch.full((4,), 5, dtype=torch.long)
        any_active = torch.zeros(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)

        torch.testing.assert_close(result, hidden)
        assert result.data_ptr() != hidden.data_ptr()

    def test_active_flag_matches_indexed_gather(self):
        hidden = torch.randn(3, 4, dtype=torch.float32)
        table = torch.randn(6, 4, dtype=torch.float32)
        index = torch.tensor([0, 2, 5], dtype=torch.long)
        any_active = torch.ones(1, dtype=torch.bool)

        result = apply_steering(hidden, table, index, any_active)

        torch.testing.assert_close(result, hidden + table[index])

    def test_register_steering_buffers_adds_any_active_flags(self):
        module = nn.Module()

        register_steering_buffers(
            module,
            hidden_size=4,
            max_steering_tokens=8,
            max_steering_configs=1,
            dtype=torch.float32,
        )

        for hp in SteeringHookPoint:
            flag = getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hp])
            assert flag.dtype == torch.bool
            assert flag.shape == (1,)
            assert not bool(flag.item())


class TestApplyLayerSteeringHotPath:
    def test_no_buffers_does_not_import_sae_steering(self, monkeypatch):
        module = nn.Module()
        module.layer_idx = 0
        hidden = torch.randn(2, 4)

        self._forbid_sae_import(monkeypatch)

        out = apply_layer_steering(module, hidden, SteeringHookPoint.POST_MLP)
        assert out is hidden

    def test_additive_only_does_not_import_sae_steering(self, monkeypatch):
        module = nn.Module()
        module.layer_idx = 0
        register_steering_buffers(
            module,
            hidden_size=4,
            max_steering_tokens=8,
            max_steering_configs=1,
            dtype=torch.float32,
        )
        hidden = torch.randn(2, 4)

        self._forbid_sae_import(monkeypatch)

        out = apply_layer_steering(module, hidden, SteeringHookPoint.POST_MLP)
        assert torch.allclose(out, hidden)

    @staticmethod
    def _forbid_sae_import(monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "vllm.model_executor.layers.sae_steering":
                raise AssertionError("no-SAE hot path imported sae_steering")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
