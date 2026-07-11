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

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    SteeringOpArgs,
    apply_layer_steering,
    apply_steering,
    register_steering_buffers,
    steering_monitor,
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


def _apply_steering_scaled(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    steering_scales: torch.Tensor,
) -> torch.Tensor:
    """Reference for the §5.3 per-row scale math the kernel computes:
    ``hidden + table[index] * scales[index]``."""
    N = hidden_states.shape[0]
    rows = steering_index[:N]
    return hidden_states + steering_table[rows] * steering_scales[rows].unsqueeze(-1)


class TestPerRowScale:
    """Per-row strength scale (§5.3) math."""

    def test_scale_one_is_identity(self):
        hidden = torch.randn(4, 8)
        table = torch.randn(6, 8)
        index = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        scales = torch.ones(6)
        torch.testing.assert_close(
            _apply_steering_scaled(hidden, table, index, scales),
            _apply_steering(hidden, table, index),
        )

    def test_scale_multiplies_gathered_row(self):
        hidden = torch.zeros(3, 4)
        table = torch.zeros(6, 4)
        table[2] = torch.ones(4) * 5.0
        index = torch.full((3,), 2, dtype=torch.long)
        scales = torch.ones(6)
        scales[2] = 0.5
        result = _apply_steering_scaled(hidden, table, index, scales)
        assert torch.allclose(result, torch.ones(3, 4) * 2.5)  # 5.0 * 0.5

    def test_per_row_scales_are_independent(self):
        hidden = torch.zeros(3, 4)
        table = torch.zeros(6, 4)
        table[1] = torch.ones(4) * 2.0
        table[2] = torch.ones(4) * 4.0
        index = torch.tensor([1, 2, 0], dtype=torch.long)
        scales = torch.ones(6)
        scales[1] = 3.0  # row 1 tripled
        scales[2] = 0.0  # row 2 disabled
        result = _apply_steering_scaled(hidden, table, index, scales)
        assert torch.allclose(result[0], torch.ones(4) * 6.0)  # 2.0 * 3.0
        assert torch.allclose(result[1], torch.zeros(4))  # 4.0 * 0.0
        assert torch.allclose(result[2], torch.zeros(4))  # row 0 sentinel


def _apply_full(hidden, table, index, scales, dvec, tscales, row_gate=None):
    """Reference for the full kernel math (§5.3 row scale + §5.4 tier +
    Phase-2 row gate): ``hidden + table[row]*scales[row]*row_gate[token]
    + dvec * tscales[token]``. ``row_gate`` defaults to 1.0 (rows full)."""
    n = hidden.shape[0]
    rows = index[:n]
    if row_gate is None:
        row_gate = torch.ones(n)
    row_term = table[rows] * scales[rows].unsqueeze(-1) * row_gate[:n].unsqueeze(-1)
    tier_term = dvec.unsqueeze(0) * tscales[:n].unsqueeze(-1)
    return hidden + row_term + tier_term


class TestRowGate:
    """Phase-2 per-token row gate: row term *= row_gate[token]."""

    def test_gate_one_is_identity(self):
        hidden = torch.randn(3, 4)
        table = torch.randn(6, 4)
        index = torch.tensor([1, 2, 3], dtype=torch.long)
        scales = torch.ones(6)
        dvec = torch.zeros(4)
        tscales = torch.zeros(3)
        rgate = torch.ones(3)
        torch.testing.assert_close(
            _apply_full(hidden, table, index, scales, dvec, tscales, rgate),
            _apply_full(hidden, table, index, scales, dvec, tscales),
        )

    def test_gate_scales_the_row_per_token(self):
        hidden = torch.zeros(3, 4)
        table = torch.zeros(6, 4)
        table[2] = torch.full((4,), 4.0)  # decode row
        index = torch.full((3,), 2, dtype=torch.long)
        scales = torch.ones(6)
        dvec = torch.zeros(4)
        tscales = torch.zeros(3)
        # Token 0 full (1.0), token 1 prefill (1.0 — never gated), token 2 half.
        rgate = torch.tensor([1.0, 1.0, 0.5])
        result = _apply_full(hidden, table, index, scales, dvec, tscales, rgate)
        assert torch.allclose(result[0], torch.full((4,), 4.0))
        assert torch.allclose(result[1], torch.full((4,), 4.0))
        assert torch.allclose(result[2], torch.full((4,), 2.0))  # 4 * 0.5

    def test_gate_zero_disables_the_row(self):
        hidden = torch.zeros(2, 4)
        table = torch.zeros(6, 4)
        table[3] = torch.full((4,), 7.0)
        index = torch.full((2,), 3, dtype=torch.long)
        scales = torch.ones(6)
        rgate = torch.tensor([0.0, 1.0])
        result = _apply_full(hidden, table, index, scales, torch.zeros(4),
                             torch.zeros(2), rgate)
        assert torch.allclose(result[0], torch.zeros(4))   # gated off
        assert torch.allclose(result[1], torch.full((4,), 7.0))

    def test_gate_and_scale_and_tier_compose(self):
        hidden = torch.zeros(1, 4)
        table = torch.zeros(6, 4)
        table[2] = torch.full((4,), 2.0)
        index = torch.zeros(1, dtype=torch.long)
        index[0] = 2
        scales = torch.ones(6)
        scales[2] = 3.0  # row scale ×3
        dvec = torch.full((4,), 5.0)
        tscales = torch.tensor([1.0])  # tier ×1
        rgate = torch.tensor([0.5])    # row gate ×0.5
        result = _apply_full(hidden, table, index, scales, dvec, tscales, rgate)
        # row: 2*3*0.5 = 3 ; tier: 5*1 = 5 ; total 8
        assert torch.allclose(result, torch.full((1, 4), 8.0))


class TestDynamicTierTerm:
    """Dedicated-gather dynamic tier (§5.4): out += dvec * token_scale."""

    def test_zero_gate_is_no_tier(self):
        hidden = torch.randn(3, 4)
        table = torch.randn(6, 4)
        index = torch.tensor([1, 2, 0], dtype=torch.long)
        scales = torch.ones(6)
        dvec = torch.full((4,), 9.0)
        tscales = torch.zeros(3)  # gate off everywhere
        result = _apply_full(hidden, table, index, scales, dvec, tscales)
        torch.testing.assert_close(
            result, hidden + table[index] * scales[index].unsqueeze(-1)
        )

    def test_tier_added_per_gated_token(self):
        hidden = torch.zeros(3, 4)
        table = torch.zeros(6, 4)
        index = torch.zeros(3, dtype=torch.long)  # sentinel rows
        scales = torch.ones(6)
        dvec = torch.full((4,), 2.0)
        # Token 0 decode (gain 3), token 1 prefill (0), token 2 decode (3).
        tscales = torch.tensor([3.0, 0.0, 3.0])
        result = _apply_full(hidden, table, index, scales, dvec, tscales)
        assert torch.allclose(result[0], torch.full((4,), 6.0))  # 2*3
        assert torch.allclose(result[1], torch.zeros(4))  # gated off
        assert torch.allclose(result[2], torch.full((4,), 6.0))

    def test_tier_composes_with_row(self):
        hidden = torch.zeros(2, 4)
        table = torch.zeros(6, 4)
        table[2] = torch.full((4,), 1.0)  # operator decode row
        index = torch.full((2,), 2, dtype=torch.long)
        scales = torch.ones(6)
        dvec = torch.full((4,), 5.0)
        tscales = torch.tensor([1.0, 1.0])
        result = _apply_full(hidden, table, index, scales, dvec, tscales)
        # row (1.0) + tier (5.0) = 6.0
        assert torch.allclose(result, torch.full((2, 4), 6.0))


def _eager_args(
    hidden: torch.Tensor,
    table: torch.Tensor,
    index: torch.Tensor,
    **overrides,
) -> SteeringOpArgs:
    """Build the 15 ``apply_steering`` tensors with inert defaults.

    Monitors off, scales 1.0, tier 0, ``any_active`` True. ``overrides``
    replace individual fields by name (e.g. ``steering_dynamic_vec=...``).
    """
    n, h = hidden.shape[0], hidden.shape[1]
    rows = table.shape[0]
    args = SteeringOpArgs(
        hidden_states=hidden,
        steering_table=table,
        steering_index=index,
        any_active=torch.tensor([True]),
        steering_scales=torch.ones(rows, dtype=torch.float32),
        steering_dynamic_vec=torch.zeros(h, dtype=torch.float32),
        steering_token_scales=torch.zeros(n, dtype=torch.float32),
        steering_row_gate=torch.ones(n, dtype=torch.float32),
        steering_monitor_probe=torch.zeros(h, dtype=torch.float32),
        steering_monitor_params=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        steering_monitor_active=torch.tensor([False]),
        steering_decode_mask=torch.zeros(n, dtype=torch.float32),
        steering_monitor_probe_table=torch.zeros(1, 1, dtype=torch.float32),
        steering_monitor_row_params=torch.tensor(
            [[-1.0e30, 1.0]], dtype=torch.float32
        ),
        steering_monitor_row_active=torch.tensor([False]),
    )
    return args._replace(**overrides)


class TestEagerDtypeParity:
    """Eager output dtype/cast-order parity with the frozen Triton kernel."""

    @pytest.mark.parametrize("hdtype", [torch.bfloat16, torch.float16])
    def test_output_dtype_matches_hidden_with_fp32_table(self, hdtype):
        """A mismatched fp32 table + low-precision hidden must not promote
        the eager output to fp32 (matches the kernel + ``empty_like`` fake)."""
        hidden = torch.randn(4, 8, dtype=hdtype)
        table = torch.randn(6, 8, dtype=torch.float32)  # fp32 table
        index = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        out = apply_steering(*_eager_args(hidden, table, index))
        assert out.dtype == hdtype

    def test_tier_term_matches_kernel_cast_order(self):
        """The kernel computes the tier as ``dvec.to(h) * tscale.to(h)`` — a
        low-precision multiply. Eager must match that cast order exactly, not
        multiply in fp32 and cast the product. Zeroing hidden + table isolates
        the tier so ``out == tier``."""
        hdtype = torch.bfloat16
        hidden = torch.zeros(1, 4, dtype=hdtype)
        table = torch.zeros(6, 4, dtype=torch.float32)
        index = torch.zeros(1, dtype=torch.long)
        # 0.1/0.3 are not bf16-exact, so bf16(0.1)*bf16(0.3) != bf16(0.1*0.3).
        dvec = torch.tensor([0.1, 0.2, 0.3, 0.7], dtype=torch.float32)
        tscale = torch.tensor([0.3], dtype=torch.float32)
        out = apply_steering(
            *_eager_args(
                hidden,
                table,
                index,
                steering_dynamic_vec=dvec,
                steering_token_scales=tscale,
            )
        )
        tier_ref = dvec.to(hdtype).unsqueeze(0) * tscale.to(hdtype).unsqueeze(-1)
        torch.testing.assert_close(out, tier_ref, atol=0.0, rtol=0.0)


class TestMonitorParamsGuard:
    """A ``(2,)`` monitor-params buffer must fail loudly on both paths — the
    kernel reads slot 2 (``gate_rows``) unconditionally, so a short buffer
    would silently OOB-read on GPU."""

    def test_apply_steering_rejects_two_element_params(self):
        hidden = torch.zeros(2, 4)
        table = torch.zeros(6, 4)
        index = torch.zeros(2, dtype=torch.long)
        with pytest.raises(ValueError):
            apply_steering(
                *_eager_args(
                    hidden,
                    table,
                    index,
                    steering_monitor_params=torch.tensor([0.0, 1.0]),
                )
            )

    def test_steering_monitor_rejects_two_element_params(self):
        with pytest.raises(ValueError):
            steering_monitor(
                torch.zeros(1, 4),
                torch.zeros(4),
                torch.tensor([0.0, 1.0]),  # (2,) — missing gate_rows
                torch.tensor([True]),
                torch.zeros(1),
                torch.zeros(1),
                torch.ones(1),
            )


class TestApplyLayerSteeringHotPath:
    """The no-SAE steering hot path must never import the SAE modules.

    ``apply_layer_steering`` short-circuits SAE feature-surgery on a marker-
    buffer presence check (:func:`_maybe_apply_layer_sae`), so a layer with
    no SAE buffers attached must return without importing ``sae_steering`` —
    keeping the additive-only / disabled forward free of the SAE import cost.
    """

    def test_no_buffers_does_not_import_sae_steering(self, monkeypatch):
        module = nn.Module()
        module.layer_idx = 0
        hidden = torch.randn(2, 4)

        self._forbid_sae_import(monkeypatch)

        out = apply_layer_steering(module, hidden, SteeringHookPoint.POST_BLOCK)
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

        out = apply_layer_steering(module, hidden, SteeringHookPoint.POST_BLOCK)
        assert torch.allclose(out, hidden)

    @staticmethod
    def _forbid_sae_import(monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "vllm.model_executor.layers.sae_steering":
                raise AssertionError("no-SAE hot path imported sae_steering")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
