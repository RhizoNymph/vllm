# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the activation-patching custom ops (CPU eager path).

Activation patching overwrites (``alpha == 1``) or interpolates toward
(``0 < alpha < 1``) the residual at selected token rows with a source vector::

    out[t] = lerp(hs[t], table[idx[t]], alpha[idx[t]])

Slot 0 is the passthrough sentinel (``alpha[0] == 0``). ``post_block`` uses a
two-tensor op that reconstructs the deferred-MLP-add block output
(``residual + hidden``) so replace/lerp lands on the true block output.

Like the steering op tests, these exercise the CPU eager Python implementations
directly (``apply_patch`` / ``apply_patch_block``); the registered
``torch.ops.vllm.*`` ops are CUDA-dispatch-only.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers.patch import (
    PATCH_ALPHA_ATTR,
    PATCH_ANY_ACTIVE_ATTR,
    PATCH_INDEX_ATTR,
    PATCH_TABLE_ATTR,
    apply_patch,
    apply_patch_block,
    get_patch_buffer_slots,
    maybe_apply_patch,
    maybe_apply_patch_block,
    register_patch_buffers,
    set_patch_buffer_slots,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    register_steering_buffers,
)


def _active(flag: bool) -> torch.Tensor:
    return torch.tensor([flag], dtype=torch.bool)


class TestApplyPatchSingleTensor:
    """CPU eager path of the single-tensor lerp op."""

    def test_replace_alpha_one(self):
        """alpha == 1 fully replaces the activation with the source row."""
        n, h = 4, 8
        hidden = torch.randn(n, h)
        table = torch.randn(3, h)
        table[0] = 0.0
        alpha = torch.tensor([0.0, 1.0, 1.0])
        index = torch.tensor([0, 1, 2, 1], dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(True))

        assert torch.allclose(out[0], hidden[0])  # slot 0 passthrough
        assert torch.allclose(out[1], table[1])
        assert torch.allclose(out[2], table[2])
        assert torch.allclose(out[3], table[1])

    def test_interpolation(self):
        """0 < alpha < 1 interpolates between activation and source."""
        n, h = 2, 4
        hidden = torch.ones(n, h)
        table = torch.zeros(3, h)
        table[1] = torch.full((h,), 4.0)
        alpha = torch.tensor([0.0, 0.25, 0.0])
        index = torch.tensor([1, 1], dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(True))
        # 0.75 * 1 + 0.25 * 4 = 1.75
        assert torch.allclose(out, torch.full((n, h), 1.75))

    def test_slot_zero_passthrough(self):
        """Slot 0 leaves the row unchanged even with garbage in table[0]."""
        n, h = 3, 4
        hidden = torch.randn(n, h)
        table = torch.randn(2, h)
        table[0] = 999.0  # must never be read for slot-0 rows
        alpha = torch.tensor([0.0, 1.0])
        index = torch.zeros(n, dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(True))
        assert torch.allclose(out, hidden)

    def test_any_active_false_is_clone(self):
        """any_active False returns a clone of hidden, ignoring the table."""
        n, h = 4, 8
        hidden = torch.randn(n, h)
        table = torch.randn(3, h)  # arbitrary garbage
        alpha = torch.tensor([0.0, 1.0, 1.0])
        index = torch.tensor([1, 2, 1, 2], dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(False))
        assert torch.allclose(out, hidden)
        assert out.data_ptr() != hidden.data_ptr()  # fresh tensor, not alias

    def test_mixed_batch_some_patched(self):
        """Only the indexed rows are patched; others pass through."""
        n, h = 5, 4
        hidden = torch.arange(n * h, dtype=torch.float32).reshape(n, h)
        table = torch.zeros(3, h)
        table[1] = torch.full((h,), 7.0)
        table[2] = torch.full((h,), -3.0)
        alpha = torch.tensor([0.0, 1.0, 1.0])
        index = torch.tensor([0, 1, 0, 2, 0], dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(True))
        assert torch.allclose(out[0], hidden[0])
        assert torch.allclose(out[1], table[1])
        assert torch.allclose(out[2], hidden[2])
        assert torch.allclose(out[3], table[2])
        assert torch.allclose(out[4], hidden[4])

    def test_index_larger_than_batch(self):
        """Index buffer may exceed N; only [:N] is read."""
        n, h = 3, 4
        hidden = torch.ones(n, h)
        table = torch.zeros(3, h)
        table[1] = torch.full((h,), 5.0)
        alpha = torch.tensor([0.0, 1.0, 0.0])
        index = torch.full((100,), 2, dtype=torch.int32)  # out-of-range padding
        index[:n] = 1

        out = apply_patch(hidden, table, index, alpha, _active(True))
        assert torch.allclose(out, torch.full((n, h), 5.0))

    def test_output_shape_dtype(self):
        n, h = 5, 16
        hidden = torch.randn(n, h, dtype=torch.float32)
        table = torch.randn(4, h, dtype=torch.float32)
        alpha = torch.zeros(4)
        index = torch.zeros(n, dtype=torch.int32)

        out = apply_patch(hidden, table, index, alpha, _active(True))
        assert out.shape == hidden.shape
        assert out.dtype == hidden.dtype

    def test_inplace_updates_visible(self):
        """In-place table/index/alpha edits show up on the next call."""
        n, h = 2, 4
        hidden = torch.zeros(n, h)
        table = torch.zeros(3, h)
        alpha = torch.zeros(3)
        index = torch.zeros(n, dtype=torch.int32)

        out1 = apply_patch(hidden, table, index, alpha, _active(True))
        assert torch.allclose(out1, hidden)

        table[1] = torch.full((h,), 9.0)
        alpha[1] = 1.0
        index[0] = 1
        out2 = apply_patch(hidden, table, index, alpha, _active(True))
        assert torch.allclose(out2[0], torch.full((h,), 9.0))
        assert torch.allclose(out2[1], hidden[1])


class TestApplyPatchBlock:
    """CPU eager path of the two-tensor post_block op (the deferred-add trap)."""

    def test_block_output_equals_source_alpha_one(self):
        """The KEY invariant: out_res + hidden == source when alpha == 1.

        This is what distinguishes correct post_block patching from the naive
        (wrong) approach of replacing the bare residual.
        """
        n, h = 4, 8
        hidden = torch.randn(n, h)
        residual = torch.randn(n, h)
        table = torch.randn(3, h)
        table[0] = 0.0
        alpha = torch.tensor([0.0, 1.0, 1.0])
        index = torch.tensor([0, 1, 2, 1], dtype=torch.int32)

        out_res = apply_patch_block(
            hidden, residual, table, index, alpha, _active(True)
        )
        block_out = out_res + hidden  # what the next layer's fused add produces

        assert torch.allclose(block_out[0], residual[0] + hidden[0])  # passthrough
        assert torch.allclose(block_out[1], table[1], atol=1e-6)
        assert torch.allclose(block_out[2], table[2], atol=1e-6)
        assert torch.allclose(block_out[3], table[1], atol=1e-6)

    def test_block_interpolation(self):
        """alpha interpolates the BLOCK OUTPUT, not the bare residual."""
        n, h = 2, 4
        hidden = torch.full((n, h), 2.0)
        residual = torch.full((n, h), 1.0)  # block_out = 3.0
        table = torch.zeros(2, h)
        table[1] = torch.full((h,), 7.0)
        alpha = torch.tensor([0.0, 0.5])
        index = torch.ones(n, dtype=torch.int32)

        out_res = apply_patch_block(
            hidden, residual, table, index, alpha, _active(True)
        )
        block_out = out_res + hidden
        # lerp(3, 7, 0.5) = 5
        assert torch.allclose(block_out, torch.full((n, h), 5.0))

    def test_hidden_states_untouched(self):
        """The op returns a new residual and never mutates hidden_states."""
        n, h = 3, 4
        hidden = torch.randn(n, h)
        hidden_orig = hidden.clone()
        residual = torch.randn(n, h)
        table = torch.randn(2, h)
        alpha = torch.tensor([0.0, 1.0])
        index = torch.ones(n, dtype=torch.int32)

        apply_patch_block(hidden, residual, table, index, alpha, _active(True))
        assert torch.allclose(hidden, hidden_orig)

    def test_passthrough_returns_residual(self):
        n, h = 3, 4
        hidden = torch.randn(n, h)
        residual = torch.randn(n, h)
        table = torch.full((2, h), 999.0)
        alpha = torch.tensor([0.0, 1.0])
        index = torch.zeros(n, dtype=torch.int32)

        out_res = apply_patch_block(
            hidden, residual, table, index, alpha, _active(True)
        )
        assert torch.allclose(out_res, residual)

    def test_any_active_false_is_residual_clone(self):
        n, h = 3, 4
        hidden = torch.randn(n, h)
        residual = torch.randn(n, h)
        table = torch.full((2, h), 999.0)
        alpha = torch.tensor([0.0, 1.0])
        index = torch.ones(n, dtype=torch.int32)

        out_res = apply_patch_block(
            hidden, residual, table, index, alpha, _active(False)
        )
        assert torch.allclose(out_res, residual)
        assert out_res.data_ptr() != residual.data_ptr()


class TestRegisterPatchBuffers:
    def test_buffer_shapes_and_alpha_invariant(self):
        hidden_size, max_tokens, max_slots = 16, 32, 8
        mod = nn.Module()
        register_patch_buffers(
            mod,
            hidden_size,
            max_patch_tokens=max_tokens,
            max_patch_slots=max_slots,
            dtype=torch.float32,
        )
        for hp in HOOK_POINT_TABLE_ATTR:
            table = getattr(mod, PATCH_TABLE_ATTR[hp])
            alpha = getattr(mod, PATCH_ALPHA_ATTR[hp])
            index = getattr(mod, PATCH_INDEX_ATTR[hp])
            flag = getattr(mod, PATCH_ANY_ACTIVE_ATTR[hp])
            assert table.shape == (max_slots, hidden_size)
            assert alpha.shape == (max_slots,)
            assert alpha.dtype == torch.float32
            assert index.shape == (max_tokens,)
            assert index.dtype == torch.int32
            assert flag.shape == (1,)
            assert flag.dtype == torch.bool
            # alpha[0] passthrough invariant
            assert float(alpha[0]) == 0.0

    def test_disabled_is_noop(self):
        mod = nn.Module()
        register_patch_buffers(
            mod, 16, max_patch_tokens=32, max_patch_slots=0, dtype=torch.float32
        )
        for hp in HOOK_POINT_TABLE_ATTR:
            assert not hasattr(mod, PATCH_TABLE_ATTR[hp])

    def test_maybe_apply_patch_short_circuits_without_buffers(self):
        """No patch buffers -> maybe_apply_patch returns the same tensor."""
        mod = nn.Module()
        hidden = torch.randn(3, 8)
        out = maybe_apply_patch(mod, hidden, SteeringHookPoint.POST_ATTN)
        assert out is hidden

    def test_maybe_apply_patch_block_short_circuits_without_buffers(self):
        mod = nn.Module()
        hidden = torch.randn(3, 8)
        residual = torch.randn(3, 8)
        out = maybe_apply_patch_block(mod, hidden, residual)
        assert out is residual


class TestGlobalConfigFold:
    """Patch buffers piggyback on register_steering_buffers via the global."""

    def test_patch_buffers_attach_even_with_steering_disabled(self):
        prev = get_patch_buffer_slots()
        try:
            set_patch_buffer_slots(8)
            mod = nn.Module()
            # max_steering_configs == 0 => steering disabled, but patch
            # buffers must still attach (registration runs before the
            # steering early-return).
            register_steering_buffers(
                mod,
                16,
                max_steering_tokens=32,
                max_steering_configs=0,
                dtype=torch.float32,
            )
            # No steering tables...
            for hp in HOOK_POINT_TABLE_ATTR.values():
                assert not hasattr(mod, hp)
            # ...but patch tables are present.
            for hp in HOOK_POINT_TABLE_ATTR:
                assert hasattr(mod, PATCH_TABLE_ATTR[hp])
                assert getattr(mod, PATCH_TABLE_ATTR[hp]).shape == (8, 16)
                assert getattr(mod, PATCH_INDEX_ATTR[hp]).shape == (32,)
        finally:
            set_patch_buffer_slots(prev)

    def test_global_disabled_attaches_nothing(self):
        prev = get_patch_buffer_slots()
        try:
            set_patch_buffer_slots(0)
            mod = nn.Module()
            register_steering_buffers(
                mod,
                16,
                max_steering_tokens=32,
                max_steering_configs=0,
                dtype=torch.float32,
            )
            for hp in HOOK_POINT_TABLE_ATTR:
                assert not hasattr(mod, PATCH_TABLE_ATTR[hp])
        finally:
            set_patch_buffer_slots(prev)
