# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the directional-clamp custom ops.

The registered ``torch.ops.vllm.apply_clamp`` / ``apply_clamp_block`` ops are
CUDA-dispatch-only (like ``apply_steering`` / ``apply_patch``); these tests
call the eager Python functions directly and assert against explicit
reference math.  The eager path is the frozen reference the Triton kernels
must match (fp32 projection accumulate, correction summed in fp32 and cast
once to the hidden dtype).
"""

import pytest
import torch

from vllm.model_executor.layers.clamp import (
    CLAMP_ANY_ACTIVE_ATTR,
    CLAMP_BOUNDS_ATTR,
    CLAMP_DIRS_ATTR,
    CLAMP_GATE_ACTIVE_ATTR,
    CLAMP_STRENGTH_ATTR,
    ClampBlockOpArgs,
    ClampOpArgs,
    apply_clamp,
    apply_clamp_block,
    maybe_register_clamp_buffers,
    register_clamp_buffers,
    set_clamp_buffer_directions,
)
from vllm.model_executor.layers.steering import SteeringHookPoint

ROWS = 6
K = 4
HIDDEN = 8
MAX_TOKENS = 16


def _make_buffers(rows=ROWS, k=K, hidden=HIDDEN, dtype=torch.float32):
    """Build the clamp buffer set at their registration defaults."""
    dirs = torch.zeros(rows, k, hidden, dtype=dtype)
    bounds = torch.empty(rows, k, 2, dtype=torch.float32)
    bounds[..., 0] = -float("inf")
    bounds[..., 1] = float("inf")
    strength = torch.ones(rows, k, dtype=torch.float32)
    index = torch.zeros(MAX_TOKENS, dtype=torch.long)
    active = torch.zeros(1, dtype=torch.bool)
    return dirs, bounds, strength, index, active


def _unit(vec):
    v = torch.as_tensor(vec, dtype=torch.float32)
    return v / v.norm()


class TestApplyClampSemantics:
    def test_inactive_returns_fresh_copy(self):
        dirs, bounds, strength, index, active = _make_buffers()
        hidden = torch.randn(3, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        assert out is not hidden
        torch.testing.assert_close(out, hidden)

    def test_active_output_is_fresh(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        hidden = torch.randn(3, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        assert out is not hidden

    def test_row_zero_is_passthrough(self):
        """Row 0 is the no-steering sentinel; its dirs are all-zero."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        # Poison row 0 bounds: a zero direction must still be a no-op.
        bounds[0, :, 0] = 2.0
        bounds[0, :, 1] = 3.0
        hidden = torch.randn(3, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        torch.testing.assert_close(out, hidden)

    def test_pin_sets_projection_exactly(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[3, 0] = v
        bounds[3, 0, 0] = 5.0
        bounds[3, 0, 1] = 5.0
        index[:2] = 3
        hidden = torch.randn(2, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        proj = out.to(torch.float32) @ v
        torch.testing.assert_close(proj, torch.full((2,), 5.0))
        # Orthogonal complement untouched.
        torch.testing.assert_close(out[:, 1:], hidden[:, 1:])

    def test_range_clamp_only_touches_out_of_bounds(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([0.0, 1.0] + [0.0] * (HIDDEN - 2))
        dirs[2, 0] = v
        bounds[2, 0, 0] = -1.0
        bounds[2, 0, 1] = 1.0
        index[:3] = 2
        hidden = torch.zeros(3, HIDDEN)
        hidden[0, 1] = 0.5  # in bounds
        hidden[1, 1] = 4.0  # above hi
        hidden[2, 1] = -9.0  # below lo
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        proj = out.to(torch.float32) @ v
        torch.testing.assert_close(proj, torch.tensor([0.5, 1.0, -1.0]))
        # The in-bounds token is bit-identical (delta == 0).
        torch.testing.assert_close(out[0], hidden[0])

    def test_one_sided_upper_bound(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0, 1.0] + [0.0] * (HIDDEN - 2))
        dirs[1, 0] = v
        bounds[1, 0, 1] = 2.0  # lo stays -inf
        index[:2] = 1
        hidden = torch.zeros(2, HIDDEN)
        hidden[0] = 10.0 * v  # proj 10 -> clamp to 2
        hidden[1] = -10.0 * v  # proj -10 -> untouched
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        proj = out.to(torch.float32) @ v
        torch.testing.assert_close(proj, torch.tensor([2.0, -10.0]))

    def test_strength_interpolates(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[1, 0] = v
        bounds[1, 0, 0] = 0.0
        bounds[1, 0, 1] = 0.0
        strength[1, 0] = 0.5
        index[:1] = 1
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 8.0
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        # Full clamp would take proj 8 -> 0; strength 0.5 -> 4.
        assert out[0, 0].item() == pytest.approx(4.0)

    def test_multiple_directions_independent(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v0 = _unit([1.0] + [0.0] * (HIDDEN - 1))
        v1 = _unit([0.0, 1.0] + [0.0] * (HIDDEN - 2))
        dirs[4, 0] = v0
        dirs[4, 1] = v1
        bounds[4, 0, 0] = 1.0
        bounds[4, 0, 1] = 1.0
        bounds[4, 1, 0] = -2.0
        bounds[4, 1, 1] = 2.0
        index[:1] = 4
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 7.0
        hidden[0, 1] = -6.0
        hidden[0, 2] = 3.0
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        assert out[0, 0].item() == pytest.approx(1.0)  # pinned
        assert out[0, 1].item() == pytest.approx(-2.0)  # range-clamped
        assert out[0, 2].item() == pytest.approx(3.0)  # orthogonal untouched

    def test_zero_padded_directions_are_noop(self):
        """Unused K slots stay zero and must contribute nothing, even with
        poisoned bounds."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[2, 0] = v
        bounds[2, 0, 0] = 0.0
        bounds[2, 0, 1] = 0.0
        # Slots 1..K-1 keep zero dirs; poison their bounds.
        bounds[2, 1:, 0] = 5.0
        bounds[2, 1:, 1] = 5.0
        index[:1] = 2
        hidden = torch.randn(1, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        expected = hidden.clone()
        expected[0, 0] = 0.0
        torch.testing.assert_close(out, expected)

    def test_mixed_rows_in_batch(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[3, 0] = v
        bounds[3, 0, 0] = 0.0
        bounds[3, 0, 1] = 0.0
        index[0] = 0  # unclamped token
        index[1] = 3  # clamped token
        hidden = torch.ones(2, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        torch.testing.assert_close(out[0], hidden[0])
        assert out[1, 0].item() == pytest.approx(0.0)
        torch.testing.assert_close(out[1, 1:], hidden[1, 1:])

    def test_output_dtype_matches_hidden(self):
        dirs, bounds, strength, index, active = _make_buffers(dtype=torch.bfloat16)
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[1, 0] = v.to(torch.bfloat16)
        bounds[1, 0, 0] = 0.0
        bounds[1, 0, 1] = 0.0
        index[:1] = 1
        hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        assert out.dtype == torch.bfloat16

    def test_empty_batch(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        hidden = torch.randn(0, HIDDEN)
        out = apply_clamp(hidden, dirs, bounds, strength, index, active)
        assert out.shape == (0, HIDDEN)


class TestApplyClampBlock:
    """post_block variant: clamps the true block output
    (residual + hidden) and folds the delta back into residual."""

    def test_inactive_returns_fresh_residual(self):
        dirs, bounds, strength, index, active = _make_buffers()
        hidden = torch.randn(2, HIDDEN)
        residual = torch.randn(2, HIDDEN)
        out = apply_clamp_block(hidden, residual, dirs, bounds, strength, index, active)
        assert out is not residual
        torch.testing.assert_close(out, residual)

    def test_block_invariant_matches_single_tensor_op(self):
        """out_res + hidden == apply_clamp(residual + hidden)."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit(torch.randn(HIDDEN))
        dirs[2, 0] = v
        bounds[2, 0, 0] = -0.5
        bounds[2, 0, 1] = 0.5
        strength[2, 0] = 0.75
        index[:3] = 2
        hidden = torch.randn(3, HIDDEN)
        residual = torch.randn(3, HIDDEN)
        out_res = apply_clamp_block(
            hidden, residual, dirs, bounds, strength, index, active
        )
        block_out = apply_clamp(
            residual + hidden, dirs, bounds, strength, index, active
        )
        torch.testing.assert_close(out_res + hidden, block_out)

    def test_clamps_block_output_not_bare_residual(self):
        """The projection is measured on residual + hidden, not residual."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[1, 0] = v
        bounds[1, 0, 0] = 0.0
        bounds[1, 0, 1] = 0.0
        index[:1] = 1
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 3.0
        residual = torch.zeros(1, HIDDEN)
        residual[0, 0] = 4.0
        out_res = apply_clamp_block(
            hidden, residual, dirs, bounds, strength, index, active
        )
        # block output proj = 7 -> pinned to 0 -> out_res proj = -hidden proj.
        assert (out_res[0, 0] + hidden[0, 0]).item() == pytest.approx(0.0)


class TestClampBufferRegistration:
    def test_register_shapes_and_defaults(self):
        module = torch.nn.Module()
        register_clamp_buffers(
            module, HIDDEN, num_rows=ROWS, max_directions=K, dtype=torch.bfloat16
        )
        for hp in SteeringHookPoint:
            dirs = getattr(module, CLAMP_DIRS_ATTR[hp])
            bounds = getattr(module, CLAMP_BOUNDS_ATTR[hp])
            strength = getattr(module, CLAMP_STRENGTH_ATTR[hp])
            active = getattr(module, CLAMP_ANY_ACTIVE_ATTR[hp])
            assert dirs.shape == (ROWS, K, HIDDEN)
            assert dirs.dtype == torch.bfloat16
            assert torch.all(dirs == 0)
            assert bounds.shape == (ROWS, K, 2)
            assert torch.all(bounds[..., 0] == -float("inf"))
            assert torch.all(bounds[..., 1] == float("inf"))
            assert strength.shape == (ROWS, K)
            assert torch.all(strength == 1.0)
            assert active.shape == (1,)
            assert not bool(active.item())

    def test_maybe_register_noop_when_k_zero(self):
        set_clamp_buffer_directions(0)
        try:
            module = torch.nn.Module()
            maybe_register_clamp_buffers(module, HIDDEN, num_rows=ROWS)
            for hp in SteeringHookPoint:
                assert not hasattr(module, CLAMP_DIRS_ATTR[hp])
        finally:
            set_clamp_buffer_directions(0)

    def test_maybe_register_uses_global_without_config(self):
        set_clamp_buffer_directions(2)
        try:
            module = torch.nn.Module()
            maybe_register_clamp_buffers(
                module, HIDDEN, num_rows=ROWS, dtype=torch.float32
            )
            hp = SteeringHookPoint.POST_BLOCK
            assert getattr(module, CLAMP_DIRS_ATTR[hp]).shape == (ROWS, 2, HIDDEN)
        finally:
            set_clamp_buffer_directions(0)


class TestClampOpSchemas:
    def test_apply_clamp_op_schema_arity(self):
        import vllm.model_executor.layers.clamp  # noqa: F401  registers ops

        schema = torch.ops.vllm.apply_clamp.default._schema
        assert len(schema.arguments) == len(ClampOpArgs._fields)

    def test_apply_clamp_op_args_fields_match_schema_order(self):
        import vllm.model_executor.layers.clamp  # noqa: F401

        schema = torch.ops.vllm.apply_clamp.default._schema
        schema_names = tuple(arg.name for arg in schema.arguments)
        assert ClampOpArgs._fields == schema_names

    def test_apply_clamp_block_op_schema_arity(self):
        import vllm.model_executor.layers.clamp  # noqa: F401

        schema = torch.ops.vllm.apply_clamp_block.default._schema
        assert len(schema.arguments) == len(ClampBlockOpArgs._fields)

    def test_apply_clamp_block_op_args_fields_match_schema_order(self):
        import vllm.model_executor.layers.clamp  # noqa: F401

        schema = torch.ops.vllm.apply_clamp_block.default._schema
        schema_names = tuple(arg.name for arg in schema.arguments)
        assert ClampBlockOpArgs._fields == schema_names


class TestClampEmission:
    """apply_layer_steering / apply_block_steering emit the clamp AFTER the
    additive steering op when clamp buffers are present."""

    def _steered_layer(self, k=K):
        from vllm.model_executor.layers.steering import (
            register_steering_buffers,
        )

        set_clamp_buffer_directions(k)
        try:
            module = torch.nn.Module()
            module.layer_idx = 0
            register_steering_buffers(
                module,
                HIDDEN,
                max_steering_tokens=MAX_TOKENS,
                max_steering_configs=ROWS - 3,
            )
        finally:
            set_clamp_buffer_directions(0)
        return module

    def test_clamp_applied_after_additive_steering(self):
        from vllm.model_executor.layers.steering import (
            HOOK_POINT_ANY_ACTIVE_ATTR,
            HOOK_POINT_TABLE_ATTR,
            apply_layer_steering,
        )

        module = self._steered_layer()
        hp = SteeringHookPoint.PRE_ATTN
        # Additive vector pushes the projection along e0 up by 10.
        table = getattr(module, HOOK_POINT_TABLE_ATTR[hp])
        table[3, 0] = 10.0
        getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hp]).fill_(True)
        module.steering_index[:1] = 3
        # Clamp pins the projection along e0 to 1.0.
        dirs = getattr(module, CLAMP_DIRS_ATTR[hp])
        bounds = getattr(module, CLAMP_BOUNDS_ATTR[hp])
        dirs[3, 0, 0] = 1.0
        bounds[3, 0, 0] = 1.0
        bounds[3, 0, 1] = 1.0
        getattr(module, CLAMP_ANY_ACTIVE_ATTR[hp]).fill_(True)

        hidden = torch.zeros(1, HIDDEN)
        out = apply_layer_steering(module, hidden, hp)
        # If the clamp ran first, the add would land after it and the
        # projection would be 11; clamp-last pins it to 1.
        assert out[0, 0].item() == pytest.approx(1.0)

    def test_block_clamp_applied_after_block_steering(self):
        from vllm.model_executor.layers.steering import (
            HOOK_POINT_ANY_ACTIVE_ATTR,
            HOOK_POINT_TABLE_ATTR,
            apply_block_steering,
        )

        module = self._steered_layer()
        hp = SteeringHookPoint.POST_BLOCK
        table = getattr(module, HOOK_POINT_TABLE_ATTR[hp])
        table[3, 0] = 10.0
        getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hp]).fill_(True)
        module.steering_index[:1] = 3
        dirs = getattr(module, CLAMP_DIRS_ATTR[hp])
        bounds = getattr(module, CLAMP_BOUNDS_ATTR[hp])
        dirs[3, 0, 0] = 1.0
        bounds[3, 0, 0] = 1.0
        bounds[3, 0, 1] = 1.0
        getattr(module, CLAMP_ANY_ACTIVE_ATTR[hp]).fill_(True)

        hidden = torch.zeros(1, HIDDEN)
        residual = torch.zeros(1, HIDDEN)
        out_hidden, out_res = apply_block_steering(module, hidden, residual)
        # Block output (residual + hidden) projection pinned to 1.
        assert (out_res[0, 0] + out_hidden[0, 0]).item() == pytest.approx(1.0)

    def test_no_clamp_buffers_short_circuits(self):
        from vllm.model_executor.layers.steering import (
            apply_layer_steering,
            register_steering_buffers,
        )

        module = torch.nn.Module()
        module.layer_idx = 0
        register_steering_buffers(
            module,
            HIDDEN,
            max_steering_tokens=MAX_TOKENS,
            max_steering_configs=ROWS - 3,
        )
        hp = SteeringHookPoint.PRE_ATTN
        assert not hasattr(module, CLAMP_DIRS_ATTR[hp])
        hidden = torch.randn(1, HIDDEN)
        out = apply_layer_steering(module, hidden, hp)
        torch.testing.assert_close(out, hidden)


def _gate_buffers(n_tokens=MAX_TOKENS):
    """Shared per-token row gate (default 1.0) + its activity flag."""
    row_gate = torch.ones(n_tokens, dtype=torch.float32)
    gate_active = torch.zeros(1, dtype=torch.bool)
    return row_gate, gate_active


class TestClampGating:
    """Declarative gates modulate clamp strength row-level:
    effective strength = clamp_strength * row_gate[token] when ``gate_active``.
    """

    def _pinned_to_zero(self):
        """Row 1 pins the e0 projection to 0 at full strength; token 0 -> row 1."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[1, 0] = v
        bounds[1, 0, 0] = 0.0
        bounds[1, 0, 1] = 0.0
        index[:1] = 1
        return dirs, bounds, strength, index, active

    def test_gate_scales_correction_linearly(self):
        dirs, bounds, strength, index, active = self._pinned_to_zero()
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 0.5
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 8.0
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        # Full clamp: proj 8 -> 0 (delta -8). Gate 0.5 -> delta -4 -> 4.0.
        assert out[0, 0].item() == pytest.approx(4.0)

    def test_gate_zero_leaves_token_untouched(self):
        dirs, bounds, strength, index, active = self._pinned_to_zero()
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 0.0
        hidden = torch.randn(1, HIDDEN)
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        # gate * strength == 0 -> delta exactly 0 -> bit-identical passthrough.
        torch.testing.assert_close(out, hidden)

    def test_gate_one_matches_ungated(self):
        dirs, bounds, strength, index, active = self._pinned_to_zero()
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)  # row_gate all 1.0
        hidden = torch.randn(2, HIDDEN)
        index[:2] = 1
        gated = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        ungated = apply_clamp(hidden, dirs, bounds, strength, index, active)
        torch.testing.assert_close(gated, ungated)

    def test_gate_inactive_ignores_row_gate(self):
        dirs, bounds, strength, index, active = self._pinned_to_zero()
        row_gate, gate_active = _gate_buffers()
        # gate_active False: the row_gate value is ignored (ungated).
        row_gate[0] = 0.0
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 8.0
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        assert out[0, 0].item() == pytest.approx(0.0)  # full clamp, gate ignored

    def test_gate_row_zero_passthrough_regardless(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        # Poison row 0 bounds; token 0 -> row 0 sentinel (dirs zero).
        bounds[0, :, 0] = 2.0
        bounds[0, :, 1] = 3.0
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 0.0
        hidden = torch.randn(1, HIDDEN)
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        torch.testing.assert_close(out, hidden)

    def test_gate_is_row_level_all_k(self):
        """The token's gate scales every K entry of its row uniformly."""
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v0 = _unit([1.0] + [0.0] * (HIDDEN - 1))
        v1 = _unit([0.0, 1.0] + [0.0] * (HIDDEN - 2))
        dirs[4, 0] = v0
        dirs[4, 1] = v1
        bounds[4, 0, 0] = 0.0
        bounds[4, 0, 1] = 0.0
        bounds[4, 1, 0] = 0.0
        bounds[4, 1, 1] = 0.0
        index[:1] = 4
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 0.25
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 4.0
        hidden[0, 1] = 8.0
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        # Both entries pinned to 0 at gate 0.25: 4 -> 3.0, 8 -> 6.0.
        assert out[0, 0].item() == pytest.approx(3.0)
        assert out[0, 1].item() == pytest.approx(6.0)

    def test_mixed_gates_per_token(self):
        dirs, bounds, strength, index, active = self._pinned_to_zero()
        index[:2] = 1
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 1.0
        row_gate[1] = 0.0
        hidden = torch.zeros(2, HIDDEN)
        hidden[:, 0] = 8.0
        out = apply_clamp(
            hidden, dirs, bounds, strength, index, active, row_gate, gate_active
        )
        assert out[0, 0].item() == pytest.approx(0.0)  # fully clamped
        assert out[1, 0].item() == pytest.approx(8.0)  # gate 0 -> untouched

    def test_block_gate_scales_correction(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit([1.0] + [0.0] * (HIDDEN - 1))
        dirs[1, 0] = v
        bounds[1, 0, 0] = 0.0
        bounds[1, 0, 1] = 0.0
        index[:1] = 1
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[0] = 0.5
        hidden = torch.zeros(1, HIDDEN)
        hidden[0, 0] = 3.0
        residual = torch.zeros(1, HIDDEN)
        residual[0, 0] = 5.0
        out_res = apply_clamp_block(
            hidden,
            residual,
            dirs,
            bounds,
            strength,
            index,
            active,
            row_gate,
            gate_active,
        )
        # block proj 8 -> full clamp delta -8; gate 0.5 -> -4 -> out_res proj 1.
        assert (out_res[0, 0] + hidden[0, 0]).item() == pytest.approx(4.0)

    def test_block_gate_matches_single_tensor_op(self):
        dirs, bounds, strength, index, active = _make_buffers()
        active.fill_(True)
        v = _unit(torch.randn(HIDDEN))
        dirs[2, 0] = v
        bounds[2, 0, 0] = -0.5
        bounds[2, 0, 1] = 0.5
        index[:3] = 2
        row_gate, gate_active = _gate_buffers()
        gate_active.fill_(True)
        row_gate[:3] = torch.tensor([1.0, 0.5, 0.0])
        hidden = torch.randn(3, HIDDEN)
        residual = torch.randn(3, HIDDEN)
        out_res = apply_clamp_block(
            hidden,
            residual,
            dirs,
            bounds,
            strength,
            index,
            active,
            row_gate,
            gate_active,
        )
        block_out = apply_clamp(
            residual + hidden,
            dirs,
            bounds,
            strength,
            index,
            active,
            row_gate,
            gate_active,
        )
        torch.testing.assert_close(out_res + hidden, block_out)


class TestClampGateSchema:
    """Sibling of TestClampOpSchemas pinning the NEW gate args (the existing
    arity-lock tests compare _fields to the schema dynamically and stay green;
    these pin the gate fields' presence and position)."""

    def test_apply_clamp_gate_args_last(self):
        import vllm.model_executor.layers.clamp  # noqa: F401

        schema = torch.ops.vllm.apply_clamp.default._schema
        names = [arg.name for arg in schema.arguments]
        assert names[-2:] == ["steering_row_gate", "gate_active"]

    def test_apply_clamp_block_gate_args_last(self):
        import vllm.model_executor.layers.clamp  # noqa: F401

        schema = torch.ops.vllm.apply_clamp_block.default._schema
        names = [arg.name for arg in schema.arguments]
        assert names[-2:] == ["steering_row_gate", "gate_active"]

    def test_opargs_end_with_gate_fields(self):
        assert ClampOpArgs._fields[-2:] == ("steering_row_gate", "gate_active")
        assert ClampBlockOpArgs._fields[-2:] == ("steering_row_gate", "gate_active")

    def test_register_attaches_gate_active_flag(self):
        module = torch.nn.Module()
        register_clamp_buffers(
            module, HIDDEN, num_rows=ROWS, max_directions=K, dtype=torch.float32
        )
        for hp in SteeringHookPoint:
            flag = getattr(module, CLAMP_GATE_ACTIVE_ATTR[hp])
            assert flag.shape == (1,)
            assert flag.dtype == torch.bool
            assert not bool(flag.item())
