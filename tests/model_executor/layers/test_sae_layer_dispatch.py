# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``apply_layer_sae_delta`` (the layer-hook dispatch shim)
and the per-layer SAE clamp-table populator.

The shim is the lifecycle bridge between buffer state and the math
primitive: pull per-token clamps from each slot's table via
``sae_index``, hand them to ``apply_sae_delta``.  The populator is the
inverse: project ``SAEClampSpec`` row content into the per-(layer,
hook, slot) buffers so the shim can read them on the next forward.
Multiple SAE modules may share a site — one op call is chained per
slot in registration order.

These tests exercise both in isolation, without a worker mixin or a
real model.  Buffers are constructed directly so the data flow is
visible end-to-end.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
    CLAMP_KIND_NONE,
    SAESlotState,
    apply_layer_sae_delta,
    get_sae_slot_state,
    populate_sae_clamp_table,
    register_sae_buffers,
    register_sae_index_buffer,
)
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_layer_steering,
    register_steering_buffers,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager


def _layer_with_sae(
    *,
    hook: SteeringHookPoint,
    module_name: str = "g",
    n_clamp: int = 2,
    hidden_size: int = 4,
    max_sae_configs: int = 2,
    max_tokens: int = 16,
    activation: SAEActivation = SAEActivation.RELU,
    activation_params: dict | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    m = nn.Module()
    register_sae_buffers(
        m,
        hook_point=hook,
        module_name=module_name,
        activation=activation,
        activation_params=activation_params or {},
        n_clamp=n_clamp,
        hidden_size=hidden_size,
        max_sae_configs=max_sae_configs,
        dtype=dtype,
    )
    register_sae_index_buffer(m, max_tokens=max_tokens)
    return m


def _slot(
    m: nn.Module,
    hook: SteeringHookPoint = SteeringHookPoint.POST_BLOCK,
    module_name: str = "g",
) -> SAESlotState:
    state = get_sae_slot_state(m, hook, module_name)
    assert state is not None, f"no slot for {module_name!r} at {hook.value!r}"
    return state


class TestApplyLayerSaeDeltaShortCircuits:
    """Disabled / no-op paths short-circuit without changing the residual."""

    def test_no_buffers_returns_input_unchanged(self):
        m = nn.Module()
        h = torch.randn(3, 4)
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)
        # ``is`` would be too strict (custom op semantics may clone);
        # equality with no copy is the contract.
        assert torch.equal(out, h)

    def test_all_tokens_at_row_zero_returns_input_unchanged(self):
        m = _layer_with_sae(hook=SteeringHookPoint.POST_BLOCK)
        # sae_index already zeros — every token reads row 0 (no-op).
        h = torch.randn(3, 4)
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)
        # Encoder weights are zeros (default), clamp_kind row 0 is NONE,
        # so delta is exactly zero.
        assert torch.allclose(out, h)

    def test_inactive_flag_suppresses_stale_rows(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=2,
            max_sae_configs=1,
            max_tokens=4,
        )
        state = _slot(m)
        state.encoder_weight.copy_(torch.tensor([[1.0, 0.0]]))
        state.decoder_weight.copy_(torch.tensor([[0.0, 1.0]]))
        state.clamp_kind[1] = torch.tensor([CLAMP_KIND_ABSOLUTE], dtype=torch.int8)
        state.clamp_value[1] = torch.tensor([9.0])
        m.sae_index[0] = 1

        h = torch.tensor([[2.0, 0.0]])
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)

        assert torch.allclose(out, h)

    def test_other_hook_point_buffers_unaffected(self):
        m = _layer_with_sae(hook=SteeringHookPoint.POST_BLOCK)
        h = torch.randn(2, 4)
        # POST_ATTN has no SAE buffers; shim must short-circuit.
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_ATTN)
        assert torch.equal(out, h)


class TestApplyLayerSaeDeltaDispatch:
    """End-to-end: buffer state → math primitive → updated residual."""

    def test_absolute_clamp_via_buffers(self):
        # n_clamp=1, hidden=2, two rows (no-op + one active).
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=2,
            max_sae_configs=1,
            max_tokens=4,
        )
        state = _slot(m)
        # Encoder picks h[0]; decoder writes into h[1].
        state.encoder_weight.copy_(torch.tensor([[1.0, 0.0]]))
        state.decoder_weight.copy_(torch.tensor([[0.0, 1.0]]))
        state.encoder_bias.zero_()
        # Row 1: feature 0 absolute-clamped to 7.0.
        state.clamp_kind[1] = torch.tensor([CLAMP_KIND_ABSOLUTE], dtype=torch.int8)
        state.clamp_value[1] = torch.tensor([7.0])
        state.clamp_only_if_active[1] = torch.tensor([False])
        state.any_active.fill_(True)
        # Token 0: routed to row 1 (clamp active); token 1: row 0 (no-op).
        m.sae_index[0] = 1
        m.sae_index[1] = 0

        h = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)
        # Token 0: f = ReLU(2.0)=2.0; delta = 7.0-2.0=5.0 along [0,1].
        # Token 1: row 0 → no clamp, residual unchanged.
        expected = torch.tensor([[2.0, 5.0], [3.0, 0.0]])
        assert torch.allclose(out, expected)

    def test_per_token_routing_independent(self):
        # Two rows clamping different feature indices.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=3,
            max_sae_configs=2,
            max_tokens=4,
        )
        state = _slot(m)
        # Encoder picks h[0] for feature 0, h[1] for feature 1.
        state.encoder_weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        # Decoder writes to h[2] for feature 0 (additive=+1), to h[2] for
        # feature 1 (additive=+10).
        state.decoder_weight.copy_(
            torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        )
        # Row 1: feature 0 additive +1 (independent of f).
        state.clamp_kind[1] = torch.tensor(
            [CLAMP_KIND_ADDITIVE, CLAMP_KIND_NONE], dtype=torch.int8
        )
        state.clamp_value[1] = torch.tensor([1.0, 0.0])
        # Row 2: feature 1 additive +10 (independent of f).
        state.clamp_kind[2] = torch.tensor(
            [CLAMP_KIND_NONE, CLAMP_KIND_ADDITIVE], dtype=torch.int8
        )
        state.clamp_value[2] = torch.tensor([0.0, 10.0])
        state.any_active.fill_(True)

        m.sae_index[0] = 1  # token 0 -> row 1 (delta +1 in h[2])
        m.sae_index[1] = 2  # token 1 -> row 2 (delta +10 in h[2])
        m.sae_index[2] = 0  # token 2 -> row 0 (no delta)

        h = torch.zeros(3, 3)
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)
        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        )
        assert torch.allclose(out, expected)

    def test_torch_compile_observes_late_sae_buffer_attach(self):
        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer_idx = 0
                register_steering_buffers(
                    self,
                    hidden_size=2,
                    max_steering_tokens=4,
                    max_steering_configs=1,
                    dtype=torch.float32,
                )

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return apply_layer_steering(
                    self, hidden_states, SteeringHookPoint.POST_BLOCK
                )

        layer = Layer()
        for hook in SteeringHookPoint:
            getattr(layer, f"steering_table_{hook.value}_any_active").zero_()
        compiled = torch.compile(layer, backend="eager")

        h = torch.tensor([[2.0, 0.0]])
        torch.testing.assert_close(compiled(h), h)

        register_sae_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=1,
            hidden_size=2,
            max_sae_configs=1,
            dtype=torch.float32,
        )
        register_sae_index_buffer(layer, max_tokens=4)
        state = _slot(layer)
        state.encoder_weight.copy_(torch.tensor([[1.0, 0.0]]))
        state.decoder_weight.copy_(torch.tensor([[0.0, 1.0]]))
        state.clamp_kind[1] = torch.tensor([CLAMP_KIND_ABSOLUTE], dtype=torch.int8)
        state.clamp_value[1] = torch.tensor([7.0])
        state.any_active.fill_(True)
        layer.sae_index[0] = 1

        torch.testing.assert_close(compiled(h), torch.tensor([[2.0, 5.0]]))


def _register_second_module(
    m: nn.Module,
    *,
    module_name: str,
    n_clamp: int,
    hidden_size: int,
    max_sae_configs: int = 2,
) -> None:
    register_sae_buffers(
        m,
        hook_point=SteeringHookPoint.POST_BLOCK,
        module_name=module_name,
        activation=SAEActivation.RELU,
        activation_params={},
        n_clamp=n_clamp,
        hidden_size=hidden_size,
        max_sae_configs=max_sae_configs,
        dtype=torch.float32,
    )


def _configure_absolute_clamp(
    state: SAESlotState,
    *,
    enc_row: int,
    dec_col: int,
    value: float,
    hidden_size: int,
    row: int = 1,
    feature_pos: int = 0,
) -> None:
    """Encoder reads h[enc_row]; decoder writes h[dec_col]; row 1 clamps."""
    enc = torch.zeros_like(state.encoder_weight)
    enc[feature_pos, enc_row] = 1.0
    state.encoder_weight.copy_(enc)
    dec = torch.zeros_like(state.decoder_weight)
    dec[feature_pos, dec_col] = 1.0
    state.decoder_weight.copy_(dec)
    state.clamp_kind[row, feature_pos] = CLAMP_KIND_ABSOLUTE
    state.clamp_value[row, feature_pos] = value
    state.any_active.fill_(True)


class TestMultiModuleComposition:
    """Two delta modules sharing a site compose in registration order."""

    def _two_module_layer(self, *, hidden_size: int = 4) -> nn.Module:
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="a",
            n_clamp=1,
            hidden_size=hidden_size,
            max_sae_configs=2,
            max_tokens=8,
        )
        _register_second_module(
            m, module_name="b", n_clamp=1, hidden_size=hidden_size
        )
        return m

    def test_composition_equals_sequential_eager_application(self):
        # A reads h[0] and writes h[1]; B reads h[1] and writes h[2].
        # Because B's encoder sees A's decoder output, the composed
        # result is order-dependent — this pins registration order.
        m = self._two_module_layer()
        state_a = _slot(m, module_name="a")
        state_b = _slot(m, module_name="b")
        _configure_absolute_clamp(
            state_a, enc_row=0, dec_col=1, value=7.0, hidden_size=4
        )
        _configure_absolute_clamp(
            state_b, enc_row=1, dec_col=2, value=11.0, hidden_size=4
        )
        m.sae_index[0] = 1

        h = torch.tensor([[2.0, 1.0, 0.0, 0.0]])
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)

        # Sequential eager reference in registration order (A then B)
        # via the public math primitive.
        from vllm.model_executor.layers.sae_steering import apply_sae_delta

        def _eager(hs: torch.Tensor, state: SAESlotState) -> torch.Tensor:
            row = 1
            return apply_sae_delta(
                hs,
                state.encoder_weight,
                state.encoder_bias,
                state.decoder_weight,
                SAEActivation.RELU,
                {},
                state.clamp_kind[row : row + 1],
                state.clamp_value[row : row + 1],
                state.clamp_only_if_active[row : row + 1],
            )

        expected = _eager(_eager(h, state_a), state_b)
        assert torch.allclose(out, expected)
        # Concretely: A clamps f_a (=h[0]=2) to 7 → h[1] += 5 → h[1]=6.
        # B then sees h[1]=6, clamps f_b to 11 → h[2] += 5.
        assert torch.allclose(out, torch.tensor([[2.0, 6.0, 5.0, 0.0]]))
        # Order matters: B-then-A differs (B would see h[1]=1, delta 10).
        swapped = _eager(_eager(h, state_b), state_a)
        assert not torch.allclose(out, swapped)

    def test_unclamped_token_bit_identical_through_both(self):
        m = self._two_module_layer()
        state_a = _slot(m, module_name="a")
        state_b = _slot(m, module_name="b")
        _configure_absolute_clamp(
            state_a, enc_row=0, dec_col=1, value=7.0, hidden_size=4
        )
        _configure_absolute_clamp(
            state_b, enc_row=1, dec_col=2, value=11.0, hidden_size=4
        )
        # Token 0 clamped, token 1 routed to the no-op sentinel row.
        m.sae_index[0] = 1
        m.sae_index[1] = 0

        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)
        # The no-clamp token passes through both slots bit-identically.
        assert torch.equal(out[1], h[1])
        # The clamped token was actually modified (sanity).
        assert not torch.equal(out[0], h[0])

    def test_idle_module_composes_as_identity(self):
        # A idle (any_active False), B active: output equals B alone.
        m = self._two_module_layer()
        state_a = _slot(m, module_name="a")
        state_b = _slot(m, module_name="b")
        _configure_absolute_clamp(
            state_a, enc_row=0, dec_col=1, value=7.0, hidden_size=4
        )
        _configure_absolute_clamp(
            state_b, enc_row=2, dec_col=3, value=11.0, hidden_size=4
        )
        state_a.any_active.zero_()  # A idle
        m.sae_index[0] = 1

        h = torch.tensor([[2.0, 1.0, 3.0, 0.0]])
        out = apply_layer_sae_delta(m, h, SteeringHookPoint.POST_BLOCK)

        # Reference: a layer with only B registered, same content.
        m_b_only = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="b",
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            max_tokens=8,
        )
        state_b_only = _slot(m_b_only, module_name="b")
        _configure_absolute_clamp(
            state_b_only, enc_row=2, dec_col=3, value=11.0, hidden_size=4
        )
        m_b_only.sae_index[0] = 1
        expected = apply_layer_sae_delta(m_b_only, h, SteeringHookPoint.POST_BLOCK)
        assert torch.allclose(out, expected)
        # f_b = ReLU(3) = 3; delta = 11 - 3 = 8 into h[3].
        assert torch.allclose(out, torch.tensor([[2.0, 1.0, 3.0, 8.0]]))


class TestPopulateSaeClampTable:
    """Manager rows projected into per-(layer, hook, slot) buffers."""

    def test_row_zero_remains_zero(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        # No registered configs — populate must leave row 0 zero.
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0, 1),
            worker_phase="prefill",
        )
        state = _slot(m)
        assert torch.equal(state.clamp_kind[0], torch.zeros(2, dtype=torch.int8))
        assert not state.any_active.item()

    def test_active_row_writes_clamp_for_matching_module(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=7.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "prefill")
        # clampable_features=(2, 5) means feature_idx=5 maps to position 1.
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(2, 5),
            worker_phase="prefill",
            layer_idx=20,
        )
        state = _slot(m)
        # The per-request row, position 1 (feature_idx 5), gets the clamp.
        assert state.clamp_kind[row, 1].item() == CLAMP_KIND_ABSOLUTE
        assert state.clamp_value[row, 1].item() == 7.0
        # Position 0 (feature_idx 2) stays NONE.
        assert state.clamp_kind[row, 0].item() == CLAMP_KIND_NONE
        assert state.any_active.item()

    def test_populate_for_module_without_slot_is_noop(self):
        # This buffer site holds a slot only for 'other_module'; a
        # populate call naming module 'g' must be a no-op (no slot).
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="other_module",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",  # spec targets module 'g'
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=7.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "prefill")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0, 5),
            worker_phase="prefill",
            layer_idx=20,
        )
        state = _slot(m, module_name="other_module")
        assert torch.equal(state.clamp_kind[row], torch.zeros(2, dtype=torch.int8))
        assert not state.any_active.item()

    def test_active_row_zeroed_for_non_matching_module(self):
        # The slot belongs to 'other_module'; populating that slot with
        # a manager holding only 'g' specs zeroes the row at this slot.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="other_module",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",  # spec targets module 'g'
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=7.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "prefill")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="other_module",
            clampable_features=(0, 5),
            worker_phase="prefill",
            layer_idx=20,
        )
        state = _slot(m, module_name="other_module")
        assert torch.equal(state.clamp_kind[row], torch.zeros(2, dtype=torch.int8))
        assert not state.any_active.item()

    def test_populator_isolation_between_shared_site_slots(self):
        # Two modules share the site; module A's rows must land only in
        # A's slot tables, never in B's.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="a",
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            max_tokens=8,
        )
        _register_second_module(m, module_name="b", n_clamp=1, hidden_size=4)
        manager = SAEClampManager(max_sae_configs=2)
        spec_a = SAEClampSpec(
            module_name="a",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=7.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec_a,), "prefill")
        for name in ("a", "b"):
            populate_sae_clamp_table(
                manager=manager,
                module=m,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name=name,
                clampable_features=(0,),
                layer_idx=20,
            )
        state_a = _slot(m, module_name="a")
        state_b = _slot(m, module_name="b")
        assert state_a.clamp_kind[row, 0].item() == CLAMP_KIND_ABSOLUTE
        assert state_a.any_active.item()
        # B's tables never see A's rows.
        assert torch.equal(
            state_b.clamp_kind, torch.zeros_like(state_b.clamp_kind)
        )
        assert not state_b.any_active.item()

    def test_phase_filter_decode_only(self):
        manager = SAEClampManager(max_sae_configs=2)
        # A decode-only spec has no prefill table content.  The manager
        # should reject this direct invalid registration instead of
        # allocating a row that would only ever be zeroed.  Production
        # callers phase-filter before registration and never make this
        # call for a phase-empty spec.
        spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        with pytest.raises(ValueError, match="do not apply"):
            manager.register_clamp_spec(123, (spec,), "prefill")

    def test_phase_both_applies_in_either_worker_phase(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",
            phase="both",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "decode")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0,),
            worker_phase="decode",
            layer_idx=20,
        )
        state = _slot(m)
        assert state.clamp_kind[row, 0].item() == CLAMP_KIND_ABSOLUTE
        assert state.clamp_value[row, 0].item() == 9.0
        assert state.any_active.item()

    def test_phase_filtered_populate_preserves_other_phase_any_active(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        decode_spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (decode_spec,), "decode")

        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0,),
            worker_phase="decode",
            layer_idx=20,
        )
        state = _slot(m)
        assert state.any_active.item()

        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0,),
            worker_phase="prefill",
            layer_idx=20,
        )

        assert state.clamp_kind[row, 0].item() == CLAMP_KIND_ABSOLUTE
        assert state.clamp_value[row, 0].item() == 9.0
        assert state.any_active.item()

    def test_layer_with_no_clamps_in_spec_zeroed(self):
        # Spec covers layer 20; site is layer 21 — the per-request row
        # must be zero at the layer-21 site.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "prefill")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0,),
            worker_phase="prefill",
            layer_idx=21,  # spec targets 20, this site is 21
        )
        state = _slot(m)
        assert torch.equal(state.clamp_kind[row], torch.zeros(1, dtype=torch.int8))

    def test_only_if_active_flag_propagates(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (
                        SAEClampEntry(
                            feature_idx=0,
                            kind="additive",
                            value=2.0,
                            only_if_active=True,
                        ),
                    )
                }
            },
        )
        row = manager.register_clamp_spec(123, (spec,), "prefill")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(0,),
            worker_phase="prefill",
            layer_idx=20,
        )
        state = _slot(m)
        assert bool(state.clamp_only_if_active[row, 0].item()) is True

    def test_global_clamp_lands_on_phase_global_rows(self):
        # Global clamp at the manager level → populator writes it
        # into rows 1/2 so tokens with no per-request SAE clamps gather
        # phase-specific global state.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        global_spec = SAEClampSpec(
            module_name="g",
            phase="both",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=3.5),)
                }
            },
        )
        manager.set_global_clamps(
            prefill_specs=(global_spec,),
            decode_specs=(global_spec,),
        )
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(2, 5),
            layer_idx=20,
        )
        state = _slot(m)
        kind = state.clamp_kind
        value = state.clamp_value
        # Position 1 corresponds to feature_idx=5 in clampable_features.
        assert int(kind[1, 1].item()) == 1  # prefill global row
        assert float(value[1, 1].item()) == 3.5
        assert int(kind[2, 1].item()) == 1  # decode global row
        assert float(value[2, 1].item()) == 3.5
        assert int(kind[0, 1].item()) == 0  # row 0 remains no-op
        assert int(kind[1, 0].item()) == 0  # other features untouched
        assert bool(state.any_active.item()) is True

    def test_phase_specific_globals_do_not_share_row(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        prefill_spec = SAEClampSpec(
            module_name="g",
            phase="prefill",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=3.5),)
                }
            },
        )
        decode_spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=9.0),)
                }
            },
        )
        manager.set_global_clamps(
            prefill_specs=(prefill_spec,),
            decode_specs=(decode_spec,),
        )
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(5,),
            layer_idx=20,
        )
        state = _slot(m)
        kind = state.clamp_kind
        value = state.clamp_value
        assert int(kind[0, 0].item()) == CLAMP_KIND_NONE
        assert int(kind[1, 0].item()) == CLAMP_KIND_ABSOLUTE
        assert float(value[1, 0].item()) == 3.5
        assert int(kind[2, 0].item()) == CLAMP_KIND_ABSOLUTE
        assert float(value[2, 0].item()) == 9.0

    def test_global_clamp_merges_into_per_request_row(self):
        # A request with its own SAE clamp on a *different* feature
        # gets the global stacked on top — same row, both clamps active.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        global_spec = SAEClampSpec(
            module_name="g",
            phase="both",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=2, kind="absolute", value=1.0),)
                }
            },
        )
        req_spec = SAEClampSpec(
            module_name="g",
            phase="both",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="additive", value=2.0),)
                }
            },
        )
        manager.set_global_clamps(
            prefill_specs=(global_spec,), decode_specs=(global_spec,)
        )
        row = manager.register_clamp_spec(0xCAFE, (req_spec,), "prefill")
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(2, 5),
            layer_idx=20,
        )
        state = _slot(m)
        kind = state.clamp_kind
        value = state.clamp_value
        # The prefill global row carries the global clamp.
        assert int(kind[1, 0].item()) == 1
        assert float(value[1, 0].item()) == 1.0
        assert int(kind[0, 0].item()) == 0
        # Per-request row carries BOTH the global (position 0) AND the
        # request's own clamp (position 1).
        assert int(kind[row, 0].item()) == 1  # global applied here too
        assert float(value[row, 0].item()) == 1.0
        assert int(kind[row, 1].item()) == 2  # CLAMP_KIND_ADDITIVE
        assert float(value[row, 1].item()) == 2.0

    def test_clearing_globals_re_zeros_phase_global_rows(self):
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        global_spec = SAEClampSpec(
            module_name="g",
            phase="both",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=5, kind="absolute", value=3.5),)
                }
            },
        )
        manager.set_global_clamps(
            prefill_specs=(global_spec,), decode_specs=(global_spec,)
        )
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(2, 5),
            layer_idx=20,
        )
        state = _slot(m)
        kind = state.clamp_kind
        # Global active.
        assert int(kind[1, 1].item()) == 1
        assert int(kind[2, 1].item()) == 1
        manager.clear_global_clamps()
        populate_sae_clamp_table(
            manager=manager,
            module=m,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="g",
            clampable_features=(2, 5),
            layer_idx=20,
        )
        # Phase global rows re-zeroed; row 0 stays zero throughout.
        assert int(kind[1, 1].item()) == 0
        assert int(kind[2, 1].item()) == 0
        assert int(kind[0, 1].item()) == 0

    def test_unknown_feature_idx_in_spec_raises(self):
        # spec references feature 99 but clampable_features only has (0,)
        # — admission-time bug should fail loud.
        m = _layer_with_sae(
            hook=SteeringHookPoint.POST_BLOCK,
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            module_name="g",
        )
        manager = SAEClampManager(max_sae_configs=2)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=99, kind="absolute", value=1.0),)
                }
            },
        )
        manager.register_clamp_spec(123, (spec,), "prefill")

        with pytest.raises(ValueError, match="not in clampable_features"):
            populate_sae_clamp_table(
                manager=manager,
                module=m,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="g",
                clampable_features=(0,),
                worker_phase="prefill",
                layer_idx=20,
            )
