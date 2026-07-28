# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitor-gated SAE delta clamps (the ``gated`` opt-in).

Covers the delta path's gate participation end-to-end on CPU:

* buffer contract — the per-row ``sae_clamp_row_gated`` table is always
  registered with a slot, zero-filled, zeroed on slot reuse / deactivate
  / spare claim,
* op numerics — a gated row's clamp delta scales by ``row_gate[token]``
  (``g = 1 - gated[row] * (1 - row_gate[token])``); ungated rows are
  unaffected by any ``row_gate`` value; ``row_gate == 0`` on a gated row
  removes the clamp effect entirely,
* default-argument back-compat — omitting the gate arguments reproduces
  the pre-``gated`` output bit-for-bit,
* layer dispatch — the shim feeds the slot's gate table and the shared
  ``steering_row_gate`` buffer to the indexed op; layers without the
  buffer behave ungated,
* populator — rows fed by a ``gated=True`` spec get gate participation
  1.0; ungated rows and the sentinel stay 0.0.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    SAESlotState,
    apply_layer_sae_delta,
    apply_sae_delta,
    claim_sae_spare_slot,
    get_sae_slot_state,
    populate_sae_clamp_table,
    register_sae_buffers,
    register_sae_index_buffer,
    sae_site_slots,
    unregister_sae_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.sae_clamp_manager import SAEClampManager

HOOK = SteeringHookPoint.POST_BLOCK


def _layer(
    *,
    n_clamp: int = 1,
    hidden_size: int = 4,
    max_sae_configs: int = 2,
    max_tokens: int = 8,
    module_name: str = "g",
    spare: bool = False,
) -> nn.Module:
    m = nn.Module()
    register_sae_buffers(
        m,
        hook_point=HOOK,
        module_name=module_name,
        activation=SAEActivation.RELU,
        activation_params={},
        n_clamp=n_clamp,
        hidden_size=hidden_size,
        max_sae_configs=max_sae_configs,
        dtype=torch.float32,
        spare=spare,
    )
    register_sae_index_buffer(m, max_tokens=max_tokens)
    return m


def _slot(m: nn.Module, module_name: str = "g") -> SAESlotState:
    state = get_sae_slot_state(m, HOOK, module_name)
    assert state is not None
    return state


def _arm_slot(state: SAESlotState, row: int, value: float = 5.0) -> None:
    """Write a simple absolute clamp into ``row`` with unit weights."""
    n_clamp, hidden = state.encoder_weight.shape
    enc = torch.zeros(n_clamp, hidden)
    enc[0, 0] = 1.0
    dec = torch.zeros(n_clamp, hidden)
    dec[0, hidden - 1] = 1.0
    state.encoder_weight.copy_(enc)
    state.decoder_weight.copy_(dec)
    state.clamp_kind[row, 0] = CLAMP_KIND_ABSOLUTE
    state.clamp_value[row, 0] = value
    state.any_active.fill_(True)


class TestGateBufferContract:
    def test_gate_table_registered_and_zero(self):
        m = _layer(max_sae_configs=3)
        state = _slot(m)
        gate = state.clamp_row_gated
        assert gate.dtype == torch.float32
        assert tuple(gate.shape) == (3 + 3,)
        assert torch.all(gate == 0.0)

    def test_gate_table_registered_for_spare_slots(self):
        m = _layer(module_name="", spare=True)
        (record,) = sae_site_slots(m, HOOK)
        assert record.spare
        state = get_sae_slot_state(m, HOOK, record.module_name)
        assert state is not None
        assert torch.all(state.clamp_row_gated == 0.0)

    def test_gate_table_zeroed_on_deactivate(self):
        m = _layer()
        state = _slot(m)
        state.clamp_row_gated.fill_(1.0)
        unregister_sae_buffers(
            m, hook_point=HOOK, module_name="g", deactivate_only=True
        )
        assert torch.all(state.clamp_row_gated == 0.0)

    def test_gate_table_zeroed_on_reuse(self):
        m = _layer()
        state = _slot(m)
        state.clamp_row_gated.fill_(1.0)
        register_sae_buffers(
            m,
            hook_point=HOOK,
            module_name="g",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=1,
            hidden_size=4,
            max_sae_configs=2,
            dtype=torch.float32,
            allow_reuse=True,
        )
        assert torch.all(state.clamp_row_gated == 0.0)

    def test_gate_table_zeroed_on_spare_claim(self):
        m = _layer(module_name="", spare=True)
        (record,) = sae_site_slots(m, HOOK)
        state = get_sae_slot_state(m, HOOK, record.module_name)
        state.clamp_row_gated.fill_(1.0)
        claimed = claim_sae_spare_slot(m, HOOK, "claimer")
        assert claimed is not None
        state = _slot(m, "claimer")
        assert torch.all(state.clamp_row_gated == 0.0)


class TestGatedOpNumerics:
    """Direct-tensor API numerics for the per-token gate."""

    def _delta_out(self, h, gated=None, row_gate=None):
        n_tokens = h.shape[0]
        enc_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        enc_b = torch.zeros(1)
        dec_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        kind = torch.full((n_tokens, 1), CLAMP_KIND_ABSOLUTE, dtype=torch.int8)
        value = torch.full((n_tokens, 1), 5.0)
        only = torch.zeros(n_tokens, 1, dtype=torch.bool)
        return apply_sae_delta(
            h,
            enc_w,
            enc_b,
            dec_w,
            SAEActivation.RELU,
            {},
            kind,
            value,
            only,
            clamp_row_gated=gated,
            row_gate=row_gate,
        )

    def test_ungated_ignores_row_gate(self):
        h = torch.randn(3, 4)
        base = self._delta_out(h)
        out = self._delta_out(
            h,
            gated=torch.zeros(3),
            row_gate=torch.tensor([0.0, 0.5, 1.0]),
        )
        assert torch.equal(out, base)

    def test_gated_scales_delta_by_row_gate(self):
        h = torch.randn(4, 4)
        base = self._delta_out(h)
        gates = torch.tensor([0.0, 0.25, 0.5, 1.0])
        out = self._delta_out(h, gated=torch.ones(4), row_gate=gates)
        base_delta = base - h
        out_delta = out - h
        assert torch.allclose(
            out_delta, gates.unsqueeze(1) * base_delta, atol=1e-6
        )

    def test_zero_row_gate_removes_clamp_entirely(self):
        h = torch.randn(2, 4)
        out = self._delta_out(h, gated=torch.ones(2), row_gate=torch.zeros(2))
        assert torch.allclose(out, h, atol=1e-7)

    def test_full_row_gate_matches_ungated(self):
        h = torch.randn(2, 4)
        base = self._delta_out(h)
        out = self._delta_out(h, gated=torch.ones(2), row_gate=torch.ones(2))
        assert torch.allclose(out, base, atol=1e-7)

    def test_omitted_gate_args_bit_identical_to_legacy_call(self):
        h = torch.randn(3, 4)
        assert torch.equal(self._delta_out(h), self._delta_out(h, None, None))


class TestGatedLayerDispatch:
    def _armed_layer(self, *, with_row_gate: bool) -> nn.Module:
        m = _layer(max_tokens=4)
        state = _slot(m)
        _arm_slot(state, row=3)
        m.sae_index[:2] = 3
        if with_row_gate:
            m.register_buffer(
                "steering_row_gate", torch.ones(4, dtype=torch.float32)
            )
        return m

    def test_gated_row_scaled_by_shared_row_gate(self):
        m = self._armed_layer(with_row_gate=True)
        state = _slot(m)
        state.clamp_row_gated[3] = 1.0
        m.steering_row_gate[0] = 0.25
        m.steering_row_gate[1] = 1.0
        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, HOOK)

        state.clamp_row_gated[3] = 0.0
        base = apply_layer_sae_delta(m, h, HOOK)
        base_delta = base - h
        out_delta = out - h
        assert torch.allclose(out_delta[0], 0.25 * base_delta[0], atol=1e-6)
        assert torch.allclose(out_delta[1], base_delta[1], atol=1e-6)

    def test_ungated_row_unaffected_by_row_gate(self):
        m = self._armed_layer(with_row_gate=True)
        m.steering_row_gate.fill_(0.0)
        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, HOOK)
        assert not torch.allclose(out, h)  # clamp still applies

    def test_layer_without_row_gate_buffer_behaves_ungated(self):
        m = self._armed_layer(with_row_gate=False)
        state = _slot(m)
        state.clamp_row_gated[3] = 1.0
        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, HOOK)
        assert not torch.allclose(out, h)  # full-strength clamp

    def test_gated_zero_row_gate_is_noop(self):
        m = self._armed_layer(with_row_gate=True)
        state = _slot(m)
        state.clamp_row_gated[3] = 1.0
        m.steering_row_gate.fill_(0.0)
        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, HOOK)
        assert torch.allclose(out, h, atol=1e-6)


def _spec(gated: bool, *, phase: str = "both", value: float = 5.0) -> SAEClampSpec:
    return SAEClampSpec(
        module_name="g",
        clamps={
            HOOK.value: {
                0: [SAEClampEntry(feature_idx=0, kind="absolute", value=value)]
            }
        },
        phase=phase,
        gated=gated,
    )


class TestGatedPopulate:
    def test_gated_spec_marks_row(self):
        m = _layer()
        mgr = SAEClampManager(max_sae_configs=2)
        row = mgr.register_clamp_spec(11, (_spec(True),), "decode")
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        assert state.clamp_row_gated[row].item() == 1.0
        assert state.clamp_row_gated[0].item() == 0.0

    def test_ungated_spec_leaves_row_zero(self):
        m = _layer()
        mgr = SAEClampManager(max_sae_configs=2)
        row = mgr.register_clamp_spec(12, (_spec(False),), "decode")
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        assert state.clamp_row_gated[row].item() == 0.0

    def test_release_then_repopulate_clears_gate(self):
        m = _layer()
        mgr = SAEClampManager(max_sae_configs=2)
        row = mgr.register_clamp_spec(13, (_spec(True),), "decode")
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        assert state.clamp_row_gated[row].item() == 1.0
        mgr.release_clamp_spec(13, "decode")
        # The freed row re-registers with an ungated spec; the populate
        # pass must clear the stale gate participation.
        row2 = mgr.register_clamp_spec(14, (_spec(False, value=2.0),), "decode")
        assert row2 == row
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        assert state.clamp_row_gated[row].item() == 0.0

    def test_gated_global_marks_global_row(self):
        m = _layer()
        mgr = SAEClampManager(max_sae_configs=2)
        mgr.set_global_clamps(decode_specs=(_spec(True, phase="decode"),))
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        assert state.clamp_row_gated[2].item() == 1.0  # decode global row
        assert state.clamp_row_gated[1].item() == 0.0  # prefill untouched

    def test_other_module_spec_does_not_mark(self):
        m = _layer()
        mgr = SAEClampManager(max_sae_configs=2)
        other = SAEClampSpec(
            module_name="other",
            clamps={
                HOOK.value: {
                    0: [SAEClampEntry(feature_idx=0, kind="absolute", value=1.0)]
                }
            },
            gated=True,
        )
        row = mgr.register_clamp_spec(15, (other,), "decode")
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        assert state.clamp_row_gated[row].item() == 0.0


class TestGatedEndToEnd:
    def test_populate_then_dispatch_scales_gated_row(self):
        m = _layer(max_tokens=4)
        m.register_buffer("steering_row_gate", torch.ones(4, dtype=torch.float32))
        mgr = SAEClampManager(max_sae_configs=2)
        row = mgr.register_clamp_spec(21, (_spec(True),), "decode")
        populate_sae_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="g",
            clampable_features=(0,),
            layer_idx=0,
        )
        state = _slot(m)
        enc = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        dec = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        state.encoder_weight.copy_(enc)
        state.decoder_weight.copy_(dec)
        m.sae_index[:2] = row
        m.steering_row_gate[0] = 0.5
        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(m, h, HOOK)

        m.steering_row_gate.fill_(1.0)
        full = apply_layer_sae_delta(m, h, HOOK)
        assert torch.allclose(out[0] - h[0], 0.5 * (full[0] - h[0]), atol=1e-6)
        assert torch.allclose(out[1], full[1], atol=1e-6)
