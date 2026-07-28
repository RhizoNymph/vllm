# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitor-gated SAE full-reconstruction clamps (the ``gated`` opt-in).

FR gating scales the CLAMP-induced feature delta only — the
reconstruction itself always applies for opted-in tokens:

    f_used = f + g * (f_clamped - f),   g = 1 - gated[row] * (1 - row_gate[t])

Because the decoder is linear, ``out(g) = out_unclamped + g *
(out_clamped - out_unclamped)`` is the reference identity the tests
check.  Also covers the per-row gate table's buffer lifecycle, the
compaction (CUDA-path body, runs on CPU tensors here) parity, layer
dispatch through the shared ``steering_row_gate``, and the populator.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampEntry,
    SAEFullReconstructionSpec,
)
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR,
    HOOK_POINT_FR_CLAMP_VALUE_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ROW_ACTIVE_ATTR,
    apply_layer_sae_full_reconstruction,
    apply_sae_full_reconstruction,
    populate_sae_full_recon_clamp_table,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
    unregister_sae_full_recon_buffers,
)
from vllm.model_executor.layers.sae_steering import CLAMP_KIND_ABSOLUTE
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)

HOOK = SteeringHookPoint.POST_BLOCK
D_SAE = 3
HIDDEN = 4


def _weights():
    torch.manual_seed(7)
    enc_w = torch.randn(D_SAE, HIDDEN)
    enc_b = torch.randn(D_SAE)
    dec_w = torch.randn(D_SAE, HIDDEN)
    dec_b = torch.randn(HIDDEN)
    return enc_w, enc_b, dec_w, dec_b


def _fr_out(h, kind, value, *, gated=None, row_gate=None, recon_mask=None):
    enc_w, enc_b, dec_w, dec_b = _weights()
    n = h.shape[0]
    if recon_mask is None:
        recon_mask = torch.ones(n, dtype=torch.bool)
    return apply_sae_full_reconstruction(
        h,
        enc_w,
        enc_b,
        dec_w,
        dec_b,
        SAEActivation.RELU,
        {},
        torch.tensor([0], dtype=torch.int64),
        kind,
        value,
        torch.zeros(n, 1, dtype=torch.bool),
        recon_mask,
        clamp_row_gated=gated,
        row_gate=row_gate,
    )


class TestGatedFrOpNumerics:
    def _clamped_unclamped(self, h):
        n = h.shape[0]
        kind = torch.full((n, 1), CLAMP_KIND_ABSOLUTE, dtype=torch.int8)
        value = torch.full((n, 1), 4.0)
        no_kind = torch.zeros(n, 1, dtype=torch.int8)
        clamped = _fr_out(h, kind, value)
        unclamped = _fr_out(h, no_kind, torch.zeros(n, 1))
        return kind, value, clamped, unclamped

    def test_gate_blends_clamp_effect_linearly(self):
        h = torch.randn(4, HIDDEN)
        kind, value, clamped, unclamped = self._clamped_unclamped(h)
        gates = torch.tensor([0.0, 0.25, 0.5, 1.0])
        out = _fr_out(h, kind, value, gated=torch.ones(4), row_gate=gates)
        expected = unclamped + gates.unsqueeze(1) * (clamped - unclamped)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_zero_gate_keeps_reconstruction(self):
        # g = 0 removes the clamp effect but the token is still fully
        # reconstructed (never equal to the raw residual).
        h = torch.randn(2, HIDDEN)
        kind, value, clamped, unclamped = self._clamped_unclamped(h)
        out = _fr_out(
            h, kind, value, gated=torch.ones(2), row_gate=torch.zeros(2)
        )
        assert torch.allclose(out, unclamped, atol=1e-5)
        assert not torch.allclose(out, h)

    def test_ungated_ignores_row_gate(self):
        h = torch.randn(3, HIDDEN)
        kind, value, clamped, _ = self._clamped_unclamped(h)
        out = _fr_out(
            h,
            kind,
            value,
            gated=torch.zeros(3),
            row_gate=torch.tensor([0.0, 0.5, 1.0]),
        )
        assert torch.allclose(out, clamped, atol=1e-6)

    def test_inactive_tokens_pass_through_regardless_of_gate(self):
        h = torch.randn(2, HIDDEN)
        kind = torch.full((2, 1), CLAMP_KIND_ABSOLUTE, dtype=torch.int8)
        value = torch.full((2, 1), 4.0)
        mask = torch.tensor([True, False])
        out = _fr_out(
            h,
            kind,
            value,
            gated=torch.ones(2),
            row_gate=torch.full((2,), 0.5),
            recon_mask=mask,
        )
        assert torch.equal(out[1], h[1])

    def test_omitted_gate_args_bit_identical_to_legacy_call(self):
        h = torch.randn(3, HIDDEN)
        kind, value, clamped, _ = self._clamped_unclamped(h)
        again = _fr_out(h, kind, value, gated=None, row_gate=None)
        assert torch.equal(clamped, again)


class TestCompactionPathParity:
    """The CUDA-path body (compaction) must match the eager body with
    gating; it runs fine on CPU tensors."""

    def test_compaction_matches_eager_with_gates(self):
        from vllm.model_executor.layers.sae_full_reconstruction import (
            _apply_sae_full_reconstruction_eager,
        )
        from vllm.model_executor.layers.sae_full_reconstruction_kernel import (
            apply_sae_full_recon_triton,
        )

        enc_w, enc_b, dec_w, dec_b = _weights()
        h = torch.randn(4, HIDDEN)
        threshold = torch.zeros(D_SAE, dtype=torch.float32)
        feats = torch.tensor([0], dtype=torch.int64)
        kind = torch.full((4, 1), CLAMP_KIND_ABSOLUTE, dtype=torch.int8)
        value = torch.full((4, 1), 4.0)
        only = torch.zeros(4, 1, dtype=torch.bool)
        mask = torch.tensor([True, False, True, True])
        gated = torch.tensor([1.0, 1.0, 0.0, 1.0])
        row_gate = torch.tensor([0.5, 0.5, 0.5, 0.0])
        args = (
            h,
            enc_w,
            enc_b,
            threshold,
            dec_w,
            dec_b,
            feats,
            kind,
            value,
            only,
            mask,
            0,
            0.0,
        )
        eager = _apply_sae_full_reconstruction_eager(
            *args, clamp_row_gated=gated, row_gate=row_gate
        )
        compacted = apply_sae_full_recon_triton(
            *args, clamp_row_gated=gated, row_gate=row_gate
        )
        assert torch.allclose(compacted, eager, atol=1e-5)


def _fr_layer(*, max_recon_configs: int = 2, max_tokens: int = 8) -> nn.Module:
    m = nn.Module()
    enc_w, enc_b, dec_w, dec_b = _weights()
    register_sae_full_recon_buffers(
        m,
        hook_point=HOOK,
        module_name="fr",
        activation=SAEActivation.RELU,
        activation_params={},
        d_sae=D_SAE,
        n_clamp=1,
        hidden_size=HIDDEN,
        max_recon_configs=max_recon_configs,
        clampable_features=torch.tensor([0], dtype=torch.int64),
        dtype=torch.float32,
    )
    getattr(m, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[HOOK]).copy_(enc_w)
    m.sae_fr_encoder_bias_post_block.copy_(enc_b)
    m.sae_fr_decoder_weight_post_block.copy_(dec_w)
    m.sae_fr_decoder_bias_post_block.copy_(dec_b)
    register_sae_recon_index_buffer(m, max_tokens=max_tokens)
    return m


class TestFrGateBufferContract:
    def test_gate_table_registered_and_zero(self):
        m = _fr_layer(max_recon_configs=3)
        gate = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        assert gate.dtype == torch.float32
        assert tuple(gate.shape) == (3 + 1,)
        assert torch.all(gate == 0.0)

    def test_gate_table_zeroed_on_deactivate(self):
        m = _fr_layer()
        gate = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        gate.fill_(1.0)
        unregister_sae_full_recon_buffers(m, hook_point=HOOK, deactivate_only=True)
        assert torch.all(gate == 0.0)

    def test_gate_table_removed_on_unregister(self):
        m = _fr_layer()
        unregister_sae_full_recon_buffers(m, hook_point=HOOK)
        assert not hasattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])


class TestFrGatedLayerDispatch:
    def _armed(self, *, with_row_gate: bool) -> nn.Module:
        m = _fr_layer()
        kind_table = getattr(m, HOOK_POINT_FR_CLAMP_KIND_ATTR[HOOK])
        value_table = getattr(m, HOOK_POINT_FR_CLAMP_VALUE_ATTR[HOOK])
        active_table = getattr(m, HOOK_POINT_FR_ROW_ACTIVE_ATTR[HOOK])
        kind_table[1, 0] = CLAMP_KIND_ABSOLUTE
        value_table[1, 0] = 4.0
        active_table[1] = True
        m.sae_recon_index[:2] = 1
        if with_row_gate:
            m.register_buffer(
                "steering_row_gate", torch.ones(8, dtype=torch.float32)
            )
        return m

    def test_gated_row_blends_by_row_gate(self):
        m = self._armed(with_row_gate=True)
        gate_table = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        gate_table[1] = 1.0
        m.steering_row_gate[0] = 0.25
        h = torch.randn(2, HIDDEN)
        out = apply_layer_sae_full_reconstruction(m, h, HOOK)

        gate_table[1] = 0.0
        clamped = apply_layer_sae_full_reconstruction(m, h, HOOK)
        kind_table = getattr(m, HOOK_POINT_FR_CLAMP_KIND_ATTR[HOOK])
        kind_table[1, 0] = 0
        unclamped = apply_layer_sae_full_reconstruction(m, h, HOOK)
        expected0 = unclamped[0] + 0.25 * (clamped[0] - unclamped[0])
        assert torch.allclose(out[0], expected0, atol=1e-5)
        assert torch.allclose(out[1], clamped[1], atol=1e-5)

    def test_layer_without_row_gate_buffer_behaves_ungated(self):
        m = self._armed(with_row_gate=False)
        gate_table = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        gate_table[1] = 1.0
        h = torch.randn(2, HIDDEN)
        out = apply_layer_sae_full_reconstruction(m, h, HOOK)
        gate_table[1] = 0.0
        clamped = apply_layer_sae_full_reconstruction(m, h, HOOK)
        assert torch.allclose(out, clamped, atol=1e-6)


def _fr_spec(gated: bool, *, with_clamps: bool = True) -> SAEFullReconstructionSpec:
    clamps = (
        {HOOK.value: {0: [SAEClampEntry(feature_idx=0, kind="absolute", value=4.0)]}}
        if with_clamps
        else {}
    )
    return SAEFullReconstructionSpec(module_name="fr", clamps=clamps, gated=gated)


class TestFrGatedPopulate:
    def _populate(self, m, mgr):
        populate_sae_full_recon_clamp_table(
            manager=mgr,
            module=m,
            hook_point=HOOK,
            module_name="fr",
            clampable_features=(0,),
            layer_idx=0,
        )

    def test_gated_spec_marks_row(self):
        m = _fr_layer()
        mgr = SAEFullReconstructionManager(max_recon_configs=2)
        row = mgr.register_recon_spec(31, (_fr_spec(True),), "decode")
        self._populate(m, mgr)
        gate_table = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        assert gate_table[row].item() == 1.0
        assert gate_table[0].item() == 0.0

    def test_ungated_spec_leaves_row_zero(self):
        m = _fr_layer()
        mgr = SAEFullReconstructionManager(max_recon_configs=2)
        row = mgr.register_recon_spec(32, (_fr_spec(False),), "decode")
        self._populate(m, mgr)
        gate_table = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        assert gate_table[row].item() == 0.0

    def test_release_then_repopulate_clears_gate(self):
        m = _fr_layer()
        mgr = SAEFullReconstructionManager(max_recon_configs=2)
        row = mgr.register_recon_spec(33, (_fr_spec(True),), "decode")
        self._populate(m, mgr)
        gate_table = getattr(m, HOOK_POINT_FR_CLAMP_ROW_GATED_ATTR[HOOK])
        assert gate_table[row].item() == 1.0
        mgr.release_recon_spec(33, "decode")
        row2 = mgr.register_recon_spec(34, (_fr_spec(False),), "decode")
        assert row2 == row
        self._populate(m, mgr)
        assert gate_table[row].item() == 0.0
