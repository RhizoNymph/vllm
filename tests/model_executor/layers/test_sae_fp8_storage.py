# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 (e4m3) storage for SAE weight tables — CPU contracts.

Covers the opt-in ``storage_dtype="fp8_e4m3"`` path:

* per-row quantize/dequant round-trip accuracy on realistic weight
  distributions (zero rows, tiny rows, large rows included);
* buffer registration with fp8 storage (weight dtype, always-present
  fp32 scale buffers, zeroing includes scales);
* CPU op numerics: fp8-stored weights dequantize inside the ops and
  stay close to the unquantized reference (delta + full recon);
* layer dispatch passes the slot scale buffers through.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm.config.sae_steering_types import (
    SAE_STORAGE_DTYPE_AUTO,
    SAE_STORAGE_DTYPE_FP8_E4M3,
    VALID_SAE_STORAGE_DTYPES,
    SAEActivation,
)
from vllm.model_executor.layers.sae_fp8 import (
    FP8_E4M3_MAX,
    FP8_STORAGE_DTYPE,
    dequantize_fp8_rowwise,
    maybe_dequantize_rowwise,
    quantize_fp8_rowwise,
    resolve_sae_storage_dtype,
)
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_DECODER_SCALE_ATTR,
    HOOK_POINT_FR_ENCODER_SCALE_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    apply_sae_full_reconstruction,
    register_sae_full_recon_buffers,
    unregister_sae_full_recon_buffers,
)
from vllm.model_executor.layers.sae_steering import (
    apply_sae_delta,
    get_sae_slot_state,
    register_sae_buffers,
    register_sae_index_buffer,
    unregister_sae_buffers,
)
from vllm.model_executor.layers.steering import SteeringHookPoint

POST_BLOCK = SteeringHookPoint.POST_BLOCK

# Error bounds for e4m3 round-to-nearest: 3 mantissa bits give a
# relative half-ulp of 2^-4 for normal values; subnormal spacing is
# 2^-9 in scaled units.
_REL_TOL = 2.0**-4 + 1e-6
_SUBNORMAL_ATOL_FACTOR = 2.0**-9


def _assert_roundtrip_close(w: torch.Tensor) -> None:
    q, scale = quantize_fp8_rowwise(w)
    assert q.dtype == FP8_STORAGE_DTYPE
    assert scale.dtype == torch.float32
    assert scale.shape == (w.shape[0],)
    assert bool((scale > 0).all())
    deq = dequantize_fp8_rowwise(q, scale)
    w32 = w.to(torch.float32)
    err = (deq - w32).abs()
    bound = torch.maximum(
        w32.abs() * _REL_TOL,
        scale.unsqueeze(1) * _SUBNORMAL_ATOL_FACTOR * (1 + 1e-6),
    )
    assert bool((err <= bound).all()), (
        f"max err {err.max().item():.3e} exceeds bound "
        f"(max bound violation {(err - bound).max().item():.3e})"
    )


class TestQuantizeRoundTrip:
    def test_gaussian_rows(self):
        torch.manual_seed(0)
        _assert_roundtrip_close(torch.randn(64, 128) * 0.02)

    def test_mixed_magnitude_rows(self):
        torch.manual_seed(1)
        w = torch.randn(8, 32)
        w[1] *= 1e-6  # tiny row
        w[2] *= 1e4  # large row
        w[3] = 0.0  # zero row
        _assert_roundtrip_close(w)

    def test_zero_row_dequantizes_to_exact_zero(self):
        w = torch.zeros(3, 16)
        w[1] = torch.randn(16)
        q, scale = quantize_fp8_rowwise(w)
        deq = dequantize_fp8_rowwise(q, scale)
        assert bool((deq[0] == 0).all())
        assert bool((deq[2] == 0).all())
        # Guard: no NaN/Inf anywhere (a zero amax must not divide by 0).
        assert bool(torch.isfinite(deq).all())

    def test_row_max_maps_to_fp8_max(self):
        w = torch.zeros(1, 4)
        w[0, 2] = 7.0
        q, scale = quantize_fp8_rowwise(w)
        assert scale[0].item() == pytest.approx(7.0 / FP8_E4M3_MAX)
        assert q.to(torch.float32)[0, 2].item() == pytest.approx(FP8_E4M3_MAX)
        deq = dequantize_fp8_rowwise(q, scale)
        assert deq[0, 2].item() == pytest.approx(7.0, rel=1e-6)

    def test_bf16_input_accepted(self):
        torch.manual_seed(2)
        _assert_roundtrip_close(torch.randn(4, 8).to(torch.bfloat16))

    def test_no_nans_on_extreme_values(self):
        w = torch.tensor([[3e38, -3e38, 1.0, 0.0]])
        q, scale = quantize_fp8_rowwise(w)
        assert bool(torch.isfinite(q.to(torch.float32)).all())

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            quantize_fp8_rowwise(torch.zeros(4))

    def test_maybe_dequantize_passthrough_for_non_fp8(self):
        w = torch.randn(2, 4)
        assert maybe_dequantize_rowwise(w, None) is w

    def test_maybe_dequantize_requires_scale_for_fp8(self):
        q, scale = quantize_fp8_rowwise(torch.randn(2, 4))
        with pytest.raises(ValueError, match="scale"):
            maybe_dequantize_rowwise(q, None)
        out = maybe_dequantize_rowwise(q, scale)
        assert out.dtype == torch.float32


class TestResolveStorageDtype:
    def test_auto_returns_compute_dtype(self):
        assert (
            resolve_sae_storage_dtype(SAE_STORAGE_DTYPE_AUTO, torch.bfloat16)
            is torch.bfloat16
        )

    def test_fp8_returns_fp8(self):
        assert (
            resolve_sae_storage_dtype(SAE_STORAGE_DTYPE_FP8_E4M3, torch.bfloat16)
            is FP8_STORAGE_DTYPE
        )

    def test_unknown_value_fails_loudly(self):
        with pytest.raises(ValueError, match="storage_dtype"):
            resolve_sae_storage_dtype("fp4", torch.float32)

    def test_valid_values_tuple(self):
        assert SAE_STORAGE_DTYPE_AUTO in VALID_SAE_STORAGE_DTYPES
        assert SAE_STORAGE_DTYPE_FP8_E4M3 in VALID_SAE_STORAGE_DTYPES


def _register_delta_slot(
    module: nn.Module,
    *,
    n_clamp: int = 3,
    hidden: int = 8,
    storage_dtype: torch.dtype | None = None,
) -> None:
    register_sae_buffers(
        module,
        hook_point=POST_BLOCK,
        module_name="m",
        activation=SAEActivation.RELU,
        activation_params={},
        n_clamp=n_clamp,
        hidden_size=hidden,
        max_sae_configs=2,
        dtype=torch.float32,
        storage_dtype=storage_dtype,
    )


class TestDeltaBufferRegistration:
    def test_fp8_storage_dtype_on_weight_buffers(self):
        m = nn.Module()
        _register_delta_slot(m, storage_dtype=FP8_STORAGE_DTYPE)
        state = get_sae_slot_state(m, POST_BLOCK, "m")
        assert state is not None
        assert state.encoder_weight.dtype == FP8_STORAGE_DTYPE
        assert state.decoder_weight.dtype == FP8_STORAGE_DTYPE
        # Bias / threshold / clamp tables keep their dtypes.
        assert state.encoder_bias.dtype == torch.float32
        assert state.threshold.dtype == torch.float32
        assert state.clamp_kind.dtype == torch.int8

    def test_scale_buffers_always_present_fp32(self):
        for storage in (None, FP8_STORAGE_DTYPE):
            m = nn.Module()
            _register_delta_slot(m, storage_dtype=storage)
            state = get_sae_slot_state(m, POST_BLOCK, "m")
            assert state.encoder_scale.dtype == torch.float32
            assert state.decoder_scale.dtype == torch.float32
            assert tuple(state.encoder_scale.shape) == (3,)
            assert tuple(state.decoder_scale.shape) == (3,)

    def test_deactivate_zeroes_scales(self):
        m = nn.Module()
        _register_delta_slot(m, storage_dtype=FP8_STORAGE_DTYPE)
        state = get_sae_slot_state(m, POST_BLOCK, "m")
        state.encoder_scale.fill_(0.5)
        state.decoder_scale.fill_(0.5)
        unregister_sae_buffers(
            m, hook_point=POST_BLOCK, module_name="m", deactivate_only=True
        )
        after = get_sae_slot_state(m, POST_BLOCK, "m")
        assert not after.encoder_scale.any()
        assert not after.decoder_scale.any()
        # Zeroed fp8 slot dequantizes to exact zero (no division hazard).
        deq = dequantize_fp8_rowwise(after.encoder_weight, after.encoder_scale)
        assert bool((deq == 0).all())

    def test_frozen_reuse_rejects_storage_dtype_drift(self):
        m = nn.Module()
        _register_delta_slot(m, storage_dtype=FP8_STORAGE_DTYPE)
        with pytest.raises(ValueError, match="storage dtype"):
            register_sae_buffers(
                m,
                hook_point=POST_BLOCK,
                module_name="m",
                activation=SAEActivation.RELU,
                activation_params={},
                n_clamp=3,
                hidden_size=8,
                max_sae_configs=2,
                dtype=torch.float32,
                allow_reuse=True,
            )

    def test_frozen_reuse_matching_dtype_zeroes_in_place(self):
        m = nn.Module()
        _register_delta_slot(m, storage_dtype=FP8_STORAGE_DTYPE)
        before = get_sae_slot_state(m, POST_BLOCK, "m")
        before.encoder_scale.fill_(1.0)
        register_sae_buffers(
            m,
            hook_point=POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=3,
            hidden_size=8,
            max_sae_configs=2,
            dtype=torch.float32,
            storage_dtype=FP8_STORAGE_DTYPE,
            allow_reuse=True,
        )
        after = get_sae_slot_state(m, POST_BLOCK, "m")
        assert after.encoder_weight is before.encoder_weight
        assert after.encoder_scale is before.encoder_scale
        assert not after.encoder_scale.any()


class TestFRBufferRegistration:
    def _register(self, m: nn.Module, storage_dtype: torch.dtype | None = None):
        register_sae_full_recon_buffers(
            m,
            hook_point=POST_BLOCK,
            module_name="fr",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=6,
            n_clamp=2,
            hidden_size=8,
            max_recon_configs=2,
            clampable_features=torch.tensor([0, 3], dtype=torch.int64),
            dtype=torch.float32,
            storage_dtype=storage_dtype,
        )

    def test_fp8_weights_and_scale_buffers(self):
        m = nn.Module()
        self._register(m, storage_dtype=FP8_STORAGE_DTYPE)
        enc = getattr(m, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK])
        assert enc.dtype == FP8_STORAGE_DTYPE
        enc_scale = getattr(m, HOOK_POINT_FR_ENCODER_SCALE_ATTR[POST_BLOCK])
        dec_scale = getattr(m, HOOK_POINT_FR_DECODER_SCALE_ATTR[POST_BLOCK])
        assert enc_scale.dtype == torch.float32
        assert tuple(enc_scale.shape) == (6,)
        assert tuple(dec_scale.shape) == (6,)

    def test_scale_buffers_present_for_auto(self):
        m = nn.Module()
        self._register(m)
        enc = getattr(m, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK])
        assert enc.dtype == torch.float32
        assert hasattr(m, HOOK_POINT_FR_ENCODER_SCALE_ATTR[POST_BLOCK])

    def test_deactivate_zeroes_scales(self):
        m = nn.Module()
        self._register(m, storage_dtype=FP8_STORAGE_DTYPE)
        getattr(m, HOOK_POINT_FR_ENCODER_SCALE_ATTR[POST_BLOCK]).fill_(0.7)
        unregister_sae_full_recon_buffers(
            m, hook_point=POST_BLOCK, deactivate_only=True
        )
        assert not getattr(m, HOOK_POINT_FR_ENCODER_SCALE_ATTR[POST_BLOCK]).any()

    def test_frozen_reuse_rejects_storage_dtype_drift(self):
        m = nn.Module()
        self._register(m, storage_dtype=FP8_STORAGE_DTYPE)
        with pytest.raises(ValueError, match="storage dtype"):
            register_sae_full_recon_buffers(
                m,
                hook_point=POST_BLOCK,
                module_name="fr",
                activation=SAEActivation.RELU,
                activation_params={},
                d_sae=6,
                n_clamp=2,
                hidden_size=8,
                max_recon_configs=2,
                clampable_features=torch.tensor([0, 3], dtype=torch.int64),
                dtype=torch.float32,
                allow_reuse=True,
            )


def _delta_reference_inputs(
    seed: int = 0, n_tokens: int = 5, d_model: int = 16, n_clamp: int = 4
):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tokens, d_model, generator=g)
    enc_w = torch.randn(n_clamp, d_model, generator=g) * 0.1
    enc_b = torch.randn(n_clamp, generator=g) * 0.01
    dec_w = torch.randn(n_clamp, d_model, generator=g) * 0.1
    kind = torch.zeros(n_tokens, n_clamp, dtype=torch.int8)
    kind[:, 1] = 1
    kind[:, 2] = 2
    value = torch.full((n_tokens, n_clamp), 2.0)
    only = torch.zeros(n_tokens, n_clamp, dtype=torch.bool)
    return hidden, enc_w, enc_b, dec_w, kind, value, only


class TestDeltaOpFp8Numerics:
    def test_fp8_matches_dequantized_reference_exactly(self):
        hidden, enc_w, enc_b, dec_w, kind, value, only = _delta_reference_inputs()
        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(dec_w)

        out_fp8 = apply_sae_delta(
            hidden,
            q_enc,
            enc_b,
            q_dec,
            SAEActivation.RELU,
            {},
            kind,
            value,
            only,
            encoder_scale=s_enc,
            decoder_scale=s_dec,
        )
        # Reference: same math on explicitly-dequantized weights.
        out_ref = apply_sae_delta(
            hidden,
            dequantize_fp8_rowwise(q_enc, s_enc),
            enc_b,
            dequantize_fp8_rowwise(q_dec, s_dec),
            SAEActivation.RELU,
            {},
            kind,
            value,
            only,
        )
        torch.testing.assert_close(out_fp8, out_ref, rtol=0, atol=0)

    def test_fp8_close_to_unquantized(self):
        hidden, enc_w, enc_b, dec_w, kind, value, only = _delta_reference_inputs()
        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(dec_w)
        out_fp8 = apply_sae_delta(
            hidden,
            q_enc,
            enc_b,
            q_dec,
            SAEActivation.RELU,
            {},
            kind,
            value,
            only,
            encoder_scale=s_enc,
            decoder_scale=s_dec,
        )
        out_ref = apply_sae_delta(
            hidden, enc_w, enc_b, dec_w, SAEActivation.RELU, {}, kind, value, only
        )
        # Loose tolerance: fp8 storage is lossy by design.
        torch.testing.assert_close(out_fp8, out_ref, rtol=0.1, atol=0.05)

    def test_fp8_without_scale_raises(self):
        hidden, enc_w, enc_b, dec_w, kind, value, only = _delta_reference_inputs()
        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, _ = quantize_fp8_rowwise(dec_w)
        with pytest.raises(ValueError, match="scale"):
            apply_sae_delta(
                hidden,
                q_enc,
                enc_b,
                q_dec,
                SAEActivation.RELU,
                {},
                kind,
                value,
                only,
                encoder_scale=s_enc,
            )

    def test_jumprelu_with_fp8(self):
        hidden, enc_w, enc_b, dec_w, kind, value, only = _delta_reference_inputs(seed=3)
        thr = torch.full((4,), 0.05, dtype=torch.float32)
        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(dec_w)
        out_fp8 = apply_sae_delta(
            hidden,
            q_enc,
            enc_b,
            q_dec,
            SAEActivation.JUMPRELU,
            {},
            kind,
            value,
            only,
            threshold=thr,
            encoder_scale=s_enc,
            decoder_scale=s_dec,
        )
        out_ref = apply_sae_delta(
            hidden,
            dequantize_fp8_rowwise(q_enc, s_enc),
            enc_b,
            dequantize_fp8_rowwise(q_dec, s_dec),
            SAEActivation.JUMPRELU,
            {},
            kind,
            value,
            only,
            threshold=thr,
        )
        torch.testing.assert_close(out_fp8, out_ref, rtol=0, atol=0)


class TestFROpFp8Numerics:
    def test_fp8_matches_dequantized_reference_exactly(self):
        g = torch.Generator().manual_seed(7)
        n_tokens, d_model, d_sae = 4, 8, 12
        hidden = torch.randn(n_tokens, d_model, generator=g)
        enc_w = torch.randn(d_sae, d_model, generator=g) * 0.1
        enc_b = torch.randn(d_sae, generator=g) * 0.01
        dec_w = torch.randn(d_sae, d_model, generator=g) * 0.1
        dec_b = torch.randn(d_model, generator=g) * 0.01
        feats = torch.tensor([1, 5], dtype=torch.int64)
        kind = torch.zeros(n_tokens, 2, dtype=torch.int8)
        kind[:, 0] = 1
        value = torch.full((n_tokens, 2), 3.0)
        only = torch.zeros(n_tokens, 2, dtype=torch.bool)
        mask = torch.tensor([True, False, True, True])

        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(dec_w)

        out_fp8 = apply_sae_full_reconstruction(
            hidden,
            q_enc,
            enc_b,
            q_dec,
            dec_b,
            SAEActivation.RELU,
            {},
            feats,
            kind,
            value,
            only,
            mask,
            encoder_scale=s_enc,
            decoder_scale=s_dec,
        )
        out_ref = apply_sae_full_reconstruction(
            hidden,
            dequantize_fp8_rowwise(q_enc, s_enc),
            enc_b,
            dequantize_fp8_rowwise(q_dec, s_dec),
            dec_b,
            SAEActivation.RELU,
            {},
            feats,
            kind,
            value,
            only,
            mask,
        )
        torch.testing.assert_close(out_fp8, out_ref, rtol=0, atol=0)
        # Unmasked token passes through bit-identically.
        torch.testing.assert_close(out_fp8[1], hidden[1], rtol=0, atol=0)

    def test_fp8_without_scale_raises(self):
        hidden = torch.randn(2, 4)
        enc_w = torch.randn(6, 4)
        q_enc, _ = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(torch.randn(6, 4))
        with pytest.raises(ValueError, match="scale"):
            apply_sae_full_reconstruction(
                hidden,
                q_enc,
                torch.zeros(6),
                q_dec,
                torch.zeros(4),
                SAEActivation.RELU,
                {},
                torch.tensor([0], dtype=torch.int64),
                torch.zeros(2, 1, dtype=torch.int8),
                torch.zeros(2, 1),
                torch.zeros(2, 1, dtype=torch.bool),
                torch.ones(2, dtype=torch.bool),
                decoder_scale=s_dec,
            )


class TestLayerDispatchFp8:
    def test_delta_layer_dispatch_uses_slot_scales(self):
        torch.manual_seed(11)
        n_clamp, hidden_size, n_tokens = 3, 8, 4
        m = nn.Module()
        _register_delta_slot(
            m, n_clamp=n_clamp, hidden=hidden_size, storage_dtype=FP8_STORAGE_DTYPE
        )
        register_sae_index_buffer(m, max_tokens=8)
        state = get_sae_slot_state(m, POST_BLOCK, "m")

        enc_w = torch.randn(n_clamp, hidden_size) * 0.1
        dec_w = torch.randn(n_clamp, hidden_size) * 0.1
        q_enc, s_enc = quantize_fp8_rowwise(enc_w)
        q_dec, s_dec = quantize_fp8_rowwise(dec_w)
        state.encoder_weight.copy_(q_enc)
        state.decoder_weight.copy_(q_dec)
        state.encoder_scale.copy_(s_enc)
        state.decoder_scale.copy_(s_dec)
        enc_b = torch.randn(n_clamp) * 0.01
        state.encoder_bias.copy_(enc_b)
        # Row 3 (first per-request row) applies an absolute clamp.
        state.clamp_kind[3, 0] = 1
        state.clamp_value[3, 0] = 4.0
        state.any_active.fill_(True)
        m.sae_index[:n_tokens] = 3

        from vllm.model_executor.layers.sae_steering import apply_layer_sae_delta

        hidden = torch.randn(n_tokens, hidden_size)
        out = apply_layer_sae_delta(m, hidden, POST_BLOCK)

        ref = apply_sae_delta(
            hidden,
            dequantize_fp8_rowwise(q_enc, s_enc),
            enc_b,
            dequantize_fp8_rowwise(q_dec, s_dec),
            SAEActivation.RELU,
            {},
            torch.stack([state.clamp_kind[3]] * n_tokens),
            torch.stack([state.clamp_value[3]] * n_tokens),
            torch.stack([state.clamp_only_if_active[3]] * n_tokens),
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        assert not torch.equal(out, hidden)
