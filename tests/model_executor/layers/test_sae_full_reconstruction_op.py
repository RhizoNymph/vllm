# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric tests for the eager SAE full-reconstruction op.

Phase 4 Stage 1: math primitive only.  These tests verify the
vectorized PyTorch path against a hand-rolled per-(token, feature)
reference loop, plus the activation functions, clamp variants, the
per-token ``recon_mask`` gate, and the dtype / shape contracts.
Layer-hook integration, custom-op registration, and the Triton kernel
land in subsequent stages — exactly mirroring the Phase-1A → Phase-1B
→ Phase-2 progression for the delta path.

The op math is:

    pre_act     = h_t @ W_enc.T + b_enc          # (d_sae,)
    f           = activation(pre_act)            # (d_sae,)
    f' [c]      = apply_clamp(f[clampable_features[c]],
                              clamp_kind[t, c],
                              clamp_value[t, c],
                              only_if_active[t, c])
    h_t_new     = f' @ W_dec + b_dec             # (d_model,)
    out[t]      = h_t_new if recon_mask[t] else h_t

The reference loop reproduces this token-by-token without any
vectorisation, so the implementation under test is checked against a
transparent definition.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm.config.sae_steering_types import SAEActivation, SAEFullReconstructionSpec
from vllm.model_executor.layers.sae_full_reconstruction import (
    apply_layer_sae_full_reconstruction,
    apply_sae_full_reconstruction,
    populate_sae_full_recon_clamp_table,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
    sae_encode_full,
)
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
    CLAMP_KIND_NONE,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)


def _ref_apply_full_reconstruction(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: dict[str, float],
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-rolled per-(token, feature) reference.

    Uses Python loops on purpose — the vectorised op under test is
    being checked against a transparent definition; this loop must
    promote to fp32 in the same places the contract demands so a
    pure-fp32 expectation cannot mask a bug in the production code's
    promotion seams.
    """
    n_tokens, d_model = hidden_states.shape
    d_sae = encoder_weight.shape[0]
    out = hidden_states.clone()
    h_fp32 = hidden_states.to(torch.float32)
    enc_w_fp32 = encoder_weight.to(torch.float32)
    enc_b_fp32 = encoder_bias.to(torch.float32)
    dec_w_compute = decoder_weight.to(hidden_states.dtype)
    dec_b_compute = decoder_bias.to(hidden_states.dtype)
    feat_list = [int(x) for x in clampable_features.tolist()]
    for t in range(n_tokens):
        if not bool(recon_mask[t]):
            continue
        # Encode the full d_sae for this token.
        pre_act = h_fp32[t] @ enc_w_fp32.t() + enc_b_fp32  # (d_sae,)
        if activation is SAEActivation.RELU:
            f = torch.clamp(pre_act, min=0.0)
        elif activation is SAEActivation.JUMPRELU:
            threshold = activation_params["threshold"]
            f = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
        elif activation is SAEActivation.TOPK:
            k = int(activation_params["k"])
            if k >= d_sae:
                f = pre_act.clone()
            else:
                top_idx = sorted(
                    range(d_sae), key=lambda j: (-float(pre_act[j]), j)
                )[:k]
                mask = torch.zeros_like(pre_act, dtype=torch.bool)
                mask[top_idx] = True
                f = torch.where(mask, pre_act, torch.zeros_like(pre_act))
        else:
            raise AssertionError(f"unhandled activation {activation}")
        # Apply clamps at the clampable positions.
        for c, feat_idx in enumerate(feat_list):
            kind = int(clamp_kind[t, c])
            if kind == CLAMP_KIND_NONE:
                continue
            v = float(clamp_value[t, c])
            gated = bool(clamp_only_if_active[t, c])
            f_val = float(f[feat_idx])
            active = f_val != 0.0 if activation is SAEActivation.TOPK else f_val > 0.0
            if gated and not active:
                continue
            if kind == CLAMP_KIND_ABSOLUTE:
                new_f = v
            elif kind == CLAMP_KIND_ADDITIVE:
                new_f = f_val + v
            else:
                raise AssertionError(f"unhandled clamp kind {kind}")
            f[feat_idx] = new_f
        # Decode in compute dtype.
        f_compute = f.to(hidden_states.dtype)
        out[t] = f_compute @ dec_w_compute + dec_b_compute
    return out


def _make_random_inputs(
    *,
    n_tokens: int = 4,
    d_model: int = 6,
    d_sae: int = 16,
    n_clamp: int = 3,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randperm(d_sae, generator=g)[:n_clamp].sort().values.to(torch.int64)
    return {
        "hidden_states": torch.randn(n_tokens, d_model, generator=g, dtype=dtype),
        "encoder_weight": torch.randn(d_sae, d_model, generator=g, dtype=dtype),
        "encoder_bias": torch.randn(d_sae, generator=g, dtype=dtype),
        "decoder_weight": torch.randn(d_sae, d_model, generator=g, dtype=dtype),
        "decoder_bias": torch.randn(d_model, generator=g, dtype=dtype),
        "clampable_features": feats,
    }


def _zero_clamps(n_tokens: int, n_clamp: int) -> dict[str, torch.Tensor]:
    return {
        "clamp_kind": torch.zeros(n_tokens, n_clamp, dtype=torch.int8),
        "clamp_value": torch.zeros(n_tokens, n_clamp, dtype=torch.float32),
        "clamp_only_if_active": torch.zeros(n_tokens, n_clamp, dtype=torch.bool),
    }


def _all_recon_mask(n_tokens: int) -> torch.Tensor:
    return torch.ones(n_tokens, dtype=torch.bool)


# ---------------------------------------------------------------------------
# sae_encode_full: activation function correctness
# ---------------------------------------------------------------------------


class TestSaeEncodeFull:
    """Activation function correctness — full-d_sae variant."""

    def test_relu_clamps_negative_to_zero(self):
        h = torch.tensor([[1.0, -1.0]])
        W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_enc = torch.tensor([0.0, 0.0])
        f = sae_encode_full(h, W_enc, b_enc, SAEActivation.RELU, {})
        assert torch.allclose(f, torch.tensor([[1.0, 0.0]]))

    def test_jumprelu_zeros_below_threshold(self):
        h = torch.tensor([[1.0, 0.5]])
        W_enc = torch.eye(2)
        b_enc = torch.zeros(2)
        f = sae_encode_full(h, W_enc, b_enc, SAEActivation.JUMPRELU, {"threshold": 0.7})
        assert torch.allclose(f, torch.tensor([[1.0, 0.0]]))

    def test_topk_keeps_only_largest_full_d_sae(self):
        # The full-reconstruction TopK runs across the *full* d_sae,
        # not a clampable subset — this is the principal semantic
        # difference from sae_steering.sae_encode.
        h = torch.tensor([[3.0, 1.0, 2.0, 0.0]])
        W_enc = torch.eye(4)
        b_enc = torch.zeros(4)
        f = sae_encode_full(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 2})
        # Top-2 by magnitude across all 4 features → indices {0, 2}.
        expected = torch.tensor([[3.0, 0.0, 2.0, 0.0]])
        assert torch.allclose(f, expected)

    def test_topk_ties_use_lowest_feature_indices(self):
        h = torch.zeros(1, 4)
        W_enc = torch.zeros(4, 4)
        b_enc = torch.zeros(4)
        f = sae_encode_full(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 2})
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(f, expected)

        b_enc = torch.ones(4)
        f = sae_encode_full(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 2})
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        assert torch.allclose(f, expected)

    def test_topk_k_equals_d_sae_keeps_all(self):
        h = torch.randn(2, 3)
        W_enc = torch.randn(3, 3)
        b_enc = torch.randn(3)
        f_topk = sae_encode_full(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 3})
        pre_act = h.to(torch.float32) @ W_enc.t().to(torch.float32) + b_enc.to(
            torch.float32
        )
        assert torch.allclose(f_topk, pre_act)

    def test_unsupported_activation_raises(self):
        h = torch.tensor([[1.0]])
        W_enc = torch.tensor([[1.0]])
        b_enc = torch.tensor([0.0])

        class _Bogus:
            value = "bogus"

        with pytest.raises(ValueError, match="Unsupported"):
            sae_encode_full(h, W_enc, b_enc, _Bogus(), {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per-token recon_mask gate
# ---------------------------------------------------------------------------


class TestReconMaskGate:
    """Tokens with ``recon_mask=False`` must pass through unchanged."""

    def test_all_false_returns_input(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(4, 3)
        recon_mask = torch.zeros(4, dtype=torch.bool)
        out = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
            recon_mask=recon_mask,
        )
        assert torch.equal(out, inputs["hidden_states"])

    def test_all_false_skips_encoder_work(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(4, 3)
        recon_mask = torch.zeros(4, dtype=torch.bool)
        with patch(
            "vllm.model_executor.layers.sae_full_reconstruction.sae_encode_full",
            side_effect=AssertionError("encoder should be skipped"),
        ):
            out = apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=recon_mask,
            )
        assert torch.equal(out, inputs["hidden_states"])

    def test_partial_mask_only_modifies_masked_rows(self):
        inputs = _make_random_inputs(n_tokens=4)
        clamps = _zero_clamps(4, 3)
        # Reconstruct only rows 1 and 3.
        recon_mask = torch.tensor([False, True, False, True])
        out = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
            recon_mask=recon_mask,
        )
        # Unmasked rows are bit-identical to input.
        assert torch.equal(out[0], inputs["hidden_states"][0])
        assert torch.equal(out[2], inputs["hidden_states"][2])
        # Masked rows (with no clamps) are still replaced by
        # reconstruction and so generally differ from input.
        assert not torch.allclose(out[1], inputs["hidden_states"][1])
        assert not torch.allclose(out[3], inputs["hidden_states"][3])

    def test_all_true_with_zero_clamps_is_pure_reconstruction(self):
        # No clamps, mask=all-True → output == decode(activate(encode(h))) + b_dec.
        inputs = _make_random_inputs(n_tokens=3, d_model=4, d_sae=6, n_clamp=0)
        clamps = {
            "clamp_kind": torch.zeros(3, 0, dtype=torch.int8),
            "clamp_value": torch.zeros(3, 0, dtype=torch.float32),
            "clamp_only_if_active": torch.zeros(3, 0, dtype=torch.bool),
        }
        out = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
            recon_mask=_all_recon_mask(3),
        )
        # Build the expected reconstruction by hand.
        h_fp32 = inputs["hidden_states"].to(torch.float32)
        f = torch.clamp(
            h_fp32 @ inputs["encoder_weight"].to(torch.float32).t()
            + inputs["encoder_bias"].to(torch.float32),
            min=0.0,
        )
        expected = (
            f.to(inputs["hidden_states"].dtype) @ inputs["decoder_weight"]
            + inputs["decoder_bias"]
        )
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


class TestLayerRouting:
    def test_row_for_other_module_does_not_reconstruct_this_site(self):
        module = nn.Module()
        register_sae_full_recon_buffers(
            module,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="site_b",
            activation=SAEActivation.RELU,
            activation_params={},
            d_sae=2,
            n_clamp=0,
            hidden_size=2,
            max_recon_configs=2,
            clampable_features=torch.zeros(0, dtype=torch.int64),
            dtype=torch.float32,
        )
        register_sae_recon_index_buffer(module, max_tokens=1)
        module.sae_recon_index[0] = 1

        manager = SAEFullReconstructionManager(max_recon_configs=2)
        manager.register_recon_spec(
            123,
            (SAEFullReconstructionSpec(module_name="site_a"),),
            "prefill",
        )
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=module,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="site_b",
            clampable_features=(),
            layer_idx=0,
        )

        hidden = torch.tensor([[1.0, 2.0]])
        out = apply_layer_sae_full_reconstruction(
            module, hidden, SteeringHookPoint.POST_BLOCK
        )

        assert torch.equal(out, hidden)


# ---------------------------------------------------------------------------
# Clamp variants
# ---------------------------------------------------------------------------


class TestApplyFullReconAbsoluteClamp:
    """Absolute clamps must replace ``f[feature]`` before the decoder pass."""

    def test_absolute_clamp_changes_only_target_feature(self):
        # Single token, identity encoder for d_sae=3, decoder = scaled identity.
        # Clamp feature 1 to value 10.0; verify the decoder output is
        # decoder[1] * 10 + decoder_bias.
        d_model, d_sae = 3, 3
        h = torch.tensor([[1.0, 2.0, 3.0]])
        W_enc = torch.eye(d_sae)  # f = ReLU(h)
        b_enc = torch.zeros(d_sae)
        # Decoder rows are e_x, e_y, e_z (identity); decoder_bias = 0.
        W_dec = torch.eye(d_sae)
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([1], dtype=torch.int64)
        clamp_kind = torch.tensor([[CLAMP_KIND_ABSOLUTE]], dtype=torch.int8)
        clamp_value = torch.tensor([[10.0]])
        only = torch.tensor([[False]])
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=_all_recon_mask(1),
        )
        # f after clamp = [1, 10, 3]; decoder is identity → out = [1, 10, 3].
        assert torch.allclose(out, torch.tensor([[1.0, 10.0, 3.0]]))

    def test_absolute_clamp_with_inactive_only_if_active_skips(self):
        # f=0 (h drives pre_act ≤ 0); only_if_active=True suppresses
        # the clamp; output is the pure reconstruction (f stays 0).
        d_model, d_sae = 2, 2
        h = torch.tensor([[-1.0, 0.0]])
        W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_enc = torch.zeros(d_sae)
        W_dec = torch.eye(d_sae)
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.tensor([[CLAMP_KIND_ABSOLUTE]], dtype=torch.int8)
        clamp_value = torch.tensor([[10.0]])
        only = torch.tensor([[True]])
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=_all_recon_mask(1),
        )
        # Both f are 0 (ReLU of negative or 0); decoder gives [0, 0].
        assert torch.allclose(out, torch.zeros(1, 2))


class TestApplyFullReconAdditiveClamp:
    """Additive clamps must shift ``f[feature]`` by ``value``."""

    def test_additive_clamp_shifts_target_feature(self):
        d_model, d_sae = 2, 2
        h = torch.tensor([[2.0, 0.0]])
        W_enc = torch.eye(d_sae)
        b_enc = torch.zeros(d_sae)
        W_dec = torch.eye(d_sae)
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.tensor([[CLAMP_KIND_ADDITIVE]], dtype=torch.int8)
        clamp_value = torch.tensor([[3.0]])
        only = torch.tensor([[False]])
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=_all_recon_mask(1),
        )
        # f = [2, 0]; additive +3 on feature 0 → [5, 0]; decoder identity → [5, 0].
        assert torch.allclose(out, torch.tensor([[5.0, 0.0]]))

    def test_additive_with_only_if_active_zero_below(self):
        d_model, d_sae = 2, 2
        h = torch.tensor([[-1.0, 0.0]])
        W_enc = torch.eye(d_sae)
        b_enc = torch.zeros(d_sae)
        W_dec = torch.eye(d_sae)
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.tensor([[CLAMP_KIND_ADDITIVE]], dtype=torch.int8)
        clamp_value = torch.tensor([[3.0]])
        only = torch.tensor([[True]])
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=_all_recon_mask(1),
        )
        # f = [0, 0] (ReLU of [-1, 0]); only_if_active suppresses clamp;
        # decoder identity → [0, 0].
        assert torch.allclose(out, torch.zeros(1, 2))

    def test_topk_only_if_active_allows_selected_negative_feature(self):
        d_model, d_sae = 2, 2
        h = torch.tensor([[-2.0, -1.0]])
        W_enc = torch.eye(d_sae)
        b_enc = torch.zeros(d_sae)
        W_dec = torch.eye(d_sae)
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.tensor([[CLAMP_KIND_ADDITIVE]], dtype=torch.int8)
        clamp_value = torch.tensor([[3.0]])
        only = torch.tensor([[True]])

        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.TOPK,
            activation_params={"k": 2},
            clampable_features=feats,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=_all_recon_mask(1),
        )

        # TopK keeps both negative pre-activations; feature 0 is selected,
        # so the gated additive clamp applies before full reconstruction.
        assert torch.allclose(out, torch.tensor([[1.0, -1.0]]))


# ---------------------------------------------------------------------------
# Reference parity for random inputs
# ---------------------------------------------------------------------------


class TestApplyFullReconMatchesReference:
    """Vectorized op must match the per-(token, feature) reference."""

    @pytest.mark.parametrize(
        "activation,params",
        [
            (SAEActivation.RELU, {}),
            (SAEActivation.JUMPRELU, {"threshold": 0.5}),
            (SAEActivation.TOPK, {"k": 4}),
        ],
    )
    def test_random_inputs_match_reference(self, activation, params):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 5, 6, 12, 3
        inputs = _make_random_inputs(
            n_tokens=n_tokens,
            d_model=d_model,
            d_sae=d_sae,
            n_clamp=n_clamp,
            seed=42,
        )
        rng = torch.Generator(device="cpu").manual_seed(7)
        clamp_kind = torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        )
        clamp_value = torch.randn(n_tokens, n_clamp, generator=rng)
        only = torch.randint(0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool)
        recon_mask = torch.randint(0, 2, (n_tokens,), generator=rng, dtype=torch.bool)
        ref = _ref_apply_full_reconstruction(
            **inputs,
            activation=activation,
            activation_params=params,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=recon_mask,
        )
        got = apply_sae_full_reconstruction(
            **inputs,
            activation=activation,
            activation_params=params,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=recon_mask,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_per_token_clamp_independence(self):
        # Token A has an absolute clamp; token B has none.  Token B
        # is also masked off (recon_mask=False), so it must pass
        # through bit-identically regardless of A's clamp.
        torch.manual_seed(0)
        inputs = _make_random_inputs(n_tokens=2, d_model=4, d_sae=6, n_clamp=2, seed=11)
        clamp_kind = torch.tensor(
            [
                [CLAMP_KIND_ABSOLUTE, CLAMP_KIND_NONE],
                [CLAMP_KIND_NONE, CLAMP_KIND_NONE],
            ],
            dtype=torch.int8,
        )
        clamp_value = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
        only = torch.zeros(2, 2, dtype=torch.bool)
        recon_mask = torch.tensor([True, False])
        out = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=recon_mask,
        )
        assert torch.equal(out[1], inputs["hidden_states"][1])


# ---------------------------------------------------------------------------
# Dtype + shape / contract
# ---------------------------------------------------------------------------


class TestApplyFullReconDtype:
    """Numeric dtype contract: GEMMs in compute dtype, activation/clamp in fp32."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_input_matches_reference(self, dtype):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 4, 6, 10, 3
        inputs = _make_random_inputs(
            n_tokens=n_tokens,
            d_model=d_model,
            d_sae=d_sae,
            n_clamp=n_clamp,
            dtype=dtype,
            seed=99,
        )
        rng = torch.Generator(device="cpu").manual_seed(101)
        clamp_kind = torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        )
        clamp_value = torch.randn(n_tokens, n_clamp, generator=rng)
        only = torch.randint(0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool)
        recon_mask = _all_recon_mask(n_tokens)
        ref = _ref_apply_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=recon_mask,
        )
        got = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only,
            recon_mask=recon_mask,
        )
        atol = 5e-2 if dtype is torch.bfloat16 else 1e-2
        assert got.dtype is dtype
        assert torch.allclose(got.float(), ref.float(), atol=atol, rtol=atol)

    def test_output_dtype_matches_input(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            inputs = _make_random_inputs(dtype=dtype)
            clamps = _zero_clamps(4, 3)
            out = apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=_all_recon_mask(4),
            )
            assert out.dtype is dtype


class TestApplyFullReconShapes:
    """Shape and contract validation."""

    def test_rejects_mismatched_d_model_encoder(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(8, 5)  # wrong d_model
        b_enc = torch.randn(8)
        W_dec = torch.randn(8, 4)
        b_dec = torch.randn(4)
        feats = torch.tensor([0, 1], dtype=torch.int64)
        clamps = _zero_clamps(2, 2)
        with pytest.raises(ValueError, match="encoder_weight"):
            apply_sae_full_reconstruction(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                decoder_bias=b_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=feats,
                **clamps,
                recon_mask=_all_recon_mask(2),
            )

    def test_rejects_mismatched_decoder_shape(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(8, 4)
        b_enc = torch.randn(8)
        W_dec = torch.randn(7, 4)  # wrong d_sae
        b_dec = torch.randn(4)
        feats = torch.tensor([0, 1], dtype=torch.int64)
        clamps = _zero_clamps(2, 2)
        with pytest.raises(ValueError, match="decoder_weight"):
            apply_sae_full_reconstruction(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                decoder_bias=b_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=feats,
                **clamps,
                recon_mask=_all_recon_mask(2),
            )

    def test_rejects_mismatched_decoder_bias(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(8, 4)
        b_enc = torch.randn(8)
        W_dec = torch.randn(8, 4)
        b_dec = torch.randn(5)  # wrong d_model
        feats = torch.tensor([0], dtype=torch.int64)
        clamps = _zero_clamps(2, 1)
        with pytest.raises(ValueError, match="decoder_bias"):
            apply_sae_full_reconstruction(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                decoder_bias=b_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=feats,
                **clamps,
                recon_mask=_all_recon_mask(2),
            )

    def test_rejects_clampable_features_wrong_dtype(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(4, 3)
        bad_feats = inputs["clampable_features"].to(torch.int32)
        with pytest.raises(ValueError, match="int64"):
            apply_sae_full_reconstruction(
                hidden_states=inputs["hidden_states"],
                encoder_weight=inputs["encoder_weight"],
                encoder_bias=inputs["encoder_bias"],
                decoder_weight=inputs["decoder_weight"],
                decoder_bias=inputs["decoder_bias"],
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=bad_feats,
                **clamps,
                recon_mask=_all_recon_mask(4),
            )

    def test_rejects_clampable_features_out_of_range(self):
        inputs = _make_random_inputs(d_sae=8, n_clamp=2)
        bad_feats = torch.tensor([0, 99], dtype=torch.int64)
        clamps = _zero_clamps(4, 2)
        with pytest.raises(ValueError, match=r"\[0, d_sae"):
            apply_sae_full_reconstruction(
                hidden_states=inputs["hidden_states"],
                encoder_weight=inputs["encoder_weight"],
                encoder_bias=inputs["encoder_bias"],
                decoder_weight=inputs["decoder_weight"],
                decoder_bias=inputs["decoder_bias"],
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=bad_feats,
                **clamps,
                recon_mask=torch.zeros(4, dtype=torch.bool),
            )

    def test_rejects_duplicate_clampable_features(self):
        inputs = _make_random_inputs(d_sae=8, n_clamp=2)
        bad_feats = torch.tensor([1, 1], dtype=torch.int64)
        clamps = _zero_clamps(4, 2)
        with pytest.raises(ValueError, match="duplicates"):
            apply_sae_full_reconstruction(
                hidden_states=inputs["hidden_states"],
                encoder_weight=inputs["encoder_weight"],
                encoder_bias=inputs["encoder_bias"],
                decoder_weight=inputs["decoder_weight"],
                decoder_bias=inputs["decoder_bias"],
                activation=SAEActivation.RELU,
                activation_params={},
                clampable_features=bad_feats,
                **clamps,
                recon_mask=torch.zeros(4, dtype=torch.bool),
            )

    def test_rejects_wrong_clamp_dtypes(self):
        inputs = _make_random_inputs(n_tokens=4, n_clamp=3)
        clamps = _zero_clamps(4, 3)
        cases = [
            ("clamp_kind", clamps["clamp_kind"].to(torch.int16), "int8"),
            ("clamp_value", clamps["clamp_value"].to(torch.int32), "floating"),
            (
                "clamp_only_if_active",
                clamps["clamp_only_if_active"].to(torch.int8),
                "bool",
            ),
        ]
        for field_name, bad_tensor, match in cases:
            bad_clamps = dict(clamps)
            bad_clamps[field_name] = bad_tensor
            with pytest.raises(ValueError, match=match):
                apply_sae_full_reconstruction(
                    **inputs,
                    activation=SAEActivation.RELU,
                    activation_params={},
                    **bad_clamps,
                    recon_mask=_all_recon_mask(4),
                )

    def test_rejects_unknown_clamp_kind_value(self):
        inputs = _make_random_inputs(n_tokens=4, n_clamp=3)
        clamps = _zero_clamps(4, 3)
        clamps["clamp_kind"][0, 0] = 99
        with pytest.raises(ValueError, match="clamp_kind entries"):
            apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=_all_recon_mask(4),
            )

    def test_rejects_recon_mask_wrong_dtype(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(4, 3)
        with pytest.raises(ValueError, match="bool"):
            apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=torch.zeros(4, dtype=torch.int8),
            )

    def test_rejects_recon_mask_wrong_shape(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(4, 3)
        with pytest.raises(ValueError, match="recon_mask"):
            apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=torch.zeros(7, dtype=torch.bool),
            )

    def test_rejects_clamp_shape_mismatch(self):
        inputs = _make_random_inputs(n_tokens=4, n_clamp=3)
        # Build clamps with mismatched n_tokens.
        clamps = _zero_clamps(5, 3)
        with pytest.raises(ValueError, match="clamp_kind"):
            apply_sae_full_reconstruction(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=_all_recon_mask(4),
            )

    def test_empty_n_clamp_with_recon_mask_true(self):
        # n_clamp=0: no clampable features. The reconstruction is
        # just decode(activate(encode(h))).  Output must equal that.
        n_tokens, d_model, d_sae = 3, 4, 5
        torch.manual_seed(0)
        h = torch.randn(n_tokens, d_model)
        W_enc = torch.randn(d_sae, d_model)
        b_enc = torch.randn(d_sae)
        W_dec = torch.randn(d_sae, d_model)
        b_dec = torch.randn(d_model)
        feats = torch.zeros(0, dtype=torch.int64)
        clamps = {
            "clamp_kind": torch.zeros(n_tokens, 0, dtype=torch.int8),
            "clamp_value": torch.zeros(n_tokens, 0, dtype=torch.float32),
            "clamp_only_if_active": torch.zeros(n_tokens, 0, dtype=torch.bool),
        }
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            **clamps,
            recon_mask=_all_recon_mask(n_tokens),
        )
        # Build expected reconstruction.
        h_fp32 = h.to(torch.float32)
        f = torch.clamp(h_fp32 @ W_enc.float().t() + b_enc.float(), min=0.0)
        expected = f @ W_dec + b_dec
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)

    def test_empty_n_tokens_returns_empty(self):
        h = torch.zeros(0, 4)
        W_enc = torch.zeros(8, 4)
        b_enc = torch.zeros(8)
        W_dec = torch.zeros(8, 4)
        b_dec = torch.zeros(4)
        feats = torch.tensor([0], dtype=torch.int64)
        clamps = {
            "clamp_kind": torch.zeros(0, 1, dtype=torch.int8),
            "clamp_value": torch.zeros(0, 1, dtype=torch.float32),
            "clamp_only_if_active": torch.zeros(0, 1, dtype=torch.bool),
        }
        out = apply_sae_full_reconstruction(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            decoder_bias=b_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clampable_features=feats,
            **clamps,
            recon_mask=torch.zeros(0, dtype=torch.bool),
        )
        assert out.shape == (0, 4)
