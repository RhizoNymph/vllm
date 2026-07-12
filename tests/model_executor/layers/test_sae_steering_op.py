# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric tests for the eager SAE feature-surgery delta op.

The op math is:

    pre_act_i  = W_enc[i, :] · h + b_enc[i]
    f_i        = activation(pre_act_i)
    new_f_i    = clamp_kind == ABSOLUTE  ? clamp_value
                 clamp_kind == ADDITIVE  ? f_i + clamp_value
                 (otherwise)             : f_i
    delta_i    = (new_f_i - f_i) gated by only_if_active when set
    h_new      = h + Σ_i delta_i · W_dec[i, :]

These tests verify a fully-vectorized eager implementation against a
hand-rolled per-(token, feature) reference loop, plus the activation
functions and clamp variants individually.  Phase 1 implements only
the math primitive; layer-hook integration and the Triton swap belong
to Phase 1B / Phase 2.
"""

from __future__ import annotations

import pytest
import torch

import vllm.model_executor.layers.sae_steering as sae_steering
from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
    CLAMP_KIND_NONE,
    apply_sae_delta,
    sae_encode,
)


def _ref_apply_sae_delta(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    activation: SAEActivation,
    activation_params: dict[str, float],
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    threshold: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hand-rolled per-(token, feature) reference.

    Uses Python loops on purpose so the implementation under test can
    be verified against a transparent definition.  Promotes to fp32 in
    the same places the contract demands so a pure-fp32 expectation
    cannot mask a bug in the production code's promotion seams.
    JumpReLU uses a per-feature ``threshold[i]``.
    """
    n_tokens = hidden_states.shape[0]
    n_clamp = encoder_weight.shape[0]
    out = hidden_states.clone()
    h_fp32 = hidden_states.to(torch.float32)
    enc_w_fp32 = encoder_weight.to(torch.float32)
    enc_b_fp32 = encoder_bias.to(torch.float32)
    dec_w_compute = decoder_weight.to(hidden_states.dtype)
    for t in range(n_tokens):
        for i in range(n_clamp):
            pre_act = float(h_fp32[t] @ enc_w_fp32[i] + enc_b_fp32[i])
            if activation is SAEActivation.RELU:
                f = max(0.0, pre_act)
            elif activation is SAEActivation.JUMPRELU:
                assert threshold is not None
                thr_i = float(threshold[i])
                f = pre_act if pre_act > thr_i else 0.0
            elif activation is SAEActivation.TOPK:
                # TopK among the clampable subset for this token.
                k = int(activation_params["k"])
                acts = []
                for j in range(n_clamp):
                    pj = float(h_fp32[t] @ enc_w_fp32[j] + enc_b_fp32[j])
                    acts.append(pj)
                top_indices = sorted(
                    range(n_clamp), key=lambda j: acts[j], reverse=True
                )[:k]
                f = acts[i] if i in top_indices else 0.0
            else:
                raise AssertionError(f"unhandled activation {activation}")

            kind = int(clamp_kind[t, i])
            if kind == CLAMP_KIND_NONE:
                continue
            v = float(clamp_value[t, i])
            gated = bool(clamp_only_if_active[t, i])
            active = f != 0.0 if activation is SAEActivation.TOPK else f > 0.0
            if gated and not active:
                continue
            if kind == CLAMP_KIND_ABSOLUTE:
                new_f = v
            elif kind == CLAMP_KIND_ADDITIVE:
                new_f = f + v
            else:
                raise AssertionError(f"unhandled clamp kind {kind}")
            delta = new_f - f
            out[t] = out[t] + torch.tensor(delta, dtype=out.dtype) * dec_w_compute[i]
    return out


def _make_random_inputs(
    *,
    n_tokens: int = 4,
    d_model: int = 8,
    n_clamp: int = 3,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(n_tokens, d_model, generator=g, dtype=dtype),
        "encoder_weight": torch.randn(n_clamp, d_model, generator=g, dtype=dtype),
        "encoder_bias": torch.randn(n_clamp, generator=g, dtype=dtype),
        "decoder_weight": torch.randn(n_clamp, d_model, generator=g, dtype=dtype),
    }


def _zero_clamps(n_tokens: int, n_clamp: int) -> dict[str, torch.Tensor]:
    return {
        "clamp_kind": torch.zeros(n_tokens, n_clamp, dtype=torch.int8),
        "clamp_value": torch.zeros(n_tokens, n_clamp, dtype=torch.float32),
        "clamp_only_if_active": torch.zeros(n_tokens, n_clamp, dtype=torch.bool),
    }


class TestSaeEncode:
    """Activation function correctness."""

    def test_relu_clamps_negative_to_zero(self):
        h = torch.tensor([[1.0, -1.0]])
        W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # picks h[0], h[1]
        b_enc = torch.tensor([0.0, 0.0])
        f = sae_encode(h, W_enc, b_enc, SAEActivation.RELU, {})
        assert torch.allclose(f, torch.tensor([[1.0, 0.0]]))

    def test_jumprelu_zeros_below_threshold(self):
        h = torch.tensor([[1.0, 0.5]])
        W_enc = torch.eye(2)
        b_enc = torch.zeros(2)
        thr = torch.full((2,), 0.7)
        f = sae_encode(h, W_enc, b_enc, SAEActivation.JUMPRELU, {}, threshold=thr)
        # 1.0 > 0.7 -> kept; 0.5 <= 0.7 -> zeroed.
        assert torch.allclose(f, torch.tensor([[1.0, 0.0]]))

    def test_jumprelu_per_feature_thresholds(self):
        # Same pre-activation for both features; only their thresholds
        # differ, so a per-feature comparison must gate them apart.
        h = torch.tensor([[0.5, 0.5]])
        W_enc = torch.eye(2)
        b_enc = torch.zeros(2)
        thr = torch.tensor([0.2, 0.8])
        f = sae_encode(h, W_enc, b_enc, SAEActivation.JUMPRELU, {}, threshold=thr)
        # 0.5 > 0.2 -> kept; 0.5 <= 0.8 -> zeroed.
        assert torch.allclose(f, torch.tensor([[0.5, 0.0]]))

    def test_jumprelu_negative_pre_act_is_zeroed(self):
        h = torch.tensor([[-2.0]])
        W_enc = torch.tensor([[1.0]])
        b_enc = torch.tensor([0.0])
        thr = torch.zeros(1)
        f = sae_encode(h, W_enc, b_enc, SAEActivation.JUMPRELU, {}, threshold=thr)
        # -2.0 <= 0.0 -> zeroed.
        assert torch.allclose(f, torch.tensor([[0.0]]))

    def test_topk_keeps_only_largest(self):
        # Two tokens, four features, k=2.
        # pre_acts row 0 = [3, 1, 2, 0]; top-2 indices = {0, 2}.
        # pre_acts row 1 = [-1, 4, 0, 5]; top-2 indices = {1, 3}.
        h = torch.tensor([[3.0, 1.0, 2.0, 0.0], [-1.0, 4.0, 0.0, 5.0]])
        W_enc = torch.eye(4)
        b_enc = torch.zeros(4)
        f = sae_encode(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 2.0})
        expected = torch.tensor([[3.0, 0.0, 2.0, 0.0], [0.0, 4.0, 0.0, 5.0]])
        assert torch.allclose(f, expected)

    def test_topk_ties_use_lowest_feature_indices(self):
        h = torch.zeros(1, 4)
        W_enc = torch.zeros(4, 4)
        b_enc = torch.ones(4)
        f = sae_encode(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 2.0})
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        assert torch.allclose(f, expected)

    def test_topk_k_equals_n_clamp_keeps_all(self):
        h = torch.randn(2, 3)
        W_enc = torch.randn(3, 3)
        b_enc = torch.randn(3)
        # k = n_clamp -> all features kept (activation is identity).
        f_topk = sae_encode(h, W_enc, b_enc, SAEActivation.TOPK, {"k": 3.0})
        # Identity reference: pre_act with no zeroing.
        pre_act = h.to(torch.float32) @ W_enc.t().to(torch.float32) + b_enc.to(
            torch.float32
        )
        assert torch.allclose(f_topk, pre_act)

    def test_jumprelu_requires_threshold(self):
        h = torch.tensor([[1.0]])
        W_enc = torch.tensor([[1.0]])
        b_enc = torch.tensor([0.0])
        with pytest.raises(ValueError, match="threshold"):
            sae_encode(h, W_enc, b_enc, SAEActivation.JUMPRELU, {})

    def test_topk_requires_k(self):
        h = torch.tensor([[1.0]])
        W_enc = torch.tensor([[1.0]])
        b_enc = torch.tensor([0.0])
        with pytest.raises(KeyError):
            sae_encode(h, W_enc, b_enc, SAEActivation.TOPK, {})


class TestApplySaeDeltaNoOp:
    """When no clamp is active the residual is returned unchanged."""

    def test_all_kind_none_returns_input(self):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(n_tokens=4, n_clamp=3)
        out = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
        )
        assert torch.equal(out, inputs["hidden_states"])

    def test_any_active_false_returns_input(self):
        inputs = _make_random_inputs(n_tokens=4, d_model=6, n_clamp=3, seed=12)
        clamp_kind = torch.full((4, 3), CLAMP_KIND_ABSOLUTE, dtype=torch.int8)
        clamp_value = torch.full((4, 3), 7.0)
        only_if_active = torch.zeros(4, 3, dtype=torch.bool)

        out = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
            any_active=torch.zeros(1, dtype=torch.bool),
        )

        assert torch.equal(out, inputs["hidden_states"])

    def test_all_kind_none_skips_encoder(self, monkeypatch):
        inputs = _make_random_inputs()
        clamps = _zero_clamps(n_tokens=4, n_clamp=3)
        calls = {"n": 0}
        original = sae_steering.sae_encode

        def counting_encode(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(sae_steering, "sae_encode", counting_encode)

        out = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
        )
        assert torch.equal(out, inputs["hidden_states"])
        assert calls["n"] == 0

    def test_only_if_active_with_dead_feature_is_no_op(self):
        # Construct an h that drives every clampable pre_act below zero,
        # so ReLU yields f=0 and only_if_active gates everything off.
        n_tokens, d_model, n_clamp = 2, 4, 3
        h = torch.zeros(n_tokens, d_model)
        W_enc = torch.zeros(n_clamp, d_model)
        b_enc = torch.full((n_clamp,), -1.0)
        W_dec = torch.randn(n_clamp, d_model)
        clamp_kind = torch.full(
            (n_tokens, n_clamp), CLAMP_KIND_ABSOLUTE, dtype=torch.int8
        )
        clamp_value = torch.full((n_tokens, n_clamp), 5.0)
        only_if_active = torch.ones(n_tokens, n_clamp, dtype=torch.bool)
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        assert torch.equal(out, h)

    def test_topk_only_if_active_allows_selected_negative_feature(self):
        h = torch.tensor([[-2.0, -1.0]])
        W_enc = torch.eye(2)
        b_enc = torch.zeros(2)
        W_dec = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        clamp_kind = torch.tensor(
            [[CLAMP_KIND_ADDITIVE, CLAMP_KIND_NONE]], dtype=torch.int8
        )
        clamp_value = torch.tensor([[3.0, 0.0]])
        only_if_active = torch.tensor([[True, False]])

        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.TOPK,
            activation_params={"k": 2},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )

        # TopK keeps both negative pre-activations. Feature 0 is active
        # because it was selected, so the gated additive clamp applies.
        expected = torch.tensor([[-2.0, 2.0]])
        torch.testing.assert_close(out, expected)


class TestApplySaeDeltaAbsolute:
    """Absolute clamps replace f_i with the target value."""

    def test_absolute_clamp_single_feature(self):
        # Single token, single feature, identity encoder: f = ReLU(h[0]).
        # We clamp f := 7.0; delta = 7.0 - f.  W_dec = unit vector → h
        # gains exactly (7.0 - f) in that direction.
        h = torch.tensor([[2.0, 0.0, 0.0]])
        W_enc = torch.tensor([[1.0, 0.0, 0.0]])
        b_enc = torch.tensor([0.0])
        W_dec = torch.tensor([[0.0, 1.0, 0.0]])
        clamp_kind = torch.tensor([[CLAMP_KIND_ABSOLUTE]], dtype=torch.int8)
        clamp_value = torch.tensor([[7.0]])
        only_if_active = torch.tensor([[False]])
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        # f = ReLU(2.0) = 2.0; delta = 5.0 along W_dec[0] = e_y.
        expected = torch.tensor([[2.0, 5.0, 0.0]])
        assert torch.allclose(out, expected)

    def test_absolute_clamp_with_inactive_only_if_active_skips(self):
        # f = 0; absolute clamp with only_if_active should NOT apply.
        h = torch.tensor([[-1.0, 0.0]])
        W_enc = torch.tensor([[1.0, 0.0]])
        b_enc = torch.tensor([0.0])
        W_dec = torch.tensor([[0.0, 1.0]])
        clamp_kind = torch.tensor([[CLAMP_KIND_ABSOLUTE]], dtype=torch.int8)
        clamp_value = torch.tensor([[10.0]])
        only_if_active = torch.tensor([[True]])
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        assert torch.equal(out, h)


class TestApplySaeDeltaAdditive:
    """Additive clamps shift f_i by a constant; delta is W_dec[i] · value."""

    def test_additive_clamp_independent_of_f(self):
        # Additive without only_if_active does NOT depend on f.  Two
        # configurations of h that produce different f values should
        # produce the same delta.
        W_enc = torch.tensor([[1.0, 0.0, 0.0]])
        b_enc = torch.tensor([0.0])
        W_dec = torch.tensor([[0.0, 1.0, 0.0]])
        clamp_kind = torch.tensor([[CLAMP_KIND_ADDITIVE]], dtype=torch.int8)
        clamp_value = torch.tensor([[3.0]])
        only_if_active = torch.tensor([[False]])

        h_a = torch.tensor([[2.0, 0.0, 0.0]])  # f = 2.0
        h_b = torch.tensor([[5.0, 0.0, 0.0]])  # f = 5.0

        out_a = apply_sae_delta(
            hidden_states=h_a,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        out_b = apply_sae_delta(
            hidden_states=h_b,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        delta_a = out_a - h_a
        delta_b = out_b - h_b
        assert torch.allclose(delta_a, delta_b)
        assert torch.allclose(delta_a, torch.tensor([[0.0, 3.0, 0.0]]))

    def test_additive_with_only_if_active_zero_below(self):
        h = torch.tensor([[-1.0, 0.0]])  # f = ReLU(-1) = 0
        W_enc = torch.tensor([[1.0, 0.0]])
        b_enc = torch.tensor([0.0])
        W_dec = torch.tensor([[0.0, 1.0]])
        clamp_kind = torch.tensor([[CLAMP_KIND_ADDITIVE]], dtype=torch.int8)
        clamp_value = torch.tensor([[3.0]])
        only_if_active = torch.tensor([[True]])
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        # Inactive ⇒ no delta.
        assert torch.equal(out, h)


class TestApplySaeDeltaMatchesReference:
    """Vectorized op must match the per-(token, feature) reference."""

    @pytest.mark.parametrize(
        "activation,params,threshold_value",
        [
            (SAEActivation.RELU, {}, None),
            (SAEActivation.JUMPRELU, {}, 0.5),
            (SAEActivation.TOPK, {"k": 2.0}, None),
        ],
    )
    def test_random_inputs_match_reference(self, activation, params, threshold_value):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 5, 7, 4
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=42
        )
        threshold = (
            torch.full((n_clamp,), threshold_value)
            if threshold_value is not None
            else None
        )
        # Mix of all three clamp kinds plus inactive entries.
        rng = torch.Generator().manual_seed(7)
        clamp_kind = torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        )
        clamp_value = torch.randn(n_tokens, n_clamp, generator=rng)
        only_if_active = torch.randint(
            0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
        )

        ref = _ref_apply_sae_delta(
            **inputs,
            activation=activation,
            activation_params=params,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
            threshold=threshold,
        )
        got = apply_sae_delta(
            **inputs,
            activation=activation,
            activation_params=params,
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
            threshold=threshold,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_jumprelu_per_feature_thresholds_match_reference(self):
        """Random non-constant threshold vector must match the reference."""
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 6, 7, 5
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=13
        )
        rng = torch.Generator().manual_seed(29)
        threshold = torch.rand(n_clamp, generator=rng) * 2.0 - 1.0
        clamp_kind = torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        )
        clamp_value = torch.randn(n_tokens, n_clamp, generator=rng)
        only_if_active = torch.randint(
            0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
        )

        ref = _ref_apply_sae_delta(
            **inputs,
            activation=SAEActivation.JUMPRELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
            threshold=threshold,
        )
        got = apply_sae_delta(
            **inputs,
            activation=SAEActivation.JUMPRELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
            threshold=threshold,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_jumprelu_per_feature_differs_from_median_scalar(self):
        """Per-feature thresholds must be observably different from the
        old behaviour of folding the vector to its median.

        Feature 0 has a low threshold (feature live), feature 1 a high
        one (feature dead); the median-scalar fold would gate both
        identically and — with ``only_if_active`` clamps — produce a
        provably different output.
        """
        h = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        W_enc = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        b_enc = torch.zeros(2)
        W_dec = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        per_feature = torch.tensor([0.5, 1.5])
        # The old fold used np.median (interpolating) → 1.0.
        median_scalar = torch.full((2,), 1.0)
        clamp_kind = torch.full((1, 2), CLAMP_KIND_ADDITIVE, dtype=torch.int8)
        clamp_value = torch.full((1, 2), 3.0)
        only_if_active = torch.ones(1, 2, dtype=torch.bool)

        common = dict(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.JUMPRELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        got = apply_sae_delta(**common, threshold=per_feature)
        got_median = apply_sae_delta(**common, threshold=median_scalar)

        # Per-feature: pre_act 1.0 > 0.5 → feature 0 live, gated clamp
        # fires (+3 along e_z); 1.0 <= 1.5 → feature 1 dead, clamp
        # suppressed.  Median scalar 1.0 would gate both features dead.
        expected = torch.tensor([[1.0, 1.0, 3.0, 0.0]])
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(got_median, h)
        assert not torch.allclose(got, got_median)

    def test_per_token_clamp_independence(self):
        # Token A has an absolute clamp; token B has none. The op must
        # leave B exactly unchanged regardless of A's clamp.
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 2, 4, 2
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=11
        )
        clamp_kind = torch.tensor(
            [
                [CLAMP_KIND_ABSOLUTE, CLAMP_KIND_NONE],
                [CLAMP_KIND_NONE, CLAMP_KIND_NONE],
            ],
            dtype=torch.int8,
        )
        clamp_value = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
        only_if_active = torch.zeros(n_tokens, n_clamp, dtype=torch.bool)
        out = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        assert torch.equal(out[1], inputs["hidden_states"][1])


class TestApplySaeDeltaDtype:
    """Numeric dtype contract: GEMMs in compute dtype, activation/clamp in fp32."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_input_matches_reference(self, dtype):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 4, 6, 3
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, dtype=dtype, seed=99
        )
        rng = torch.Generator().manual_seed(101)
        clamp_kind = torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        )
        clamp_value = torch.randn(n_tokens, n_clamp, generator=rng)
        only_if_active = torch.randint(
            0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
        )
        ref = _ref_apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        got = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        # bf16/fp16 tolerances are necessarily looser than fp32.
        atol = 1e-2 if dtype is torch.bfloat16 else 5e-3
        assert got.dtype is dtype
        assert torch.allclose(got.float(), ref.float(), atol=atol, rtol=atol)

    def test_output_dtype_matches_input(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            inputs = _make_random_inputs(dtype=dtype)
            clamps = _zero_clamps(4, 3)
            out = apply_sae_delta(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
            )
            assert out.dtype is dtype


class TestApplySaeDeltaShapes:
    """Shape and contract validation."""

    def test_rejects_mismatched_d_model(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(3, 5)  # d_model mismatch
        b_enc = torch.randn(3)
        W_dec = torch.randn(3, 4)
        clamps = _zero_clamps(2, 3)
        with pytest.raises((RuntimeError, ValueError)):
            apply_sae_delta(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
            )

    def test_rejects_mismatched_n_clamp_decoder(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(3, 4)
        b_enc = torch.randn(3)
        W_dec = torch.randn(2, 4)  # rows must equal n_clamp=3
        clamps = _zero_clamps(2, 3)
        with pytest.raises((RuntimeError, ValueError)):
            apply_sae_delta(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
            )

    def test_rejects_mismatched_clamp_kind_shape(self):
        h = torch.randn(2, 4)
        W_enc = torch.randn(3, 4)
        b_enc = torch.randn(3)
        W_dec = torch.randn(3, 4)
        clamps = _zero_clamps(3, 3)  # n_tokens mismatch
        with pytest.raises((RuntimeError, ValueError)):
            apply_sae_delta(
                hidden_states=h,
                encoder_weight=W_enc,
                encoder_bias=b_enc,
                decoder_weight=W_dec,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
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
                apply_sae_delta(
                    **inputs,
                    activation=SAEActivation.RELU,
                    activation_params={},
                    **bad_clamps,
                )

    def test_rejects_unknown_clamp_kind_value(self):
        inputs = _make_random_inputs(n_tokens=4, n_clamp=3)
        clamps = _zero_clamps(4, 3)
        clamps["clamp_kind"][0, 0] = 99
        with pytest.raises(ValueError, match="clamp_kind entries"):
            apply_sae_delta(
                **inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
            )

    def test_empty_n_clamp_is_a_no_op(self):
        # n_clamp=0: registry case where no features are clampable.
        # The op should still run and return hidden_states unchanged.
        h = torch.randn(3, 5)
        W_enc = torch.zeros(0, 5)
        b_enc = torch.zeros(0)
        W_dec = torch.zeros(0, 5)
        clamp_kind = torch.zeros(3, 0, dtype=torch.int8)
        clamp_value = torch.zeros(3, 0)
        only_if_active = torch.zeros(3, 0, dtype=torch.bool)
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=clamp_kind,
            clamp_value=clamp_value,
            clamp_only_if_active=only_if_active,
        )
        assert torch.equal(out, h)
