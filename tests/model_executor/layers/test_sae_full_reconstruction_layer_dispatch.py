# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``apply_layer_sae_full_reconstruction`` (dispatch shim).

The shim:

* Short-circuits when no full-reconstruction buffers are attached
  for the requested hook point — the disabled-mode path emits zero
  kernel work so engines without full-reconstruction stay free of
  overhead.
* Pulls every per-(layer, hook) buffer plus the shared
  ``sae_recon_index`` and gathers per-token clamp tensors from the
  row tables.
* Derives ``recon_mask`` from this site's active-row table so row 0
  (never marked active) is the no-reconstruction sentinel and rows
  owned by other modules' sites pass through bit-identically.
* Dispatches via ``torch.ops.vllm.apply_sae_full_reconstruction``
  (the registered custom op) so :mod:`torch.compile` treats the
  call as an opaque splitting point.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_FR_CLAMP_VALUE_ATTR,
    HOOK_POINT_FR_DECODER_BIAS_ATTR,
    HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ROW_ACTIVE_ATTR,
    apply_layer_sae_full_reconstruction,
    register_sae_full_recon_buffers,
    register_sae_recon_index_buffer,
)
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
)
from vllm.model_executor.layers.steering import SteeringHookPoint


def _attach_identity_sae(
    layer: nn.Module,
    *,
    hook_point: SteeringHookPoint,
    d_model: int,
    d_sae: int,
    clampable_features: list[int],
    activation: SAEActivation = SAEActivation.RELU,
    activation_params: dict[str, float] | None = None,
    max_tokens: int = 8,
    max_recon_configs: int = 4,
) -> None:
    """Set up an SAE site whose decoder bias = -h, encoder = scaled identity.

    This makes the eager reconstruction recoverable in tests: the
    unclamped reconstruction sums features chosen by the encoder back
    into the residual.  Concrete weight values are filled in by
    individual tests.
    """
    feats = torch.tensor(clampable_features, dtype=torch.int64)
    register_sae_full_recon_buffers(
        layer,
        hook_point=hook_point,
        module_name="m",
        activation=activation,
        activation_params=activation_params or {},
        d_sae=d_sae,
        n_clamp=len(feats),
        hidden_size=d_model,
        max_recon_configs=max_recon_configs,
        clampable_features=feats,
        dtype=torch.float32,
    )
    register_sae_recon_index_buffer(layer, max_tokens=max_tokens)


def _make_layer(layer_idx: int = 0) -> nn.Module:
    layer = nn.Module()
    layer.layer_idx = layer_idx  # type: ignore[attr-defined]
    return layer


class TestLayerDispatchDisabled:
    """When buffers aren't attached, the shim must pass through."""

    def test_no_buffers_returns_input_unchanged(self):
        layer = _make_layer()
        h = torch.randn(3, 4)
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        assert out is h or torch.equal(out, h)

    def test_buffers_attached_for_other_hook_still_passes_through(self):
        # An attached SAE on POST_ATTN must not affect POST_BLOCK traffic.
        layer = _make_layer()
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_ATTN,
            d_model=4,
            d_sae=8,
            clampable_features=[0, 1],
        )
        h = torch.randn(3, 4)
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        assert torch.equal(out, h)

    def test_empty_token_batch_short_circuits(self):
        layer = _make_layer()
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            d_model=4,
            d_sae=8,
            clampable_features=[0, 1],
        )
        h = torch.zeros(0, 4)
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        assert out.shape == (0, 4)


class TestReconIndexRouting:
    """Token-row routing via ``sae_recon_index``."""

    def test_row_zero_passes_through_unchanged(self):
        # All tokens map to row 0 → all pass through bit-identically
        # even though every other buffer is attached.
        layer = _make_layer()
        d_model, d_sae = 4, 6
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            d_model=d_model,
            d_sae=d_sae,
            clampable_features=[0, 1],
        )
        # Fill encoder/decoder with non-trivial values so any
        # accidental reconstruction would diverge.
        torch.manual_seed(0)
        getattr(
            layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.randn(d_sae, d_model))
        getattr(
            layer, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.randn(d_sae, d_model))
        getattr(
            layer, HOOK_POINT_FR_DECODER_BIAS_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.randn(d_model))
        # sae_recon_index is all zeros (default) → all rows = no-op.
        h = torch.randn(3, d_model)
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        assert torch.equal(out, h)

    def test_partial_recon_index_routes_per_token(self):
        # Set sae_recon_index = [0, 1, 0]; only row 1 reconstructs.
        # With identity encoder + decoder, no clamps in row 1 → the
        # masked-row output is `decode(activate(encode(h_t)))`, not h_t.
        layer = _make_layer()
        d_model, d_sae = 3, 3
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            d_model=d_model,
            d_sae=d_sae,
            clampable_features=[0],
        )
        # Identity encoder, identity decoder, zero biases — pure
        # ReLU reconstruction.
        getattr(
            layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.eye(d_sae))
        getattr(
            layer, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.eye(d_sae))
        h = torch.tensor(
            [
                [1.0, -1.0, 2.0],
                [3.0, -2.0, 0.5],
                [4.0, 0.0, -3.0],
            ]
        )
        layer.sae_recon_index[:3].copy_(  # type: ignore[union-attr]
            torch.tensor([0, 1, 0], dtype=torch.long)
        )
        # Row 1 belongs to this site — mark it in the active-row table
        # (the populator does this in production).
        getattr(layer, HOOK_POINT_FR_ROW_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK])[
            1
        ] = True
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        # Row 0 (no-op) → out[0] == h[0].
        assert torch.allclose(out[0], h[0])
        # Row 1 (recon) → ReLU(h[1]) since enc/dec are identity, no clamps.
        assert torch.allclose(out[1], torch.tensor([3.0, 0.0, 0.5]))
        # Row 0 again → out[2] == h[2].
        assert torch.allclose(out[2], h[2])

    def test_row_with_clamp_applies_modification(self):
        # One token routed to row 1 with an absolute clamp on feat 0.
        layer = _make_layer()
        d_model, d_sae = 3, 3
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            d_model=d_model,
            d_sae=d_sae,
            clampable_features=[0],
        )
        getattr(
            layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.eye(d_sae))
        getattr(
            layer, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_BLOCK]
        ).copy_(torch.eye(d_sae))
        # Clamp row 1, position 0 → absolute value 7.
        getattr(layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK])[
            1, 0
        ] = CLAMP_KIND_ABSOLUTE
        getattr(layer, HOOK_POINT_FR_CLAMP_VALUE_ATTR[SteeringHookPoint.POST_BLOCK])[
            1, 0
        ] = 7.0
        # Token 0 → row 1, marked active for this site.
        h = torch.tensor([[1.0, -1.0, 2.0]])
        layer.sae_recon_index[:1].copy_(torch.tensor([1], dtype=torch.long))  # type: ignore[union-attr]
        getattr(layer, HOOK_POINT_FR_ROW_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK])[
            1
        ] = True
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        # f = ReLU([1, -1, 2]) = [1, 0, 2]; absolute clamp on feat 0 →
        # [7, 0, 2]; identity decoder → [7, 0, 2].
        assert torch.allclose(out[0], torch.tensor([7.0, 0.0, 2.0]))


class TestLayerDispatchInvokesCustomOp:
    """The shim must call through ``torch.ops.vllm.apply_sae_full_reconstruction``.

    Verified by a monkeypatch hook on the registered op that records
    call counts.  This guards against a regression where the layer
    shim calls the eager Python function directly and skips the
    ``torch.compile`` fence.
    """

    def test_dispatch_invokes_torch_ops(self, monkeypatch):
        layer = _make_layer()
        _attach_identity_sae(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            d_model=4,
            d_sae=6,
            clampable_features=[0, 1],
        )
        original = torch.ops.vllm.apply_sae_full_reconstruction
        calls = {"n": 0}

        def counting_op(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            torch.ops.vllm, "apply_sae_full_reconstruction", counting_op
        )

        # Route at least one token to row 1 so the shim has work.
        layer.sae_recon_index[:2].copy_(  # type: ignore[union-attr]
            torch.tensor([1, 0], dtype=torch.long)
        )
        h = torch.randn(2, 4)
        out = apply_layer_sae_full_reconstruction(layer, h, SteeringHookPoint.POST_BLOCK)
        assert out.shape == h.shape
        assert calls["n"] == 1


class TestCustomOpRegistration:
    """``torch.ops.vllm.apply_sae_full_reconstruction`` must be live."""

    def test_custom_op_is_registered(self):
        # Importing the module is enough — the registration runs at
        # import time.
        from vllm.model_executor.layers import sae_full_reconstruction  # noqa: F401

        assert hasattr(torch.ops.vllm, "apply_sae_full_reconstruction"), (
            "torch.ops.vllm.apply_sae_full_reconstruction should be "
            "registered by importing "
            "vllm.model_executor.layers.sae_full_reconstruction."
        )

    def test_op_func_runs_on_cpu_tensors(self):
        # The op-func is invocable directly with CPU tensors; this
        # mirrors the public-API path that bypasses torch.ops to
        # avoid CPU/CUDA dispatch-key mismatches in test envs.
        from vllm.model_executor.layers.sae_full_reconstruction import (
            ACTIVATION_CODE_RELU,
            apply_sae_full_reconstruction_op,
        )

        n_tokens, d_model, d_sae, n_clamp = 2, 3, 4, 1
        h = torch.randn(n_tokens, d_model)
        W_enc = torch.eye(d_sae)[:, :d_model]
        b_enc = torch.zeros(d_sae)
        W_dec = torch.eye(d_sae)[:, :d_model]
        b_dec = torch.zeros(d_model)
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.zeros(n_tokens, n_clamp, dtype=torch.int8)
        clamp_value = torch.zeros(n_tokens, n_clamp, dtype=torch.float32)
        only = torch.zeros(n_tokens, n_clamp, dtype=torch.bool)
        recon_mask = torch.tensor([True, False])
        out = apply_sae_full_reconstruction_op(
            h,
            W_enc,
            b_enc,
            torch.zeros(d_sae),
            W_dec,
            b_dec,
            feats,
            clamp_kind,
            clamp_value,
            only,
            recon_mask,
            ACTIVATION_CODE_RELU,
            0.0,
        )
        # Token 1 is unmasked → equal to input.
        assert torch.equal(out[1], h[1])
        # Token 0 reconstructed; can't predict exact value but shape ok.
        assert out.shape == h.shape


class TestPublicAPIStillValidatesShapes:
    """The public ``apply_sae_full_reconstruction`` keeps its validation.

    These tests confirm that wrapping the op-func behind the public
    Python API didn't drop the shape-checking surface that Stage 1
    established.
    """

    def test_rejects_mismatched_decoder_bias(self):
        from vllm.model_executor.layers.sae_full_reconstruction import (
            apply_sae_full_reconstruction,
        )

        h = torch.randn(2, 4)
        W_enc = torch.randn(8, 4)
        b_enc = torch.randn(8)
        W_dec = torch.randn(8, 4)
        b_dec = torch.randn(5)  # wrong d_model
        feats = torch.tensor([0], dtype=torch.int64)
        clamp_kind = torch.zeros(2, 1, dtype=torch.int8)
        clamp_value = torch.zeros(2, 1, dtype=torch.float32)
        only = torch.zeros(2, 1, dtype=torch.bool)
        recon_mask = torch.zeros(2, dtype=torch.bool)
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
                clamp_kind=clamp_kind,
                clamp_value=clamp_value,
                clamp_only_if_active=only,
                recon_mask=recon_mask,
            )
