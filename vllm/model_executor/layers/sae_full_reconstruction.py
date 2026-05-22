# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager reference op for SAE feature-surgery via full reconstruction.

Sibling to :mod:`sae_steering` (the *delta* path).  The math, per
token ``t`` whose ``recon_mask[t]`` is ``True``:

    pre_act     = h_t @ W_enc.T + b_enc          # (d_sae,)
    f           = activation(pre_act)            # (d_sae,)
    f' [c]      = apply_clamp(f[clampable_features[c]],
                              clamp_kind[t, c],
                              clamp_value[t, c],
                              clamp_only_if_active[t, c])
                  # for c ∈ [0, n_clamp); other features are unchanged
    h_t_new     = f' @ W_dec + b_dec             # (d_model,)

Tokens where ``recon_mask[t]`` is ``False`` pass through unchanged
(``out[t] = h[t]``).  This per-token gate is what makes per-request
opt-in tractable: a request that does not register a full-
reconstruction spec keeps its residual stream identical to a build
without the SAE module attached, the same per-token-row-indexing
contract the additive and delta paths use.

Compared to the delta variant (:func:`apply_sae_delta`):

* Encoder / decoder use the **full** ``d_sae`` rows, not just a
  ``clampable_features`` subset.  The reconstructed residual carries
  the SAE's reconstruction error along with whatever clamp
  modifications the caller requested, replacing the original
  residual entirely (Anthropic Scaling Monosemanticity / Golden Gate
  Claude semantics).
* The decoder bias ``b_dec`` participates — it does not in the delta
  path, where the result is a perturbation around ``h``.
* Per-token opt-in is via ``recon_mask`` (a bool tensor) rather than
  via row 0 of an indirection table — the math primitive is
  layer-buffer-agnostic so the eventual worker integration can wire
  ``recon_mask`` from any source (an explicit per-token flag, a
  derived ``recon_index != 0`` mask, etc.).

Phase-4 Stage 1 ships only this math primitive.  Per-(layer, hook)
buffer registration, the layer-hook dispatch shim, the custom-op
registration as ``torch.ops.vllm.apply_sae_full_reconstruction``, the
fused Triton kernel, and the worker mixin integration land in the
follow-up stages — exactly the cadence Phase-1A → Phase-1B → Phase-2
followed for the delta path.

Numeric dtype contract (matches ``docs/features/sae_steering.md``):

* Encoder GEMM and decoder GEMM run in the model's compute dtype.
* The activation tensor and clamp arithmetic are promoted to fp32 in
  the eager body to match the contract the kernel will use; results
  are cast back to compute dtype before the decoder GEMM.

Activation support: ``ReLU``, ``JumpReLU``
(``activation_params['threshold']``), and ``TopK``
(``activation_params['k']``).  Unlike the delta path, TopK runs over
the **full** ``d_sae`` here — true Golden-Gate-style semantics —
because we have the full encoder available.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
    CLAMP_KIND_NONE,
    _topk_mask_lowest_indices,
    _validate_clamp_kind_values,
)


def sae_encode_full(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
) -> torch.Tensor:
    """Project ``hidden_states`` through the full SAE encoder.

    Mirrors :func:`sae_steering.sae_encode` but operates on the full
    ``d_sae`` encoder rather than a clampable-subset slice; the
    reconstruction op needs every feature's activation to recompute
    the residual.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.
        encoder_weight: ``(d_sae, d_model)``.
        encoder_bias: ``(d_sae,)``.
        activation: encoder activation function.
        activation_params: ``{"threshold": float}`` for JumpReLU,
            ``{"k": float}`` for TopK (cast to int internally), ignored
            for ReLU.

    Returns:
        ``(n_tokens, d_sae)`` activation tensor in fp32.  The full-
        reconstruction caller modifies this in place at the
        clampable-feature positions before the decoder pass.
    """
    h_fp32 = hidden_states.to(torch.float32)
    enc_w_fp32 = encoder_weight.to(torch.float32)
    enc_b_fp32 = encoder_bias.to(torch.float32)
    pre_act = h_fp32 @ enc_w_fp32.t() + enc_b_fp32
    if activation is SAEActivation.RELU:
        return torch.clamp(pre_act, min=0.0)
    if activation is SAEActivation.JUMPRELU:
        threshold = float(activation_params["threshold"])
        return torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
    if activation is SAEActivation.TOPK:
        k = int(activation_params["k"])
        d_sae = pre_act.shape[1]
        if k >= d_sae:
            return pre_act
        mask = _topk_mask_lowest_indices(pre_act, k)
        return torch.where(mask, pre_act, torch.zeros_like(pre_act))
    raise ValueError(f"Unsupported SAE activation: {activation!r}")


def apply_sae_full_reconstruction(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
) -> torch.Tensor:
    """Eager reference for the SAE full-reconstruction op.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.  Output
            is the same shape and dtype.
        encoder_weight: ``(d_sae, d_model)`` — *full* encoder rows.
        encoder_bias: ``(d_sae,)``.
        decoder_weight: ``(d_sae, d_model)`` — *full* decoder rows.
        decoder_bias: ``(d_model,)``.
        activation: encoder activation function.
        activation_params: parameters for ``activation`` (see
            :func:`sae_encode_full`).
        clampable_features: ``(n_clamp,)`` int64 tensor of *global*
            feature indices that may be clamped per-token.  Indexes
            into the ``d_sae`` activation tensor; values must lie in
            ``[0, d_sae)`` and be unique.
        clamp_kind: ``(n_tokens, n_clamp)`` int8.  Per-(token, clamp-
            position) clamp kind (``CLAMP_KIND_NONE`` /
            ``CLAMP_KIND_ABSOLUTE`` / ``CLAMP_KIND_ADDITIVE``).
        clamp_value: ``(n_tokens, n_clamp)`` float — target / offset.
        clamp_only_if_active: ``(n_tokens, n_clamp)`` bool — when True,
            suppress the clamp at positions where the live ``f`` for
            that feature is ≤ 0 ("amplify when present" semantics).
        recon_mask: ``(n_tokens,)`` bool — per-token gate.  Tokens
            with ``recon_mask[t] == False`` pass through unchanged;
            tokens with ``recon_mask[t] == True`` have their residual
            replaced by ``decode(clamp(encode(h_t)))``.

    Returns:
        ``(n_tokens, d_model)`` tensor in the same dtype as
        ``hidden_states``.  Values for unmasked tokens are
        bit-identical to the corresponding rows of ``hidden_states``.

    Notes:
        Always runs the encoder pass for *all* tokens, even those
        with ``recon_mask=False``.  The pass-through tokens have
        their reconstruction discarded by the final ``where`` gate.
        The "skip encoder for unmasked tokens" optimisation is a
        kernel-level concern reserved for the Stage-2 fused kernel,
        where it pays off via per-program gating.
    """
    n_tokens, d_model = hidden_states.shape
    d_sae = encoder_weight.shape[0]
    n_clamp = clampable_features.shape[0]

    if encoder_weight.shape != (d_sae, d_model):
        raise ValueError(
            "encoder_weight must be (d_sae, d_model); "
            f"got {tuple(encoder_weight.shape)} vs d_model={d_model}."
        )
    if encoder_bias.shape != (d_sae,):
        raise ValueError(
            "encoder_bias must be (d_sae,); "
            f"got {tuple(encoder_bias.shape)} vs d_sae={d_sae}."
        )
    if decoder_weight.shape != (d_sae, d_model):
        raise ValueError(
            "decoder_weight must be (d_sae, d_model) aligned with encoder; "
            f"got {tuple(decoder_weight.shape)} vs (d_sae={d_sae}, "
            f"d_model={d_model})."
        )
    if decoder_bias.shape != (d_model,):
        raise ValueError(
            "decoder_bias must be (d_model,); "
            f"got {tuple(decoder_bias.shape)} vs d_model={d_model}."
        )
    if clampable_features.dtype != torch.int64:
        raise ValueError(
            f"clampable_features must be int64; got dtype={clampable_features.dtype}."
        )
    if clampable_features.ndim != 1:
        raise ValueError(
            "clampable_features must be 1-D; "
            f"got shape={tuple(clampable_features.shape)}."
        )
    if recon_mask.shape != (n_tokens,):
        raise ValueError(
            "recon_mask must be (n_tokens,); "
            f"got {tuple(recon_mask.shape)} vs n_tokens={n_tokens}."
        )
    if recon_mask.dtype != torch.bool:
        raise ValueError(
            f"recon_mask must be torch.bool; got dtype={recon_mask.dtype}."
        )
    expected_clamp_shape = (n_tokens, n_clamp)
    for name, t in (
        ("clamp_kind", clamp_kind),
        ("clamp_value", clamp_value),
        ("clamp_only_if_active", clamp_only_if_active),
    ):
        if tuple(t.shape) != expected_clamp_shape:
            raise ValueError(
                f"{name} must be {expected_clamp_shape}; got {tuple(t.shape)}."
            )
    if clamp_kind.dtype != torch.int8:
        raise ValueError(f"clamp_kind must be torch.int8; got {clamp_kind.dtype}.")
    _validate_clamp_kind_values(clamp_kind)
    if not clamp_value.dtype.is_floating_point:
        raise ValueError(
            f"clamp_value must be a floating dtype; got {clamp_value.dtype}."
        )
    if clamp_only_if_active.dtype != torch.bool:
        raise ValueError(
            "clamp_only_if_active must be torch.bool; "
            f"got {clamp_only_if_active.dtype}."
        )

    if n_clamp > 0 and not clampable_features.is_cuda:
        min_feature = int(clampable_features.min().item())
        max_feature = int(clampable_features.max().item())
        if min_feature < 0 or max_feature >= d_sae:
            raise ValueError(
                f"clampable_features must lie in [0, d_sae={d_sae}); "
                f"got min={min_feature}, max={max_feature}."
            )
        if torch.unique(clampable_features).numel() != n_clamp:
            raise ValueError("clampable_features must not contain duplicates.")

    # Empty token batch short-circuit.
    if n_tokens == 0:
        return hidden_states.clone()
    # No token opted into full reconstruction: preserve value semantics
    # without paying the full encoder/decoder cost that will be discarded by
    # the final mask.  Keep this CPU-only; on CUDA, ``.item()`` would
    # synchronize the stream and regress the common active-mask path.
    if not recon_mask.is_cuda and not bool(torch.any(recon_mask).item()):
        return hidden_states.clone()

    # Encode every token (we mask via ``recon_mask`` at the end).
    f = sae_encode_full(
        hidden_states,
        encoder_weight,
        encoder_bias,
        activation,
        activation_params,
    )
    # ``f`` is fp32; clamp arithmetic stays in fp32 to match the
    # numeric contract the eventual Triton kernel will adopt.

    if n_clamp > 0:
        # Gather the (n_tokens, n_clamp) subset of ``f`` keyed by the
        # clampable feature indices, apply clamp logic, then scatter
        # the modified values back into ``f``.
        idx_2d = clampable_features.unsqueeze(0).expand(n_tokens, -1)
        f_subset = f.gather(1, idx_2d)  # (n_tokens, n_clamp), fp32

        kind = clamp_kind.to(torch.int8)
        value = clamp_value.to(torch.float32)
        gated = clamp_only_if_active.to(torch.bool)
        active = f_subset != 0.0 if activation is SAEActivation.TOPK else f_subset > 0.0

        new_f_absolute = value
        new_f_additive = f_subset + value
        new_f = torch.where(
            kind == CLAMP_KIND_ABSOLUTE,
            new_f_absolute,
            torch.where(kind == CLAMP_KIND_ADDITIVE, new_f_additive, f_subset),
        )
        apply_clamp = (kind != CLAMP_KIND_NONE) & (~gated | active)
        new_f_subset = torch.where(apply_clamp, new_f, f_subset)

        f = f.scatter(1, idx_2d, new_f_subset)

    # Decode in the model's compute dtype, per the dtype contract.
    f_compute = f.to(hidden_states.dtype)
    decoder_compute = decoder_weight.to(hidden_states.dtype)
    decoder_bias_compute = decoder_bias.to(hidden_states.dtype)
    # (n_tokens, d_sae) @ (d_sae, d_model) → (n_tokens, d_model)
    reconstruction = f_compute @ decoder_compute + decoder_bias_compute

    # Per-token gate: replace residual where recon_mask is True,
    # else pass hidden_states through unchanged.
    return torch.where(recon_mask.unsqueeze(1), reconstruction, hidden_states)
