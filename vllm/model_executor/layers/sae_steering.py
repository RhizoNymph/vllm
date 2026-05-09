# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager reference op for SAE feature-surgery (delta) steering.

The op math, per token ``t`` with clamp set ``I``:

    pre_act_i  = W_enc[i, :] · h_t + b_enc[i]
    f_i        = activation(pre_act_i)
    new_f_i    = clamp_kind == ABSOLUTE  ? clamp_value
                 clamp_kind == ADDITIVE  ? f_i + clamp_value
                 (kind == NONE)         : f_i
    delta_i    = (new_f_i - f_i) gated by only_if_active when set
    h_t_new    = h_t + Σ_{i ∈ I} delta_i · W_dec[i, :]

This is the **eager reference implementation** for Phase 1 of the SAE
steering rollout: a vectorized PyTorch path with the same input shape
and dtype contract that the Phase-2 Triton kernel will adopt.  Layer-
hook integration (per-layer buffers, custom-op registration with
``torch.ops.vllm.apply_sae_delta``, and the model_runner_mixin wires)
arrives in a follow-up so this op stays standalone, unit-testable, and
reusable in tests that assemble inputs directly.

Numeric dtype contract (matches ``docs/features/sae_steering.md``):

* Encoder GEMM and decoder GEMM run in the model's compute dtype.
* The ``(n_tokens, n_clamp)`` activation tensor is promoted to fp32
  for the activation function and the ``delta = clamp(f, target) − f``
  subtraction; results are cast back to compute dtype before the
  decoder GEMM.

Phase-1 supports ``ReLU``, ``JumpReLU`` (``activation_params['threshold']``),
and ``TopK`` (``activation_params['k']``).  TopK selects k largest
pre-activations across the **encoder rows passed in** — for a partial
encoder this is "TopK among the clampable subset".  Operators who
need full-d_sae TopK semantics must load the full encoder.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from vllm.config.sae_steering_types import SAEActivation

# Integer codes for ``clamp_kind`` tensors.  Kept explicit so the
# Triton kernel (Phase 2) and any other consumer can use the same
# values without a Python enum lookup.
CLAMP_KIND_NONE = 0
CLAMP_KIND_ABSOLUTE = 1
CLAMP_KIND_ADDITIVE = 2


def sae_encode(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
) -> torch.Tensor:
    """Project ``hidden_states`` through the (partial) encoder and apply activation.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.
        encoder_weight: ``(n_clamp, d_model)``.  These are the encoder
            rows for the clampable feature subset.
        encoder_bias: ``(n_clamp,)``.
        activation: encoder activation function.
        activation_params: ``{"threshold": float}`` for JumpReLU,
            ``{"k": float}`` for TopK (cast to int internally), ignored
            for ReLU.

    Returns:
        ``(n_tokens, n_clamp)`` activation tensor in fp32.  Callers
        cast back to compute dtype after applying clamps.
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
        n_clamp = pre_act.shape[1]
        if k >= n_clamp:
            return pre_act
        # Per-row TopK mask.
        _, top_idx = torch.topk(pre_act, k=k, dim=1, largest=True)
        mask = torch.zeros_like(pre_act, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return torch.where(mask, pre_act, torch.zeros_like(pre_act))
    raise ValueError(f"Unsupported SAE activation: {activation!r}")


def apply_sae_delta(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
) -> torch.Tensor:
    """Eager reference for the SAE feature-surgery delta op.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.  Output
            is the same shape and dtype.
        encoder_weight: ``(n_clamp, d_model)`` encoder rows for the
            clampable feature subset.
        encoder_bias: ``(n_clamp,)``.
        decoder_weight: ``(n_clamp, d_model)`` decoder rows aligned
            with ``encoder_weight``: row ``i`` is the decoder direction
            for the same feature whose encoder row is at index ``i``.
        activation: encoder activation function.
        activation_params: parameters for ``activation`` (see
            :func:`sae_encode`).
        clamp_kind: ``(n_tokens, n_clamp)`` int8.  Per-token, per-
            feature clamp kind: ``CLAMP_KIND_NONE`` (skip),
            ``CLAMP_KIND_ABSOLUTE`` (set ``f := value``), or
            ``CLAMP_KIND_ADDITIVE`` (set ``f := f + value``).
        clamp_value: ``(n_tokens, n_clamp)`` float.  Target value for
            absolute clamps; offset for additive clamps.  Ignored where
            ``clamp_kind == CLAMP_KIND_NONE``.
        clamp_only_if_active: ``(n_tokens, n_clamp)`` bool.  When True,
            the clamp is suppressed at positions where ``f <= 0`` in
            the live encoder pass — "amplify when present" semantics.

    Returns:
        ``hidden_states + Σ_i delta_i · W_dec[i]`` in the same dtype
        as ``hidden_states``.

    Notes:
        Always runs the encoder pass.  The "skip encoder when no clamp
        needs it" optimization (a kernel-level concern; cf.
        ``SAEClampEntry.requires_encoder_pass``) is reserved for the
        Phase-2 fused kernel where it pays off.
    """
    n_tokens, d_model = hidden_states.shape
    n_clamp = encoder_weight.shape[0]

    if encoder_weight.shape != (n_clamp, d_model):
        raise ValueError(
            "encoder_weight must be (n_clamp, d_model) matching hidden_states; "
            f"got {tuple(encoder_weight.shape)} vs d_model={d_model}."
        )
    if encoder_bias.shape != (n_clamp,):
        raise ValueError(
            "encoder_bias must be (n_clamp,); "
            f"got {tuple(encoder_bias.shape)} vs n_clamp={n_clamp}."
        )
    if decoder_weight.shape != (n_clamp, d_model):
        raise ValueError(
            "decoder_weight must be (n_clamp, d_model) aligned with encoder; "
            f"got {tuple(decoder_weight.shape)} vs (n_clamp={n_clamp}, "
            f"d_model={d_model})."
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

    # n_clamp == 0 short-circuit: no features to clamp, no work to do.
    if n_clamp == 0:
        return hidden_states.clone()

    # (n_tokens, n_clamp) fp32 — encoder pass.
    f = sae_encode(
        hidden_states, encoder_weight, encoder_bias, activation, activation_params
    )

    kind = clamp_kind.to(torch.int8)
    value = clamp_value.to(torch.float32)
    gated = clamp_only_if_active.to(torch.bool)
    active = f > 0.0

    # Per-(token, feature) new-f computation.  Branchless via where().
    new_f_absolute = value
    new_f_additive = f + value
    new_f = torch.where(
        kind == CLAMP_KIND_ABSOLUTE,
        new_f_absolute,
        torch.where(kind == CLAMP_KIND_ADDITIVE, new_f_additive, f),
    )
    # only_if_active gates everything: when gated and not active, the
    # clamp is suppressed (delta = 0).
    apply_clamp = (kind != CLAMP_KIND_NONE) & (~gated | active)
    delta = torch.where(apply_clamp, new_f - f, torch.zeros_like(f))

    # Cast tiny (n_tokens, n_clamp) tensor back to compute dtype before
    # the d_model GEMM, per the dtype contract.
    delta_compute = delta.to(hidden_states.dtype)
    decoder_compute = decoder_weight.to(hidden_states.dtype)
    # (n_tokens, n_clamp) @ (n_clamp, d_model) → (n_tokens, d_model)
    residual_delta = delta_compute @ decoder_compute
    return hidden_states + residual_delta
