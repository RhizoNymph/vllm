# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA path for the SAE full-reconstruction custom op.

The main optimisation here is a **per-token short-circuit on
``recon_mask``**.  The eager body always encodes / decodes every
token, then masks out the unmodified positions at the end — for a
batch where only a few requests opted into reconstruction, that's a
lot of wasted full-``d_sae`` × ``d_model`` matmuls.

Stage 4 ships a *compaction-based* path that keeps the math identical
but only runs it on the active subset:

* Gather the active token rows (``recon_mask == True``) into a dense
  ``(n_active, d_model)`` tensor.
* Run the existing encoder / clamp / decoder math on the subset
  using PyTorch (which dispatches to cuBLAS for the two large
  GEMMs — already near-optimal at the typical Gemma-Scope shape of
  ``d_sae=16384``, ``d_model=2304``).
* Scatter the reconstructed rows back into the output tensor;
  unmasked positions retain the original ``hidden_states``.

A custom Triton kernel doesn't have an obvious advantage at these
shapes — cuBLAS already saturates the SMs on the encoder / decoder
matmuls, and the per-token short-circuit is what the design doc
identifies as the actual win.  The body behind
``apply_sae_full_reconstruction_op`` is a clean swap point if a
future Triton-kernel optimisation lands; the surface stays stable.

Empty-recon-mask short-circuits the full path: no compaction, no
matmul, just a clone.  Empty token batches return an empty tensor
without dispatching anywhere.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.sae_steering import _topk_mask_lowest_indices


def apply_sae_full_recon_triton(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """CUDA path for the SAE full-reconstruction op.

    Compacts active tokens, runs the encoder / clamp / decoder math
    on the active subset, and scatters the reconstructed rows back
    into a fresh output tensor.  Inactive tokens keep their original
    residual.

    The math is identical to
    :func:`vllm.model_executor.layers.sae_full_reconstruction._apply_sae_full_reconstruction_eager`
    on the active subset; this wrapper just narrows the work to the
    rows that actually need it.
    """
    n_tokens = hidden_states.shape[0]
    out = torch.empty_like(hidden_states)
    if n_tokens == 0:
        return out

    # Initialise output to the pass-through residual; reconstructed
    # rows are overwritten below.  ``copy_`` is a single contiguous
    # memcpy so the no-active-rows fast path is essentially free.
    out.copy_(hidden_states)

    active_idx = torch.nonzero(recon_mask, as_tuple=False).flatten()
    n_active = active_idx.numel()
    if n_active == 0:
        return out

    n_clamp = clampable_features.shape[0]
    h_active = hidden_states.index_select(0, active_idx)

    # Encoder pass on the active subset.  Promote to fp32 for the
    # activation + clamp arithmetic so the numerics match the eager
    # body bit-identically (modulo the cuBLAS tile differences that
    # affect the eager path too).
    h_fp32 = h_active.to(torch.float32)
    enc_w_fp32 = encoder_weight.to(torch.float32)
    enc_b_fp32 = encoder_bias.to(torch.float32)
    pre_act = h_fp32 @ enc_w_fp32.t() + enc_b_fp32

    if activation_code == 0:  # ReLU
        f = torch.clamp(pre_act, min=0.0)
    elif activation_code == 1:  # JumpReLU
        threshold = float(activation_param)
        f = torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
    elif activation_code == 2:  # TopK over the full d_sae
        k = int(activation_param)
        d_sae = pre_act.shape[1]
        if k >= d_sae:
            f = pre_act
        else:
            mask = _topk_mask_lowest_indices(pre_act, k)
            f = torch.where(mask, pre_act, torch.zeros_like(pre_act))
    else:
        raise ValueError(f"Unsupported activation_code: {activation_code!r}")

    if n_clamp > 0:
        # Apply per-(active-token, clamp-position) modifications to
        # ``f`` at the clampable feature indices.  ``clamp_*`` tables
        # are sized ``(n_tokens, n_clamp)`` covering the *full* batch;
        # gather their active-token rows.
        kind_active = clamp_kind.index_select(0, active_idx).to(torch.int8)
        value_active = clamp_value.index_select(0, active_idx).to(torch.float32)
        only_active = clamp_only_if_active.index_select(0, active_idx).to(torch.bool)

        idx_2d = clampable_features.unsqueeze(0).expand(n_active, -1)
        f_subset = f.gather(1, idx_2d)
        active_flag = (
            f_subset != 0.0 if activation_code == 2 else f_subset > 0.0
        )
        new_f_absolute = value_active
        new_f_additive = f_subset + value_active
        new_f = torch.where(
            kind_active == 1,
            new_f_absolute,
            torch.where(kind_active == 2, new_f_additive, f_subset),
        )
        apply_clamp = (kind_active != 0) & (~only_active | active_flag)
        new_f_subset = torch.where(apply_clamp, new_f, f_subset)
        f = f.scatter(1, idx_2d, new_f_subset)

    # Decoder pass — back to compute dtype so cuBLAS dispatches to
    # the right tensorcore tile.
    f_compute = f.to(hidden_states.dtype)
    dec_w_compute = decoder_weight.to(hidden_states.dtype)
    dec_b_compute = decoder_bias.to(hidden_states.dtype)
    reconstructed = f_compute @ dec_w_compute + dec_b_compute

    # Scatter the reconstructed rows back into ``out`` at the active
    # token positions.  ``index_copy_`` is the in-place counterpart
    # of ``index_select`` and is straight-line memcpy on row-
    # contiguous tensors.
    out.index_copy_(0, active_idx, reconstructed)
    return out


def warmup_apply_sae_full_recon_kernel(
    *,
    hidden_size: int,
    d_sae: int,
    n_clamp: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    activation_code: int = 0,
    activation_param: float = 0.0,
) -> None:
    """Pre-warm the CUDA path so its first call lands outside any captured forward.

    Mirrors ``warmup_apply_steering_kernel`` in
    :mod:`vllm.model_executor.layers.sae_steering_kernel` in spirit:
    a tiny dummy launch on CUDA so the dispatch path's
    workspace allocation, cuBLAS handle creation, and any first-call
    autotune happens *before* the engine starts capturing forward
    passes for CUDA-graph replay.

    Phase-4 Stage 4 routes the CUDA path through PyTorch matmuls
    (which call into cuBLAS), so this warmup is the cuBLAS
    equivalent of the delta-path's Triton JIT pre-compile.  A future
    Triton-kernel swap behind ``apply_sae_full_recon_triton`` would
    extend this helper to JIT-compile that kernel here too.

    No-op on CPU and on degenerate ``d_sae == 0`` / ``n_clamp == 0``
    shapes that the runtime treats as disabled-mode.
    """
    if device.type != "cuda":
        return
    if d_sae <= 0 or hidden_size <= 0:
        return
    n_active = 1
    dummy_h = torch.zeros(n_active, hidden_size, dtype=compute_dtype, device=device)
    dummy_W_enc = torch.zeros(d_sae, hidden_size, dtype=table_dtype, device=device)
    dummy_b_enc = torch.zeros(d_sae, dtype=table_dtype, device=device)
    dummy_W_dec = torch.zeros(d_sae, hidden_size, dtype=table_dtype, device=device)
    dummy_b_dec = torch.zeros(hidden_size, dtype=table_dtype, device=device)
    feats = torch.zeros(max(n_clamp, 0), dtype=torch.int64, device=device)
    dummy_kind = torch.zeros(n_active, max(n_clamp, 0), dtype=torch.int8, device=device)
    dummy_val = torch.zeros(
        n_active, max(n_clamp, 0), dtype=torch.float32, device=device
    )
    dummy_only = torch.zeros(n_active, max(n_clamp, 0), dtype=torch.bool, device=device)
    mask = torch.ones(n_active, dtype=torch.bool, device=device)
    apply_sae_full_recon_triton(
        dummy_h,
        dummy_W_enc,
        dummy_b_enc,
        dummy_W_dec,
        dummy_b_dec,
        feats,
        dummy_kind,
        dummy_val,
        dummy_only,
        mask,
        int(activation_code),
        float(activation_param),
    )
