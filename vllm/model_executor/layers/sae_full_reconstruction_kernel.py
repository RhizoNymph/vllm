# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture-safe CUDA path for the SAE full-reconstruction custom op.

A dense masked Triton kernel, structurally mirroring the delta kernel
in :mod:`sae_steering_kernel`: one program per token over the padded
batch (``grid = (N,)``), no data-dependent shapes, no host sync — so
the op runs *inside* compiled / CUDA-graph-captured regions.

Per-program flow:

* Load the token's ``recon_mask`` scalar.  Inactive tokens copy their
  hidden row through in ``BLOCK_H`` tiles and return — a mask-scalar
  load plus one row memcpy, no ``d_sae`` work.
* Active tokens stream the full ``d_sae`` feature axis in ``BLOCK_S``
  tiles.  Each tile computes its encoder pre-activations (inner
  ``BLOCK_H`` sweep over the hidden row), applies the activation,
  applies any clamp whose ``clampable_features`` index falls inside
  the tile, and accumulates the decoder contribution into a per-token
  fp32 scratch row (``acc``, caller-allocated ``(N, H)``) seeded with
  ``b_dec``.  A final sweep casts the scratch row to the residual
  dtype and stores it.

The scratch row exists because a register-resident ``(d_model,)``
decoder accumulator cannot be dynamically sliced per ``BLOCK_H`` tile
in Triton; a single program owns its row, so the read-modify-write
accumulation is race-free, and its traffic is negligible next to the
``2 * d_sae * d_model`` weight loads that dominate an active token.

Numeric dtype contract (matches the delta kernel): encoder / decoder
accumulate in fp32 even for bf16/fp16 weights, activation + clamp
arithmetic in fp32, one cast back to the residual dtype at the store.

Activation encoding (``ACTIVATION_CODE`` constexpr): ``0`` = ReLU,
``1`` = JumpReLU (per-feature ``(d_sae,)`` fp32 ``threshold``).  TopK
(``2``) needs a global rank over all ``d_sae`` features, which does
not fit a streaming per-token program; TopK sites route to the dense
eager body — itself capture-safe (``torch.topk`` has static output
shapes) but without the inactive-token short-circuit.  The same dense
fallback serves clamp counts beyond ``_MAX_BLOCK_C``.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.intervention_kernel_common import run_kernel_warmup
from vllm.model_executor.layers.sae_steering import (
    ACTIVATION_CODE_TOPK,
)
from vllm.triton_utils import tl, triton

# Feature-axis tile width.  Together with ``BLOCK_H`` this bounds the
# (BLOCK_S, BLOCK_H) weight tile the kernel stages per inner step.
_FR_BLOCK_S = 32

# Cap on the hidden-axis tile.  The weight tiles are contiguous along
# the hidden axis, so a wider BLOCK_H improves coalescing; 256 keeps
# the (BLOCK_S, BLOCK_H) fp32 tile at 8K elements.
_FR_MAX_BLOCK_H = 256

# Cap on BLOCK_C, mirroring the delta kernel: clampable subsets beyond
# this fall back to the dense eager body (capture-safe, just not
# per-token gated).
_MAX_BLOCK_C = 256


@triton.jit
def _apply_sae_full_recon_kernel(
    hidden_ptr,
    enc_w_ptr,
    enc_b_ptr,
    threshold_ptr,
    dec_w_ptr,
    dec_b_ptr,
    feat_ptr,
    kind_ptr,
    value_ptr,
    only_ptr,
    mask_ptr,
    gate_ptr,
    acc_ptr,
    out_ptr,
    N,
    H,
    D_SAE,
    n_clamp,
    h_stride_n,
    h_stride_h,
    enc_stride_s,
    enc_stride_h,
    enc_b_stride,
    thr_stride,
    dec_stride_s,
    dec_stride_h,
    dec_b_stride,
    feat_stride,
    kind_stride_n,
    kind_stride_c,
    value_stride_n,
    value_stride_c,
    only_stride_n,
    only_stride_c,
    mask_stride,
    gate_stride,
    acc_stride_n,
    acc_stride_h,
    out_stride_n,
    out_stride_h,
    ACTIVATION_CODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute one token row of the SAE full-reconstruction op."""
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    h_row_ptr = hidden_ptr + pid * h_stride_n
    out_row_ptr = out_ptr + pid * out_stride_n
    if tl.load(mask_ptr + pid * mask_stride) == 0:
        # Inactive token: pass the residual through unchanged.
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            h_mask = h_idx < H
            h_vals = tl.load(h_row_ptr + h_idx * h_stride_h, mask=h_mask)
            tl.store(out_row_ptr + h_idx * out_stride_h, h_vals, mask=h_mask)
        return

    # Seed the fp32 scratch row with the decoder bias.
    acc_row_ptr = acc_ptr + pid * acc_stride_n
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H
        b_vals = tl.load(dec_b_ptr + h_idx * dec_b_stride, mask=h_mask, other=0.0).to(
            tl.float32
        )
        tl.store(acc_row_ptr + h_idx * acc_stride_h, b_vals, mask=h_mask)

    # Per-token monitor gate (1.0 = full clamp strength).  Scales only
    # the clamp-induced feature delta below — never the reconstruction.
    g_tok = tl.load(gate_ptr + pid * gate_stride).to(tl.float32)

    # Per-token clamp state, register-resident across the feature sweep.
    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < n_clamp
    feat = tl.load(feat_ptr + c_idx * feat_stride, mask=c_mask, other=0).to(tl.int32)
    kind_t = tl.load(
        kind_ptr + pid * kind_stride_n + c_idx * kind_stride_c,
        mask=c_mask,
        other=0,
    ).to(tl.int32)
    value_t = tl.load(
        value_ptr + pid * value_stride_n + c_idx * value_stride_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    only_t = tl.load(
        only_ptr + pid * only_stride_n + c_idx * only_stride_c,
        mask=c_mask,
        other=0,
    ).to(tl.int32)

    s_range = tl.arange(0, BLOCK_S)
    for s_off in range(0, D_SAE, BLOCK_S):
        s_idx = s_off + s_range
        s_mask = s_idx < D_SAE

        # Encoder pass for this feature tile.
        pre = tl.load(enc_b_ptr + s_idx * enc_b_stride, mask=s_mask, other=0.0).to(
            tl.float32
        )
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            h_mask = h_idx < H
            h_vals = tl.load(h_row_ptr + h_idx * h_stride_h, mask=h_mask, other=0.0).to(
                tl.float32
            )
            enc_off = s_idx[:, None] * enc_stride_s + h_idx[None, :] * enc_stride_h
            enc_mask = s_mask[:, None] & h_mask[None, :]
            enc_block = tl.load(enc_w_ptr + enc_off, mask=enc_mask, other=0.0).to(
                tl.float32
            )
            pre += tl.sum(enc_block * h_vals[None, :], axis=1)

        if ACTIVATION_CODE == 0:  # ReLU
            f_tile = tl.maximum(pre, 0.0)
        else:  # JumpReLU — per-feature thresholds.
            thr = tl.load(
                threshold_ptr + s_idx * thr_stride, mask=s_mask, other=0.0
            ).to(tl.float32)
            f_tile = tl.where(pre > thr, pre, 0.0)
        f_tile = tl.where(s_mask, f_tile, 0.0)

        # Apply clamps whose feature index lands in this tile.  ``sel``
        # is the one-hot (BLOCK_C, BLOCK_S) lane match; each clampable
        # feature is unique so it fires in exactly one tile.
        in_tile = c_mask & (feat >= s_off) & (feat < s_off + BLOCK_S)
        sel = in_tile[:, None] & ((feat[:, None] - s_off) == s_range[None, :])
        f_at = tl.sum(tl.where(sel, f_tile[None, :], 0.0), axis=1)
        active_for_gate = f_at > 0.0
        new_f = tl.where(
            kind_t == 1,
            value_t,
            tl.where(kind_t == 2, f_at + value_t, f_at),
        )
        apply_c = in_tile & (kind_t != 0) & ((only_t == 0) | active_for_gate)
        delta_c = tl.where(apply_c, (new_f - f_at) * g_tok, 0.0)
        f_tile += tl.sum(tl.where(sel, delta_c[:, None], 0.0), axis=0)

        # Decoder pass: accumulate this tile's contribution into the
        # scratch row (single program owns the row — race-free RMW).
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            h_mask = h_idx < H
            dec_off = s_idx[:, None] * dec_stride_s + h_idx[None, :] * dec_stride_h
            dec_mask = s_mask[:, None] & h_mask[None, :]
            dec_block = tl.load(dec_w_ptr + dec_off, mask=dec_mask, other=0.0).to(
                tl.float32
            )
            partial = tl.sum(dec_block * f_tile[:, None], axis=0)
            acc_vals = tl.load(
                acc_row_ptr + h_idx * acc_stride_h, mask=h_mask, other=0.0
            )
            tl.store(
                acc_row_ptr + h_idx * acc_stride_h, acc_vals + partial, mask=h_mask
            )

    # Final sweep: cast the fp32 reconstruction to the residual dtype.
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H
        acc_vals = tl.load(acc_row_ptr + h_idx * acc_stride_h, mask=h_mask, other=0.0)
        tl.store(
            out_row_ptr + h_idx * out_stride_h,
            acc_vals.to(out_ptr.dtype.element_ty),
            mask=h_mask,
        )


def _next_power_of_two(value: int) -> int:
    """Round ``value`` up to the next power of two (≥ 1).

    Implemented manually so the module stays importable on CPU-only
    builds where ``triton.next_power_of_2`` may be a stub.
    """
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _choose_fr_block_h(hidden_size: int) -> int:
    """Pick ``BLOCK_H`` — next power of two, capped at ``_FR_MAX_BLOCK_H``."""
    return min(_next_power_of_two(hidden_size), _FR_MAX_BLOCK_H)


def _choose_block_c(n_clamp: int) -> int:
    """Pick ``BLOCK_C`` for the kernel given ``n_clamp`` (≥ 0)."""
    return _next_power_of_two(max(n_clamp, 1))


def _fr_kernel_supports(n_clamp: int, activation_code: int) -> bool:
    """Whether the dense masked kernel serves this site.

    TopK needs a global rank over the full ``d_sae`` axis — not
    expressible in a streaming per-token program — and oversized clamp
    subsets blow the register tile; both route to the dense eager body,
    which is equally capture-safe.
    """
    if int(activation_code) == ACTIVATION_CODE_TOPK:
        return False
    return _choose_block_c(n_clamp) <= _MAX_BLOCK_C


def apply_sae_full_recon_triton(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    threshold: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    activation_param: float,
    clamp_row_gated: torch.Tensor | None = None,
    row_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA path for the SAE full-reconstruction op (capture-safe).

    ``clamp_row_gated`` / ``row_gate`` are optional per-token fp32
    ``(n_tokens,)`` monitor-gating tensors; the clamp-induced feature
    delta is scaled by ``g = 1 - gated[t] * (1 - row_gate[t])``.  The
    reconstruction itself is never gated.  ``None`` (either) means
    ungated — the kernel receives an all-ones gate, which is exactly
    the legacy math.

    Launches the dense masked Triton kernel over the padded token
    batch; each program gates on its own ``recon_mask`` scalar, so
    inactive tokens cost a row copy-through and active tokens run the
    full encode / clamp / decode.  Everything is fixed-shape and
    device-side — legal inside CUDA-graph capture.

    TopK sites and clampable subsets beyond ``_MAX_BLOCK_C`` route to
    :func:`vllm.model_executor.layers.sae_full_reconstruction._apply_sae_full_reconstruction_eager`
    — dense over all tokens but equally capture-safe.
    """
    out = torch.empty_like(hidden_states)
    n_tokens = hidden_states.shape[0]
    if n_tokens == 0:
        return out
    n_clamp = clampable_features.shape[0]
    if not _fr_kernel_supports(n_clamp, activation_code):
        from vllm.model_executor.layers.sae_full_reconstruction import (
            _apply_sae_full_reconstruction_eager,
        )

        return _apply_sae_full_reconstruction_eager(
            hidden_states,
            encoder_weight,
            encoder_bias,
            threshold,
            decoder_weight,
            decoder_bias,
            clampable_features,
            clamp_kind,
            clamp_value,
            clamp_only_if_active,
            recon_mask,
            activation_code,
            activation_param,
            clamp_row_gated,
            row_gate,
        )

    h_size = hidden_states.shape[1]
    d_sae = encoder_weight.shape[0]
    block_h = _choose_fr_block_h(h_size)
    block_c = _choose_block_c(n_clamp)
    only_int = clamp_only_if_active.view(torch.int8)
    mask_int = recon_mask.view(torch.int8)
    if clamp_row_gated is not None and row_gate is not None:
        gate = 1.0 - clamp_row_gated.to(torch.float32) * (
            1.0 - row_gate.to(torch.float32)
        )
    else:
        gate = torch.ones(n_tokens, dtype=torch.float32, device=hidden_states.device)
    acc = torch.empty(
        n_tokens, h_size, dtype=torch.float32, device=hidden_states.device
    )

    _apply_sae_full_recon_kernel[(n_tokens,)](
        hidden_states,
        encoder_weight,
        encoder_bias,
        threshold,
        decoder_weight,
        decoder_bias,
        clampable_features,
        clamp_kind,
        clamp_value,
        only_int,
        mask_int,
        gate,
        acc,
        out,
        n_tokens,
        h_size,
        d_sae,
        n_clamp,
        hidden_states.stride(0),
        hidden_states.stride(1),
        encoder_weight.stride(0),
        encoder_weight.stride(1),
        encoder_bias.stride(0),
        threshold.stride(0),
        decoder_weight.stride(0),
        decoder_weight.stride(1),
        decoder_bias.stride(0),
        clampable_features.stride(0),
        clamp_kind.stride(0),
        clamp_kind.stride(1),
        clamp_value.stride(0),
        clamp_value.stride(1),
        only_int.stride(0),
        only_int.stride(1),
        mask_int.stride(0),
        gate.stride(0),
        acc.stride(0),
        acc.stride(1),
        out.stride(0),
        out.stride(1),
        ACTIVATION_CODE=int(activation_code),
        BLOCK_H=block_h,
        BLOCK_S=_FR_BLOCK_S,
        BLOCK_C=block_c,
    )
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
    """JIT-compile the FR kernel ahead of CUDA-graph capture.

    Mirrors :func:`sae_steering_kernel.warmup_apply_sae_delta_kernel`:
    a tiny single-token launch with an active mask drives the full
    kernel body through Triton's first-call JIT outside any captured
    forward.  Sites the kernel does not serve (TopK, oversized clamp
    subsets) drive the dense eager body once instead, pre-warming its
    cuBLAS dispatch.

    No-op on CPU and on degenerate ``d_sae == 0`` / ``hidden_size ==
    0`` shapes that the runtime treats as disabled-mode.
    """
    if device.type != "cuda":
        return
    if d_sae <= 0 or hidden_size <= 0:
        return
    n_tokens = 1
    dummy_h = torch.zeros(n_tokens, hidden_size, dtype=compute_dtype, device=device)
    dummy_enc_w = torch.zeros(d_sae, hidden_size, dtype=table_dtype, device=device)
    dummy_enc_b = torch.zeros(d_sae, dtype=table_dtype, device=device)
    dummy_threshold = torch.zeros(d_sae, dtype=torch.float32, device=device)
    dummy_dec_w = torch.zeros(d_sae, hidden_size, dtype=table_dtype, device=device)
    dummy_dec_b = torch.zeros(hidden_size, dtype=table_dtype, device=device)
    feats = torch.arange(max(n_clamp, 0), dtype=torch.int64, device=device)
    dummy_kind = torch.zeros(n_tokens, max(n_clamp, 0), dtype=torch.int8, device=device)
    dummy_value = torch.zeros(
        n_tokens, max(n_clamp, 0), dtype=torch.float32, device=device
    )
    dummy_only = torch.zeros(n_tokens, max(n_clamp, 0), dtype=torch.bool, device=device)
    mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

    def drive(n: int) -> None:
        apply_sae_full_recon_triton(
            dummy_h[:n],
            dummy_enc_w,
            dummy_enc_b,
            dummy_threshold,
            dummy_dec_w,
            dummy_dec_b,
            feats,
            dummy_kind[:n],
            dummy_value[:n],
            dummy_only[:n],
            mask[:n],
            int(activation_code),
            float(activation_param),
        )

    if not _fr_kernel_supports(n_clamp, activation_code):
        # Dense eager fallback path — warm cuBLAS, no Triton JIT.
        drive(n_tokens)
        return

    # Single-token drive by design: the binary specialises on
    # ``ACTIVATION_CODE`` and the block constexprs, not the batch dim.
    run_kernel_warmup(
        label="sae full-recon",
        kernels=(_apply_sae_full_recon_kernel,),
        sizes=[n_tokens],
        drive=drive,
        dump_env_var="VLLM_SAE_DUMP_JIT_CACHE",
    )
