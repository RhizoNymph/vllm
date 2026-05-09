# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the SAE feature-surgery (delta) custom op.

The kernel fuses, for each token row, the operations performed by the
eager :func:`apply_sae_delta` reference in :mod:`sae_steering`:

* Encoder GEMM for the clampable feature subset
  (``pre_act = h @ W_enc.T + b_enc``).
* Activation (ReLU, JumpReLU, or partial-encoder TopK).
* Per-(token, feature) clamp logic — absolute / additive / no-op,
  optionally gated by ``only_if_active``.
* Decoder GEMM and add-back to the residual
  (``h_new = h + delta @ W_dec``).

A separate program is launched per token (``grid = (N,)``) and the
hidden dimension is walked in ``BLOCK_H`` tiles so non-power-of-two
``d_model`` works correctly.  The clamp dimension is staged in
registers as a ``BLOCK_C``-wide tile, where ``BLOCK_C`` is the next
power of two ≥ ``n_clamp`` and lanes past ``n_clamp`` are masked out.
This keeps the encoder-rows / decoder-rows / clamp-state arrays
register-resident through the per-token computation.

Numeric dtype contract (matches ``docs/features/sae_steering.md``):

* Encoder GEMM and decoder GEMM accumulate in fp32 inside the kernel
  even when the weight tensors are bf16/fp16.
* The activation and clamp arithmetic happen in fp32.
* Decoder-direction sum is cast back to ``hidden_states.dtype`` at the
  store site so the output dtype matches the input.

Output is always written to a freshly allocated tensor — the kernel
preserves the eager reference's "no in-place" contract so the
``torch.compile`` graph keeps value semantics.

Activation encoding (``ACTIVATION_CODE`` constexpr): ``0`` = ReLU,
``1`` = JumpReLU (``activation_param`` = threshold), ``2`` = TopK
(``activation_param`` = k, cast to int).  ``activation_code`` is a
constexpr so each (activation, hook) site compiles to its own
specialised binary; Triton's JIT cache reuses the binary across
launches with identical specialisation.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

# Activation codes — kept in sync with ``sae_steering.py``.  A constexpr
# in the kernel switches on these values, so the binary specialises per
# activation and the disabled branches are eliminated at compile time.
ACTIVATION_CODE_RELU = 0
ACTIVATION_CODE_JUMPRELU = 1
ACTIVATION_CODE_TOPK = 2


@triton.jit
def _apply_sae_delta_kernel(
    hidden_ptr,
    enc_w_ptr,
    enc_b_ptr,
    dec_w_ptr,
    kind_ptr,
    value_ptr,
    only_ptr,
    out_ptr,
    N,
    H,
    n_clamp,
    h_stride_n,
    h_stride_h,
    enc_stride_c,
    enc_stride_h,
    enc_b_stride,
    dec_stride_c,
    dec_stride_h,
    kind_stride_n,
    kind_stride_c,
    value_stride_n,
    value_stride_c,
    only_stride_n,
    only_stride_c,
    out_stride_n,
    out_stride_h,
    activation_param,
    ACTIVATION_CODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Compute one token row of the SAE feature-surgery delta op."""
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < n_clamp

    # Initialise pre-activations with the encoder bias.
    pre_acts = tl.load(enc_b_ptr + c_idx * enc_b_stride, mask=c_mask, other=0.0).to(
        tl.float32
    )

    h_row_ptr = hidden_ptr + pid * h_stride_n
    out_row_ptr = out_ptr + pid * out_stride_n

    # Pass 1: accumulate encoder dot products.  The hidden row is
    # streamed through in BLOCK_H tiles; for each tile we load the
    # corresponding (BLOCK_C, BLOCK_H) slice of the encoder weights and
    # contract on the hidden axis.
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H
        h_vals = tl.load(h_row_ptr + h_idx * h_stride_h, mask=h_mask, other=0.0).to(
            tl.float32
        )
        enc_off = c_idx[:, None] * enc_stride_c + h_idx[None, :] * enc_stride_h
        enc_mask = c_mask[:, None] & h_mask[None, :]
        enc_block = tl.load(enc_w_ptr + enc_off, mask=enc_mask, other=0.0).to(
            tl.float32
        )
        pre_acts += tl.sum(enc_block * h_vals[None, :], axis=1)

    # Apply activation function to obtain f.
    if ACTIVATION_CODE == 0:  # ReLU
        f = tl.maximum(pre_acts, 0.0)
    elif ACTIVATION_CODE == 1:  # JumpReLU
        f = tl.where(pre_acts > activation_param, pre_acts, 0.0)
    else:  # TopK among the clampable subset.
        # rank[i] = number of valid lanes j with pre_act[j] > pre_act[i].
        # Lanes past n_clamp are excluded by setting them to -inf so they
        # never beat a real lane and never get counted as a beater.
        masked = tl.where(c_mask, pre_acts, float("-inf"))
        cmp = (masked[None, :] > masked[:, None]).to(tl.int32)
        rank = tl.sum(cmp, axis=1)
        k_int = tl.cast(activation_param, tl.int32)
        in_topk = (rank < k_int) & c_mask
        f = tl.where(in_topk, pre_acts, 0.0)

    # Force out-of-range lanes to zero so they can never produce a
    # decoder contribution even if the masking below glitches.
    f = tl.where(c_mask, f, 0.0)

    # Per-feature clamp state for this token.
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

    new_f_abs = value_t
    new_f_add = f + value_t
    new_f = tl.where(
        kind_t == 1,
        new_f_abs,
        tl.where(kind_t == 2, new_f_add, f),
    )
    apply_clamp = (kind_t != 0) & ((only_t == 0) | (f > 0.0)) & c_mask
    delta = tl.where(apply_clamp, new_f - f, 0.0)

    # Pass 2: write hidden + (delta @ W_dec).  The decoder rows are
    # streamed in the same BLOCK_H tiles as the encoder pass; each tile
    # contracts the BLOCK_C clamp axis so the per-token output row is
    # produced in one streaming sweep.
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H
        h_vals = tl.load(h_row_ptr + h_idx * h_stride_h, mask=h_mask, other=0.0)

        dec_off = c_idx[:, None] * dec_stride_c + h_idx[None, :] * dec_stride_h
        dec_mask = c_mask[:, None] & h_mask[None, :]
        dec_block = tl.load(dec_w_ptr + dec_off, mask=dec_mask, other=0.0).to(
            tl.float32
        )

        residual_delta = tl.sum(dec_block * delta[:, None], axis=0)
        result = h_vals.to(tl.float32) + residual_delta
        tl.store(
            out_row_ptr + h_idx * out_stride_h,
            result.to(h_vals.dtype),
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


def _choose_block_h(hidden_size: int) -> int:
    """Pick ``BLOCK_H`` for the kernel given ``hidden_size``.

    Mirrors :func:`steering_kernel._choose_block_h`: cap at 2048 for
    large hidden sizes (the loop walks the row in tiles); round up to
    the next power of two below that so a single iteration covers the
    row when possible.
    """
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


# Cap on BLOCK_C; beyond this, the (BLOCK_C, BLOCK_H) tile is too large
# to hold in registers efficiently and the eager fallback wins.  In
# practice clampable feature counts are small (≤ 64), so this cap only
# bites in pathological configurations.
_MAX_BLOCK_C = 256


def _choose_block_c(n_clamp: int) -> int:
    """Pick ``BLOCK_C`` for the kernel given ``n_clamp`` (≥ 1)."""
    return _next_power_of_two(max(n_clamp, 1))


def _kernel_supports(n_clamp: int) -> bool:
    """Return True iff the Triton kernel can handle this clamp count.

    Beyond ``_MAX_BLOCK_C`` we fall back to the eager path; the kernel
    body would still produce the right answer but the register tile
    becomes large enough that the eager GEMM path is competitive and
    less risky.
    """
    return _choose_block_c(n_clamp) <= _MAX_BLOCK_C


def apply_sae_delta_triton(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Compute the SAE feature-surgery delta on CUDA via a Triton kernel.

    Inputs match the eager reference :func:`apply_sae_delta` (except for
    the activation enum being replaced by an integer code and a single
    float parameter; see :mod:`sae_steering` for the encoding).  The
    output is a freshly allocated tensor with the same shape and dtype
    as ``hidden_states``.

    Empty token batches and empty clamp sets short-circuit before the
    kernel launch — Triton can fail on zero-sized grids and the math is
    a no-op in either case.
    """
    out = torch.empty_like(hidden_states)
    n_tokens = hidden_states.shape[0]
    if n_tokens == 0:
        return out
    n_clamp = encoder_weight.shape[0]
    if n_clamp == 0:
        out.copy_(hidden_states)
        return out

    h_size = hidden_states.shape[1]
    block_h = _choose_block_h(h_size)
    block_c = _choose_block_c(n_clamp)

    # Bool tensors map to 1-byte storage; ``view(int8)`` is zero-copy
    # and lets Triton load into an int8 register (operands cast to int32
    # inside the kernel).
    only_int = clamp_only_if_active.view(torch.int8)

    _apply_sae_delta_kernel[(n_tokens,)](
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        clamp_kind,
        clamp_value,
        only_int,
        out,
        n_tokens,
        h_size,
        n_clamp,
        hidden_states.stride(0),
        hidden_states.stride(1),
        encoder_weight.stride(0),
        encoder_weight.stride(1),
        encoder_bias.stride(0),
        decoder_weight.stride(0),
        decoder_weight.stride(1),
        clamp_kind.stride(0),
        clamp_kind.stride(1),
        clamp_value.stride(0),
        clamp_value.stride(1),
        only_int.stride(0),
        only_int.stride(1),
        out.stride(0),
        out.stride(1),
        float(activation_param),
        ACTIVATION_CODE=int(activation_code),
        BLOCK_H=block_h,
        BLOCK_C=block_c,
    )
    return out


def warmup_apply_sae_delta_kernel(
    *,
    hidden_size: int,
    n_clamp: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    activation_code: int = ACTIVATION_CODE_RELU,
    activation_param: float = 0.0,
) -> None:
    """JIT-compile the SAE kernel ahead of CUDA-graph capture.

    Mirrors :func:`steering_kernel.warmup_apply_steering_kernel`.  A
    tiny single-token launch is enough for Triton's first-call JIT to
    happen outside any captured forward, so subsequent CUDA-graph
    capture steps don't trigger a compile mid-capture.

    The warmup binary specialises on ``activation_code``, ``BLOCK_H``,
    and ``BLOCK_C``; the production launch reuses the cached binary
    when those constexprs match.  We therefore warm up once per
    activation-code site at startup; mismatches re-JIT lazily but
    outside graph capture (per the warmup contract).
    """
    if device.type != "cuda":
        return
    if n_clamp <= 0:
        return
    dummy_hidden = torch.zeros(1, hidden_size, dtype=compute_dtype, device=device)
    dummy_enc_w = torch.zeros(n_clamp, hidden_size, dtype=table_dtype, device=device)
    dummy_enc_b = torch.zeros(n_clamp, dtype=table_dtype, device=device)
    dummy_dec_w = torch.zeros(n_clamp, hidden_size, dtype=table_dtype, device=device)
    dummy_kind = torch.zeros(1, n_clamp, dtype=torch.int8, device=device)
    dummy_value = torch.zeros(1, n_clamp, dtype=torch.float32, device=device)
    dummy_only = torch.zeros(1, n_clamp, dtype=torch.bool, device=device)
    apply_sae_delta_triton(
        dummy_hidden,
        dummy_enc_w,
        dummy_enc_b,
        dummy_dec_w,
        dummy_kind,
        dummy_value,
        dummy_only,
        activation_code,
        activation_param,
    )
