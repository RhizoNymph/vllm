# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel that fuses ``apply_steering`` with the next consumer's
fused-add RMSNorm.

The pre-fusion pattern (Gemma 3 ``post_attn`` site, mirrored in every
RMSNorm-using model that exercises the ``post_attn`` steering hook) is:

    residual = apply_steering(residual, table, index)
    hidden_states, residual = layernorm(hidden_states, residual)

Two opaque kernels per (hook, layer) per forward step:
    - ``apply_steering`` reads + writes the residual tensor
    - ``fused_add_rms_norm`` reads hidden + residual, writes a fresh
      normalized output (and updates residual in place)

This kernel collapses both into one launch with a single read of
``hidden_states`` and ``residual``, one indexed gather from the steering
table, and one fp32 variance accumulation.

Math (matches ``GemmaRMSNorm.forward(hidden_states, residual)`` chained
after ``apply_steering``):

    new_residual = hidden + residual + cast_to_compute(table[index[i]])
    var = mean(new_residual.fp32() ** 2)
    rstd = rsqrt(var + eps)
    out = (new_residual.fp32() * rstd) * (weight.fp32() + 1.0)
    return out.to(compute_dtype), new_residual

The ``(weight + 1.0)`` term encodes Gemma's "x * (1 + w) instead of
x * w" departure from vanilla RMSNorm (see
``vllm/model_executor/layers/layernorm.py`` ``GemmaRMSNorm``). This is
the only RMSNorm flavor wired up for the MVP — non-Gemma RMSNorm
follow-ups would either pass a flag or land a sibling kernel.

Layout assumptions:
- ``hidden_states`` is row-contiguous ``[N, H]`` in compute dtype
  (typically ``bfloat16``).
- ``residual`` is row-contiguous ``[N, H]`` in compute dtype.
- ``steering_table`` is row-contiguous ``[num_rows, H]`` in any dtype;
  values are cast to compute dtype inside the kernel.
- ``steering_index`` is ``int64`` and may be longer than ``N``; only
  the first ``N`` entries are read.
- ``weight`` is ``[H]`` in compute dtype (matches ``RMSNorm.weight``).

The kernel uses one program per token row (grid ``(N,)``) and walks the
hidden dimension in ``BLOCK_H`` chunks. Pass 1 builds ``new_residual``,
writes it to ``out_residual``, and accumulates the fp32 sum of squares.
Pass 2 re-loads ``new_residual`` from ``out_residual``, applies the
RMSNorm, and writes ``out_norm``. Two passes is simpler than online
variance and matches vLLM's CUDA ``fused_add_rms_norm`` pattern.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _apply_steering_and_norm_kernel(
    hidden_ptr,
    residual_ptr,
    table_ptr,
    index_ptr,
    weight_ptr,
    out_norm_ptr,
    out_residual_ptr,
    N,
    H,
    eps,
    h_stride_n,
    h_stride_h,
    r_stride_n,
    r_stride_h,
    t_stride_r,
    t_stride_h,
    on_stride_n,
    on_stride_h,
    or_stride_n,
    or_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Fused ``apply_steering`` + ``GemmaRMSNorm`` (residual variant)."""
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    row = tl.load(index_ptr + pid_n)

    hidden_row = hidden_ptr + pid_n * h_stride_n
    residual_row = residual_ptr + pid_n * r_stride_n
    table_row = table_ptr + row * t_stride_r
    out_norm_row = out_norm_ptr + pid_n * on_stride_n
    out_residual_row = out_residual_ptr + pid_n * or_stride_n

    # ---- Pass 1: hidden + residual + table[idx] -> out_residual,
    #              accumulate variance in fp32.
    var_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H

        h_vals = tl.load(hidden_row + h_idx * h_stride_h, mask=mask, other=0.0)
        r_vals = tl.load(residual_row + h_idx * r_stride_h, mask=mask, other=0.0)
        t_vals = tl.load(table_row + h_idx * t_stride_h, mask=mask, other=0.0)

        # Cast everything to the compute dtype implicitly via the add;
        # `t_vals.to(h_vals.dtype)` mirrors the existing apply_steering
        # kernel's defensive cast for dtype-mismatched tables.
        new_residual = h_vals + r_vals + t_vals.to(h_vals.dtype)
        tl.store(out_residual_row + h_idx * or_stride_h, new_residual, mask=mask)

        nr_fp32 = new_residual.to(tl.float32)
        var_acc += tl.where(mask, nr_fp32 * nr_fp32, 0.0)

    var = tl.sum(var_acc) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # ---- Pass 2: load new_residual, apply RMSNorm with (weight + 1),
    #              cast to compute dtype, store into out_norm.
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H

        nr = tl.load(out_residual_row + h_idx * or_stride_h, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + h_idx, mask=mask, other=0.0)

        # Gemma's `(1 + w) * x / rstd` performed in fp32, cast back at
        # the store. Matches GemmaRMSNorm.forward_native exactly:
        # `out.to(orig_dtype)` after a fp32 multiply by (weight + 1).
        normed = (nr.to(tl.float32) * rstd) * (w_vals.to(tl.float32) + 1.0)
        tl.store(out_norm_row + h_idx * on_stride_h, normed.to(nr.dtype), mask=mask)


def _choose_block_h(hidden_size: int) -> int:
    """Same policy as :mod:`steering_kernel`: round up to a power of two
    capped at 2048 so the kernel walks larger hidden sizes in chunks
    without churning the configuration cache.
    """
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


def apply_steering_and_norm_triton(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``GemmaRMSNorm(hidden + residual + table[idx[:N]])``.

    Returns ``(normed, new_residual)`` — both freshly allocated, both in
    ``hidden_states.dtype``. Empty batches short-circuit before the
    Triton launch (zero-grid launches can fail).
    """
    N = hidden_states.shape[0]
    out_norm = torch.empty_like(hidden_states)
    out_residual = torch.empty_like(residual)
    if N == 0:
        return out_norm, out_residual

    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _apply_steering_and_norm_kernel[(N,)](
        hidden_states,
        residual,
        steering_table,
        steering_index,
        weight,
        out_norm,
        out_residual,
        N,
        H,
        float(eps),
        hidden_states.stride(0),
        hidden_states.stride(1),
        residual.stride(0),
        residual.stride(1),
        steering_table.stride(0),
        steering_table.stride(1),
        out_norm.stride(0),
        out_norm.stride(1),
        out_residual.stride(0),
        out_residual.stride(1),
        BLOCK_H=block_h,
    )
    return out_norm, out_residual


def warmup_apply_steering_and_norm_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """JIT-compile the fused kernel ahead of CUDA graph capture.

    Called from ``_init_steering_state`` next to the
    ``warmup_apply_steering_kernel`` call, so first-call JIT happens
    before the captured forward path runs.
    """
    if device.type != "cuda":
        return
    dummy_hidden = torch.zeros(1, hidden_size, dtype=compute_dtype, device=device)
    dummy_residual = torch.zeros(1, hidden_size, dtype=compute_dtype, device=device)
    dummy_table = torch.zeros(
        max(table_rows, 1), hidden_size, dtype=table_dtype, device=device
    )
    dummy_index = torch.zeros(1, dtype=torch.long, device=device)
    dummy_weight = torch.zeros(hidden_size, dtype=weight_dtype, device=device)
    apply_steering_and_norm_triton(
        dummy_hidden, dummy_residual, dummy_table, dummy_index, dummy_weight, 1e-6
    )
