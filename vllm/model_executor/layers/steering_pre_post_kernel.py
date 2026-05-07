# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel that fuses PRE_ATTN + POST_ATTN steering into one launch.

For models like Gemma3 where the residual skip tensor is *not* read or
mutated between the pre-attention and post-attention hook points, the
two steering applications are mathematically equivalent to one combined
update::

    new_residual = residual + table_pre[index[i]] + table_post[index[i]]

This kernel performs that combined gather+add in a single pass over the
residual tensor, eliminating one launch per decoder layer per forward
(28 launches per Gemma-3-4B forward — graph-stable, regardless of
whether either hook is active).

Layout assumptions match :mod:`steering_kernel`:

- ``hidden_states`` is row-contiguous ``[N, H]`` in compute dtype.
- Both tables are row-contiguous ``[num_rows, H]``; values are cast to
  ``hidden_states.dtype`` inside the kernel.
- ``steering_index`` is ``int64`` and may be longer than ``N``; only
  the first ``N`` entries are read.

The accumulation is performed in fp32 to keep numerics tight when the
compute dtype is bf16 / fp16; the result is cast back to the compute
dtype on store. The output is always a freshly allocated tensor.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _apply_steering_pre_post_kernel(
    hidden_ptr,
    table_pre_ptr,
    table_post_ptr,
    index_ptr,
    out_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    tp_stride_r,
    tp_stride_h,
    tq_stride_r,
    tq_stride_h,
    o_stride_n,
    o_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Compute ``out[i, j] = hidden[i, j] + pre[idx[i], j] + post[idx[i], j]``.

    Both table rows are loaded once per program; the add is performed in
    fp32 then cast back to the compute dtype on store.
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    row = tl.load(index_ptr + pid_n)

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    pre_row_ptr = table_pre_ptr + row * tp_stride_r
    post_row_ptr = table_post_ptr + row * tq_stride_r
    out_row_ptr = out_ptr + pid_n * o_stride_n

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
        pre_vals = tl.load(pre_row_ptr + h_idx * tp_stride_h, mask=mask)
        post_vals = tl.load(post_row_ptr + h_idx * tq_stride_h, mask=mask)
        # Accumulate in fp32 to preserve numerics when compute dtype is
        # bf16 / fp16; cast back to the hidden dtype on store.
        result_f32 = (
            h_vals.to(tl.float32) + pre_vals.to(tl.float32) + post_vals.to(tl.float32)
        )
        tl.store(
            out_row_ptr + h_idx * o_stride_h,
            result_f32.to(h_vals.dtype),
            mask=mask,
        )


def _choose_block_h(hidden_size: int) -> int:
    """Pick a sensible ``BLOCK_H`` for the kernel given the hidden size.

    Mirrors :func:`steering_kernel._choose_block_h` so the two kernels
    pick identical block sizes for matching hidden sizes.
    """
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


def apply_steering_pre_post_triton(
    hidden_states: torch.Tensor,
    table_pre: torch.Tensor,
    table_post: torch.Tensor,
    steering_index: torch.Tensor,
) -> torch.Tensor:
    """Compute ``hidden_states + table_pre[idx] + table_post[idx]`` fused.

    Returns a freshly allocated output tensor with the same shape and
    dtype as ``hidden_states``. Empty batches (``N == 0``) short-circuit
    without launching the kernel.
    """
    out = torch.empty_like(hidden_states)
    N = hidden_states.shape[0]
    if N == 0:
        return out

    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _apply_steering_pre_post_kernel[(N,)](
        hidden_states,
        table_pre,
        table_post,
        steering_index,
        out,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        table_pre.stride(0),
        table_pre.stride(1),
        table_post.stride(0),
        table_post.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def warmup_apply_steering_pre_post_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """JIT-compile the fused kernel ahead of CUDA graph capture.

    Mirrors :func:`steering_kernel.warmup_apply_steering_kernel`.
    """
    if device.type != "cuda":
        return
    dummy_hidden = torch.zeros(1, hidden_size, dtype=compute_dtype, device=device)
    rows = max(table_rows, 1)
    dummy_table_pre = torch.zeros(rows, hidden_size, dtype=table_dtype, device=device)
    dummy_table_post = torch.zeros(rows, hidden_size, dtype=table_dtype, device=device)
    dummy_index = torch.zeros(1, dtype=torch.long, device=device)
    apply_steering_pre_post_triton(
        dummy_hidden, dummy_table_pre, dummy_table_post, dummy_index
    )
