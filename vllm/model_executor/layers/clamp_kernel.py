# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the directional-clamp custom ops.

Directional clamping constrains the hidden state's scalar projection along
up to K unit directions per steering row (see
:mod:`vllm.model_executor.layers.clamp`).  Two ops back the feature:

- :func:`apply_clamp_triton` — single-tensor variant for ``pre_attn`` /
  ``post_attn``, where the residual *is* the quantity to clamp::

      p[t, k]  = hs[t] @ dirs[row[t], k]                (fp32 accumulate)
      delta    = strength[row, k] * (clip(p, lo, hi) - p)
      out[t]   = hs[t] + sum_k delta[k] * dirs[row[t], k]

- :func:`apply_clamp_block_triton` — two-tensor variant for ``post_block``.
  vLLM defers the MLP-branch add into the next fused add+norm, so the true
  block output is ``residual + hidden_states`` while the op acts on
  ``residual`` alone.  Clamping does not commute through the deferred add,
  so this variant measures the projection of the reconstructed block output
  and folds the correction into ``residual``::

      out_res[t] = res[t] + sum_k delta[k] * dirs[row[t], k]
      # delta from p = (res[t] + hs[t]) @ dirs[row[t], k]

Both kernels mirror the steering/patch kernel contract: one program per
token row (``grid = (N,)``), masked ``BLOCK_H`` walks over the hidden dim,
a freshly allocated output (never in place) for ``torch.compile`` value
semantics, a whole-step ``any_active`` short-circuit, and a per-token
``row == 0`` passthrough (row 0 is the no-steering sentinel whose dirs are
all-zero forever, so skipping it is exact).  The K direction lanes are a
``BLOCK_K`` (next pow2 of K) vector axis: pass 1 accumulates all K dots in
one walk over the hidden dim, pass 2 applies the summed correction.  The
correction is accumulated in fp32 and cast to the hidden dtype once before
the add — matching the eager reference in ``clamp.py`` exactly.

Layout assumptions: row-contiguous ``[N, H]`` hidden/residual in compute
dtype; ``dirs`` ``[rows, K, H]`` cast to fp32 in-kernel for the dot;
``bounds`` fp32 ``[rows, K, 2]``; ``strength`` fp32 ``[rows, K]``;
``steering_index`` int64 (the SHARED steering token->row buffer, only the
first ``N`` entries read).
"""

from __future__ import annotations

import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.steering_kernel import (
    _choose_block_h,
    _default_warmup_sizes,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _apply_clamp_kernel(
    hidden_ptr,
    dirs_ptr,
    bounds_ptr,
    strength_ptr,
    index_ptr,
    active_ptr,
    out_ptr,
    N,
    H,
    K,
    h_stride_n,
    h_stride_h,
    d_stride_r,
    d_stride_k,
    d_stride_h,
    b_stride_r,
    b_stride_k,
    b_stride_x,
    s_stride_r,
    s_stride_k,
    o_stride_n,
    o_stride_h,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``out[t] = hs[t] + sum_k strength*(clip(hs[t]@v_k, lo, hi) - hs[t]@v_k)*v_k``."""
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    # ``steering_index`` is int64; the initializer must match its dtype or
    # Triton rejects the branch redefinition (int32[] vs int64[]).
    active = tl.load(active_ptr)
    row = tl.full([], 0, tl.int64)
    if active != 0:
        row = tl.load(index_ptr + pid_n)

    if row == 0:
        # Passthrough: whole-step inactive, or the no-steering sentinel row
        # (dirs[0] is all-zero forever, so skipping is exact). Uniform within
        # a program — no divergence.
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            hmask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=hmask)
            tl.store(out_row_ptr + h_idx * o_stride_h, h_vals, mask=hmask)
        return

    k_idx = tl.arange(0, BLOCK_K)
    kmask = k_idx < K
    dirs_row_ptr = dirs_ptr + row * d_stride_r

    # Pass 1: all K projections in one walk over the hidden dim (fp32).
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        hmask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=hmask, other=0.0).to(
            tl.float32
        )
        d_vals = tl.load(
            dirs_row_ptr + k_idx[:, None] * d_stride_k + h_idx[None, :] * d_stride_h,
            mask=kmask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(h_vals[None, :] * d_vals, axis=1)

    lo = tl.load(
        bounds_ptr + row * b_stride_r + k_idx * b_stride_k, mask=kmask, other=0.0
    )
    hi = tl.load(
        bounds_ptr + row * b_stride_r + k_idx * b_stride_k + b_stride_x,
        mask=kmask,
        other=0.0,
    )
    strength = tl.load(
        strength_ptr + row * s_stride_r + k_idx * s_stride_k, mask=kmask, other=0.0
    )
    p_clamped = tl.minimum(tl.maximum(acc, lo), hi)
    delta = strength * (p_clamped - acc)
    delta = tl.where(kmask, delta, 0.0)

    # Pass 2: fold the summed fp32 correction back, cast once to hidden dtype.
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        hmask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=hmask)
        d_vals = tl.load(
            dirs_row_ptr + k_idx[:, None] * d_stride_k + h_idx[None, :] * d_stride_h,
            mask=kmask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        corr = tl.sum(delta[:, None] * d_vals, axis=0)
        result = h_vals + corr.to(h_vals.dtype)
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=hmask)


@triton.jit
def _apply_clamp_block_kernel(
    hidden_ptr,
    residual_ptr,
    dirs_ptr,
    bounds_ptr,
    strength_ptr,
    index_ptr,
    active_ptr,
    out_ptr,
    N,
    H,
    K,
    h_stride_n,
    h_stride_h,
    r_stride_n,
    r_stride_h,
    d_stride_r,
    d_stride_k,
    d_stride_h,
    b_stride_r,
    b_stride_k,
    b_stride_x,
    s_stride_r,
    s_stride_k,
    o_stride_n,
    o_stride_h,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """post_block variant: projection measured on ``residual + hidden``,
    correction folded into ``residual`` (``out + hidden == clamp(res + hs)``)."""
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    residual_row_ptr = residual_ptr + pid_n * r_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    # ``steering_index`` is int64; the initializer must match its dtype or
    # Triton rejects the branch redefinition (int32[] vs int64[]).
    active = tl.load(active_ptr)
    row = tl.full([], 0, tl.int64)
    if active != 0:
        row = tl.load(index_ptr + pid_n)

    if row == 0:
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            hmask = h_idx < H
            r_vals = tl.load(residual_row_ptr + h_idx * r_stride_h, mask=hmask)
            tl.store(out_row_ptr + h_idx * o_stride_h, r_vals, mask=hmask)
        return

    k_idx = tl.arange(0, BLOCK_K)
    kmask = k_idx < K
    dirs_row_ptr = dirs_ptr + row * d_stride_r

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        hmask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=hmask, other=0.0).to(
            tl.float32
        )
        r_vals = tl.load(
            residual_row_ptr + h_idx * r_stride_h, mask=hmask, other=0.0
        ).to(tl.float32)
        b_vals = r_vals + h_vals  # true block output, fp32
        d_vals = tl.load(
            dirs_row_ptr + k_idx[:, None] * d_stride_k + h_idx[None, :] * d_stride_h,
            mask=kmask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(b_vals[None, :] * d_vals, axis=1)

    lo = tl.load(
        bounds_ptr + row * b_stride_r + k_idx * b_stride_k, mask=kmask, other=0.0
    )
    hi = tl.load(
        bounds_ptr + row * b_stride_r + k_idx * b_stride_k + b_stride_x,
        mask=kmask,
        other=0.0,
    )
    strength = tl.load(
        strength_ptr + row * s_stride_r + k_idx * s_stride_k, mask=kmask, other=0.0
    )
    p_clamped = tl.minimum(tl.maximum(acc, lo), hi)
    delta = strength * (p_clamped - acc)
    delta = tl.where(kmask, delta, 0.0)

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        hmask = h_idx < H
        r_vals = tl.load(residual_row_ptr + h_idx * r_stride_h, mask=hmask)
        d_vals = tl.load(
            dirs_row_ptr + k_idx[:, None] * d_stride_k + h_idx[None, :] * d_stride_h,
            mask=kmask[:, None] & hmask[None, :],
            other=0.0,
        ).to(tl.float32)
        corr = tl.sum(delta[:, None] * d_vals, axis=0)
        result = r_vals + corr.to(r_vals.dtype)
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=hmask)


def _choose_clamp_blocks(hidden_size: int, k: int) -> tuple[int, int]:
    """Pick ``(BLOCK_K, BLOCK_H)`` for the clamp kernels.

    ``BLOCK_K`` is the next power of two >= K (Triton requires pow2 block
    axes).  ``BLOCK_H`` reuses the steering kernel's policy but is capped
    at 1024 because the dirs load is a 2-D ``(BLOCK_K, BLOCK_H)`` tile —
    K lanes multiply the per-iteration register footprint.
    """
    block_k = max(1, 1 << (max(1, k) - 1).bit_length())
    block_h = min(_choose_block_h(hidden_size), 1024)
    return block_k, block_h


def apply_clamp_triton(
    hidden_states: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """CUDA dispatch of :func:`vllm.model_executor.layers.clamp.apply_clamp`.

    Returns a freshly allocated output tensor with the same shape and dtype
    as ``hidden_states``.  Empty batches short-circuit without launching
    (Triton can fail on zero-sized grids).
    """
    out = torch.empty_like(hidden_states)
    N = hidden_states.shape[0]
    if N == 0:
        return out

    H = hidden_states.shape[1]
    K = clamp_dirs.shape[1]
    block_k, block_h = _choose_clamp_blocks(H, K)

    _apply_clamp_kernel[(N,)](
        hidden_states,
        clamp_dirs,
        clamp_bounds,
        clamp_strength,
        steering_index,
        any_active,
        out,
        N,
        H,
        K,
        hidden_states.stride(0),
        hidden_states.stride(1),
        clamp_dirs.stride(0),
        clamp_dirs.stride(1),
        clamp_dirs.stride(2),
        clamp_bounds.stride(0),
        clamp_bounds.stride(1),
        clamp_bounds.stride(2),
        clamp_strength.stride(0),
        clamp_strength.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_K=block_k,
        BLOCK_H=block_h,
    )
    return out


def apply_clamp_block_triton(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """CUDA dispatch of ``apply_clamp_block``; returns a fresh ``residual``."""
    out = torch.empty_like(residual)
    N = residual.shape[0]
    if N == 0:
        return out

    H = residual.shape[1]
    K = clamp_dirs.shape[1]
    block_k, block_h = _choose_clamp_blocks(H, K)

    _apply_clamp_block_kernel[(N,)](
        hidden_states,
        residual,
        clamp_dirs,
        clamp_bounds,
        clamp_strength,
        steering_index,
        any_active,
        out,
        N,
        H,
        K,
        hidden_states.stride(0),
        hidden_states.stride(1),
        residual.stride(0),
        residual.stride(1),
        clamp_dirs.stride(0),
        clamp_dirs.stride(1),
        clamp_dirs.stride(2),
        clamp_bounds.stride(0),
        clamp_bounds.stride(1),
        clamp_bounds.stride(2),
        clamp_strength.stride(0),
        clamp_strength.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_K=block_k,
        BLOCK_H=block_h,
    )
    return out


def warmup_apply_clamp_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    max_directions: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile both clamp kernels ahead of CUDA graph capture.

    Mirrors ``warmup_apply_steering_kernel``: drive every batch shape vLLM
    will subsequently capture, through the *registered ops* (the dispatched
    runtime path), in both ``any_active`` states, so no served request pays
    first-call JIT cost.  Buffer shapes must match what the runner keeps
    (``(rows, K, hidden)``) so Triton's stride specialization compiles the
    variant the runtime actually hits.
    """
    if device.type != "cuda" or max_directions <= 0:
        return

    sizes = capture_sizes if capture_sizes else _default_warmup_sizes()
    sizes = sorted({int(s) for s in sizes if int(s) > 0})
    if not sizes:
        return

    max_n = max(sizes)
    rows = max(table_rows, 1)
    hidden_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    residual_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    dirs_buf = torch.zeros(
        rows, max_directions, hidden_size, dtype=table_dtype, device=device
    )
    bounds_buf = torch.empty(
        rows, max_directions, 2, dtype=torch.float32, device=device
    )
    bounds_buf[..., 0] = -float("inf")
    bounds_buf[..., 1] = float("inf")
    strength_buf = torch.ones(rows, max_directions, dtype=torch.float32, device=device)
    index_buf = torch.zeros(max_n, dtype=torch.long, device=device)
    active_flag = torch.zeros(1, dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    for n in sizes:
        for active in (False, True):
            active_flag.fill_(active)
            torch.ops.vllm.apply_clamp(
                hidden_buf[:n],
                dirs_buf,
                bounds_buf,
                strength_buf,
                index_buf[:n],
                active_flag,
            )
            torch.ops.vllm.apply_clamp_block(
                hidden_buf[:n],
                residual_buf[:n],
                dirs_buf,
                bounds_buf,
                strength_buf,
                index_buf[:n],
                active_flag,
            )
    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "clamp kernel warmup: shapes=%d K=%d elapsed_ms=%.1f",
        len(sizes),
        max_directions,
        elapsed_ms,
    )
