# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the activation-patching custom ops.

Activation patching overwrites (or interpolates toward) the residual-stream
activation at selected token rows with a source vector captured from a prior
"clean" run. Two ops back the feature:

- :func:`apply_patch_triton` — single-tensor lerp for the ``pre_attn`` /
  ``post_attn`` hook points, where the residual *is* the quantity to patch::

      out[t] = lerp(hs[t], table[idx[t]], alpha[idx[t]])
             = hs[t] + alpha[idx[t]] * (table[idx[t]] - hs[t])

- :func:`apply_patch_block_triton` — two-tensor variant for ``post_block``.
  vLLM defers each layer's MLP-branch add into the next fused add+norm, so the
  true block output is ``residual + hidden_states`` while the op acts on
  ``residual`` alone. Additive steering commutes through the deferred add, but
  replace/lerp does **not**, so this op reconstructs the block output, lerps
  that, and folds the delta back into ``residual``::

      out_res[t] = res[t] + alpha[idx[t]] * (table[idx[t]] - (res[t] + hs[t]))
      # block_out = out_res + hs = lerp(res + hs, table, alpha)

Both kernels mirror the steering kernel contract: one program per token row
(``grid = (N,)``), a masked ``BLOCK_H`` walk over the hidden dim, a freshly
allocated output (never in place) to keep ``torch.compile`` value semantics,
and two passthrough layers — a whole-step ``any_active`` short-circuit and a
per-token ``slot == 0`` skip. Slot 0 is the passthrough sentinel; ``alpha[0]``
is pinned to ``0.0`` so even a stray index on slot 0 yields the input.

Layout assumptions match the steering kernel: row-contiguous ``[N, H]`` tensors
in compute dtype; ``table`` row-contiguous ``[num_slots, H]`` cast to the
hidden dtype inside the kernel; ``index`` int32 (only the first ``N`` entries
read); ``alpha`` fp32 ``[num_slots]``.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.steering_kernel import _choose_block_h
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _apply_patch_kernel(
    hidden_ptr,
    table_ptr,
    index_ptr,
    alpha_ptr,
    active_ptr,
    out_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    t_stride_r,
    t_stride_h,
    o_stride_n,
    o_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Single-tensor lerp: ``out[i] = lerp(hidden[i], table[idx[i]], a[idx[i]])``.

    The ``slot == 0`` branch is uniform within a program (one program per row,
    every lane reads the same ``slot``), so it introduces no warp divergence
    and skips the table gather + cast entirely for unpatched rows.
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    active = tl.load(active_ptr)
    slot = 0
    if active != 0:
        slot = tl.load(index_ptr + pid_n)

    if slot == 0:
        # Passthrough: whole-step inactive, or this row is unpatched.
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
            tl.store(out_row_ptr + h_idx * o_stride_h, h_vals, mask=mask)
        return

    alpha = tl.load(alpha_ptr + slot)
    table_row_ptr = table_ptr + slot * t_stride_r

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
        t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)
        a = alpha.to(h_vals.dtype)
        # Precise lerp form: exact at the endpoints (a==1 -> table, a==0 -> hs),
        # unlike ``h + a*(t-h)`` which loses the endpoints to rounding.
        result = (1.0 - a) * h_vals + a * t_vals.to(h_vals.dtype)
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)


@triton.jit
def _apply_patch_block_kernel(
    hidden_ptr,
    residual_ptr,
    table_ptr,
    index_ptr,
    alpha_ptr,
    active_ptr,
    out_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    r_stride_n,
    r_stride_h,
    t_stride_r,
    t_stride_h,
    o_stride_n,
    o_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Two-tensor block patch over the deferred-MLP-add residual.

    ``out_res[i] = res[i] + a[idx[i]] * (table[idx[i]] - (res[i] + hidden[i]))``
    so that ``out_res + hidden == lerp(res + hidden, table, a)``. ``hidden`` is
    not modified (the caller's deferred add still lands correctly).
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    residual_row_ptr = residual_ptr + pid_n * r_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    active = tl.load(active_ptr)
    slot = 0
    if active != 0:
        slot = tl.load(index_ptr + pid_n)

    if slot == 0:
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            r_vals = tl.load(residual_row_ptr + h_idx * r_stride_h, mask=mask)
            tl.store(out_row_ptr + h_idx * o_stride_h, r_vals, mask=mask)
        return

    alpha = tl.load(alpha_ptr + slot)
    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    table_row_ptr = table_ptr + slot * t_stride_r

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        r_vals = tl.load(residual_row_ptr + h_idx * r_stride_h, mask=mask)
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
        t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)
        a = alpha.to(r_vals.dtype)
        # out_res such that out_res + h == lerp(r + h, table, a). Written as
        # (1-a)*r + a*(t-h) so a==0 yields r exactly (passthrough).
        result = (1.0 - a) * r_vals + a * (t_vals.to(r_vals.dtype) - h_vals)
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)


def apply_patch_triton(
    hidden_states: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Single-tensor lerp patch. Returns a fresh output, never in place.

    Empty batches (``N == 0``) short-circuit without launching the kernel —
    Triton can fail on zero-sized grids.
    """
    out = torch.empty_like(hidden_states)
    N = hidden_states.shape[0]
    if N == 0:
        return out

    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _apply_patch_kernel[(N,)](
        hidden_states,
        patch_table,
        patch_index,
        patch_alpha,
        any_active,
        out,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        patch_table.stride(0),
        patch_table.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def apply_patch_block_triton(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Two-tensor block patch. Returns a fresh patched ``residual``."""
    out = torch.empty_like(residual)
    N = residual.shape[0]
    if N == 0:
        return out

    H = residual.shape[1]
    block_h = _choose_block_h(H)

    _apply_patch_block_kernel[(N,)](
        hidden_states,
        residual,
        patch_table,
        patch_index,
        patch_alpha,
        any_active,
        out,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        residual.stride(0),
        residual.stride(1),
        patch_table.stride(0),
        patch_table.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def _default_warmup_sizes() -> list[int]:
    """Fallback warmup batch sizes when no capture-size list is supplied."""
    return [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def _kernel_cache_size() -> int:
    """Total compiled variants across devices for both patch kernels."""
    total = 0
    for kern in (_apply_patch_kernel, _apply_patch_block_kernel):
        cache = getattr(kern, "cache", None)
        if cache is None:
            continue
        for device_cache in cache.values():
            try:
                total += len(device_cache)
            except TypeError:
                continue
    return total


def warmup_apply_patch_kernel(
    *,
    hidden_size: int,
    table_slots: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile both patch kernels ahead of CUDA graph capture.

    Mirrors :func:`warmup_apply_steering_kernel`. Drives both ops at every
    batch dim vLLM will capture, exercising ``any_active in {0, 1}`` and both
    ``slot == 0`` (passthrough) and ``slot > 0`` (gather) rows, so first-call
    JIT cost never lands inside a captured forward (which would fail capture).
    Routes through the registered ops so the dispatched stride specialization
    matches runtime.
    """
    if device.type != "cuda":
        return

    sizes = capture_sizes if capture_sizes else _default_warmup_sizes()
    sizes = sorted({int(s) for s in sizes if int(s) > 0})
    if not sizes:
        return

    cache_before = _kernel_cache_size()
    max_n = max(sizes)
    n_slots = max(table_slots, 2)
    hidden_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    residual_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    table_buf = torch.zeros(n_slots, hidden_size, dtype=table_dtype, device=device)
    alpha_buf = torch.zeros(n_slots, dtype=torch.float32, device=device)
    active_flag = torch.zeros(1, dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    for n in sizes:
        hidden_view = hidden_buf[:n]
        residual_view = residual_buf[:n]
        # Index alternates slot 0 (passthrough) and slot 1 (gather) so both
        # per-token branches are exercised within a single launch.
        index_view = torch.arange(n, device=device, dtype=torch.int32) % 2
        # Inactive (whole-step short-circuit), then active.
        active_flag.fill_(False)
        torch.ops.vllm.apply_patch(
            hidden_view, table_buf, index_view, alpha_buf, active_flag
        )
        torch.ops.vllm.apply_patch_block(
            hidden_view, residual_view, table_buf, index_view, alpha_buf, active_flag
        )
        active_flag.fill_(True)
        torch.ops.vllm.apply_patch(
            hidden_view, table_buf, index_view, alpha_buf, active_flag
        )
        torch.ops.vllm.apply_patch_block(
            hidden_view, residual_view, table_buf, index_view, alpha_buf, active_flag
        )
    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    cache_after = _kernel_cache_size()
    logger.info(
        "patch kernel warmup: shapes=%d variants_compiled=%d "
        "cache_total=%d elapsed_ms=%.1f",
        len(sizes),
        cache_after - cache_before,
        cache_after,
        elapsed_ms,
    )

    if os.environ.get("VLLM_PATCH_DUMP_JIT_CACHE", "0") == "1":
        for kern in (_apply_patch_kernel, _apply_patch_block_kernel):
            cache = getattr(kern, "cache", None)
            if cache is None:
                continue
            for device_id, device_cache in cache.items():
                try:
                    keys = list(device_cache.keys())
                except AttributeError:
                    keys = []
                logger.info(
                    "patch JIT cache: kernel=%s device=%s variants=%d",
                    kern.__name__,
                    device_id,
                    len(keys),
                )
