# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for ``torch.ops.vllm.apply_neuron_clamp``.

One program per token row. Each program reads the device-side
``any_active`` flag first and exits immediately when it is clear, so the
inactive cost is one tiny launch with no activation traffic. When active,
the program gathers its steering row, loads that row's ``K`` clamp slots
and rewrites at most ``K`` scalars of the activation row in place.
"""

from __future__ import annotations

import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _apply_neuron_clamp_kernel(
    act_ptr,
    cidx_ptr,
    cmul_ptr,
    cadd_ptr,
    sidx_ptr,
    active_ptr,
    a_stride_n,
    a_stride_i,
    c_stride_r,
    c_stride_k,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    active = tl.load(active_ptr)
    if active == 0:
        return
    row = tl.load(sidx_ptr + pid_n)
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    slot = row * c_stride_r + k * c_stride_k
    nidx = tl.load(cidx_ptr + slot, mask=kmask, other=-1)
    valid = nidx >= 0
    mul = tl.load(cmul_ptr + slot, mask=kmask, other=1.0)
    add = tl.load(cadd_ptr + slot, mask=kmask, other=0.0)
    safe_idx = tl.where(valid, nidx, 0).to(tl.int64)
    ptrs = act_ptr + pid_n.to(tl.int64) * a_stride_n + safe_idx * a_stride_i
    cur = tl.load(ptrs, mask=valid, other=0.0).to(tl.float32)
    new = tl.where(mul == 0.0, add, cur * mul + add)
    tl.store(ptrs, new.to(act_ptr.dtype.element_ty), mask=valid)


def _block_k(k: int) -> int:
    return max(1, 1 << (max(k, 1) - 1).bit_length())


def apply_neuron_clamp_triton(
    act: torch.Tensor,
    clamp_index: torch.Tensor,
    clamp_mul: torch.Tensor,
    clamp_add: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> None:
    """In-place clamp of ``act`` (``[N, I_local]``). ``N == 0`` is a no-op."""
    n = act.shape[0]
    if n == 0:
        return
    k = clamp_index.shape[1]
    assert clamp_mul.stride() == clamp_index.stride()
    assert clamp_add.stride() == clamp_index.stride()
    _apply_neuron_clamp_kernel[(n,)](
        act,
        clamp_index,
        clamp_mul,
        clamp_add,
        steering_index,
        any_active,
        act.stride(0),
        act.stride(1),
        clamp_index.stride(0),
        clamp_index.stride(1),
        k,
        BLOCK_K=_block_k(k),
    )


def warmup_neuron_clamp_kernel(
    *,
    intermediate_size_local: int,
    table_rows: int,
    max_clamps_per_layer: int,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile the clamp kernel before CUDA-graph capture.

    Routed through ``torch.ops.vllm.apply_neuron_clamp`` (the runtime
    dispatch path) for both flag states at every captured batch size.
    """
    if device.type != "cuda" or max_clamps_per_layer <= 0:
        return
    sizes = sorted({int(s) for s in (capture_sizes or [1, 8, 64, 256]) if int(s) > 0})
    if not sizes:
        return
    max_n = max(sizes)
    act = torch.zeros(
        max_n, intermediate_size_local, dtype=compute_dtype, device=device
    )
    cidx = torch.full(
        (table_rows, max_clamps_per_layer), -1, dtype=torch.int32, device=device
    )
    cmul = torch.ones(
        table_rows, max_clamps_per_layer, dtype=torch.float32, device=device
    )
    cadd = torch.zeros(
        table_rows, max_clamps_per_layer, dtype=torch.float32, device=device
    )
    sidx = torch.zeros(max_n, dtype=torch.long, device=device)
    flag = torch.zeros(1, dtype=torch.bool, device=device)
    t0 = time.perf_counter()
    for n in sizes:
        for state in (False, True):
            flag.fill_(state)
            torch.ops.vllm.apply_neuron_clamp(act[:n], cidx, cmul, cadd, sidx[:n], flag)
    torch.accelerator.synchronize()
    logger.info(
        "neuron clamp kernel warmup: shapes=%d elapsed_ms=%.1f",
        len(sizes),
        (time.perf_counter() - t0) * 1000.0,
    )
