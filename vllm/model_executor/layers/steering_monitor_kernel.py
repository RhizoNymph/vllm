# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the in-graph steering monitor op (Phase 2, §8).

The monitor runs at a probe site (one ``(layer, hook)``) and turns the
runner-written flat decode gate in ``steering_token_scales`` into a
*per-token* gate computed from a probe, in the same forward pass:

    score[i] = sum_h hidden[i, h] * probe[h]            # fp32 reduction
    gate[i]  = sigmoid(sharpness * (score[i] - threshold))
    token_scales[i] *= gate[i]                          # read-modify-write

The multiply (not overwrite) is deliberate — the runner writes
``token_scales`` fresh each step (``gain`` for decode tokens, ``0`` for
prefill), which is the per-step reset; the monitor modulates within the
step. Prefill cache-safety (§7) holds because ``0 * gate == 0``
regardless of the probe.

Graph safety: fixed-shape loads against persistent buffers, no
allocation, no host sync, no cross-token reduction (each program owns one
token), no data-dependent control flow beyond the ``monitor_active``
short-circuit (a tensor flag, so the compiled topology is stable). Every
TP rank records it; the residual is rank-identical post-all-reduce and
the probe is replicated, so every rank computes the same gate (§3.1).

Layout assumptions mirror ``steering_kernel.py``:

- ``hidden_states`` is row-contiguous ``[N, H]`` in compute dtype.
- ``probe`` is contiguous ``[H]`` (fp32).
- ``token_scales`` is contiguous ``[max_tokens]`` (fp32); only the first
  ``N`` entries are touched.
"""

from __future__ import annotations

import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _steering_monitor_kernel(
    hidden_ptr,
    probe_ptr,
    params_ptr,
    active_ptr,
    tscale_ptr,
    dmask_ptr,
    rgate_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    p_stride_h,
    ts_stride_n,
    dm_stride_n,
    rg_stride_n,
    BLOCK_H: tl.constexpr,
):
    """Per-token gate: ``token_scales[i] *= g`` (tier) and, when
    ``params[2]`` (gate_rows) is set, ``row_gate[i] *= mask[i]·g + (1−mask[i])``
    (per-request row term, decode-only via the mask). ``g = sigmoid(sharp·
    (hidden[i]@probe − thr))``.

    One program per token row. When the byte at ``active_ptr`` is zero the
    kernel returns immediately, leaving both gates as the runner wrote them.
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    active = tl.load(active_ptr)
    if active == 0:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n

    # fp32 dot product over the hidden dim (one token's score).
    acc = tl.zeros((), dtype=tl.float32)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask, other=0.0)
        p_vals = tl.load(probe_ptr + h_idx * p_stride_h, mask=mask, other=0.0)
        acc += tl.sum(h_vals.to(tl.float32) * p_vals.to(tl.float32))

    threshold = tl.load(params_ptr + 0)
    sharpness = tl.load(params_ptr + 1)
    gate_rows = tl.load(params_ptr + 2)
    gate = tl.sigmoid(sharpness * (acc - threshold))

    ts_ptr = tscale_ptr + pid_n * ts_stride_n
    tl.store(ts_ptr, tl.load(ts_ptr) * gate)

    if gate_rows != 0.0:
        # decode → ·gate ; prefill (mask 0) → ·1 (row stays full strength).
        dm = tl.load(dmask_ptr + pid_n * dm_stride_n)
        rg_ptr = rgate_ptr + pid_n * rg_stride_n
        tl.store(rg_ptr, tl.load(rg_ptr) * (dm * gate + (1.0 - dm)))


def _choose_block_h(hidden_size: int) -> int:
    """Pick a ``BLOCK_H`` for the reduction; mirrors steering_kernel.py."""
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


def steering_monitor_triton(
    hidden_states: torch.Tensor,
    probe: torch.Tensor,
    params: torch.Tensor,
    monitor_active: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_decode_mask: torch.Tensor,
    steering_row_gate: torch.Tensor,
) -> None:
    """In-place per-token gate write into ``steering_token_scales[:N]`` and,
    when ``params[2]`` is set, ``steering_row_gate[:N]`` (decode-only via
    ``steering_decode_mask``).

    Empty batches (``N == 0``) short-circuit — Triton can fail on
    zero-sized grids. ``monitor_active`` is a single-element bool tensor;
    when ``False`` the kernel launches but returns without touching either
    gate.
    """
    N = hidden_states.shape[0]
    if N == 0:
        return
    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _steering_monitor_kernel[(N,)](
        hidden_states,
        probe,
        params,
        monitor_active,
        steering_token_scales,
        steering_decode_mask,
        steering_row_gate,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        probe.stride(0),
        steering_token_scales.stride(0),
        steering_decode_mask.stride(0),
        steering_row_gate.stride(0),
        BLOCK_H=block_h,
    )


def warmup_steering_monitor_kernel(
    *,
    hidden_size: int,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile the monitor kernel ahead of CUDA graph capture.

    Routes warmup through the registered op
    (``torch.ops.vllm.steering_monitor``) for the same reason the steering
    warmup does — to compile the stride-class specialization the dispatched
    runtime call triggers, not a different one. Both ``monitor_active``
    states are driven (they share the compiled artifact; the flag is a
    tensor, not a constexpr).
    """
    if device.type != "cuda":
        return
    sizes = capture_sizes if capture_sizes else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    sizes = sorted({int(s) for s in sizes if int(s) > 0})
    if not sizes:
        return

    max_n = max(sizes)
    hidden_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    probe_buf = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    # [threshold, sharpness, gate_rows]; gate_rows=1 so the row-gating
    # branch is compiled too (the active/inactive flag shares the artifact).
    params_buf = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32, device=device)
    active_flag = torch.zeros(1, dtype=torch.bool, device=device)
    tscale_buf = torch.zeros(max_n, dtype=torch.float32, device=device)
    dmask_buf = torch.zeros(max_n, dtype=torch.float32, device=device)
    rgate_buf = torch.ones(max_n, dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    for n in sizes:
        hidden_view = hidden_buf[:n]
        tscale_view = tscale_buf[:n]
        dmask_view = dmask_buf[:n]
        rgate_view = rgate_buf[:n]
        active_flag.fill_(False)
        torch.ops.vllm.steering_monitor(
            hidden_view, probe_buf, params_buf, active_flag, tscale_view,
            dmask_view, rgate_view,
        )
        active_flag.fill_(True)
        torch.ops.vllm.steering_monitor(
            hidden_view, probe_buf, params_buf, active_flag, tscale_view,
            dmask_view, rgate_view,
        )
    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "steering monitor kernel warmup: shapes=%d elapsed_ms=%.1f",
        len(sizes),
        elapsed_ms,
    )
