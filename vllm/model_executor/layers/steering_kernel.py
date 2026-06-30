# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for the per-request activation-steering custom op.

The kernel fuses the gather, dtype cast, and add operations performed by
``apply_steering`` into a single launch and a single pass over
``hidden_states``. The eager Python implementation produces a fresh
output tensor (matching ``hidden_states.dtype``); this kernel preserves
that contract — output is written to a freshly allocated tensor, never
in place, to keep the ``torch.compile`` graph contract stable.

Layout assumptions:

- ``hidden_states`` is row-contiguous ``[N, H]`` in compute dtype.
- ``steering_table`` is row-contiguous ``[num_rows, H]`` in any dtype;
  values are cast to ``hidden_states.dtype`` inside the kernel.
- ``steering_index`` is ``int64`` and may be longer than ``N``; only
  the first ``N`` entries are read.

The kernel launches one program per token row (``grid = (N,)``) and
walks the hidden dimension in ``BLOCK_H`` chunks with masked loads
and stores so non-power-of-two hidden sizes are handled correctly.
"""

from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _apply_steering_kernel(
    hidden_ptr,
    table_ptr,
    index_ptr,
    active_ptr,
    scales_ptr,
    dvec_ptr,
    tscale_ptr,
    rgate_ptr,
    probe_ptr,
    mparams_ptr,
    mactive_ptr,
    dmask_ptr,
    rprobe_ptr,
    rparams_ptr,
    ractive_ptr,
    out_ptr,
    N,
    H,
    h_stride_n,
    h_stride_h,
    t_stride_r,
    t_stride_h,
    s_stride_r,
    dv_stride_h,
    ts_stride_n,
    rg_stride_n,
    p_stride_h,
    dm_stride_n,
    rp_stride_r,
    rp_stride_h,
    rpp_stride_r,
    o_stride_n,
    o_stride_h,
    BLOCK_H: tl.constexpr,
):
    """Compute ``out = hidden + table[index]*scale*row_gate + dvec*token_scale``
    with the in-graph monitor gate **fused in** (non-mutating).

    - the per-row gather ``table[index[i]] * scales[index[i]] * row_gate[i]``
      (§5.3 per-row scale + Phase-2 row gate), plus
    - the **dedicated dynamic tier** ``dvec * token_scales[i]`` (§5.4):
      ``token_scales[i] = 0`` (default / prefill) ⇒ no tier, and
    - the **fused monitor gate**: when ``mactive`` is set, compute
      ``g = sigmoid(sharp·(hidden[i]@probe − thr))`` from the *pre-steering*
      residual and fold it into ``token_scale`` (always) and ``row_gate``
      (when ``params[2]`` gate_rows, decode-only via ``dmask``) — locally,
      in registers. Nothing is written back to the shared
      ``token_scales``/``row_gate`` buffers, so the op stays non-mutating
      (cudagraph-fusable). The gate affects only this ``(layer, hook)``
      (same-hook gating); cross-layer gating is handled separately.

    When the byte at ``active_ptr`` is zero, the kernel skips the gather
    and emits ``out[i, j] = hidden[i, j]``. The ``mactive`` reduction is
    also skipped when inactive, so an unconfigured monitor costs nothing.
    """
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    hidden_row_ptr = hidden_ptr + pid_n * h_stride_n
    out_row_ptr = out_ptr + pid_n * o_stride_n

    active = tl.load(active_ptr)
    if active == 0:
        # Inactive: skip the table gather and the dtype cast entirely.
        # We still must produce ``out == hidden_states`` so the gather-
        # path callers see consistent value semantics; this branch
        # eliminates the table memory traffic and the cast, which is
        # the dominant cost when the table is bf16/fp16 and hidden_size
        # is large.  Combine with the in-place sibling branch
        # (``mutates_args=["hidden_states"]``) for a full skip with no
        # memcpy.
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
            tl.store(out_row_ptr + h_idx * o_stride_h, h_vals, mask=mask)
        return

    row = tl.load(index_ptr + pid_n)
    table_row_ptr = table_ptr + row * t_stride_r
    # Per-row scale (runtime "how much" knob); fp32, default 1.0.
    scale = tl.load(scales_ptr + row * s_stride_r)
    # Dedicated dynamic tier (§5.4): per-token gate, fp32, default 0.0
    # (and 0.0 for prefill tokens) ⇒ the tier add is a no-op.
    tscale = tl.load(tscale_ptr + pid_n * ts_stride_n)
    # Per-token row gate (Phase 2 row gating): fp32, default 1.0 (and 1.0
    # for prefill tokens) ⇒ the row applies at full strength.
    rgate = tl.load(rgate_ptr + pid_n * rg_stride_n)

    # Fused in-graph monitor (§8): gate from the pre-steering residual,
    # folded into tscale/rgate locally (never written back ⇒ non-mutating).
    # Skipped entirely when inactive, so it's free unless a probe is set here.
    mactive = tl.load(mactive_ptr)
    if mactive != 0:
        acc = tl.zeros((), dtype=tl.float32)
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask, other=0.0)
            p_vals = tl.load(probe_ptr + h_idx * p_stride_h, mask=mask, other=0.0)
            acc += tl.sum(h_vals.to(tl.float32) * p_vals.to(tl.float32))
        threshold = tl.load(mparams_ptr + 0)
        sharpness = tl.load(mparams_ptr + 1)
        gate_rows = tl.load(mparams_ptr + 2)
        gate = tl.sigmoid(sharpness * (acc - threshold))
        tscale = tscale * gate
        if gate_rows != 0.0:
            # decode → ·gate ; prefill (mask 0) → ·1 (row stays full strength).
            dm = tl.load(dmask_ptr + pid_n * dm_stride_n)
            rgate = rgate * (dm * gate + (1.0 - dm))

    # Per-row (per-request) monitor: gate THIS token's row term by its own
    # row's probe + params (probe_table[row], row_params[row]). Orthogonal to
    # the global monitor above; decode-only. Skipped when inactive (incl. the
    # (1,1) dummy buffers when the row monitor is disabled).
    ractive = tl.load(ractive_ptr)
    if ractive != 0:
        racc = tl.zeros((), dtype=tl.float32)
        rprobe_row_ptr = rprobe_ptr + row * rp_stride_r
        for h_off in range(0, H, BLOCK_H):
            h_idx = h_off + tl.arange(0, BLOCK_H)
            mask = h_idx < H
            h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask, other=0.0)
            rp_vals = tl.load(
                rprobe_row_ptr + h_idx * rp_stride_h, mask=mask, other=0.0
            )
            racc += tl.sum(h_vals.to(tl.float32) * rp_vals.to(tl.float32))
        rthr = tl.load(rparams_ptr + row * rpp_stride_r + 0)
        rsharp = tl.load(rparams_ptr + row * rpp_stride_r + 1)
        rgval = tl.sigmoid(rsharp * (racc - rthr))
        dm = tl.load(dmask_ptr + pid_n * dm_stride_n)
        rgate = rgate * (dm * rgval + (1.0 - dm))

    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + tl.arange(0, BLOCK_H)
        mask = h_idx < H
        h_vals = tl.load(hidden_row_ptr + h_idx * h_stride_h, mask=mask)
        t_vals = tl.load(table_row_ptr + h_idx * t_stride_h, mask=mask)
        d_vals = tl.load(dvec_ptr + h_idx * dv_stride_h, mask=mask)
        # Cast table values to hidden dtype so dtype-mismatched tables
        # (fp32 table + bf16 hidden, common before PR 1 lands) work.
        result = (
            h_vals
            + t_vals.to(h_vals.dtype) * scale.to(h_vals.dtype) * rgate.to(h_vals.dtype)
            + d_vals.to(h_vals.dtype) * tscale.to(h_vals.dtype)
        )
        tl.store(out_row_ptr + h_idx * o_stride_h, result, mask=mask)


def _choose_block_h(hidden_size: int) -> int:
    """Pick a sensible ``BLOCK_H`` for the kernel given the hidden size.

    For small hidden sizes (< 2048) round up to the next power of two so
    a single iteration covers the row. For larger hidden sizes cap at
    2048 — the loop in the kernel handles multi-iteration walks.

    Uses a manual power-of-two computation rather than
    ``triton.next_power_of_2`` so the module remains importable on
    environments where Triton is disabled (e.g. CPU-only test runs);
    the kernel itself is only ever launched on CUDA.
    """
    if hidden_size >= 2048:
        return 2048
    if hidden_size <= 1:
        return 1
    return 1 << (hidden_size - 1).bit_length()


def apply_steering_triton(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_scales: torch.Tensor,
    steering_dynamic_vec: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_row_gate: torch.Tensor,
    steering_monitor_probe: torch.Tensor,
    steering_monitor_params: torch.Tensor,
    steering_monitor_active: torch.Tensor,
    steering_decode_mask: torch.Tensor,
    steering_monitor_probe_table: torch.Tensor,
    steering_monitor_row_params: torch.Tensor,
    steering_monitor_row_active: torch.Tensor,
) -> torch.Tensor:
    """Compute the steered output with the fused in-graph monitor gate.

    ``hidden + table[idx]*scales[idx]*row_gate[:N] + dvec*token_scales[:N]``,
    with ``row_gate``/``token_scales`` modulated per token by the monitor
    gate ``sigmoid(sharp·(hidden@probe − thr))`` when
    ``steering_monitor_active`` is set (computed in-kernel, never written
    back ⇒ non-mutating).

    Returns a freshly allocated output tensor with the same shape and
    dtype as ``hidden_states``. Empty batches (``N == 0``) short-circuit
    without launching the kernel — Triton can fail on zero-sized grids.

    ``any_active`` is a single-element bool tensor; when ``False`` the
    kernel still launches but skips the table gather and emits
    ``hidden_states`` into the freshly-allocated output.
    """
    out = torch.empty_like(hidden_states)
    N = hidden_states.shape[0]
    if N == 0:
        return out

    H = hidden_states.shape[1]
    block_h = _choose_block_h(H)

    _apply_steering_kernel[(N,)](
        hidden_states,
        steering_table,
        steering_index,
        any_active,
        steering_scales,
        steering_dynamic_vec,
        steering_token_scales,
        steering_row_gate,
        steering_monitor_probe,
        steering_monitor_params,
        steering_monitor_active,
        steering_decode_mask,
        steering_monitor_probe_table,
        steering_monitor_row_params,
        steering_monitor_row_active,
        out,
        N,
        H,
        hidden_states.stride(0),
        hidden_states.stride(1),
        steering_table.stride(0),
        steering_table.stride(1),
        steering_scales.stride(0),
        steering_dynamic_vec.stride(0),
        steering_token_scales.stride(0),
        steering_row_gate.stride(0),
        steering_monitor_probe.stride(0),
        steering_decode_mask.stride(0),
        steering_monitor_probe_table.stride(0),
        steering_monitor_probe_table.stride(1),
        steering_monitor_row_params.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def _default_warmup_sizes() -> list[int]:
    """Fallback warmup batch sizes when no capture-size list is supplied.

    Mirrors the powers-of-two and small-batch shapes that vLLM commonly
    captures when ``cudagraph_capture_sizes`` is left to its default.
    Used only when the caller cannot pass an explicit list (e.g.
    standalone tests).
    """
    return [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def _dump_jit_cache_keys() -> None:
    """One-shot diagnostic — log the keys present in the kernel cache.

    Enabled when ``VLLM_STEERING_DUMP_JIT_CACHE=1``.  Walks
    ``_apply_steering_kernel.cache`` (a per-device dict from Triton's
    JITFunction) and emits each variant key at INFO so a benchmark run
    can reveal what specializations Triton actually built.
    """
    cache = getattr(_apply_steering_kernel, "cache", None)
    if cache is None:
        logger.info(
            "steering JIT cache dump requested but kernel has no cache "
            "attribute (Triton may be disabled)"
        )
        return
    total = 0
    for device_id, device_cache in cache.items():
        try:
            keys = list(device_cache.keys())
        except AttributeError:
            keys = []
        total += len(keys)
        logger.info(
            "steering JIT cache: device=%s variants=%d",
            device_id,
            len(keys),
        )
        for i, key in enumerate(keys):
            logger.info("  variant[%d]: %r", i, key)
    logger.info("steering JIT cache: total_variants=%d", total)


def _kernel_cache_size() -> int:
    """Return the total number of compiled variants across all devices.

    Returns 0 when the kernel has not yet been built (no ``cache``
    attribute, e.g. when Triton is disabled in the importing process).
    """
    cache = getattr(_apply_steering_kernel, "cache", None)
    if cache is None:
        return 0
    total = 0
    for device_cache in cache.values():
        try:
            total += len(device_cache)
        except TypeError:
            continue
    return total


def warmup_apply_steering_kernel(
    *,
    hidden_size: int,
    table_rows: int,
    table_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
    capture_sizes: list[int] | None = None,
) -> None:
    """JIT-compile the kernel ahead of CUDA graph capture.

    The kernel is launched with dummy tensors at every shape vLLM will
    subsequently hit so Triton's first-call JIT cost — observed as
    ~10 ms ``cuLibraryLoadData`` events on a 3090 with gemma-3-4b-it,
    even in modes that never apply a non-zero steering vector — happens
    before any served request window.

    *capture_sizes* is the full list of batch dims that warmup should
    drive the kernel for.  Pass ``vllm_config.compilation_config.
    cudagraph_capture_sizes`` so the warmup stays in sync with the
    captured shapes.  When ``None``, falls back to a representative
    powers-of-two list.

    The ``apply_steering`` registered op is the path used at runtime; we
    route warmup through it (``torch.ops.vllm.apply_steering``) rather
    than calling the Triton wrapper directly, because going direct would
    compile a different stride-class specialization than what the
    dispatched runtime call ends up triggering.

    Both ``any_active`` states are exercised at every batch size — the
    inactive branch and the active branch share the same compiled
    artifact (the flag is a tensor, not a constexpr) but driving both
    flags is cheap insurance.

    Total compile count and cumulative warmup wall-clock are logged at
    INFO so the cost is visible.
    """
    if device.type != "cuda":
        return

    sizes = capture_sizes if capture_sizes else _default_warmup_sizes()
    # Defensive: deduplicate and sort to drive smaller shapes first
    # (smaller compiles tend to be slightly cheaper).
    sizes = sorted({int(s) for s in sizes if int(s) > 0})
    if not sizes:
        return

    cache_before = _kernel_cache_size()
    max_n = max(sizes)
    # One large allocation each; we only ever read/write the leading N
    # rows per launch, so reusing a single buffer per dtype avoids
    # ``max_n`` independent allocations.
    hidden_buf = torch.zeros(max_n, hidden_size, dtype=compute_dtype, device=device)
    table_buf = torch.zeros(
        max(table_rows, 1), hidden_size, dtype=table_dtype, device=device
    )
    index_buf = torch.zeros(max_n, dtype=torch.long, device=device)
    active_flag = torch.zeros(1, dtype=torch.bool, device=device)
    scales_buf = torch.ones(max(table_rows, 1), dtype=torch.float32, device=device)
    dvec_buf = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    tscale_buf = torch.zeros(max_n, dtype=torch.float32, device=device)
    rgate_buf = torch.ones(max_n, dtype=torch.float32, device=device)
    # Fused in-graph monitor buffers (inactive during warmup; the GEMV
    # branch is in the same compiled artifact regardless of the flag).
    probe_buf = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    mparams_buf = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    mactive_flag = torch.zeros(1, dtype=torch.bool, device=device)
    dmask_buf = torch.zeros(max_n, dtype=torch.float32, device=device)
    # Per-row monitor buffers (inactive during warmup; same compiled artifact).
    rprobe_buf = torch.zeros(
        max(table_rows, 1), hidden_size, dtype=torch.float32, device=device
    )
    rparams_buf = (
        torch.tensor([-1.0e30, 1.0], dtype=torch.float32, device=device)
        .expand(max(table_rows, 1), 2)
        .clone()
    )
    ractive_flag = torch.zeros(1, dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    for n in sizes:
        hidden_view = hidden_buf[:n]
        index_view = index_buf[:n]
        tscale_view = tscale_buf[:n]
        rgate_view = rgate_buf[:n]
        dmask_view = dmask_buf[:n]
        # Inactive path first — exercises the short-circuit branch.
        active_flag.fill_(False)
        torch.ops.vllm.apply_steering(
            hidden_view,
            table_buf,
            index_view,
            active_flag,
            scales_buf,
            dvec_buf,
            tscale_view,
            rgate_view,
            probe_buf,
            mparams_buf,
            mactive_flag,
            dmask_view,
            rprobe_buf,
            rparams_buf,
            ractive_flag,
        )
        # Active path — exercises the gather + add.
        active_flag.fill_(True)
        torch.ops.vllm.apply_steering(
            hidden_view,
            table_buf,
            index_view,
            active_flag,
            scales_buf,
            dvec_buf,
            tscale_view,
            rgate_view,
            probe_buf,
            mparams_buf,
            mactive_flag,
            dmask_view,
            rprobe_buf,
            rparams_buf,
            ractive_flag,
        )
    # Block until every JIT compile (and cuLibraryLoadData) has retired so
    # the wall-clock measurement and cache-size readback reflect reality.
    torch.accelerator.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    cache_after = _kernel_cache_size()
    new_variants = cache_after - cache_before
    logger.info(
        "steering kernel warmup: shapes=%d variants_compiled=%d "
        "cache_total=%d elapsed_ms=%.1f",
        len(sizes),
        new_variants,
        cache_after,
        elapsed_ms,
    )

    if os.environ.get("VLLM_STEERING_DUMP_JIT_CACHE", "0") == "1":
        _dump_jit_cache_keys()
