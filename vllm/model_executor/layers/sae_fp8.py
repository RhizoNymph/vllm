# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-wise fp8 (e4m3) storage helpers for SAE weight tables.

SAE modules can opt into storing their large per-feature weight
matrices (encoder/decoder rows for the delta path, full W_enc/W_dec
for full reconstruction) as ``torch.float8_e4m3fn`` with per-row fp32
scales via the manifest field ``storage_dtype: "fp8_e4m3"``, halving
GPU memory versus bf16.

Scheme (per row ``r`` — i.e. per SAE feature):

    scale[r] = max(amax(|w[r]|) / 448, tiny)   # 448 = e4m3 finite max
    q[r]     = round_to_e4m3(w[r] / scale[r])  # clamped to ±448
    w[r]     ≈ q[r].to(fp32) * scale[r]

Zero rows quantize to all-zero ``q`` with a tiny positive scale, so
dequantization is an exact zero and no division hazard exists
anywhere (division by ``scale`` happens only at quantization time,
where the clamp guarantees positivity).

Quantization happens worker-side at weight-attach time; weights still
arrive over the wire as bf16/fp32.  Note that fp8 *arithmetic* is not
required anywhere — consumers convert to fp32 first (``q.to(fp32) *
scale``), which CPU torch supports for ``float8_e4m3fn`` storage.
"""

from __future__ import annotations

import torch

from vllm.config.sae_steering_types import (
    SAE_STORAGE_DTYPE_AUTO,
    SAE_STORAGE_DTYPE_FP8_E4M3,
    VALID_SAE_STORAGE_DTYPES,
)

FP8_STORAGE_DTYPE = torch.float8_e4m3fn
"""Storage dtype backing ``storage_dtype="fp8_e4m3"``."""

FP8_E4M3_MAX = 448.0
"""Largest finite magnitude representable in ``float8_e4m3fn``."""


def resolve_sae_storage_dtype(
    storage_dtype: str, compute_dtype: torch.dtype
) -> torch.dtype:
    """Map a manifest ``storage_dtype`` string to a torch dtype."""
    if storage_dtype == SAE_STORAGE_DTYPE_AUTO:
        return compute_dtype
    if storage_dtype == SAE_STORAGE_DTYPE_FP8_E4M3:
        return FP8_STORAGE_DTYPE
    raise ValueError(
        f"Unknown SAE storage_dtype {storage_dtype!r}; expected one of "
        f"{list(VALID_SAE_STORAGE_DTYPES)}."
    )


def quantize_fp8_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight matrix to fp8 with per-row fp32 scales.

    Args:
        weight: ``(n_rows, n_cols)`` tensor in any floating dtype
            except fp8 (typically the bf16/fp32 wire form).

    Returns:
        ``(q, scale)`` where ``q`` is ``(n_rows, n_cols)``
        ``float8_e4m3fn`` and ``scale`` is ``(n_rows,)`` fp32.  The
        pre-cast values are clamped to ±448 so the fp8 cast can never
        produce NaN (torch's e4m3 cast does not saturate).
    """
    if weight.ndim != 2:
        raise ValueError(
            f"fp8 row-wise quantization expects a 2-D weight; got "
            f"shape {tuple(weight.shape)}."
        )
    if weight.dtype == FP8_STORAGE_DTYPE:
        raise ValueError("weight is already fp8; quantize from bf16/fp32.")
    w32 = weight.detach().to(torch.float32)
    amax = w32.abs().amax(dim=1)
    scale = (amax / FP8_E4M3_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = (
        (w32 / scale.unsqueeze(1))
        .clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(FP8_STORAGE_DTYPE)
    )
    return q, scale


def dequantize_fp8_rowwise(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize row-wise fp8 back to fp32: ``q.to(fp32) * scale[row]``."""
    return q.to(torch.float32) * scale.unsqueeze(1)


def maybe_dequantize_rowwise(
    weight: torch.Tensor, scale: torch.Tensor | None
) -> torch.Tensor:
    """Dequantize ``weight`` when it is fp8-stored; pass through otherwise.

    The eager op bodies call this on their weight arguments so the
    downstream math is dtype-agnostic.  A fp8 weight without a scale
    tensor is a wiring bug and fails loudly.
    """
    if weight.dtype != FP8_STORAGE_DTYPE:
        return weight
    if scale is None:
        raise ValueError(
            "fp8-stored SAE weight requires its per-row scale tensor; got None."
        )
    return dequantize_fp8_rowwise(weight, scale)
