# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused PRE_ATTN + POST_ATTN steering op.

The fused op is mathematically equivalent to two back-to-back calls of
the single-table op against ``table_pre`` then ``table_post`` when the
residual is not mutated between them (true for Gemma3). These tests
exercise both the CPU eager fallback and the CUDA Triton path against a
direct PyTorch reference within the bf16 tolerance budget.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.steering import (
    apply_steering,
    apply_steering_pre_post,
)

# Tolerances per dtype. Both the fused and unfused paths are
# approximations of the true fp32 sum ``h + pre + post``; each can
# disagree from ground truth by ~1 ULP, so they can disagree from
# *each other* by up to ~2 ULPs at the result magnitude. We compare
# both against the high-precision fp32 reference and accept a single
# ULP-scaled envelope that bounds both.
_TOL: dict[torch.dtype, tuple[float, float]] = {
    # bf16: 8-bit mantissa, ULP at magnitude 4 is 2^-5 = 0.03125 — leave
    # generous headroom for sums of three bf16 values where the result
    # may be in the [2, 8) bin.
    torch.bfloat16: (5e-2, 5e-2),
    # fp16: 11-bit mantissa, ULP at magnitude 4 is 2^-8 ≈ 0.004.
    torch.float16: (5e-3, 5e-3),
    # fp32: tight.
    torch.float32: (1e-6, 1e-6),
}


def _fp32_reference(
    hidden: torch.Tensor,
    table_pre: torch.Tensor,
    table_post: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """High-precision reference: sum in fp32 then cast back to compute dtype.

    Both the unfused two-call path and the fused op are approximations
    of this; the fused op is in fact closer to it (single rounding step
    vs two). Comparing both implementations against this ground truth
    gives a stable ULP-scaled tolerance envelope that doesn't depend on
    the order in which the unfused path happens to round.
    """
    n = hidden.shape[0]
    idx = index[:n]
    h_f32 = hidden.to(torch.float32)
    pre_f32 = table_pre[idx].to(torch.float32)
    post_f32 = table_post[idx].to(torch.float32)
    return (h_f32 + pre_f32 + post_f32).to(hidden.dtype)


def _two_call_reference(
    hidden: torch.Tensor,
    table_pre: torch.Tensor,
    table_post: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Unfused two-call reference, kept for the API-equivalence smoke test."""
    out = apply_steering(hidden, table_pre, index)
    out = apply_steering(out, table_post, index)
    return out


def _make_inputs(
    *,
    n_tokens: int,
    hidden_size: int,
    n_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(n_tokens, hidden_size, dtype=dtype, device=device, generator=g)
    # Tables are allocated in the compute dtype in production; mirror
    # that here so the gather doesn't need a cast.
    table_pre = torch.randn(
        n_rows, hidden_size, dtype=dtype, device=device, generator=g
    )
    table_post = torch.randn(
        n_rows, hidden_size, dtype=dtype, device=device, generator=g
    )
    # Row 0 is the no-steering sentinel.
    table_pre[0].zero_()
    table_post[0].zero_()
    if n_tokens == 0:
        index = torch.zeros(0, dtype=torch.long, device=device)
    else:
        # Spread tokens across rows so we exercise distinct gathers.
        index = torch.randint(
            0, n_rows, (n_tokens,), dtype=torch.long, device=device, generator=g
        )
    return hidden, table_pre, table_post, index


_HIDDEN_SIZES = [2048, 2560, 3072, 4096]
_NUM_TOKENS = [1, 4, 128]
_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("n_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cpu_eager_matches_reference(
    hidden_size: int, n_tokens: int, dtype: torch.dtype
) -> None:
    """CPU eager path matches the unfused two-call reference."""
    device = torch.device("cpu")
    hidden, table_pre, table_post, index = _make_inputs(
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        n_rows=8,
        dtype=dtype,
        device=device,
        seed=0xABCD,
    )

    rtol, atol = _TOL[dtype]
    fp32_ref = _fp32_reference(hidden, table_pre, table_post, index)
    two_call_ref = _two_call_reference(hidden, table_pre, table_post, index)
    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
    # Fused must agree with fp32 ground truth within ULP envelope.
    torch.testing.assert_close(actual, fp32_ref, rtol=rtol, atol=atol)
    # Unfused two-call path must also agree with ground truth within
    # the same envelope — locks in that "numerically equivalent to the
    # original unfused path" means equivalent to the fp32 ideal, not
    # bit-equivalent to a particular rounding order.
    torch.testing.assert_close(two_call_ref, fp32_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("n_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_triton_matches_reference(
    hidden_size: int, n_tokens: int, dtype: torch.dtype
) -> None:
    """CUDA Triton path matches the fp32 reference within ULP envelope."""
    device = torch.device("cuda")
    hidden, table_pre, table_post, index = _make_inputs(
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        n_rows=8,
        dtype=dtype,
        device=device,
        seed=0xBEEF,
    )

    rtol, atol = _TOL[dtype]
    fp32_ref = _fp32_reference(hidden, table_pre, table_post, index)
    two_call_ref = _two_call_reference(hidden, table_pre, table_post, index)
    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
    torch.testing.assert_close(actual, fp32_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(two_call_ref, fp32_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_empty_batch_returns_empty_tensor(device_type: str) -> None:
    """N == 0 short-circuits and returns an empty tensor of the right shape."""
    device = torch.device(device_type)
    hidden, table_pre, table_post, index = _make_inputs(
        n_tokens=0,
        hidden_size=2048,
        n_rows=4,
        dtype=torch.float32,
        device=device,
        seed=1,
    )

    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)
    assert actual.shape == (0, 2048)
    assert actual.dtype == hidden.dtype


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_zero_tables_are_noop(device_type: str) -> None:
    """All-zero tables leave the residual unchanged regardless of index."""
    device = torch.device(device_type)
    n_tokens, hidden_size = 16, 2048
    g = torch.Generator(device=device).manual_seed(7)
    hidden = torch.randn(
        n_tokens, hidden_size, dtype=torch.float32, device=device, generator=g
    )
    table_pre = torch.zeros(8, hidden_size, dtype=torch.float32, device=device)
    table_post = torch.zeros(8, hidden_size, dtype=torch.float32, device=device)
    index = torch.randint(
        0, 8, (n_tokens,), dtype=torch.long, device=device, generator=g
    )

    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)
    torch.testing.assert_close(actual, hidden, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_distinct_per_token_rows(device_type: str) -> None:
    """Each token can independently select a row from each table.

    The fused op must apply *the same* index to both tables (the design
    constraint — index buffer is shared across hook points) so the
    expected result is row-aligned.
    """
    device = torch.device(device_type)
    hidden_size = 2048
    hidden = torch.zeros(4, hidden_size, dtype=torch.float32, device=device)
    table_pre = torch.zeros(4, hidden_size, dtype=torch.float32, device=device)
    table_post = torch.zeros(4, hidden_size, dtype=torch.float32, device=device)
    table_pre[1] = 2.0
    table_pre[2] = 3.0
    table_post[1] = 5.0
    table_post[2] = 7.0
    index = torch.tensor([0, 1, 2, 1], dtype=torch.long, device=device)

    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)

    expected = torch.zeros(4, hidden_size, dtype=torch.float32, device=device)
    expected[0] = 0.0
    expected[1] = 2.0 + 5.0
    expected[2] = 3.0 + 7.0
    expected[3] = 2.0 + 5.0
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_index_buffer_longer_than_batch() -> None:
    """Only ``index[:N]`` is read; trailing entries must be ignored."""
    device = torch.device("cuda")
    n_tokens, hidden_size = 8, 2048
    g = torch.Generator(device=device).manual_seed(11)
    hidden = torch.randn(
        n_tokens, hidden_size, dtype=torch.bfloat16, device=device, generator=g
    )
    table_pre = torch.randn(
        4, hidden_size, dtype=torch.bfloat16, device=device, generator=g
    )
    table_post = torch.randn(
        4, hidden_size, dtype=torch.bfloat16, device=device, generator=g
    )
    table_pre[0].zero_()
    table_post[0].zero_()

    index = torch.zeros(64, dtype=torch.long, device=device)
    index[:n_tokens] = torch.tensor(
        [0, 1, 2, 3, 1, 2, 3, 0], dtype=torch.long, device=device
    )

    rtol, atol = _TOL[torch.bfloat16]
    expected = _fp32_reference(hidden, table_pre, table_post, index)
    actual = apply_steering_pre_post(hidden, table_pre, table_post, index)

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
