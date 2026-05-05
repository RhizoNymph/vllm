# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical-equivalence tests for the fused
``apply_steering_and_norm`` Triton kernel.

The kernel collapses the unfused Gemma 3 ``post_attn`` pattern

    residual = apply_steering(residual, table, idx)
    hidden_states, residual = gemma_rms_norm(hidden_states, residual)

into one launch. Tests assert that the fused output matches an eager
reference implementation within a tolerance that's loose enough to
absorb fp32-vs-bf16 ordering differences but tight enough to catch
real bugs (e.g. a missing ``+ 1.0`` on the weight, a missing residual
write-back, a swapped grid).

These run on CUDA only (the kernel itself is CUDA-only). When CUDA is
unavailable the tests are skipped — the eager fallback path in
``apply_steering_and_norm`` is exercised indirectly by
``test_steering_op.py``'s reference and by the higher-level steering
unit tests.
"""

from __future__ import annotations

import pytest
import torch


def _reference_apply_steering_and_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager reference matching ``apply_steering`` chained with
    ``GemmaRMSNorm.forward(hidden_states, residual)``.

    Math (matches ``GemmaRMSNorm.forward_native`` with residual, on
    bf16 / fp32 input — the fp16 upcast branch is intentionally left
    out because gemma-3-4b-it loads in bf16):

        new_residual = hidden + residual + table[idx[:N]]
        var = mean(new_residual.fp32() ** 2)
        rstd = 1 / sqrt(var + eps)
        out = (new_residual.fp32() * rstd) * (weight.fp32() + 1.0)
        return out.to(input_dtype), new_residual
    """
    N = hidden_states.shape[0]
    new_residual = (
        hidden_states
        + residual
        + steering_table[steering_index[:N]].to(hidden_states.dtype)
    )
    nr_fp32 = new_residual.to(torch.float32)
    var = nr_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    weight_fp32 = weight.to(torch.float32) + 1.0
    normed = (nr_fp32 * rstd * weight_fp32).to(hidden_states.dtype)
    return normed, new_residual


def _has_cuda() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _has_cuda(), reason="apply_steering_and_norm Triton kernel requires CUDA"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("hidden_size", [2048, 2560, 3072, 4096])
@pytest.mark.parametrize("num_tokens", [1, 4, 128])
def test_matches_reference_random(
    dtype: torch.dtype, hidden_size: int, num_tokens: int
):
    """Random hidden / residual / table — fused output matches eager
    reference within fp tolerance."""
    from vllm.model_executor.layers.steering_norm_kernel import (
        apply_steering_and_norm_triton,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    eps = 1e-6
    num_rows = 16

    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    table = torch.randn(num_rows, hidden_size, dtype=dtype, device=device)
    table[0] = 0.0  # zeros sentinel
    weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
    index = torch.randint(0, num_rows, (num_tokens,), dtype=torch.long, device=device)

    out_norm, out_residual = apply_steering_and_norm_triton(
        hidden, residual, table, index, weight, eps
    )
    ref_norm, ref_residual = _reference_apply_steering_and_norm(
        hidden, residual, table, index, weight, eps
    )

    # bf16 / fp16 accumulation order differs slightly between Triton and
    # eager torch; allow a generous absolute tolerance on the normed
    # output. The residual is just an add, no division — tighter rtol.
    norm_atol = {torch.bfloat16: 5e-2, torch.float16: 5e-3, torch.float32: 1e-4}[dtype]
    norm_rtol = {torch.bfloat16: 5e-2, torch.float16: 5e-3, torch.float32: 1e-4}[dtype]
    res_atol = {torch.bfloat16: 5e-2, torch.float16: 1e-2, torch.float32: 1e-5}[dtype]

    assert torch.allclose(out_norm, ref_norm, rtol=norm_rtol, atol=norm_atol), (
        f"normed mismatch: max diff "
        f"{(out_norm.float() - ref_norm.float()).abs().max().item():.6f}"
    )
    assert torch.allclose(out_residual, ref_residual, rtol=norm_rtol, atol=res_atol), (
        f"residual mismatch: max diff "
        f"{(out_residual.float() - ref_residual.float()).abs().max().item():.6f}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_zero_table_is_pure_norm(dtype: torch.dtype):
    """Index 0 (zeros sentinel) must reduce to plain
    ``GemmaRMSNorm(hidden + residual)`` — i.e. steering contributes
    nothing to either output."""
    from vllm.model_executor.layers.steering_norm_kernel import (
        apply_steering_and_norm_triton,
    )

    torch.manual_seed(1)
    device = torch.device("cuda")
    num_tokens, hidden_size = 8, 2560
    eps = 1e-6

    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    table = torch.randn(4, hidden_size, dtype=dtype, device=device)
    table[0] = 0.0
    weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.1
    index = torch.zeros(num_tokens, dtype=torch.long, device=device)

    out_norm, out_residual = apply_steering_and_norm_triton(
        hidden, residual, table, index, weight, eps
    )

    # Reference: same kernel with zero steering = standard GemmaRMSNorm.
    expected_residual = hidden + residual
    nr_fp32 = expected_residual.to(torch.float32)
    var = nr_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    expected_norm = (nr_fp32 * rstd * (weight.to(torch.float32) + 1.0)).to(dtype)

    assert torch.allclose(out_residual, expected_residual, rtol=5e-3, atol=5e-3)
    atol = {torch.bfloat16: 5e-2, torch.float32: 1e-4}[dtype]
    assert torch.allclose(out_norm, expected_norm, rtol=atol, atol=atol)


def test_index_buffer_larger_than_batch():
    """Only first ``N`` entries of ``steering_index`` are read; trailing
    bytes are ignored."""
    from vllm.model_executor.layers.steering_norm_kernel import (
        apply_steering_and_norm_triton,
    )

    torch.manual_seed(2)
    device = torch.device("cuda")
    num_tokens, hidden_size = 4, 2560
    eps = 1e-6
    max_index_len = 256

    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    table = torch.randn(8, hidden_size, dtype=torch.bfloat16, device=device)
    table[0] = 0.0
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.1

    index = torch.zeros(max_index_len, dtype=torch.long, device=device)
    index[:num_tokens] = torch.tensor([2, 5, 1, 3], dtype=torch.long, device=device)
    # Garbage past N — should be unread.
    index[num_tokens:] = 999_999

    out_norm, out_residual = apply_steering_and_norm_triton(
        hidden, residual, table, index, weight, eps
    )
    ref_norm, ref_residual = _reference_apply_steering_and_norm(
        hidden, residual, table, index, weight, eps
    )

    # If the kernel mistakenly read past N, indexing 999_999 into an
    # 8-row table would be UB / a wild read. Surviving this with a
    # matching reference is the assertion.
    assert torch.allclose(out_norm, ref_norm, rtol=5e-2, atol=5e-2)
    assert torch.allclose(out_residual, ref_residual, rtol=5e-2, atol=5e-2)


def test_empty_batch_is_noop():
    """N=0 short-circuits before kernel launch; outputs are still
    correctly shaped."""
    from vllm.model_executor.layers.steering_norm_kernel import (
        apply_steering_and_norm_triton,
    )

    device = torch.device("cuda")
    hidden = torch.empty(0, 2560, dtype=torch.bfloat16, device=device)
    residual = torch.empty(0, 2560, dtype=torch.bfloat16, device=device)
    table = torch.zeros(4, 2560, dtype=torch.bfloat16, device=device)
    weight = torch.zeros(2560, dtype=torch.bfloat16, device=device)
    index = torch.zeros(8, dtype=torch.long, device=device)

    out_norm, out_residual = apply_steering_and_norm_triton(
        hidden, residual, table, index, weight, 1e-6
    )

    assert out_norm.shape == (0, 2560)
    assert out_residual.shape == (0, 2560)


def test_distinct_per_token_rows():
    """Different rows for different tokens — fused output matches
    eager reference exactly within fp tolerance."""
    from vllm.model_executor.layers.steering_norm_kernel import (
        apply_steering_and_norm_triton,
    )

    torch.manual_seed(3)
    device = torch.device("cuda")
    num_tokens, hidden_size = 16, 3072
    eps = 1e-6

    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    table = torch.randn(8, hidden_size, dtype=torch.bfloat16, device=device)
    table[0] = 0.0
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    # Spread rows across the table so the kernel must actually gather
    # different rows (not just constant-fold to a single load).
    index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, dtype=torch.long, device=device)[
        :num_tokens
    ]

    out_norm, out_residual = apply_steering_and_norm_triton(
        hidden, residual, table, index, weight, eps
    )
    ref_norm, ref_residual = _reference_apply_steering_and_norm(
        hidden, residual, table, index, weight, eps
    )

    assert torch.allclose(out_norm, ref_norm, rtol=5e-2, atol=5e-2)
    assert torch.allclose(out_residual, ref_residual, rtol=5e-2, atol=5e-2)
