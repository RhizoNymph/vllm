# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity tests for the directional-clamp Triton kernels.

The CPU tests in ``test_clamp_op.py`` cover the clamp math via the eager
path; these exercise the real Triton kernels on CUDA against that eager
reference, including the exactness edges (row-0 passthrough, inactive
flag, zero-padded K slots). Ported from the manual
``tests/gpu_clamp_validate.py --mode kernel`` script.
"""

import pytest
import torch

from vllm.model_executor.layers.clamp import apply_clamp, apply_clamp_block

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="clamp GPU parity needs CUDA"
)

ROWS, K, HIDDEN, N = 8, 4, 1024, 33


def _make(dtype):
    torch.manual_seed(0)
    dirs = torch.zeros(ROWS, K, HIDDEN, dtype=dtype)
    for r in range(1, 6):
        for j in range(1 + r % 3):
            v = torch.randn(HIDDEN)
            dirs[r, j] = (v / v.norm()).to(dtype)
    bounds = torch.empty(ROWS, K, 2, dtype=torch.float32)
    bounds[..., 0] = -float("inf")
    bounds[..., 1] = float("inf")
    bounds[1, 0] = torch.tensor([0.0, 0.0])
    bounds[2, 0] = torch.tensor([-1.0, 1.0])
    bounds[3, 0, 1] = 2.0
    bounds[4, 0] = torch.tensor([5.0, 5.0])
    # Poisoned bounds on a zero-padded slot must stay a no-op.
    bounds[5, 3] = torch.tensor([7.0, 7.0])
    strength = torch.ones(ROWS, K, dtype=torch.float32)
    strength[4, 0] = 0.5
    index = torch.randint(0, 6, (N,), dtype=torch.long)
    index[0] = 0
    active = torch.ones(1, dtype=torch.bool)
    return dirs, bounds, strength, index, active


def _to_cuda(*tensors):
    return tuple(t.cuda() for t in tensors)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_clamp_matches_eager(dtype):
    dirs, bounds, strength, index, active = _make(dtype)
    h = torch.randn(N, HIDDEN, dtype=dtype)

    cpu = apply_clamp(h, dirs, bounds, strength, index, active)
    gpu = apply_clamp(*_to_cuda(h, dirs, bounds, strength, index, active)).cpu()

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    max_err = (cpu.float() - gpu.float()).abs().max().item()
    assert max_err < tol, f"max_err={max_err:.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_clamp_block_matches_eager(dtype):
    dirs, bounds, strength, index, active = _make(dtype)
    h = torch.randn(N, HIDDEN, dtype=dtype)
    res = torch.randn(N, HIDDEN, dtype=dtype)

    cpu = apply_clamp_block(h, res, dirs, bounds, strength, index, active)
    gpu = apply_clamp_block(
        *_to_cuda(h, res, dirs, bounds, strength, index, active)
    ).cpu()

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    max_err = (cpu.float() - gpu.float()).abs().max().item()
    assert max_err < tol, f"max_err={max_err:.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_zero_bitwise_passthrough(dtype):
    dirs, bounds, strength, index, active = _make(dtype)
    h = torch.randn(N, HIDDEN, dtype=dtype)

    h_c, dirs_c, bounds_c, strength_c, index_c, active_c = _to_cuda(
        h, dirs, bounds, strength, index, active
    )
    gpu = apply_clamp(h_c, dirs_c, bounds_c, strength_c, index_c, active_c)

    sentinel = index_c == 0
    assert torch.equal(gpu[sentinel], h_c[sentinel])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inactive_flag_bitwise_passthrough(dtype):
    dirs, bounds, strength, index, _active = _make(dtype)
    h = torch.randn(N, HIDDEN, dtype=dtype)

    h_c, dirs_c, bounds_c, strength_c, index_c = _to_cuda(
        h, dirs, bounds, strength, index
    )
    inactive = torch.zeros(1, dtype=torch.bool, device="cuda")
    gpu = apply_clamp(h_c, dirs_c, bounds_c, strength_c, index_c, inactive)

    assert torch.equal(gpu, h_c)


def test_partial_strength_pin_projection():
    """Row 4 pins to 5.0 at strength 0.5: the projection must land
    halfway between the natural projection and the target."""
    dtype = torch.float32
    dirs, bounds, strength, index, active = _make(dtype)
    h = torch.randn(N, HIDDEN, dtype=dtype)

    h_c, dirs_c, bounds_c, strength_c, index_c, active_c = _to_cuda(
        h, dirs, bounds, strength, index, active
    )
    gpu = apply_clamp(h_c, dirs_c, bounds_c, strength_c, index_c, active_c)

    row4 = index_c == 4
    if not row4.any():
        pytest.skip("Seeded index draw assigned no tokens to row 4")
    direction = dirs_c[4, 0].float()
    proj = gpu[row4].float() @ direction
    proj_in = h_c[row4].float() @ direction
    expect = proj_in + 0.5 * (5.0 - proj_in)
    err = (proj - expect).abs().max().item()
    assert err < 1e-3, f"err={err:.2e}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
