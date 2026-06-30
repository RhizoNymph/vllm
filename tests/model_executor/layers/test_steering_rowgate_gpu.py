# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU parity tests for Phase 2 per-request-row token gating.

The CPU tests in ``test_steering_op.py`` / ``test_steering_monitor_op.py``
cover the math via the eager path; these exercise the real Triton kernels
on CUDA. Skipped unless a GPU is present. See
docs/design/dynamic_steering_row_gating.md.
"""

import pytest
import torch

import vllm.model_executor.layers.steering  # noqa: F401 — registers the ops

H = 5376

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="row-gating GPU parity needs CUDA"
)


def _eager(hidden, table, index, scales, dvec, tscale, rgate):
    n = hidden.shape[0]
    rows = index[:n]
    row = (
        table[rows].to(torch.float32)
        * scales[rows].unsqueeze(-1)
        * rgate[:n].unsqueeze(-1)
    )
    tier = dvec.to(torch.float32).unsqueeze(0) * tscale[:n].unsqueeze(-1)
    return hidden.to(torch.float32) + row + tier


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1, 8, 64, 256])
def test_kernel_row_gate_matches_eager(dtype, n):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    hidden = torch.randn(n, H, dtype=dtype, device=dev)
    table = torch.randn(9, H, dtype=dtype, device=dev)
    table[0] = 0.0
    index = torch.randint(0, 9, (n,), dtype=torch.long, device=dev)
    any_active = torch.ones(1, dtype=torch.bool, device=dev)
    scales = torch.rand(9, device=dev) + 0.5
    dvec = torch.randn(H, dtype=torch.float32, device=dev)
    tscale = torch.zeros(n, dtype=torch.float32, device=dev)
    rgate = torch.rand(n, dtype=torch.float32, device=dev)  # varied gates
    probe = torch.zeros(H, dtype=torch.float32, device=dev)
    mparams = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=dev)
    mactive = torch.zeros(1, dtype=torch.bool, device=dev)  # monitor off
    dmask = torch.zeros(n, dtype=torch.float32, device=dev)
    rprobe = torch.zeros(9, H, dtype=torch.float32, device=dev)
    rparams = (
        torch.tensor([-1.0e30, 1.0], dtype=torch.float32, device=dev)
        .expand(9, 2)
        .clone()
    )
    ractive = torch.zeros(1, dtype=torch.bool, device=dev)  # row monitor off
    out = torch.ops.vllm.apply_steering(
        hidden,
        table,
        index,
        any_active,
        scales,
        dvec,
        tscale,
        rgate,
        probe,
        mparams,
        mactive,
        dmask,
        rprobe,
        rparams,
        ractive,
    )
    exp = _eager(hidden, table, index, scales, dvec, tscale, rgate)
    rel = (out.to(torch.float32) - exp).abs().max() / (exp.abs().max() + 1e-6)
    assert rel < (3e-2 if dtype == torch.bfloat16 else 2e-4), f"rel={rel:.2e}"


def test_monitor_gate_rows_decode_only():
    dev = torch.device("cuda")
    n = 4
    probe = torch.ones(H, dtype=torch.float32, device=dev)
    hidden = torch.stack(
        [
            torch.full((H,), 1.0),  # decode, engaged
            torch.full((H,), 1.0),  # prefill (mask 0) — must stay 1.0
            torch.full((H,), -1.0),  # decode, disengaged
            torch.full((H,), -1.0),  # prefill (mask 0) — must stay 1.0
        ]
    ).to(dtype=torch.bfloat16, device=dev)
    params = torch.tensor([0.0, 50.0, 1.0], dtype=torch.float32, device=dev)
    active = torch.ones(1, dtype=torch.bool, device=dev)
    tscale = torch.zeros(n, dtype=torch.float32, device=dev)
    dmask = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=dev)
    rgate = torch.ones(n, dtype=torch.float32, device=dev)
    torch.ops.vllm.steering_monitor(hidden, probe, params, active, tscale, dmask, rgate)
    rg = rgate.cpu()
    assert rg[0] > 0.99  # decode engaged
    assert rg[1].item() == 1.0  # prefill never gated
    assert rg[2] < 0.01  # decode disengaged
    assert rg[3].item() == 1.0  # prefill never gated


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
