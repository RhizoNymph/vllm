# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests for the neuron-clamp Triton kernel (parity, graphs, compile)."""

import pytest
import torch

import vllm.model_executor.layers.neuron_clamp  # noqa: F401  # registers op
from vllm.model_executor.layers.neuron_clamp import neuron_clamp_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda"


def _random_tables(rows, k, inter, *, fill=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.full((rows, k), -1, dtype=torch.int32)
    mul = torch.ones(rows, k)
    add = torch.zeros(rows, k)
    for r in range(3, rows):
        n = int(torch.randint(0, k + 1, (1,), generator=g))
        neurons = torch.randperm(inter, generator=g)[:n]
        for s, nid in enumerate(neurons.tolist()):
            if torch.rand(1, generator=g).item() > fill:
                continue
            mode = int(torch.randint(0, 3, (1,), generator=g))
            val = float(torch.randn(1, generator=g) * 5)
            idx[r, s] = nid
            if mode == 0:
                mul[r, s], add[r, s] = 0.0, val
            elif mode == 1:
                mul[r, s], add[r, s] = 1.0, val
            else:
                mul[r, s], add[r, s] = val, 0.0
    return idx.to(DEV), mul.to(DEV), add.to(DEV)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "n,inter,k", [(1, 3072, 16), (7, 3072, 4), (513, 12288, 16), (4096, 6144, 1)]
)
def test_triton_matches_eager(dtype, n, inter, k):
    rows = 7
    idx, mul, add = _random_tables(rows, k, inter, seed=n + k)
    sidx = torch.randint(0, rows, (n + 5,), device=DEV, dtype=torch.long)
    flag = torch.ones(1, dtype=torch.bool, device=DEV)
    act = (torch.randn(n, inter, device=DEV) * 3).to(dtype)
    ref = neuron_clamp_reference(
        act.cpu(), idx.cpu(), mul.cpu(), add.cpu(), sidx.cpu(), flag.cpu()
    )
    out = act.clone()
    torch.ops.vllm.apply_neuron_clamp(out, idx, mul, add, sidx, flag)
    torch.cuda.synchronize()
    # Both compute x*mul+add in fp32 and round once to the activation dtype;
    # Triton may fuse the FMA (one fp32 rounding instead of two), which is
    # invisible after a bf16/fp16 round but can differ by an fp32 ulp.
    tol = 1e-6 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(out.cpu(), ref, rtol=tol, atol=tol)
    # Untouched coordinates are bit-identical.
    touched = torch.zeros_like(act, dtype=torch.bool).cpu()
    rows_t = sidx[:n].cpu()
    for t in range(n):
        for nid in idx.cpu()[rows_t[t]].tolist():
            if nid >= 0:
                touched[t, nid] = True
    assert torch.equal(out.cpu()[~touched], act.cpu()[~touched])


def test_flag_off_untouched():
    idx, mul, add = _random_tables(7, 16, 3072)
    sidx = torch.full((64,), 5, device=DEV, dtype=torch.long)
    act = torch.randn(64, 3072, device=DEV, dtype=torch.bfloat16)
    out = act.clone()
    torch.ops.vllm.apply_neuron_clamp(
        out, idx, mul, add, sidx, torch.zeros(1, dtype=torch.bool, device=DEV)
    )
    assert torch.equal(out, act)


def test_empty_batch():
    idx, mul, add = _random_tables(7, 4, 64)
    act = torch.empty(0, 64, device=DEV, dtype=torch.bfloat16)
    torch.ops.vllm.apply_neuron_clamp(
        act,
        idx,
        mul,
        add,
        torch.zeros(4, device=DEV, dtype=torch.long),
        torch.ones(1, dtype=torch.bool, device=DEV),
    )


def test_cuda_graph_replay_reads_live_tables():
    rows, k, inter, n = 7, 4, 1024, 32
    idx = torch.full((rows, k), -1, dtype=torch.int32, device=DEV)
    mul = torch.ones(rows, k, device=DEV)
    add = torch.zeros(rows, k, device=DEV)
    sidx = torch.zeros(n, dtype=torch.long, device=DEV)
    flag = torch.zeros(1, dtype=torch.bool, device=DEV)
    static_in = torch.randn(n, inter, device=DEV, dtype=torch.bfloat16)
    static_out = torch.empty_like(static_in)

    def step():
        x = static_in * 1.0
        torch.ops.vllm.apply_neuron_clamp(x, idx, mul, add, sidx, flag)
        static_out.copy_(x)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()  # warm up / JIT outside capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, static_in)

    # Mutate tables/index/flag in place after capture: replay must see them.
    idx[4, 0] = 17
    mul[4, 0] = 0.0
    add[4, 0] = -30.0
    sidx[::2] = 4
    flag.fill_(True)
    graph.replay()
    torch.cuda.synchronize()
    exp = static_in.clone()
    exp[::2, 17] = -30.0
    assert torch.equal(static_out, exp)

    flag.fill_(False)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, static_in)


def test_torch_compile_matches_eager():
    rows, k, inter, n = 7, 4, 512, 16
    idx, mul, add = _random_tables(rows, k, inter, fill=1.0, seed=3)
    sidx = torch.randint(3, rows, (n,), device=DEV, dtype=torch.long)
    flag = torch.ones(1, dtype=torch.bool, device=DEV)
    # fp32 so inductor's fused (fp32-intermediate) silu*mul and eager agree
    # tightly; the point is clamp placement/semantics under compile.
    w = torch.randn(inter, 64, device=DEV, dtype=torch.float32)

    def mlp_tail(gate_up):
        d = gate_up.shape[-1] // 2
        act = torch.nn.functional.silu(gate_up[..., :d]) * gate_up[..., d:]
        torch.ops.vllm.apply_neuron_clamp(act, idx, mul, add, sidx, flag)
        return act @ w

    x = torch.randn(n, 2 * inter, device=DEV, dtype=torch.float32)
    eager = mlp_tail(x)
    compiled = torch.compile(mlp_tail, fullgraph=True)(x)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-4)
    # And the clamp really happened inside the compiled graph.
    d = inter
    act = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    act_c = neuron_clamp_reference(act, idx, mul, add, sidx, flag)
    torch.testing.assert_close(compiled, act_c @ w, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(act_c @ w, act @ w, rtol=1e-2, atol=1e-2)
