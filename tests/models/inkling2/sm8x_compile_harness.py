# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the Inkling2 Triton kernels of the SM8x path for an emulated GPU.

Run as a script (`test_sm8x_compile.py` runs it in a subprocess): it
installs a fake Triton CUDA driver that reports compute capability
`--arch` (86 by default) with that architecture's shared-memory limit and a
no-op launcher, makes vLLM's platform report the same capability, then
calls the kernel wrappers the SM8x path uses with CPU tensors of the Inkling
2 Mini shapes. Every launch is compiled through ptxas for that target and
checked against the per-block shared-memory limit, so PTX that needs a
newer architecture (`griddepcontrol`, TMA, FP8 conversions) or tiles that
do not fit fail here without a GPU. Nothing executes.

Prints one line per kernel specialization and exits non-zero on any
failure.
"""

import argparse
import sys
import traceback

# Max dynamic shared memory per block (bytes, opt-in) per architecture.
_MAX_SHARED = {80: 166912, 86: 101376, 87: 166912, 89: 101376, 90: 232448}


def _install_fake_driver(arch: int) -> list[tuple[str, int]]:
    """Make Triton (and vLLM's Triton check) see one CUDA device of `arch`;
    returns the list each compiled kernel's (name, shared bytes) is appended
    to."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia import driver as nv_driver

    nv_driver.CudaDriver.is_active = staticmethod(lambda: True)
    compiled: list[tuple[str, int]] = []

    class _Utils:
        def get_device_properties(self, device):
            return {
                "max_shared_mem": _MAX_SHARED[arch],
                "multiprocessor_count": 82,
                "max_num_regs": 65536,
                "warpSize": 32,
                "sm_clock_rate": 1,
                "mem_clock_rate": 1,
                "mem_bus_width": 1,
            }

        def load_binary(self, name, kernel, shared, device):
            compiled.append((name, shared))
            return 0, 0, 0, 0, 1024

    class _FakeDriver:
        utils = _Utils()

        def __init__(self):
            self.launcher_cls = lambda src, metadata: (lambda *args, **kw: None)

        def get_current_device(self):
            return 0

        def set_current_device(self, device):
            pass

        def get_current_stream(self, device=None):
            return 0

        def get_current_target(self):
            return GPUTarget("cuda", arch, 32)

        def get_device_capability(self, device=0):
            return divmod(arch, 10)

        def get_active_torch_device(self):
            import torch

            return torch.device("cpu")

        def map_python_to_cpp_type(self, ty):
            return nv_driver.ty_to_cpp(ty)

    triton.runtime.driver.set_active(_FakeDriver())
    return compiled


def _patch_platform(arch: int) -> None:
    import vllm.platforms

    # A source checkout without an installed vllm distribution raises
    # PackageNotFoundError (an ImportError) here, which would make
    # `vllm.triton_utils` disable Triton.
    vllm.platforms.vllm_version_matches_substr = lambda substr: False
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    cap = DeviceCapability(major=arch // 10, minor=arch % 10)
    cls = type(current_platform)
    cls.get_device_capability = classmethod(lambda c, device_id=0: cap)


def _cases(arch: int):
    """(label, thunk) pairs over the SM8x-path kernels."""
    import torch

    from vllm.models.inkling2.nvidia.ops import attention as attn_ops
    from vllm.models.inkling2.nvidia.ops import dspark as dspark_ops
    from vllm.models.inkling2.nvidia.ops import loop as loop_ops
    from vllm.models.inkling2.nvidia.ops import vision as vision_ops
    from vllm.models.inkling2.nvidia.ops.attention_sm90 import (
        MODE_GATED,
        MODE_OUT_LSE,
        MODE_PARTIAL,
        rel_attention,
    )
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
    from vllm.models.inkling2.nvidia.ops.linear import (
        k_group_partials,
        linear_k_groups,
    )
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts
    from vllm.models.inkling2.nvidia.ops.norm import rms_norm
    from vllm.models.inkling2.nvidia.ops.sconv import (
        attention_prologue,
        shift_rel_proj,
        short_conv,
    )

    bf16, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
    hidden, heads, kv_heads, head_dim, d_rel, extent = 2304, 24, 8, 128, 16, 1024
    page, width = 64, 4

    def empty(*shape, dtype=bf16):
        return torch.zeros(shape, dtype=dtype)

    token_counts = (1, 8, 64, 300, 2048, 8192)

    # Projections (router, controllers, dynamic conv taps).
    for t in token_counts:
        yield (
            f"k_group_partials T={t} (router 128x2304)",
            lambda t=t: k_group_partials(empty(t, hidden), empty(128, hidden)),
        )
        yield (
            f"linear_k_groups T={t} (controller 64x4608)",
            lambda t=t: linear_k_groups(
                empty(t, 2 * hidden), empty(64, 2 * hidden), bf16
            ),
        )
        yield (
            f"select_experts T={t}",
            lambda t=t: select_experts(
                empty(t, hidden),
                empty(128, hidden),
                empty(128, dtype=f32),
                8,
                8.0,
                torch.tensor([0, 128], dtype=i32),
            ),
        )
        yield (
            f"rms_norm T={t}",
            lambda t=t: rms_norm(empty(t, hidden), empty(hidden), 1e-6),
        )

    # Block attention residual mixes (1 and 3 queries, with / without norm).
    for t in (1, 300, 8192):
        for nq in (1, 3, 4):
            for blocks in (0, 1, 3):
                for norm in (True, False):
                    yield (
                        f"attn_res T={t} Q={nq} blocks={blocks} norm={norm}",
                        lambda t=t, nq=nq, blocks=blocks, norm=norm: attn_res(
                            empty(blocks, t, hidden),
                            empty(t, hidden),
                            empty(t, hidden),
                            empty(nq, hidden, dtype=f32),
                            empty(nq, hidden, dtype=f32) if norm else None,
                            1e-6,
                            True,
                        ),
                    )

    # Loop message controllers.
    for t in (1, 300, 8192):
        for streams in (1, 3):
            yield (
                f"pair_rms_norm T={t} S={streams}",
                lambda t=t, s=streams: loop_ops.pair_rms_norm(
                    empty(s, t, hidden, dtype=f32), empty(s, t, hidden, dtype=f32)
                ),
            )
            yield (
                f"blend_rms_norm T={t} S={streams}",
                lambda t=t, s=streams: loop_ops.blend_rms_norm(
                    empty(s, t, hidden, dtype=f32),
                    empty(s, t, hidden, dtype=f32),
                    empty(s, t, 2 * hidden, dtype=f32),
                    empty(s, hidden),
                    1e-6,
                ),
            )

    # Short convs (conv-state cache: [blocks, heads, page, head_size]).
    conv_heads = 1
    for t in token_counts:
        cache = empty(16, conv_heads, page, 4 * hidden)
        yield (
            f"short_conv T={t} (hidden stream, residual)",
            lambda t=t, cache=cache: short_conv(
                empty(t, hidden),
                empty(hidden, width),
                cache,
                torch.zeros(t, dtype=i64),
                torch.zeros(1, 16, dtype=i32),
                torch.zeros(t, dtype=i32),
                torch.zeros(t, dtype=i64),
                torch.zeros(2, dtype=i32),
                2 * hidden,
                hidden,
                empty(t, hidden),
                empty(t, hidden),
            ),
        )

    # Attention prologue (clip, rel logits, K/V convs, cache insert).
    for t in token_counts:
        for local, dynamic in ((True, False), (False, True)):
            rel_proj = shift_rel_proj(empty(d_rel, extent, dtype=f32), local)
            rel_proj = rel_proj.to(bf16)
            kv_cache = empty(32, page, kv_heads, 2 * head_dim)
            key_cache, value_cache = kv_cache.split(head_dim, dim=-1)
            conv_cache = empty(16, kv_heads, page, 2 * head_dim)
            if dynamic:
                k_w = empty(width, kv_heads * head_dim)
                dyn = empty(t, kv_heads * width)
            else:
                k_w = empty(kv_heads * head_dim, width)
                dyn = None
            yield (
                f"attention_prologue T={t} local={local}",
                lambda t=t,
                rel_proj=rel_proj,
                key_cache=key_cache,
                value_cache=value_cache,
                conv_cache=conv_cache,
                k_w=k_w,
                dyn=dyn,
                local=local: attention_prologue(
                    empty(t, heads, head_dim),
                    empty(t, heads, d_rel),
                    empty(t, kv_heads * head_dim),
                    empty(t, kv_heads * head_dim),
                    empty(heads, head_dim),
                    empty(heads, d_rel),
                    None if local else empty(t, dtype=f32),
                    rel_proj,
                    k_w,
                    k_w.clone(),
                    dyn,
                    None if dyn is None else dyn.clone(),
                    conv_cache,
                    torch.zeros(t, dtype=i64),
                    torch.zeros(1, 16, dtype=i32),
                    torch.zeros(t, dtype=i32),
                    torch.zeros(t, dtype=i64),
                    torch.zeros(2, dtype=i32),
                    key_cache,
                    value_cache,
                    torch.zeros(t, dtype=i64),
                    empty(t, heads, head_dim),
                    empty(t + 128, heads, extent + 256),
                    torch.tensor([0, t], dtype=i32),
                    torch.tensor([t], dtype=i32),
                    clip_threshold=22.6,
                    rel_threshold=8.0,
                    norm_eps=1e-6,
                ),
            )

    # Relative attention: prefill (fp32 out + LSE), decode (gated / split
    # partials) and the DSpark draft (no relative logits).
    kv_cache = empty(64, page, kv_heads, 2 * head_dim)
    key_cache, value_cache = kv_cache.split(head_dim, dim=-1)
    block_table = torch.zeros(1, 64, dtype=i32)
    for q_len in (1, 2, 8, 16, 64, 300, 2048):
        for window in (0, 1024):
            yield (
                f"rel_attention prefill q_len={q_len} window={window}",
                lambda q_len=q_len, window=window: rel_attention(
                    empty(q_len, heads, head_dim),
                    key_cache,
                    value_cache,
                    empty(q_len, heads, head_dim, dtype=f32),
                    empty(heads, q_len, dtype=f32),
                    empty(q_len + 128, heads, extent + 256),
                    block_table=block_table,
                    cache_seqlens=torch.tensor([q_len], dtype=i32),
                    cu_seqlens_q=torch.tensor([0, q_len], dtype=i32),
                    max_query_len=q_len,
                    softmax_scale=head_dim**-0.5,
                    window=window,
                    num_splits=1,
                    mode=MODE_OUT_LSE,
                ),
            )
    for reqs, q_len in ((1, 1), (8, 1), (4, 2), (2, 8)):
        t = reqs * q_len
        for splits in (1, 9):
            mode = MODE_GATED if splits == 1 else MODE_PARTIAL
            out = (
                empty(t, heads * head_dim)
                if splits == 1
                else empty(splits, t, heads, head_dim, dtype=f32)
            )
            lse = None if splits == 1 else empty(splits, t, heads, dtype=f32)
            yield (
                f"rel_attention decode reqs={reqs} q_len={q_len} splits={splits}",
                lambda reqs=reqs,
                q_len=q_len,
                t=t,
                splits=splits,
                mode=mode,
                out=out,
                lse=lse: rel_attention(
                    empty(t, heads, head_dim),
                    key_cache,
                    value_cache,
                    out,
                    lse,
                    empty(t + 128, heads, extent + 256),
                    empty(t, heads * head_dim),
                    empty(t, heads),
                    block_table=block_table.expand(reqs, -1).contiguous(),
                    cache_seqlens=torch.full((reqs,), 2000, dtype=i32),
                    cu_seqlens_q=torch.arange(reqs + 1, dtype=i32) * q_len,
                    max_query_len=q_len,
                    softmax_scale=head_dim**-0.5,
                    window=1024,
                    num_splits=splits,
                    mode=mode,
                ),
            )
            yield (
                f"rel_decode_attention reqs={reqs} q_len={q_len} splits={splits}",
                lambda reqs=reqs, q_len=q_len, t=t, splits=splits: (
                    attn_ops.rel_decode_attention(
                        empty(t, heads, head_dim),
                        key_cache,
                        value_cache,
                        empty(t + 128, heads, extent + 256),
                        empty(t, heads * head_dim),
                        empty(t, heads),
                        empty(t, heads * head_dim),
                        block_table=block_table.expand(reqs, -1).contiguous(),
                        cache_seqlens=torch.full((reqs,), 2000, dtype=i32),
                        cu_seqlens_q=torch.arange(reqs + 1, dtype=i32) * q_len,
                        max_query_len=q_len,
                        num_actual_tokens=t,
                        softmax_scale=head_dim**-0.5,
                        is_local=False,
                        num_splits=splits,
                    )
                ),
            )
    for splits in (1, 4):
        yield (
            f"block_decode_attention (DSpark draft) splits={splits}",
            lambda splits=splits: attn_ops.block_decode_attention(
                empty(16, heads, head_dim),
                key_cache,
                value_cache,
                empty(16, heads * head_dim),
                block_table=block_table.expand(2, -1).contiguous(),
                cache_seqlens=torch.full((2,), 900, dtype=i32),
                cu_seqlens_q=torch.tensor([0, 8, 16], dtype=i32),
                block=8,
                window=1024,
                num_splits=splits,
            ),
        )
    for t in (1, 300, 8192):
        yield (
            f"gate_sink T={t}",
            lambda t=t: attn_ops.gate_sink(
                empty(t, heads, head_dim, dtype=f32),
                empty(t, heads * head_dim),
                empty(t, heads, dtype=f32),
                empty(t, heads),
                empty(t, heads * head_dim),
            ),
        )
        yield (
            f"compute_log_scaling_tau T={t}",
            lambda t=t: attn_ops.compute_log_scaling_tau(
                torch.zeros(t, dtype=i64), 8192, 0.1
            ),
        )

    # DSpark draft ops.
    for t in (8, 300):
        yield (
            f"dspark mix_rmsnorm T={t}",
            lambda t=t: dspark_ops.mix_rmsnorm(
                empty(t, 3, hidden), empty(3, dtype=f32), empty(hidden), 1e-6
            ),
        )
        yield (
            f"dspark silu_and_mul T={t}",
            lambda t=t: dspark_ops.silu_and_mul(empty(t, 2 * 9216)),
        )
        yield (
            f"dspark add_softcap T={t}",
            lambda t=t: dspark_ops.add_softcap(
                empty(t, 1024, dtype=f32), empty(t, 1024, dtype=f32), 30.0
            ),
        )
        yield (
            f"dspark markov_gate T={t}",
            lambda t=t: dspark_ops.markov_gate(
                empty(t, 3 * 256, dtype=f32), empty(t, 256)
            ),
        )
        yield (
            f"dspark rope_ T={t}",
            lambda t=t: dspark_ops.rope_(
                empty(t, heads, head_dim),
                torch.zeros(t, dtype=i64),
                empty(4096, head_dim, dtype=f32),
            ),
        )

    # Vision tower (ViT width 1024, 16 heads, 20 px native patches).
    for rows in (64, 4096):
        yield (
            f"vision scaled_residual_norm_ rows={rows}",
            lambda rows=rows: vision_ops.scaled_residual_norm_(
                empty(rows, 1024),
                empty(rows, 1024),
                empty(1024),
                torch.nn.LayerNorm(1024, dtype=bf16),
            ),
        )
        yield (
            f"vision apply_rotary_qk_ rows={rows}",
            lambda rows=rows: vision_ops.apply_rotary_qk_(
                empty(rows, 3 * 1024), torch.zeros(rows, 4, dtype=i32), 16
            ),
        )
        yield (
            f"vision gather_patches patches={rows}",
            lambda rows=rows: vision_ops.gather_patches(
                empty(1, 40, 40, 3),
                torch.zeros(rows, dtype=i64),
                20,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=int, default=86)
    parser.add_argument("-k", default="", help="only labels containing this")
    args = parser.parse_args()
    compiled = _install_fake_driver(args.arch)
    _patch_platform(args.arch)
    try:
        from vllm.models.inkling2.nvidia.ops import pdl, use_pdl
    except ImportError:
        print("note: no PDL guard in this tree (vllm.models.inkling2...ops.pdl)")
    else:
        assert pdl.LAUNCH_PDL == use_pdl() == (args.arch >= 90)
    failures = 0
    total = 0
    for label, thunk in _cases(args.arch):
        if args.k not in label:
            continue
        total += 1
        start = len(compiled)
        try:
            thunk()
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures += 1
            msg = " ".join(str(exc).split())[:400]
            print(f"FAIL {label}: {type(exc).__name__}: {msg}")
            if "-v" in sys.argv:
                traceback.print_exc()
            continue
        kernels = ", ".join(f"{n}:{s}B" for n, s in compiled[start:])
        print(f"ok   {label}: {kernels or '(cached)'}")
    print(f"{total - failures}/{total} cases compiled for sm_{args.arch}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
