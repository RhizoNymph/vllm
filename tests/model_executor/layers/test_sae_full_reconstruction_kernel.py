# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the capture-safe SAE full-reconstruction CUDA path.

Coverage layers:

* **CPU (always runs).**  The Triton kernel itself cannot execute on
  CPU, so the CPU layer covers everything around it: the wrapper's
  pre-launch short-circuits and dense-fallback routing (TopK and
  oversized clamp subsets route to the eager body, which runs on CPU
  tensors), reference numerics of the eager body across activation
  kinds and mask patterns, and a pure-torch simulation of the
  kernel's tile-streaming algorithm (per-token mask gate, ``BLOCK_S``
  feature tiles, in-tile clamp application, fp32 decoder
  accumulation) checked against the eager reference.
* **CUDA-only (skipped without GPU).**  Kernel-vs-eager parity for
  ReLU / JumpReLU across mixed ``recon_mask`` patterns; the TopK
  dense route on device; warmup sanity.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_full_reconstruction import (
    _apply_sae_full_reconstruction_eager,
    apply_sae_full_reconstruction,
)
from vllm.model_executor.layers.sae_full_reconstruction_kernel import (
    _fr_kernel_supports,
    apply_sae_full_recon_triton,
    warmup_apply_sae_full_recon_kernel,
)
from vllm.model_executor.layers.sae_steering import (
    ACTIVATION_CODE_JUMPRELU,
    ACTIVATION_CODE_RELU,
    ACTIVATION_CODE_TOPK,
)


def _make_inputs(
    *,
    n_tokens: int = 4,
    d_model: int = 6,
    d_sae: int = 12,
    n_clamp: int = 2,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randperm(d_sae, generator=g)[:n_clamp].sort().values.to(torch.int64)
    return {
        "hidden_states": torch.randn(n_tokens, d_model, generator=g, dtype=dtype).to(
            device
        ),
        "encoder_weight": torch.randn(d_sae, d_model, generator=g, dtype=dtype).to(
            device
        ),
        "encoder_bias": torch.randn(d_sae, generator=g, dtype=dtype).to(device),
        # Non-constant per-feature JumpReLU thresholds (fp32).  Read
        # only under the JumpReLU activation; other activations ignore
        # the tensor but still require the argument (fixed op arity).
        "threshold": (torch.rand(d_sae, generator=g) - 0.5).to(device),
        "decoder_weight": torch.randn(d_sae, d_model, generator=g, dtype=dtype).to(
            device
        ),
        "decoder_bias": torch.randn(d_model, generator=g, dtype=dtype).to(device),
        "clampable_features": feats.to(device),
    }


def _random_clamps(
    n_tokens: int, n_clamp: int, seed: int = 7, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "clamp_kind": torch.randint(
            0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
        ).to(device),
        "clamp_value": torch.randn(n_tokens, n_clamp, generator=rng).to(device),
        "clamp_only_if_active": torch.randint(
            0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
        ).to(device),
    }


def _zero_clamps(
    n_tokens: int, n_clamp: int, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    return {
        "clamp_kind": torch.zeros(n_tokens, n_clamp, dtype=torch.int8, device=device),
        "clamp_value": torch.zeros(
            n_tokens, n_clamp, dtype=torch.float32, device=device
        ),
        "clamp_only_if_active": torch.zeros(
            n_tokens, n_clamp, dtype=torch.bool, device=device
        ),
    }


def _simulate_fr_kernel(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    threshold: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    block_s: int = 8,
) -> torch.Tensor:
    """Torch mirror of ``_apply_sae_full_recon_kernel``'s algorithm.

    Per-token mask gate, ``block_s``-wide feature-tile streaming with
    in-tile clamp application, and fp32 decoder accumulation seeded
    with ``b_dec`` — the same decomposition the Triton kernel runs, so
    the tiling and clamp-scatter logic get CPU coverage even though
    the kernel itself only executes on CUDA.
    """
    n_tokens = hidden_states.shape[0]
    d_sae = encoder_weight.shape[0]
    out = torch.empty_like(hidden_states)
    for t in range(n_tokens):
        if not bool(recon_mask[t]):
            out[t] = hidden_states[t]
            continue
        acc = decoder_bias.to(torch.float32).clone()
        h32 = hidden_states[t].to(torch.float32)
        for s_off in range(0, d_sae, block_s):
            s_end = min(s_off + block_s, d_sae)
            pre = (
                encoder_bias[s_off:s_end].to(torch.float32)
                + encoder_weight[s_off:s_end].to(torch.float32) @ h32
            )
            if activation_code == ACTIVATION_CODE_RELU:
                f_tile = torch.clamp(pre, min=0.0)
            else:
                thr = threshold[s_off:s_end].to(torch.float32)
                f_tile = torch.where(pre > thr, pre, torch.zeros_like(pre))
            for c in range(clampable_features.shape[0]):
                fc = int(clampable_features[c])
                if not (s_off <= fc < s_end):
                    continue
                kind = int(clamp_kind[t, c])
                if kind == 0:
                    continue
                f_at = float(f_tile[fc - s_off])
                if bool(clamp_only_if_active[t, c]) and not f_at > 0.0:
                    continue
                value = float(clamp_value[t, c])
                f_tile[fc - s_off] = value if kind == 1 else f_at + value
            acc += f_tile @ decoder_weight[s_off:s_end].to(torch.float32)
        out[t] = acc.to(hidden_states.dtype)
    return out


# ---------------------------------------------------------------------------
# CPU layer (always runs)
# ---------------------------------------------------------------------------


class TestKernelRouting:
    """Which sites the dense masked kernel serves vs the dense fallback."""

    def test_relu_and_jumprelu_supported(self):
        assert _fr_kernel_supports(4, ACTIVATION_CODE_RELU)
        assert _fr_kernel_supports(0, ACTIVATION_CODE_RELU)
        assert _fr_kernel_supports(256, ACTIVATION_CODE_JUMPRELU)

    def test_topk_routes_to_dense_fallback(self):
        assert not _fr_kernel_supports(4, ACTIVATION_CODE_TOPK)

    def test_oversized_clamp_subset_routes_to_dense_fallback(self):
        assert not _fr_kernel_supports(257, ACTIVATION_CODE_RELU)


class TestWrapperShortCircuits:
    def test_empty_token_batch(self):
        # Pre-launch short-circuit: no kernel launch, empty output.
        h = torch.zeros(0, 4)
        out = apply_sae_full_recon_triton(
            h,
            torch.zeros(8, 4),
            torch.zeros(8),
            torch.zeros(8),
            torch.zeros(8, 4),
            torch.zeros(4),
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, 0, dtype=torch.int8),
            torch.zeros(0, 0, dtype=torch.float32),
            torch.zeros(0, 0, dtype=torch.bool),
            torch.zeros(0, dtype=torch.bool),
            ACTIVATION_CODE_RELU,
            0.0,
        )
        assert out.shape == (0, 4)


class TestDenseFallbackParity:
    """Fallback routes must match the public API on CPU tensors."""

    def test_topk_route_matches_public_api(self):
        n_tokens, n_clamp = 5, 3
        inputs = _make_inputs(n_tokens=n_tokens, n_clamp=n_clamp, seed=42)
        clamps = _random_clamps(n_tokens, n_clamp)
        rng = torch.Generator(device="cpu").manual_seed(9)
        recon_mask = torch.randint(0, 2, (n_tokens,), generator=rng, dtype=torch.bool)
        ref = apply_sae_full_reconstruction(
            **inputs,
            activation=SAEActivation.TOPK,
            activation_params={"k": 4},
            **clamps,
            recon_mask=recon_mask,
        )
        got = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_TOPK,
            activation_param=4.0,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_oversized_clamp_subset_matches_eager(self):
        n_tokens, d_sae, n_clamp = 3, 512, 300
        inputs = _make_inputs(
            n_tokens=n_tokens, d_model=6, d_sae=d_sae, n_clamp=n_clamp, seed=1
        )
        clamps = _random_clamps(n_tokens, n_clamp)
        recon_mask = torch.tensor([True, False, True])
        ref = _apply_sae_full_reconstruction_eager(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        got = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


class TestReferenceNumerics:
    """Eager-body numerics across mask patterns and activation kinds.

    The eager body is both the CPU dispatch target and the ground
    truth the CUDA kernel is validated against, so its behavior across
    the contract matrix is pinned here.
    """

    @pytest.mark.parametrize(
        "activation,params,code,param",
        [
            (SAEActivation.RELU, {}, ACTIVATION_CODE_RELU, 0.0),
            (SAEActivation.JUMPRELU, {}, ACTIVATION_CODE_JUMPRELU, 0.0),
            (SAEActivation.TOPK, {"k": 4}, ACTIVATION_CODE_TOPK, 4.0),
        ],
    )
    @pytest.mark.parametrize(
        "mask",
        [
            [True, False, True, False, True],  # mixed
            [False] * 5,  # zero active
            [True] * 5,  # all active
        ],
    )
    def test_eager_matches_public_api(self, activation, params, code, param, mask):
        n_tokens, n_clamp = 5, 3
        inputs = _make_inputs(n_tokens=n_tokens, n_clamp=n_clamp, seed=42)
        clamps = _random_clamps(n_tokens, n_clamp)
        recon_mask = torch.tensor(mask)
        ref = apply_sae_full_reconstruction(
            **inputs,
            activation=activation,
            activation_params=params,
            **clamps,
            recon_mask=recon_mask,
        )
        got = _apply_sae_full_reconstruction_eager(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=code,
            activation_param=param,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
        # Inactive rows pass through bit-identically.
        for t, active in enumerate(mask):
            if not active:
                assert torch.equal(got[t], inputs["hidden_states"][t])

    def test_unclamped_rows_are_pure_reconstruction(self):
        # kind == 0 everywhere → decode(activation(encode(h))).
        inputs = _make_inputs(n_tokens=3, n_clamp=2, seed=3)
        clamps = _zero_clamps(3, 2)
        recon_mask = torch.ones(3, dtype=torch.bool)
        got = _apply_sae_full_reconstruction_eager(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        h32 = inputs["hidden_states"].to(torch.float32)
        f = torch.clamp(
            h32 @ inputs["encoder_weight"].t() + inputs["encoder_bias"], min=0.0
        )
        expected = f @ inputs["decoder_weight"] + inputs["decoder_bias"]
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


class TestKernelAlgorithmSimulation:
    """Torch mirror of the tile-streaming kernel vs the eager body."""

    @pytest.mark.parametrize("code", [ACTIVATION_CODE_RELU, ACTIVATION_CODE_JUMPRELU])
    @pytest.mark.parametrize("block_s", [1, 4, 8])
    def test_tiled_algorithm_matches_eager(self, code, block_s):
        # Non-power-of-two d_sae/d_model exercise the tile-tail masks.
        n_tokens, d_model, d_sae, n_clamp = 6, 5, 13, 3
        inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=11
        )
        clamps = _random_clamps(n_tokens, n_clamp, seed=13)
        rng = torch.Generator(device="cpu").manual_seed(17)
        recon_mask = torch.randint(0, 2, (n_tokens,), generator=rng, dtype=torch.bool)
        ref = _apply_sae_full_reconstruction_eager(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=code,
            activation_param=0.0,
        )
        got = _simulate_fr_kernel(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=code,
            block_s=block_s,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_zero_active_is_pure_copy_through(self):
        inputs = _make_inputs(n_tokens=4, seed=5)
        clamps = _random_clamps(4, 2)
        got = _simulate_fr_kernel(
            **inputs,
            **clamps,
            recon_mask=torch.zeros(4, dtype=torch.bool),
            activation_code=ACTIVATION_CODE_RELU,
        )
        assert torch.equal(got, inputs["hidden_states"])

    def test_no_clampable_features(self):
        inputs = _make_inputs(n_tokens=3, n_clamp=0, seed=8)
        clamps = _zero_clamps(3, 0)
        recon_mask = torch.tensor([True, False, True])
        ref = _apply_sae_full_reconstruction_eager(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        got = _simulate_fr_kernel(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_only_if_active_gate_respected(self):
        # Feature 0 inactive (f == 0 under ReLU for negative pre-act);
        # only_if_active suppresses the clamp there.
        hidden = torch.tensor([[-2.0, 3.0]])
        inputs = {
            "hidden_states": hidden,
            "encoder_weight": torch.eye(2),
            "encoder_bias": torch.zeros(2),
            "threshold": torch.zeros(2),
            "decoder_weight": torch.eye(2),
            "decoder_bias": torch.zeros(2),
            "clampable_features": torch.tensor([0, 1], dtype=torch.int64),
        }
        clamps = {
            "clamp_kind": torch.tensor([[1, 1]], dtype=torch.int8),
            "clamp_value": torch.tensor([[5.0, 7.0]], dtype=torch.float32),
            "clamp_only_if_active": torch.tensor([[True, True]], dtype=torch.bool),
        }
        got = _simulate_fr_kernel(
            **inputs,
            **clamps,
            recon_mask=torch.ones(1, dtype=torch.bool),
            activation_code=ACTIVATION_CODE_RELU,
            block_s=1,
        )
        # f = ReLU([-2, 3]) = [0, 3]; clamp on feat 0 suppressed
        # (inactive), feat 1 clamped to 7 → decode = [0, 7].
        assert torch.equal(got, torch.tensor([[0.0, 7.0]]))


class TestMultipleSites:
    """Sequential FR sites compose exactly like sequential eager calls."""

    def test_two_sites_sequential(self):
        n_tokens = 4
        site_a = _make_inputs(n_tokens=n_tokens, seed=21)
        site_b = _make_inputs(n_tokens=n_tokens, seed=22)
        site_b["hidden_states"] = site_a["hidden_states"]
        clamps = _random_clamps(n_tokens, 2, seed=23)
        mask_a = torch.tensor([True, False, True, False])
        mask_b = torch.tensor([True, True, False, False])

        mid_ref = _apply_sae_full_reconstruction_eager(
            **site_a,
            **clamps,
            recon_mask=mask_a,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        site_b_after = dict(site_b)
        site_b_after["hidden_states"] = mid_ref
        final_ref = _apply_sae_full_reconstruction_eager(
            **site_b_after,
            **clamps,
            recon_mask=mask_b,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )

        mid = _simulate_fr_kernel(
            **site_a,
            **clamps,
            recon_mask=mask_a,
            activation_code=ACTIVATION_CODE_RELU,
        )
        site_b_sim = dict(site_b)
        site_b_sim["hidden_states"] = mid
        final = _simulate_fr_kernel(
            **site_b_sim,
            **clamps,
            recon_mask=mask_b,
            activation_code=ACTIVATION_CODE_RELU,
        )
        assert torch.allclose(final, final_ref, atol=1e-5, rtol=1e-5)
        # Token 3 opted into neither site → untouched end to end.
        assert torch.equal(final[3], site_a["hidden_states"][3])


class TestCpuDispatchDtype:
    def test_dtype_preserved_via_cpu_dispatch(self):
        from vllm.model_executor.layers.sae_full_reconstruction import (
            apply_sae_full_reconstruction_op,
        )

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            inputs = _make_inputs(dtype=dtype)
            clamps = _zero_clamps(4, 2)
            out = apply_sae_full_reconstruction_op(
                inputs["hidden_states"],
                inputs["encoder_weight"],
                inputs["encoder_bias"],
                inputs["threshold"],
                inputs["decoder_weight"],
                inputs["decoder_bias"],
                inputs["clampable_features"],
                clamps["clamp_kind"],
                clamps["clamp_value"],
                clamps["clamp_only_if_active"],
                torch.ones(4, dtype=torch.bool),
                ACTIVATION_CODE_RELU,
                0.0,
            )
            assert out.dtype is dtype


class TestWarmupCpu:
    def test_cpu_warmup_is_no_op(self):
        # Warmup on a CPU device must not raise / not import triton.
        warmup_apply_sae_full_recon_kernel(
            hidden_size=64,
            d_sae=128,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_warmup_skipped_for_zero_d_sae(self):
        warmup_apply_sae_full_recon_kernel(
            hidden_size=64,
            d_sae=0,  # disabled-mode equivalent
            n_clamp=0,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
        )


# ---------------------------------------------------------------------------
# CUDA-only layer
# ---------------------------------------------------------------------------


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the SAE full-reconstruction CUDA-path tests.",
)


@cuda_required
class TestCudaParity:
    @pytest.mark.parametrize(
        "activation,params,code,param",
        [
            (SAEActivation.RELU, {}, ACTIVATION_CODE_RELU, 0.0),
            (SAEActivation.JUMPRELU, {}, ACTIVATION_CODE_JUMPRELU, 0.0),
            (SAEActivation.TOPK, {"k": 6}, ACTIVATION_CODE_TOPK, 6.0),
        ],
    )
    @pytest.mark.parametrize(
        "mask_pattern",
        ["mixed", "none", "all"],
    )
    def test_cuda_matches_cpu_eager(
        self, activation, params, code, param, mask_pattern
    ):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 8, 16, 32, 4
        cpu_inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=42
        )
        cpu_clamps = _random_clamps(n_tokens, n_clamp)
        if mask_pattern == "mixed":
            rng = torch.Generator(device="cpu").manual_seed(7)
            recon_mask = torch.randint(
                0, 2, (n_tokens,), generator=rng, dtype=torch.bool
            )
        elif mask_pattern == "none":
            recon_mask = torch.zeros(n_tokens, dtype=torch.bool)
        else:
            recon_mask = torch.ones(n_tokens, dtype=torch.bool)
        ref = apply_sae_full_reconstruction(
            **cpu_inputs,
            activation=activation,
            activation_params=params,
            **cpu_clamps,
            recon_mask=recon_mask,
        )
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}
        got = apply_sae_full_recon_triton(
            **gpu_inputs,
            **gpu_clamps,
            recon_mask=recon_mask.cuda(),
            activation_code=code,
            activation_param=param,
        )
        assert got.is_cuda
        assert torch.allclose(got.cpu(), ref, atol=1e-4, rtol=1e-4)
        # Inactive rows pass through bit-identically.
        got_cpu = got.cpu()
        for t in range(n_tokens):
            if not bool(recon_mask[t]):
                assert torch.equal(got_cpu[t], cpu_inputs["hidden_states"][t])

    def test_cuda_large_shapes_multi_tile(self):
        # d_sae and d_model beyond one tile each; bf16 weights.
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 4, 300, 1000, 3
        cpu_inputs = _make_inputs(
            n_tokens=n_tokens,
            d_model=d_model,
            d_sae=d_sae,
            n_clamp=n_clamp,
            seed=42,
            dtype=torch.bfloat16,
        )
        # Pre-activations here have std ~sqrt(d_model) >> the random
        # thresholds, so bf16 input rounding alone can flip JumpReLU on
        # borderline features between the two accumulation orders.  Push
        # any threshold within 0.5 of a pre-activation safely below all
        # of them so the comparison is deterministic.
        pre = (
            cpu_inputs["hidden_states"].float()
            @ cpu_inputs["encoder_weight"].float().T
            + cpu_inputs["encoder_bias"].float()
        )
        thr = cpu_inputs["threshold"]
        margin = (pre - thr).abs().amin(dim=0)
        cpu_inputs["threshold"] = torch.where(
            margin < 0.5, pre.amin(dim=0) - 1.0, thr
        )
        cpu_clamps = _random_clamps(n_tokens, n_clamp)
        recon_mask = torch.tensor([True, False, True, False])
        # Reference in fp32 from the same bf16 inputs: the kernel
        # accumulates fp32 end-to-end with a single output cast, while
        # the eager body on bf16 tensors rounds every intermediate to
        # bf16 — comparing against the latter measures the reference's
        # rounding, not the kernel's.
        ref = apply_sae_full_reconstruction(
            **{
                k: v.float() if torch.is_floating_point(v) else v
                for k, v in cpu_inputs.items()
            },
            activation=SAEActivation.JUMPRELU,
            activation_params={},
            **cpu_clamps,
            recon_mask=recon_mask,
        )
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}
        got = apply_sae_full_recon_triton(
            **gpu_inputs,
            **gpu_clamps,
            recon_mask=recon_mask.cuda(),
            activation_code=ACTIVATION_CODE_JUMPRELU,
            activation_param=0.0,
        )
        assert got.dtype is torch.bfloat16
        assert torch.allclose(got.cpu().float(), ref, atol=5e-2, rtol=1e-2)
        assert torch.equal(got.cpu()[1], cpu_inputs["hidden_states"][1])


@cuda_required
class TestCudaWarmup:
    def test_warmup_runs_without_error(self):
        warmup_apply_sae_full_recon_kernel(
            hidden_size=64,
            d_sae=128,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
        )

    def test_warmup_topk_route_runs_without_error(self):
        warmup_apply_sae_full_recon_kernel(
            hidden_size=64,
            d_sae=128,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_TOPK,
            activation_param=8.0,
        )


@cuda_required
class TestKernelCudaGraph:
    """The FR kernel must capture and replay correctly under a CUDA graph.

    The delta kernel has this coverage in
    ``test_sae_steering_kernel.py::TestKernelCudaGraph``; this is the
    full-reconstruction analogue, including the per-token ``recon_mask``
    copy-through inside the captured graph.
    """

    def test_capture_and_replay_matches_eager(self):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 4, 32, 64, 4
        cpu_inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=2024
        )
        cpu_clamps = _random_clamps(n_tokens, n_clamp, seed=2025)
        recon_mask = torch.tensor([True, False, True, True])
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}
        gpu_mask = recon_mask.cuda()

        # Warm the kernel JIT before capture; capture cannot include
        # Triton compilation.
        warmup_apply_sae_full_recon_kernel(
            hidden_size=d_model,
            d_sae=d_sae,
            n_clamp=n_clamp,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        out_buf = torch.empty_like(gpu_inputs["hidden_states"])
        with torch.cuda.graph(graph):
            captured = apply_sae_full_recon_triton(
                **gpu_inputs,
                **gpu_clamps,
                recon_mask=gpu_mask,
                activation_code=ACTIVATION_CODE_RELU,
                activation_param=0.0,
            )
            out_buf.copy_(captured)
        graph.replay()
        torch.cuda.synchronize()

        ref = apply_sae_full_reconstruction(
            **cpu_inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **cpu_clamps,
            recon_mask=recon_mask,
        )
        assert torch.allclose(out_buf.cpu(), ref, atol=1e-4, rtol=1e-4)
        # Masked-off rows pass through bit-identically inside the graph too.
        out_cpu = out_buf.cpu()
        for t in range(n_tokens):
            if not bool(recon_mask[t]):
                assert torch.equal(out_cpu[t], cpu_inputs["hidden_states"][t])

    def test_replay_sees_updated_clamp_tables(self):
        """A replay after in-place clamp-table mutation must read the new
        values — the property FULL-cudagraph serving relies on."""
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 4, 32, 64, 4
        cpu_inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=2026
        )
        clamps_a = _random_clamps(n_tokens, n_clamp, seed=2027)
        clamps_b = _random_clamps(n_tokens, n_clamp, seed=2028)
        recon_mask = torch.ones(n_tokens, dtype=torch.bool)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in clamps_a.items()}
        gpu_mask = recon_mask.cuda()

        warmup_apply_sae_full_recon_kernel(
            hidden_size=d_model,
            d_sae=d_sae,
            n_clamp=n_clamp,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        out_buf = torch.empty_like(gpu_inputs["hidden_states"])
        with torch.cuda.graph(graph):
            captured = apply_sae_full_recon_triton(
                **gpu_inputs,
                **gpu_clamps,
                recon_mask=gpu_mask,
                activation_code=ACTIVATION_CODE_RELU,
                activation_param=0.0,
            )
            out_buf.copy_(captured)

        def _ref(clamps: dict) -> torch.Tensor:
            return apply_sae_full_reconstruction(
                **cpu_inputs,
                activation=SAEActivation.RELU,
                activation_params={},
                **clamps,
                recon_mask=recon_mask,
            )

        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out_buf.cpu(), _ref(clamps_a), atol=1e-4, rtol=1e-4)

        for k, v in clamps_b.items():
            gpu_clamps[k].copy_(v.cuda())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out_buf.cpu(), _ref(clamps_b), atol=1e-4, rtol=1e-4)
