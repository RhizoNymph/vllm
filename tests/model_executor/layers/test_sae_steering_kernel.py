# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-2 tests for the fused SAE feature-surgery Triton kernel.

Two layers of coverage:

* **CPU layer (always runs).**  The activation-code helpers
  round-trip, the registered custom op
  ``torch.ops.vllm.apply_sae_delta`` is invocable, and the
  ``apply_sae_delta_op`` eager body produces the same output as the
  public :func:`apply_sae_delta` for representative inputs.  The
  custom-op registration is exercised as a live import so a regression
  in :func:`direct_register_custom_op` plumbing fails fast.

* **CUDA layer (skipped without GPU).**  Numeric parity between the
  Triton kernel and the eager body across activations, dtypes and
  shapes; CUDA-graph capture-and-replay; and warmup sanity.  These
  tests exist so the kernel is verified end-to-end when a GPU is
  available; they don't run in the CPU-only test environment.

The kernel under test is :func:`apply_sae_delta_triton` in
:mod:`vllm.model_executor.layers.sae_steering_kernel`; the eager
reference it must match is :func:`apply_sae_delta` in
:mod:`vllm.model_executor.layers.sae_steering`.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    _ACTIVATION_TO_CODE,
    _CODE_TO_ACTIVATION,
    ACTIVATION_CODE_JUMPRELU,
    ACTIVATION_CODE_RELU,
    ACTIVATION_CODE_TOPK,
    _activation_to_scalar,
    _scalar_to_activation_params,
    apply_sae_delta,
    apply_sae_delta_indexed_op,
    apply_sae_delta_op,
)
from vllm.model_executor.layers.sae_steering_kernel import (
    _choose_block_c,
    _choose_block_h,
    _kernel_supports,
    _next_power_of_two,
)


# Some CUDA-only tests use a small synthetic input set.  Reuse the
# random-input helper structure from ``test_sae_steering_op`` for
# parity, but inline here so the file is self-contained.
def _make_random_inputs(
    *,
    n_tokens: int = 4,
    d_model: int = 8,
    n_clamp: int = 3,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "hidden_states": torch.randn(n_tokens, d_model, generator=g, dtype=dtype).to(
            device
        ),
        "encoder_weight": torch.randn(n_clamp, d_model, generator=g, dtype=dtype).to(
            device
        ),
        "encoder_bias": torch.randn(n_clamp, generator=g, dtype=dtype).to(device),
        "decoder_weight": torch.randn(n_clamp, d_model, generator=g, dtype=dtype).to(
            device
        ),
    }


def _random_clamps(
    *,
    n_tokens: int,
    n_clamp: int,
    seed: int,
    device: torch.device | str = "cpu",
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


class TestActivationCodeHelpers:
    """The packed (code, scalar) form must round-trip the enum + dict."""

    def test_relu_round_trips_with_no_param(self):
        scalar = _activation_to_scalar(SAEActivation.RELU, {})
        assert scalar == 0.0
        params = _scalar_to_activation_params(SAEActivation.RELU, scalar)
        assert params == {}

    def test_jumprelu_round_trips_threshold(self):
        scalar = _activation_to_scalar(SAEActivation.JUMPRELU, {"threshold": 0.7})
        assert scalar == pytest.approx(0.7)
        params = _scalar_to_activation_params(SAEActivation.JUMPRELU, scalar)
        assert params == {"threshold": pytest.approx(0.7)}

    def test_topk_round_trips_k(self):
        scalar = _activation_to_scalar(SAEActivation.TOPK, {"k": 4})
        assert scalar == 4.0
        params = _scalar_to_activation_params(SAEActivation.TOPK, scalar)
        assert params == {"k": pytest.approx(4.0)}

    def test_code_table_is_complete(self):
        # Every activation must have a forward and inverse mapping.
        for activation in SAEActivation:
            code = _ACTIVATION_TO_CODE[activation]
            assert _CODE_TO_ACTIVATION[code] is activation

    def test_activation_codes_match_kernel_module(self):
        # The kernel module owns its own copy of these constants; the
        # two must agree or the kernel's switch will pick the wrong
        # branch.
        from vllm.model_executor.layers import sae_steering_kernel as kern

        assert kern.ACTIVATION_CODE_RELU == ACTIVATION_CODE_RELU
        assert kern.ACTIVATION_CODE_JUMPRELU == ACTIVATION_CODE_JUMPRELU
        assert kern.ACTIVATION_CODE_TOPK == ACTIVATION_CODE_TOPK


class TestKernelHelpers:
    """Block-size pickers and capability gates."""

    @pytest.mark.parametrize(
        "x,expected",
        [(0, 1), (1, 1), (2, 2), (3, 4), (8, 8), (9, 16), (255, 256), (256, 256)],
    )
    def test_next_power_of_two(self, x, expected):
        assert _next_power_of_two(x) == expected

    @pytest.mark.parametrize(
        "h,expected",
        [(1, 1), (2, 2), (3, 4), (1024, 1024), (2048, 2048), (4096, 2048)],
    )
    def test_block_h_picker(self, h, expected):
        assert _choose_block_h(h) == expected

    @pytest.mark.parametrize(
        "n_clamp,expected", [(1, 1), (3, 4), (8, 8), (9, 16), (64, 64)]
    )
    def test_block_c_picker(self, n_clamp, expected):
        assert _choose_block_c(n_clamp) == expected

    def test_kernel_supports_typical_clamp_counts(self):
        # The common case (≤ 64 clampable features) is supported.
        for n in (1, 8, 16, 32, 64, 128, 256):
            assert _kernel_supports(n) is True

    def test_kernel_rejects_pathological_clamp_counts(self):
        # 257 rounds up to 512, which exceeds the cap.
        assert _kernel_supports(257) is False
        assert _kernel_supports(1024) is False


class TestCustomOpRegistration:
    """``torch.ops.vllm.apply_sae_delta`` must be registered and callable."""

    def test_custom_op_is_registered(self):
        # Importing the module is enough; ``direct_register_custom_op``
        # runs at import time.
        from vllm.model_executor.layers import sae_steering  # noqa: F401

        assert hasattr(torch.ops.vllm, "apply_sae_delta"), (
            "torch.ops.vllm.apply_sae_delta should be registered "
            "by importing vllm.model_executor.layers.sae_steering."
        )
        assert hasattr(torch.ops.vllm, "apply_sae_delta_indexed"), (
            "torch.ops.vllm.apply_sae_delta_indexed should be registered "
            "by importing vllm.model_executor.layers.sae_steering."
        )

    def test_op_func_equivalent_to_public_api_on_cpu(self):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 4, 6, 3
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=42
        )
        clamps = _random_clamps(n_tokens=n_tokens, n_clamp=n_clamp, seed=7)

        # Public API path (validates + translates + delegates).
        public = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
        )

        # Direct op-func path (skips validation, takes int/float params).
        op = apply_sae_delta_op(
            inputs["hidden_states"],
            inputs["encoder_weight"],
            inputs["encoder_bias"],
            inputs["decoder_weight"],
            clamps["clamp_kind"],
            clamps["clamp_value"],
            clamps["clamp_only_if_active"],
            torch.ones(1, dtype=torch.bool),
            ACTIVATION_CODE_RELU,
            0.0,
        )

        assert torch.allclose(public, op)

    def test_indexed_op_func_equivalent_to_public_api_on_cpu(self):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 4, 6, 3
        inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=42
        )
        clamps = _random_clamps(n_tokens=n_tokens, n_clamp=n_clamp, seed=7)

        public = apply_sae_delta(
            **inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
        )
        indexed = apply_sae_delta_indexed_op(
            inputs["hidden_states"],
            inputs["encoder_weight"],
            inputs["encoder_bias"],
            inputs["decoder_weight"],
            clamps["clamp_kind"],
            clamps["clamp_value"],
            clamps["clamp_only_if_active"],
            torch.arange(n_tokens, dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            ACTIVATION_CODE_RELU,
            0.0,
        )

        assert torch.allclose(public, indexed)

    def test_indexed_op_inactive_skips_table_gather_on_cpu(self):
        hidden = torch.randn(3, 4)
        encoder_weight = torch.randn(2, 4)
        encoder_bias = torch.randn(2)
        decoder_weight = torch.randn(2, 4)
        kind_table = torch.full((2, 2), 1, dtype=torch.int8)
        value_table = torch.full((2, 2), float("nan"), dtype=torch.float32)
        only_table = torch.zeros(2, 2, dtype=torch.bool)
        # Out-of-range rows prove the inactive path returns before indexing.
        index = torch.full((3,), 99, dtype=torch.long)

        out = apply_sae_delta_indexed_op(
            hidden,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            kind_table,
            value_table,
            only_table,
            index,
            torch.zeros(1, dtype=torch.bool),
            ACTIVATION_CODE_RELU,
            0.0,
        )

        torch.testing.assert_close(out, hidden)
        assert out.data_ptr() != hidden.data_ptr()

    def test_op_func_jumprelu_threshold_param(self):
        # Distinct threshold values must produce distinct outputs to
        # confirm the scalar parameter actually flows through.
        torch.manual_seed(1)
        inputs = _make_random_inputs(n_tokens=2, d_model=4, n_clamp=2, seed=1)
        clamps = _random_clamps(n_tokens=2, n_clamp=2, seed=2)

        loose = apply_sae_delta(
            **inputs,
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": -10.0},  # admits everything
            **clamps,
        )
        strict = apply_sae_delta(
            **inputs,
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": 10.0},  # gates almost everything
            **clamps,
        )
        # If threshold weren't being threaded through, the two would
        # come out identical.  They almost certainly should not.
        assert not torch.allclose(loose, strict)


class TestLayerDispatchUsesCustomOp:
    """``apply_layer_sae_delta`` must call through the indexed registered op.

    Verified by a monkeypatch hook on ``torch.ops.vllm.apply_sae_delta_indexed``
    that records call counts; this guards against a regression where
    the layer shim calls the eager Python function directly and skips
    the torch.compile fence.
    """

    def test_layer_dispatch_invokes_torch_ops(self, monkeypatch):
        from torch import nn

        from vllm.model_executor.layers.sae_steering import (
            apply_layer_sae_delta,
            register_sae_buffers,
            register_sae_index_buffer,
        )
        from vllm.model_executor.layers.steering import SteeringHookPoint

        layer = nn.Module()
        layer.layer_idx = 0
        register_sae_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=2,
            hidden_size=4,
            max_sae_configs=4,
            dtype=torch.float32,
        )
        register_sae_index_buffer(layer, max_tokens=8)

        original = torch.ops.vllm.apply_sae_delta_indexed
        calls = {"n": 0}

        def counting_op(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        # Patch the OpOverloadPacket attribute the layer shim resolves
        # at call time.  We can't reassign the C++ op directly, so we
        # patch the Python-visible accessor on the ops module.
        monkeypatch.setattr(torch.ops.vllm, "apply_sae_delta_indexed", counting_op)

        h = torch.randn(3, 4)
        out = apply_layer_sae_delta(layer, h, SteeringHookPoint.POST_BLOCK)
        assert out.shape == h.shape
        assert calls["n"] == 1

    def test_layer_dispatch_short_circuits_when_n_clamp_zero(self):
        # The guard before the op call must avoid launching a kernel
        # with an empty clamp set.  Verified by checking that the
        # output is the input unchanged (clone semantics).
        from torch import nn

        from vllm.model_executor.layers.sae_steering import (
            apply_layer_sae_delta,
            register_sae_buffers,
            register_sae_index_buffer,
        )
        from vllm.model_executor.layers.steering import SteeringHookPoint

        layer = nn.Module()
        layer.layer_idx = 0
        register_sae_buffers(
            layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            activation=SAEActivation.RELU,
            activation_params={},
            n_clamp=0,
            hidden_size=4,
            max_sae_configs=4,
            dtype=torch.float32,
        )
        register_sae_index_buffer(layer, max_tokens=8)

        h = torch.randn(2, 4)
        out = apply_layer_sae_delta(layer, h, SteeringHookPoint.POST_BLOCK)
        # n_clamp == 0 ⇒ same tensor (no op + no allocation).
        assert torch.equal(out, h)


# ---------------------------------------------------------------------------
# CUDA-only kernel parity tests
# ---------------------------------------------------------------------------


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the SAE Triton kernel parity tests.",
)


@cuda_required
class TestKernelParityFp32:
    """Triton kernel must match the eager body within fp32 tolerance."""

    @pytest.mark.parametrize(
        "activation,params",
        [
            (SAEActivation.RELU, {}),
            (SAEActivation.JUMPRELU, {"threshold": 0.5}),
            (SAEActivation.TOPK, {"k": 2}),
        ],
    )
    @pytest.mark.parametrize(
        "n_tokens,d_model,n_clamp", [(1, 8, 1), (4, 16, 3), (8, 64, 5), (16, 1024, 8)]
    )
    def test_random_inputs(self, activation, params, n_tokens, d_model, n_clamp):
        cpu_inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=42
        )
        cpu_clamps = _random_clamps(n_tokens=n_tokens, n_clamp=n_clamp, seed=7)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}

        # Eager body via public API on CPU.
        ref = apply_sae_delta(
            **cpu_inputs,
            activation=activation,
            activation_params=params,
            **cpu_clamps,
        )
        # Triton kernel via public API on CUDA.
        got = apply_sae_delta(
            **gpu_inputs,
            activation=activation,
            activation_params=params,
            **gpu_clamps,
        )
        assert got.is_cuda
        assert torch.allclose(got.cpu(), ref, atol=1e-4, rtol=1e-4)


@cuda_required
class TestKernelParityLowPrecision:
    """bf16 / fp16 inputs must match eager body within looser tolerance."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "activation,params",
        [
            (SAEActivation.RELU, {}),
            (SAEActivation.JUMPRELU, {"threshold": 0.0}),
        ],
    )
    def test_low_precision(self, dtype, activation, params):
        cpu_inputs = _make_random_inputs(
            n_tokens=4, d_model=64, n_clamp=4, dtype=dtype, seed=99
        )
        cpu_clamps = _random_clamps(n_tokens=4, n_clamp=4, seed=101)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}

        ref = apply_sae_delta(
            **cpu_inputs,
            activation=activation,
            activation_params=params,
            **cpu_clamps,
        )
        got = apply_sae_delta(
            **gpu_inputs,
            activation=activation,
            activation_params=params,
            **gpu_clamps,
        )
        assert got.dtype is dtype
        atol = 5e-2 if dtype is torch.bfloat16 else 1e-2
        assert torch.allclose(got.cpu().float(), ref.float(), atol=atol, rtol=atol)


@cuda_required
class TestKernelEdgeCases:
    """Boundaries that the host wrapper short-circuits or pads."""

    def test_empty_token_batch_returns_empty(self):
        h = torch.zeros(0, 8, device="cuda")
        W_enc = torch.zeros(2, 8, device="cuda")
        b_enc = torch.zeros(2, device="cuda")
        W_dec = torch.zeros(2, 8, device="cuda")
        clamps = {
            "clamp_kind": torch.zeros(0, 2, dtype=torch.int8, device="cuda"),
            "clamp_value": torch.zeros(0, 2, device="cuda"),
            "clamp_only_if_active": torch.zeros(0, 2, dtype=torch.bool, device="cuda"),
        }
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=W_enc,
            encoder_bias=b_enc,
            decoder_weight=W_dec,
            activation=SAEActivation.RELU,
            activation_params={},
            **clamps,
        )
        assert out.shape == (0, 8)

    def test_zero_n_clamp_returns_clone(self):
        h = torch.randn(3, 8, device="cuda")
        out = apply_sae_delta(
            hidden_states=h,
            encoder_weight=torch.zeros(0, 8, device="cuda"),
            encoder_bias=torch.zeros(0, device="cuda"),
            decoder_weight=torch.zeros(0, 8, device="cuda"),
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=torch.zeros(3, 0, dtype=torch.int8, device="cuda"),
            clamp_value=torch.zeros(3, 0, device="cuda"),
            clamp_only_if_active=torch.zeros(3, 0, dtype=torch.bool, device="cuda"),
        )
        assert torch.equal(out, h)

    def test_non_power_of_two_d_model(self):
        # d_model=130 forces the BLOCK_H loop to do a partial second
        # tile (BLOCK_H=128 covers the first 128, then 2 lanes masked).
        torch.manual_seed(0)
        cpu_inputs = _make_random_inputs(n_tokens=3, d_model=130, n_clamp=2, seed=5)
        cpu_clamps = _random_clamps(n_tokens=3, n_clamp=2, seed=11)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}

        ref = apply_sae_delta(
            **cpu_inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **cpu_clamps,
        )
        got = apply_sae_delta(
            **gpu_inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **gpu_clamps,
        )
        assert torch.allclose(got.cpu(), ref, atol=1e-4, rtol=1e-4)


@cuda_required
class TestIndexedKernelParity:
    """The production indexed-table kernel must match gathered eager inputs."""

    def test_indexed_random_rows_match_eager(self):
        n_tokens, d_model, n_clamp, n_rows = 5, 17, 3, 4
        cpu_inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=123
        )
        row_clamps = _random_clamps(n_tokens=n_rows, n_clamp=n_clamp, seed=321)
        index = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

        ref = apply_sae_delta(
            **cpu_inputs,
            activation=SAEActivation.JUMPRELU,
            activation_params={"threshold": 0.25},
            clamp_kind=row_clamps["clamp_kind"][index],
            clamp_value=row_clamps["clamp_value"][index],
            clamp_only_if_active=row_clamps["clamp_only_if_active"][index],
        )

        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        got = apply_sae_delta_indexed_op(
            gpu_inputs["hidden_states"],
            gpu_inputs["encoder_weight"],
            gpu_inputs["encoder_bias"],
            gpu_inputs["decoder_weight"],
            row_clamps["clamp_kind"].cuda(),
            row_clamps["clamp_value"].cuda(),
            row_clamps["clamp_only_if_active"].cuda(),
            index.cuda(),
            torch.ones(1, dtype=torch.bool, device="cuda"),
            ACTIVATION_CODE_JUMPRELU,
            0.25,
        )

        assert got.is_cuda
        torch.testing.assert_close(got.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_indexed_inactive_skips_out_of_range_rows(self):
        hidden = torch.randn(3, 8, device="cuda")
        encoder_weight = torch.randn(2, 8, device="cuda")
        encoder_bias = torch.randn(2, device="cuda")
        decoder_weight = torch.randn(2, 8, device="cuda")
        kind_table = torch.ones(2, 2, dtype=torch.int8, device="cuda")
        value_table = torch.full((2, 2), float("nan"), device="cuda")
        only_table = torch.zeros(2, 2, dtype=torch.bool, device="cuda")
        index = torch.full((3,), 99, dtype=torch.long, device="cuda")

        out = apply_sae_delta_indexed_op(
            hidden,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            kind_table,
            value_table,
            only_table,
            index,
            torch.zeros(1, dtype=torch.bool, device="cuda"),
            ACTIVATION_CODE_RELU,
            0.0,
        )

        torch.testing.assert_close(out, hidden)

    def test_indexed_unsupported_inactive_skips_out_of_range_rows(self):
        """The large-n_clamp fallback must preserve kernel inactive behavior."""
        n_tokens, d_model, n_clamp, n_rows = 3, 8, 257, 2
        hidden = torch.randn(n_tokens, d_model, device="cuda")
        encoder_weight = torch.randn(n_clamp, d_model, device="cuda")
        encoder_bias = torch.randn(n_clamp, device="cuda")
        decoder_weight = torch.randn(n_clamp, d_model, device="cuda")
        kind_table = torch.ones(n_rows, n_clamp, dtype=torch.int8, device="cuda")
        value_table = torch.ones(n_rows, n_clamp, device="cuda")
        only_table = torch.zeros(n_rows, n_clamp, dtype=torch.bool, device="cuda")
        index = torch.full((n_tokens,), 99, dtype=torch.long, device="cuda")

        from vllm.model_executor.layers.sae_steering_kernel import (
            apply_sae_delta_indexed_triton,
        )

        assert not _kernel_supports(n_clamp)
        out = apply_sae_delta_indexed_triton(
            hidden,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            kind_table,
            value_table,
            only_table,
            index,
            torch.zeros(1, dtype=torch.bool, device="cuda"),
            ACTIVATION_CODE_RELU,
            0.0,
        )

        torch.testing.assert_close(out, hidden)


@cuda_required
class TestKernelCudaGraph:
    """The kernel must capture and replay correctly under a CUDA graph."""

    def test_capture_and_replay_matches_eager(self):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp = 4, 32, 3
        cpu_inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=2024
        )
        cpu_clamps = _random_clamps(n_tokens=n_tokens, n_clamp=n_clamp, seed=2025)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}

        # Warm the kernel JIT before capture; capture cannot include
        # Triton compilation.
        from vllm.model_executor.layers.sae_steering_kernel import (
            apply_sae_delta_triton,
            warmup_apply_sae_delta_kernel,
        )

        warmup_apply_sae_delta_kernel(
            hidden_size=d_model,
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
        any_active = torch.ones(1, dtype=torch.bool, device="cuda")
        with torch.cuda.graph(graph):
            captured = apply_sae_delta_triton(
                gpu_inputs["hidden_states"],
                gpu_inputs["encoder_weight"],
                gpu_inputs["encoder_bias"],
                gpu_inputs["decoder_weight"],
                gpu_clamps["clamp_kind"],
                gpu_clamps["clamp_value"],
                gpu_clamps["clamp_only_if_active"],
                any_active,
                ACTIVATION_CODE_RELU,
                0.0,
            )
            out_buf.copy_(captured)
        graph.replay()
        torch.cuda.synchronize()

        # Compare against eager body via public API on CPU.
        ref = apply_sae_delta(
            **cpu_inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            **cpu_clamps,
        )
        assert torch.allclose(out_buf.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_indexed_capture_and_replay_matches_eager(self):
        torch.manual_seed(0)
        n_tokens, d_model, n_clamp, n_rows = 4, 32, 3, 3
        cpu_inputs = _make_random_inputs(
            n_tokens=n_tokens, d_model=d_model, n_clamp=n_clamp, seed=2026
        )
        row_clamps = _random_clamps(n_tokens=n_rows, n_clamp=n_clamp, seed=2027)
        index = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}

        from vllm.model_executor.layers.sae_steering_kernel import (
            apply_sae_delta_indexed_triton,
            warmup_apply_sae_delta_kernel,
        )

        warmup_apply_sae_delta_kernel(
            hidden_size=d_model,
            n_clamp=n_clamp,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )

        kind_table = row_clamps["clamp_kind"].cuda()
        value_table = row_clamps["clamp_value"].cuda()
        only_table = row_clamps["clamp_only_if_active"].cuda()
        gpu_index = index.cuda()
        any_active = torch.ones(1, dtype=torch.bool, device="cuda")

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        out_buf = torch.empty_like(gpu_inputs["hidden_states"])
        with torch.cuda.graph(graph):
            captured = apply_sae_delta_indexed_triton(
                gpu_inputs["hidden_states"],
                gpu_inputs["encoder_weight"],
                gpu_inputs["encoder_bias"],
                gpu_inputs["decoder_weight"],
                kind_table,
                value_table,
                only_table,
                gpu_index,
                any_active,
                ACTIVATION_CODE_RELU,
                0.0,
            )
            out_buf.copy_(captured)
        graph.replay()
        torch.cuda.synchronize()

        ref = apply_sae_delta(
            **cpu_inputs,
            activation=SAEActivation.RELU,
            activation_params={},
            clamp_kind=row_clamps["clamp_kind"][index],
            clamp_value=row_clamps["clamp_value"][index],
            clamp_only_if_active=row_clamps["clamp_only_if_active"][index],
        )
        torch.testing.assert_close(out_buf.cpu(), ref, atol=1e-4, rtol=1e-4)


@cuda_required
class TestKernelWarmup:
    """Warmup is a no-op on CPU and must not raise on CUDA."""

    def test_warmup_runs_without_error(self):
        from vllm.model_executor.layers.sae_steering_kernel import (
            warmup_apply_sae_delta_kernel,
        )

        warmup_apply_sae_delta_kernel(
            hidden_size=64,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        # JumpReLU specialisation should JIT independently.
        warmup_apply_sae_delta_kernel(
            hidden_size=64,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_JUMPRELU,
            activation_param=0.5,
        )

    def test_warmup_is_noop_on_cpu(self):
        # Calling on a CPU device must be a no-op (and not import
        # triton or fail).
        from vllm.model_executor.layers.sae_steering_kernel import (
            warmup_apply_sae_delta_kernel,
        )

        warmup_apply_sae_delta_kernel(
            hidden_size=64,
            n_clamp=4,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cpu"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )

    def test_warmup_skipped_when_no_clamp_features(self):
        from vllm.model_executor.layers.sae_steering_kernel import (
            warmup_apply_sae_delta_kernel,
        )

        # n_clamp=0 → no kernel JIT possible; must short-circuit.
        warmup_apply_sae_delta_kernel(
            hidden_size=64,
            n_clamp=0,
            table_dtype=torch.float32,
            compute_dtype=torch.float32,
            device=torch.device("cuda"),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
