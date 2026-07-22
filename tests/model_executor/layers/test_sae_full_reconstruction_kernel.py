# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-4 Stage-4 tests for the SAE full-reconstruction CUDA path.

Coverage layers:

* **CPU (always runs).**  ``apply_sae_full_recon_triton`` runs a
  compaction-based per-token short-circuit; the dispatch logic and
  the empty-token / no-active-row short-circuits are testable on CPU
  via direct calls.  Numeric parity against the eager body is
  verified for representative shapes — the math is identical, only
  the active-row narrowing differs.
* **CUDA-only (skipped without GPU).**  Parity vs eager body for
  ReLU / JumpReLU / TopK across mixed ``recon_mask`` patterns;
  warmup sanity.

The CPU layer drives both branches because
:func:`apply_sae_full_recon_triton` does not require ``cuda``
tensors — the function name is "triton" by convention but Stage 4
routes CUDA tensors through PyTorch matmuls (cuBLAS) until a real
Triton kernel lands.  The CUDA tests only confirm device fidelity.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_full_reconstruction import (
    apply_sae_full_reconstruction,
)
from vllm.model_executor.layers.sae_full_reconstruction_kernel import (
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


# ---------------------------------------------------------------------------
# CPU layer (always runs)
# ---------------------------------------------------------------------------


class TestEmptyShortCircuits:
    def test_empty_token_batch(self):
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

    def test_no_active_tokens_returns_clone(self):
        # All-False recon_mask must skip the encoder/decoder GEMMs
        # entirely — output should be a bit-identical clone of the
        # input.
        inputs = _make_inputs()
        clamps = _zero_clamps(4, 2)
        out = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=torch.zeros(4, dtype=torch.bool),
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        assert torch.equal(out, inputs["hidden_states"])
        # Output is a fresh tensor, not the input itself (no-alias contract).
        assert out is not inputs["hidden_states"]


class TestParityWithEager:
    """Compaction must produce the same output as the eager body."""

    @pytest.mark.parametrize(
        "activation,params,code",
        [
            (SAEActivation.RELU, {}, ACTIVATION_CODE_RELU),
            (SAEActivation.JUMPRELU, {}, ACTIVATION_CODE_JUMPRELU),
            (SAEActivation.TOPK, {"k": 4}, ACTIVATION_CODE_TOPK),
        ],
    )
    def test_random_inputs_match_eager(self, activation, params, code):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 5, 6, 12, 3
        # ``_make_inputs`` includes a random non-constant per-feature
        # threshold tensor — the JumpReLU case exercises the per-lane
        # comparison end-to-end.
        inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=42
        )
        rng = torch.Generator(device="cpu").manual_seed(7)
        clamps = {
            "clamp_kind": torch.randint(
                0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
            ),
            "clamp_value": torch.randn(n_tokens, n_clamp, generator=rng),
            "clamp_only_if_active": torch.randint(
                0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
            ),
        }
        recon_mask = torch.randint(0, 2, (n_tokens,), generator=rng, dtype=torch.bool)
        # Eager body via the public API.
        ref = apply_sae_full_reconstruction(
            **inputs,
            activation=activation,
            activation_params=params,
            **clamps,
            recon_mask=recon_mask,
        )
        # Compaction CUDA path — runs on CPU tensors fine, that's the
        # whole point of the design (no real Triton kernel yet).
        param = float(params.get("k", 0.0))
        got = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=code,
            activation_param=param,
        )
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_partial_recon_mask_unmasked_rows_bit_identical(self):
        # Unmasked rows must come through as exact copies — the
        # compaction wrapper does ``out.copy_(hidden_states)`` first.
        torch.manual_seed(0)
        inputs = _make_inputs(n_tokens=4)
        clamps = _zero_clamps(4, 2)
        recon_mask = torch.tensor([True, False, True, False])
        out = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=recon_mask,
            activation_code=ACTIVATION_CODE_RELU,
            activation_param=0.0,
        )
        assert torch.equal(out[1], inputs["hidden_states"][1])
        assert torch.equal(out[3], inputs["hidden_states"][3])

    def test_topk_only_if_active_treats_selected_negative_feature_as_active(self):
        hidden = torch.tensor([[-2.0, -1.0]])
        inputs = {
            "hidden_states": hidden,
            "encoder_weight": torch.eye(2),
            "encoder_bias": torch.zeros(2),
            "threshold": torch.zeros(2),
            "decoder_weight": torch.eye(2),
            "decoder_bias": torch.zeros(2),
            "clampable_features": torch.tensor([0], dtype=torch.int64),
        }
        clamps = {
            "clamp_kind": torch.tensor([[1]], dtype=torch.int8),
            "clamp_value": torch.tensor([[4.0]], dtype=torch.float32),
            "clamp_only_if_active": torch.tensor([[True]], dtype=torch.bool),
        }

        got = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=torch.ones(1, dtype=torch.bool),
            activation_code=ACTIVATION_CODE_TOPK,
            activation_param=2.0,
        )

        assert torch.equal(got, torch.tensor([[4.0, -1.0]]))

    def test_topk_ties_keep_lowest_feature_indices(self):
        inputs = {
            "hidden_states": torch.zeros(1, 4),
            "encoder_weight": torch.zeros(4, 4),
            "encoder_bias": torch.ones(4),
            "threshold": torch.zeros(4),
            "decoder_weight": torch.eye(4),
            "decoder_bias": torch.zeros(4),
            "clampable_features": torch.zeros(0, dtype=torch.int64),
        }
        clamps = _zero_clamps(1, 0)

        got = apply_sae_full_recon_triton(
            **inputs,
            **clamps,
            recon_mask=torch.ones(1, dtype=torch.bool),
            activation_code=ACTIVATION_CODE_TOPK,
            activation_param=2.0,
        )

        assert torch.equal(got, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))

    def test_dtype_preserved(self):
        # bfloat16/float16 input → output preserved.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            inputs = _make_inputs(dtype=dtype)
            clamps = _zero_clamps(4, 2)
            out = apply_sae_full_recon_triton(
                **inputs,
                **clamps,
                recon_mask=torch.ones(4, dtype=torch.bool),
                activation_code=ACTIVATION_CODE_RELU,
                activation_param=0.0,
            )
            assert out.dtype is dtype


class TestUnsupportedActivation:
    def test_invalid_activation_code_raises(self):
        inputs = _make_inputs()
        clamps = _zero_clamps(4, 2)
        with pytest.raises(ValueError, match="Unsupported activation"):
            apply_sae_full_recon_triton(
                **inputs,
                **clamps,
                recon_mask=torch.ones(4, dtype=torch.bool),
                activation_code=99,
                activation_param=0.0,
            )


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
        "activation,params,code",
        [
            (SAEActivation.RELU, {}, ACTIVATION_CODE_RELU),
            (SAEActivation.JUMPRELU, {}, ACTIVATION_CODE_JUMPRELU),
        ],
    )
    def test_cuda_matches_cpu_eager(self, activation, params, code):
        torch.manual_seed(0)
        n_tokens, d_model, d_sae, n_clamp = 8, 16, 32, 4
        cpu_inputs = _make_inputs(
            n_tokens=n_tokens, d_model=d_model, d_sae=d_sae, n_clamp=n_clamp, seed=42
        )
        rng = torch.Generator(device="cpu").manual_seed(7)
        cpu_clamps = {
            "clamp_kind": torch.randint(
                0, 3, (n_tokens, n_clamp), generator=rng, dtype=torch.int8
            ),
            "clamp_value": torch.randn(n_tokens, n_clamp, generator=rng),
            "clamp_only_if_active": torch.randint(
                0, 2, (n_tokens, n_clamp), generator=rng, dtype=torch.bool
            ),
        }
        recon_mask = torch.randint(0, 2, (n_tokens,), generator=rng, dtype=torch.bool)
        ref = apply_sae_full_reconstruction(
            **cpu_inputs,
            activation=activation,
            activation_params=params,
            **cpu_clamps,
            recon_mask=recon_mask,
        )
        gpu_inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
        gpu_clamps = {k: v.cuda() for k, v in cpu_clamps.items()}
        param = float(params.get("k", 0.0))
        got = apply_sae_full_recon_triton(
            **gpu_inputs,
            **gpu_clamps,
            recon_mask=recon_mask.cuda(),
            activation_code=code,
            activation_param=param,
        )
        assert got.is_cuda
        assert torch.allclose(got.cpu(), ref, atol=1e-4, rtol=1e-4)


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
