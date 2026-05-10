# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-4 Stage-5 real-weights end-to-end test for SAE full-reconstruction.

Sibling of ``test_sae_steering_real_weights.py`` for the
*replacement* variant.  Loads a Gemma Scope SAE for one site of
Gemma 2-2B, registers it with ``kind="sae_full_reconstruction"``,
attaches the full encoder / decoder / biases, and asserts that
submitting the same prompt with a non-trivial
``SamplingParams.sae_full_reconstruction_specs`` shifts the
generated tokens or cumulative logprob relative to baseline.

Two orthogonal checks the design doc calls out for full
reconstruction (vs delta):

1. **Pure reconstruction shifts the output.**  The replacement
   variant injects the SAE's reconstruction error even with no
   clamps active, so a request that opts into the module *without*
   modifying any feature should already diverge from a baseline
   request that doesn't enable the module at all.  This is the key
   semantic difference from the delta path, where the
   no-modification case is bit-identical to baseline.

2. **A high-magnitude clamp drives the output further.**  Same as
   the delta-path test: clamping a feature to a large absolute
   value produces a noticeable steering effect in the decoder
   direction, on top of the baseline reconstruction.

Skip semantics mirror the delta-path real-weights test: no CUDA /
not enough memory / model not registered in HF_EXAMPLE_MODELS /
download failure all surface as ``pytest.skip``.

The CPU loader unit tests in ``test_sae_loader.py`` cover the
read / subset contract for both the delta and full-reconstruction
loaders without any GPU.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import torch

from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEFullReconstructionSpec,
)
from vllm.entrypoints.openai.steering.sae_loader import (
    LoadedSAEModule,
    load_gemma_scope_sae_full_recon,
)

from ...registry import HF_EXAMPLE_MODELS

_MODEL = "google/gemma-2-2b"
_SAE_REPO = "google/gemma-scope-2b-pt-res"
_SAE_LAYER = 12
_SAE_HOOK = "post_mlp"
_SAE_RELATIVE_PATH = "layer_12/width_16k/average_l0_82/params.npz"
_SAE_MODULE_NAME = "gemma_scope_full_recon_layer12_post_mlp"
_NUM_CLAMPABLE_FEATURES = 4
_CLAMP_TARGET_VALUE = 50.0


def _skip_if_cuda_unavailable_or_below(min_memory_gib: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{_MODEL} real-weights SAE full-recon test requires CUDA.")
    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_memory_gib < min_memory_gib:
        pytest.skip(
            f"{_MODEL} real-weights SAE full-recon test requires a GPU with "
            f"at least {min_memory_gib:.0f} GiB of memory."
        )


def _maybe_skip_if_model_not_registered(model_id: str) -> None:
    """Skip if ``model_id`` isn't in the test-suite's HF model registry."""
    try:
        info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    except ValueError as exc:
        pytest.skip(
            f"{model_id} is not registered in HF_EXAMPLE_MODELS for this "
            f"test suite; add an entry to tests/models/registry.py to "
            f"enable the SAE full-reconstruction real-weights test.  ({exc})"
        )
        return
    info.check_available_online(on_fail="skip")


def _maybe_skip_model_access_failure(exc: Exception, label: str) -> None:
    if isinstance(exc, requests.exceptions.RequestException):
        pytest.skip(f"{label} skipped due to network error: {exc}")
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if (
            "gated repo" in msg
            or "connection error" in msg
            or "read timeout" in msg
            or "401" in msg
            or "403" in msg
        ):
            pytest.skip(f"{label} skipped due to access error: {exc}")
    if isinstance(exc, FileNotFoundError):
        pytest.skip(f"{label} skipped due to missing artifact: {exc}")


def _download_gemma_scope_npz() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        pytest.skip(
            f"huggingface_hub is required for the SAE full-reconstruction "
            f"real-weights test: {exc}"
        )
    local_path = hf_hub_download(
        repo_id=_SAE_REPO,
        filename=_SAE_RELATIVE_PATH,
        token=os.getenv("HF_TOKEN"),
    )
    return Path(local_path)


def _load_full_recon_module() -> LoadedSAEModule:
    """Download + parse a Gemma Scope NPZ for the full-reconstruction path.

    The full-recon loader returns the **complete** ``d_sae`` × ``d_model``
    encoder + decoder + biases, not a clampable-features subset — every
    feature participates in the reconstruction.  The clampable subset
    here is just the indices the per-request spec may modify.
    """
    npz_path = _download_gemma_scope_npz()
    feats = list(range(_NUM_CLAMPABLE_FEATURES))
    return load_gemma_scope_sae_full_recon(
        npz_path,
        layer_idx=_SAE_LAYER,
        hook_str=_SAE_HOOK,
        clampable_features=feats,
        weights_dtype=torch.bfloat16,
    )


def _payload_from_loaded(loaded: LoadedSAEModule) -> dict[str, Any]:
    """Marshal the manifest into the broadcast payload for the worker.

    Same shape as the delta path uses, but with
    ``kind="sae_full_reconstruction"``.
    """
    from vllm.entrypoints.openai.steering.registry import _sae_manifest_to_dict

    return {
        "kind": "sae_full_reconstruction",
        "sae_manifest": _sae_manifest_to_dict(loaded.manifest),
    }


def _gen_tokens_and_logprob(
    llm,
    prompt: str,
    sampling: SamplingParams,
) -> tuple[list[int], float | None]:
    result = llm.llm.generate([prompt], sampling)
    output = result[0].outputs[0]
    return list(output.token_ids), output.cumulative_logprob


def _build_clamped_spec(
    clampable_features: tuple[int, ...],
    *,
    target: float,
) -> SAEFullReconstructionSpec:
    entry = SAEClampEntry(
        feature_idx=int(clampable_features[0]),
        kind="absolute",
        value=float(target),
        only_if_active=False,
    )
    return SAEFullReconstructionSpec(
        module_name=_SAE_MODULE_NAME,
        clamps={_SAE_HOOK: {_SAE_LAYER: (entry,)}},
        phase="both",
    )


def _build_pure_recon_spec() -> SAEFullReconstructionSpec:
    """Spec with no clamps — pure reconstruction.

    The replacement variant runs the SAE forward and discards the
    original residual even when no features are modified, so this
    spec should still shift the output relative to baseline because
    the SAE has non-zero reconstruction error.  This is the principal
    semantic difference from the delta path.
    """
    return SAEFullReconstructionSpec(module_name=_SAE_MODULE_NAME, phase="both")


def test_full_reconstruction_pure_replacement_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """A pure-reconstruction spec must shift the output via reconstruction error.

    With no clamps active, the SAE replaces the residual with its
    reconstruction at every covered (layer, hook) site.  The
    reconstruction has non-zero error, so the post-replacement
    residual stream differs from the original — and the generation
    diverges from a baseline that doesn't enable the module.

    This is the key test that distinguishes full-reconstruction from
    delta: the delta path's no-modification case is bit-identical to
    baseline, the full-reconstruction path's is not.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)

    try:
        loaded = _load_full_recon_module()
    except Exception as exc:  # noqa: BLE001
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE full-recon")
        raise

    payload = _payload_from_loaded(loaded)

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "The fox jumped over the " * 8
        baseline_sampling = SamplingParams(max_tokens=12, temperature=0.0, logprobs=5)

        with vllm_runner(
            _MODEL,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_logprob(
                llm, prompt, baseline_sampling
            )
            assert llm.llm.reset_prefix_cache()

            llm.llm.collective_rpc(
                "register_steering_modules",
                kwargs={
                    "modules": {_SAE_MODULE_NAME: payload},
                    "replace": False,
                },
            )
            llm.llm.collective_rpc(
                "attach_sae_full_recon_weights",
                kwargs={
                    "module_name": _SAE_MODULE_NAME,
                    "weights": loaded.weights,
                },
            )
            assert llm.llm.reset_prefix_cache()

            pure_spec = _build_pure_recon_spec()
            recon_sampling = SamplingParams(
                max_tokens=12,
                temperature=0.0,
                logprobs=5,
                sae_full_reconstruction_specs=(pure_spec,),
            )
            recon_tokens, recon_logprob = _gen_tokens_and_logprob(
                llm, prompt, recon_sampling
            )

            assert recon_tokens != baseline_tokens or not (
                recon_logprob is not None
                and baseline_logprob is not None
                and math.isclose(
                    recon_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), (
                "Pure reconstruction (no clamps) should still shift the output: "
                "the SAE replaces the residual with its reconstruction, which "
                "has non-zero reconstruction error.  This is the semantic "
                "difference from the delta path."
            )

            # An unclamped baseline-style request after the recon one must
            # match the original baseline — the SAE delta is per-request,
            # not global.
            assert llm.llm.reset_prefix_cache()
            restored_tokens, _ = _gen_tokens_and_logprob(llm, prompt, baseline_sampling)
            assert restored_tokens == baseline_tokens, (
                "An unclamped (no-spec) request after a reconstruction one "
                "should match the original baseline (full-reconstruction is "
                "per-request, not global)."
            )


def test_full_reconstruction_clamp_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """A high-magnitude absolute clamp must change the output further than baseline.

    On top of the pure-reconstruction shift verified above, the
    explicit clamp drives a feature's activation to a large value
    before the decoder pass.  The output should diverge from a
    pure-reconstruction (no-clamp) request, demonstrating that the
    clamp itself contributes to the steering — not just the
    reconstruction-error baseline.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)

    try:
        loaded = _load_full_recon_module()
    except Exception as exc:  # noqa: BLE001
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE full-recon")
        raise

    payload = _payload_from_loaded(loaded)

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "The fox jumped over the " * 8
        sampling_kwargs = {"max_tokens": 12, "temperature": 0.0, "logprobs": 5}

        with vllm_runner(
            _MODEL,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            llm.llm.collective_rpc(
                "register_steering_modules",
                kwargs={
                    "modules": {_SAE_MODULE_NAME: payload},
                    "replace": False,
                },
            )
            llm.llm.collective_rpc(
                "attach_sae_full_recon_weights",
                kwargs={
                    "module_name": _SAE_MODULE_NAME,
                    "weights": loaded.weights,
                },
            )

            # Pure reconstruction (no clamps) — the "baseline with
            # reconstruction" against which the clamp's incremental
            # effect is measured.
            pure_sampling = SamplingParams(
                **sampling_kwargs,
                sae_full_reconstruction_specs=(_build_pure_recon_spec(),),
            )
            pure_tokens, pure_logprob = _gen_tokens_and_logprob(
                llm, prompt, pure_sampling
            )
            assert llm.llm.reset_prefix_cache()

            # With a high-magnitude clamp.
            clamped_spec = _build_clamped_spec(
                loaded.manifest.clampable_features, target=_CLAMP_TARGET_VALUE
            )
            clamped_sampling = SamplingParams(
                **sampling_kwargs,
                sae_full_reconstruction_specs=(clamped_spec,),
            )
            clamped_tokens, clamped_logprob = _gen_tokens_and_logprob(
                llm, prompt, clamped_sampling
            )

            # The explicit clamp must do *something* on top of the
            # pure-reconstruction baseline.
            assert clamped_tokens != pure_tokens or not (
                clamped_logprob is not None
                and pure_logprob is not None
                and math.isclose(
                    clamped_logprob,
                    pure_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), (
                "An absolute clamp on a real Gemma Scope feature should "
                "change the output beyond what pure reconstruction alone "
                "produces."
            )


def test_full_reconstruction_clamp_magnitude_scales_with_target_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """Larger absolute clamp magnitude should produce more divergence from pure-recon.

    Tests monotonicity: a target=50 clamp's logprob delta from the
    pure-reconstruction baseline should be at least as large as a
    target=5 clamp's delta.  Same shape as the delta-path test, with
    the pure-reconstruction request as the reference (rather than
    a no-spec baseline) so we measure only the clamp's incremental
    effect.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)

    try:
        loaded = _load_full_recon_module()
    except Exception as exc:  # noqa: BLE001
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE full-recon")
        raise

    payload = _payload_from_loaded(loaded)

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "The fox jumped over the " * 8
        sampling_kwargs = {"max_tokens": 12, "temperature": 0.0, "logprobs": 5}

        with vllm_runner(
            _MODEL,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            llm.llm.collective_rpc(
                "register_steering_modules",
                kwargs={
                    "modules": {_SAE_MODULE_NAME: payload},
                    "replace": False,
                },
            )
            llm.llm.collective_rpc(
                "attach_sae_full_recon_weights",
                kwargs={
                    "module_name": _SAE_MODULE_NAME,
                    "weights": loaded.weights,
                },
            )

            pure_sampling = SamplingParams(
                **sampling_kwargs,
                sae_full_reconstruction_specs=(_build_pure_recon_spec(),),
            )
            _, pure_logprob = _gen_tokens_and_logprob(llm, prompt, pure_sampling)
            assert llm.llm.reset_prefix_cache()

            small_spec = _build_clamped_spec(
                loaded.manifest.clampable_features, target=5.0
            )
            small_sampling = SamplingParams(
                **sampling_kwargs,
                sae_full_reconstruction_specs=(small_spec,),
            )
            _, small_logprob = _gen_tokens_and_logprob(llm, prompt, small_sampling)
            assert llm.llm.reset_prefix_cache()

            large_spec = _build_clamped_spec(
                loaded.manifest.clampable_features, target=_CLAMP_TARGET_VALUE
            )
            large_sampling = SamplingParams(
                **sampling_kwargs,
                sae_full_reconstruction_specs=(large_spec,),
            )
            _, large_logprob = _gen_tokens_and_logprob(llm, prompt, large_sampling)

            if pure_logprob is None or small_logprob is None or large_logprob is None:
                pytest.skip(
                    "Cumulative logprob unavailable — skipping monotonicity check."
                )
            small_delta = abs(small_logprob - pure_logprob)
            large_delta = abs(large_logprob - pure_logprob)
            assert large_delta + 1e-3 >= small_delta, (
                "Larger clamp magnitude should not produce a smaller divergence "
                f"from the pure-reconstruction baseline: target=5 → "
                f"Δ={small_delta}, target={_CLAMP_TARGET_VALUE} → Δ={large_delta}."
            )
