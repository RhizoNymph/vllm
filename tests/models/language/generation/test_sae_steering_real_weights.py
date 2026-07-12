# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-3 real-weights end-to-end test for SAE feature-surgery steering.

Loads a Gemma Scope SAE for one site of Gemma 2-2B, registers it with
the worker through ``register_steering_modules`` + ``attach_sae_weights``,
submits a generation request whose ``SamplingParams.sae_clamp_specs``
clamps one feature to a high target value, and asserts the output
shifts relative to a baseline generation with no clamp.  Per the
design doc, the goal is **qualitative** — the test verifies that
steering *happens* on a real checkpoint, not that the output text
acquires a specific topical bias.

Skip semantics (matching the existing ``*_real_weights`` tests in
``test_steering.py``):

* No CUDA, or a GPU with too little memory → skip.
* Gemma 2 / Gemma Scope download failures
  (gated repo, network error, etc.) → skip.

The CPU loader unit tests in
``tests/entrypoints/openai/test_sae_loader.py`` cover the read /
subset / merge contract on synthetic on-disk fixtures and do not
require a GPU.

Per-feature JumpReLU thresholds: Gemma Scope ships a per-feature
``threshold`` array, while the current SAE kernel takes a scalar
``threshold`` parameter (kept simple so each ``(activation, n_clamp,
d_model)`` site JIT-specialises on a single constexpr).  The loader
falls back to the median per-feature threshold over the clampable
subset (see :func:`load_gemma_scope_sae` for the full rationale).
For absolute clamps with a high target value, the steering effect
is dominated by ``target`` rather than the live activation ``f``
(``delta = target - f``), so the qualitative output-shift assertion
in this test is robust to the threshold simplification.  Tracked as
a follow-up.
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
from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec
from vllm.entrypoints.openai.steering.registry import (
    pack_sae_weights_for_broadcast,
)
from vllm.entrypoints.openai.steering.sae_loader import (
    LoadedSAEModule,
    load_gemma_scope_sae,
)

from ...registry import HF_EXAMPLE_MODELS

# Gemma 2-2B is the smallest model that Gemma Scope was trained for.
# 26 layers, hidden size 2304.  Pick a mid-stack layer; layer 12 has a
# 16k-width SAE released at average_l0=82.  Both numbers are part of
# the upstream release and stable; if Google ever revises the path
# layout the test skips on the download error.
_MODEL = "google/gemma-2-2b"
_SAE_REPO = "google/gemma-scope-2b-pt-res"
_SAE_LAYER = 12
_SAE_HOOK = "post_block"
_SAE_RELATIVE_PATH = "layer_12/width_16k/average_l0_82/params.npz"
_SAE_MODULE_NAME = "gemma_scope_layer12_post_block"
_NUM_CLAMPABLE_FEATURES = 4
_CLAMP_TARGET_VALUE = 50.0


def _skip_if_cuda_unavailable_or_below(min_memory_gib: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{_MODEL} real-weights SAE test requires CUDA.")
    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_memory_gib < min_memory_gib:
        pytest.skip(
            f"{_MODEL} real-weights SAE test requires a GPU with at least "
            f"{min_memory_gib:.0f} GiB of memory."
        )


def _maybe_skip_if_model_not_registered(model_id: str) -> None:
    """Skip if ``model_id`` isn't in the test-suite's HF model registry.

    The registry is the gate that ``vllm_runner`` consults to decide
    whether to materialise the model.  ``google/gemma-2-2b`` may or
    may not be registered depending on which branch the test runs on;
    rather than mutate ``tests/models/registry.py`` (a registry edit
    that this PR shouldn't own), we surface the missing-entry case as
    a skip so the test stays opt-in.
    """
    try:
        info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    except ValueError as exc:
        pytest.skip(
            f"{model_id} is not registered in HF_EXAMPLE_MODELS for this "
            f"test suite; add an entry to tests/models/registry.py to "
            f"enable the SAE real-weights test.  ({exc})"
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
    """Download one Gemma Scope NPZ from HuggingFace.

    Returns a ``Path`` to the local NPZ file.  Caller is responsible
    for handling download failures via :func:`_maybe_skip_model_access_failure`.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        pytest.skip(f"huggingface_hub is required for the SAE real-weights test: {exc}")
    local_path = hf_hub_download(
        repo_id=_SAE_REPO,
        filename=_SAE_RELATIVE_PATH,
        token=os.getenv("HF_TOKEN"),
    )
    return Path(local_path)


def _load_sae_module() -> LoadedSAEModule:
    """Download + parse one Gemma Scope SAE site, subsetting to a few features.

    The test only needs a handful of clampable features; subsetting
    keeps the buffer footprint tiny (``n_clamp = 4`` here) so the
    fused kernel stays in its happy path and the kernel-tile sizes
    match the warmup the worker mixin issues.
    """
    npz_path = _download_gemma_scope_npz()
    feats = list(range(_NUM_CLAMPABLE_FEATURES))
    return load_gemma_scope_sae(
        npz_path,
        layer_idx=_SAE_LAYER,
        hook_str=_SAE_HOOK,
        clampable_features=feats,
        weights_dtype=torch.bfloat16,
    )


def _sae_payload_from_loaded(loaded: LoadedSAEModule) -> dict[str, Any]:
    """Marshal a :class:`LoadedSAEModule` into the broadcast payload shape.

    ``register_steering_modules`` on the worker mixin consumes a
    JSON-safe dict; the manifest is round-tripped through
    :func:`_sae_manifest_to_dict` so the on-the-wire representation
    matches what the API server would send.
    """
    from vllm.entrypoints.openai.steering.registry import _sae_manifest_to_dict

    return {
        "kind": "sae_delta",
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


def _build_clamp_spec(
    clampable_features: tuple[int, ...],
    *,
    target: float,
) -> SAEClampSpec:
    """Construct a single-feature absolute clamp spec for the test.

    Picks the first clampable feature; absolute clamp with
    ``target=50`` produces a delta dominated by the target value, so
    the test is robust to the JumpReLU threshold simplification
    discussed at module top.
    """
    entry = SAEClampEntry(
        feature_idx=int(clampable_features[0]),
        kind="absolute",
        value=float(target),
        only_if_active=False,
    )
    return SAEClampSpec(
        module_name=_SAE_MODULE_NAME,
        clamps={_SAE_HOOK: {_SAE_LAYER: (entry,)}},
        phase="both",
    )


def test_gemma_scope_sae_clamp_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """A real Gemma Scope SAE clamp must change the generated tokens.

    Loads Gemma 2-2B with steering enabled, registers + attaches a
    single-(layer, hook) Gemma Scope SAE, generates a baseline
    completion, then submits the same prompt with a high-magnitude
    absolute clamp on one feature.  The test asserts that *something*
    in the output diverges (token sequence or cumulative logprob);
    the qualitative direction of the shift (golden-gate-style topical
    bias) is feature-dependent and not asserted.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)

    try:
        loaded = _load_sae_module()
    except Exception as exc:  # noqa: BLE001 - intentionally broad to skip on any
        # download/network/parse failure.
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE")
        raise

    payload = _sae_payload_from_loaded(loaded)

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
            # Baseline generation with no SAE clamp active.
            baseline_tokens, baseline_logprob = _gen_tokens_and_logprob(
                llm, prompt, baseline_sampling
            )
            assert llm.llm.reset_prefix_cache()

            # Register the SAE module and attach weights.  The dispatch
            # path runs once per worker rank; raw tensors do not survive
            # collective_rpc's msgpack hop to an out-of-process engine
            # core, so weights cross in the packed wire form.
            llm.llm.collective_rpc(
                "register_steering_modules",
                kwargs={
                    "modules": {_SAE_MODULE_NAME: payload},
                    "replace": False,
                },
            )
            llm.llm.collective_rpc(
                "attach_sae_weights",
                kwargs={
                    "module_name": _SAE_MODULE_NAME,
                    "weights": pack_sae_weights_for_broadcast(loaded.weights),
                },
            )
            assert llm.llm.reset_prefix_cache()

            clamp_spec = _build_clamp_spec(
                loaded.manifest.clampable_features, target=_CLAMP_TARGET_VALUE
            )
            steered_sampling = SamplingParams(
                max_tokens=12,
                temperature=0.0,
                logprobs=5,
                sae_clamp_specs=(clamp_spec,),
            )
            steered_tokens, steered_logprob = _gen_tokens_and_logprob(
                llm, prompt, steered_sampling
            )

            # Either the token sequence diverges or the logprob shifts;
            # the steering effect must be visible somewhere in the
            # output for a high-magnitude absolute clamp.
            assert steered_tokens != baseline_tokens or not (
                steered_logprob is not None
                and baseline_logprob is not None
                and math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ), (
                "SAE clamp on a real Gemma Scope feature should change "
                "the generated tokens or cumulative logprob."
            )

            # Submit the prompt again *without* the clamp spec — the
            # buffer state is unchanged (rows still allocated for the
            # request that just completed), but no per-token routing
            # to a clamp row should mean the output matches baseline.
            assert llm.llm.reset_prefix_cache()
            restored_tokens, restored_logprob = _gen_tokens_and_logprob(
                llm, prompt, baseline_sampling
            )
            assert restored_tokens == baseline_tokens, (
                "An unclamped request after a clamped one should match "
                "the original baseline (the SAE delta is per-request, "
                "not global)."
            )
            if restored_logprob is not None and baseline_logprob is not None:
                assert math.isclose(
                    restored_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ), (
                    "Restored cumulative logprob should match baseline "
                    "for an unclamped request."
                )


def test_gemma_scope_sae_clamp_magnitude_scales_with_target_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """A larger absolute clamp value should produce more divergence.

    Asserts that the total absolute logprob delta between baseline and
    steered is monotonic-ish in the clamp target: a clamp of 5 should
    produce a smaller (or equal) shift than a clamp of 50.  Robust
    against fluky tokenisations because we compare cumulative
    logprob rather than tokens directly; if both clamps happen to
    produce the *same* token sequence, the assertion degrades to "the
    larger clamp's logprob is at least as far from baseline as the
    smaller clamp's".
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)

    try:
        loaded = _load_sae_module()
    except Exception as exc:  # noqa: BLE001
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE")
        raise

    payload = _sae_payload_from_loaded(loaded)

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
            llm.llm.collective_rpc(
                "register_steering_modules",
                kwargs={
                    "modules": {_SAE_MODULE_NAME: payload},
                    "replace": False,
                },
            )
            llm.llm.collective_rpc(
                "attach_sae_weights",
                kwargs={
                    "module_name": _SAE_MODULE_NAME,
                    "weights": pack_sae_weights_for_broadcast(loaded.weights),
                },
            )

            _, baseline_logprob = _gen_tokens_and_logprob(
                llm, prompt, baseline_sampling
            )
            assert llm.llm.reset_prefix_cache()

            small_spec = _build_clamp_spec(
                loaded.manifest.clampable_features, target=5.0
            )
            small_sampling = SamplingParams(
                max_tokens=12,
                temperature=0.0,
                logprobs=5,
                sae_clamp_specs=(small_spec,),
            )
            _, small_logprob = _gen_tokens_and_logprob(llm, prompt, small_sampling)
            assert llm.llm.reset_prefix_cache()

            large_spec = _build_clamp_spec(
                loaded.manifest.clampable_features, target=_CLAMP_TARGET_VALUE
            )
            large_sampling = SamplingParams(
                max_tokens=12,
                temperature=0.0,
                logprobs=5,
                sae_clamp_specs=(large_spec,),
            )
            _, large_logprob = _gen_tokens_and_logprob(llm, prompt, large_sampling)

            if (
                baseline_logprob is None
                or small_logprob is None
                or large_logprob is None
            ):
                pytest.skip(
                    "Cumulative logprob unavailable — skipping monotonicity "
                    "check (would require greedy-token comparison instead)."
                )
            small_delta = abs(small_logprob - baseline_logprob)
            large_delta = abs(large_logprob - baseline_logprob)
            # Use a small slack so floating-point noise on near-zero
            # deltas doesn't trip the assertion.
            assert large_delta + 1e-3 >= small_delta, (
                f"Larger SAE clamp magnitude should not produce a smaller "
                f"divergence from baseline: target=5 → Δ={small_delta}, "
                f"target={_CLAMP_TARGET_VALUE} → Δ={large_delta}."
            )
