# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weights breadth tests for SAE steering beyond the delta happy path.

Extends ``test_sae_steering_real_weights.py`` (same model, SAE site,
harness, and skip stack) to cover surfaces that previously had only
mocked or kernel-level coverage:

* fp8 (``storage_dtype="fp8_e4m3"``) weight-table storage on a real
  Gemma Scope SAE, checked for steering-effect consistency against the
  bf16 module on the same site.
* The global SAE clamp tier (``set_sae_global_clamps`` /
  ``clear_sae_global_clamps`` worker RPCs — rows 1/2), shifting a
  request that carries no per-request specs.
* ``gated=True`` specs with no monitor active: the row gate resets to
  1.0 every step, so a gated clamp must match its ungated twin
  token-for-token.  (Monitor-driven ON/OFF gating on a real model
  still requires the dynamic-steering consumer harness and is not
  covered here.)
* Prefix-cache isolation between clamped and unclamped requests with
  no manual ``reset_prefix_cache`` calls between them.
"""

from __future__ import annotations

import math

import pytest
from vllm import SamplingParams
from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec
from vllm.entrypoints.openai.steering.registry import (
    pack_sae_weights_for_broadcast,
)

from .test_sae_steering_real_weights import (
    _CLAMP_TARGET_VALUE,
    _MODEL,
    _SAE_HOOK,
    _SAE_LAYER,
    _SAE_MODULE_NAME,
    _gen_tokens_and_logprob,
    _load_sae_module,
    _maybe_skip_if_model_not_registered,
    _maybe_skip_model_access_failure,
    _sae_payload_from_loaded,
    _skip_if_cuda_unavailable_or_below,
)

_PROMPT = "The fox jumped over the " * 8
_RUNNER_KWARGS = dict(
    max_model_len=256,
    enable_steering=True,
    max_steering_configs=4,
    enable_prefix_caching=True,
    enforce_eager=True,
    # The model + tiny KV cache fit comfortably; a low target keeps the
    # test runnable on a GPU shared with a desktop session or sibling jobs.
    gpu_memory_utilization=0.55,
)


def _clamp_spec_for(
    module_name: str,
    feature_idx: int,
    *,
    target: float = _CLAMP_TARGET_VALUE,
    gated: bool = False,
    phase: str = "both",
) -> SAEClampSpec:
    entry = SAEClampEntry(
        feature_idx=feature_idx,
        kind="absolute",
        value=float(target),
        only_if_active=False,
    )
    return SAEClampSpec(
        module_name=module_name,
        clamps={_SAE_HOOK: {_SAE_LAYER: (entry,)}},
        phase=phase,
        gated=gated,
    )


def _sampling(clamp_specs: tuple = ()) -> SamplingParams:
    kwargs: dict = dict(max_tokens=12, temperature=0.0, logprobs=5)
    if clamp_specs:
        kwargs["sae_clamp_specs"] = clamp_specs
    return SamplingParams(**kwargs)


def _load_module_or_skip():
    try:
        return _load_sae_module()
    except Exception as exc:  # noqa: BLE001 - broad to skip on any download failure
        _maybe_skip_model_access_failure(exc, "Gemma Scope SAE")
        raise


def _register_and_attach(llm, name: str, payload: dict, weights) -> None:
    llm.llm.collective_rpc(
        "register_steering_modules",
        kwargs={"modules": {name: payload}, "replace": False},
    )
    llm.llm.collective_rpc(
        "attach_sae_weights",
        kwargs={
            "module_name": name,
            "weights": pack_sae_weights_for_broadcast(weights),
        },
    )


def test_fp8_storage_clamp_consistent_with_bf16_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """fp8 weight tables must steer, and steer consistently with bf16.

    Registers the same Gemma Scope site twice — once with the default
    (bf16) storage and once with ``storage_dtype="fp8_e4m3"`` (the
    worker quantizes the broadcast bf16 tensors at attach time).  The
    same absolute clamp routed through either module must move the
    output, and the fp8 module's logprob shift must land within a
    loose band of the bf16 shift — catching both a silent no-op
    (zeroed or mis-scaled fp8 tables) and a wildly mis-quantized
    decoder direction, without asserting bit-level parity the fp8
    round-trip cannot provide.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)
    loaded = _load_module_or_skip()

    bf16_payload = _sae_payload_from_loaded(loaded)
    fp8_payload = _sae_payload_from_loaded(loaded)
    fp8_payload["sae_manifest"]["storage_dtype"] = "fp8_e4m3"
    fp8_name = f"{_SAE_MODULE_NAME}_fp8"
    feature = int(loaded.manifest.clampable_features[0])

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(_MODEL, **_RUNNER_KWARGS) as llm:
            _register_and_attach(llm, _SAE_MODULE_NAME, bf16_payload, loaded.weights)
            _register_and_attach(llm, fp8_name, fp8_payload, loaded.weights)
            assert llm.llm.reset_prefix_cache()

            _, baseline_logprob = _gen_tokens_and_logprob(llm, _PROMPT, _sampling())
            assert llm.llm.reset_prefix_cache()

            bf16_spec = _clamp_spec_for(_SAE_MODULE_NAME, feature)
            _, bf16_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling((bf16_spec,))
            )
            assert llm.llm.reset_prefix_cache()

            fp8_spec = _clamp_spec_for(fp8_name, feature)
            _, fp8_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling((fp8_spec,))
            )

            if baseline_logprob is None or bf16_logprob is None or fp8_logprob is None:
                pytest.skip("Cumulative logprob unavailable for delta comparison.")
            bf16_delta = abs(bf16_logprob - baseline_logprob)
            fp8_delta = abs(fp8_logprob - baseline_logprob)
            assert bf16_delta > 1e-3, (
                "bf16 SAE clamp produced no measurable shift — the "
                "consistency band below would be vacuous."
            )
            assert fp8_delta > 1e-3, (
                "fp8 SAE clamp produced no measurable shift; fp8 weight "
                "tables are likely zero-filled or mis-scaled at attach."
            )
            assert 0.25 * bf16_delta <= fp8_delta <= 4.0 * bf16_delta, (
                f"fp8 clamp shift (Δ={fp8_delta}) is not within a 4x band "
                f"of the bf16 shift (Δ={bf16_delta}); quantized decoder "
                "directions are inconsistent with the bf16 tables."
            )


def test_global_sae_clamp_tier_shifts_specless_requests_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """Global SAE clamps (rows 1/2) must steer requests with no specs.

    Real-model analogue of ``test_steering.py``'s
    ``test_global_prefill_steering_via_worker_api`` for the SAE tier:
    ``set_sae_global_clamps`` shifts a spec-less generation, and
    ``clear_sae_global_clamps`` restores the exact baseline.  Prefix
    cache is reset after every set/clear, as the API router mandates.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)
    loaded = _load_module_or_skip()
    payload = _sae_payload_from_loaded(loaded)
    feature = int(loaded.manifest.clampable_features[0])
    global_specs = [
        {
            "module_name": _SAE_MODULE_NAME,
            "clamps": {
                _SAE_HOOK: {
                    str(_SAE_LAYER): [
                        {
                            "feature_idx": feature,
                            "kind": "absolute",
                            "value": _CLAMP_TARGET_VALUE,
                        }
                    ]
                }
            },
        }
    ]

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(_MODEL, **_RUNNER_KWARGS) as llm:
            _register_and_attach(llm, _SAE_MODULE_NAME, payload, loaded.weights)
            assert llm.llm.reset_prefix_cache()

            baseline_tokens, baseline_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling()
            )

            llm.llm.collective_rpc(
                "set_sae_global_clamps",
                kwargs={
                    "prefill_specs_raw": global_specs,
                    "decode_specs_raw": global_specs,
                },
            )
            assert llm.llm.reset_prefix_cache()
            global_tokens, global_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling()
            )
            assert global_tokens != baseline_tokens or not (
                global_logprob is not None
                and baseline_logprob is not None
                and math.isclose(
                    global_logprob, baseline_logprob, rel_tol=0.0, abs_tol=1e-6
                )
            ), (
                "Global SAE clamps should shift a request that carries no "
                "per-request specs (rows 1/2 gather on every token)."
            )

            llm.llm.collective_rpc("clear_sae_global_clamps")
            assert llm.llm.reset_prefix_cache()
            restored_tokens, restored_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling()
            )
            assert restored_tokens == baseline_tokens, (
                "Clearing global SAE clamps should restore the exact "
                "baseline generation."
            )
            if restored_logprob is not None and baseline_logprob is not None:
                assert math.isclose(
                    restored_logprob, baseline_logprob, rel_tol=0.0, abs_tol=1e-6
                )


def test_gated_spec_without_monitor_matches_ungated_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """``gated=True`` with no monitor active must equal the ungated clamp.

    The shared ``steering_row_gate`` resets to 1.0 at the top of every
    step and is only reduced by an active row-gating monitor, so a
    gated spec on an engine with no monitor configured must reproduce
    its ungated twin token-for-token.  Catches a gate-participation
    buffer that silently zeroes (or fails to apply) the clamp effect.
    Monitor-driven suppression on a real model is deliberately out of
    scope — it needs the dynamic-steering consumer harness.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)
    loaded = _load_module_or_skip()
    payload = _sae_payload_from_loaded(loaded)
    feature = int(loaded.manifest.clampable_features[0])

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(_MODEL, **_RUNNER_KWARGS) as llm:
            _register_and_attach(llm, _SAE_MODULE_NAME, payload, loaded.weights)
            assert llm.llm.reset_prefix_cache()

            baseline_tokens, _ = _gen_tokens_and_logprob(llm, _PROMPT, _sampling())
            assert llm.llm.reset_prefix_cache()

            ungated = _clamp_spec_for(_SAE_MODULE_NAME, feature)
            ungated_tokens, ungated_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling((ungated,))
            )
            assert llm.llm.reset_prefix_cache()

            gated = _clamp_spec_for(_SAE_MODULE_NAME, feature, gated=True)
            gated_tokens, gated_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling((gated,))
            )

            assert ungated_tokens != baseline_tokens, (
                "Ungated clamp produced no shift — the gated-equivalence "
                "check below would be vacuous."
            )
            assert gated_tokens == ungated_tokens, (
                "A gated clamp with no monitor active must match the "
                "ungated clamp token-for-token (row gate resets to 1.0)."
            )
            if gated_logprob is not None and ungated_logprob is not None:
                assert math.isclose(
                    gated_logprob, ungated_logprob, rel_tol=0.0, abs_tol=1e-4
                )


def test_prefix_cache_isolation_with_sae_clamps_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """SAE specs must key the prefix cache — no resets between requests.

    All comparisons are between *cache-hit* runs: a cold prefill and a
    cache-hit continuation are not bitwise-comparable on real bf16
    weights (the recomputed partial-block tail can flip a near-tie
    greedy token), so a warm-up request populates the cache and every
    asserted request after it takes the same full-hit path.  The
    clamped request must diverge from the unsteered cache-hit
    reference even though an unsteered prefill for the identical
    prompt sits in the cache (its spec hash must miss that entry), and
    a final unclamped request must reproduce the reference exactly
    even though the clamped run cached its own steered blocks.
    """
    _skip_if_cuda_unavailable_or_below(min_memory_gib=10.0)
    _maybe_skip_if_model_not_registered(_MODEL)
    loaded = _load_module_or_skip()
    payload = _sae_payload_from_loaded(loaded)
    feature = int(loaded.manifest.clampable_features[0])

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(_MODEL, **_RUNNER_KWARGS) as llm:
            _register_and_attach(llm, _SAE_MODULE_NAME, payload, loaded.weights)
            assert llm.llm.reset_prefix_cache()

            # Warm-up: populates the unsteered cache entry; not asserted.
            _gen_tokens_and_logprob(llm, _PROMPT, _sampling())
            # Cache-hit reference for every comparison below.
            reference_tokens, reference_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling()
            )

            clamp = _clamp_spec_for(_SAE_MODULE_NAME, feature)
            clamped_tokens, clamped_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling((clamp,))
            )
            assert clamped_tokens != reference_tokens or not (
                clamped_logprob is not None
                and reference_logprob is not None
                and math.isclose(
                    clamped_logprob, reference_logprob, rel_tol=0.0, abs_tol=1e-6
                )
            ), (
                "Clamped request reproduced the unsteered reference exactly "
                "— its steered prefill likely reused the unsteered cached KV."
            )

            restored_tokens, restored_logprob = _gen_tokens_and_logprob(
                llm, _PROMPT, _sampling()
            )
            assert restored_tokens == reference_tokens, (
                "Unclamped request after a clamped one diverged from the "
                "unsteered cache-hit reference — steered KV blocks leaked "
                "into the unsteered cache key (prefix-cache poisoning)."
            )
            if restored_logprob is not None and reference_logprob is not None:
                assert math.isclose(
                    restored_logprob, reference_logprob, rel_tol=0.0, abs_tol=1e-6
                )
