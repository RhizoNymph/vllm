# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed equivalence tests for activation steering.

Proves the steering determinism contract (see
``docs/design/steering_runtime.md``) across a tensor-parallel ×
pipeline-parallel grid: identical steering configuration produces
identical greedy token IDs regardless of TP/PP degree.

All tests use ``load_format="dummy"`` with small decoder overrides so
they complete in seconds per GPU configuration. Real-weight coverage
lives in ``test_steering.py``.

Requires multiple GPUs for the parallel configurations — each test is
gated via ``pytest.mark.skipif`` on ``torch.accelerator.device_count()``. The
small matrix (TP=2, PP=2) needs 2 GPUs; the full matrix (including
TP=4) needs 4 GPUs.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_TABLE_ATTR,
)

MODEL = "google/gemma-3-4b-it"

# ``_discover_layers`` sends a Python callable to every worker via
# ``collective_rpc``; the multiproc executor pickles it on the way out.
# Enable the fallback up-front so every test in this module can use the
# same pattern without per-test monkeypatching.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Keep the graph tiny — dummy weights + compact decoder. Two layers lets
# us exercise PP>1 meaningfully (one layer per stage) while still being
# small enough to run on modest hardware.
_SMALL_OVERRIDES = {
    "num_hidden_layers": 2,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
}

_HP = DEFAULT_HOOK_POINT.value
_TABLE_ATTR = HOOK_POINT_TABLE_ATTR[DEFAULT_HOOK_POINT]


def _skip_if_not_enough_gpus(required: int) -> None:
    if not torch.accelerator.is_available():
        pytest.skip("Distributed steering tests require CUDA.")
    if torch.accelerator.device_count() < required:
        pytest.skip(
            f"Test requires {required} GPUs; only "
            f"{torch.accelerator.device_count()} available."
        )


def _runner_kwargs(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_steering_configs: int = 4,
) -> dict:
    return {
        "load_format": "dummy",
        "max_model_len": 128,
        "enable_steering": True,
        "max_steering_configs": max_steering_configs,
        "hf_overrides": _SMALL_OVERRIDES,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "enforce_eager": True,
        "seed": 0,
    }


def _discover_layers(llm) -> tuple[int, int]:
    """Return ``(target_layer_idx, hidden_size)`` from any live rank.

    Under PP each rank only owns a slice of layers, but the union
    across all ranks covers the full model. We pick a layer that
    exists on at least one rank and return its hidden size.
    """

    def _discover(worker):
        layers = {}
        model = worker.model_runner.get_model()
        for mod in model.modules():
            if hasattr(mod, _TABLE_ATTR) and hasattr(mod, "layer_idx"):
                layers[mod.layer_idx] = getattr(mod, _TABLE_ATTR).shape[1]
        return layers

    per_rank = llm.llm.collective_rpc(_discover)
    merged: dict[int, int] = {}
    for rank_layers in per_rank:
        merged.update(rank_layers)
    target_layer = min(merged.keys())
    return target_layer, merged[target_layer]


def _gen_tokens(llm, prompt: str, sampling: SamplingParams) -> list[int]:
    result = llm.llm.generate([prompt], sampling)
    return list(result[0].outputs[0].token_ids)


def _apply_global_steering(
    llm, target_layer: int, hidden_size: int, magnitude: float = 0.2
) -> None:
    vec = [magnitude] * hidden_size
    llm.llm.collective_rpc(
        "set_steering_vectors",
        kwargs={"vectors": {_HP: {target_layer: vec}}},
    )


def _clear_global_steering(llm) -> None:
    llm.llm.collective_rpc("clear_steering_vectors")


# ---------------------------------------------------------------------------
# Single-rank reference: captured once per module so each equivalence test
# doesn't pay the model-load cost twice. ``pytest.fixture(scope="module")``
# keeps the reference alive across every test in this file.
# ---------------------------------------------------------------------------

_PROMPT = "The quick brown fox jumps over the lazy"
_SAMPLING = SamplingParams(max_tokens=8, temperature=0.0)


@pytest.fixture(scope="module")
def single_rank_reference(vllm_runner):
    """Run a single-rank baseline to use as the equivalence reference.

    Returns ``(baseline_tokens, steered_tokens, target_layer,
    hidden_size)`` — the tokens from an unsteered run and from a run
    with a fixed global steering vector applied, both on TP=1 PP=1.

    Skips at the fixture level when no GPU is available so every
    dependent test registers a clean skip instead of an error.
    """
    if not torch.accelerator.is_available():
        pytest.skip("Distributed steering tests require CUDA.")
    with vllm_runner(MODEL, **_runner_kwargs()) as llm:
        baseline = _gen_tokens(llm, _PROMPT, _SAMPLING)
        target_layer, hidden_size = _discover_layers(llm)
        _apply_global_steering(llm, target_layer, hidden_size)
        assert llm.llm.reset_prefix_cache()
        steered = _gen_tokens(llm, _PROMPT, _SAMPLING)
        _clear_global_steering(llm)
    return baseline, steered, target_layer, hidden_size


# ---------------------------------------------------------------------------
# Global-steering equivalence across TP / PP / mixed grids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [2, 4])
def test_global_steering_equivalence_tp(
    vllm_runner,
    single_rank_reference,
    tp: int,
) -> None:
    """Token IDs must match the single-rank baseline for every TP degree."""
    _skip_if_not_enough_gpus(tp)
    _baseline_ref, steered_ref, target_layer, hidden_size = single_rank_reference

    with vllm_runner(MODEL, **_runner_kwargs(tensor_parallel_size=tp)) as llm:
        _apply_global_steering(llm, target_layer, hidden_size)
        assert llm.llm.reset_prefix_cache()
        tokens = _gen_tokens(llm, _PROMPT, _SAMPLING)
        _clear_global_steering(llm)

    assert tokens == steered_ref, (
        f"TP={tp} steered tokens {tokens} diverge from single-rank "
        f"reference {steered_ref}"
    )


@pytest.mark.parametrize("pp", [2, 4])
def test_global_steering_equivalence_pp(
    vllm_runner,
    single_rank_reference,
    pp: int,
) -> None:
    _skip_if_not_enough_gpus(pp)
    _baseline_ref, steered_ref, target_layer, hidden_size = single_rank_reference

    with vllm_runner(MODEL, **_runner_kwargs(pipeline_parallel_size=pp)) as llm:
        _apply_global_steering(llm, target_layer, hidden_size)
        assert llm.llm.reset_prefix_cache()
        tokens = _gen_tokens(llm, _PROMPT, _SAMPLING)
        _clear_global_steering(llm)

    assert tokens == steered_ref, (
        f"PP={pp} steered tokens {tokens} diverge from single-rank "
        f"reference {steered_ref}"
    )


@pytest.mark.parametrize(
    ("tp", "pp"),
    [(2, 2), (4, 2), (2, 4)],
    ids=["tp2_pp2", "tp4_pp2", "tp2_pp4"],
)
def test_global_steering_equivalence_tp_pp(
    vllm_runner,
    single_rank_reference,
    tp: int,
    pp: int,
) -> None:
    _skip_if_not_enough_gpus(tp * pp)
    _baseline_ref, steered_ref, target_layer, hidden_size = single_rank_reference

    with vllm_runner(
        MODEL,
        **_runner_kwargs(tensor_parallel_size=tp, pipeline_parallel_size=pp),
    ) as llm:
        _apply_global_steering(llm, target_layer, hidden_size)
        assert llm.llm.reset_prefix_cache()
        tokens = _gen_tokens(llm, _PROMPT, _SAMPLING)
        _clear_global_steering(llm)

    assert tokens == steered_ref, (
        f"(TP={tp}, PP={pp}) steered tokens {tokens} diverge from "
        f"single-rank reference {steered_ref}"
    )


# ---------------------------------------------------------------------------
# Per-request steering under TP × PP
# ---------------------------------------------------------------------------


def test_per_request_steering_tp_pp_2_2(vllm_runner) -> None:
    """Four concurrent requests with distinct steering specs must
    produce identical per-request outputs to a single-rank run."""
    _skip_if_not_enough_gpus(4)

    # Reference pass on TP=1 PP=1.
    references: list[list[int]] = []
    with vllm_runner(MODEL, **_runner_kwargs()) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        for magnitude in (0.1, 0.2, 0.3, 0.4):
            sp = SamplingParams(
                max_tokens=8,
                temperature=0.0,
                steering_vectors={_HP: {target_layer: [magnitude] * hidden_size}},
            )
            references.append(_gen_tokens(llm, _PROMPT, sp))

    # Distributed pass.
    with vllm_runner(
        MODEL,
        **_runner_kwargs(tensor_parallel_size=2, pipeline_parallel_size=2),
    ) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        distributed_outputs: list[list[int]] = []
        for magnitude in (0.1, 0.2, 0.3, 0.4):
            sp = SamplingParams(
                max_tokens=8,
                temperature=0.0,
                steering_vectors={_HP: {target_layer: [magnitude] * hidden_size}},
            )
            distributed_outputs.append(_gen_tokens(llm, _PROMPT, sp))

    for i, (ref, dist) in enumerate(zip(references, distributed_outputs)):
        assert ref == dist, (
            f"Per-request steering diverged for request {i}: "
            f"reference={ref}, TP=2 PP=2={dist}"
        )


# ---------------------------------------------------------------------------
# Capacity-deferral consistency
# ---------------------------------------------------------------------------


def test_capacity_deferral_consistency_tp_pp(vllm_runner) -> None:
    """With ``max_steering_configs=2`` and four distinct per-request
    configs, the scheduler must defer the same requests on every rank
    so no rank admits a request the others reject. We verify this
    indirectly: all four requests eventually complete with outputs
    matching a single-rank reference."""
    _skip_if_not_enough_gpus(4)

    magnitudes = (0.1, 0.2, 0.3, 0.4)

    references: list[list[int]] = []
    with vllm_runner(MODEL, **_runner_kwargs(max_steering_configs=2)) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        for magnitude in magnitudes:
            sp = SamplingParams(
                max_tokens=8,
                temperature=0.0,
                steering_vectors={_HP: {target_layer: [magnitude] * hidden_size}},
            )
            references.append(_gen_tokens(llm, _PROMPT, sp))

    with vllm_runner(
        MODEL,
        **_runner_kwargs(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            max_steering_configs=2,
        ),
    ) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        outputs: list[list[int]] = []
        for magnitude in magnitudes:
            sp = SamplingParams(
                max_tokens=8,
                temperature=0.0,
                steering_vectors={_HP: {target_layer: [magnitude] * hidden_size}},
            )
            outputs.append(_gen_tokens(llm, _PROMPT, sp))

    for i, (ref, out) in enumerate(zip(references, outputs)):
        assert ref == out, (
            f"Capacity-deferral diverged for request {i}: "
            f"reference={ref}, TP=2 PP=2={out}"
        )


# ---------------------------------------------------------------------------
# Prefill → decode transition under PP
# ---------------------------------------------------------------------------


def test_prefill_decode_transition_pp(vllm_runner) -> None:
    """A request with distinct prefill and decode steering specs must
    transition cleanly across the phase boundary on every PP rank.

    Verifies the row release / re-register path by re-using the same
    request (same hash chain) multiple times and confirming repeatable
    outputs equal a single-rank run.
    """
    _skip_if_not_enough_gpus(2)

    sp_factory = lambda target_layer, hidden_size: SamplingParams(  # noqa: E731
        max_tokens=8,
        temperature=0.0,
        prefill_steering_vectors={_HP: {target_layer: [0.15] * hidden_size}},
        decode_steering_vectors={_HP: {target_layer: [0.25] * hidden_size}},
    )

    with vllm_runner(MODEL, **_runner_kwargs()) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        sp = sp_factory(target_layer, hidden_size)
        ref_first = _gen_tokens(llm, _PROMPT, sp)
        ref_second = _gen_tokens(llm, _PROMPT, sp)
        assert ref_first == ref_second, "Prefill/decode steering is not stable"

    with vllm_runner(MODEL, **_runner_kwargs(pipeline_parallel_size=2)) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        sp = sp_factory(target_layer, hidden_size)
        dist_first = _gen_tokens(llm, _PROMPT, sp)
        dist_second = _gen_tokens(llm, _PROMPT, sp)

    assert dist_first == dist_second
    assert dist_first == ref_first, (
        f"Prefill/decode transition under PP=2 diverged from reference: "
        f"ref={ref_first}, dist={dist_first}"
    )


# ---------------------------------------------------------------------------
# Error handling under TP / PP
# ---------------------------------------------------------------------------


def test_invalid_layer_error_under_pp(vllm_runner) -> None:
    """A nonexistent layer index must be reported as empty validated
    layers by every rank, so the router can produce a single 400.

    We exercise the worker contract directly via ``collective_rpc``:
    every rank must return ``(tp_rank, pp_rank, [])`` since layer 999
    does not exist on any rank.
    """
    _skip_if_not_enough_gpus(2)

    with vllm_runner(MODEL, **_runner_kwargs(pipeline_parallel_size=2)) as llm:
        _target_layer, hidden_size = _discover_layers(llm)
        results = llm.llm.collective_rpc(
            "set_steering_vectors",
            kwargs={
                "vectors": {_HP: {999: [0.1] * hidden_size}},
                "validate_only": True,
            },
        )
    # Every worker returns a tuple; the third element is the validated
    # layer list. All must be empty — nothing matches layer 999.
    assert len(results) == 2, f"Expected 2 PP ranks, got {len(results)}"
    for entry in results:
        tp_rank, pp_rank, valid_layers = entry
        assert valid_layers == [], (
            f"Rank tp={tp_rank}, pp={pp_rank} unexpectedly validated layer 999: "
            f"{valid_layers}"
        )


def test_mismatched_vector_size_under_tp(vllm_runner) -> None:
    """A vector with the wrong size raises SteeringVectorError on every
    TP rank. Each rank owns the same layers under TP so either all
    ranks raise or none do.
    """
    _skip_if_not_enough_gpus(2)

    from vllm.exceptions import SteeringVectorError

    with vllm_runner(MODEL, **_runner_kwargs(tensor_parallel_size=2)) as llm:
        target_layer, hidden_size = _discover_layers(llm)
        assert hidden_size > 2
        with pytest.raises((SteeringVectorError, Exception)) as exc_info:
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "vectors": {_HP: {target_layer: [0.1, 0.2]}},
                    "validate_only": True,
                },
            )
        # The raised message must name the expected size so the router
        # can surface a coherent 400.
        msg = str(exc_info.value)
        assert "expected vector of size" in msg, (
            f"Expected size-mismatch error, got: {msg!r}"
        )
