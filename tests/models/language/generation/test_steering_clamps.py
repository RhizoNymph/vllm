# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for directional projection clamping on Gemma 3.

Pytest-collected counterpart of the manual ``tests/gpu_clamp_validate.py``
HTTP script: every behavior here runs through a real engine forward pass
(dummy weights) instead of a hand-started server. Kernel-level math lives
in ``tests/model_executor/layers/test_clamp_op.py`` (eager) and
``test_clamp_gpu.py`` (Triton parity).

Exactness tests lean on the clamp op's in-bounds guarantee: when the
projection is inside ``[min, max]`` (or ``strength == 0``) the applied
delta is exactly zero, so hidden states — and therefore token ids and
logprobs — are bitwise unchanged. Identity tests use one-hot directions
so both sides of the comparison write bitwise-identical hidden states
regardless of dtype.
"""

import math

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_TABLE_ATTR,
)

MODEL = "google/gemma-3-4b-it"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="clamp e2e tests require CUDA"
)

_HP = DEFAULT_HOOK_POINT.value
_TABLE_ATTR = HOOK_POINT_TABLE_ATTR[DEFAULT_HOOK_POINT]

_PROMPT = "What does the fox say? " * 32

# Dummy (random) weights need large perturbations to overcome logit-space
# noise; matches the [500]*H additive vectors used in test_steering.py
# (L2 ~ 500 * sqrt(hidden)).
_PIN = 30000.0

_LOGPROB_TOL = 1e-6


def _discover_layers(llm):
    """Return (target_layer, hidden_size) for the default hook point."""

    def _discover(worker):
        layers = {}
        model_inst = worker.model_runner.get_model()
        for mod in model_inst.modules():
            if hasattr(mod, _TABLE_ATTR) and hasattr(mod, "layer_idx"):
                layers[mod.layer_idx] = getattr(mod, _TABLE_ATTR).shape[1]
        return layers

    layer_info = llm.llm.collective_rpc(_discover)[0]
    target_layer = max(layer_info.keys()) // 2
    hidden_size = layer_info[target_layer]
    return target_layer, hidden_size


def _gen(llm, prompt, sampling):
    """Generate and return (token_ids, cumulative_logprob)."""
    result = llm.llm.generate([prompt], sampling)
    output = result[0].outputs[0]
    return list(output.token_ids), output.cumulative_logprob


def _unit(hidden: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden)
    return (v / np.linalg.norm(v)).tolist()


def _one_hot(hidden: int, axis: int) -> list[float]:
    v = [0.0] * hidden
    v[axis] = 1.0
    return v


def _clamps(layer: int, entries: list[dict]) -> dict:
    return {_HP: {layer: entries}}


def _sampling(max_tokens: int = 10, **extra) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_tokens, temperature=0.0, logprobs=5, **extra
    )


def _runner_kwargs(**extra) -> dict:
    kwargs = {
        "load_format": "dummy",
        "max_model_len": 512,
        "enable_prefix_caching": True,
        "enable_steering": True,
        "max_steering_configs": 4,
    }
    kwargs.update(extra)
    return kwargs


@pytest.mark.parametrize("model", [MODEL])
def test_wide_bounds_clamp_matches_baseline(vllm_runner, monkeypatch, model):
    """A clamp whose bounds contain the natural projection is a no-op:
    token ids match and the cumulative logprob is unchanged."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            wide = _sampling(
                steering_clamps=_clamps(
                    target_layer,
                    [{"vector": _unit(hidden_size, 1), "min": -1e9, "max": 1e9}],
                )
            )
            tokens, lp = _gen(llm, _PROMPT, wide)

            assert tokens == base_tokens
            assert math.isclose(lp, base_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_strength_zero_pin_matches_baseline(vllm_runner, monkeypatch, model):
    """An aggressive pin with strength=0 applies no delta at all."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            s0 = _sampling(
                steering_clamps=_clamps(
                    target_layer,
                    [
                        {
                            "vector": _unit(hidden_size, 1),
                            "value": _PIN,
                            "strength": 0.0,
                        }
                    ],
                )
            )
            tokens, lp = _gen(llm, _PROMPT, s0)

            assert tokens == base_tokens
            assert math.isclose(lp, base_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_aggressive_pin_changes_output_no_contamination(
    vllm_runner, monkeypatch, model
):
    """An aggressive per-request pin changes generation; a subsequent
    unclamped request reproduces the baseline exactly."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            pinned = _sampling(
                steering_clamps=_clamps(
                    target_layer,
                    [{"vector": _unit(hidden_size, 1), "value": _PIN}],
                )
            )
            pinned_tokens, _ = _gen(llm, _PROMPT, pinned)
            assert pinned_tokens != base_tokens, "Aggressive pin should bite"

            assert llm.llm.reset_prefix_cache()
            restored_tokens, restored_lp = _gen(llm, _PROMPT, _sampling())
            assert restored_tokens == base_tokens
            assert math.isclose(restored_lp, base_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_one_sided_bounds(vllm_runner, monkeypatch, model):
    """A max bound far above the natural projection is a no-op; a max
    bound far below equals a pin at that bound (one-hot direction, so
    both writes are bitwise identical)."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            axis_dir = _one_hot(hidden_size, hidden_size // 2)
            assert llm.llm.reset_prefix_cache()

            cap_high = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": axis_dir, "max": 1e9}]
                )
            )
            tokens_high, lp_high = _gen(llm, _PROMPT, cap_high)
            assert tokens_high == base_tokens
            assert math.isclose(lp_high, base_lp, abs_tol=_LOGPROB_TOL)

            assert llm.llm.reset_prefix_cache()
            cap_low = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": axis_dir, "max": -_PIN}]
                )
            )
            tokens_cap, lp_cap = _gen(llm, _PROMPT, cap_low)

            assert llm.llm.reset_prefix_cache()
            pin_low = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": axis_dir, "value": -_PIN}]
                )
            )
            tokens_pin, lp_pin = _gen(llm, _PROMPT, pin_low)

            assert tokens_cap == tokens_pin
            assert math.isclose(lp_cap, lp_pin, abs_tol=_LOGPROB_TOL)
            assert tokens_cap != base_tokens, "Binding one-sided cap should bite"


@pytest.mark.parametrize("model", [MODEL])
def test_clamp_after_additive_erases_component(vllm_runner, monkeypatch, model):
    """The clamp runs after the additive tier at the same site, so a pin
    along the steered direction erases the additive component: steer+pin
    must equal pin-only. One-hot direction keeps the comparison bitwise."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, _ = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            axis = hidden_size // 2
            axis_dir = _one_hot(hidden_size, axis)
            add_vec = [0.0] * hidden_size
            add_vec[axis] = _PIN
            assert llm.llm.reset_prefix_cache()

            add_only = _sampling(steering_vectors={_HP: {target_layer: add_vec}})
            add_tokens, _ = _gen(llm, _PROMPT, add_only)
            assert add_tokens != base_tokens, "Additive component should bite"

            assert llm.llm.reset_prefix_cache()
            pin_only = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": axis_dir, "value": 0.0}]
                )
            )
            pin_tokens, pin_lp = _gen(llm, _PROMPT, pin_only)

            assert llm.llm.reset_prefix_cache()
            add_and_pin = _sampling(
                steering_vectors={_HP: {target_layer: add_vec}},
                steering_clamps=_clamps(
                    target_layer, [{"vector": axis_dir, "value": 0.0}]
                ),
            )
            both_tokens, both_lp = _gen(llm, _PROMPT, add_and_pin)

            assert both_tokens == pin_tokens
            assert math.isclose(both_lp, pin_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_decode_only_clamp_preserves_first_token(vllm_runner, monkeypatch, model):
    """Decode-only clamps leave the prefill forward untouched — the first
    sampled token matches baseline — while the continuation diverges."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, _ = _gen(llm, _PROMPT, _sampling(max_tokens=16))
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            decode_pinned = _sampling(
                max_tokens=16,
                decode_steering_clamps=_clamps(
                    target_layer,
                    [{"vector": _unit(hidden_size, 1), "value": _PIN}],
                ),
            )
            pinned_tokens, _ = _gen(llm, _PROMPT, decode_pinned)

            assert pinned_tokens[0] == base_tokens[0], (
                "Decode-only clamp must not perturb the prefill-sampled token"
            )
            assert pinned_tokens != base_tokens, (
                "Decode-only clamp should change the continuation"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_global_clamps_via_worker_rpc(vllm_runner, monkeypatch, model):
    """Global clamps set through the worker RPC (the /v1/steering/set
    path) shift generation on an otherwise steering-free engine, and
    clearing restores the baseline exactly."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "clamps": _clamps(
                        target_layer,
                        [{"vector": _unit(hidden_size, 1), "value": _PIN}],
                    )
                },
            )
            clamped_tokens, _ = _gen(llm, _PROMPT, _sampling())
            assert clamped_tokens != base_tokens, "Global clamp should bite"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()
            restored_tokens, restored_lp = _gen(llm, _PROMPT, _sampling())
            assert restored_tokens == base_tokens
            assert math.isclose(restored_lp, base_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_global_clamps_validate_only_does_not_mutate(vllm_runner, monkeypatch, model):
    """validate_only reports the target layer without applying clamps."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, base_lp = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            results = llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "clamps": _clamps(
                        target_layer,
                        [{"vector": _unit(hidden_size, 1), "value": _PIN}],
                    ),
                    "validate_only": True,
                },
            )
            for _tp_rank, _pp_rank, valid_layers in results:
                assert valid_layers == [target_layer]

            tokens, lp = _gen(llm, _PROMPT, _sampling())
            assert tokens == base_tokens
            assert math.isclose(lp, base_lp, abs_tol=_LOGPROB_TOL)


@pytest.mark.parametrize("model", [MODEL])
def test_prefix_cache_respects_clamps(vllm_runner, monkeypatch, model):
    """Clamps participate in the steering config hash: same prompt with
    different clamps must not share cached prefill KV, while identical
    clamps hit the cache and reproduce exactly."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            base_tokens, _ = _gen(llm, _PROMPT, _sampling())
            target_layer, hidden_size = _discover_layers(llm)
            assert llm.llm.reset_prefix_cache()

            pinned = _sampling(
                steering_clamps=_clamps(
                    target_layer,
                    [{"vector": _unit(hidden_size, 1), "value": _PIN}],
                )
            )
            tokens_a, _ = _gen(llm, _PROMPT, pinned)
            assert tokens_a != base_tokens, "Clamped prefill should bite"

            tokens_b, _ = _gen(llm, _PROMPT, _sampling())
            assert tokens_b == base_tokens, (
                "An unclamped request must not reuse clamped prefill KV"
            )

            tokens_c, _ = _gen(llm, _PROMPT, pinned)
            assert tokens_c == tokens_a, (
                "Identical clamps should hit the prefix cache and reproduce"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_clamps_concurrent_with_cuda_graphs(
    vllm_runner, monkeypatch, model
):
    """Distinct per-request clamp configs batched together produce
    distinct outputs under CUDA graph replay, and a later unclamped
    batch reproduces the unclamped baseline."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs(enable_prefix_caching=False)) as llm:
            target_layer, hidden_size = _discover_layers(llm)
            direction = _unit(hidden_size, 1)

            no_clamp = _sampling()
            pin_pos = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": direction, "value": _PIN}]
                )
            )
            pin_neg = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": direction, "value": -_PIN}]
                )
            )

            outputs = llm.llm.generate(
                [_PROMPT, _PROMPT, _PROMPT], [no_clamp, pin_pos, pin_neg]
            )
            tokens_none = list(outputs[0].outputs[0].token_ids)
            tokens_pos = list(outputs[1].outputs[0].token_ids)
            tokens_neg = list(outputs[2].outputs[0].token_ids)

            assert tokens_pos != tokens_neg, (
                "Opposite pins should produce different outputs"
            )
            assert tokens_pos != tokens_none or tokens_neg != tokens_none, (
                "At least one clamped request should differ from unclamped"
            )

            outputs2 = llm.llm.generate([_PROMPT, _PROMPT], [no_clamp, no_clamp])
            assert list(outputs2[0].outputs[0].token_ids) == tokens_none
            assert list(outputs2[1].outputs[0].token_ids) == tokens_none


@pytest.mark.parametrize("model", [MODEL])
def test_packed_clamps_match_json(vllm_runner, monkeypatch, model):
    """The legacy base64-packed clamp submission materializes the same
    device state as the JSON entry-list form."""
    import base64

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(model, **_runner_kwargs()) as llm:
            target_layer, hidden_size = _discover_layers(llm)
            direction = _unit(hidden_size, 1)

            json_pinned = _sampling(
                steering_clamps=_clamps(
                    target_layer, [{"vector": direction, "value": _PIN}]
                )
            )
            tokens_json, lp_json = _gen(llm, _PROMPT, json_pinned)

            assert llm.llm.reset_prefix_cache()
            rows = np.asarray([direction], dtype=np.float64)
            packed = {
                _HP: {
                    "dtype": "float64",
                    "shape": [1, hidden_size],
                    "layer_indices": [target_layer],
                    "data": base64.b64encode(rows.tobytes()).decode("ascii"),
                    "bounds": [[_PIN, _PIN]],
                    "strengths": [1.0],
                }
            }
            packed_pinned = _sampling(steering_clamps=packed)
            tokens_packed, lp_packed = _gen(llm, _PROMPT, packed_pinned)

            assert tokens_packed == tokens_json
            assert math.isclose(lp_packed, lp_json, abs_tol=_LOGPROB_TOL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
