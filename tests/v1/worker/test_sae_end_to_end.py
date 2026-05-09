# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end Stage-3 tests for the SAE feature-surgery path.

Exercises the full per-step pipeline against a synthetic decoder
layer — no real model needed.  The harness reuses the ``_RichHarness``
pattern from Stage 2 (mock ``vllm_config``, synthetic
``_steerable_layers_cache``) and adds the missing pieces Stage 3
relies on: a stub ``input_batch`` carrying per-request hashes and
phase state, a stub ``scheduler_output``, and a tiny forward harness
that calls ``apply_layer_steering`` (which now also dispatches SAE
delta) on a fixed residual.

What's covered:

* End-to-end forward: register an SAE module, attach weights, admit
  a request with an absolute clamp, run the per-step buffer update,
  call ``apply_layer_steering`` on a synthetic residual and verify
  the output reflects the clamp.
* Per-token routing: two requests with different SAE rows produce
  different per-token deltas in a single forward.
* Prefill→decode transition: a request that completes prefill in a
  step transitions cleanly so the next decode step still applies its
  decode-phase clamp.
* Disabled-mode parity: when no SAE module is registered, the
  composed ``apply_layer_steering`` is bit-for-bit identical to the
  pre-Stage-3 additive path (no SAE kernel runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.model_executor.layers.sae_steering import (
    HOOK_POINT_SAE_CLAMP_KIND_ATTR,
    sae_buffers_attached,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    apply_layer_steering,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin


@dataclass
class _StubModelConfig:
    dtype: torch.dtype = torch.float32


@dataclass
class _StubSchedulerConfig:
    max_num_batched_tokens: int = 16
    max_num_seqs: int = 8


@dataclass
class _StubSteeringConfig:
    max_steering_configs: int = 4


@dataclass
class _StubVllmConfig:
    model_config: _StubModelConfig = field(default_factory=_StubModelConfig)
    scheduler_config: _StubSchedulerConfig = field(default_factory=_StubSchedulerConfig)
    steering_config: _StubSteeringConfig = field(default_factory=_StubSteeringConfig)


@dataclass
class _StubInputBatch:
    """Just enough of vllm.v1.worker.gpu_input_batch.InputBatch."""

    num_reqs: int = 0
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    num_computed_tokens_cpu: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int64)
    )
    num_prompt_tokens: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int64)
    )
    request_prefill_steering_hash: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int64)
    )
    request_decode_steering_hash: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.int64)
    )


@dataclass
class _StubSchedulerOutput:
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)


def _make_decoder_layer(layer_idx: int, hidden_size: int = 4) -> nn.Module:
    m = nn.Module()
    m.layer_idx = layer_idx  # type: ignore[attr-defined]
    for hp in SteeringHookPoint:
        m.register_buffer(
            HOOK_POINT_TABLE_ATTR[hp],
            torch.zeros(7, hidden_size, dtype=torch.float32),
            persistent=False,
        )
    m.register_buffer(
        "steering_index", torch.zeros(16, dtype=torch.long), persistent=False
    )
    return m


class _E2EHarness(SteeringModelRunnerMixin):
    """Mixin instance with full Stage-3 surface populated."""

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...] = (20,),
        hidden_size: int = 4,
        max_sae_configs: int = 4,
        max_tokens: int = 16,
    ) -> None:
        self.vllm_config = _StubVllmConfig(
            steering_config=_StubSteeringConfig(max_steering_configs=max_sae_configs),
            scheduler_config=_StubSchedulerConfig(max_num_batched_tokens=max_tokens),
        )
        self._steerable_layers_cache = {
            idx: _make_decoder_layer(idx, hidden_size=hidden_size)
            for idx in layer_indices
        }
        self._locally_owned_layers = frozenset(layer_indices)
        self._steering_module_registry: dict = {}
        self._sae_module_registry: dict = {}
        self._sae_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._req_steering_phase: dict = {}
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(max_sae_configs)
        self._steering_manager = None  # SAE-only path for these tests
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
        self._sae_rows_scratch = np.zeros(8, dtype=np.int64)
        self._sae_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
        self.input_batch = _StubInputBatch()
        self.requests: dict = {}


def _sae_payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_mlp"),),
    clampable: tuple[int, ...] = (0,),
    activation: str = "relu",
    d_model: int = 4,
) -> dict:
    return {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": 64,
            "activation": activation,
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable),
            "activation_params": {},
            "weights_uri": None,
        },
    }


def _attach_identity_weights(
    h: _E2EHarness,
    name: str,
    *,
    n_clamp: int,
    hidden_size: int,
    layers: tuple[tuple[int, str], ...] = ((20, "post_mlp"),),
) -> None:
    """Attach an encoder that picks h[0..n_clamp-1] and a decoder
    that writes feature i back into h[i].

    Convenient default: feature i's encoder dot product is exactly
    h[i], and feature i's decoder direction is the unit vector e_i.
    Clamping feature i to value v adds (v - h[i]) to h[i].
    """
    enc_w = torch.zeros(n_clamp, hidden_size)
    dec_w = torch.zeros(n_clamp, hidden_size)
    for i in range(n_clamp):
        enc_w[i, i] = 1.0
        dec_w[i, i] = 1.0
    weights = {
        layer_hook: {
            "encoder_weight": enc_w.clone(),
            "encoder_bias": torch.zeros(n_clamp),
            "decoder_weight": dec_w.clone(),
        }
        for layer_hook in layers
    }
    h.attach_sae_weights(name, weights)


class TestSingleRequestEndToEnd:
    """One request, one SAE clamp, exercising the full pipeline."""

    def test_absolute_clamp_applied_to_residual(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        # Register SAE module for layer 20 / post_mlp covering feature 0.
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        # Admit a request that clamps feature 0 to 7.0 in prefill.
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=7.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=True,
        )
        # Build the input batch with one request, two tokens scheduled.
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 111
        h.input_batch.request_decode_steering_hash[0] = 222
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 2})
        # Per-step buffer update.
        h._update_sae_buffers(scheduler_output)

        # Forward: residual where h[0] varies per token.  Both tokens
        # should be clamped to 7.0 in dim 0.
        residual = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        )
        site = h._steerable_layers_cache[20]
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_MLP)
        # Token 0: f = ReLU(2) = 2; delta = 7-2 = 5.  out[0,0] = 7.
        # Token 1: f = ReLU(3) = 3; delta = 7-3 = 4.  out[1,0] = 7.
        assert torch.allclose(out[:, 0], torch.tensor([7.0, 7.0]))
        # Other dims untouched by the decoder unit row.
        assert torch.allclose(out[:, 1:], residual[:, 1:])

    def test_disabled_mode_unchanged_behavior(self):
        """No SAE module registered → forward is bit-identical to additive-only."""
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        site = h._steerable_layers_cache[20]
        residual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_MLP)
        # Additive table is zero-initialised → no change; SAE not
        # attached → no change.  Output equals input.
        assert torch.allclose(out, residual)
        # And: no SAE buffers were attached as a side effect.
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_MLP)


class TestPerTokenRouting:
    """Two requests with different SAE configs route to different rows."""

    def test_two_requests_different_clamps(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        _attach_identity_weights(h, "g", n_clamp=2, hidden_size=4)

        spec_a = SAEClampSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=10.0),)
                }
            },
        )
        spec_b = SAEClampSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=1, kind="absolute", value=20.0),)
                }
            },
        )
        h._register_initial_sae_clamps(
            "req-a",
            SamplingParams(sae_clamp_specs=(spec_a,)),
            prefill_hash=100,
            decode_hash=0,
            is_prefilling=True,
        )
        h._register_initial_sae_clamps(
            "req-b",
            SamplingParams(sae_clamp_specs=(spec_b,)),
            prefill_hash=200,
            decode_hash=0,
            is_prefilling=True,
        )

        h.input_batch.num_reqs = 2
        h.input_batch.req_ids = ["req-a", "req-b"]
        h.input_batch.req_id_to_index = {"req-a": 0, "req-b": 1}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_computed_tokens_cpu[1] = 0
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.num_prompt_tokens[1] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_prefill_steering_hash[1] = 200
        scheduler_output = _StubSchedulerOutput(
            num_scheduled_tokens={"req-a": 1, "req-b": 1}
        )
        h._update_sae_buffers(scheduler_output)

        # One token per request; req-a is token index 0, req-b is index 1.
        residual = torch.zeros(2, 4)
        site = h._steerable_layers_cache[20]
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_MLP)
        # req-a (token 0): feature 0 clamped to 10 → out[0, 0] = 10.
        # req-b (token 1): feature 1 clamped to 20 → out[1, 1] = 20.
        assert out[0, 0].item() == 10.0
        assert out[1, 1].item() == 20.0
        # Cross-routing must NOT happen.
        assert out[0, 1].item() == 0.0
        assert out[1, 0].item() == 0.0


class TestPrefillToDecodeTransition:
    """A request that completes prefill in this step decodes correctly next step."""

    def test_transition_releases_prefill_registers_decode(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)

        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        # Pre-transition: prefill row registered.
        assert (100, "prefill") in h._sae_clamp_manager.config_to_row
        assert (200, "decode") not in h._sae_clamp_manager.config_to_row

        # Drive a transition manually (the additive path normally does
        # this from _update_steering_buffers; we exercise the SAE half
        # directly because the additive path is None in this harness).
        # Mirror what _handle_steering_transition's SAE block does.
        h._sae_clamp_manager.release_clamp_spec(100, "prefill")
        h._sae_clamp_manager.register_clamp_spec(200, sp.sae_clamp_specs, "decode")
        h._req_sae_phase["req-1"] = "decode"

        # Post-transition: decode row replaces prefill row.
        assert (100, "prefill") not in h._sae_clamp_manager.config_to_row
        assert (200, "decode") in h._sae_clamp_manager.config_to_row

        # Decode step: request is past prefill, hash 200 is the decode hash.
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 4
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_decode_steering_hash[0] = 200
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        h._update_sae_buffers(scheduler_output)

        residual = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        site = h._steerable_layers_cache[20]
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_MLP)
        # Feature 0 clamped to 5: out[0,0] = 5.
        assert out[0, 0].item() == 5.0


class TestPopulatorWritesIntoBuffers:
    """The Stage-3 update flow writes manager rows into per-layer buffers."""

    def test_populate_writes_clamp_kind(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        _attach_identity_weights(h, "g", n_clamp=2, hidden_size=4)

        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=1, kind="absolute", value=9.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_prompt_tokens[0] = 2
        h.input_batch.request_prefill_steering_hash[0] = 111
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        h._update_sae_buffers(scheduler_output)

        site = h._steerable_layers_cache[20]
        kind = getattr(site, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP])
        # Row 1 (the prefill row) has feature 1 clamped (not feature 0).
        from vllm.model_executor.layers.sae_steering import (
            CLAMP_KIND_ABSOLUTE,
            CLAMP_KIND_NONE,
        )

        assert kind[1, 0].item() == CLAMP_KIND_NONE
        assert kind[1, 1].item() == CLAMP_KIND_ABSOLUTE


class TestNoActiveStateShortCircuit:
    """``_update_sae_buffers`` short-circuits when nothing is active."""

    def test_no_sae_module_registered_is_noop(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        # No register call; _sae_steerable_sites is empty.
        h._update_sae_buffers(_StubSchedulerOutput())
        site = h._steerable_layers_cache[20]
        # No SAE buffers should ever have been allocated.
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_MLP)
