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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
)
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.sae_steering import (
    HOOK_POINT_SAE_ANY_ACTIVE_ATTR,
    HOOK_POINT_SAE_CLAMP_KIND_ATTR,
    sae_buffers_attached,
)
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_layer_steering,
    register_steering_buffers,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.steering_manager import SteeringManager
from vllm.v1.worker.steering_model_runner_mixin import (
    SteeringModelRunnerMixin,
    _SteeringReqState,
)


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
    register_steering_buffers(
        m,
        hidden_size,
        max_steering_tokens=16,
        max_steering_configs=4,
        dtype=torch.float32,
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
        self._steering_module_resolved_cache: dict = {}
        self._steering_module_pinned_rows: dict = {}
        self._sae_module_registry: dict = {}
        self._sae_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._steering_reqs: dict = {}
        self._req_dynamic_decode: dict = {}
        self._req_override_source: dict = {}
        self._req_decode_sig_reported: dict = {}
        self._req_transition_scan_candidates: set[str] = set()
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(max_sae_configs)
        self._steering_manager = None  # SAE-only path for these tests
        self._steering_rows_scratch = np.zeros(8, dtype=np.int64)
        self._steering_n_tokens_scratch = np.zeros(8, dtype=np.int64)
        self._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
        self._steering_tier_gain_scratch = np.zeros(8, dtype=np.float32)
        self._steering_decode_mask_scratch = np.zeros(8, dtype=np.float32)
        self._steering_token_scales_pinned = torch.zeros(
            max_tokens, dtype=torch.float32
        )
        self._steering_decode_mask_pinned = torch.zeros(
            max_tokens, dtype=torch.float32
        )
        self._sae_rows_scratch = np.zeros(8, dtype=np.int64)
        self._sae_index_pinned = torch.zeros(max_tokens, dtype=torch.long)
        self.input_batch = _StubInputBatch()
        self.requests: dict = {}

    @property
    def _req_steering_phase(self) -> dict:
        """Legacy view of per-request additive phase.

        The rebuilt mixin tracks phase on the canonical ``_steering_reqs``
        store; this read-only view keeps the original assertions'
        semantics (request -> registered additive phase).
        """
        return {rid: rs.phase for rid, rs in self._steering_reqs.items()}


def _sync_reqs_from_batch(h: "_E2EHarness") -> None:
    """Mirror the stub input_batch's per-request combined hashes into the
    canonical ``_steering_reqs`` store.

    The rebuilt mixin reads per-request steering identity (combined
    per-phase hashes + prompt length + phase) from ``_steering_reqs``
    rather than InputBatch columns, so tests that stage identity on the
    stub columns must sync before driving a per-step update.  Requests
    with both hashes zero are skipped, mirroring the mixin's own
    registration guard (untracked requests route with hash 0).
    """
    ib = h.input_batch
    for i, rid in enumerate(ib.req_ids):
        if rid is None:
            continue
        prefill_hash = int(ib.request_prefill_steering_hash[i])
        decode_hash = int(ib.request_decode_steering_hash[i])
        if prefill_hash == 0 and decode_hash == 0:
            continue
        req_state = h.requests.get(rid)
        sp = getattr(req_state, "sampling_params", None)
        if sp is None:
            sp = SamplingParams()
        num_computed = int(ib.num_computed_tokens_cpu[i])
        num_prompt = int(ib.num_prompt_tokens[i])
        h._steering_reqs[rid] = _SteeringReqState(
            sampling_params=sp,
            prefill_hash=prefill_hash,
            decode_hash=decode_hash,
            num_prompt_tokens=num_prompt,
            phase="prefill" if num_computed < num_prompt else "decode",
        )


def _assert_no_additive_steering(site, additive_row: int) -> None:
    """Assert the additive index carries no steering contribution.

    Row 0 is the no-steer sentinel; row 1/2 are the global effective rows,
    which are all-zero unless global vectors were set.  Either routing means
    the additive tier contributed nothing.
    """
    from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR

    assert additive_row in (0, 1, 2)
    if additive_row != 0:
        table = getattr(site, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK])
        assert torch.all(table[additive_row] == 0)


def _sae_payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_block"),),
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
    layers: tuple[tuple[int, str], ...] = ((20, "post_block"),),
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
        # Register SAE module for layer 20 / post_block covering feature 0.
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        # Admit a request that clamps feature 0 to 7.0 in prefill.
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
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
        _sync_reqs_from_batch(h)
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
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
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
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        # Additive table is zero-initialised → no change; SAE not
        # attached → no change.  Output equals input.
        assert torch.allclose(out, residual)
        # And: no SAE buffers were attached as a side effect.
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)


class TestPerTokenRouting:
    """Two requests with different SAE configs route to different rows."""

    def test_two_requests_different_clamps(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        _attach_identity_weights(h, "g", n_clamp=2, hidden_size=4)

        spec_a = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=10.0),)
                }
            },
        )
        spec_b = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
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
        _sync_reqs_from_batch(h)
        h._update_sae_buffers(scheduler_output)

        # One token per request; req-a is token index 0, req-b is index 1.
        residual = torch.zeros(2, 4)
        site = h._steerable_layers_cache[20]
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
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
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        prefill_sae_hash = sp.prefill_sae_clamp_config_hash
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        # Pre-transition: prefill row registered.
        assert (prefill_sae_hash, "prefill") in h._sae_clamp_manager.config_to_row
        assert (decode_sae_hash, "decode") not in h._sae_clamp_manager.config_to_row

        # Drive a transition manually (the additive path normally does
        # this from _update_steering_buffers; we exercise the SAE half
        # directly because the additive path is None in this harness).
        h._handle_sae_transition("req-1", prefill_hash=100, decode_hash=200, sp=sp)

        # Post-transition: decode row replaces prefill row.
        assert (prefill_sae_hash, "prefill") not in h._sae_clamp_manager.config_to_row
        assert (decode_sae_hash, "decode") in h._sae_clamp_manager.config_to_row

        # Decode step: request is past prefill, hash 200 is the decode hash.
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 4
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_decode_steering_hash[0] = 200
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        _sync_reqs_from_batch(h)
        h._update_sae_buffers(scheduler_output)

        residual = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        site = h._steerable_layers_cache[20]
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        # Feature 0 clamped to 5: out[0,0] = 5.
        assert out[0, 0].item() == 5.0

    def test_update_steering_buffers_keeps_sae_on_final_prefill_token(self):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)

        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        h.requests["req-1"] = SimpleNamespace(sampling_params=sp)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_decode_steering_hash[0] = 200

        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )

        site = h._steerable_layers_cache[20]
        assert int(site.sae_index[0].item()) == 3
        prefill_sae_hash = sp.prefill_sae_clamp_config_hash
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        assert (prefill_sae_hash, "prefill") not in h._sae_clamp_manager.config_to_row
        assert (decode_sae_hash, "decode") in h._sae_clamp_manager.config_to_row

        residual = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        out = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        assert out[0, 0].item() == 5.0

    def test_final_prefill_uses_prefill_content_then_decode_repopulates(self):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)

        prefill_spec = SAEClampSpec(
            module_name="g",
            phase="prefill",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        decode_spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(prefill_spec, decode_spec))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        h.requests["req-1"] = SimpleNamespace(sampling_params=sp)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_decode_steering_hash[0] = 200

        site = h._steerable_layers_cache[20]
        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )

        residual = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        final_prefill = apply_layer_steering(
            site, residual, SteeringHookPoint.POST_BLOCK
        )
        assert final_prefill[0, 0].item() == 5.0

        h.input_batch.num_computed_tokens_cpu[0] = 4
        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )
        decode = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        assert decode[0, 0].item() == 9.0

    def test_sae_only_no_active_additive_shortcut_registers_decode_row(self):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)

        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        h.requests["req-1"] = SimpleNamespace(sampling_params=sp)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 100
        h.input_batch.request_decode_steering_hash[0] = 200

        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )

        site = h._steerable_layers_cache[20]
        # An SAE-only request defeats the additive nothing-active shortcut
        # (nonzero combined hash), so the rebuilt main path routes its token
        # to row 1 (global prefill effective) instead of leaving the row-0
        # sentinel.  With no globals set that row is all zeros — the additive
        # contribution is still nil.
        _assert_no_additive_steering(site, int(site.steering_index[0].item()))
        assert int(site.sae_index[0].item()) == 3
        prefill_sae_hash = sp.prefill_sae_clamp_config_hash
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        assert (prefill_sae_hash, "prefill") not in h._sae_clamp_manager.config_to_row
        assert (decode_sae_hash, "decode") in h._sae_clamp_manager.config_to_row


class TestGlobalSaeRows:
    def test_hash_zero_routes_to_phase_specific_global_rows(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)

        prefill_spec = SAEClampSpec(
            module_name="g",
            phase="prefill",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        decode_spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=9.0),)
                }
            },
        )
        h._sae_clamp_manager.set_global_clamps(
            prefill_specs=(prefill_spec,),
            decode_specs=(decode_spec,),
        )
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[0] = 0
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})

        site = h._steerable_layers_cache[20]
        residual = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

        h.input_batch.num_computed_tokens_cpu[0] = 3
        h._update_sae_buffers(scheduler_output)
        assert int(site.sae_index[0].item()) == 1
        prefill = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        assert prefill[0, 0].item() == 5.0

        h.input_batch.num_computed_tokens_cpu[0] = 4
        h._update_sae_buffers(scheduler_output)
        assert int(site.sae_index[0].item()) == 2
        decode = apply_layer_steering(site, residual, SteeringHookPoint.POST_BLOCK)
        assert decode[0, 0].item() == 9.0


class TestPopulatorWritesIntoBuffers:
    """The Stage-3 update flow writes manager rows into per-layer buffers."""

    def test_populate_writes_clamp_kind(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        _attach_identity_weights(h, "g", n_clamp=2, hidden_size=4)

        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
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
        _sync_reqs_from_batch(h)
        h._update_sae_buffers(scheduler_output)

        site = h._steerable_layers_cache[20]
        kind = getattr(site, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK])
        row = h._sae_clamp_manager.config_to_row[
            (sp.prefill_sae_clamp_config_hash, "prefill")
        ]
        # The prefill row has feature 1 clamped (not feature 0).
        from vllm.model_executor.layers.sae_steering import (
            CLAMP_KIND_ABSOLUTE,
            CLAMP_KIND_NONE,
        )

        assert kind[row, 0].item() == CLAMP_KIND_NONE
        assert kind[row, 1].item() == CLAMP_KIND_ABSOLUTE


class TestNoActiveStateShortCircuit:
    """``_update_sae_buffers`` short-circuits when nothing is active."""

    def test_no_sae_module_registered_is_noop(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        # No register call; _sae_steerable_sites is empty.
        h._update_sae_buffers(_StubSchedulerOutput())
        site = h._steerable_layers_cache[20]
        # No SAE buffers should ever have been allocated.
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)

    def test_active_to_inactive_clears_any_active_flag(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=7.0),)
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
        _sync_reqs_from_batch(h)
        h._update_sae_buffers(scheduler_output)

        site = h._steerable_layers_cache[20]
        any_active = getattr(
            site, HOOK_POINT_SAE_ANY_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert any_active.item()

        h._release_sae_for_request("req-1", prefill_hash=111, decode_hash=222)
        h._update_sae_buffers(_StubSchedulerOutput())

        assert not any_active.item()
        assert int(site.sae_index[0].item()) == 0


class TestCrossManagerHashIsolation:
    """The combined ``prefill_steering_config_hash`` is shared between
    the additive and SAE managers.  A request that only carries SAE
    clamps still produces a nonzero hash, but the additive manager has
    no row for it — and vice versa.  Both ``_update_steering_buffers``
    and ``_update_sae_buffers`` must gate their lookups on
    ``config_to_row`` membership rather than raising."""

    def test_additive_only_request_does_not_raise_in_sae_lookup(self):
        """An additive-only request still hits ``_update_sae_buffers``
        once SAE buffers are attached; its nonzero hash is absent from
        the SAE manager's row map.  The lookup must fall back to
        row 0 (no-op) instead of raising."""
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        # Attach SAE buffers via registration, but admit a request with
        # no SAE clamps — only an additive hash.
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_prompt_tokens[0] = 4
        # Nonzero hash (request carries additive state) but never
        # registered with the SAE manager.
        h.input_batch.request_prefill_steering_hash[0] = 999
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        # Must not raise.
        h._update_sae_buffers(scheduler_output)
        # And the per-token SAE index must point to the no-op row (0).
        site = h._steerable_layers_cache[20]
        assert int(site.sae_index[0].item()) == 0

    def test_sae_only_request_does_not_raise_in_additive_lookup(self):
        """An SAE-only request reaches ``_update_steering_buffers``
        with a nonzero hash but no additive row registered.  Must fall
        back to global rows (hash==0 semantics) instead of raising."""
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        # Real additive manager so the additive path executes; SAE
        # manager already exists from the harness ctor.
        h._steering_manager = SteeringManager(max_steering_configs=4)
        # Register an SAE module and admit an SAE-only request.
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=3.0),)
                }
            },
        )
        h._register_initial_sae_clamps(
            "req-1",
            SamplingParams(sae_clamp_specs=(spec,)),
            prefill_hash=777,
            decode_hash=0,
            is_prefilling=True,
        )
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 777
        scheduler_output = _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        # Must not raise on the additive lookup; the request has no
        # additive state and no global vectors, so the additive tier
        # contributes nothing while the SAE side still updates.
        _sync_reqs_from_batch(h)
        h._update_steering_buffers(scheduler_output)
        site = h._steerable_layers_cache[20]
        _assert_no_additive_steering(site, int(site.steering_index[0].item()))
        # SAE side still routes to the request's first per-request row.
        assert int(site.sae_index[0].item()) == 3

    def test_released_sae_row_clears_index_on_additive_no_active_shortcut(self):
        """After the last SAE row is released, the additive no-active
        shortcut still has to clear ``sae_index``.  Otherwise future
        unsteered tokens can keep gathering a stale nonzero SAE row."""
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=3.0),)
                }
            },
        )
        h._register_initial_sae_clamps(
            "req-1",
            SamplingParams(sae_clamp_specs=(spec,)),
            prefill_hash=777,
            decode_hash=0,
            is_prefilling=True,
        )
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 0
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 777
        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )
        site = h._steerable_layers_cache[20]
        assert int(site.sae_index[0].item()) == 3

        h._release_sae_for_request("req-1", prefill_hash=777, decode_hash=0)
        h.input_batch.req_ids = ["req-2"]
        h.input_batch.req_id_to_index = {"req-2": 0}
        h.input_batch.request_prefill_steering_hash[0] = 0
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-2": 1})
        )
        assert int(site.sae_index[0].item()) == 0

    def test_sae_index_build_grows_token_count_scratch(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_tokens=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=3.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=777, decode_hash=0, is_prefilling=True
        )
        h._register_initial_sae_clamps(
            "req-2", sp, prefill_hash=777, decode_hash=0, is_prefilling=True
        )
        h._sae_rows_scratch = np.zeros(8, dtype=np.int64)
        h._steering_n_tokens_scratch = np.zeros(1, dtype=np.int64)
        h.input_batch = _StubInputBatch(
            num_reqs=2,
            req_ids=["req-1", "req-2"],
            req_id_to_index={"req-1": 0, "req-2": 1},
            num_computed_tokens_cpu=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            num_prompt_tokens=np.array([4, 4, 0, 0, 0, 0, 0, 0]),
            request_prefill_steering_hash=np.array([777, 777, 0, 0, 0, 0, 0, 0]),
            request_decode_steering_hash=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        )

        _sync_reqs_from_batch(h)
        h._update_sae_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1, "req-2": 1})
        )

        site = h._steerable_layers_cache[20]
        assert site.sae_index[:2].tolist() == [3, 3]
        assert h._sae_rows_scratch.shape[0] == 2
        assert h._steering_n_tokens_scratch is not None
        assert h._steering_n_tokens_scratch.shape[0] == 2

    def test_no_active_shortcut_skips_unsteered_transition_scan(self):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["plain"]
        h.input_batch.req_id_to_index = {"plain": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[0] = 0

        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"plain": 1})
        )

        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}

    def test_no_active_shortcut_ignores_stale_decode_hash_tail(self):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["plain"]
        h.input_batch.req_id_to_index = {"plain": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[7] = 999

        assert not h._may_need_prefill_completion_transition_scan()
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"plain": 1})
        )

        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}

    def test_no_active_shortcut_skips_decode_hash_scan_without_sae_rows(
        self, monkeypatch
    ):
        from vllm.v1.worker.steering_manager import SteeringManager

        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["plain"]
        h.input_batch.req_id_to_index = {"plain": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 3
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[0] = 123

        def fail_transition_scan(_scheduler_output):
            raise AssertionError("inactive path should not scan decode hashes")

        monkeypatch.setattr(
            h,
            "_handle_sae_transitions_for_scheduled_prefill_completions",
            fail_transition_scan,
        )

        assert not h._may_need_prefill_completion_transition_scan()
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"plain": 1})
        )

        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}

    def test_no_active_shortcut_skips_transition_scan_for_decode_sae_row(
        self, monkeypatch
    ):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h._steering_manager = SteeringManager(max_steering_configs=4)
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            phase="decode",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=0, decode_hash=200, is_prefilling=False
        )
        h.input_batch.num_reqs = 1
        h.input_batch.req_ids = ["req-1"]
        h.input_batch.req_id_to_index = {"req-1": 0}
        h.input_batch.num_computed_tokens_cpu[0] = 4
        h.input_batch.num_prompt_tokens[0] = 4
        h.input_batch.request_prefill_steering_hash[0] = 0
        h.input_batch.request_decode_steering_hash[0] = 200

        def fail_transition_scan(_scheduler_output):
            raise AssertionError("decode SAE rows should not scan transitions")

        monkeypatch.setattr(
            h,
            "_handle_sae_transitions_for_scheduled_prefill_completions",
            fail_transition_scan,
        )

        assert not h._may_need_prefill_completion_transition_scan()
        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )

        site = h._steerable_layers_cache[20]
        assert int(site.sae_index[0].item()) == 3
        assert h._req_sae_phase == {"req-1": "decode"}


class TestBatchedAdditiveTransitions:
    """Additive prefill->decode transitions are applied batch-wise."""

    def test_shared_sae_prefill_row_frees_before_decode_registration(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        _attach_identity_weights(h, "g", n_clamp=1, hidden_size=4)
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(sae_clamp_specs=(spec,))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        h._register_initial_sae_clamps(
            "req-2", sp, prefill_hash=100, decode_hash=200, is_prefilling=True
        )
        h.requests = {
            "req-1": SimpleNamespace(sampling_params=sp),
            "req-2": SimpleNamespace(sampling_params=sp),
        }
        h.input_batch = _StubInputBatch(
            num_reqs=2,
            req_ids=["req-1", "req-2"],
            req_id_to_index={"req-1": 0, "req-2": 1},
            num_computed_tokens_cpu=np.array([9, 9, 0, 0, 0, 0, 0, 0]),
            num_prompt_tokens=np.array([10, 10, 0, 0, 0, 0, 0, 0]),
            request_prefill_steering_hash=np.array([100, 100, 0, 0, 0, 0, 0, 0]),
            request_decode_steering_hash=np.array([200, 200, 0, 0, 0, 0, 0, 0]),
        )

        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1, "req-2": 1})
        )

        mgr = h._sae_clamp_manager
        assert mgr is not None
        prefill_sae_hash = sp.prefill_sae_clamp_config_hash
        decode_sae_hash = sp.decode_sae_clamp_config_hash
        assert (prefill_sae_hash, "prefill") not in mgr.config_to_row
        assert mgr.config_to_row[(decode_sae_hash, "decode")] == 3
        assert mgr.config_refcounts[(decode_sae_hash, "decode")] == 2
        assert h._req_sae_phase == {"req-1": "decode", "req-2": "decode"}

    def test_decode_only_additive_transitions_from_no_active_shortcut(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        sp = SamplingParams(
            decode_steering_vectors={"post_block": {20: [1.0, 0.0, 0.0, 0.0]}}
        )
        h.requests = {"req-1": SimpleNamespace(sampling_params=sp)}
        # Phase tracking lives on _steering_reqs now (seeded by the sync
        # below); the request starts in prefill by its token counts.
        h._req_transition_scan_candidates.add("req-1")
        h.input_batch = _StubInputBatch(
            num_reqs=1,
            req_ids=["req-1"],
            req_id_to_index={"req-1": 0},
            num_computed_tokens_cpu=np.array([9, 0, 0, 0, 0, 0, 0, 0]),
            num_prompt_tokens=np.array([10, 0, 0, 0, 0, 0, 0, 0]),
            request_prefill_steering_hash=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            request_decode_steering_hash=np.array([200, 0, 0, 0, 0, 0, 0, 0]),
        )

        _sync_reqs_from_batch(h)
        h._update_steering_buffers(
            _StubSchedulerOutput(num_scheduled_tokens={"req-1": 1})
        )

        mgr = h._steering_manager
        assert mgr is not None
        assert mgr.config_to_row[(200, "decode")] == 3
        assert h._req_steering_phase == {"req-1": "decode"}
        assert h._req_transition_scan_candidates == set()

    def test_initial_sae_admission_failure_rolls_back_additive_row(
        self, monkeypatch
    ):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(
            steering_vectors={"post_block": {20: [1.0, 0.0, 0.0, 0.0]}},
            sae_clamp_specs=(spec,),
        )

        def fail_sae_admission(*_args, **_kwargs):
            raise RuntimeError("sae capacity full")

        monkeypatch.setattr(h, "_register_initial_sae_clamps", fail_sae_admission)

        with pytest.raises(RuntimeError, match="sae capacity full"):
            h._steering_register_request(
                "req-1",
                sampling_params=sp,
                prefill_hash=100,
                decode_hash=200,
                num_prompt_tokens=10,
                num_computed_tokens=0,
            )

        mgr = h._steering_manager
        assert mgr is not None
        assert mgr.config_to_row == {}
        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}

    def test_streaming_sae_refresh_failure_rolls_back_new_additive_row(
        self, monkeypatch
    ):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(
            steering_vectors={"post_block": {20: [1.0, 0.0, 0.0, 0.0]}},
            sae_clamp_specs=(spec,),
        )

        def fail_sae_register(*_args, **_kwargs):
            raise RuntimeError("sae refresh capacity full")

        assert h._sae_clamp_manager is not None
        monkeypatch.setattr(
            h._sae_clamp_manager, "register_clamp_spec", fail_sae_register
        )

        # Streaming re-adds route through the canonical registration path.
        with pytest.raises(RuntimeError, match="sae refresh capacity full"):
            h._steering_register_request(
                "req-1",
                sampling_params=sp,
                prefill_hash=100,
                decode_hash=200,
                num_prompt_tokens=10,
                num_computed_tokens=0,
            )

        mgr = h._steering_manager
        assert mgr is not None
        assert mgr.config_to_row == {}
        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}

    def test_streaming_sae_refresh_validates_before_releasing_old_row(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        old_spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        old_sp = SamplingParams(sae_clamp_specs=(old_spec,))
        h._register_initial_sae_clamps(
            "req-1",
            old_sp,
            prefill_hash=old_sp.prefill_steering_config_hash,
            decode_hash=old_sp.decode_steering_config_hash,
            is_prefilling=True,
        )
        old_sae_hash = old_sp.prefill_sae_clamp_config_hash
        assert h._sae_clamp_manager is not None
        assert h._sae_clamp_manager.config_to_row == {(old_sae_hash, "prefill"): 3}

        new_spec = SAEClampSpec(
            module_name="missing",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=7.0),)
                }
            },
        )
        new_sp = SamplingParams(sae_clamp_specs=(new_spec,))

        # The canonical registration path validates the incoming spec
        # BEFORE releasing the old instance's rows, so a rejected
        # streaming continuation leaves the old row intact.
        with pytest.raises(SteeringVectorError, match="unknown module 'missing'"):
            h._steering_register_request(
                "req-1",
                sampling_params=new_sp,
                prefill_hash=new_sp.prefill_steering_config_hash,
                decode_hash=new_sp.decode_steering_config_hash,
                num_prompt_tokens=10,
                num_computed_tokens=0,
            )

        assert h._sae_clamp_manager.config_to_row == {(old_sae_hash, "prefill"): 3}
        assert h._req_sae_phase == {"req-1": "prefill"}

    def test_resumption_sae_failure_rolls_back_new_additive_row(
        self, monkeypatch
    ):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4, max_sae_configs=1)
        h._steering_manager = SteeringManager(
            max_steering_configs=1,
            device=torch.device("cpu"),
        )
        h.register_steering_modules({"g": _sae_payload(clampable=(0,))})
        spec = SAEClampSpec(
            module_name="g",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )
        sp = SamplingParams(
            steering_vectors={"post_block": {20: [1.0, 0.0, 0.0, 0.0]}},
            sae_clamp_specs=(spec,),
        )
        req_state = SimpleNamespace(
            sampling_params=sp,
            num_prompt_tokens=10,
            prefill_steering_config_hash=100,
            decode_steering_config_hash=200,
        )
        # Preempted mid-decode; the canonical store tracks the phase.
        h._steering_reqs["req-1"] = _SteeringReqState(
            sampling_params=sp,
            prefill_hash=100,
            decode_hash=200,
            num_prompt_tokens=10,
            phase="decode",
        )

        def fail_sae_register(*_args, **_kwargs):
            raise RuntimeError("sae resumption capacity full")

        assert h._sae_clamp_manager is not None
        monkeypatch.setattr(
            h._sae_clamp_manager, "register_clamp_spec", fail_sae_register
        )

        with pytest.raises(RuntimeError, match="sae resumption capacity full"):
            h._reset_steering_for_resumption(
                "req-1", req_state, new_num_computed_tokens=0
            )

        mgr = h._steering_manager
        assert mgr is not None
        assert mgr.config_to_row == {}
        assert h._req_steering_phase == {}
        assert h._req_sae_phase == {}


class TestMultiHookSaeRegistration:
    """A manifest covering more than one hook on the same layer must
    register without raising.  ``sae_index`` is one shared buffer per
    layer, so attaching the second hook needs the helper to no-op."""

    def test_two_hooks_on_one_layer_attach_cleanly(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "pre_attn"), (20, "post_block")),
                    clampable=(0, 1),
                )
            },
        )
        site = h._steerable_layers_cache[20]
        assert sae_buffers_attached(site, SteeringHookPoint.PRE_ATTN)
        assert sae_buffers_attached(site, SteeringHookPoint.POST_BLOCK)
        # Only one sae_index buffer materialised on the layer.
        assert hasattr(site, "sae_index")

    def test_overlapping_sae_modules_rejected_worker_side(self):
        h = _E2EHarness(layer_indices=(20,), hidden_size=4)
        h.register_steering_modules({"a": _sae_payload(clampable=(0,))})

        with pytest.raises(SteeringVectorError, match="overlap"):
            h.register_steering_modules({"b": _sae_payload(clampable=(1,))})

        assert "a" in h._sae_module_registry
        assert "b" not in h._sae_module_registry
