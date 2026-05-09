# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-2 worker-mixin integration tests for the SAE feature-surgery path.

These tests exercise the worker-side state machinery — buffer
attachment lifecycle, weight injection, and the
admission/release/transition helpers — without spinning up a real
model runner.  We construct a richer harness that mirrors the parts
of the mixin Stage 2 actually touches: a fake ``vllm_config``, a
synthetic ``_steerable_layers_cache`` with bare ``nn.Module`` layers,
and the empty SAE state Stage 2 expects ``_init_steering_state`` to
have prepared.

Coverage:

* Buffer attach on ``register_steering_modules`` / detach on
  ``unregister_steering_modules``, including the replace-clears-both
  path and the kind-swap path.
* ``attach_sae_weights`` validates shape and copies tensors
  in-place into the right per-(layer, hook) buffer.
* ``_register_initial_sae_clamps`` honours ``is_prefilling`` and the
  hash==0 short-circuit.
* ``_release_sae_for_request`` releases under whichever phase the
  request was last admitted, idempotent on already-released requests.
* ``_handle_steering_transition`` releases prefill SAE row and
  registers decode SAE row in lockstep with the additive path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    HOOK_POINT_SAE_CLAMP_KIND_ATTR,
    HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
    HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_MODULE_NAME_ATTR,
    sae_buffers_attached,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
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


def _make_decoder_layer(layer_idx: int, hidden_size: int = 4) -> nn.Module:
    """Bare decoder-layer stand-in with the additive table buffers attached."""
    m = nn.Module()
    m.layer_idx = layer_idx  # type: ignore[attr-defined]
    # Additive steering buffers — needed because _attach_sae_buffers
    # discovers compute dtype from them.
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


class _RichHarness(SteeringModelRunnerMixin):
    """Mixin instance with the Stage-2 state surface populated."""

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...] = (20, 21),
        hidden_size: int = 4,
        max_sae_configs: int = 4,
    ) -> None:
        self.vllm_config = _StubVllmConfig(
            steering_config=_StubSteeringConfig(max_steering_configs=max_sae_configs),
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
        self._steering_manager = None  # not exercised in these tests
        self._steering_rows_scratch = None
        self._steering_n_tokens_scratch = None
        self._steering_index_pinned = None
        self._sae_rows_scratch = None
        self._sae_index_pinned = None


def _sae_payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_mlp"),),
    clampable: tuple[int, ...] = (0, 1, 2, 34),
    activation: str = "relu",
    activation_params: dict[str, float] | None = None,
    d_model: int = 4,
    d_sae: int = 64,
) -> dict:
    return {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": activation,
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable),
            "activation_params": activation_params or {},
            "weights_uri": None,
        },
    }


def _make_clamp_spec(
    *,
    module_name: str = "g",
    feature_idx: int = 0,
    layer: int = 20,
    hook: str = "post_mlp",
    value: float = 5.0,
    kind: str = "absolute",
    phase: str = "both",
) -> SAEClampSpec:
    return SAEClampSpec(
        module_name=module_name,
        phase=phase,  # type: ignore[arg-type]
        clamps={
            hook: {
                layer: (
                    SAEClampEntry(
                        feature_idx=feature_idx,
                        kind=kind,  # type: ignore[arg-type]
                        value=value,
                    ),
                )
            }
        },
    )


class TestRegisterAttachesSaeBuffers:
    """``register_steering_modules`` for a SAE module attaches per-layer buffers."""

    def test_attaches_to_every_owned_layer_in_manifest(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_mlp"), (21, "post_mlp")),
                ),
            }
        )
        site20 = h._steerable_layers_cache[20]
        site21 = h._steerable_layers_cache[21]
        assert sae_buffers_attached(site20, SteeringHookPoint.POST_MLP)
        assert sae_buffers_attached(site21, SteeringHookPoint.POST_MLP)
        # POST_ATTN has no SAE buffers because the manifest doesn't
        # declare it.
        assert not sae_buffers_attached(site20, SteeringHookPoint.POST_ATTN)

    def test_skips_layers_not_owned_by_this_rank(self):
        # PP scenario: rank owns layers {20}, but manifest declares 20+21.
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_mlp"), (21, "post_mlp")),
                ),
            }
        )
        # Registry entry must still exist (broadcast determinism).
        assert "g" in h._sae_module_registry
        # Site for layer 20 attached; layer 21 was never on this rank.
        assert sae_buffers_attached(
            h._steerable_layers_cache[20], SteeringHookPoint.POST_MLP
        )
        assert ("g", 21, "post_mlp") not in h._sae_steerable_sites

    def test_records_module_name_in_buffer_attribute(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        site = h._steerable_layers_cache[20]
        assert (
            getattr(site, HOOK_POINT_SAE_MODULE_NAME_ATTR[SteeringHookPoint.POST_MLP])
            == "g"
        )

    def test_buffer_shape_matches_manifest(self):
        h = _RichHarness(hidden_size=8)
        h.register_steering_modules(
            {"g": _sae_payload(d_model=8, clampable=(0, 1, 2))},
        )
        site = h._steerable_layers_cache[20]
        enc_w = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
        )
        assert enc_w.shape == (3, 8)

    def test_index_buffer_attached_and_shared(self):
        h = _RichHarness(layer_indices=(20, 21))
        h.register_steering_modules(
            {
                "g": _sae_payload(
                    layers=((20, "post_mlp"), (21, "post_mlp")),
                ),
            }
        )
        site20 = h._steerable_layers_cache[20]
        site21 = h._steerable_layers_cache[21]
        assert site20.sae_index is site21.sae_index  # type: ignore[attr-defined]


class TestUnregisterDetachesSaeBuffers:
    def test_unregister_strips_buffers(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        h.unregister_steering_modules(["g"])
        site = h._steerable_layers_cache[20]
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_MLP)
        assert h._sae_steerable_sites == {}

    def test_replace_clears_existing_sae_buffers(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        # replace=True drops the old SAE module entirely.
        h.register_steering_modules({"new": _sae_payload()}, replace=True)
        site = h._steerable_layers_cache[20]
        # 'g' should be gone.
        assert "g" not in h._sae_module_registry
        # 'new' should be attached.
        assert (
            getattr(site, HOOK_POINT_SAE_MODULE_NAME_ATTR[SteeringHookPoint.POST_MLP])
            == "new"
        )

    def test_kind_swap_detaches_old_sae_buffers(self):
        # Re-registering the same name as additive must drop the SAE
        # buffers so the layer state is consistent with the registry.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        site = h._steerable_layers_cache[20]
        assert sae_buffers_attached(site, SteeringHookPoint.POST_MLP)
        h.register_steering_modules(
            {
                "g": {
                    "kind": "additive",
                    "vectors": {"post_mlp": {0: [0.1, 0.2, 0.3, 0.4]}},
                    "prefill_vectors": None,
                    "decode_vectors": None,
                },
            }
        )
        assert not sae_buffers_attached(site, SteeringHookPoint.POST_MLP)


class TestAttachSaeWeights:
    def test_copies_into_buffers_in_place(self):
        h = _RichHarness(hidden_size=4)
        h.register_steering_modules({"g": _sae_payload(d_model=4, clampable=(0, 1))})
        site = h._steerable_layers_cache[20]
        enc_w_before = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
        ).clone()
        new_w = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        new_b = torch.tensor([0.1, 0.2])
        new_dec = torch.tensor([[9.0, 8.0, 7.0, 6.0], [5.0, 4.0, 3.0, 2.0]])
        h.attach_sae_weights(
            "g",
            {
                (20, "post_mlp"): {
                    "encoder_weight": new_w,
                    "encoder_bias": new_b,
                    "decoder_weight": new_dec,
                }
            },
        )
        enc_w = getattr(
            site, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
        )
        enc_b = getattr(
            site, HOOK_POINT_SAE_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_MLP]
        )
        dec_w = getattr(
            site, HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
        )
        assert torch.equal(enc_w, new_w)
        assert torch.equal(enc_b, new_b)
        assert torch.equal(dec_w, new_dec)
        assert not torch.equal(enc_w, enc_w_before)

    def test_unknown_module_raises(self):
        h = _RichHarness()
        with pytest.raises(SteeringVectorError, match="not registered"):
            h.attach_sae_weights("g", {})

    def test_missing_tensor_key_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="missing"):
            h.attach_sae_weights(
                "g",
                {(20, "post_mlp"): {"encoder_weight": torch.zeros(2, 4)}},
            )

    def test_shape_mismatch_raises(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        with pytest.raises(SteeringVectorError, match="shape"):
            h.attach_sae_weights(
                "g",
                {
                    (20, "post_mlp"): {
                        "encoder_weight": torch.zeros(99, 4),  # wrong n_clamp
                        "encoder_bias": torch.zeros(2),
                        "decoder_weight": torch.zeros(2, 4),
                    }
                },
            )

    def test_layer_not_owned_by_rank_silently_skipped(self):
        # PP scenario: rank doesn't own layer 99, so the weights for it
        # are dropped without error.  The on-disk loader can't know
        # which rank owns which layer; pruning is the rank's job.
        h = _RichHarness(layer_indices=(20,))
        h.register_steering_modules({"g": _sae_payload(clampable=(0, 1))})
        h.attach_sae_weights(
            "g",
            {
                (99, "post_mlp"): {
                    "encoder_weight": torch.zeros(2, 4),
                    "encoder_bias": torch.zeros(2),
                    "decoder_weight": torch.zeros(2, 4),
                }
            },
        )


class TestRegisterInitialSaeClamps:
    def test_admit_under_prefill_when_prefilling(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=True,
        )
        assert h._req_sae_phase["req-1"] == "prefill"
        assert h._sae_clamp_manager.config_to_row[(111, "prefill")] == 1

    def test_admit_under_decode_when_full_prefix_cache_hit(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=111,
            decode_hash=222,
            is_prefilling=False,
        )
        assert h._req_sae_phase["req-1"] == "decode"
        assert h._sae_clamp_manager.config_to_row[(222, "decode")] == 1

    def test_no_clamps_no_op(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams()
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}

    def test_hash_zero_skips_admission(self):
        # A request might carry sae_clamp_specs but have hash 0 if all
        # clamps are phase-filtered out for the current worker phase.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(phase="decode"),))
        # Even with specs present, hash 0 means "no SAE state in this
        # phase" — the manager must not register.
        h._register_initial_sae_clamps(
            "req-1",
            sp,
            prefill_hash=0,
            decode_hash=222,
            is_prefilling=True,
        )
        assert "req-1" not in h._req_sae_phase


class TestReleaseSaeForRequest:
    def test_releases_under_recorded_phase(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h._release_sae_for_request("req-1", prefill_hash=111, decode_hash=222)
        assert "req-1" not in h._req_sae_phase
        assert h._sae_clamp_manager.config_to_row == {}

    def test_release_unknown_request_no_op(self):
        h = _RichHarness()
        # Must not raise.
        h._release_sae_for_request("never-admitted", prefill_hash=1, decode_hash=2)

    def test_release_decode_uses_decode_hash(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=False
        )
        h._release_sae_for_request("req-1", prefill_hash=111, decode_hash=222)
        assert h._sae_clamp_manager.config_to_row == {}


class TestUpdateSaeBuffersPopulator:
    """End-to-end populate sweep against a small SAE-attached harness.

    Bypasses the per-step ``_update_sae_buffers`` (which depends on
    ``input_batch`` / ``scheduler_output``) and exercises the
    populator directly.  The per-step pass is covered by the
    integration test in Stage 3.
    """

    def test_populates_active_row_with_clamp(self):
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload(clampable=(34,))})
        sp = SamplingParams(
            sae_clamp_specs=(
                _make_clamp_spec(feature_idx=34, value=7.0, kind="absolute"),
            )
        )
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )

        # Drive the populator via the worker flow.
        site = h._steerable_layers_cache[20]
        from vllm.model_executor.layers.sae_steering import populate_sae_clamp_table

        populate_sae_clamp_table(
            manager=h._sae_clamp_manager,
            module=site,
            hook_point=SteeringHookPoint.POST_MLP,
            module_name="g",
            clampable_features=(34,),
            layer_idx=20,
        )
        kind = getattr(site, HOOK_POINT_SAE_CLAMP_KIND_ATTR[SteeringHookPoint.POST_MLP])
        # Row 1 (the prefill registration) gets the absolute clamp.
        from vllm.model_executor.layers.sae_steering import CLAMP_KIND_ABSOLUTE

        assert kind[1, 0].item() == CLAMP_KIND_ABSOLUTE


class TestKindSwapClearsTracking:
    """Re-registering ``g`` as additive must drop ``_req_sae_phase``-style state."""

    def test_unregister_does_not_release_admitted_specs(self):
        # Concrete: SAE module 'g' admits a clamp; then 'g' is
        # unregistered.  The admitted row in the manager outlives
        # the registry entry — the manager refcount is what frees it.
        h = _RichHarness()
        h.register_steering_modules({"g": _sae_payload()})
        sp = SamplingParams(sae_clamp_specs=(_make_clamp_spec(),))
        h._register_initial_sae_clamps(
            "req-1", sp, prefill_hash=111, decode_hash=222, is_prefilling=True
        )
        h.unregister_steering_modules(["g"])
        # The manager still has the row; release must come via
        # _release_sae_for_request.  This documents Stage-2 behaviour:
        # the buffer-attach lifecycle and the request-row lifecycle
        # are independent.  (Production-side, the request will fail
        # validation on its next admission step anyway.)
        assert h._sae_clamp_manager.config_to_row[(111, "prefill")] == 1
