# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-3b worker-mixin integration tests for the full-reconstruction path.

Mirrors :mod:`tests.v1.worker.test_sae_mixin_integration` for Phase-4
Stage-3b: register / unregister / attach weights / admission / release
exercised via the ``SteeringModelRunnerMixin`` with a synthetic
harness, no real model runner.

What this stage covers:

* Registry branching on ``kind="sae_full_reconstruction"`` attaches
  per-(layer, hook) full-reconstruction buffers, broadcast-replays
  cleanly, and detaches on unregister.
* ``attach_sae_full_recon_weights`` injects encoder + decoder weight
  + bias tensors into the right per-(layer, hook) buffer with shape
  validation.
* ``_register_initial_sae_full_recon`` honours ``is_prefilling`` and
  the hash==0 short-circuit; ``_release_sae_full_recon_for_request``
  releases under the recorded phase and is idempotent.
* The new full-reconstruction registry is **disjoint** from the delta
  registry: re-registering a name with a different kind drops the
  prior entry and detaches its buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn

from vllm import SamplingParams
from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEFullReconstructionSpec,
)
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_DECODER_BIAS_ATTR,
    HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ENCODER_BIAS_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_MODULE_NAME_ATTR,
    sae_full_recon_buffers_attached,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)
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
    """Bare decoder-layer stand-in with the additive table buffers attached.

    The full-reconstruction buffer attach path discovers compute dtype
    via the additive table buffers (existing source of truth), so
    each synthetic layer carries them too.
    """
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


class _Harness(SteeringModelRunnerMixin):
    """Mixin instance with the Stage-3b full-reconstruction state populated.

    Sets up both the delta and full-reconstruction state in parallel,
    so the kind-swap test can verify the registries stay disjoint.
    """

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
        # Both delta and full-reconstruction surfaces present.
        self._steering_module_registry: dict = {}
        self._sae_module_registry: dict = {}
        self._sae_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._req_steering_phase: dict = {}
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(max_sae_configs)
        self._steering_manager = None
        self._steering_rows_scratch = None
        self._steering_n_tokens_scratch = None
        self._steering_index_pinned = None
        self._sae_rows_scratch = None
        self._sae_index_pinned = None
        # Full-reconstruction surface — Stage 3b adds these.
        self._sae_fr_module_registry: dict = {}
        self._sae_fr_steerable_sites: dict = {}
        self._req_sae_fr_phase: dict = {}
        self._sae_fr_clamp_manager = SAEFullReconstructionManager(max_sae_configs)
        self._sae_fr_rows_scratch = None
        self._sae_fr_index_pinned = None


def _payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_mlp"),),
    clampable: tuple[int, ...] = (0, 1, 34),
    activation: str = "relu",
    activation_params: dict[str, float] | None = None,
    d_model: int = 4,
    d_sae: int = 64,
) -> dict:
    return {
        "kind": "sae_full_reconstruction",
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


class TestRegisterAttachesFullReconBuffers:
    def test_register_creates_per_site_buffers(self):
        h = _Harness()
        h.register_steering_modules(
            modules={"g": _payload(layers=((20, "post_mlp"), (21, "post_attn")))}
        )
        assert "g" in h._sae_fr_module_registry
        layer20 = h._steerable_layers_cache[20]
        layer21 = h._steerable_layers_cache[21]
        assert sae_full_recon_buffers_attached(layer20, SteeringHookPoint.POST_MLP)
        assert sae_full_recon_buffers_attached(layer21, SteeringHookPoint.POST_ATTN)
        # The owning module name is recorded as a python attribute.
        assert (
            getattr(layer20, HOOK_POINT_FR_MODULE_NAME_ATTR[SteeringHookPoint.POST_MLP])
            == "g"
        )

    def test_unregister_detaches(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        h.unregister_steering_modules(["g"])
        assert "g" not in h._sae_fr_module_registry
        layer = h._steerable_layers_cache[20]
        assert not sae_full_recon_buffers_attached(layer, SteeringHookPoint.POST_MLP)

    def test_replace_clears_old_full_recon_modules(self):
        h = _Harness()
        h.register_steering_modules(modules={"a": _payload()})
        h.register_steering_modules(modules={"b": _payload()}, replace=True)
        assert "a" not in h._sae_fr_module_registry
        assert "b" in h._sae_fr_module_registry

    def test_pp_filtered_layers_skipped(self):
        # Manifest declares layer 99 (not in this rank's owned layers);
        # registration must succeed but no buffer is attached for it.
        h = _Harness(layer_indices=(20,))
        h.register_steering_modules(
            modules={
                "g": _payload(
                    layers=((20, "post_mlp"), (99, "post_mlp")),
                )
            }
        )
        assert "g" in h._sae_fr_module_registry
        layer20 = h._steerable_layers_cache[20]
        assert sae_full_recon_buffers_attached(layer20, SteeringHookPoint.POST_MLP)
        assert ("g", 99, "post_mlp") not in h._sae_fr_steerable_sites


class TestKindSwapDisjoint:
    """A name re-registered as a different kind must drop the old entry.

    This guards against two registries holding the same name —
    request resolution would otherwise collide and the worker can't
    tell which path to dispatch.
    """

    def test_swap_full_recon_to_delta_drops_full_recon(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        assert "g" in h._sae_fr_module_registry
        # Re-register as delta.
        delta_payload = {
            "kind": "sae_delta",
            "sae_manifest": _payload()["sae_manifest"],
        }
        h.register_steering_modules(modules={"g": delta_payload})
        assert "g" not in h._sae_fr_module_registry
        assert "g" in h._sae_module_registry

    def test_swap_delta_to_full_recon_drops_delta(self):
        h = _Harness()
        delta_payload = {
            "kind": "sae_delta",
            "sae_manifest": _payload()["sae_manifest"],
        }
        h.register_steering_modules(modules={"g": delta_payload})
        assert "g" in h._sae_module_registry
        h.register_steering_modules(modules={"g": _payload()})
        assert "g" not in h._sae_module_registry
        assert "g" in h._sae_fr_module_registry


class TestAttachWeights:
    def test_copies_tensors_in_place(self):
        h = _Harness()
        h.register_steering_modules(
            modules={"g": _payload(d_model=4, d_sae=8, clampable=(0, 1, 2))}
        )
        layer = h._steerable_layers_cache[20]
        d_sae, d_model = 8, 4
        torch.manual_seed(0)
        enc_w = torch.randn(d_sae, d_model)
        enc_b = torch.randn(d_sae)
        dec_w = torch.randn(d_sae, d_model)
        dec_b = torch.randn(d_model)
        h.attach_sae_full_recon_weights(
            "g",
            {
                (20, "post_mlp"): {
                    "encoder_weight": enc_w,
                    "encoder_bias": enc_b,
                    "decoder_weight": dec_w,
                    "decoder_bias": dec_b,
                },
            },
        )
        assert torch.allclose(
            getattr(
                layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
            ),
            enc_w,
        )
        assert torch.allclose(
            getattr(layer, HOOK_POINT_FR_ENCODER_BIAS_ATTR[SteeringHookPoint.POST_MLP]),
            enc_b,
        )
        assert torch.allclose(
            getattr(
                layer, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[SteeringHookPoint.POST_MLP]
            ),
            dec_w,
        )
        assert torch.allclose(
            getattr(layer, HOOK_POINT_FR_DECODER_BIAS_ATTR[SteeringHookPoint.POST_MLP]),
            dec_b,
        )

    def test_rejects_unknown_module(self):
        h = _Harness()
        with pytest.raises(SteeringVectorError, match="not registered"):
            h.attach_sae_full_recon_weights("missing", {})

    def test_missing_decoder_bias_rejected(self):
        h = _Harness()
        h.register_steering_modules(
            modules={"g": _payload(d_model=4, d_sae=8, clampable=(0, 1, 2))}
        )
        with pytest.raises(SteeringVectorError, match="decoder_bias"):
            h.attach_sae_full_recon_weights(
                "g",
                {
                    (20, "post_mlp"): {
                        "encoder_weight": torch.zeros(8, 4),
                        "encoder_bias": torch.zeros(8),
                        "decoder_weight": torch.zeros(8, 4),
                        # decoder_bias deliberately absent
                    }
                },
            )

    def test_shape_mismatch_rejected(self):
        h = _Harness()
        h.register_steering_modules(
            modules={"g": _payload(d_model=4, d_sae=8, clampable=(0, 1, 2))}
        )
        with pytest.raises(SteeringVectorError, match="encoder_weight"):
            h.attach_sae_full_recon_weights(
                "g",
                {
                    (20, "post_mlp"): {
                        "encoder_weight": torch.zeros(7, 4),  # wrong d_sae
                        "encoder_bias": torch.zeros(8),
                        "decoder_weight": torch.zeros(8, 4),
                        "decoder_bias": torch.zeros(4),
                    }
                },
            )


class TestAdmissionAndRelease:
    def _spec(self) -> SAEFullReconstructionSpec:
        return SAEFullReconstructionSpec(
            module_name="g",
            clamps={
                "post_mlp": {
                    20: (SAEClampEntry(feature_idx=0, kind="absolute", value=5.0),)
                }
            },
        )

    def test_register_initial_prefill_admits_under_prefill_phase(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        sp = SamplingParams(sae_full_reconstruction_specs=[self._spec()])
        h._register_initial_sae_full_recon(
            "req-1",
            sp,
            prefill_hash=0xCAFE,
            decode_hash=0xDEAD,
            is_prefilling=True,
        )
        assert h._req_sae_fr_phase["req-1"] == "prefill"
        assert (0xCAFE, "prefill") in h._sae_fr_clamp_manager.config_to_row

    def test_register_initial_decode_only_when_not_prefilling(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        sp = SamplingParams(sae_full_reconstruction_specs=[self._spec()])
        h._register_initial_sae_full_recon(
            "req-1",
            sp,
            prefill_hash=0xCAFE,
            decode_hash=0xDEAD,
            is_prefilling=False,
        )
        assert h._req_sae_fr_phase["req-1"] == "decode"
        assert (0xDEAD, "decode") in h._sae_fr_clamp_manager.config_to_row
        # Prefill row must not be allocated.
        assert (0xCAFE, "prefill") not in h._sae_fr_clamp_manager.config_to_row

    def test_release_under_recorded_phase(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        sp = SamplingParams(sae_full_reconstruction_specs=[self._spec()])
        h._register_initial_sae_full_recon(
            "req-1",
            sp,
            prefill_hash=0xCAFE,
            decode_hash=0xDEAD,
            is_prefilling=True,
        )
        h._release_sae_full_recon_for_request("req-1", 0xCAFE, 0xDEAD)
        assert "req-1" not in h._req_sae_fr_phase
        assert (0xCAFE, "prefill") not in h._sae_fr_clamp_manager.config_to_row

    def test_release_unknown_request_is_no_op(self):
        h = _Harness()
        # Calling on a request that never registered must be silent
        # (production path calls this unconditionally on completion).
        h._release_sae_full_recon_for_request("never-was", 0xAB, 0xCD)

    def test_register_skips_when_no_specs(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        sp = SamplingParams()  # no sae_full_reconstruction_specs
        h._register_initial_sae_full_recon(
            "req-1", sp, prefill_hash=0xCAFE, decode_hash=0, is_prefilling=True
        )
        assert "req-1" not in h._req_sae_fr_phase

    def test_register_skips_when_hash_zero(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload()})
        sp = SamplingParams(sae_full_reconstruction_specs=[self._spec()])
        h._register_initial_sae_full_recon(
            "req-1", sp, prefill_hash=0, decode_hash=0, is_prefilling=True
        )
        # Hash 0 is the no-reconstruction sentinel; admitting it would
        # be a manager-side violation, so the helper short-circuits.
        assert "req-1" not in h._req_sae_fr_phase


class TestAssertSpecsCanBeApplied:
    def test_unknown_module_rejected(self):
        h = _Harness()
        sp = SamplingParams(
            sae_full_reconstruction_specs=[
                SAEFullReconstructionSpec(module_name="missing")
            ]
        )
        with pytest.raises(SteeringVectorError, match="unknown module"):
            h._assert_sae_full_recon_specs_can_be_applied(sp)

    def test_uncovered_site_rejected(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload(layers=((20, "post_mlp"),))})
        sp = SamplingParams(
            sae_full_reconstruction_specs=[
                SAEFullReconstructionSpec(
                    module_name="g",
                    clamps={
                        "post_attn": {  # not covered by manifest
                            20: (
                                SAEClampEntry(
                                    feature_idx=0, kind="absolute", value=1.0
                                ),
                            )
                        }
                    },
                )
            ]
        )
        with pytest.raises(SteeringVectorError, match="not declared"):
            h._assert_sae_full_recon_specs_can_be_applied(sp)

    def test_unclampable_feature_rejected(self):
        h = _Harness()
        h.register_steering_modules(modules={"g": _payload(clampable=(0, 1, 34))})
        sp = SamplingParams(
            sae_full_reconstruction_specs=[
                SAEFullReconstructionSpec(
                    module_name="g",
                    clamps={
                        "post_mlp": {
                            20: (
                                SAEClampEntry(
                                    feature_idx=99,  # not clampable
                                    kind="absolute",
                                    value=1.0,
                                ),
                            )
                        }
                    },
                )
            ]
        )
        with pytest.raises(SteeringVectorError, match="not in the module"):
            h._assert_sae_full_recon_specs_can_be_applied(sp)

    def test_no_specs_passes_silently(self):
        h = _Harness()
        h._assert_sae_full_recon_specs_can_be_applied(SamplingParams())
