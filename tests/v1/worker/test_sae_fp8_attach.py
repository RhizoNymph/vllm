# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker attach-path tests for fp8 SAE weight storage.

Quantization happens worker-side at weight-attach time: weights arrive
over the wire as bf16/fp32 and ``attach_sae_weights`` /
``attach_sae_full_recon_weights`` quantize on copy when the slot is
fp8.  Covers scale correctness, refresh re-quantization, deactivate
zeroing, snapshot/rollback round-trips, frozen-topology storage-dtype
mismatch rejection, and spare-slot ineligibility of fp8 modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn

from vllm.config.steering import SAEModuleTopology
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.sae_fp8 import (
    FP8_E4M3_MAX,
    FP8_STORAGE_DTYPE,
    dequantize_fp8_rowwise,
)
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_DECODER_SCALE_ATTR,
    HOOK_POINT_FR_ENCODER_SCALE_ATTR,
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
)
from vllm.model_executor.layers.sae_steering import get_sae_slot_state
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    register_steering_buffers,
)
from vllm.v1.worker.sae_clamp_manager import SAEClampManager
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)
from vllm.v1.worker.steering_model_runner_mixin import SteeringModelRunnerMixin

POST_BLOCK = SteeringHookPoint.POST_BLOCK


@dataclass
class _StubModelConfig:
    dtype: torch.dtype = torch.float32

    def get_hidden_size(self) -> int:
        return 4


@dataclass
class _StubSchedulerConfig:
    max_num_batched_tokens: int = 16
    max_num_seqs: int = 8


@dataclass
class _StubSteeringConfig:
    max_steering_configs: int = 4
    sae_module_topology: list = field(default_factory=list)
    sae_spare_slot_sites: list = field(default_factory=list)
    sae_spare_slots_per_site: int = 1
    sae_spare_slot_features: int = 0


@dataclass
class _StubVllmConfig:
    model_config: _StubModelConfig = field(default_factory=_StubModelConfig)
    scheduler_config: _StubSchedulerConfig = field(default_factory=_StubSchedulerConfig)
    steering_config: _StubSteeringConfig = field(default_factory=_StubSteeringConfig)


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


def _fp8_delta_topo(name: str = "g") -> SAEModuleTopology:
    return SAEModuleTopology(
        name=name,
        kind="sae_delta",
        layers=((20, "post_block"),),
        d_model=4,
        d_sae=64,
        n_clamp=2,
        activation="relu",
        activation_params={},
        storage_dtype="fp8_e4m3",
    )


def _fp8_fr_topo(name: str = "fr") -> SAEModuleTopology:
    return SAEModuleTopology(
        name=name,
        kind="sae_full_reconstruction",
        layers=((21, "post_block"),),
        d_model=4,
        d_sae=8,
        n_clamp=2,
        activation="relu",
        activation_params={},
        storage_dtype="fp8_e4m3",
    )


class _Harness(SteeringModelRunnerMixin):
    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...] = (20, 21),
        topology: tuple[SAEModuleTopology, ...] = (),
        spare_sites: tuple[str, ...] = (),
        spare_features: int = 0,
        frozen: bool = False,
    ) -> None:
        self.vllm_config = _StubVllmConfig(
            steering_config=_StubSteeringConfig(
                sae_module_topology=list(topology),
                sae_spare_slot_sites=list(spare_sites),
                sae_spare_slot_features=spare_features,
            ),
        )
        self._steerable_layers_cache = {
            idx: _make_decoder_layer(idx) for idx in layer_indices
        }
        self._locally_owned_layers = frozenset(layer_indices)
        self._steering_module_registry: dict = {}
        self._steering_module_resolved_cache: dict = {}
        self._steering_module_pinned_rows: dict = {}
        self._sae_module_registry: dict = {}
        self._sae_steerable_sites: dict = {}
        self._sae_fr_module_registry: dict = {}
        self._sae_fr_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._req_sae_fr_phase: dict = {}
        self._steering_reqs: dict = {}
        self._req_transition_scan_candidates: set[str] = set()
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(4)
        self._sae_fr_clamp_manager = SAEFullReconstructionManager(4)
        self._steering_manager = None
        self._steering_topology_frozen = frozen
        self._sae_declared_topology: dict = {}
        self._sae_spare_layers: dict = {}
        self.requests: dict = {}
        self._preallocate_sae_topology(self.vllm_config.steering_config)


def _fp8_delta_payload(weights: dict | None = None) -> dict:
    body = {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": 4,
            "d_sae": 64,
            "activation": "relu",
            "layers": [[20, "post_block"]],
            "clampable_features": [0, 1],
            "activation_params": {},
            "weights_uri": None,
            "storage_dtype": "fp8_e4m3",
        },
    }
    if weights is not None:
        body["sae_weights"] = weights
    return body


def _delta_weights(enc: torch.Tensor, dec: torch.Tensor) -> dict:
    return {
        (20, "post_block"): {
            "encoder_weight": enc,
            "encoder_bias": torch.zeros(enc.shape[0]),
            "decoder_weight": dec,
        }
    }


class TestFp8AttachDelta:
    def test_preallocated_buffers_are_fp8(self):
        h = _Harness(topology=(_fp8_delta_topo(),))
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        assert state is not None
        assert state.encoder_weight.dtype == FP8_STORAGE_DTYPE
        assert state.decoder_weight.dtype == FP8_STORAGE_DTYPE
        assert state.encoder_scale.dtype == torch.float32

    def test_attach_quantizes_and_writes_scales(self):
        h = _Harness(topology=(_fp8_delta_topo(),))
        enc = torch.tensor([[1.0, -2.0, 0.5, 0.25], [0.0, 0.0, 0.0, 0.0]])
        dec = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.1, 0.2, -0.3, 0.4]])
        h.register_steering_modules(
            {"g": _fp8_delta_payload(weights=_delta_weights(enc, dec))}
        )
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        assert state.encoder_scale[0].item() == pytest.approx(2.0 / FP8_E4M3_MAX)
        assert state.decoder_scale[0].item() == pytest.approx(4.0 / FP8_E4M3_MAX)
        # Zero row: exact-zero dequant, positive scale (no div hazard).
        assert state.encoder_scale[1].item() > 0
        deq_enc = dequantize_fp8_rowwise(state.encoder_weight, state.encoder_scale)
        torch.testing.assert_close(deq_enc, enc, rtol=0.07, atol=1e-6)
        assert bool((deq_enc[1] == 0).all())
        deq_dec = dequantize_fp8_rowwise(state.decoder_weight, state.decoder_scale)
        torch.testing.assert_close(deq_dec, dec, rtol=0.07, atol=1e-6)

    def test_refresh_requantizes(self):
        h = _Harness(topology=(_fp8_delta_topo(),), frozen=True)
        enc1 = torch.full((2, 4), 1.0)
        dec1 = torch.full((2, 4), 2.0)
        h.register_steering_modules(
            {"g": _fp8_delta_payload(weights=_delta_weights(enc1, dec1))}
        )
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        buf_before = state.encoder_weight
        scale_before = state.encoder_scale.clone()
        enc2 = torch.full((2, 4), 3.0)
        dec2 = torch.full((2, 4), 6.0)
        h.register_steering_modules(
            {"g": _fp8_delta_payload(weights=_delta_weights(enc2, dec2))}
        )
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        assert state.encoder_weight is buf_before  # in-place refresh
        assert not torch.equal(state.encoder_scale, scale_before)
        deq = dequantize_fp8_rowwise(state.encoder_weight, state.encoder_scale)
        torch.testing.assert_close(deq, enc2, rtol=0.07, atol=1e-6)

    def test_frozen_unregister_zeroes_scales(self):
        h = _Harness(topology=(_fp8_delta_topo(),), frozen=True)
        enc = torch.full((2, 4), 1.0)
        dec = torch.full((2, 4), 2.0)
        h.register_steering_modules(
            {"g": _fp8_delta_payload(weights=_delta_weights(enc, dec))}
        )
        layer = h._steerable_layers_cache[20]
        h.unregister_steering_modules(["g"])
        state = get_sae_slot_state(layer, POST_BLOCK, "g")
        assert state is not None  # frozen: deactivated, not deleted
        assert not state.encoder_scale.any()
        assert not state.decoder_scale.any()
        deq = dequantize_fp8_rowwise(state.encoder_weight, state.encoder_scale)
        assert bool((deq == 0).all())

    def test_snapshot_restores_dequantized_weights(self):
        h = _Harness(topology=(_fp8_delta_topo(),))
        enc = torch.randn(2, 4)
        dec = torch.randn(2, 4)
        h.register_steering_modules(
            {"g": _fp8_delta_payload(weights=_delta_weights(enc, dec))}
        )
        snap = h._snapshot_sae_weights("g")
        site = snap[(20, "post_block")]
        # Snapshot carries dequantized weights so a restore re-quantizes.
        assert site["encoder_weight"].dtype == torch.float32
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        torch.testing.assert_close(
            site["encoder_weight"],
            dequantize_fp8_rowwise(state.encoder_weight, state.encoder_scale),
        )
        # Restore round-trip is stable (attach re-quantizes to same values).
        q_before = state.encoder_weight.clone()
        scale_before = state.encoder_scale.clone()
        h.attach_sae_weights("g", snap)
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        torch.testing.assert_close(
            state.encoder_weight.to(torch.float32), q_before.to(torch.float32)
        )
        torch.testing.assert_close(state.encoder_scale, scale_before)

    def test_wire_packed_bf16_weights_quantize(self):
        # bf16 packed wire form (the Rust frontend / broadcast path).
        h = _Harness(topology=(_fp8_delta_topo(),))
        enc = torch.randn(2, 4).to(torch.bfloat16)
        dec = torch.randn(2, 4).to(torch.bfloat16)

        def _pack(t: torch.Tensor) -> dict:
            return {
                "dtype": "bfloat16",
                "shape": list(t.shape),
                "data": t.contiguous().view(torch.uint8).numpy().tobytes(),
            }

        weights = {
            "20:post_block": {
                "encoder_weight": _pack(enc),
                "encoder_bias": {
                    "dtype": "float32",
                    "shape": [2],
                    "data": torch.zeros(2).numpy().tobytes(),
                },
                "decoder_weight": _pack(dec),
            }
        }
        h.register_steering_modules({"g": _fp8_delta_payload(weights=weights)})
        state = get_sae_slot_state(h._steerable_layers_cache[20], POST_BLOCK, "g")
        deq = dequantize_fp8_rowwise(state.encoder_weight, state.encoder_scale)
        torch.testing.assert_close(deq, enc.to(torch.float32), rtol=0.07, atol=1e-3)


class TestFp8AttachFR:
    def _fr_payload(self, weights: dict | None = None) -> dict:
        body = {
            "kind": "sae_full_reconstruction",
            "sae_manifest": {
                "d_model": 4,
                "d_sae": 8,
                "activation": "relu",
                "layers": [[21, "post_block"]],
                "clampable_features": [0, 1],
                "activation_params": {},
                "weights_uri": None,
                "storage_dtype": "fp8_e4m3",
            },
        }
        if weights is not None:
            body["sae_weights"] = weights
        return body

    def _fr_weights(self, enc: torch.Tensor, dec: torch.Tensor) -> dict:
        return {
            (21, "post_block"): {
                "encoder_weight": enc,
                "encoder_bias": torch.zeros(8),
                "decoder_weight": dec,
                "decoder_bias": torch.zeros(4),
            }
        }

    def test_attach_quantizes_full_matrices(self):
        h = _Harness(topology=(_fp8_fr_topo(),))
        enc = torch.randn(8, 4)
        dec = torch.randn(8, 4)
        h.register_steering_modules(
            {"fr": self._fr_payload(weights=self._fr_weights(enc, dec))}
        )
        layer = h._steerable_layers_cache[21]
        w = getattr(layer, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK])
        assert w.dtype == FP8_STORAGE_DTYPE
        s_enc = getattr(layer, HOOK_POINT_FR_ENCODER_SCALE_ATTR[POST_BLOCK])
        s_dec = getattr(layer, HOOK_POINT_FR_DECODER_SCALE_ATTR[POST_BLOCK])
        torch.testing.assert_close(
            dequantize_fp8_rowwise(w, s_enc), enc, rtol=0.07, atol=1e-6
        )
        assert s_dec.dtype == torch.float32

    def test_snapshot_dequantizes(self):
        h = _Harness(topology=(_fp8_fr_topo(),))
        enc = torch.randn(8, 4)
        dec = torch.randn(8, 4)
        h.register_steering_modules(
            {"fr": self._fr_payload(weights=self._fr_weights(enc, dec))}
        )
        snap = h._snapshot_sae_full_recon_weights("fr")
        site = snap[(21, "post_block")]
        assert site["encoder_weight"].dtype == torch.float32
        torch.testing.assert_close(site["encoder_weight"], enc, rtol=0.07, atol=1e-6)
        # Restore path re-quantizes without error.
        h.attach_sae_full_recon_weights("fr", snap)


class TestFrozenTopologyFp8:
    def test_storage_dtype_drift_is_rejected(self):
        h = _Harness(topology=(_fp8_delta_topo(),), frozen=True)
        payload = _fp8_delta_payload()
        payload["sae_manifest"]["storage_dtype"] = "auto"
        with pytest.raises(SteeringVectorError, match="storage_dtype"):
            h.register_steering_modules({"g": payload})

    def test_fp8_module_cannot_claim_spare_slots(self):
        h = _Harness(
            topology=(),
            spare_sites=("20:post_block",),
            spare_features=4,
            frozen=True,
        )
        with pytest.raises(
            SteeringVectorError, match="fp8 modules must be declared at startup"
        ):
            h.register_steering_modules({"g": _fp8_delta_payload()})

    def test_auto_module_still_claims_spares(self):
        h = _Harness(
            topology=(),
            spare_sites=("20:post_block",),
            spare_features=4,
            frozen=True,
        )
        payload = _fp8_delta_payload()
        payload["sae_manifest"]["storage_dtype"] = "auto"
        h.register_steering_modules({"g": payload})
        assert "g" in h._sae_module_registry
