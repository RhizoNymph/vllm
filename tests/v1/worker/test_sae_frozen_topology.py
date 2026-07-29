# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen-topology (compiled-engine) tests for SAE buffer pre-allocation.

Covers the compile-safety contract added for compiled serving:

* ``_preallocate_sae_topology`` attaches zero-filled delta/FR buffers
  for startup-declared modules (owned layers only) plus spare slots.
* Under frozen topology, a weight-bearing re-registration of a
  declared module reuses the pre-allocated buffers in place (tensor
  identity preserved — what a captured graph requires).
* The refresh-vs-reject matrix: shape/activation/site drift and
  undeclared modules raise ``SteeringVectorError`` before registry
  mutation; eligible undeclared delta modules claim spare slots with
  zero-padded weights and release them back to the pool on
  unregister.
* Frozen detach deactivates in place (buffers zeroed, slot records
  kept); eager detach still deletes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
import torch.nn as nn

from vllm.config.steering import SAEModuleTopology
from vllm.exceptions import SteeringVectorError
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ROW_ACTIVE_ATTR,
    sae_full_recon_buffers_attached,
)
from vllm.model_executor.layers.sae_steering import (
    SAE_SPARE_NAME_PREFIX,
    get_sae_slot_state,
    sae_buffers_attached,
    sae_site_slots,
)
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


def _delta_topo(
    name: str = "g",
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_block"),),
    n_clamp: int = 2,
    d_sae: int = 64,
    activation: str = "relu",
) -> SAEModuleTopology:
    return SAEModuleTopology(
        name=name,
        kind="sae_delta",
        layers=tuple(sorted(layers)),
        d_model=4,
        d_sae=d_sae,
        n_clamp=n_clamp,
        activation=activation,
        activation_params={},
    )


def _fr_topo(
    name: str = "fr",
    *,
    layers: tuple[tuple[int, str], ...] = ((21, "post_block"),),
    n_clamp: int = 2,
    d_sae: int = 8,
) -> SAEModuleTopology:
    return SAEModuleTopology(
        name=name,
        kind="sae_full_reconstruction",
        layers=tuple(sorted(layers)),
        d_model=4,
        d_sae=d_sae,
        n_clamp=n_clamp,
        activation="relu",
        activation_params={},
    )


class _FrozenHarness(SteeringModelRunnerMixin):
    """Mixin instance with pre-allocation + frozen-topology state."""

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...] = (20, 21),
        topology: tuple[SAEModuleTopology, ...] = (),
        spare_sites: tuple[str, ...] = (),
        spare_features: int = 0,
        spares_per_site: int = 1,
        frozen: bool = True,
        hidden_size: int = 4,
        max_sae_configs: int = 4,
    ) -> None:
        self.vllm_config = _StubVllmConfig(
            steering_config=_StubSteeringConfig(
                max_steering_configs=max_sae_configs,
                sae_module_topology=list(topology),
                sae_spare_slot_sites=list(spare_sites),
                sae_spare_slots_per_site=spares_per_site,
                sae_spare_slot_features=spare_features,
            ),
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
        self._sae_fr_module_registry: dict = {}
        self._sae_fr_steerable_sites: dict = {}
        self._req_sae_phase: dict = {}
        self._req_sae_fr_phase: dict = {}
        self._steering_reqs: dict = {}
        self._req_transition_scan_candidates: set[str] = set()
        self._steering_index_dirty = False
        self._sae_clamp_manager = SAEClampManager(max_sae_configs)
        self._sae_fr_clamp_manager = SAEFullReconstructionManager(max_sae_configs)
        self._steering_manager = None
        self._steering_topology_frozen = frozen
        self._sae_declared_topology: dict = {}
        self._sae_spare_layers: dict = {}
        self.requests: dict = {}
        self._preallocate_sae_topology(self.vllm_config.steering_config)


def _delta_payload(
    *,
    layers: tuple[tuple[int, str], ...] = ((20, "post_block"),),
    clampable: tuple[int, ...] = (0, 1),
    activation: str = "relu",
    d_model: int = 4,
    d_sae: int = 64,
    weights: dict | None = None,
) -> dict:
    body = {
        "kind": "sae_delta",
        "sae_manifest": {
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": activation,
            "layers": [list(p) for p in layers],
            "clampable_features": list(clampable),
            "activation_params": {},
            "weights_uri": None,
        },
    }
    if weights is not None:
        body["sae_weights"] = weights
    return body


def _delta_weights(
    *,
    n_clamp: int = 2,
    d_model: int = 4,
    site: tuple[int, str] = (20, "post_block"),
    fill: float = 1.0,
) -> dict:
    return {
        site: {
            "encoder_weight": torch.full((n_clamp, d_model), fill),
            "encoder_bias": torch.full((n_clamp,), fill / 2),
            "decoder_weight": torch.full((n_clamp, d_model), fill * 2),
        }
    }


class TestPreallocation:
    def test_declared_delta_buffers_attach_zero_filled(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        site = h._steerable_layers_cache[20]
        assert sae_buffers_attached(site, POST_BLOCK)
        state = get_sae_slot_state(site, POST_BLOCK, "g")
        assert state is not None
        assert tuple(state.encoder_weight.shape) == (2, 4)
        assert not state.encoder_weight.any()
        assert not state.any_active.any()
        assert ("g", 20, "post_block") in h._sae_steerable_sites
        assert "g" in h._sae_declared_topology
        # Registry stays empty until the weight broadcast lands.
        assert "g" not in h._sae_module_registry

    def test_declared_fr_buffers_attach_zero_filled(self):
        h = _FrozenHarness(topology=(_fr_topo("fr"),))
        site = h._steerable_layers_cache[21]
        assert sae_full_recon_buffers_attached(site, POST_BLOCK)
        enc = getattr(site, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK])
        assert tuple(enc.shape) == (8, 4)
        assert not enc.any()
        assert not getattr(site, HOOK_POINT_FR_ROW_ACTIVE_ATTR[POST_BLOCK]).any()
        assert "fr" not in h._sae_fr_module_registry

    def test_unowned_layers_are_skipped(self):
        h = _FrozenHarness(
            layer_indices=(20,),
            topology=(
                _delta_topo("g", layers=((20, "post_block"), (99, "post_block"))),
            ),
        )
        assert ("g", 20, "post_block") in h._sae_steerable_sites
        assert ("g", 99, "post_block") not in h._sae_steerable_sites

    def test_spare_slots_allocated_unclaimed_jumprelu(self):
        h = _FrozenHarness(
            spare_sites=("20:post_block",),
            spare_features=3,
            spares_per_site=2,
        )
        site = h._steerable_layers_cache[20]
        slots = sae_site_slots(site, POST_BLOCK)
        assert len(slots) == 2
        for record in slots:
            assert record.spare
            assert record.module_name.startswith(SAE_SPARE_NAME_PREFIX)
            assert record.activation.value == "jumprelu"
        assert hasattr(site, "sae_index")

    def test_bad_spare_site_string_raises(self):
        with pytest.raises(SteeringVectorError, match="not a valid"):
            _FrozenHarness(spare_sites=("nonsense",), spare_features=3)


class TestFrozenWeightRefresh:
    def test_declared_module_registration_reuses_buffers_in_place(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        site = h._steerable_layers_cache[20]
        before = get_sae_slot_state(site, POST_BLOCK, "g")
        assert before is not None
        h.register_steering_modules({"g": _delta_payload(weights=_delta_weights())})
        after = get_sae_slot_state(site, POST_BLOCK, "g")
        assert after is not None
        assert after.encoder_weight is before.encoder_weight
        assert torch.equal(after.encoder_weight, torch.ones(2, 4))
        assert "g" in h._sae_module_registry

    def test_replace_broadcast_round_trip_preserves_buffer_identity(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        site = h._steerable_layers_cache[20]
        h.register_steering_modules({"g": _delta_payload(weights=_delta_weights())})
        buf = get_sae_slot_state(site, POST_BLOCK, "g").encoder_weight
        # Startup-style full-sync push: detach-all + re-add must reuse
        # the same tensors under frozen topology.
        h.register_steering_modules(
            {"g": _delta_payload(weights=_delta_weights(fill=3.0))},
            replace=True,
        )
        after = get_sae_slot_state(site, POST_BLOCK, "g")
        assert after.encoder_weight is buf
        assert torch.equal(after.encoder_weight, torch.full((2, 4), 3.0))


class TestFrozenRejectMatrix:
    def test_undeclared_module_without_spares_rejected(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        with pytest.raises(SteeringVectorError, match="not declared at startup"):
            h.register_steering_modules({"other": _delta_payload()})
        assert "other" not in h._sae_module_registry

    def test_declared_module_with_extra_site_rejected(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        with pytest.raises(SteeringVectorError, match="sites"):
            h.register_steering_modules(
                {"g": _delta_payload(layers=((20, "post_block"), (21, "post_block")))}
            )

    def test_declared_module_with_d_sae_drift_rejected(self):
        h = _FrozenHarness(topology=(_delta_topo("g", d_sae=64),))
        with pytest.raises(SteeringVectorError, match="d_sae"):
            h.register_steering_modules({"g": _delta_payload(d_sae=128)})

    def test_declared_module_with_activation_drift_rejected(self):
        h = _FrozenHarness(topology=(_delta_topo("g", activation="relu"),))
        with pytest.raises(SteeringVectorError, match="activation"):
            h.register_steering_modules({"g": _delta_payload(activation="jumprelu")})

    def test_declared_module_with_n_clamp_drift_rejected(self):
        h = _FrozenHarness(topology=(_delta_topo("g", n_clamp=2),))
        with pytest.raises(SteeringVectorError, match="n_clamp"):
            h.register_steering_modules({"g": _delta_payload(clampable=(0, 1, 2))})

    def test_undeclared_fr_module_rejected_even_with_spares(self):
        h = _FrozenHarness(spare_sites=("21:post_block",), spare_features=8)
        payload = {
            "kind": "sae_full_reconstruction",
            "sae_manifest": {
                "d_model": 4,
                "d_sae": 8,
                "activation": "relu",
                "layers": [[21, "post_block"]],
                "clampable_features": [0, 1],
                "activation_params": {},
                "weights_uri": None,
            },
        }
        with pytest.raises(SteeringVectorError, match="not declared at startup"):
            h.register_steering_modules({"fr2": payload})

    def test_topk_module_cannot_claim_spares(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=4)
        payload = _delta_payload(activation="topk")
        payload["sae_manifest"]["activation_params"] = {"k": 1.0}
        with pytest.raises(SteeringVectorError, match="relu/jumprelu"):
            h.register_steering_modules({"other": payload})

    def test_oversized_module_cannot_claim_spares(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=2)
        with pytest.raises(SteeringVectorError, match="reserve only 2"):
            h.register_steering_modules({"other": _delta_payload(clampable=(0, 1, 2))})

    def test_uncovered_site_cannot_claim_spares(self):
        h = _FrozenHarness(spare_sites=("21:post_block",), spare_features=4)
        with pytest.raises(SteeringVectorError, match="no spare slots reserved"):
            h.register_steering_modules({"other": _delta_payload()})


class TestSpareClaim:
    def test_eligible_module_claims_spare_and_pads_weights(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=4)
        site = h._steerable_layers_cache[20]
        spare_buf = get_sae_slot_state(
            site, POST_BLOCK, sae_site_slots(site, POST_BLOCK)[0].module_name
        ).encoder_weight
        h.register_steering_modules(
            {"other": _delta_payload(weights=_delta_weights(n_clamp=2))}
        )
        state = get_sae_slot_state(site, POST_BLOCK, "other")
        assert state is not None
        assert state.slot.spare
        # Same tensor as the pre-allocated spare, reserved width kept.
        assert state.encoder_weight is spare_buf
        assert tuple(state.encoder_weight.shape) == (4, 4)
        assert torch.equal(state.encoder_weight[:2], torch.ones(2, 4))
        assert not state.encoder_weight[2:].any()
        # Trace-time constant untouched: the slot stays JumpReLU.
        assert state.slot.activation.value == "jumprelu"

    def test_unregister_returns_spare_to_pool(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=4)
        site = h._steerable_layers_cache[20]
        h.register_steering_modules(
            {"other": _delta_payload(weights=_delta_weights(n_clamp=2))}
        )
        h.unregister_steering_modules(["other"])
        assert "other" not in h._sae_module_registry
        slots = sae_site_slots(site, POST_BLOCK)
        assert len(slots) == 1
        assert slots[0].module_name.startswith(SAE_SPARE_NAME_PREFIX)
        state = get_sae_slot_state(site, POST_BLOCK, slots[0].module_name)
        assert not state.encoder_weight.any()
        # Pool restored: another module can claim.
        h.register_steering_modules({"third": _delta_payload()})
        assert get_sae_slot_state(site, POST_BLOCK, "third") is not None

    def test_second_module_rejected_when_site_pool_exhausted(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=4)
        h.register_steering_modules({"a": _delta_payload()})
        with pytest.raises(SteeringVectorError, match="claimed"):
            h.register_steering_modules({"b": _delta_payload()})

    def test_reregistering_claimer_reuses_its_own_slot(self):
        h = _FrozenHarness(spare_sites=("20:post_block",), spare_features=4)
        site = h._steerable_layers_cache[20]
        h.register_steering_modules(
            {"other": _delta_payload(weights=_delta_weights(fill=1.0))}
        )
        h.register_steering_modules(
            {"other": _delta_payload(weights=_delta_weights(fill=5.0))}
        )
        state = get_sae_slot_state(site, POST_BLOCK, "other")
        assert torch.equal(state.encoder_weight[:2], torch.full((2, 4), 5.0))
        assert len(sae_site_slots(site, POST_BLOCK)) == 1


class TestFrozenDetach:
    def test_frozen_unregister_deactivates_in_place(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),))
        site = h._steerable_layers_cache[20]
        h.register_steering_modules({"g": _delta_payload(weights=_delta_weights())})
        buf = get_sae_slot_state(site, POST_BLOCK, "g").encoder_weight
        h.unregister_steering_modules(["g"])
        assert "g" not in h._sae_module_registry
        # Slot survives, zeroed, same tensor object.
        state = get_sae_slot_state(site, POST_BLOCK, "g")
        assert state is not None
        assert state.encoder_weight is buf
        assert not state.encoder_weight.any()
        # Declared name reclaims its slot.
        h.register_steering_modules(
            {"g": _delta_payload(weights=_delta_weights(fill=7.0))}
        )
        assert get_sae_slot_state(site, POST_BLOCK, "g").encoder_weight is buf
        assert torch.equal(buf, torch.full((2, 4), 7.0))

    def test_frozen_fr_unregister_deactivates_in_place(self):
        h = _FrozenHarness(topology=(_fr_topo("fr"),))
        site = h._steerable_layers_cache[21]
        payload = {
            "kind": "sae_full_reconstruction",
            "sae_manifest": {
                "d_model": 4,
                "d_sae": 8,
                "activation": "relu",
                "layers": [[21, "post_block"]],
                "clampable_features": [0, 1],
                "activation_params": {},
                "weights_uri": None,
            },
            "sae_weights": {
                (21, "post_block"): {
                    "encoder_weight": torch.ones(8, 4),
                    "encoder_bias": torch.zeros(8),
                    "decoder_weight": torch.ones(8, 4),
                    "decoder_bias": torch.zeros(4),
                }
            },
        }
        h.register_steering_modules({"fr": payload})
        enc = getattr(site, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK])
        assert enc.any()
        h.unregister_steering_modules(["fr"])
        assert sae_full_recon_buffers_attached(site, POST_BLOCK)
        assert getattr(site, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK]) is enc
        assert not enc.any()
        # Declared owner re-registers into the same site.
        h.register_steering_modules({"fr": payload})
        assert getattr(site, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[POST_BLOCK]) is enc
        assert enc.any()

    def test_eager_unregister_still_deletes(self):
        h = _FrozenHarness(topology=(_delta_topo("g"),), frozen=False)
        site = h._steerable_layers_cache[20]
        h.register_steering_modules({"g": _delta_payload(weights=_delta_weights())})
        h.unregister_steering_modules(["g"])
        assert not sae_buffers_attached(site, POST_BLOCK)

    def test_eager_register_after_prealloc_allows_topology_change(self):
        # Eager engines keep fully dynamic registration even when the
        # module was pre-allocated at init: the first weight-bearing
        # registration may reshape freely (delete + fresh attach).
        h = _FrozenHarness(topology=(_delta_topo("g", n_clamp=2),), frozen=False)
        site = h._steerable_layers_cache[20]
        h.register_steering_modules({"g": _delta_payload(clampable=(0, 1, 2))})
        state = get_sae_slot_state(site, POST_BLOCK, "g")
        assert tuple(state.encoder_weight.shape) == (3, 4)

    def test_eager_fr_register_after_prealloc_succeeds(self):
        h = _FrozenHarness(topology=(_fr_topo("fr"),), frozen=False)
        site = h._steerable_layers_cache[21]
        payload = {
            "kind": "sae_full_reconstruction",
            "sae_manifest": {
                "d_model": 4,
                "d_sae": 8,
                "activation": "relu",
                "layers": [[21, "post_block"]],
                "clampable_features": [0, 1],
                "activation_params": {},
                "weights_uri": None,
            },
        }
        h.register_steering_modules({"fr": payload})
        assert "fr" in h._sae_fr_module_registry
        assert sae_full_recon_buffers_attached(site, POST_BLOCK)
