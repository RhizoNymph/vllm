# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``populate_sae_full_recon_clamp_table``.

Mirrors :mod:`tests.model_executor.layers.test_sae_layer_dispatch`'s
populator coverage for the full-reconstruction path: rows in the
manager are projected into the per-(layer, hook) clamp tables, gated
by module-match, phase-match, and layer-match.  Pure-reconstruction
specs (empty clamps) leave the row's clamp tables zeroed but the row
itself is allocated, so the dispatch shim's
``recon_mask = (recon_index != 0)`` triggers reconstruction even
without modifications.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm.config.sae_steering_types import (
    SAEActivation,
    SAEClampEntry,
    SAEFullReconstructionSpec,
)
from vllm.model_executor.layers.sae_full_reconstruction import (
    HOOK_POINT_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR,
    HOOK_POINT_FR_CLAMP_VALUE_ATTR,
    populate_sae_full_recon_clamp_table,
    register_sae_full_recon_buffers,
)
from vllm.model_executor.layers.sae_steering import (
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.v1.worker.sae_full_reconstruction_manager import (
    SAEFullReconstructionManager,
)


def _attach(
    layer: nn.Module,
    *,
    hook: SteeringHookPoint,
    module_name: str,
    clampable_features: list[int],
    d_sae: int = 16,
    hidden_size: int = 4,
    max_recon_configs: int = 4,
) -> None:
    register_sae_full_recon_buffers(
        layer,
        hook_point=hook,
        module_name=module_name,
        activation=SAEActivation.RELU,
        activation_params={},
        d_sae=d_sae,
        n_clamp=len(clampable_features),
        hidden_size=hidden_size,
        max_recon_configs=max_recon_configs,
        clampable_features=torch.tensor(clampable_features, dtype=torch.int64),
        dtype=torch.float32,
    )


class TestPopulator:
    def test_populator_writes_clamp_to_correct_position(self):
        layer = nn.Module()
        layer.layer_idx = 20
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[3, 7, 11],
        )
        manager = SAEFullReconstructionManager(4)
        spec = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=7, kind="absolute", value=5.0),)
                }
            },
        )
        manager.register_recon_spec(0xCAFE, (spec,), "prefill")
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=(3, 7, 11),
            layer_idx=20,
        )
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        value_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_VALUE_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        # Position 1 in the clampable tuple is feature_idx=7.
        assert kind_table[1, 1].item() == CLAMP_KIND_ABSOLUTE
        assert value_table[1, 1].item() == pytest.approx(5.0)
        # Other positions stay zero.
        assert kind_table[1, 0].item() == 0
        assert kind_table[1, 2].item() == 0

    def test_pure_reconstruction_row_stays_zero(self):
        # Spec with empty clamps must allocate a row but leave the
        # clamp tables for that row zero — reconstruction still
        # happens because recon_mask = (recon_index != 0).
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0, 1],
        )
        manager = SAEFullReconstructionManager(4)
        bare = SAEFullReconstructionSpec(module_name="m")
        manager.register_recon_spec(0x1234, (bare,), "decode")
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=(0, 1),
            layer_idx=0,
        )
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(kind_table[1], torch.zeros(2, dtype=torch.int8))

    def test_other_module_specs_skipped(self):
        # Specs targeting a different module must not write to this site.
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m_a",
            clampable_features=[0, 1],
        )
        manager = SAEFullReconstructionManager(4)
        spec = SAEFullReconstructionSpec(
            module_name="m_b",  # different module
            clamps={
                "post_block": {
                    0: (SAEClampEntry(feature_idx=0, kind="absolute", value=99.0),)
                }
            },
        )
        manager.register_recon_spec(0x9999, (spec,), "prefill")
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m_a",
            clampable_features=(0, 1),
            layer_idx=0,
        )
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert torch.equal(kind_table[1], torch.zeros(2, dtype=torch.int8))

    def test_phase_filter_only_writes_matching_rows(self):
        # Two requests, two rows; one prefill, one decode.  When the
        # populator is invoked with worker_phase="prefill", only the
        # prefill row's content gets written; the decode row stays at
        # whatever it had previously (here, all zero).
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0],
        )
        manager = SAEFullReconstructionManager(4)
        prefill = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    0: (SAEClampEntry(feature_idx=0, kind="absolute", value=11.0),)
                }
            },
        )
        decode = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    0: (SAEClampEntry(feature_idx=0, kind="additive", value=22.0),)
                }
            },
        )
        prefill_row = manager.register_recon_spec(0x1, (prefill,), "prefill")
        decode_row = manager.register_recon_spec(0x2, (decode,), "decode")
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=(0,),
            layer_idx=0,
            worker_phase="prefill",
        )
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        value_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_VALUE_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert kind_table[prefill_row, 0].item() == CLAMP_KIND_ABSOLUTE
        assert value_table[prefill_row, 0].item() == pytest.approx(11.0)
        # Decode row was filtered out → still zero.
        assert kind_table[decode_row, 0].item() == 0
        assert value_table[decode_row, 0].item() == pytest.approx(0.0)

    def test_row_zero_always_zeroed(self):
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0],
        )
        # Manually corrupt row 0 — populator must defensively reset it.
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        kind_table[0, 0] = CLAMP_KIND_ABSOLUTE
        manager = SAEFullReconstructionManager(4)
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=(0,),
            layer_idx=0,
        )
        assert kind_table[0, 0].item() == 0

    def test_unknown_feature_idx_raises(self):
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0, 1],
        )
        manager = SAEFullReconstructionManager(4)
        spec = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    0: (
                        SAEClampEntry(
                            feature_idx=99,  # not in clampable_features
                            kind="absolute",
                            value=1.0,
                        ),
                    )
                }
            },
        )
        manager.register_recon_spec(0xAA, (spec,), "prefill")
        with pytest.raises(ValueError, match="not in clampable_features"):
            populate_sae_full_recon_clamp_table(
                manager=manager,
                module=layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="m",
                clampable_features=(0, 1),
                layer_idx=0,
            )

    def test_invalid_worker_phase_rejected(self):
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0],
        )
        manager = SAEFullReconstructionManager(4)
        with pytest.raises(ValueError, match="worker_phase"):
            populate_sae_full_recon_clamp_table(
                manager=manager,
                module=layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="m",
                clampable_features=(0,),
                layer_idx=0,
                worker_phase="bogus",
            )

    def test_clampable_features_length_mismatch_rejected(self):
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0, 1],
        )
        manager = SAEFullReconstructionManager(4)
        with pytest.raises(ValueError, match="clampable_features length"):
            populate_sae_full_recon_clamp_table(
                manager=manager,
                module=layer,
                hook_point=SteeringHookPoint.POST_BLOCK,
                module_name="m",
                clampable_features=(0, 1, 2),  # wrong length
                layer_idx=0,
            )

    def test_only_if_active_passed_through(self):
        layer = nn.Module()
        layer.layer_idx = 0
        _attach(
            layer,
            hook=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=[0],
        )
        manager = SAEFullReconstructionManager(4)
        spec = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    0: (
                        SAEClampEntry(
                            feature_idx=0,
                            kind="additive",
                            value=3.0,
                            only_if_active=True,
                        ),
                    )
                }
            },
        )
        row = manager.register_recon_spec(0xBB, (spec,), "prefill")
        populate_sae_full_recon_clamp_table(
            manager=manager,
            module=layer,
            hook_point=SteeringHookPoint.POST_BLOCK,
            module_name="m",
            clampable_features=(0,),
            layer_idx=0,
        )
        only_table = getattr(
            layer,
            HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR[SteeringHookPoint.POST_BLOCK],
        )
        kind_table = getattr(
            layer, HOOK_POINT_FR_CLAMP_KIND_ATTR[SteeringHookPoint.POST_BLOCK]
        )
        assert kind_table[row, 0].item() == CLAMP_KIND_ADDITIVE
        assert only_table[row, 0].item() is True
