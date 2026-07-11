# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SAEFullReconstructionSpec type and its hash + coerce.

The full-reconstruction spec is structurally similar to
:class:`SAEClampSpec` but allows empty ``clamps`` (pure reconstruction
is a meaningful op for this kind).  These tests cover validation,
coercion from JSON shapes, and hash determinism with a distinct
domain separator from the delta path.
"""

from __future__ import annotations

import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
    SAEFullReconstructionSpec,
    SteeringModuleKind,
    coerce_sae_full_reconstruction_specs,
    hash_sae_clamp_specs,
    hash_sae_full_reconstruction_specs,
)


class TestEnumExtended:
    def test_full_reconstruction_kind_present(self):
        assert SteeringModuleKind.SAE_FULL_RECONSTRUCTION.value == (
            "sae_full_reconstruction"
        )

    def test_kinds_are_distinct(self):
        kinds = {k.value for k in SteeringModuleKind}
        assert "additive" in kinds
        assert "sae_delta" in kinds
        assert "sae_full_reconstruction" in kinds


class TestSpecValidation:
    def test_minimal_spec_with_no_clamps_is_valid(self):
        # Pure reconstruction — empty clamps must be accepted.
        spec = SAEFullReconstructionSpec(module_name="m")
        assert spec.module_name == "m"
        assert spec.clamps == {}
        assert spec.phase == "both"

    def test_spec_with_clamps_is_valid(self):
        spec = SAEFullReconstructionSpec(
            module_name="m",
            clamps={
                "post_block": {
                    20: (SAEClampEntry(feature_idx=3, kind="absolute", value=5.0),)
                }
            },
        )
        assert "post_block" in spec.clamps
        assert 20 in spec.clamps["post_block"]

    def test_empty_module_name_rejected(self):
        with pytest.raises(ValueError, match="module_name"):
            SAEFullReconstructionSpec(module_name="")

    def test_invalid_phase_rejected(self):
        with pytest.raises(ValueError, match="phase"):
            SAEFullReconstructionSpec(module_name="m", phase="bogus")  # type: ignore[arg-type]

    def test_invalid_hook_in_clamps_rejected(self):
        with pytest.raises(ValueError, match="not a valid hook"):
            SAEFullReconstructionSpec(
                module_name="m",
                clamps={
                    "bogus": {
                        0: (SAEClampEntry(feature_idx=0, kind="absolute", value=1.0),)
                    }
                },
            )

    def test_empty_layer_map_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            SAEFullReconstructionSpec(
                module_name="m",
                clamps={"post_block": {}},
            )

    def test_empty_entries_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            SAEFullReconstructionSpec(
                module_name="m",
                clamps={"post_block": {0: ()}},
            )

    def test_duplicate_feature_idx_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            SAEFullReconstructionSpec(
                module_name="m",
                clamps={
                    "post_block": {
                        0: (
                            SAEClampEntry(feature_idx=3, kind="absolute", value=1.0),
                            SAEClampEntry(feature_idx=3, kind="additive", value=2.0),
                        )
                    }
                },
            )

    def test_negative_layer_idx_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            SAEFullReconstructionSpec(
                module_name="m",
                clamps={
                    "post_block": {
                        -1: (SAEClampEntry(feature_idx=0, kind="absolute", value=1.0),)
                    }
                },
            )


class TestCoercion:
    def test_none_returns_none(self):
        assert coerce_sae_full_reconstruction_specs(None) is None

    def test_empty_list_returns_none(self):
        assert coerce_sae_full_reconstruction_specs([]) is None

    def test_dict_payload_round_trips(self):
        raw = [
            {
                "module_name": "m",
                "phase": "both",
                "clamps": {
                    "post_block": {
                        "20": [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                    }
                },
            }
        ]
        out = coerce_sae_full_reconstruction_specs(raw)
        assert out is not None
        assert len(out) == 1
        spec = out[0]
        # Layer-key string was coerced to int.
        assert 20 in spec.clamps["post_block"]

    def test_already_typed_passthrough(self):
        spec = SAEFullReconstructionSpec(module_name="m")
        out = coerce_sae_full_reconstruction_specs([spec])
        assert out == (spec,)

    def test_clamps_field_optional(self):
        # Pure-reconstruction payload with no clamps.
        out = coerce_sae_full_reconstruction_specs([{"module_name": "m"}])
        assert out is not None
        assert out[0].clamps == {}

    def test_invalid_top_level_type_rejected(self):
        with pytest.raises(ValueError, match="list of spec"):
            coerce_sae_full_reconstruction_specs("not a list")  # type: ignore[arg-type]

    def test_missing_module_name_rejected(self):
        with pytest.raises(ValueError, match="module_name"):
            coerce_sae_full_reconstruction_specs([{"phase": "both"}])


class TestHashDeterminism:
    def test_none_and_empty_hash_zero(self):
        assert hash_sae_full_reconstruction_specs(None) == 0
        assert hash_sae_full_reconstruction_specs(()) == 0

    def test_same_spec_same_hash(self):
        spec = (SAEFullReconstructionSpec(module_name="m"),)
        h1 = hash_sae_full_reconstruction_specs(spec)
        h2 = hash_sae_full_reconstruction_specs(spec)
        assert h1 == h2
        assert h1 != 0

    def test_different_module_names_different_hash(self):
        h1 = hash_sae_full_reconstruction_specs(
            (SAEFullReconstructionSpec(module_name="a"),)
        )
        h2 = hash_sae_full_reconstruction_specs(
            (SAEFullReconstructionSpec(module_name="b"),)
        )
        assert h1 != h2

    def test_different_phase_different_hash(self):
        h1 = hash_sae_full_reconstruction_specs(
            (SAEFullReconstructionSpec(module_name="m", phase="prefill"),)
        )
        h2 = hash_sae_full_reconstruction_specs(
            (SAEFullReconstructionSpec(module_name="m", phase="decode"),)
        )
        assert h1 != h2

    def test_different_clamps_different_hash(self):
        h1 = hash_sae_full_reconstruction_specs(
            (
                SAEFullReconstructionSpec(
                    module_name="m",
                    clamps={
                        "post_block": {
                            20: (
                                SAEClampEntry(
                                    feature_idx=3, kind="absolute", value=5.0
                                ),
                            )
                        }
                    },
                ),
            )
        )
        h2 = hash_sae_full_reconstruction_specs(
            (
                SAEFullReconstructionSpec(
                    module_name="m",
                    clamps={
                        "post_block": {
                            20: (
                                SAEClampEntry(
                                    feature_idx=3, kind="absolute", value=5.5
                                ),
                            )
                        }
                    },
                ),
            )
        )
        assert h1 != h2

    def test_does_not_collide_with_delta_hash_on_identical_clamps(self):
        # A delta SAEClampSpec and a full-reconstruction spec with
        # identical clamp content must hash differently — the two
        # paths produce different residual streams (perturbation vs
        # replacement) and must not share prefix-cache keys.
        clamps = {
            "post_block": {
                20: (SAEClampEntry(feature_idx=3, kind="absolute", value=5.0),)
            }
        }
        delta = (SAEClampSpec(module_name="m", clamps=clamps),)
        recon = (SAEFullReconstructionSpec(module_name="m", clamps=clamps),)
        assert hash_sae_clamp_specs(delta) != hash_sae_full_reconstruction_specs(recon)


class TestHashSteeringConfigComposition:
    def test_full_recon_hash_folds_into_steering_config(self):
        # A request that adds a full-reconstruction spec must produce
        # a different combined hash than a request without it.
        from vllm.config.steering_types import hash_steering_config

        h_no = hash_steering_config(None)
        h_recon = hash_steering_config(
            None,
            sae_full_reconstruction_specs=(SAEFullReconstructionSpec(module_name="m"),),
        )
        assert h_no == 0
        assert h_recon != 0
        assert h_recon != h_no

    def test_delta_and_recon_blocks_compose_independently(self):
        from vllm.config.steering_types import hash_steering_config

        delta_only = hash_steering_config(
            None,
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="m",
                    clamps={
                        "post_block": {
                            20: (
                                SAEClampEntry(
                                    feature_idx=3, kind="absolute", value=5.0
                                ),
                            )
                        }
                    },
                ),
            ),
        )
        recon_only = hash_steering_config(
            None,
            sae_full_reconstruction_specs=(SAEFullReconstructionSpec(module_name="m"),),
        )
        both = hash_steering_config(
            None,
            sae_clamp_specs=(
                SAEClampSpec(
                    module_name="m",
                    clamps={
                        "post_block": {
                            20: (
                                SAEClampEntry(
                                    feature_idx=3, kind="absolute", value=5.0
                                ),
                            )
                        }
                    },
                ),
            ),
            sae_full_reconstruction_specs=(SAEFullReconstructionSpec(module_name="m"),),
        )
        assert delta_only != recon_only
        assert both != delta_only
        assert both != recon_only
