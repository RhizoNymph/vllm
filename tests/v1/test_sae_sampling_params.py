# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SamplingParams.sae_clamp_specs plumbing."""

from __future__ import annotations

import pytest

from vllm import SamplingParams
from vllm.config.sae_steering_types import SAEClampEntry, SAEClampSpec


def _spec(*, phase: str = "both") -> SAEClampSpec:
    return SAEClampSpec(
        module_name="g",
        phase=phase,  # type: ignore[arg-type]
        clamps={
            "post_mlp": {
                20: (SAEClampEntry(feature_idx=34, kind="absolute", value=5.0),)
            }
        },
    )


class TestSamplingParamsSAEField:
    def test_default_is_none(self):
        sp = SamplingParams()
        assert sp.sae_clamp_specs is None

    def test_typed_input_passthrough(self):
        s = _spec()
        sp = SamplingParams(sae_clamp_specs=(s,))
        assert sp.sae_clamp_specs == (s,)

    def test_dict_input_coerced(self):
        """Direct construction with a list-of-dicts must be normalized to
        the typed form by ``_validate_steering_vectors``."""
        sp = SamplingParams(
            sae_clamp_specs=[
                {
                    "module_name": "g",
                    "clamps": {
                        "post_mlp": {
                            20: [
                                {
                                    "feature_idx": 34,
                                    "kind": "absolute",
                                    "value": 5.0,
                                }
                            ]
                        }
                    },
                }
            ],
        )
        assert sp.sae_clamp_specs is not None
        assert isinstance(sp.sae_clamp_specs[0], SAEClampSpec)
        assert sp.sae_clamp_specs[0].module_name == "g"

    def test_invalid_dict_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(
                sae_clamp_specs=[
                    {
                        "module_name": "g",
                        "clamps": {
                            "bogus_hook": {
                                0: [
                                    {"feature_idx": 0, "kind": "absolute", "value": 1.0}
                                ]
                            }
                        },
                    }
                ],
            )

    def test_clone_reuses_immutable_sae_specs(self):
        s = _spec()
        sp = SamplingParams(sae_clamp_specs=(s,))
        clone = sp.clone()
        assert clone is not sp
        assert clone.sae_clamp_specs is sp.sae_clamp_specs

    def test_clone_carries_cached_sae_phase_hashes(self):
        s = _spec()
        sp = SamplingParams(sae_clamp_specs=(s,))
        prefill_hash = sp.prefill_sae_clamp_config_hash
        decode_hash = sp.decode_sae_clamp_config_hash

        clone = sp.clone()

        assert clone.__dict__["prefill_sae_clamp_config_hash"] == prefill_hash
        assert clone.__dict__["decode_sae_clamp_config_hash"] == decode_hash

    def test_steering_module_ref_rejects_bool_scale(self):
        with pytest.raises(ValueError, match="steering_module_ref"):
            SamplingParams(steering_module_ref=("m", True))


class TestPhaseFilteredHashing:
    def test_no_sae_hash_unchanged(self):
        """Inline-only sampling params (no SAE) must hash identically
        to a build without SAE support."""
        a = SamplingParams(steering_vectors={"post_mlp": {0: [0.1]}})
        b = SamplingParams(steering_vectors={"post_mlp": {0: [0.1]}})
        assert a.prefill_steering_config_hash == b.prefill_steering_config_hash
        assert a.decode_steering_config_hash == b.decode_steering_config_hash

    def test_phase_both_affects_both_hashes(self):
        s = _spec(phase="both")
        sp_no = SamplingParams(steering_vectors={"post_mlp": {0: [0.1]}})
        sp_sae = SamplingParams(
            steering_vectors={"post_mlp": {0: [0.1]}}, sae_clamp_specs=(s,)
        )
        assert sp_no.prefill_steering_config_hash != sp_sae.prefill_steering_config_hash
        assert sp_no.decode_steering_config_hash != sp_sae.decode_steering_config_hash

    def test_phase_prefill_affects_only_prefill(self):
        s = _spec(phase="prefill")
        sp_no = SamplingParams(steering_vectors={"post_mlp": {0: [0.1]}})
        sp_sae = SamplingParams(
            steering_vectors={"post_mlp": {0: [0.1]}}, sae_clamp_specs=(s,)
        )
        assert sp_no.prefill_steering_config_hash != sp_sae.prefill_steering_config_hash
        # Decode-tier hash must be untouched: the SAE spec is prefill-only.
        assert sp_no.decode_steering_config_hash == sp_sae.decode_steering_config_hash

    def test_phase_decode_affects_only_decode(self):
        s = _spec(phase="decode")
        sp_no = SamplingParams(steering_vectors={"post_mlp": {0: [0.1]}})
        sp_sae = SamplingParams(
            steering_vectors={"post_mlp": {0: [0.1]}}, sae_clamp_specs=(s,)
        )
        assert sp_no.prefill_steering_config_hash == sp_sae.prefill_steering_config_hash
        assert sp_no.decode_steering_config_hash != sp_sae.decode_steering_config_hash
