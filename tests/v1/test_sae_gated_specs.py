# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spec-surface tests for the monitor-gated SAE clamp opt-in (``gated``).

Covers the request-side contract of the ``gated`` field on
``SAEClampSpec`` / ``SAEFullReconstructionSpec``:

* dataclass validation and default (``False``),
* coercer acceptance from JSON-shaped payloads,
* hash separation (``gated=True`` never shares a row hash with an
  otherwise-identical ungated spec) and back-compat (``gated=False``
  hashes bit-identically to a spec built before the field existed),
* real-codec wire round-trip through the APIServer->EngineCore hop and
  the type-less collective_rpc revive path used by the global tier.
"""

from __future__ import annotations

import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
    SAEFullReconstructionSpec,
    coerce_sae_clamp_specs,
    coerce_sae_full_reconstruction_specs,
    hash_sae_clamp_specs,
    hash_sae_clamp_specs_for_phase,
    hash_sae_full_reconstruction_specs,
    hash_sae_full_reconstruction_specs_for_phase,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

_CLAMPS = {
    "post_block": {
        7: [SAEClampEntry(feature_idx=3, kind="absolute", value=5.0)],
    }
}


def _clamp_spec(**kwargs) -> SAEClampSpec:
    base = dict(module_name="m", clamps=dict(_CLAMPS))
    base.update(kwargs)
    return SAEClampSpec(**base)


def _fr_spec(**kwargs) -> SAEFullReconstructionSpec:
    base = dict(module_name="fr", clamps=dict(_CLAMPS))
    base.update(kwargs)
    return SAEFullReconstructionSpec(**base)


class TestGatedField:
    def test_default_is_false(self):
        assert _clamp_spec().gated is False
        assert _fr_spec().gated is False

    def test_explicit_true(self):
        assert _clamp_spec(gated=True).gated is True
        assert _fr_spec(gated=True).gated is True

    def test_non_bool_rejected(self):
        with pytest.raises(ValueError, match="gated"):
            _clamp_spec(gated=1)
        with pytest.raises(ValueError, match="gated"):
            _fr_spec(gated="yes")


class TestCoercers:
    def test_clamp_coercer_reads_gated(self):
        raw = [
            {
                "module_name": "m",
                "gated": True,
                "clamps": {
                    "post_block": {
                        "7": [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                    }
                },
            }
        ]
        (spec,) = coerce_sae_clamp_specs(raw)
        assert spec.gated is True

    def test_clamp_coercer_default_false(self):
        raw = [
            {
                "module_name": "m",
                "clamps": {
                    "post_block": {
                        7: [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                    }
                },
            }
        ]
        (spec,) = coerce_sae_clamp_specs(raw)
        assert spec.gated is False

    def test_clamp_coercer_rejects_non_bool_gated(self):
        raw = [
            {
                "module_name": "m",
                "gated": "yes",
                "clamps": {
                    "post_block": {
                        7: [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                    }
                },
            }
        ]
        with pytest.raises(ValueError, match="gated"):
            coerce_sae_clamp_specs(raw)

    def test_fr_coercer_reads_gated(self):
        raw = [{"module_name": "fr", "gated": True}]
        (spec,) = coerce_sae_full_reconstruction_specs(raw)
        assert spec.gated is True

    def test_fr_coercer_default_false(self):
        raw = [{"module_name": "fr"}]
        (spec,) = coerce_sae_full_reconstruction_specs(raw)
        assert spec.gated is False

    def test_fr_coercer_rejects_non_bool_gated(self):
        raw = [{"module_name": "fr", "gated": 2}]
        with pytest.raises(ValueError, match="gated"):
            coerce_sae_full_reconstruction_specs(raw)


class TestHashes:
    def test_gated_changes_clamp_hash(self):
        ungated = (_clamp_spec(),)
        gated = (_clamp_spec(gated=True),)
        assert hash_sae_clamp_specs(ungated) != hash_sae_clamp_specs(gated)
        for phase in ("prefill", "decode"):
            assert hash_sae_clamp_specs_for_phase(
                ungated, phase
            ) != hash_sae_clamp_specs_for_phase(gated, phase)

    def test_gated_changes_fr_hash(self):
        ungated = (_fr_spec(),)
        gated = (_fr_spec(gated=True),)
        assert hash_sae_full_reconstruction_specs(
            ungated
        ) != hash_sae_full_reconstruction_specs(gated)
        for phase in ("prefill", "decode"):
            assert hash_sae_full_reconstruction_specs_for_phase(
                ungated, phase
            ) != hash_sae_full_reconstruction_specs_for_phase(gated, phase)

    def test_gated_false_hash_matches_omitted_field(self):
        # Byte-identical back-compat: gated=False must hash exactly like a
        # spec that never mentions the field (the pre-``gated`` digest).
        explicit = (_clamp_spec(gated=False),)
        omitted = (_clamp_spec(),)
        assert hash_sae_clamp_specs(explicit) == hash_sae_clamp_specs(omitted)
        assert hash_sae_full_reconstruction_specs(
            (_fr_spec(gated=False),)
        ) == hash_sae_full_reconstruction_specs((_fr_spec(),))

    def test_gated_hash_is_order_stable(self):
        a = _clamp_spec(gated=True)
        b = SAEClampSpec(
            module_name="m",
            clamps={
                "post_block": {
                    9: [SAEClampEntry(feature_idx=1, kind="additive", value=1.0)]
                }
            },
        )
        assert hash_sae_clamp_specs((a, b)) == hash_sae_clamp_specs((b, a))


class TestWireRoundTrip:
    def _wrap(self, sp: SamplingParams) -> EngineCoreRequest:
        return EngineCoreRequest(
            request_id="r-gated",
            prompt_token_ids=[1, 2],
            mm_features=None,
            sampling_params=sp,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    def test_gated_survives_engine_core_hop(self):
        sp_in = SamplingParams(
            max_tokens=4,
            sae_clamp_specs=[
                {
                    "module_name": "m",
                    "gated": True,
                    "clamps": {
                        "post_block": {
                            7: [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                        }
                    },
                }
            ],
            sae_full_reconstruction_specs=[{"module_name": "fr", "gated": True}],
        )
        enc = MsgpackEncoder()
        bufs = enc.encode(self._wrap(sp_in))
        sp_out = MsgpackDecoder(EngineCoreRequest).decode(bufs).sampling_params
        assert sp_out.sae_clamp_specs[0].gated is True
        assert sp_out.sae_full_reconstruction_specs[0].gated is True
        assert sp_out.sae_clamp_specs == sp_in.sae_clamp_specs
        assert (
            sp_out.prefill_steering_config_hash == sp_in.prefill_steering_config_hash
        )
        assert sp_out.decode_steering_config_hash == sp_in.decode_steering_config_hash

    def test_gated_survives_collective_rpc_revive(self):
        typed = coerce_sae_clamp_specs(
            [
                {
                    "module_name": "m",
                    "gated": True,
                    "clamps": {
                        "post_block": {
                            7: [{"feature_idx": 3, "kind": "absolute", "value": 5.0}]
                        }
                    },
                }
            ]
        )
        enc = MsgpackEncoder()
        bufs = enc.encode({"prefill_specs_raw": list(typed)})
        wire = MsgpackDecoder().decode(bufs)
        revived = coerce_sae_clamp_specs(wire["prefill_specs_raw"])
        assert revived == typed
        assert revived[0].gated is True

    def test_fr_gated_survives_collective_rpc_revive(self):
        typed = coerce_sae_full_reconstruction_specs(
            [{"module_name": "fr", "gated": True}]
        )
        enc = MsgpackEncoder()
        bufs = enc.encode({"specs": list(typed)})
        wire = MsgpackDecoder().decode(bufs)
        revived = coerce_sae_full_reconstruction_specs(wire["specs"])
        assert revived == typed
        assert revived[0].gated is True
