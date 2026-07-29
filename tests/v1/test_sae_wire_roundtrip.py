# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-codec wire round-trips for SAE-bearing SamplingParams.

The APIServer->EngineCore ADD hop strictly typed-decodes
``EngineCoreRequest`` (and its nested ``SamplingParams`` with the SAE
dataclass fields).  Until now nothing exercised that hop with SAE specs
through vLLM's own ``MsgpackEncoder``/``MsgpackDecoder`` pair — the same
coverage gap that let the directional-clamp wire wedge ship on the
sibling branch.  Covers:

- typed round-trip of ``sae_clamp_specs`` + ``sae_full_reconstruction_specs``
  (dataclass equality, int layer keys, ``__post_init__`` re-run) with all
  cached config hashes stable across the hop;
- strict rejection of string layer keys injected at the wire level;
- collective_rpc-shaped revive: type-less decoded spec dicts (what
  ``set_sae_global_clamps`` receives) revive equal via the coercers.
"""

import msgspec
import pytest

from vllm.config.sae_steering_types import (
    SAEClampEntry,
    SAEClampSpec,
    SAEFullReconstructionSpec,
    coerce_sae_clamp_specs,
    coerce_sae_full_reconstruction_specs,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

CLAMP_SPECS = [
    {
        "module_name": "sae_a",
        "clamps": {
            "post_block": {
                7: [
                    {"feature_idx": 3, "kind": "absolute", "value": 5.0},
                    {
                        "feature_idx": 11,
                        "kind": "additive",
                        "value": -2.5,
                        "only_if_active": True,
                    },
                ],
                12: [{"feature_idx": 3, "kind": "absolute", "value": 0.0}],
            },
            "pre_attn": {2: [{"feature_idx": 9, "kind": "additive", "value": 1.5}]},
        },
    },
    {
        "module_name": "sae_b",
        "phase": "decode",
        "clamps": {
            "post_block": {7: [{"feature_idx": 4, "kind": "absolute", "value": 3.0}]}
        },
    },
]

FR_SPECS = [
    # Pure reconstruction: empty clamps is legal for FR specs.
    {"module_name": "fr_mod", "phase": "both"},
    {
        "module_name": "fr_clamped",
        "phase": "prefill",
        "clamps": {
            "post_block": {5: [{"feature_idx": 1, "kind": "absolute", "value": 2.0}]}
        },
    },
]


def _sae_sp() -> SamplingParams:
    return SamplingParams(
        max_tokens=4,
        sae_clamp_specs=[dict(s) for s in CLAMP_SPECS],
        sae_full_reconstruction_specs=[dict(s) for s in FR_SPECS],
    )


def _wrap(sp: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="r-sae",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class TestAddRequestHop:
    def test_round_trip_typed_and_hashes_stable(self):
        sp_in = _sae_sp()
        hashes_in = (
            sp_in.prefill_steering_config_hash,
            sp_in.decode_steering_config_hash,
            sp_in.prefill_sae_clamp_config_hash,
            sp_in.decode_sae_clamp_config_hash,
            sp_in.prefill_sae_full_recon_config_hash,
            sp_in.decode_sae_full_recon_config_hash,
        )

        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        req_out = MsgpackDecoder(EngineCoreRequest).decode(bufs)
        sp_out = req_out.sampling_params

        assert sp_out.sae_clamp_specs == sp_in.sae_clamp_specs
        assert sp_out.sae_full_reconstruction_specs == (
            sp_in.sae_full_reconstruction_specs
        )
        assert all(isinstance(s, SAEClampSpec) for s in sp_out.sae_clamp_specs)
        assert all(
            isinstance(s, SAEFullReconstructionSpec)
            for s in sp_out.sae_full_reconstruction_specs
        )
        # Layer keys stay ints and entries stay typed after strict decode.
        table = sp_out.sae_clamp_specs[0].clamps["post_block"]
        assert set(table.keys()) == {7, 12}
        assert isinstance(table[7][0], SAEClampEntry)
        assert table[7][1].only_if_active is True
        # Engine side recomputes identical hashes from the decoded structs.
        hashes_out = (
            sp_out.prefill_steering_config_hash,
            sp_out.decode_steering_config_hash,
            sp_out.prefill_sae_clamp_config_hash,
            sp_out.decode_sae_clamp_config_hash,
            sp_out.prefill_sae_full_recon_config_hash,
            sp_out.decode_sae_full_recon_config_hash,
        )
        assert hashes_out == hashes_in
        assert hashes_in[0] != 0

    def test_string_layer_key_rejected_at_strict_decode(self):
        sp_in = _sae_sp()
        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        wire = MsgpackDecoder().decode(bufs)

        def _stringify_layer_key(obj) -> bool:
            if isinstance(obj, dict):
                if "module_name" in obj and "clamps" in obj:
                    for layer_map in obj["clamps"].values():
                        for key in list(layer_map.keys()):
                            layer_map[str(key)] = layer_map.pop(key)
                        return True
                return any(_stringify_layer_key(v) for v in obj.values())
            if isinstance(obj, list):
                return any(_stringify_layer_key(v) for v in obj)
            return False

        assert _stringify_layer_key(wire)
        rebuffed = msgspec.msgpack.encode(wire)
        with pytest.raises(msgspec.ValidationError):
            MsgpackDecoder(EngineCoreRequest).decode(rebuffed)

    def test_fr_specs_omitted_when_absent(self):
        sp_in = SamplingParams(
            max_tokens=4, sae_clamp_specs=[dict(s) for s in CLAMP_SPECS]
        )
        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        sp_out = MsgpackDecoder(EngineCoreRequest).decode(bufs).sampling_params
        assert sp_out.sae_full_reconstruction_specs is None
        assert sp_out.sae_clamp_specs == sp_in.sae_clamp_specs


class TestCollectiveRpcRevive:
    """The utility RPC hop decodes type-less; receivers revive via the
    coercers (mirrors set_sae_global_clamps' ingestion)."""

    def test_clamp_specs_flatten_then_revive_equal(self):
        typed = coerce_sae_clamp_specs([dict(s) for s in CLAMP_SPECS])
        enc = MsgpackEncoder()
        bufs = enc.encode({"prefill_specs_raw": list(typed)})
        wire = MsgpackDecoder().decode(bufs)
        flattened = wire["prefill_specs_raw"]
        assert isinstance(flattened[0], dict)  # dataclass -> plain map
        revived = coerce_sae_clamp_specs(flattened)
        assert revived == typed

    def test_fr_specs_flatten_then_revive_equal(self):
        typed = coerce_sae_full_reconstruction_specs([dict(s) for s in FR_SPECS])
        enc = MsgpackEncoder()
        bufs = enc.encode({"specs": list(typed)})
        wire = MsgpackDecoder().decode(bufs)
        revived = coerce_sae_full_reconstruction_specs(wire["specs"])
        assert revived == typed
