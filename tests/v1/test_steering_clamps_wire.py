# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-boundary tests for the canonical SteeringClamps type.

These cover the seams that previously had NO test coverage and where the
pre-canonical representation bugs lived:

- a clamp-bearing SamplingParams crossing the real APIServer->EngineCore
  ADD hop (``MsgpackEncoder`` -> typed ``MsgpackDecoder(EngineCoreRequest)``)
  must decode to equal Structs with ``__post_init__`` re-run and identical
  config hashes;
- collective_rpc's type-less generic decode flattens Structs to plain
  dicts (bytes preserved) — worker receivers must revive them equal via
  ``from_obj``;
- the legacy Rust verbatim entry-list wire shape no longer decodes under
  the strict typed annotation (the documented compat window until the
  Rust mirror lands);
- pydantic models embedding SamplingParams (the disagg protocol) must
  schema-generate, serialize (base64 JSON form) and validate back.
"""

import copy

import msgspec
import pytest

from vllm.config.steering_types import SteeringClamps
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

CLAMPS = {
    "post_block": {
        3: [{"vector": [3.0, 4.0, 0.0, 1.0], "min": -0.5, "max": 1.25}],
        7: [{"vector": [1.0, -2.0, 3.0, -4.0], "value": 2.0, "strength": 0.5}],
    },
    "pre_attn": {
        0: [{"vector": [-1.0, 0.5, 0.25, 2.0], "min": -3.0}],
    },
}


def _clamped_sp() -> SamplingParams:
    return SamplingParams(
        max_tokens=4,
        steering_clamps=copy.deepcopy(CLAMPS),
        decode_steering_clamps={
            "post_attn": {5: [{"vector": [1.0, 0.0, 0.0, 0.0], "value": 1.0}]}
        },
    )


def _wrap(sp: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="r1",
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
    """The real ZMQ ADD codec pair, exactly as core.py builds it."""

    def test_round_trip_typed_and_hash_stable(self):
        sp_in = _clamped_sp()
        prefill_hash = sp_in.prefill_steering_config_hash
        decode_hash = sp_in.decode_steering_config_hash

        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        req_out = MsgpackDecoder(EngineCoreRequest).decode(bufs)
        sp_out = req_out.sampling_params

        assert isinstance(sp_out.steering_clamps, SteeringClamps)
        assert sp_out.steering_clamps == sp_in.steering_clamps
        assert sp_out.decode_steering_clamps == sp_in.decode_steering_clamps
        assert sp_out.prefill_steering_clamps is None
        # The engine side recomputes hashes lazily from the decoded
        # Structs — identical bytes, identical hashes.
        assert sp_out.prefill_steering_config_hash == prefill_hash
        assert sp_out.decode_steering_config_hash == decode_hash

    def test_data_rides_msgpack_bin_not_base64(self):
        sp_in = _clamped_sp()
        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        # Type-less decode of the same frames exposes the raw wire shape.
        wire = MsgpackDecoder().decode(bufs)

        def _find_hooks(obj):
            if isinstance(obj, dict):
                if set(obj.keys()) == {"hooks"}:
                    return obj["hooks"]
                for v in obj.values():
                    found = _find_hooks(v)
                    if found is not None:
                        return found
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    found = _find_hooks(v)
                    if found is not None:
                        return found
            return None

        hooks = _find_hooks(wire)
        assert hooks is not None
        assert isinstance(hooks["post_block"]["data"], bytes)

    def test_corrupt_table_rejected_on_decode(self):
        sp_in = _clamped_sp()
        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        wire = MsgpackDecoder().decode(bufs)

        def _mutate(obj):
            if isinstance(obj, dict):
                if set(obj.keys()) == {"hooks"}:
                    blob = next(iter(obj["hooks"].values()))
                    blob["shape"] = [blob["shape"][0] + 1, blob["shape"][1]]
                    return True
                return any(_mutate(v) for v in obj.values())
            if isinstance(obj, list):
                return any(_mutate(v) for v in obj)
            return False

        assert _mutate(wire)
        rebuffed = msgspec.msgpack.encode(wire)
        with pytest.raises(msgspec.ValidationError):
            MsgpackDecoder(EngineCoreRequest).decode(rebuffed)

    def test_legacy_verbatim_entry_lists_rejected(self):
        """The old Rust frontend forwards raw entry-list dicts as the
        field value; the strict typed annotation rejects them (documented
        compat window until the Rust typed mirror lands)."""
        sp_in = _clamped_sp()
        enc = MsgpackEncoder()
        bufs = enc.encode(_wrap(sp_in))
        wire = MsgpackDecoder().decode(bufs)

        def _swap(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict) and set(v.keys()) == {"hooks"}:
                        obj[k] = {"post_block": {"3": [{"vector": [1.0, 0.0]}]}}
                        return True
                    if _swap(v):
                        return True
            if isinstance(obj, list):
                return any(_swap(v) for v in obj)
            return False

        assert _swap(wire)
        rebuffed = msgspec.msgpack.encode(wire)
        with pytest.raises(msgspec.ValidationError):
            MsgpackDecoder(EngineCoreRequest).decode(rebuffed)


class TestCollectiveRpcHop:
    """The utility RPC path decodes type-less: Structs flatten to plain
    dicts (bytes preserved) and worker receivers revive via from_obj."""

    def test_flatten_then_revive_equal(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(CLAMPS))
        enc = MsgpackEncoder()
        kwargs = {"clamps": spec, "prefill_clamps": None}
        bufs = enc.encode(kwargs)
        wire = MsgpackDecoder().decode(bufs)

        flattened = wire["clamps"]
        assert isinstance(flattened, dict)
        assert set(flattened.keys()) == {"hooks"}
        assert isinstance(flattened["hooks"]["post_block"]["data"], bytes)

        revived = SteeringClamps.from_obj(flattened)
        assert revived == spec
        assert SteeringClamps.from_obj(wire["prefill_clamps"]) is None


class TestPydanticEmbedding:
    """SamplingParams is embedded in pydantic models (disagg protocol);
    the Struct field must schema-generate and JSON round-trip."""

    def test_disagg_protocol_importable(self):
        import vllm.entrypoints.serve.disagg.protocol  # noqa: F401

    def test_pydantic_json_round_trip(self):
        from pydantic import BaseModel

        class Holder(BaseModel):
            sampling_params: SamplingParams

        sp = _clamped_sp()
        holder = Holder(sampling_params=sp)
        dumped = holder.model_dump_json()
        restored = Holder.model_validate_json(dumped)
        out = restored.sampling_params
        assert isinstance(out.steering_clamps, SteeringClamps)
        assert out.steering_clamps == sp.steering_clamps
        assert out.decode_steering_clamps == sp.decode_steering_clamps
