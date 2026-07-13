# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the packed binary wire form of directional clamps.

Covers:
- pack_steering_clamps / unpack_steering_clamps round-trips (order, bounds,
  strengths, inf<->null encoding)
- structural validation of malformed packed blobs
- coerce_clamp_spec passthrough vs unpack behaviour
- resolve_effective_clamps transparently unpacking packed tiers
- validate_clamp_row_widths on packed tiers
- CRITICAL: packed-vs-JSON hash equality (prefix-cache correctness), both at
  the hash_steering_config level and the SamplingParams cached-hash level
- maybe_pack_inline_steering_for_request packing JSON clamps in place while
  preserving the fp64-derived config hash
"""

import base64
import math

import numpy as np
import pytest

from vllm.config.steering_types import (
    coerce_clamp_spec,
    hash_steering_config,
    maybe_pack_inline_steering_for_request,
    pack_steering_clamps,
    resolve_effective_clamps,
    unpack_steering_clamps,
    validate_clamp_row_widths,
)
from vllm.sampling_params import SamplingParams


def _pack_blob(rows, layer_indices, bounds, strengths, dtype="float64"):
    """Build one ClampHookPacked dict from row vectors + metadata."""
    arr = np.asarray(rows, dtype=np.dtype(dtype))
    return {
        "dtype": dtype,
        "shape": [arr.shape[0], arr.shape[1]],
        "layer_indices": list(layer_indices),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "bounds": bounds,
        "strengths": strengths,
    }


class TestUnpackSteeringClamps:
    def test_basic_roundtrip(self):
        packed = {
            "post_attn": _pack_blob(
                rows=[[1.0, 0.0], [0.0, 1.0]],
                layer_indices=[3, 5],
                bounds=[[-2.0, 2.0], [None, 4.0]],
                strengths=[1.0, 0.5],
            )
        }
        spec = unpack_steering_clamps(packed)
        assert set(spec.keys()) == {"post_attn"}
        assert set(spec["post_attn"].keys()) == {3, 5}
        e3 = spec["post_attn"][3][0]
        assert e3["vector"] == [1.0, 0.0]
        assert e3["min"] == -2.0
        assert e3["max"] == 2.0
        assert e3["strength"] == 1.0
        e5 = spec["post_attn"][5][0]
        assert e5["min"] == -math.inf  # null -> -inf
        assert e5["max"] == 4.0
        assert e5["strength"] == 0.5

    def test_multiple_entries_same_layer_preserve_order(self):
        packed = {
            "pre_attn": _pack_blob(
                rows=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                layer_indices=[2, 2, 2],
                bounds=[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                strengths=[1.0, 1.0, 1.0],
            )
        }
        spec = unpack_steering_clamps(packed)
        entries = spec["pre_attn"][2]
        assert len(entries) == 3
        # Row order preserved: distinguishing bound is the first bound.
        assert [e["min"] for e in entries] == [0.0, 1.0, 2.0]

    def test_none_and_empty(self):
        assert unpack_steering_clamps(None) is None
        assert unpack_steering_clamps({}) is None


class TestPackRoundtrip:
    def test_pack_unpack_identity(self):
        spec = {
            "post_block": {
                7: [
                    {"vector": [0.6, 0.8], "min": -1.0, "max": 1.0, "strength": 0.5},
                    {
                        "vector": [0.8, 0.6],
                        "min": -math.inf,
                        "max": 2.0,
                        "strength": 1.0,
                    },
                ]
            }
        }
        packed = pack_steering_clamps(spec)
        back = unpack_steering_clamps(packed)
        assert set(back.keys()) == {"post_block"}
        entries = back["post_block"][7]
        assert len(entries) == 2
        assert np.allclose(entries[0]["vector"], [0.6, 0.8])
        assert entries[0]["min"] == -1.0
        assert entries[1]["min"] == -math.inf


class TestStructuralValidation:
    def test_bad_base64_length(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["shape"] = [1, 3]  # claims 3 wide but data is 2 wide
        with pytest.raises(ValueError):
            unpack_steering_clamps({"post_attn": blob})

    def test_layer_indices_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["layer_indices"] = [0, 1]
        with pytest.raises(ValueError):
            unpack_steering_clamps({"post_attn": blob})

    def test_bounds_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["bounds"] = [[0.0, 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError):
            unpack_steering_clamps({"post_attn": blob})

    def test_strengths_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["strengths"] = [1.0, 1.0]
        with pytest.raises(ValueError):
            unpack_steering_clamps({"post_attn": blob})


class TestCoerceClampSpec:
    def test_passthrough_packed_by_default(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        packed = {"post_attn": blob}
        out = coerce_clamp_spec(packed)
        # Passthrough: still the packed blob shape.
        assert out["post_attn"] is blob

    def test_unpack_packed_when_requested(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        out = coerce_clamp_spec({"post_attn": blob}, unpack_packed=True)
        assert out["post_attn"][3][0]["vector"] == [1.0, 0.0]

    def test_json_int_coerce_unchanged(self):
        out = coerce_clamp_spec(
            {"post_attn": {"3": [{"vector": [1.0, 0.0], "value": 1.0}]}}
        )
        assert set(out["post_attn"].keys()) == {3}


class TestResolveUnpacksPacked:
    def test_packed_base_resolves(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        resolved = resolve_effective_clamps(packed, None)
        assert resolved["post_attn"][3][0]["vector"] == [1.0, 0.0]

    def test_packed_concat_with_json_phase(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        phase = {"post_attn": {3: [{"vector": [0.0, 1.0], "value": 2.0}]}}
        resolved = resolve_effective_clamps(packed, phase)
        assert len(resolved["post_attn"][3]) == 2


class TestValidateWidthsPacked:
    def test_packed_width_ok(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        validate_clamp_row_widths(packed, 2, field_name="steering_clamps")

    def test_packed_width_mismatch(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        with pytest.raises(ValueError):
            validate_clamp_row_widths(packed, 4, field_name="steering_clamps")


class TestHashEquality:
    """CRITICAL: same logical config, packed vs JSON, must hash identically."""

    def test_hash_steering_config_packed_equals_json(self):
        json_spec = {
            "post_attn": {
                3: [{"vector": [3.0, 4.0], "min": -1.0, "max": 2.0, "strength": 1.0}],
                5: [{"vector": [1.0, 0.0], "min": None, "max": 5.0, "strength": 0.5}],
            }
        }
        # Build the packed form from the *same* canonical entries.
        packed = pack_steering_clamps(json_spec)
        h_json = hash_steering_config(None, clamps=json_spec)
        h_packed = hash_steering_config(None, clamps=unpack_steering_clamps(packed))
        assert h_json == h_packed
        assert h_json != 0

    def test_sampling_params_packed_equals_json_hash(self):
        raw = {
            "post_attn": {
                3: [{"vector": [3.0, 4.0], "min": -1.0, "max": 2.0}],
                5: [{"vector": [0.0, 2.0], "value": 1.5}],
            }
        }
        sp_json = SamplingParams(steering_clamps=raw)
        # Pack the canonicalized (validated, unit-normalized) form.
        packed = pack_steering_clamps(sp_json.steering_clamps)
        sp_packed = SamplingParams(steering_clamps=packed)
        assert (
            sp_json.prefill_steering_config_hash
            == sp_packed.prefill_steering_config_hash
        )
        assert (
            sp_json.decode_steering_config_hash == sp_packed.decode_steering_config_hash
        )
        assert sp_json.prefill_steering_config_hash != 0


class TestSamplingParamsPackedValidation:
    def test_packed_field_accepted_structurally(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        sp = SamplingParams(steering_clamps=packed)
        # Not eagerly decoded: still packed on the field.
        assert "data" in sp.steering_clamps["post_attn"]
        # But resolvable through effective props.
        assert sp.effective_prefill_clamps["post_attn"][3][0]["min"] == 0.0

    def test_packed_bad_structure_rejected(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        blob["bounds"] = [[0.0, 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError):
            SamplingParams(steering_clamps={"post_attn": blob})

    def test_packed_bad_hook_rejected(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        with pytest.raises(ValueError):
            SamplingParams(steering_clamps={"not_a_hook": blob})


class TestMaybePackClamps:
    def test_json_clamps_packed_in_place_preserving_hash(self):
        raw = {
            "post_attn": {
                3: [{"vector": [3.0, 4.0], "min": -1.0, "max": 2.0}],
            }
        }
        sp = SamplingParams(steering_clamps=raw)
        h_prefill = sp.prefill_steering_config_hash
        h_decode = sp.decode_steering_config_hash

        sp2 = SamplingParams(steering_clamps=raw)
        maybe_pack_inline_steering_for_request(sp2, np.float32, expected_row_width=2)
        # Field replaced with the packed form.
        assert "data" in sp2.steering_clamps["post_attn"]
        # Hash preserved bit-for-bit.
        assert sp2.prefill_steering_config_hash == h_prefill
        assert sp2.decode_steering_config_hash == h_decode

    def test_clamp_only_width_gate_rejects_packed_wrong_width(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        sp = SamplingParams(steering_clamps=packed)
        with pytest.raises(ValueError):
            maybe_pack_inline_steering_for_request(sp, np.float32, expected_row_width=8)
