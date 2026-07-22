# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the canonical SteeringClamps / ClampHookTable types.

Covers:
- from_obj over every accepted input shape: instances, the type's own
  wire map (bytes and base64/null JSON form), legacy entry-lists
  (int/str keys, value sugar, omitted bounds), legacy base64 packed
  blobs (f64/f32/f16 with exact upcast)
- empty-input collapse ({} -> None) vs preserve_empty (REPLACE clear)
- structural validation errors, field_name-prefixed messages, and the
  ingestion-eager finite/non-zero row checks
- accessors: rows zero-copy view, by_layer grouping/order, site_counts,
  concat, validate_row_width, check_max_directions
- typed msgspec round-trip (__post_init__ runs on decode) and
  to_json_obj lossless round-trip
- pickle / deepcopy
- pinned golden config-hash digests captured at the parent commit —
  guards hash parity across the representation refactor
"""

import base64
import copy
import math
import pickle

import msgspec
import numpy as np
import pytest

from vllm.config.steering_types import (
    ClampHookTable,
    SteeringClamps,
)
from vllm.sampling_params import SamplingParams

HIDDEN = 8
V1 = [0.3, -1.7, 2.5, 0.01, -4.2, 3.3, -0.9, 1.1]
V2 = [1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]
V3 = [-2.5, 0.125, 7.75, -0.001, 3.5, -1.25, 0.5, -6.0]

JSON_CANONICAL = {
    "post_block": {
        3: [{"vector": V1, "min": -0.5, "max": 1.25}],
        7: [{"vector": V2, "value": 2.0}],
    },
    "pre_attn": {
        0: [{"vector": V3, "min": -3.0}],
    },
}

SUGAR_STRENGTH = {
    "post_attn": {
        5: [{"vector": V1, "value": -1.5, "strength": 0.25}],
    },
}

INF_BOUNDS = {
    "post_block": {
        2: [
            {"vector": V2, "max": 4.0},
            {"vector": V3, "min": -2.0},
        ],
    },
}

MULTI_SAME_LAYER = {
    "post_block": {
        3: [
            {"vector": V1, "min": -0.5, "max": 0.5},
            {"vector": V2, "value": 1.0, "strength": 0.5},
        ],
    },
}

# Golden config hashes captured at the parent commit (f865486a0) via the
# then-current SamplingParams ingestion + hash path.  The canonical-type
# refactor must keep reproducing these for identical client submissions.
GOLDEN_JSON_CANONICAL = 5198768903439831645
GOLDEN_SUGAR_STRENGTH = 6374091342592873568
GOLDEN_INF_BOUNDS = 9220295498424249194
GOLDEN_MULTI_SAME_LAYER = 840792782604274689
GOLDEN_BASE_PLUS_DECODE_DECODE = 5577579003190762101
# Raw (unnormalized) rows packed at float32 quantize the direction beyond
# an fp32 ulp of the JSON submission — a distinct-but-stable hash, per the
# documented narrow-dtype packed contract (one-time prefix-cache miss).
GOLDEN_PACKED_F32_RAW = 8463674918148086811


def _str_keys(spec):
    return {
        hook: {str(layer): copy.deepcopy(entries) for layer, entries in layers.items()}
        for hook, layers in spec.items()
    }


def _legacy_packed(spec, dtype="float64"):
    """Build the legacy base64 ClampHookPacked form of an entry-list spec."""
    np_dtype = np.dtype(dtype)
    result = {}
    for hook, layers in spec.items():
        rows, layer_indices, bounds, strengths = [], [], [], []
        for layer in sorted(layers):
            for entry in layers[layer]:
                vec = np.asarray(entry["vector"], dtype=np_dtype)
                rows.append(vec)
                layer_indices.append(layer)
                if entry.get("value") is not None:
                    lo = hi = float(entry["value"])
                else:
                    lo = entry.get("min", -math.inf)
                    hi = entry.get("max", math.inf)
                bounds.append(
                    [
                        None if math.isinf(lo) else float(lo),
                        None if math.isinf(hi) else float(hi),
                    ]
                )
                strengths.append(float(entry.get("strength", 1.0)))
        stacked = np.stack(rows).astype(np_dtype)
        result[hook] = {
            "dtype": np_dtype.name,
            "shape": [stacked.shape[0], stacked.shape[1]],
            "layer_indices": layer_indices,
            "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
            "bounds": bounds,
            "strengths": strengths,
        }
    return result


class TestFromObjEntryLists:
    def test_rows_stored_raw_not_normalized(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        table = spec.hooks["post_block"]
        assert table.shape == [2, HIDDEN]
        np.testing.assert_array_equal(table.rows()[0], np.asarray(V1))
        np.testing.assert_array_equal(table.rows()[1], np.asarray(V2))

    def test_str_keys_equal_int_keys(self):
        a = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        b = SteeringClamps.from_obj(_str_keys(JSON_CANONICAL))
        assert a == b

    def test_value_sugar_and_strength(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(SUGAR_STRENGTH))
        table = spec.hooks["post_attn"]
        assert table.lo == [-1.5]
        assert table.hi == [-1.5]
        assert table.strength == [0.25]

    def test_omitted_bounds_become_inf(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(INF_BOUNDS))
        table = spec.hooks["post_block"]
        assert table.lo == [-math.inf, -2.0]
        assert table.hi == [4.0, math.inf]

    def test_multi_entry_layer_order_preserved(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(MULTI_SAME_LAYER))
        table = spec.hooks["post_block"]
        assert table.layer_indices == [3, 3]
        np.testing.assert_array_equal(table.rows()[0], np.asarray(V1))
        np.testing.assert_array_equal(table.rows()[1], np.asarray(V2))
        assert table.strength == [1.0, 0.5]

    def test_layers_packed_in_sorted_order(self):
        spec = SteeringClamps.from_obj(
            {
                "post_block": {
                    9: [{"vector": V2, "value": 1.0}],
                    1: [{"vector": V1, "value": 2.0}],
                }
            }
        )
        assert spec.hooks["post_block"].layer_indices == [1, 9]

    def test_empty_entry_list_layer_dropped(self):
        assert SteeringClamps.from_obj({"post_block": {3: []}}) is None

    def test_default_strength(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        assert spec.hooks["pre_attn"].strength == [1.0]


class TestFromObjInstanceAndEmpty:
    def test_instance_passthrough_identity(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        assert SteeringClamps.from_obj(spec) is spec

    def test_none_passthrough(self):
        assert SteeringClamps.from_obj(None) is None

    def test_empty_dict_collapses_to_none(self):
        assert SteeringClamps.from_obj({}) is None

    def test_empty_dict_preserved_on_request(self):
        out = SteeringClamps.from_obj({}, preserve_empty=True)
        assert out == SteeringClamps.empty()
        assert not out

    def test_empty_instance_collapses_to_none(self):
        assert SteeringClamps.from_obj(SteeringClamps.empty()) is None
        kept = SteeringClamps.from_obj(SteeringClamps.empty(), preserve_empty=True)
        assert kept == SteeringClamps.empty()

    def test_bool(self):
        assert not SteeringClamps.empty()
        assert SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))


class TestFromObjWireMap:
    def test_untyped_msgpack_decode_revives(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(spec))
        assert isinstance(wire, dict) and set(wire.keys()) == {"hooks"}
        blob = next(iter(wire["hooks"].values()))
        assert isinstance(blob["data"], bytes)  # msgpack bin, no base64
        assert SteeringClamps.from_obj(wire) == spec

    def test_json_obj_round_trip(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(INF_BOUNDS))
        obj = spec.to_json_obj()
        blob = obj["hooks"]["post_block"]
        assert isinstance(blob["data"], str)
        assert blob["lo"] == [None, -2.0]
        assert blob["hi"] == [4.0, None]
        assert SteeringClamps.from_obj(obj) == spec

    def test_empty_hooks_wire_map(self):
        assert SteeringClamps.from_obj({"hooks": {}}) is None
        assert (
            SteeringClamps.from_obj({"hooks": {}}, preserve_empty=True)
            == SteeringClamps.empty()
        )

    def test_malformed_hook_table_rejected(self):
        with pytest.raises(ValueError, match="clamps\\['post_block'\\]"):
            SteeringClamps.from_obj({"hooks": {"post_block": {"shape": [1, 2]}}})


class TestFromObjLegacyPacked:
    def test_f64_equals_entry_list_ingestion(self):
        a = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        b = SteeringClamps.from_obj(_legacy_packed(JSON_CANONICAL))
        assert a == b

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_narrow_dtype_exact_upcast(self, dtype):
        packed = _legacy_packed(JSON_CANONICAL, dtype=dtype)
        spec = SteeringClamps.from_obj(packed)
        expected = np.stack(
            [np.asarray(V1, dtype=dtype), np.asarray(V2, dtype=dtype)]
        ).astype(np.float64)
        np.testing.assert_array_equal(spec.hooks["post_block"].rows(), expected)

    def test_null_bounds_become_inf(self):
        spec = SteeringClamps.from_obj(_legacy_packed(INF_BOUNDS))
        table = spec.hooks["post_block"]
        assert table.lo == [-math.inf, -2.0]
        assert table.hi == [4.0, math.inf]

    def test_length_mismatch_rejected(self):
        packed = _legacy_packed(JSON_CANONICAL)
        packed["post_block"]["strengths"] = [1.0]
        with pytest.raises(ValueError, match="strengths"):
            SteeringClamps.from_obj(packed)


class TestFromObjValidation:
    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="clamps must be a dict"):
            SteeringClamps.from_obj([1, 2])

    def test_field_name_in_errors(self):
        with pytest.raises(ValueError, match="decode_steering_clamps"):
            SteeringClamps.from_obj([1], field_name="decode_steering_clamps")

    def test_bad_layer_key(self):
        with pytest.raises(ValueError, match="invalid layer index"):
            SteeringClamps.from_obj(
                {"post_block": {"x": [{"vector": V1, "value": 1.0}]}}
            )
        with pytest.raises(ValueError, match="invalid layer index"):
            SteeringClamps.from_obj(
                {"post_block": {-1: [{"vector": V1, "value": 1.0}]}}
            )

    def test_entries_must_be_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            SteeringClamps.from_obj({"post_block": {3: {"vector": V1, "value": 1.0}}})

    def test_entry_errors_prefixed_with_site(self):
        with pytest.raises(ValueError, match="clamps\\['post_block'\\]\\[3\\]"):
            SteeringClamps.from_obj({"post_block": {3: [{"vector": V1}]}})

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            SteeringClamps.from_obj(
                {"post_block": {3: [{"vector": [0.0] * HIDDEN, "value": 1.0}]}}
            )

    def test_nonfinite_vector_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            SteeringClamps.from_obj(
                {"post_block": {3: [{"vector": [math.inf] * HIDDEN, "value": 1.0}]}}
            )

    def test_packed_zero_row_rejected_eagerly(self):
        packed = _legacy_packed(
            {"post_block": {3: [{"vector": [0.0] * HIDDEN, "value": 1.0}]}}
        )
        with pytest.raises(ValueError, match="non-zero"):
            SteeringClamps.from_obj(packed)

    def test_packed_nonfinite_row_rejected_eagerly(self):
        packed = _legacy_packed(
            {"post_block": {3: [{"vector": [math.nan] * HIDDEN, "value": 1.0}]}}
        )
        with pytest.raises(ValueError, match="finite"):
            SteeringClamps.from_obj(packed)

    def test_min_above_max_rejected(self):
        with pytest.raises(ValueError, match="must be <= max"):
            SteeringClamps.from_obj(
                {"post_block": {3: [{"vector": V1, "min": 2.0, "max": 1.0}]}}
            )

    def test_strength_out_of_range(self):
        with pytest.raises(ValueError, match="strength"):
            SteeringClamps.from_obj(
                {"post_block": {3: [{"vector": V1, "value": 1.0, "strength": 1.5}]}}
            )

    def test_value_exclusive_with_bounds(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SteeringClamps.from_obj(
                {"post_block": {3: [{"vector": V1, "value": 1.0, "min": 0.0}]}}
            )

    def test_width_mismatch_within_hook(self):
        with pytest.raises(ValueError, match="width"):
            SteeringClamps.from_obj(
                {
                    "post_block": {
                        3: [{"vector": V1, "value": 1.0}],
                        4: [{"vector": V1[:4], "value": 1.0}],
                    }
                }
            )


class TestAccessors:
    def test_rows_zero_copy_view(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        rows = spec.hooks["post_block"].rows()
        assert rows.base is not None
        assert not rows.flags.writeable

    def test_by_layer_interleaved_order(self):
        table = ClampHookTable(
            shape=[4, 2],
            layer_indices=[5, 2, 5, 2],
            data=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
                dtype=np.float64,
            ).tobytes(),
            lo=[-1.0, -2.0, -3.0, -4.0],
            hi=[1.0, 2.0, 3.0, 4.0],
            strength=[1.0, 0.9, 0.8, 0.7],
        )
        groups = table.by_layer()
        assert [g[0] for g in groups] == [2, 5]
        layer2 = groups[0]
        np.testing.assert_array_equal(layer2[1], [[0.0, 1.0], [0.0, 2.0]])
        np.testing.assert_array_equal(layer2[2], [-2.0, -4.0])
        np.testing.assert_array_equal(layer2[4], [0.9, 0.7])
        layer5 = groups[1]
        np.testing.assert_array_equal(layer5[1], [[1.0, 0.0], [2.0, 0.0]])

    def test_site_counts(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(MULTI_SAME_LAYER))
        assert spec.hooks["post_block"].site_counts() == {3: 2}

    def test_concat_order_and_width_check(self):
        a = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        t1 = a.hooks["post_block"]
        merged = ClampHookTable.concat(t1, t1)
        assert merged.layer_indices == [3, 7, 3, 7]
        np.testing.assert_array_equal(merged.rows()[:2], t1.rows())
        np.testing.assert_array_equal(merged.rows()[2:], t1.rows())
        narrow = SteeringClamps.from_obj(
            {"post_block": {0: [{"vector": [1.0, 2.0], "value": 0.5}]}}
        ).hooks["post_block"]
        with pytest.raises(ValueError, match="widths"):
            ClampHookTable.concat(t1, narrow)

    def test_validate_row_width(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        spec.validate_row_width(HIDDEN)
        with pytest.raises(ValueError, match="hidden size"):
            spec.validate_row_width(16, field_name="steering_clamps")

    def test_check_max_directions(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(MULTI_SAME_LAYER))
        spec.check_max_directions(2)
        with pytest.raises(ValueError, match="max_clamp_directions=1"):
            spec.check_max_directions(1)


class TestTypedWireRoundTrip:
    def test_typed_decode_runs_post_init(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        encoded = msgspec.msgpack.encode(spec)
        decoded = msgspec.msgpack.Decoder(SteeringClamps).decode(encoded)
        assert decoded == spec

    def test_typed_decode_rejects_corrupt_table(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(spec))
        wire["hooks"]["post_block"]["shape"] = [3, HIDDEN]  # data too short
        with pytest.raises(msgspec.ValidationError):
            msgspec.msgpack.Decoder(SteeringClamps).decode(msgspec.msgpack.encode(wire))

    def test_construction_validates(self):
        with pytest.raises(ValueError, match="length"):
            ClampHookTable(
                shape=[2, 2],
                layer_indices=[0],
                data=b"\x00" * 32,
                lo=[0.0, 0.0],
                hi=[1.0, 1.0],
                strength=[1.0, 1.0],
            )


class TestCopySemantics:
    def test_pickle_round_trip(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        assert pickle.loads(pickle.dumps(spec)) == spec

    def test_deepcopy(self):
        spec = SteeringClamps.from_obj(copy.deepcopy(JSON_CANONICAL))
        assert copy.deepcopy(spec) == spec


class TestGoldenDigests:
    """Config hashes must keep matching the parent-commit values for the
    same client submissions (prefix-cache identity across the refactor)."""

    def _hashes(self, clamps, **kwargs):
        sp = SamplingParams(steering_clamps=copy.deepcopy(clamps), **kwargs)
        return sp.prefill_steering_config_hash, sp.decode_steering_config_hash

    def test_json_canonical(self):
        assert self._hashes(JSON_CANONICAL) == (
            GOLDEN_JSON_CANONICAL,
            GOLDEN_JSON_CANONICAL,
        )

    def test_str_keys_match(self):
        assert self._hashes(_str_keys(JSON_CANONICAL)) == (
            GOLDEN_JSON_CANONICAL,
            GOLDEN_JSON_CANONICAL,
        )

    def test_sugar_strength(self):
        assert self._hashes(SUGAR_STRENGTH) == (
            GOLDEN_SUGAR_STRENGTH,
            GOLDEN_SUGAR_STRENGTH,
        )

    def test_inf_bounds(self):
        assert self._hashes(INF_BOUNDS) == (
            GOLDEN_INF_BOUNDS,
            GOLDEN_INF_BOUNDS,
        )

    def test_multi_same_layer(self):
        assert self._hashes(MULTI_SAME_LAYER) == (
            GOLDEN_MULTI_SAME_LAYER,
            GOLDEN_MULTI_SAME_LAYER,
        )

    def test_packed_f64_matches_json(self):
        assert self._hashes(_legacy_packed(JSON_CANONICAL)) == (
            GOLDEN_JSON_CANONICAL,
            GOLDEN_JSON_CANONICAL,
        )

    def test_packed_f32_own_golden(self):
        assert self._hashes(_legacy_packed(JSON_CANONICAL, dtype="float32")) == (
            GOLDEN_PACKED_F32_RAW,
            GOLDEN_PACKED_F32_RAW,
        )

    def test_base_plus_decode_tier(self):
        sp = SamplingParams(
            steering_clamps=copy.deepcopy(JSON_CANONICAL),
            decode_steering_clamps=copy.deepcopy(SUGAR_STRENGTH),
        )
        assert sp.prefill_steering_config_hash == GOLDEN_JSON_CANONICAL
        assert sp.decode_steering_config_hash == GOLDEN_BASE_PLUS_DECODE_DECODE
