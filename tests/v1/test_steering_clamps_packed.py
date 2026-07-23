# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the legacy base64-packed clamp INPUT shape.

The packed dict (`{hook: {dtype, shape, layer_indices, data, bounds,
strengths}}`) remains a public client submission format; ingestion
normalizes it to the canonical ``SteeringClamps`` via ``from_obj``.
Covers structural rejection of malformed blobs, resolve-time merging
with JSON tiers, and SamplingParams-level acceptance/rejection.
(Hash parity between packed and JSON submissions is pinned by the golden
digests in ``test_steering_clamps_types.py``.)
"""

import base64
import math

import numpy as np
import pytest

from vllm.config.steering_types import (
    SteeringClamps,
    resolve_effective_clamps,
)
from vllm.sampling_params import SamplingParams


def _pack_blob(rows, layer_indices, bounds, strengths, dtype="float64"):
    """Build one legacy packed hook blob from row vectors + metadata."""
    arr = np.asarray(rows, dtype=np.dtype(dtype))
    return {
        "dtype": dtype,
        "shape": [arr.shape[0], arr.shape[1]],
        "layer_indices": list(layer_indices),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "bounds": bounds,
        "strengths": strengths,
    }


class TestPackedIngestion:
    def test_basic_ingestion(self):
        packed = {
            "post_attn": _pack_blob(
                rows=[[1.0, 0.0], [0.0, 1.0]],
                layer_indices=[3, 5],
                bounds=[[-2.0, 2.0], [None, 4.0]],
                strengths=[1.0, 0.5],
            )
        }
        spec = SteeringClamps.from_obj(packed)
        table = spec.hooks["post_attn"]
        assert table.layer_indices == [3, 5]
        assert table.rows().tolist() == [[1.0, 0.0], [0.0, 1.0]]
        assert table.lo == [-2.0, -math.inf]
        assert table.hi == [2.0, 4.0]
        assert table.strength == [1.0, 0.5]

    def test_row_order_within_layer_preserved(self):
        packed = {
            "post_attn": _pack_blob(
                rows=[[1.0, 0.0], [0.0, 1.0]],
                layer_indices=[3, 3],
                bounds=[[0.0, 1.0], [1.0, 2.0]],
                strengths=[1.0, 0.5],
            )
        }
        table = SteeringClamps.from_obj(packed).hooks["post_attn"]
        _, dirs, lo, _hi, strength = table.by_layer()[0]
        assert dirs.tolist() == [[1.0, 0.0], [0.0, 1.0]]
        assert lo.tolist() == [0.0, 1.0]
        assert strength.tolist() == [1.0, 0.5]


class TestStructuralValidation:
    def test_bad_base64_length(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["shape"] = [1, 3]  # claims 3 wide but data is 2 wide
        with pytest.raises(ValueError):
            SteeringClamps.from_obj({"post_attn": blob})

    def test_layer_indices_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["layer_indices"] = [0, 1]
        with pytest.raises(ValueError):
            SteeringClamps.from_obj({"post_attn": blob})

    def test_bounds_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["bounds"] = [[0.0, 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError):
            SteeringClamps.from_obj({"post_attn": blob})

    def test_strengths_length_mismatch(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0, 1.0]], [1.0])
        blob["strengths"] = [1.0, 1.0]
        with pytest.raises(ValueError):
            SteeringClamps.from_obj({"post_attn": blob})

    def test_bad_bound_pair(self):
        blob = _pack_blob([[1.0, 0.0]], [0], [[0.0]], [1.0])
        with pytest.raises(ValueError):
            SteeringClamps.from_obj({"post_attn": blob})


class TestResolveUnpacksPacked:
    def test_packed_base_resolves(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        resolved = resolve_effective_clamps(packed, None)
        table = resolved.hooks["post_attn"]
        assert table.layer_indices == [3]
        assert table.rows()[0].tolist() == [1.0, 0.0]

    def test_packed_concat_with_json_phase(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        phase = {"post_attn": {3: [{"vector": [0.0, 1.0], "value": 2.0}]}}
        resolved = resolve_effective_clamps(packed, phase)
        assert resolved.hooks["post_attn"].site_counts()[3] == 2


class TestSamplingParamsPackedValidation:
    def test_packed_field_accepted_structurally(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        sp = SamplingParams(steering_clamps=packed)
        # Ingestion normalizes the legacy packed blob to the canonical
        # SteeringClamps (raw fp64 rows, real ±inf bounds).
        assert isinstance(sp.steering_clamps, SteeringClamps)
        table = sp.steering_clamps.hooks["post_attn"]
        assert table.rows()[0].tolist() == [1.0, 0.0]
        eff = sp.effective_prefill_clamps
        assert eff.hooks["post_attn"].lo == [0.0]

    def test_packed_bad_structure_rejected(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        blob["bounds"] = [[0.0, 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError):
            SamplingParams(steering_clamps={"post_attn": blob})

    def test_packed_bad_hook_rejected(self):
        blob = _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        with pytest.raises(ValueError):
            SamplingParams(steering_clamps={"not_a_hook": blob})

    def test_packed_zero_row_rejected(self):
        blob = _pack_blob([[0.0, 0.0]], [3], [[0.0, 1.0]], [1.0])
        with pytest.raises(ValueError, match="non-zero"):
            SamplingParams(steering_clamps={"post_attn": blob})

    def test_mixed_packed_and_json_tiers(self):
        packed = {"post_attn": _pack_blob([[1.0, 0.0]], [3], [[0.0, 1.0]], [1.0])}
        sp = SamplingParams(
            steering_clamps=packed,
            decode_steering_clamps={
                "post_attn": {3: [{"vector": [0.0, 1.0], "value": 2.0}]}
            },
        )
        eff = sp.effective_decode_clamps
        assert eff.hooks["post_attn"].site_counts()[3] == 2
