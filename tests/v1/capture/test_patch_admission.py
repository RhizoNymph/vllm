# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for patch admission: prefix-floor classification + validation."""

import asyncio

import pytest

from vllm import SamplingParams
from vllm.v1.capture.patch_admission import (
    PatchValidationError,
    resolve_patch_prefix_flags,
)
from vllm.v1.capture.types import CaptureContext


def _ctx(num_prompt: int, num_layers: int = 8) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id="req",
        num_prompt_tokens=num_prompt,
        num_computed_tokens=0,
        num_hidden_layers=num_layers,
        hidden_size=16,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def _entry(layer=2, hook="post_block", dest=1, run="R1", src=1, alpha=1.0):
    return {
        "layer": layer,
        "hook": hook,
        "dest_position": dest,
        "source_run": run,
        "source_position": src,
        "alpha": alpha,
    }


class TestPrefixFloor:
    def test_prompt_patch_sets_floor(self):
        sp = SamplingParams(patch=[_entry(dest=3), _entry(dest=1)])
        resolve_patch_prefix_flags(sp, _ctx(num_prompt=10), max_patch_slots=64)
        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 1  # lowest patched prompt pos

    def test_generated_only_no_floor(self):
        # dest positions >= num_prompt -> generated range, no prefix clamp.
        sp = SamplingParams(patch=[_entry(dest=12), _entry(dest=15)])
        resolve_patch_prefix_flags(sp, _ctx(num_prompt=10), max_patch_slots=64)
        assert sp.patch_touches_prompt is False
        assert sp.patch_min_prompt_position is None

    def test_mixed_floor_is_lowest_prompt(self):
        sp = SamplingParams(patch=[_entry(dest=12), _entry(dest=4)])
        resolve_patch_prefix_flags(sp, _ctx(num_prompt=10), max_patch_slots=64)
        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 4

    def test_no_patch_noop(self):
        sp = SamplingParams()
        resolve_patch_prefix_flags(sp, _ctx(num_prompt=10), max_patch_slots=64)
        assert sp.patch_touches_prompt is None


class TestValidation:
    def test_bad_hook(self):
        sp = SamplingParams(patch=[_entry(hook="mlp_out")])
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_layer_out_of_range(self):
        sp = SamplingParams(patch=[_entry(layer=99)])
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10, num_layers=8), max_patch_slots=64)

    def test_negative_dest(self):
        sp = SamplingParams(patch=[_entry(dest=-1)])
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_single_request_capacity_overflow(self):
        # 3 sites at the same (layer, hook) but pool only fits 2 usable slots.
        sp = SamplingParams(
            patch=[_entry(dest=d, layer=2, hook="post_block") for d in range(3)]
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=3)

    def test_capacity_ok_across_distinct_sites(self):
        # Same count but spread across distinct (layer,hook) sites -> fine.
        sp = SamplingParams(
            patch=[
                _entry(dest=0, layer=1, hook="post_block"),
                _entry(dest=0, layer=2, hook="post_block"),
                _entry(dest=0, layer=3, hook="pre_attn"),
            ]
        )
        resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=3)
        assert sp.patch_touches_prompt is True


class TestPatchSiteDemand:
    def test_counts_positions_per_site(self):
        sp = SamplingParams(
            patch=[
                _entry(layer=2, hook="post_block", dest=0),
                _entry(layer=2, hook="post_block", dest=1),
                _entry(layer=3, hook="pre_attn", dest=0),
            ]
        )
        assert sp.patch_site_demand == {(2, "post_block"): 2, (3, "pre_attn"): 1}

    def test_empty_when_no_patch(self):
        assert SamplingParams().patch_site_demand == {}


def _pack(rows, width, dtype="float32"):
    import numpy as np
    import pybase64 as base64

    arr = np.zeros((rows, width), dtype=np.float32)
    return {
        "dtype": dtype,
        "shape": [rows, width],
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _sp_http(patch, patch_vectors=None):
    """Mirror the HTTP path: patch / patch_vectors are set post-construction,
    so structural validation first runs at admission (not at construction)."""
    sp = SamplingParams()
    sp.patch = patch
    sp.patch_vectors = patch_vectors
    return sp


class TestSamplingParamsStructural:
    """Construction-time (offline / direct) structural validation."""

    def test_run_kind_ok(self):
        SamplingParams(patch=[_entry()])  # source_run + source_position

    def test_module_kind_ok(self):
        SamplingParams(
            patch=[
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "m",
                }
            ]
        )

    def test_inline_kind_ok(self):
        SamplingParams(
            patch=[
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 8),
        )

    def test_no_source_kind_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(
                patch=[{"layer": 0, "hook": "post_block", "dest_position": 0}]
            )

    def test_two_source_kinds_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(
                patch=[
                    {
                        "layer": 0,
                        "hook": "post_block",
                        "dest_position": 0,
                        "source_run": "R1",
                        "source_position": 0,
                        "source_module": "m",
                    }
                ]
            )

    def test_inline_without_table_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(
                patch=[
                    {
                        "layer": 0,
                        "hook": "post_block",
                        "dest_position": 0,
                        "source_inline": 0,
                    }
                ]
            )

    def test_mask_both_forms_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(
                patch=[
                    {
                        "layer": 0,
                        "hook": "post_block",
                        "dest_position": 0,
                        "source_module": "zeros",
                        "mask": {"indices": [1], "inline": 0},
                    }
                ],
                patch_vectors=_pack(1, 8),
            )

    def test_bad_patch_vectors_dtype_rejected(self):
        with pytest.raises(ValueError):
            SamplingParams(patch_vectors={"dtype": "int8", "shape": [1, 4], "data": ""})


class TestNewSourceKindsAdmission:
    def test_unknown_source_module_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "x",
                }
            ]
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(
                sp, _ctx(10), max_patch_slots=64, named_module_exists=lambda n: False
            )

    def test_known_source_module_passes(self):
        sp = _sp_http(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 3,
                    "source_module": "x",
                }
            ]
        )
        resolve_patch_prefix_flags(
            sp, _ctx(10), max_patch_slots=64, named_module_exists=lambda n: n == "x"
        )
        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 3  # floor from dest, source-agnostic

    def test_zeros_module_never_registry_checked(self):
        sp = _sp_http(
            [
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "zeros",
                }
            ]
        )
        # callable rejects everything, but "zeros" bypasses the check.
        resolve_patch_prefix_flags(
            sp, _ctx(10), max_patch_slots=64, named_module_exists=lambda n: False
        )

    def test_inline_width_mismatch_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 4),  # width 4 != ctx.hidden_size 16
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_inline_width_match_passes(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 16),
        )
        resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_bad_base64_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors={"dtype": "float32", "shape": [1, 16], "data": "@@notb64"},
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_shape_mismatch_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 16) | {"shape": [2, 16]},  # data too short
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_inline_index_out_of_range_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 5,
                }
            ],
            patch_vectors=_pack(1, 16),
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_mask_negative_index_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "zeros",
                    "mask": {"indices": [-1]},
                }
            ]
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_two_source_kinds_rejected(self):
        sp = _sp_http(
            [
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "zeros",
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 16),
        )
        with pytest.raises(PatchValidationError):
            resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)

    def test_floor_source_kind_independent(self):
        # A vector-sourced (zeros) prompt patch still stamps the prefix floor.
        sp = _sp_http(
            [
                {
                    "layer": 1,
                    "hook": "post_block",
                    "dest_position": 5,
                    "source_module": "zeros",
                }
            ]
        )
        resolve_patch_prefix_flags(sp, _ctx(10), max_patch_slots=64)
        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 5


class _FakeEngine:
    """Minimal engine_client stub: collective_rpc returns per-rank manifests."""

    def __init__(self, ranks_manifests, *, fail=False):
        self._ranks = ranks_manifests
        self.fail = fail
        self.calls = 0

    async def collective_rpc(self, method, *a, **k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rpc down")
        return self._ranks


def _manifest(run, hook_layers, positions):
    return {
        "run_id": run,
        "num_prompt_tokens": max(positions) + 1,
        "hidden_size": 8,
        "hook_layers": [[hook, layer] for (hook, layer) in hook_layers],
        "positions": list(positions),
    }


def _patch_sp(run="R1", hook="post_block", layer=2, src=1):
    return SamplingParams(
        patch=[_entry(layer=layer, hook=hook, dest=0, run=run, src=src)]
    )


class TestPatchSourceCache:
    def _cache(self):
        from vllm.v1.capture.patch_admission import _PatchSourceCache

        return _PatchSourceCache()

    def test_present_source_passes(self):
        cache = self._cache()
        eng = _FakeEngine([[_manifest("R1", [("post_block", 2)], [0, 1, 2])]])
        asyncio.run(cache.validate(_patch_sp(src=1), eng))  # no raise
        # cold cache: one manifest refresh + one eviction-lease RPC
        assert eng.calls == 2

    def test_missing_run_rejects(self):
        from vllm.v1.capture.patch_admission import PatchValidationError

        cache = self._cache()
        eng = _FakeEngine([[_manifest("R1", [("post_block", 2)], [0, 1])]])
        with pytest.raises(PatchValidationError):
            asyncio.run(cache.validate(_patch_sp(run="GHOST"), eng))

    def test_missing_position_rejects(self):
        from vllm.v1.capture.patch_admission import PatchValidationError

        cache = self._cache()
        eng = _FakeEngine([[_manifest("R1", [("post_block", 2)], [0, 1])]])
        with pytest.raises(PatchValidationError):
            asyncio.run(cache.validate(_patch_sp(src=9), eng))

    def test_positive_cache_avoids_second_rpc(self):
        cache = self._cache()
        eng = _FakeEngine([[_manifest("R1", [("post_block", 2)], [0, 1, 2])]])
        asyncio.run(cache.validate(_patch_sp(src=1), eng))
        asyncio.run(cache.validate(_patch_sp(src=2), eng))
        # 2 = cold manifest refresh + one lease; the second validate is served
        # from cache AND its lease is still fresh (throttled to half-TTL), so
        # it issues no RPC at all — a sweep stays ~O(1) RPCs total.
        assert eng.calls == 2

    def test_lease_rpc_throttled_across_many_validates(self):
        cache = self._cache()
        eng = _FakeEngine([[_manifest("R1", [("post_block", 2)], [0, 1, 2])]])
        for src in (0, 1, 2, 0, 1, 2):
            asyncio.run(cache.validate(_patch_sp(src=src), eng))
        assert eng.calls == 2  # manifest + single lease, regardless of cells

    def test_rpc_failure_is_best_effort(self):
        cache = self._cache()
        eng = _FakeEngine([], fail=True)
        # Must not raise even though the run is "missing" — admission proceeds.
        asyncio.run(cache.validate(_patch_sp(run="R1"), eng))

    def test_pp_union_across_ranks(self):
        # Rank 0 has layer 2, rank 1 has layer 5 of the same run.
        cache = self._cache()
        eng = _FakeEngine(
            [
                [_manifest("R1", [("post_block", 2)], [0, 1])],
                [_manifest("R1", [("post_block", 5)], [0, 1])],
            ]
        )
        # site on rank 1's layer resolves via the union.
        asyncio.run(cache.validate(_patch_sp(layer=5, src=1), eng))
