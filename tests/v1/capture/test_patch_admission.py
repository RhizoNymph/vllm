# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for patch admission: prefix-floor classification + validation."""

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
