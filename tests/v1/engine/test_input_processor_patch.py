# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline patch admission in the v1 ``InputProcessor``.

The OpenAI serving layer resolves a request's ``patch`` spec into prefix-cache
reuse flags in ``_admit_patch``. Offline / direct ``LLM`` requests never pass
through that layer, so the ``InputProcessor`` runs the same resolution
(``_resolve_patch_prefix_flags``) before the request reaches the scheduler.
Unlike the best-effort capture path, an invalid offline patch spec raises
``ValueError`` (the engine rejects the request). These tests cover that offline
path directly, bypassing the heavy ``InputProcessor.__init__``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


def _entry(layer=2, hook="post_block", dest=1, run="R1", src=1, alpha=1.0):
    return {
        "layer": layer,
        "hook": hook,
        "dest_position": dest,
        "source_run": run,
        "source_position": src,
        "alpha": alpha,
    }


def _stub_vllm_config(patch_config):
    """Minimal ``vllm_config`` exposing what patch admission reads."""
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
    )
    model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 8,
        get_hidden_size=lambda: 16,
        dtype=SimpleNamespace(itemsize=2),
    )
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        patch_config=patch_config,
    )


def _processor(patch_config) -> InputProcessor:
    proc = InputProcessor.__new__(InputProcessor)
    proc.vllm_config = _stub_vllm_config(patch_config)
    return proc


class TestOfflinePatchPrefixFlags:
    def test_stamps_floor_for_prompt_touching_spec(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(patch=[_entry(dest=3), _entry(dest=1)])

        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 1  # lowest patched prompt pos

    def test_generated_only_no_floor(self) -> None:
        # dest positions >= num_prompt -> generated range, no prefix clamp.
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(patch=[_entry(dest=12), _entry(dest=15)])

        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

        assert sp.patch_touches_prompt is False
        assert sp.patch_min_prompt_position is None

    def test_idempotent_when_already_stamped(self) -> None:
        # A request admitted by the serving layer arrives already stamped; the
        # ``process_inputs`` gate (patch_touches_prompt is None) skips it, but
        # even calling the resolver again is a stable no-op given the same
        # spec/shape (it recomputes the same floor).
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(patch=[_entry(dest=3)])
        sp.patch_touches_prompt = True
        sp.patch_min_prompt_position = 3

        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 3

    def test_invalid_spec_raises(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        # layer 99 is out of range for an 8-layer model.
        sp = SamplingParams(patch=[_entry(layer=99)])

        with pytest.raises(ValueError, match="out of range"):
            proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

    def test_bad_hook_raises(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(patch=[_entry(hook="not_a_hook")])

        with pytest.raises(ValueError, match="not injectable"):
            proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

    def test_overflow_raises(self) -> None:
        # More distinct dest positions at one site than usable slots.
        proc = _processor(SimpleNamespace(max_patch_slots=2))
        sp = SamplingParams(
            patch=[_entry(dest=0), _entry(dest=1)]  # 2 > usable (max-1 = 1)
        )

        with pytest.raises(ValueError, match="slots"):
            proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

    def test_patch_disabled_raises(self) -> None:
        # No patch_config -> patching is not enabled; a spec is a hard error.
        proc = _processor(None)
        sp = SamplingParams(patch=[_entry()])

        with pytest.raises(ValueError, match="patching is not enabled"):
            proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)


def _pack(rows, width):
    import numpy as np
    import pybase64 as base64

    arr = np.zeros((rows, width), dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": [rows, width],
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


class TestOfflineNewSourceKinds:
    """Offline parity for module / inline sources: structural validation +
    floors. Named modules pass structurally (no frontend registry offline)."""

    def test_named_module_passes_structurally_and_stamps_floor(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(
            patch=[
                {
                    "layer": 2,
                    "hook": "post_block",
                    "dest_position": 4,
                    "source_module": "anything",  # not registry-checked offline
                }
            ]
        )
        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)
        assert sp.patch_touches_prompt is True
        assert sp.patch_min_prompt_position == 4

    def test_zeros_module_ok(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(
            patch=[
                {
                    "layer": 1,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_module": "zeros",
                }
            ]
        )
        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)
        assert sp.patch_touches_prompt is True

    def test_inline_width_match_ok(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(
            patch=[
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 16),
        )
        proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)
        assert sp.patch_touches_prompt is True

    def test_inline_width_mismatch_raises(self) -> None:
        proc = _processor(SimpleNamespace(max_patch_slots=64))
        sp = SamplingParams(
            patch=[
                {
                    "layer": 0,
                    "hook": "post_block",
                    "dest_position": 0,
                    "source_inline": 0,
                }
            ],
            patch_vectors=_pack(1, 4),  # != hidden_size 16
        )
        with pytest.raises(ValueError, match="width"):
            proc._resolve_patch_prefix_flags("r", sp, [0] * 10, None)

    def test_inline_index_out_of_range_raises(self) -> None:
        # source_inline out of range is structural -> raises at construction.
        with pytest.raises(ValueError, match="out of range"):
            SamplingParams(
                patch=[
                    {
                        "layer": 0,
                        "hook": "post_block",
                        "dest_position": 0,
                        "source_inline": 9,
                    }
                ],
                patch_vectors=_pack(1, 16),
            )
