# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache gating for capture-bearing requests (roadmap step B).

A capture tap only produces a residual for a token position that is
actually forwarded through the model, so a prefix-cache hit (which skips
the forward pass for cached prefix tokens) conflicts with capture *only*
when the capture taps a prompt-range position. These tests pin the
classification helper, the relaxed ``SamplingParams`` construction, and
the ``Request`` gate that together keep prefix caching for generated-only
captures while still skipping it for prompt-touching ones.
"""

from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.capture import CaptureSpec, spec_touches_prompt
from vllm.v1.request import Request


def _spec(positions: object) -> CaptureSpec:
    return CaptureSpec(hooks={"post_mlp": [0]}, positions=positions)  # type: ignore[arg-type]


def _request(sampling_params: SamplingParams) -> Request:
    return Request(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=sampling_params,
        pooling_params=None,
    )


class TestSpecTouchesPrompt:
    """``spec_touches_prompt`` classifies a resolved spec against the prompt."""

    def test_all_generated_never_touches_prompt(self) -> None:
        assert (
            spec_touches_prompt(_spec("all_generated"), num_prompt_tokens=10) is False
        )

    def test_all_always_touches_prompt(self) -> None:
        assert spec_touches_prompt(_spec("all"), num_prompt_tokens=10) is True

    @pytest.mark.parametrize("symbolic", ["all_prompt", "last_prompt", "bogus"])
    def test_unresolved_symbolic_is_conservative(self, symbolic: str) -> None:
        # If a consumer leaves these symbolic (or returns an unknown
        # selector), the safe answer is "touches prompt".
        assert spec_touches_prompt(_spec(symbolic), num_prompt_tokens=10) is True

    def test_explicit_positions_all_generated(self) -> None:
        # Every index is at or past the prompt boundary -> generated-only.
        assert spec_touches_prompt(_spec([10, 11, 25]), num_prompt_tokens=10) is False

    def test_explicit_positions_with_prompt_index(self) -> None:
        assert spec_touches_prompt(_spec([9, 10, 11]), num_prompt_tokens=10) is True

    def test_empty_explicit_positions(self) -> None:
        # Nothing captured -> no conflict.
        assert spec_touches_prompt(_spec([]), num_prompt_tokens=10) is False


class TestSamplingParamsConstruction:
    """Construction no longer force-disables prefix caching on capture."""

    def test_capture_does_not_force_skip(self) -> None:
        sp = SamplingParams(capture={"logging": {"level": "INFO"}})
        # Capture alone must not flip the cache off at construction; that
        # decision now waits for admission classification.
        assert sp.skip_reading_prefix_cache is False

    def test_capture_touches_prompt_defaults_none(self) -> None:
        sp = SamplingParams(capture={"logging": {"level": "INFO"}})
        assert sp.capture_touches_prompt is None

    def test_prompt_logprobs_still_skips(self) -> None:
        sp = SamplingParams(capture={"logging": {}}, prompt_logprobs=1)
        assert sp.skip_reading_prefix_cache is True

    def test_explicit_skip_is_respected(self) -> None:
        sp = SamplingParams(capture={"logging": {}}, skip_reading_prefix_cache=False)
        assert sp.skip_reading_prefix_cache is False


class TestRequestGate:
    """``Request.get_skip_reading_prefix_cache`` consults the classification."""

    def test_prompt_touching_skips(self) -> None:
        sp = SamplingParams(capture={"logging": {}})
        sp.capture_touches_prompt = True
        assert _request(sp).get_skip_reading_prefix_cache() is True

    def test_generated_only_keeps_cache(self) -> None:
        sp = SamplingParams(capture={"logging": {}})
        sp.capture_touches_prompt = False
        assert _request(sp).get_skip_reading_prefix_cache() is False

    def test_unclassified_is_conservative(self) -> None:
        # The offline ``LLM`` path leaves this ``None`` -> skip (safe).
        sp = SamplingParams(capture={"logging": {}})
        assert sp.capture_touches_prompt is None
        assert _request(sp).get_skip_reading_prefix_cache() is True

    def test_no_capture_defaults_false(self) -> None:
        sp = SamplingParams()
        assert _request(sp).get_skip_reading_prefix_cache() is False

    def test_no_capture_explicit_skip_true(self) -> None:
        sp = SamplingParams(skip_reading_prefix_cache=True)
        assert _request(sp).get_skip_reading_prefix_cache() is True
