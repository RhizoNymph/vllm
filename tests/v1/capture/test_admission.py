# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared capture-spec admission helpers.

``vllm/v1/capture/admission.py`` is the single source of truth for
resolving a request's ``capture`` dict into the prefix-cache reuse flags
on ``SamplingParams``. Both the OpenAI serving layer (``_admit_capture``)
and the offline ``InputProcessor`` call it, so these tests exercise the
resolution / floor / store-set computation and the error contract once,
independent of either entry point.
"""

from __future__ import annotations

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.capture import CaptureConsumer, CaptureContext, CaptureSpec
from vllm.v1.capture.admission import (
    build_capture_context,
    resolve_capture_prefix_flags,
)
from vllm.v1.capture.errors import (
    CaptureValidationError,
    UnknownCaptureConsumerError,
)


class _PromptConsumer(CaptureConsumer):
    """Resolves to a prompt-touching spec (``last_prompt``)."""

    reads_client_spec = True

    def __init__(self) -> None:  # noqa: D107
        pass

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        return CaptureSpec(hooks={"post_mlp": [0, 1]}, positions="last_prompt")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


class _GeneratedConsumer(CaptureConsumer):
    """Resolves to a generated-only spec (never taps the prompt)."""

    reads_client_spec = True

    def __init__(self) -> None:  # noqa: D107
        pass

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        return CaptureSpec(hooks={"post_mlp": [0]}, positions="all_generated")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


class _RejectingConsumer(CaptureConsumer):
    reads_client_spec = True

    def __init__(self) -> None:  # noqa: D107
        pass

    def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
        raise CaptureValidationError("layer out of range")

    def on_capture(self, key, tensor, sidecar):  # pragma: no cover - unused
        pass


def _ctx(num_prompt_tokens: int = 8) -> CaptureContext:
    return CaptureContext(
        vllm_internal_request_id="req-1",  # type: ignore[arg-type]
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_hidden_layers=32,
        hidden_size=4096,
        element_size_bytes=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


class TestResolveCapturePrefixFlags:
    def test_noop_when_capture_is_none(self) -> None:
        sp = SamplingParams(capture=None)
        resolve_capture_prefix_flags({}, sp, _ctx())
        assert sp.capture is None
        assert sp.capture_touches_prompt is None

    def test_generated_only_keeps_full_prefix_caching(self) -> None:
        sp = SamplingParams(capture={"gen": {}})
        resolve_capture_prefix_flags({"gen": _GeneratedConsumer()}, sp, _ctx())
        # Generated-only: classified, but no prompt floor and no store set,
        # so prefix caching is left fully enabled.
        assert sp.capture_touches_prompt is False
        assert sp.capture_min_prompt_position is None
        assert sp.capture_store_hook_layers is None
        assert sp.capture_store_positions is None

    def test_prompt_touching_records_floor_and_store_set(self) -> None:
        sp = SamplingParams(capture={"fs": {}})
        resolve_capture_prefix_flags({"fs": _PromptConsumer()}, sp, _ctx(8))
        # ``last_prompt`` over an 8-token prompt taps position 7.
        assert sp.capture_touches_prompt is True
        assert sp.capture_min_prompt_position == 7
        # Store serve-set: union of (hook, layer) and prompt positions.
        assert sp.capture_store_hook_layers == [("post_mlp", 0), ("post_mlp", 1)]
        assert sp.capture_store_positions == [7]

    def test_floor_is_min_across_consumers(self) -> None:
        # One consumer taps the whole prompt, another only the last token;
        # the request-wide floor is the lowest tapped position (0).
        class _AllPrompt(CaptureConsumer):
            reads_client_spec = True

            def __init__(self) -> None:
                pass

            def validate_client_spec(self, raw_spec, ctx):  # type: ignore[override]
                return CaptureSpec(hooks={"post_mlp": [0]}, positions="all_prompt")

            def on_capture(self, key, tensor, sidecar):  # pragma: no cover
                pass

        sp = SamplingParams(capture={"a": {}, "b": {}})
        resolve_capture_prefix_flags(
            {"a": _AllPrompt(), "b": _PromptConsumer()}, sp, _ctx(8)
        )
        assert sp.capture_touches_prompt is True
        assert sp.capture_min_prompt_position == 0
        assert sp.capture_store_positions == list(range(8))

    def test_unknown_consumer_raises_with_param(self) -> None:
        sp = SamplingParams(capture={"missing": {}})
        with pytest.raises(UnknownCaptureConsumerError) as ei:
            resolve_capture_prefix_flags({}, sp, _ctx())
        assert "missing" in str(ei.value)
        assert ei.value.capture_param == "capture.missing"  # type: ignore[attr-defined]

    def test_invalid_spec_raises_with_param(self) -> None:
        sp = SamplingParams(capture={"fs": {"bad": True}})
        with pytest.raises(CaptureValidationError) as ei:
            resolve_capture_prefix_flags({"fs": _RejectingConsumer()}, sp, _ctx())
        assert "layer out of range" in str(ei.value)
        assert ei.value.capture_param == "capture.fs"  # type: ignore[attr-defined]


class TestBuildCaptureContext:
    def test_reads_shape_from_vllm_config(self) -> None:
        class _ParallelConfig:
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            data_parallel_size = 1

        class _Dtype:
            itemsize = 2

        class _ModelConfig:
            @staticmethod
            def get_total_num_hidden_layers() -> int:
                return 40

            @staticmethod
            def get_hidden_size() -> int:
                return 5120

            dtype = _Dtype()

        class _VllmConfig:
            parallel_config = _ParallelConfig()
            model_config = _ModelConfig()

        ctx = build_capture_context(_VllmConfig(), num_prompt_tokens=11, request_id="r")
        assert ctx.vllm_internal_request_id == "r"
        assert ctx.num_prompt_tokens == 11
        assert ctx.num_computed_tokens == 0
        assert ctx.num_hidden_layers == 40
        assert ctx.hidden_size == 5120
        assert ctx.element_size_bytes == 2
        assert ctx.tensor_parallel_size == 2
        assert ctx.pipeline_parallel_size == 1
