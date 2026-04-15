# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the activation-storing OpenAI entrypoint protocol.

Scope
-----
These tests cover phase 5 of the activation-storing roadmap: the
``activation_storing`` per-request field on ``ChatCompletionRequest`` /
``CompletionRequest``, admission validation at the entrypoint boundary
with HTTP 400 translation, attachment of the validated spec to
``SamplingParams.activation_storing``, and serialization of the
runner-produced capture result back into the HTTP response body.

Mock engine strategy
--------------------
Phase 4 (runner integration) is landing in a sibling worktree and is not
merged yet. Phase 4 adds ``RequestOutput.activation_storage`` which this
module reads off the final ``RequestOutput`` when building the response.
Rather than block phase 5 on phase 4's merge, these tests stand up a
mock ``AsyncLLM`` that:

1. Matches the subset of the ``EngineClient`` protocol the serving layer
   touches (``model_config``, ``vllm_config``, ``errored``, ``renderer``,
   ``io_processor``, ``input_processor``, ``generate``).
2. Yields ``RequestOutput`` instances whose ``activation_storage``
   attribute is set manually via ``setattr`` on a local ``_CaptureResult``
   shim. Once phase 4 lands and the real ``CaptureResult`` is defined in
   ``vllm.config.activation_storing_types``, these tests continue to
   work because the helper's ``status`` / ``error`` / ``paths`` field set
   matches the serialization contract the serving layer consumes.

Integration tests exercising the real runner path will live in
``tests/v1/capture/test_runner_integration.py`` once phase 4 lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from pydantic import ValidationError

from vllm.config.activation_storing import ActivationStoringConfig
from vllm.config.activation_storing_types import ActivationStoringSpec
from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ActivationStorageResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


# ---------------------------------------------------------------------------
# CaptureResult shim
# ---------------------------------------------------------------------------
#
# Phase 4 will add ``CaptureResult`` to
# ``vllm.config.activation_storing_types`` alongside a new
# ``RequestOutput.activation_storage`` attribute. Until phase 4 lands, we
# simulate it here with a matching structural shape so the serving-layer
# serializer can walk it without caring which module it came from. The
# serving layer only reads ``.status``, ``.error``, and ``.paths``.


@dataclass
class _CaptureResult:
    status: str
    error: str | None = None
    paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock model / parallel / vllm config
# ---------------------------------------------------------------------------
#
# Structurally matches the fields the serving layer reads. Importantly,
# ``vllm_config.activation_storing_config`` must be set (with a non-``None``
# ``root_path``) for the validator to let the request through.


@dataclass
class _MockHFConfig:
    model_type: str = "any"


@dataclass
class _MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 128
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = _MockHFConfig()
    hf_text_config = _MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    # The activation-storing validator reads these via helper methods.
    num_hidden_layers: int = 12
    hidden_size: int = 32
    dtype: torch.dtype = torch.bfloat16

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}

    def get_total_num_hidden_layers(self) -> int:
        return self.num_hidden_layers

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1


@dataclass
class _MockVllmConfig:
    model_config: _MockModelConfig
    parallel_config: _MockParallelConfig
    activation_storing_config: ActivationStoringConfig | None = None


def _make_vllm_config(
    activation_storing_config: ActivationStoringConfig | None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> _MockVllmConfig:
    return _MockVllmConfig(
        model_config=_MockModelConfig(),
        parallel_config=_MockParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        activation_storing_config=activation_storing_config,
    )


def _build_renderer(model_config: _MockModelConfig):
    return HfRenderer(
        _MockVllmConfig(model_config, parallel_config=_MockParallelConfig()),
        cached_tokenizer_from_config(model_config),
    )


def _build_mock_engine(
    *,
    ast_config: ActivationStoringConfig | None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> MagicMock:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = _MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.io_processor = MagicMock()
    mock_engine.renderer = _build_renderer(mock_engine.model_config)
    mock_engine.vllm_config = _make_vllm_config(
        ast_config,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    return mock_engine


# ---------------------------------------------------------------------------
# Request/engine input helpers
# ---------------------------------------------------------------------------


def _fake_engine_input(prompt_token_ids: list[int] | None = None) -> dict[str, Any]:
    return {"prompt_token_ids": prompt_token_ids or [1, 2, 3, 4, 5]}


def _patch_chat_rendering(serving_chat: OpenAIServingChat) -> None:
    """Replace the rendering pipeline so tests don't touch a real tokenizer."""

    async def _fake_render(request: ChatCompletionRequest):
        return (
            [{"role": "user", "content": "hello"}],
            [_fake_engine_input()],
        )

    serving_chat.openai_serving_render.render_chat = AsyncMock(  # type: ignore[method-assign]
        side_effect=_fake_render
    )


def _patch_completion_rendering(serving_completion: OpenAIServingCompletion) -> None:
    async def _fake_render(request: CompletionRequest):
        return [_fake_engine_input()]

    serving_completion.openai_serving_render.render_completion = AsyncMock(  # type: ignore[method-assign]
        side_effect=_fake_render
    )


def _build_serving_chat(engine: AsyncLLM) -> OpenAIServingChat:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        io_processor=engine.io_processor,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving_chat = OpenAIServingChat(
        engine,
        models,
        response_role="assistant",
        openai_serving_render=serving_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    _patch_chat_rendering(serving_chat)
    return serving_chat


def _build_serving_completion(engine: AsyncLLM) -> OpenAIServingCompletion:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    serving_render = OpenAIServingRender(
        model_config=engine.model_config,
        renderer=engine.renderer,
        io_processor=engine.io_processor,
        model_registry=models.registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving_completion = OpenAIServingCompletion(
        engine,
        models,
        openai_serving_render=serving_render,
        request_logger=None,
    )
    _patch_completion_rendering(serving_completion)
    return serving_completion


def _build_request_output(
    *,
    prompt_token_ids: list[int],
    text: str = "world",
    token_ids: tuple[int, ...] = (42, 43),
    capture: _CaptureResult | None = None,
    finish_reason: str | None = "stop",
) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=finish_reason,
    )
    ro = RequestOutput(
        request_id="test-id",
        prompt="hello",
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
    )
    # Phase 4 adds this attribute; we set it directly so phase 5 tests
    # don't need phase 4 to be merged.
    if capture is not None:
        ro.activation_storage = capture  # type: ignore[attr-defined]
    return ro


def _install_generate(mock_engine: MagicMock, request_outputs: list[RequestOutput]):
    async def mock_generate(*args: Any, **kwargs: Any):
        for ro in request_outputs:
            yield ro

    mock_engine.generate = MagicMock(side_effect=mock_generate)


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _good_spec() -> dict[str, Any]:
    return {
        "request_id": "probe_0001",
        "tag": "unit-test",
        "hooks": {"post_mlp": [0, 1, 2]},
        "positions": "last_prompt",
    }


def _enabled_ast_config(tmp_path) -> ActivationStoringConfig:
    return ActivationStoringConfig(root_path=str(tmp_path))


# ---------------------------------------------------------------------------
# ChatCompletionRequest protocol-shape tests
# ---------------------------------------------------------------------------


class TestChatCompletionRequestSchema:
    def test_defaults_to_none(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert req.activation_storing is None

    def test_accepts_spec(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "activation_storing": _good_spec(),
            }
        )
        assert isinstance(req.activation_storing, ActivationStoringSpec)
        assert req.activation_storing.request_id == "probe_0001"
        assert req.activation_storing.tag == "unit-test"
        assert req.activation_storing.hooks == {"post_mlp": [0, 1, 2]}
        assert req.activation_storing.positions == "last_prompt"

    def test_to_sampling_params_attaches_spec(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "activation_storing": _good_spec(),
            }
        )
        sp = req.to_sampling_params(max_tokens=8, default_sampling_params={})
        assert isinstance(sp.activation_storing, ActivationStoringSpec)
        assert sp.activation_storing.request_id == "probe_0001"

    def test_to_sampling_params_none_when_absent(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        sp = req.to_sampling_params(max_tokens=8, default_sampling_params={})
        assert sp.activation_storing is None


# ---------------------------------------------------------------------------
# CompletionRequest protocol-shape tests
# ---------------------------------------------------------------------------


class TestCompletionRequestSchema:
    def test_defaults_to_none(self):
        req = CompletionRequest.model_validate({"model": MODEL_NAME, "prompt": "hi"})
        assert req.activation_storing is None

    def test_accepts_spec(self):
        req = CompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "prompt": "hi",
                "activation_storing": _good_spec(),
            }
        )
        assert isinstance(req.activation_storing, ActivationStoringSpec)

    def test_to_sampling_params_attaches_spec(self):
        req = CompletionRequest.model_validate(
            {
                "model": MODEL_NAME,
                "prompt": "hi",
                "activation_storing": _good_spec(),
            }
        )
        sp = req.to_sampling_params(max_tokens=8)
        assert isinstance(sp.activation_storing, ActivationStoringSpec)
        assert sp.activation_storing.tag == "unit-test"

    def test_to_sampling_params_none_when_absent(self):
        req = CompletionRequest.model_validate({"model": MODEL_NAME, "prompt": "hi"})
        sp = req.to_sampling_params(max_tokens=8)
        assert sp.activation_storing is None


# ---------------------------------------------------------------------------
# ActivationStorageResponse model
# ---------------------------------------------------------------------------


class TestActivationStorageResponseModel:
    def test_minimum_fields(self):
        resp = ActivationStorageResponse(status="ok")
        assert resp.status == "ok"
        assert resp.error is None
        assert resp.paths == []

    def test_all_fields(self):
        resp = ActivationStorageResponse(
            status="partial_error",
            error="disk full at layer 3",
            paths=["/mnt/a.bin", "/mnt/a.json"],
        )
        assert resp.status == "partial_error"
        assert resp.error == "disk full at layer 3"
        assert resp.paths == ["/mnt/a.bin", "/mnt/a.json"]

    def test_rejects_unknown_status(self):
        with pytest.raises(ValidationError):
            ActivationStorageResponse(status="weird")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chat completion happy-path (non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completion_attaches_activation_storage_on_success(tmp_path):
    """A request with a valid spec returns an ``activation_storage`` pointer."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    capture = _CaptureResult(
        status="ok",
        error=None,
        paths=[
            str(tmp_path / "probe_0001_layer0.bin"),
            str(tmp_path / "probe_0001_layer0.json"),
        ],
    )
    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=capture,
    )
    _install_generate(mock_engine, [request_output])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ChatCompletionResponse), response
    assert response.activation_storage is not None
    assert response.activation_storage.status == "ok"
    assert response.activation_storage.error is None
    assert response.activation_storage.paths == capture.paths

    # Engine was called with the spec attached to sampling params.
    assert mock_engine.generate.called
    call_kwargs = mock_engine.generate.call_args.args
    sampling_params = call_kwargs[1]
    assert isinstance(sampling_params.activation_storing, ActivationStoringSpec)
    assert sampling_params.activation_storing.request_id == "probe_0001"


@pytest.mark.asyncio
async def test_chat_completion_absent_field_omits_activation_storage(tmp_path):
    """A request WITHOUT ``activation_storing`` gets a ``None`` pointer."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=None,
    )
    _install_generate(mock_engine, [request_output])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ChatCompletionResponse)
    assert response.activation_storage is None


@pytest.mark.asyncio
async def test_chat_completion_partial_error_propagates(tmp_path):
    """A writer-level partial_error surfaces verbatim in the response."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    capture = _CaptureResult(
        status="partial_error",
        error="layer 1 write timed out after 180s",
        paths=[str(tmp_path / "probe_0001_layer0.bin")],
    )
    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=capture,
    )
    _install_generate(mock_engine, [request_output])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ChatCompletionResponse)
    assert response.activation_storage is not None
    assert response.activation_storage.status == "partial_error"
    assert response.activation_storage.error == "layer 1 write timed out after 180s"
    assert response.activation_storage.paths == capture.paths


@pytest.mark.asyncio
async def test_chat_completion_missing_capture_attribute_reports_error(tmp_path):
    """When the runner forgets to set ``activation_storage`` we surface error."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=None,  # attribute missing
    )
    _install_generate(mock_engine, [request_output])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ChatCompletionResponse)
    assert response.activation_storage is not None
    assert response.activation_storage.status == "error"
    assert "engine did not return" in (response.activation_storage.error or "")


# ---------------------------------------------------------------------------
# Chat completion admission failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_rejects_when_feature_disabled(tmp_path):
    """Activation storing disabled at server start → HTTP 400."""
    mock_engine = _build_mock_engine(ast_config=None)
    serving_chat = _build_serving_chat(mock_engine)

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert "activation storing enabled" in response.error.message


@pytest.mark.asyncio
async def test_chat_rejects_unknown_layer(tmp_path):
    """A layer index above ``num_hidden_layers`` yields a descriptive 400."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    spec = _good_spec()
    spec["hooks"] = {"post_mlp": [99]}  # 12 layers in mock → 99 is OOB
    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": spec,
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert "out of range" in response.error.message


@pytest.mark.asyncio
async def test_chat_rejects_byte_budget(tmp_path):
    """When estimated capture size exceeds the cap, return HTTP 400."""
    ast_config = ActivationStoringConfig(
        root_path=str(tmp_path),
        max_bytes_per_request=1,  # 1 byte cap → any capture blows past
    )
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert "max-bytes-per-request" in response.error.message


@pytest.mark.asyncio
async def test_chat_rejects_tp_greater_than_one(tmp_path):
    """TP > 1 is out of scope for v1; admission returns HTTP 400."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config, tensor_parallel_size=2)
    serving_chat = _build_serving_chat(mock_engine)

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST
    assert "tensor_parallel_size=1" in response.error.message


# ---------------------------------------------------------------------------
# Chat completion streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_streaming_final_frame_carries_activation_storage(tmp_path):
    """Streaming requests surface the pointer in the final SSE frame."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    capture = _CaptureResult(
        status="ok",
        error=None,
        paths=[str(tmp_path / "probe_0001_layer0.bin")],
    )
    # Two chunks: the first yields one token, the second has finish_reason
    # and the capture result attached.
    chunk_1 = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        text="wor",
        token_ids=(42,),
        finish_reason=None,
        capture=None,
    )
    # Non-terminal chunk: it needs finished=False for streaming semantics.
    chunk_1.finished = False
    chunk_2 = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        text="ld",
        token_ids=(43,),
        finish_reason="stop",
        capture=capture,
    )
    _install_generate(mock_engine, [chunk_1, chunk_2])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_chat.create_chat_completion(request)
    assert not isinstance(response, ErrorResponse)

    chunks: list[str] = []
    async for c in response:
        chunks.append(c)

    # Find the final-usage frame (it's the second-to-last entry before
    # [DONE]). Parse it and check the ``activation_storage`` field.
    assert chunks[-1] == "data: [DONE]\n\n"
    final_frame = chunks[-2]
    assert final_frame.startswith("data: ")
    payload = json.loads(final_frame[len("data: ") :].rstrip())
    assert "activation_storage" in payload, payload
    assert payload["activation_storage"]["status"] == "ok"
    assert payload["activation_storage"]["paths"] == capture.paths


@pytest.mark.asyncio
async def test_chat_streaming_without_spec_does_not_add_field(tmp_path):
    """A streaming request without a spec does not include the field."""
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_chat = _build_serving_chat(mock_engine)

    chunk = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        finish_reason="stop",
        capture=None,
    )
    _install_generate(mock_engine, [chunk])

    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    response = await serving_chat.create_chat_completion(request)
    assert not isinstance(response, ErrorResponse)

    chunks: list[str] = []
    async for c in response:
        chunks.append(c)

    # Scan every emitted frame: none of them should contain
    # ``activation_storage``.
    for frame in chunks:
        if frame.startswith("data: ") and frame != "data: [DONE]\n\n":
            payload = json.loads(frame[len("data: ") :].rstrip())
            assert "activation_storage" not in payload, payload


# ---------------------------------------------------------------------------
# Legacy completions parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_attaches_activation_storage_on_success(tmp_path):
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_completion = _build_serving_completion(mock_engine)

    capture = _CaptureResult(
        status="ok",
        error=None,
        paths=[str(tmp_path / "probe_0001_layer0.bin")],
    )
    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=capture,
    )
    _install_generate(mock_engine, [request_output])

    request = CompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "prompt": "hi",
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_completion.create_completion(request)

    assert isinstance(response, CompletionResponse), response
    assert response.activation_storage is not None
    assert response.activation_storage.status == "ok"
    assert response.activation_storage.paths == capture.paths


@pytest.mark.asyncio
async def test_completion_absent_field_omits_activation_storage(tmp_path):
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_completion = _build_serving_completion(mock_engine)

    request_output = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        capture=None,
    )
    _install_generate(mock_engine, [request_output])

    request = CompletionRequest.model_validate(
        {"model": MODEL_NAME, "prompt": "hi", "max_tokens": 8, "stream": False}
    )
    response = await serving_completion.create_completion(request)

    assert isinstance(response, CompletionResponse)
    assert response.activation_storage is None


@pytest.mark.asyncio
async def test_completion_rejects_when_feature_disabled(tmp_path):
    mock_engine = _build_mock_engine(ast_config=None)
    serving_completion = _build_serving_completion(mock_engine)

    request = CompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "prompt": "hi",
            "max_tokens": 8,
            "stream": False,
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_completion.create_completion(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_completion_streaming_final_frame_carries_pointer(tmp_path):
    ast_config = _enabled_ast_config(tmp_path)
    mock_engine = _build_mock_engine(ast_config=ast_config)
    serving_completion = _build_serving_completion(mock_engine)

    capture = _CaptureResult(
        status="ok",
        error=None,
        paths=[str(tmp_path / "probe_0001_layer0.bin")],
    )
    chunk_1 = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        text="wor",
        token_ids=(42,),
        finish_reason=None,
        capture=None,
    )
    chunk_1.finished = False
    chunk_2 = _build_request_output(
        prompt_token_ids=[1, 2, 3, 4, 5],
        text="ld",
        token_ids=(43,),
        finish_reason="stop",
        capture=capture,
    )
    _install_generate(mock_engine, [chunk_1, chunk_2])

    request = CompletionRequest.model_validate(
        {
            "model": MODEL_NAME,
            "prompt": "hi",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
            "activation_storing": _good_spec(),
        }
    )
    response = await serving_completion.create_completion(request)
    assert not isinstance(response, ErrorResponse)

    chunks: list[str] = []
    async for c in response:
        chunks.append(c)

    assert chunks[-1] == "data: [DONE]\n\n"
    final_frame = chunks[-2]
    assert final_frame.startswith("data: ")
    payload = json.loads(final_frame[len("data: ") :].rstrip())
    assert payload.get("activation_storage", {}).get("status") == "ok"
    assert payload["activation_storage"]["paths"] == capture.paths
