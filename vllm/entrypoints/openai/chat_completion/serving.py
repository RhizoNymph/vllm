# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pybase64 as base64
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    get_tool_call_id_type,
    make_tool_call_id,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    CaptureResultResponse,
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.inputs import EngineInput, MultiModalPlaceholders
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tool_parser
from vllm.v1.capture import (
    CaptureConsumer,
    CaptureValidationError,
    UnknownCaptureConsumerError,
)
from vllm.v1.capture import registry as capture_registry
from vllm.v1.capture.admission import (
    build_capture_context,
    resolve_capture_prefix_flags,
)

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)


def _capture_result_to_response_payload(payload: Any) -> dict[str, Any]:
    """Coerce an opaque consumer payload into a JSON-serializable dict.

    The framework's ``CaptureResult.payload`` is intentionally
    unconstrained — filesystem consumers typically produce a list of
    paths, other consumers may return ``None``, a dashboard URL, or
    arbitrary structured data. Responses need a dict for stable JSON
    emission, so:

    - ``dict`` flows through unchanged.
    - ``None`` becomes ``{}`` (consumer declined to share a payload).
    - ``list`` becomes ``{"items": <list>}``. The filesystem consumer
      emits ``list[Path]``; converting ``Path`` to ``str`` keeps the
      dict JSON-safe.
    - Everything else is wrapped as ``{"value": <payload>}``.
    """

    def _coerce(p: Any) -> Any:
        # Results that crossed the engine-IPC boundary (late finalize,
        # ``capture_wait``) are msgspec round-tripped: ``Path`` objects
        # arrive as ``bytes``/``str``, not ``Path`` -- so don't rely on
        # ``__fspath__`` alone. Anything non-JSON-primitive is stringified.
        if isinstance(p, bytes):
            return p.decode("utf-8", errors="replace")
        if isinstance(p, (str, int, float, bool)) or p is None:
            return p
        return str(p)

    if payload is None:
        return {}
    if isinstance(payload, dict):
        return {str(k): _coerce(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return {"items": [_coerce(p) for p in payload]}
    return {"value": _coerce(payload)}


def _build_capture_results_response(
    request: ChatCompletionRequest,
    final_res: RequestOutput,
) -> dict[str, CaptureResultResponse] | None:
    """Convert per-consumer capture results into the response dict.

    Returns ``None`` when the dict is empty so serializers omit the
    response field entirely, keeping the payload small for the common
    uncaptured request.
    """
    results = getattr(final_res, "capture_results", None)
    if not results:
        # The request finished generating before its captures finalized
        # (writes are asynchronous -- a request's files may land seconds
        # after the response on slow filesystems). If the request opted
        # into capture, surface that explicitly instead of omitting the
        # field, so clients can tell "capture pending" from "no capture".
        requested = getattr(request, "capture", None)
        if requested:
            return {
                name: CaptureResultResponse(
                    status="pending",
                    error=None,
                    payload={
                        "detail": (
                            "capture admitted; results are written "
                            "asynchronously and had not finalized when the "
                            "response was generated"
                        )
                    },
                )
                for name in requested
            }
        return None
    response: dict[str, CaptureResultResponse] = {}
    for name, result in results.items():
        response[name] = CaptureResultResponse(
            status=result.status,
            error=result.error,
            payload=_capture_result_to_response_payload(result.payload),
        )
    return response


def _get_mm_token_counts(engine_input: EngineInput) -> dict[str, int]:
    """Sum per-modality placeholder tokens from ``mm_placeholders``.

    Keyed by modality name; ``PlaceholderRange.length`` is the placeholder's
    prompt token span, so each sum matches the placeholder tokens already
    counted in ``usage.prompt_tokens``.
    """
    mm_placeholders = cast(
        "MultiModalPlaceholders | None", engine_input.get("mm_placeholders")
    )
    return {
        modality: sum(p.length for p in ranges)
        for modality, ranges in (mm_placeholders or {}).items()
        if ranges
    }


def _messages_have_multimodal(messages) -> bool:
    """True if any message carries a non-text content part (image/audio/...).

    Used to reject patch+multimodal BEFORE chat rendering: rendering registers
    each image in the frontend's multimodal sender cache, so rejecting after
    it leaves the cache claiming the engine holds items it never received —
    poisoning later requests that reuse the same image.
    """
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") != "text":
                return True
    return False


def _make_prompt_tokens_details(
    enable_prompt_tokens_details: bool,
    num_cached_tokens: int | None,
    mm_token_counts: dict[str, int] | None,
) -> PromptTokenUsageInfo | None:
    """Build ``prompt_tokens_details`` from cached + multimodal token counts."""
    if not enable_prompt_tokens_details:
        return None
    if num_cached_tokens is None and not mm_token_counts:
        return None
    return PromptTokenUsageInfo(
        cached_tokens=num_cached_tokens,
        multimodal_tokens=mm_token_counts or None,
    )


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        self.enable_auto_tools: bool = enable_auto_tools
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
        )
        if (
            self.parser_cls is not None
            and is_mistral_tool_parser(self.parser_cls.tool_parser_cls)
            and self.parser_cls.reasoning_parser_cls is not None
        ):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            MistralToolParser.model_can_reason = True

        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage

        # Build a capture-consumer validator cache at serving-layer init so
        # per-request ``validate_client_spec`` calls do not re-import or
        # re-instantiate consumers. Used only for admission validation — not
        # for dispatching captures. Keyed exactly as the runner's
        # ``name_to_index`` (see ``build_admission_validators``) so a client's
        # ``capture={<name>: ...}`` resolves to the same consumer here and at
        # the worker.
        self._capture_consumers: dict[str, CaptureConsumer] = (
            capture_registry.build_admission_validators(self.engine_client.vllm_config)
        )
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.tool_call_id_type = get_tool_call_id_type(self.model_config)

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        parser: Parser | None = None
        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
            )
        # Patch + multimodal is rejected BEFORE rendering (see
        # _messages_have_multimodal for why after is too late); the
        # placeholder-count check in _admit_patch stays as a backstop.
        if getattr(request, "patch", None) and _messages_have_multimodal(
            request.messages
        ):
            return self.create_error_response(
                "activation patching is not supported with multimodal "
                "prompts (positions include image placeholder tokens); "
                "use a text-only prompt",
                status_code=HTTPStatus.BAD_REQUEST,
                param="patch",
            )

        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        # Named steering module: validate name exists, then pass a
        # ``(name, scale)`` reference through to ``SamplingParams`` so
        # the worker resolves against its broadcast registry instead of
        # the server materializing the full vector spec into the
        # multiprocessing payload (perf optimization, see PR 7 of
        # the activation-steering plan).  Inline overrides ride along
        # unmodified on ``request.steering_vectors`` etc.
        steering_module_ref: tuple[str, float] | None = None
        if request.steering_name is not None:
            steering_registry = (
                None
                if raw_request is None
                else getattr(raw_request.app.state, "steering_module_registry", None)
            )
            if steering_registry is None:
                return self.create_error_response(
                    "Named steering modules are not available. "
                    "Ensure the server was started with steering enabled.",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            if steering_registry.get(request.steering_name) is None:
                return self.create_error_response(
                    (
                        f"Unknown steering module '{request.steering_name}'. "
                        f"Available: {steering_registry.list_modules() or 'none'}"
                    ),
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            steering_module_ref = (request.steering_name, 1.0)

        # Build the request-metadata channel once (declarative steering gate
        # names are resolved against the vector registry here; a malformed
        # gate spec or unknown name raises ValueError → HTTP 400).
        steering_vector_registry = (
            None
            if raw_request is None
            else getattr(raw_request.app.state, "steering_vector_registry", None)
        )
        from vllm.entrypoints.openai.steering import (
            monitor_writes_gates_from_request,
        )

        monitor_writes_gates = monitor_writes_gates_from_request(raw_request)
        try:
            req_metadata_channel = request.to_request_metadata(
                vector_registry=steering_vector_registry,
                monitor_writes_gates=monitor_writes_gates,
            )
        except ValueError as exc:
            return self.create_error_response(
                f"Invalid steering gate spec: {exc}",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        mm_token_counts: dict[str, int] | None = None
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            mm_token_counts = _get_mm_token_counts(engine_input)

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
                # Attach the named-module reference (validated above).
                # Worker resolves the (name, scale) tuple against its
                # broadcast registry; we never serialize the resolved
                # vectors over multiprocessing.
                if steering_module_ref is not None:
                    sampling_params.steering_module_ref = steering_module_ref

            # Per-request capture-consumer admission validation. Runs
            # AFTER sampling-params construction (we need the tokenized
            # prompt length) and BEFORE the request is handed to the
            # engine so admission failures never consume a slot.
            if isinstance(sampling_params, SamplingParams) and sampling_params.capture:
                error_response = self._admit_capture(
                    sampling_params=sampling_params,
                    engine_input=engine_input,
                    request_id=sub_request_id,
                )
                if error_response is not None:
                    return error_response

            if isinstance(sampling_params, SamplingParams) and sampling_params.patch:
                from vllm.v1.capture.patch_admission import (
                    make_named_module_existence,
                )

                error_response = self._admit_patch(
                    sampling_params=sampling_params,
                    engine_input=engine_input,
                    request_id=sub_request_id,
                    named_module_exists=make_named_module_existence(raw_request),
                )
                if error_response is not None:
                    return error_response
                # Reject (HTTP 400) if a referenced patch source run/site does
                # not exist, instead of silently completing unpatched.
                from vllm.v1.capture.patch_admission import (
                    PatchValidationError,
                    validate_patch_sources,
                )

                try:
                    await validate_patch_sources(self.engine_client, sampling_params)
                except PatchValidationError as exc:
                    return self.create_error_response(
                        str(exc),
                        status_code=HTTPStatus.BAD_REQUEST,
                        param="patch",
                    )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                if not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_tool_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif parser is not None and parser.reasoning_parser is not None:
                    reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if parser is not None and parser.reasoning_parser is not None
                    else None,
                    request_metadata=req_metadata_channel,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )

    def _admit_capture(
        self,
        *,
        sampling_params: SamplingParams,
        engine_input: EngineInput,
        request_id: str,
    ) -> ErrorResponse | None:
        """Run admission validation for the per-request capture dict.

        Iterates each ``(consumer_name, raw_spec)`` entry in
        ``sampling_params.capture``, looking the consumer up in the
        serving-layer cache and calling its
        ``validate_client_spec(raw, ctx)``. The raw payload is left in
        place (the worker re-validates it; see the NOTE below); on
        success the resolved positions are used to set
        ``sampling_params.capture_touches_prompt``. On failure, an HTTP
        400 ``ErrorResponse`` is returned and the request is rejected
        before reaching the engine.
        """
        if sampling_params.capture is None:
            return None

        try:
            num_prompt_tokens = self._extract_prompt_len(engine_input)
        except Exception as exc:  # pragma: no cover - defensive
            return self.create_error_response(
                f"capture: failed to determine prompt length: {exc}",
                status_code=HTTPStatus.BAD_REQUEST,
                param="capture",
            )

        try:
            ctx = build_capture_context(
                self.engine_client.vllm_config, num_prompt_tokens, request_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self.create_error_response(
                f"capture: failed to read model shape: {exc}",
                status_code=HTTPStatus.BAD_REQUEST,
                param="capture",
            )

        # Resolve every per-request spec and stamp the prefix-cache reuse
        # flags onto ``sampling_params`` (shared with the offline
        # ``InputProcessor`` path). The raw ``capture`` dict is left in place:
        # ``CaptureSpec`` is not IPC-serializable, so the worker re-validates
        # from the original dict after scheduling.
        try:
            resolve_capture_prefix_flags(self._capture_consumers, sampling_params, ctx)
        except (UnknownCaptureConsumerError, CaptureValidationError) as exc:
            return self.create_error_response(
                str(exc),
                status_code=HTTPStatus.BAD_REQUEST,
                param=getattr(exc, "capture_param", "capture"),
            )
        return None

    def _admit_patch(
        self,
        *,
        sampling_params: SamplingParams,
        engine_input: EngineInput,
        request_id: str,
        named_module_exists: Callable[[str], bool] | None = None,
    ) -> ErrorResponse | None:
        """Validate the patch spec and stamp prefix-cache flags."""
        from vllm.v1.capture.patch_admission import (
            PatchValidationError,
            resolve_patch_prefix_flags,
        )

        if not sampling_params.patch:
            return None
        # Multimodal prompts are unsupported: prompt positions include image
        # placeholder tokens, so patch positions would target placeholder
        # activations — semantically undefined and unvalidated. Fail loud.
        if _get_mm_token_counts(engine_input):
            return self.create_error_response(
                "activation patching is not supported with multimodal "
                "prompts (positions include image placeholder tokens); "
                "use a text-only prompt",
                status_code=HTTPStatus.BAD_REQUEST,
                param="patch",
            )
        try:
            num_prompt_tokens = self._extract_prompt_len(engine_input)
            ctx = build_capture_context(
                self.engine_client.vllm_config, num_prompt_tokens, request_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self.create_error_response(
                f"patch: failed to read request shape: {exc}",
                status_code=HTTPStatus.BAD_REQUEST,
                param="patch",
            )
        patch_config = getattr(self.engine_client.vllm_config, "patch_config", None)
        max_patch_slots = (
            getattr(patch_config, "max_patch_slots", 0) if patch_config else 0
        )
        try:
            resolve_patch_prefix_flags(
                sampling_params,
                ctx,
                max_patch_slots=max_patch_slots,
                named_module_exists=named_module_exists,
            )
        except PatchValidationError as exc:
            return self.create_error_response(
                str(exc), status_code=HTTPStatus.BAD_REQUEST, param="patch"
            )
        return None

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        previous_texts = [""] * num_choices

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    for _ in range(num_choices)
                ]
                for p in parsers:
                    if p is not None:
                        # NOTE: HarmonyParser ignores _stream_state (uses its own FSM).
                        p._stream_state.tool_call_id_type = self.tool_call_id_type
                        p._stream_state.history_tool_call_cnt = history_tool_call_cnt
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    if parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                            finished=output.finish_reason is not None,
                        )
                        if delta_message is not None:
                            if delta_message.tool_calls:
                                tools_streamed[i] = True

                            if (
                                delta_message.reasoning
                                and not request.include_reasoning
                            ):
                                delta_message.reasoning = None
                                if not (
                                    delta_message.content or delta_message.tool_calls
                                ):
                                    delta_message = None

                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                final_usage.prompt_tokens_details = _make_prompt_tokens_details(
                    self.enable_prompt_tokens_details,
                    num_cached_tokens,
                    mm_token_counts,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        parser: Parser | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        tool_parser_cls = (
            self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
        )
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if parser is not None:
                reasoning, content, tool_calls = parser.parse(
                    output.text,
                    request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=token_ids,
                )
                if not request.include_reasoning:
                    reasoning = None
            else:
                reasoning = None
                content = output.text
                tool_calls = []

            auto_tools_called = False

            if (not self.enable_auto_tools or not tool_parser_cls) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                tool_call_items = []
                tool_calls = tool_calls or []
                for tc in tool_calls:
                    if not tc.id:
                        tc.id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tc.name,
                            idx=history_tool_call_cnt,
                        )
                    tool_call_items.append(ToolCall(id=tc.id, function=tc))
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content or "",
                    tool_calls=tool_call_items,
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_items = []
                tool_calls = tool_calls or []
                for tool_call in tool_calls:
                    if not tool_call.id:
                        tool_call.id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tool_call.name,
                            idx=history_tool_call_cnt,
                        )
                    tool_call_items.append(
                        ToolCall(id=tool_call.id, function=tool_call)
                    )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content=content or "",
                    tool_calls=tool_call_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and tool_parser_cls
            ):
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    tool_call_items = []
                    for tc in tool_calls:
                        if not tc.id:
                            tc.id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                        tool_call_items.append(ToolCall(id=tc.id, function=tc))
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_items,
                    )

                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        usage.prompt_tokens_details = _make_prompt_tokens_details(
            self.enable_prompt_tokens_details,
            final_res.num_cached_tokens,
            mm_token_counts,
        )

        request_metadata.final_usage_info = usage

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        if (
            request.capture
            and getattr(request, "capture_wait", False)
            and not final_res.capture_results
            and hasattr(self.engine_client, "wait_for_capture_results")
        ):
            late = await self.engine_client.wait_for_capture_results(
                final_res.request_id
            )
            if late:
                final_res.capture_results = late

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
            capture_results=_build_capture_results_response(request, final_res),
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = format_token_id_placeholder(token_id)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
