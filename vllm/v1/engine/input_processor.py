# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from functools import cached_property
from typing import Any, Literal

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.steering_geometry import (
    SteeringSiteGeometry,
    resolve_steering_site_geometry,
)
from vllm.config.utils import hash_factors
from vllm.exceptions import SteeringSiteValidationError, VLLMValidationError
from vllm.inputs import (
    EngineInput,
    PromptType,
    SingletonInput,
    split_enc_dec_input,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.multimodal.utils import argsort_mm_positions
from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.async_utils import make_async
from vllm.utils.diffusion import validate_diffusion_sampling_params
from vllm.utils.jsontree import json_iter_leaves
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.request_metadata import RequestMetadata

logger = init_logger(__name__)


class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config
        self.steering_config = vllm_config.steering_config
        self.diffusion_config = vllm_config.diffusion_config
        # Load the custom logits processor classes once; the returned callable
        # runs their validate_params hooks per request at admission.
        self.validate_logits_processors_params = (
            self._build_logits_processors_params_validator()
        )

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(vllm_config)

        self.supports_mm_inputs = model_config.supports_multimodal_inputs
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        if self.supports_mm_inputs:
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore

        # per-request capture specs into prefix-cache flags before the
        # request reaches the scheduler (the offline analogue of the OpenAI
        # serving layer's ``_admit_capture``). Built lazily on the first
        # capture request; stays ``None`` for non-capture workloads so they
        # pay nothing. See ``vllm/v1/capture/admission.py``.
        self._capture_consumers: dict[str, Any] | None = None
        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers should run it on the renderer's
        # thread pool to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _build_logits_processors_params_validator(
        self,
    ) -> Callable[[SamplingParams], None]:
        """Load the custom logits processor classes and return the per-request
        params validator."""
        custom_logitsprocs = self.model_config.logits_processors
        if self.model_config.runner_type == "pooling" and not custom_logitsprocs:
            return lambda _: None

        from vllm.v1.worker.gpu.sample.logits_processor import (
            build_custom_logits_processors_params_validator,
        )

        return build_custom_logits_processors_params_validator(custom_logitsprocs)

    def resolve_watermarking(self, params: SamplingParams | BeamSearchParams) -> bool:
        """Return whether watermarking will be applied to this request.

        `None` and `True` request watermarking, while `False` opts out.
        If watermarking is unavailable or incompatible with the request, emit
        a warning and return `False`.
        """
        return self.vllm_config._check_supports_watermarking(params)

    def apply_watermarking(self, params: SamplingParams, watermarking: bool) -> None:
        """Store the resolved `watermarking` on request-local params.

        Also records whether watermarking was requested on a watermarking
        engine but could not be applied, as opposed to an explicit opt-out.
        """
        # Sticky, so resolving an already resolved request keeps the flag.
        params._watermarking_skipped = params._watermarking_skipped or (
            self.vllm_config.watermark_config is not None
            and params.watermarking is not False
            and not watermarking
        )
        params.watermarking = watermarking

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
    ) -> bool | None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid.

        Returns the resolved `watermarking` value for `SamplingParams` (`None`
        for `PoolingParams`). The caller applies it to its private copy of the
        params so that a caller-owned params object is never mutated.
        """
        watermarking: bool | None = None
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise VLLMValidationError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )
            if (
                params.prompt_logprob_token_ids is not None
                and self.vllm_config.cache_config.kv_sharing_fast_prefill
            ):
                raise VLLMValidationError(
                    "prompt_logprob_token_ids is incorrect with "
                    "--kv-sharing-fast-prefill; disable it for scoring.",
                    parameter="prompt_logprob_token_ids",
                )

            self.validate_logits_processors_params(params)

            if self.model_config.is_diffusion:
                # Without --diffusion-config the served canvas is unknown here;
                # the ids and the read-only normalisation are still checked.
                validate_diffusion_sampling_params(
                    params,
                    canvas_length=(
                        self.diffusion_config.canvas_length
                        if self.diffusion_config is not None
                        else None
                    ),
                    vocab_size=self.model_config.get_vocab_size(),
                )

            if self.model_config.return_sampling_mask:
                if params.temperature <= 0:
                    raise ValueError(
                        "sampling distribution replay requires temperature > 0"
                    )
                if params.top_k <= 0:
                    raise ValueError(
                        "sampling distribution replay requires top_k > 0 to "
                        "bound sampling mask size, reduce transfer overhead, "
                        "and avoid potential OOMs"
                    )
            if params.thinking_token_budget is not None and (
                self.vllm_config.reasoning_config is None
                or not self.vllm_config.reasoning_config.enabled
            ):
                raise VLLMValidationError(
                    "thinking_token_budget is set but reasoning_config is "
                    "not configured. Please set --reasoning-parser "
                    "and/or --reasoning-config to use thinking_token_budget."
                )
            if (
                params.trace_decode_token_ids
                and not self.model_config.enable_trace_replay
            ):
                raise VLLMValidationError(
                    "trace_decode_token_ids is set but trace replay is not "
                    "enabled. Start the engine with --enable-trace-replay "
                    "to use it."
                )
            # Resolved last so that feature gates above report the missing
            # feature rather than its incompatibility with watermarking.
            watermarking = self.resolve_watermarking(params)
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise VLLMValidationError("This model does not support pooling")

            if params.task is None:
                if "token_embed" in supported_pooling_tasks:
                    params.task = "token_embed"
                elif "token_classify" in supported_pooling_tasks:
                    params.task = "token_classify"
                elif "plugin" in supported_pooling_tasks:
                    params.task = "plugin"

            if params.task not in supported_pooling_tasks:
                raise VLLMValidationError(
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

        return watermarking

    def _normalize_trace_replay_params(
        self, sampling_params: SamplingParams, prompt_len: int
    ) -> None:
        """Apply trace replay's generation semantics to request-local params."""
        trace_token_ids = sampling_params.trace_decode_token_ids
        assert trace_token_ids
        assert sampling_params.max_tokens is not None

        max_trace_len = max(self.model_config.max_model_len - prompt_len, 1)
        trace_token_ids = trace_token_ids[:max_trace_len]
        sampling_params.trace_decode_token_ids = trace_token_ids

        # Apply this after the generation config so its EOS token cannot stop
        # replay before the trace is exhausted.
        sampling_params.max_tokens = min(
            len(trace_token_ids), sampling_params.max_tokens
        )
        sampling_params.min_tokens = 0
        sampling_params.ignore_eos = True
        sampling_params._eos_token_id = None
        sampling_params.stop = []
        sampling_params.stop_token_ids = []
        sampling_params._all_stop_token_ids = set()

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise VLLMValidationError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    @cached_property
    def steering_site_geometry(self) -> SteeringSiteGeometry | None:
        """Per-hook row widths + layer count of the model's steering sites.

        Resolved once, on first use; ``None`` iff steering is disabled.
        Every seam admitting client steering specs (this processor, the
        AsyncLLM / LLM inline packing) validates against it.
        """
        if not self.steering_config:
            return None
        return resolve_steering_site_geometry(self.model_config)

    def _validate_steering(self, params: SamplingParams | PoolingParams) -> None:
        if not isinstance(params, SamplingParams):
            return
        has_steering = (
            params.steering_vectors is not None
            or params.prefill_steering_vectors is not None
            or params.decode_steering_vectors is not None
            or params.steering_module_ref is not None
            or params.sae_clamp_specs is not None
            or params.sae_full_reconstruction_specs is not None
            or params._effective_prefill_steering_packed is not None
            or params._effective_decode_steering_packed is not None
        )
        has_clamps = (
            params.steering_clamps is not None
            or params.prefill_steering_clamps is not None
            or params.decode_steering_clamps is not None
        )
        if not has_steering and not has_clamps:
            return
        if not self.steering_config:
            raise VLLMValidationError(
                "Per-request steering vectors/clamps, named-module "
                "references, or SAE clamp specs were provided but steering "
                "is not enabled. Start the server with --enable-steering "
                "to use per-request steering."
            )
        # Site gate: SamplingParams validation is model-blind, and a
        # wrong-width row shape-crashes the worker's steering table
        # population (an out-of-range layer is dropped silently) — reject
        # here (frontend process, request-level error) on every path:
        # online HTTP, offline LLM(), and the Rust frontend. Each hook is
        # checked against its own width (mHC stream hooks: hc * hidden).
        from vllm.config.steering_types import resolve_effective_clamps

        geometry = self.steering_site_geometry
        assert geometry is not None, "resolved whenever steering is enabled"
        # ``steering_types`` raises ValueError so it can double as a pydantic
        # validator; re-raise as VLLMValidationError so the entrypoints map
        # these to 400 rather than letting them escape as a 500. The site
        # gate's SteeringSiteValidationError already is one and passes as-is.
        try:
            for field_name, spec in (
                ("steering_vectors", params.steering_vectors),
                ("prefill_steering_vectors", params.prefill_steering_vectors),
                ("decode_steering_vectors", params.decode_steering_vectors),
            ):
                geometry.validate_vectors(spec, field_name=field_name)
            if has_clamps:
                max_dirs = int(getattr(self.steering_config, "max_clamp_directions", 0))
                if max_dirs <= 0:
                    raise ValueError(
                        "Per-request clamps were provided but clamping is "
                        "disabled (steering_config.max_clamp_directions=0)."
                    )
                for field_name, cspec in (
                    ("steering_clamps", params.steering_clamps),
                    ("prefill_steering_clamps", params.prefill_steering_clamps),
                    ("decode_steering_clamps", params.decode_steering_clamps),
                ):
                    geometry.validate_clamps(cspec, field_name=field_name)
                # Per-site K cap after the tier concat — reject over-budget
                # requests here (request-level error) rather than crashing the
                # step thread at manager materialization.
                resolve_effective_clamps(
                    params.steering_clamps,
                    params.prefill_steering_clamps,
                    max_directions=max_dirs,
                )
                resolve_effective_clamps(
                    params.steering_clamps,
                    params.decode_steering_clamps,
                    max_directions=max_dirs,
                )
        except SteeringSiteValidationError:
            raise
        except ValueError as exc:
            raise VLLMValidationError(str(exc)) from exc

    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: LoRARequest | None,
    ) -> str:
        """When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the adapter name and path to prevent incorrect
        cache hits when the same name is loaded from a different path.
        Paths must identify immutable adapter versions for persistent cache reuse.
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        adapter_hash = hash_factors(
            {"lora_name": lora_request.lora_name, "lora_path": lora_request.lora_path}
        )
        return f"{adapter_hash}:{mm_hash}"

    def inject_into_mm_cache(
        self,
        mm_hashes: dict[str, list[str]],
        mm_kwargs: dict[str, list],
    ) -> None:
        """Inject pre-processed mm_kwargs into the processor cache.

        Call this when mm_kwargs have already been through the HF processor
        externally (e.g. by a frontend that transfers pre-processed tensors
        to the backend).  This ensures MM cache hit rate metrics are reported
        accurately and avoids redundant processing on subsequent requests
        with the same images.

        Uses `get_and_update_item()` with an empty prompt_updates list,
        since token expansion has already been handled externally.
        """
        cache = self.renderer.mm_processor_cache
        if cache is None:
            return
        try:
            for modality, hashes in mm_hashes.items():
                items = mm_kwargs.get(modality, [])
                for i, mm_hash in enumerate(hashes):
                    if i < len(items) and items[i] is not None:
                        # Insert into cache via get_and_update_item.
                        # Use the returned item (may be an address for SHM
                        # cache or the original item for LRU cache).
                        items[i], _ = cache.get_and_update_item(
                            (items[i], []),
                            mm_hash,
                        )
            # Update cache stats to reflect the externally processed items
            self.renderer.update_mm_cache_stats()
        except Exception:
            logger.warning(
                "Failed to inject mm_kwargs into processor cache",
                exc_info=True,
            )

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
        request_metadata: RequestMetadata | None = None,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
    ) -> EngineCoreRequest:
        watermarking = self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)
        self._validate_steering(params)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise VLLMValidationError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]

            engine_input: EngineInput = prompt  # type: ignore[assignment]
        else:
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in the future. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            if arrival_time is None:
                arrival_time = time.time()

            renderer = self.renderer
            model_config = self.model_config

            parsed_prompt = parse_model_prompt(model_config, prompt)
            tok_params = renderer.default_cmpl_tok_params.with_kwargs(
                **(tokenization_kwargs or {})
            )

            (engine_input,) = renderer.render_cmpl(
                [parsed_prompt],
                tok_params,
            )

        encoder_input, decoder_input = split_enc_dec_input(engine_input)
        try:
            current_platform.validate_request(engine_input, params)
            self._validate_model_inputs(encoder_input, decoder_input)
        except Exception:
            # Rendering recorded the multimodal items as sent to the engine,
            # which a rejected request never does: resend them next time.
            self.renderer.queue_mm_cache_invalidations(
                [
                    mm_hash
                    for singleton in (encoder_input, decoder_input)
                    if singleton is not None and singleton["type"] == "multimodal"
                    for mm_hashes in singleton["mm_hashes"].values()
                    for mm_hash in mm_hashes
                ]
            )
            raise

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_input["type"] == "embeds":
            prompt_embeds = decoder_input["prompt_embeds"]
            prompt_token_ids = decoder_input.get("prompt_token_ids")
            prompt_is_token_ids = decoder_input.get("is_token_ids")
        else:
            prompt_token_ids = decoder_input["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # Resolve on the request-local copy: `params` may be shared across
            # prompts, requests and even engines by the caller.
            assert watermarking is not None
            self.apply_watermarking(sampling_params, watermarking)
            prompt_len = length_from_prompt_token_ids_or_embeds(
                prompt_token_ids, prompt_embeds
            )
            if not 0 <= sampling_params.routed_experts_prompt_start <= prompt_len:
                raise VLLMValidationError(
                    f"routed_experts_prompt_start must be between 0 and "
                    f"the prompt length ({prompt_len}), inclusive.",
                    parameter="routed_experts_prompt_start",
                    value=sampling_params.routed_experts_prompt_start,
                )
            start = sampling_params.prompt_logprob_start or 0
            if start and start >= prompt_len - 1:
                raise VLLMValidationError(
                    f"prompt_logprob_start must be less than prompt_len - 1 "
                    f"({prompt_len - 1}) so at least one prompt row is scored.",
                    parameter="prompt_logprob_start",
                    value=start,
                )
            rows = sampling_params.prompt_logprob_token_ids
            if rows is not None:
                num_rows = max(prompt_len - 1 - start, 0)
                if len(rows) != num_rows:
                    raise VLLMValidationError(
                        f"prompt_logprob_token_ids has {len(rows)} rows, but the "
                        f"prompt has {num_rows} scored rows "
                        f"(prompt_len - 1 - prompt_logprob_start).",
                        parameter="prompt_logprob_token_ids",
                        value=len(rows),
                    )
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                sampling_params.max_tokens = (
                    self.model_config.max_model_len - prompt_len
                )
                # min_tokens is not checked while max_tokens is unset.
                if sampling_params.min_tokens > sampling_params.max_tokens:
                    raise VLLMValidationError(
                        f"min_tokens must be less than or equal to "
                        f"max_tokens={sampling_params.max_tokens}, got "
                        f"{sampling_params.min_tokens}."
                    )

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
            if sampling_params.trace_decode_token_ids:
                self._normalize_trace_replay_params(sampling_params, prompt_len)
        else:
            pooling_params = params.clone()

        # Offline capture admission: resolve the per-request capture spec into
        # prefix-cache reuse flags (the OpenAI path does this in
        # ``_admit_capture``). Idempotent — served requests arrive with the
        # flag already set, so this only fires for offline / direct ``LLM``
        # requests, which never pass through the serving-layer admission step.
        if (
            sampling_params is not None
            and sampling_params.capture is not None
            and sampling_params.capture_touches_prompt is None
        ):
            self._resolve_capture_prefix_flags(
                request_id,
                sampling_params,
                prompt_token_ids,
                prompt_embeds,
            )

        # Offline patch admission: resolve the per-request patch spec into
        # prefix-cache reuse flags (the OpenAI path does this in
        # ``_admit_patch``). Idempotent — served requests arrive already
        # stamped and skip this; it only fires for offline / direct ``LLM``
        # requests. Unlike capture's best-effort resolution, an invalid offline
        # patch spec raises ``ValueError`` (rejecting the request): patch
        # entries carry explicit positions validated against the model's real
        # shape, so a bad spec is a hard client error.
        if (
            sampling_params is not None
            and sampling_params.patch
            and sampling_params.patch_touches_prompt is None
        ):
            self._resolve_patch_prefix_flags(
                request_id,
                sampling_params,
                prompt_token_ids,
                prompt_embeds,
            )

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_input["type"] == "multimodal":
            decoder_mm_inputs = decoder_input["mm_kwargs"]
            decoder_mm_positions = decoder_input["mm_placeholders"]
            decoder_mm_hashes = decoder_input["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "
                    "This is likely due to an incorrect custom implementation of "
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_input.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            request_metadata=request_metadata,
            session_id=session_id,
            kv_hints=kv_hints,
        )

    def _resolve_capture_prefix_flags(
        self,
        request_id: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None,
        prompt_embeds: Any,
    ) -> None:
        """Offline analogue of the serving layer's ``_admit_capture``.

        Resolves ``sampling_params.capture`` into the prefix-cache reuse flags
        so offline (and any non-OpenAI) requests get the same prefix-cache
        treatment as served ones. Best-effort: an invalid spec leaves the flags
        unset so the request stays conservatively correct (prefix caching
        disabled for the capture) and the worker re-validates the raw spec at
        registration.

        Both spec/dict-form and pre-built instance-form consumers
        (``LLM(capture_consumers=[obj])``) resolve here:
        ``build_admission_validators`` keys them exactly as the runner does.
        """
        if getattr(self.vllm_config, "capture_consumers_config", None) is None:
            return

        from vllm.v1.capture.admission import (
            build_capture_context,
            resolve_capture_prefix_flags,
        )
        from vllm.v1.capture.errors import (
            CaptureValidationError,
            UnknownCaptureConsumerError,
        )
        from vllm.v1.capture.registry import build_admission_validators

        if self._capture_consumers is None:
            self._capture_consumers = build_admission_validators(self.vllm_config)
        if not self._capture_consumers:
            return

        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        try:
            ctx = build_capture_context(self.vllm_config, num_prompt_tokens, request_id)
            resolve_capture_prefix_flags(self._capture_consumers, sampling_params, ctx)
        except (UnknownCaptureConsumerError, CaptureValidationError) as exc:
            logger.debug(
                "capture: offline prefix-flag resolution skipped for request %s: %s",
                request_id,
                exc,
            )

    def _resolve_patch_prefix_flags(
        self,
        request_id: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None,
        prompt_embeds: Any,
    ) -> None:
        """Offline analogue of the serving layer's ``_admit_patch``.

        Resolves ``sampling_params.patch`` into the prefix-cache reuse flags
        (``patch_touches_prompt`` / ``patch_min_prompt_position``) so offline
        (and any non-OpenAI) requests get the same precise APC windowing served
        requests get. Unlike the best-effort capture path, an invalid patch
        spec raises ``ValueError`` — patch entries carry explicit
        ``dest_position`` ints validated against the model's real shape, so a
        bad spec is a hard client error that must fail loud (the engine rejects
        the request, mirroring the serving layer's 400).

        Deliberately narrower than the serving layer's ``_admit_patch``: it does
        NOT run the source-manifest existence pre-check or the multimodal guard.
        Those are frontend concerns — engine-side, a missing source is caught
        (loudly) by the worker resolution-failure registry as a backstop, and
        the offline/multimodal combination is out of scope.

        The new client-provided source kinds are structurally validated here
        (exactly-one-of source, mask shape, packed ``patch_vectors``, inline
        index-in-range/width). Offline has no frontend steering registry, so
        ``source_module`` names (other than the reserved ``zeros``) CANNOT be
        existence-checked here — a bad name surfaces loudly via the worker
        resolution-failure registry (``named_module_exists`` is left ``None``).
        """
        from vllm.v1.capture.admission import build_capture_context
        from vllm.v1.capture.patch_admission import (
            PatchValidationError,
            resolve_patch_prefix_flags,
        )

        patch_config = getattr(self.vllm_config, "patch_config", None)
        if patch_config is None:
            raise VLLMValidationError(
                "A patch spec was provided but patching is not enabled. Start "
                "vLLM with --enable-patching to use per-request activation "
                "patching."
            )

        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        ctx = build_capture_context(self.vllm_config, num_prompt_tokens, request_id)
        max_patch_slots = getattr(patch_config, "max_patch_slots", 0)
        try:
            resolve_patch_prefix_flags(
                sampling_params, ctx, max_patch_slots=max_patch_slots
            )
        except PatchValidationError as exc:
            raise VLLMValidationError(str(exc)) from exc

    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        if self.skip_prompt_length_check and prompt_type == "encoder":
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise VLLMValidationError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            if self.supports_mm_inputs:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "embeds":
            is_token_ids = prompt_input.get("is_token_ids")
            if is_token_ids is not None and len(is_token_ids) != prompt_len:
                raise VLLMValidationError(
                    "prompt_is_token_ids must have the same length as prompt_embeds "
                    f"(expected {prompt_len}, got {len(is_token_ids)}).",
                    parameter="prompt_is_token_ids",
                )

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            mm_hashes = prompt_input["mm_hashes"]
            num_embeds_by_hash: dict[str, int] = {}
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_hash, mm_position in zip(
                    mm_hashes[modality], mm_positions, strict=True
                ):
                    num_embeds = mm_position.get_num_embeds()
                    num_embeds_by_hash[mm_hash] = num_embeds
                    if num_embeds > self.mm_encoder_cache_size:
                        raise VLLMValidationError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

            # Without chunked prefill, the scheduler encodes all of a prompt's
            # items in one step, so they must fit in the encoder cache together.
            total_num_embeds = sum(num_embeds_by_hash.values())
            if (
                prompt_type == "decoder"
                and not self.scheduler_config.enable_chunked_prefill
                and not self.vllm_config.is_mm_encoder_only
                and total_num_embeds > self.mm_encoder_cache_size
            ):
                raise VLLMValidationError(
                    f"The prompt's multimodal items need {total_num_embeds} "
                    "encoder cache tokens, more than the encoder cache size of "
                    f"{self.mm_encoder_cache_size}. They must fit together when "
                    "chunked prefill is disabled."
                )

        # Shared by generate, embedding and pooling requests.
        if prompt_ids:
            self.renderer.validate_token_ids(prompt_ids)

    def _validate_model_inputs(
        self,
        encoder_input: SingletonInput | None,
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
