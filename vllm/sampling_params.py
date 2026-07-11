# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling parameters for text generation."""

import copy
import json as json_mod
import math
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
import numpy as np
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import ModelConfig, SpeculativeConfig, StructuredOutputsConfig
from vllm.config.sae_steering_types import (
    SAEClampSpec,
    SAEFullReconstructionSpec,
    coerce_sae_clamp_specs,
    coerce_sae_full_reconstruction_specs,
    hash_sae_clamp_specs_for_phase,
    hash_sae_full_reconstruction_specs_for_phase,
)
from vllm.config.steering_types import (
    SteeringLayerEntry,
    SteeringVectorSpec,
    hash_steering_config,
    normalize_layer_entry,
    resolve_effective_vectors,
    validate_steering_index,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import VALID_HOOK_POINT_NAMES
from vllm.tokenizers import TokenizerLike
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.serial_utils import PydanticMsgspecMixin

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on `SamplingParams.logprob_token_ids` list length. Must match
the per-request row width allocated by the sampler's `LogprobTokenIdsState`."""


def validate_thinking_token_budget(value: int | float | bool | None) -> int | None:
    """Validate ``thinking_token_budget``; return ``None`` if unset."""
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    if value == -1:
        return None
    if value < 0:
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    return value


ThinkingTokenBudget = Annotated[
    int | None,
    BeforeValidator(validate_thinking_token_budget),
]


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# maybe make msgspec?
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise ValueError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise ValueError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise ValueError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise ValueError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


def _is_non_tekken_mistral(tokenizer: TokenizerLike) -> bool:
    return is_mistral_tokenizer(tokenizer) and not tokenizer.is_tekken


def _get_llg_tokenizer(tokenizer: TokenizerLike) -> Any:
    return tokenizer.llg_tokenizer if is_mistral_tokenizer(tokenizer) else None


class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.
    """

    n: int = 1
    """Number of outputs to return for the given prompt request.

    The maximum allowed value is controlled by the ``VLLM_MAX_N_SEQUENCES``
    environment variable (default: 16384).

    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`."""
    presence_penalty: float = 0.0
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    frequency_penalty: float = 0.0
    """Penalizes new tokens based on their frequency in the generated text so
    far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    repetition_penalty: float = 1.0
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    temperature: float = 1.0
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    top_p: float = 1.0
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    top_k: int = 0
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    min_p: float = 0.0
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this."""
    seed: int | None = None
    """Random seed to use for the generation."""
    stop: str | list[str] | None = None
    """String(s) that stop the generation when they are generated. The returned
    output will not contain the stop strings."""
    stop_token_ids: list[int] | None = None
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens."""
    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated."""
    max_tokens: int | None = 16
    """Maximum number of tokens to generate per output sequence."""
    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated"""
    logprobs: int | None = None
    """Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities."""
    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities."""
    logprob_token_ids: list[int] | None = None
    """Specific token IDs to return logprobs for. More efficient than
    logprobs=-1 when you only need logprobs for a small set of tokens.
    When set, logprobs for exactly these token IDs will be returned,
    in addition to the sampled token. This is useful for scoring tasks
    where you want to compare probabilities of specific label tokens."""
    flat_logprobs: bool = False
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance.
    NOTE: GC costs of FlatLogprobs is significantly smaller than
    list[dict[int, Logprob]]. After enabled, PromptLogprobs and
    SampleLogprobs would populated as FlatLogprobs."""
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    detokenize: bool = True
    """Whether to detokenize the output."""
    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output."""
    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output."""
    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    skip_clone: bool = False
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses."""

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    output_text_buffer_length: int = 0
    _eos_token_id: int | None = None
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    logit_bias: dict[int, float] | None = None
    """If provided, the engine will construct a logits processor that applies
    these logit biases."""
    allowed_token_ids: list[int] | None = None
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids."""
    extra_args: dict[str, Any] | None = None
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations."""
    routed_experts_prompt_start: int = 0
    """When enable_return_routed_experts is active, skip the first
    routed_experts_prompt_start prompt tokens from the returned routing
    data. In multi-turn agent scenarios, set this to the length of the
    already-returned prefix to avoid duplicating routing for prompt tokens
    covered by earlier turns. Default 0 returns routing for all prompt
    tokens."""

    # Fields used for bad words
    bad_words: list[str] | None = None
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence."""
    _bad_words_token_ids: list[list[int]] | None = None

    skip_reading_prefix_cache: bool | None = None
    thinking_token_budget: int | None = None
    """Maximum number of tokens allowed for thinking operations."""

    capture: dict[str, Any] | None = None
    """Per-request opt-in for capture consumers, keyed by consumer name.

    The value at each key is the consumer-specific spec (consumers define
    their own schemas). Only consumers with ``reads_client_spec = True`` accept
    per-request specs.

    Validation at ``SamplingParams`` construction is strictly structural:
    the field must be either ``None`` or a ``dict[str, Any]`` (with
    string keys). Per-consumer validation — shape, layer-in-range,
    byte-budget, prefix-cache positions — runs at the OpenAI entrypoint
    against the active consumer registry and is *not* performed here.

    The entrypoint validates each value at admission but leaves the raw
    payload in place: :class:`CaptureSpec` is not serializable across the
    engine IPC boundary, so the worker re-validates from this raw dict.
    Admission instead records the resolved prompt-overlap on
    ``capture_touches_prompt``."""

    capture_touches_prompt: bool | None = None
    """Whether this request's capture spec taps any prompt-range position.

    Resolved by the OpenAI entrypoint's ``_admit_capture`` once the
    consumer specs are validated (it is the only place the resolved
    positions exist at admission). Drives prefix-cache reuse via
    :meth:`vllm.v1.request.Request.get_skip_reading_prefix_cache`:

    - ``True``  — a prompt position is captured; the prefix must be
      re-forwarded, so prefix-cache reuse is skipped for this request.
    - ``False`` — only generated positions are captured; prefix caching
      is safe and kept.
    - ``None``  — unclassified (e.g. the offline ``LLM`` path, which does
      not resolve specs at admission). Treated conservatively as
      prompt-touching.

    Not client-settable; ignore any value supplied at construction."""

    capture_min_prompt_position: int | None = None
    """Lowest prompt position this request's capture taps, or ``None``.

    Set by ``_admit_capture`` alongside ``capture_touches_prompt`` when the
    latter is ``True``. Prefix-cache reuse is clamped to this position so
    it (and every later position) is re-forwarded and its residual can be
    captured; positions strictly below it may still be served from cache.
    See :meth:`vllm.v1.request.Request.get_capture_prefix_cache_limit`.
    ``None`` when no capture clamp applies (no capture, generated-only, or
    unclassified). Not client-settable."""

    capture_store_hook_layers: list[tuple[str, int]] | None = None
    """Union of ``(hook, layer)`` pairs this request's capture taps.

    Set by ``_admit_capture`` for prompt-touching captures. Used by the
    scheduler with ``capture_store_positions`` and the request's block
    hashes to test whether the captured prompt prefix is wholly resident in
    the activation store (and serve it from there instead of re-forwarding).
    Not client-settable."""

    capture_store_positions: list[int] | None = None
    """Union of captured prompt positions (sorted) for activation-store
    serve. Set by ``_admit_capture`` alongside ``capture_store_hook_layers``.
    Not client-settable."""

    patch: list[dict[str, Any]] | None = None
    """Per-request activation-patching spec: a list of site entries, each
    ``{"layer": int, "hook": str, "dest_position": int, "source_run": str,
    "source_position": int, "alpha": float = 1.0}``. Each entry overwrites
    (``alpha == 1``) or interpolates toward the destination's activation at
    ``(layer, hook, dest_position)`` with the clean run ``source_run``'s
    activation at ``source_position``.

    Validation at construction is strictly structural (list of dicts with the
    required keys/types). Layer/hook/source existence and pool capacity are
    validated at the entrypoint against the model + source store; the worker
    resolves source vectors from the per-rank source store."""

    patch_touches_prompt: bool | None = None
    """Whether this request patches any prompt-range position. Resolved at
    admission (mirrors ``capture_touches_prompt``); drives prefix-cache reuse
    via :meth:`vllm.v1.request.Request.get_skip_reading_prefix_cache`. ``None``
    (offline path) is treated conservatively as prompt-touching. Not
    client-settable."""

    patch_min_prompt_position: int | None = None
    """Lowest prompt position this request patches, or ``None``. Prefix-cache
    reuse is clamped to this position so it (and later positions) are
    re-forwarded and the patch hook fires. Not client-settable."""

    patch_vectors: dict[str, Any] | None = None
    """Request-level packed table of client-provided patch vectors, referenced
    by a patch entry's ``source_inline`` / mask ``inline`` row index. Same
    binary wire form as ``SteeringHookPacked`` minus layer_indices/scales:
    ``{"dtype": "float32|float16|bfloat16", "shape": [n_rows, width],
    "data": "<base64 contiguous bytes>"}``. Travels verbatim and is decoded
    once per request at worker-side resolution. Deliberately packed-only (no
    raw float-list form — inline float lists caused a large E2EL regression)."""

    steering_vectors: SteeringVectorSpec | None = None
    """Base steering vectors applied to both prefill and decode phases.
    Keyed by hook point name (pre_attn, post_attn, post_block), then
    layer index. Values are either bare
    ``list[float]`` (scale=1.0) or ``{"vector": [...], "scale": float}``."""

    prefill_steering_vectors: SteeringVectorSpec | None = None
    """Phase-specific steering vectors added to base during prefill only.
    Same format as ``steering_vectors``."""

    decode_steering_vectors: SteeringVectorSpec | None = None
    """Phase-specific steering vectors added to base during decode only.
    Same format as ``steering_vectors``."""

    _effective_prefill_steering_packed: dict[str, dict[int, np.ndarray]] | None = None
    """In-process pre-resolved + packed prefill-phase steering, in the
    model's compute dtype.  Equivalent to
    ``effective_prefill_steering`` cast to model dtype, but produced on
    the client *before* the request crosses the multiprocessing boundary
    so the wire format carries ``len(vec) * dtype.itemsize`` bytes per
    (hook, layer) entry instead of msgpack-encoded float lists (~4.5×
    reduction at bf16).  When non-empty, takes precedence over
    ``steering_vectors`` / ``prefill_steering_vectors`` in
    :meth:`SteeringModelRunnerMixin._resolve_request_steering` and short-
    circuits the merge + resolve numpy work on the worker.

    Underscore-prefixed so :class:`PydanticMsgspecMixin` excludes the
    field from JSON / OpenAPI schema generation (``np.ndarray`` is not a
    Pydantic-friendly type).  Internal field — never populated from the
    public API; constructed by the request-preprocessing helpers in
    :mod:`vllm.config.steering_types`."""

    _effective_decode_steering_packed: dict[str, dict[int, np.ndarray]] | None = None
    """Decode-phase counterpart of ``_effective_prefill_steering_packed``."""

    steering_module_ref: tuple[str, float] | None = None
    """Optional ``(module_name, scale)`` reference to a worker-side
    named steering module.  When set, the worker resolves the named
    module against its broadcast registry and merges it with any
    inline ``steering_vectors`` / ``prefill_steering_vectors`` /
    ``decode_steering_vectors`` overrides via
    :func:`vllm.config.steering_types.merge_steering_specs` before
    calling ``SteeringManager.register_config``.  The reference, not
    the resolved vectors, is what crosses the multiprocessing boundary
    — eliminating per-request serialization of large vector blobs for
    requests that use named modules.

    Hash determinism: the request hash incorporates the
    ``(name, scale)`` tuple via :func:`hash_steering_config`, so two
    requests with the same reference and identical inline overrides
    produce the same hash regardless of when the named module was
    registered.  Inline-only requests (``steering_module_ref is None``)
    hash bit-for-bit identically to today, preserving prefix-cache
    reuse."""

    sae_full_reconstruction_specs: tuple[SAEFullReconstructionSpec, ...] | None = None
    """Optional SAE full-reconstruction directives (residual replacement).

    Each entry references a named SAE module (registered via the
    standard module-registration API with
    ``kind="sae_full_reconstruction"``) and optionally declares
    per-(hook, layer) clamps to apply to the SAE's activations
    before the decoder pass.  When ``clamps`` is empty the SAE
    reconstruction replaces the residual without modifications —
    pure ``decode(activate(encode(h))) + b_dec``.

    Hash determinism: the full-reconstruction state is folded into
    :pyattr:`prefill_steering_config_hash` /
    :pyattr:`decode_steering_config_hash` via
    :func:`hash_steering_config`'s
    ``sae_full_reconstruction_specs`` argument with a distinct
    domain separator from the delta block, so a delta-clamp request
    and a full-reconstruction request with identical clamp content
    do not collide on prefix-cache keys.  Replacement and
    perturbation produce different residual streams and must not
    share prefill cache."""

    sae_clamp_specs: tuple[SAEClampSpec, ...] | None = None
    """Optional SAE feature-surgery clamps (delta intervention).

    Each entry references a named SAE module (registered via the
    standard module-registration API with ``kind="sae_delta"``) and
    declares which feature activations to clamp on which
    (hook, layer) pairs.  See :class:`SAEClampSpec` and
    ``docs/features/sae_steering.md`` for the runtime contract.

    Hash determinism: SAE clamp state is folded into
    :pyattr:`prefill_steering_config_hash` /
    :pyattr:`decode_steering_config_hash` via
    :func:`hash_steering_config`'s ``sae_clamp_specs`` argument, so
    different clamp configurations produce different prefix-cache
    keys.  Requests that do not use SAE clamps
    (``sae_clamp_specs is None``) hash bit-for-bit identically to
    requests on a build without SAE support, preserving prefix-cache
    reuse."""

    repetition_detection: RepetitionDetectionParams | None = None
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    If such repetition is detected, generation will be ended early. LLMs can
    sometimes generate repetitive, unhelpful token patterns, stopping only
    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
    and terminate early, saving time and tokens."""

    @staticmethod
    def from_optional(
        n: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        bad_words: list[str] | None = None,
        thinking_token_budget: int | None = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int | None = 16,
        min_tokens: int = 0,
        logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        structured_outputs: StructuredOutputsParams | None = None,
        logit_bias: dict[int, float] | dict[str, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        skip_clone: bool = False,
        repetition_detection: RepetitionDetectionParams | None = None,
        capture: dict[str, Any] | None = None,
        patch: list[dict[str, Any]] | None = None,
        patch_vectors: dict[str, Any] | None = None,
        steering_vectors: SteeringVectorSpec | None = None,
        prefill_steering_vectors: SteeringVectorSpec | None = None,
        decode_steering_vectors: SteeringVectorSpec | None = None,
        steering_module_ref: tuple[str, float] | None = None,
        sae_clamp_specs: object = None,
        sae_full_reconstruction_specs: object = None,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            logit_bias = {
                int(token): min(100.0, max(-100.0, bias))
                for token, bias in logit_bias.items()
            }

        return SamplingParams(
            n=1 if n is None else n,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            thinking_token_budget=thinking_token_budget,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            logprob_token_ids=logprob_token_ids,
            prompt_logprobs=prompt_logprobs,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_kind=output_kind,
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            skip_clone=skip_clone,
            repetition_detection=repetition_detection,
            capture=capture,
            patch=patch,
            patch_vectors=patch_vectors,
            steering_vectors=steering_vectors,
            prefill_steering_vectors=prefill_steering_vectors,
            decode_steering_vectors=decode_steering_vectors,
            steering_module_ref=steering_module_ref,
            sae_clamp_specs=coerce_sae_clamp_specs(sae_clamp_specs),
            sae_full_reconstruction_specs=coerce_sae_full_reconstruction_specs(
                sae_full_reconstruction_specs
            ),
        )

    def __post_init__(self) -> None:
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.bad_words is None:
            self.bad_words = []

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()
        self._validate_capture()
        self._validate_patch()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

        if self.skip_reading_prefix_cache is None:
            # Disable prefix-cache reuse for this request when prompt_logprobs
            # is requested: with caching the number of returned prompt
            # logprobs may be less than n_prompt_tokens.
            #
            # The capture case is handled separately, not here: a prefix-cache
            # hit skips the forward pass for the cached prefix, so the hook
            # taps never fire on those positions. But that only conflicts when
            # the capture actually taps a prompt position. Whether it does is
            # not known until the consumer specs are resolved at admission, so
            # the decision lives in ``Request.get_skip_reading_prefix_cache``
            # via ``capture_touches_prompt`` rather than in this constructor.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    def _validate_capture(self) -> None:
        """Structural check on ``capture``.

        Only verifies the shape at construction time (``dict[str, Any]``
        with string keys). Per-consumer validation — against the active
        consumer registry, with access to the request context — happens
        in the OpenAI entrypoint (``_admit_capture``). Leaving full
        validation out of ``SamplingParams`` keeps the module free of
        any capture-framework imports.
        """
        capture = self.capture
        if capture is None:
            return
        if not isinstance(capture, dict):
            raise ValueError(
                "capture must be a dict keyed by consumer name, got "
                f"{type(capture).__name__}"
            )
        for key in capture:
            if not isinstance(key, str):
                raise ValueError(
                    "capture keys must be strings (consumer names), got "
                    f"{type(key).__name__} ({key!r})"
                )

    def _validate_patch(self) -> None:
        """Structural check on ``patch`` (and ``patch_vectors``).

        Verifies shape at construction: a list of dicts with the common keys,
        exactly one source kind per entry (``source_run`` + ``source_position``
        | ``source_module`` | ``source_inline``), an optional ``mask``
        (``{"indices": [...]}`` or ``{"inline": row}``), and — when present — a
        structurally-valid packed ``patch_vectors`` table whose every referenced
        row index is in range. Layer/hook/source existence, named-module
        existence, inline widths and pool capacity are validated at the
        entrypoint against the model + registries, keeping this module free of
        patch-framework imports.
        """
        patch = self.patch
        if patch is None:
            if self.patch_vectors is not None:
                self._validate_patch_vectors_table()
            return
        if not isinstance(patch, list):
            raise ValueError(
                f"patch must be a list of site dicts, got {type(patch).__name__}"
            )
        n_rows = self._validate_patch_vectors_table()
        for i, entry in enumerate(patch):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"patch[{i}] must be a dict, got {type(entry).__name__}"
                )
            for req_field in ("layer", "hook", "dest_position"):
                if req_field not in entry:
                    raise ValueError(f"patch[{i}] missing required key {req_field!r}")
            if not isinstance(entry["layer"], int):
                raise ValueError(f"patch[{i}]['layer'] must be an int")
            if not isinstance(entry["hook"], str):
                raise ValueError(f"patch[{i}]['hook'] must be a str")
            if not isinstance(entry["dest_position"], int):
                raise ValueError(f"patch[{i}]['dest_position'] must be an int")
            alpha = entry.get("alpha", 1.0)
            if not isinstance(alpha, (int, float)):
                raise ValueError(f"patch[{i}]['alpha'] must be a number")
            self._validate_patch_source_kind(i, entry, n_rows)
            self._validate_patch_mask(i, entry.get("mask"), n_rows)

    def _validate_patch_source_kind(
        self, i: int, entry: dict, n_rows: int | None
    ) -> None:
        """Enforce exactly-one-of source kinds and their per-field types."""
        has_run = entry.get("source_run") is not None
        has_module = entry.get("source_module") is not None
        has_inline = entry.get("source_inline") is not None
        n_kinds = sum((has_run, has_module, has_inline))
        if n_kinds != 1:
            raise ValueError(
                f"patch[{i}] must set exactly one source kind — "
                f"(source_run + source_position) | source_module | "
                f"source_inline; got {n_kinds}"
            )
        if has_run:
            if not isinstance(entry["source_run"], str):
                raise ValueError(f"patch[{i}]['source_run'] must be a str")
            if "source_position" not in entry:
                raise ValueError(
                    f"patch[{i}]: source_run requires source_position"
                )
            if not isinstance(entry["source_position"], int):
                raise ValueError(f"patch[{i}]['source_position'] must be an int")
        elif has_module:
            if not isinstance(entry["source_module"], str):
                raise ValueError(f"patch[{i}]['source_module'] must be a str")
        else:  # has_inline
            idx = entry["source_inline"]
            if not isinstance(idx, int):
                raise ValueError(f"patch[{i}]['source_inline'] must be an int")
            self._require_patch_row(i, "source_inline", idx, n_rows)

    def _validate_patch_mask(
        self, i: int, mask: Any, n_rows: int | None
    ) -> None:
        """Structural check on an optional per-entry ``mask``."""
        if mask is None:
            return
        if not isinstance(mask, dict):
            raise ValueError(f"patch[{i}]['mask'] must be a dict")
        has_indices = mask.get("indices") is not None
        has_inline = mask.get("inline") is not None
        if has_indices == has_inline:
            raise ValueError(
                f"patch[{i}]['mask'] must set exactly one of "
                f"'indices' | 'inline'"
            )
        if has_indices:
            indices = mask["indices"]
            if not isinstance(indices, list):
                raise ValueError(f"patch[{i}]['mask']['indices'] must be a list")
            for j in indices:
                if not isinstance(j, int) or j < 0:
                    raise ValueError(
                        f"patch[{i}]['mask']['indices'] must be non-negative ints"
                    )
        else:
            idx = mask["inline"]
            if not isinstance(idx, int):
                raise ValueError(f"patch[{i}]['mask']['inline'] must be an int")
            self._require_patch_row(i, "mask.inline", idx, n_rows)

    def _require_patch_row(
        self, i: int, what: str, idx: int, n_rows: int | None
    ) -> None:
        """A ``source_inline`` / mask inline index must reference a real row."""
        if n_rows is None:
            raise ValueError(
                f"patch[{i}]: {what} index {idx} requires patch_vectors"
            )
        if not (0 <= idx < n_rows):
            raise ValueError(
                f"patch[{i}]: {what} index {idx} out of range [0, {n_rows})"
            )

    def _validate_patch_vectors_table(self) -> int | None:
        """Structurally validate ``patch_vectors``; return its ``n_rows``.

        ``None`` when no table is set. Raises on malformed keys, dtype, shape,
        or a base64 payload whose byte length disagrees with ``shape``/dtype.
        """
        pv = self.patch_vectors
        if pv is None:
            return None
        if not isinstance(pv, dict):
            raise ValueError("patch_vectors must be a dict")
        for key in ("dtype", "shape", "data"):
            if key not in pv:
                raise ValueError(f"patch_vectors missing required key {key!r}")
        itemsize = {"float32": 4, "float16": 2, "bfloat16": 2}.get(str(pv["dtype"]))
        if itemsize is None:
            raise ValueError(
                f"patch_vectors.dtype {pv['dtype']!r} must be one of "
                f"float32 | float16 | bfloat16"
            )
        shape = pv["shape"]
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or not all(isinstance(s, int) and s >= 0 for s in shape)
        ):
            raise ValueError(
                "patch_vectors.shape must be [n_rows, width] of non-negative ints"
            )
        if not isinstance(pv["data"], str):
            raise ValueError("patch_vectors.data must be a base64 string")
        import binascii

        import pybase64 as base64

        try:
            raw = base64.b64decode(pv["data"])
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"patch_vectors.data is not valid base64: {exc}") from exc
        expected = int(shape[0]) * int(shape[1]) * itemsize
        if len(raw) != expected:
            raise ValueError(
                f"patch_vectors.data length {len(raw)} != expected {expected} "
                f"(shape={list(shape)}, dtype={pv['dtype']})"
            )
        return int(shape[0])

    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise ValueError(f"n must be an int, but is of type {type(self.n)}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        max_n = envs.VLLM_MAX_N_SEQUENCES
        if self.n > max_n:
            raise ValueError(
                f"n must be at most {max_n}, got {self.n}. "
                "To increase this limit, set the VLLM_MAX_N_SEQUENCES "
                "environment variable."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if not math.isfinite(self.repetition_penalty):
            raise ValueError(
                "repetition_penalty must be a finite number, "
                f"got {self.repetition_penalty}."
            )
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if not math.isfinite(self.temperature):
            raise VLLMValidationError(
                f"temperature must be a finite number, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature < 0.0:
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature > 2.0:
            raise VLLMValidationError(
                f"temperature must be in [0, 2], got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if not 0.0 < self.top_p <= 1.0:
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise ValueError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        if not isinstance(self.top_k, int):
            raise TypeError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        if self.min_tokens < 0:
            raise ValueError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        assert isinstance(self.stop_token_ids, list)
        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):
            raise ValueError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise ValueError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )
        assert isinstance(self.bad_words, list)
        if any(not bad_word for bad_word in self.bad_words):
            raise ValueError(
                f"bad_words cannot contain an empty string. "
                f"Got bad_words={self.bad_words}"
            )

        self._validate_steering_vectors()

    def _validate_steering_vectors(self) -> None:
        """Validate all steering vector fields if provided.

        Expected format per field:
        ``{hook_point: {layer_idx: SteeringLayerEntry}}``
        where ``SteeringLayerEntry`` is either ``list[float]`` (scale=1.0)
        or ``{"vector": list[float], "scale": float}``.
        """
        if self.steering_module_ref is not None:
            ref = self.steering_module_ref
            # Accept tuple or list (msgspec / JSON round-trips may emit
            # the latter); coerce to tuple post-validation.
            if (
                not isinstance(ref, (tuple, list))
                or len(ref) != 2
                or not isinstance(ref[0], str)
                or not isinstance(ref[1], (int, float))
                or not math.isfinite(float(ref[1]))
            ):
                raise ValueError(
                    "steering_module_ref must be a "
                    "(name: str, scale: finite float) tuple, got "
                    f"{ref!r}."
                )
            if not isinstance(ref, tuple):
                self.steering_module_ref = (ref[0], float(ref[1]))

        fields_to_check: list[tuple[str, SteeringVectorSpec | None]] = [
            ("steering_vectors", self.steering_vectors),
            ("prefill_steering_vectors", self.prefill_steering_vectors),
            ("decode_steering_vectors", self.decode_steering_vectors),
        ]
        for field_name, spec in fields_to_check:
            if spec is not None:
                self._validate_single_steering_spec(field_name, spec)

        # Cross-validate overlapping dimensions between base and phase specs.
        if self.steering_vectors:
            for phase_name, phase_spec in [
                ("prefill_steering_vectors", self.prefill_steering_vectors),
                ("decode_steering_vectors", self.decode_steering_vectors),
            ]:
                if phase_spec is None:
                    continue
                for hook, layers in self.steering_vectors.items():
                    if hook not in phase_spec:
                        continue
                    for layer_idx, base_entry in layers.items():
                        if layer_idx not in phase_spec[hook]:
                            continue
                        base_vec, _ = normalize_layer_entry(base_entry)
                        phase_vec, _ = normalize_layer_entry(
                            phase_spec[hook][layer_idx]
                        )
                        if len(base_vec) != len(phase_vec):
                            raise ValueError(
                                f"steering_vectors[{hook!r}]"
                                f"[{layer_idx}] has "
                                f"dimension {len(base_vec)} but "
                                f"{phase_name}[{hook!r}]"
                                f"[{layer_idx}] has "
                                f"dimension {len(phase_vec)}. "
                                f"Overlapping entries must have "
                                f"matching dimensions."
                            )

        # Cross-validate overlapping dimensions between prefill and decode
        # phase specs (caught even when no base ``steering_vectors`` is set).
        if self.prefill_steering_vectors and self.decode_steering_vectors:
            for hook, prefill_layers in self.prefill_steering_vectors.items():
                if hook not in self.decode_steering_vectors:
                    continue
                decode_layers = self.decode_steering_vectors[hook]
                for layer_idx, prefill_entry in prefill_layers.items():
                    if layer_idx not in decode_layers:
                        continue
                    prefill_vec, _ = normalize_layer_entry(prefill_entry)
                    decode_vec, _ = normalize_layer_entry(decode_layers[layer_idx])
                    if len(prefill_vec) != len(decode_vec):
                        raise ValueError(
                            f"prefill_steering_vectors[{hook!r}]"
                            f"[{layer_idx}] has "
                            f"dimension {len(prefill_vec)} but "
                            f"decode_steering_vectors[{hook!r}]"
                            f"[{layer_idx}] has "
                            f"dimension {len(decode_vec)}. "
                            f"Overlapping entries must have "
                            f"matching dimensions."
                        )

    def _validate_single_steering_spec(
        self, field_name: str, spec: SteeringVectorSpec
    ) -> None:
        """Validate a single steering vector spec."""
        if not isinstance(spec, dict):
            raise ValueError(
                f"{field_name} must be a dict mapping hook point "
                "names to dicts of layer vectors."
            )
        for hook_name, layer_vecs in spec.items():
            if hook_name not in VALID_HOOK_POINT_NAMES:
                raise ValueError(
                    f"{field_name} key {hook_name!r} is not a "
                    f"valid hook point. Valid values: "
                    f"{sorted(VALID_HOOK_POINT_NAMES)}."
                )
            if not isinstance(layer_vecs, dict):
                raise ValueError(
                    f"{field_name}[{hook_name!r}] must be a dict "
                    f"mapping layer indices to layer entries."
                )
            for key, value in layer_vecs.items():
                if not isinstance(key, int) or key < 0:
                    raise ValueError(
                        f"{field_name}[{hook_name!r}] keys must be "
                        f"non-negative integers, got {key!r}."
                    )
                self._validate_layer_entry(field_name, hook_name, key, value)

    def _validate_layer_entry(
        self,
        field_name: str,
        hook_name: str,
        layer_idx: int,
        entry: SteeringLayerEntry,
    ) -> None:
        """Validate a single layer entry (bare list or dict with scale)."""
        prefix = f"{field_name}[{hook_name!r}][{layer_idx}]"
        if isinstance(entry, dict):
            allowed = {"vector", "scale"}
            extra = set(entry.keys()) - allowed
            if extra:
                raise ValueError(
                    f"{prefix} dict entry has unexpected keys: {sorted(extra)}; "
                    f"allowed keys: ['scale', 'vector']"
                )
            if "vector" not in entry or "scale" not in entry:
                raise ValueError(
                    f"{prefix} dict entries must have 'vector' "
                    f"and 'scale' keys, got {sorted(entry.keys())}."
                )
            if not isinstance(entry["scale"], (int, float)):
                raise ValueError(
                    f"{prefix}['scale'] must be a finite float, got "
                    f"{type(entry['scale']).__name__}."
                )
            if not math.isfinite(entry["scale"]):
                raise ValueError(
                    f"{prefix}['scale'] must be finite, got {entry['scale']}."
                )
            self._validate_float_list(prefix + "['vector']", entry["vector"])
        elif isinstance(entry, list):
            self._validate_float_list(prefix, entry)
        else:
            # ndarray entries arrive from the binary-wire decode path
            # (``unpack_steering_vectors``).  The downstream resolver
            # already accepts ndarrays — ``np.asarray`` is a no-op on
            # them — so we just sanity-check shape/dtype here rather
            # than rejecting outright.
            import numpy as _np

            if isinstance(entry, _np.ndarray):
                if entry.ndim != 1:
                    raise ValueError(
                        f"{prefix} ndarray must be 1-D, got shape {entry.shape}."
                    )
                if entry.dtype.kind != "f":
                    raise ValueError(
                        f"{prefix} ndarray must be a floating dtype, got {entry.dtype}."
                    )
                return
            raise ValueError(
                f"{prefix} must be a list of floats or a dict with "
                f"'vector' and 'scale' keys, got "
                f"{type(entry).__name__}."
            )

    @staticmethod
    def _validate_float_list(prefix: str, values: Any) -> None:
        """Validate that *values* is a list of finite floats."""
        if not isinstance(values, list):
            raise ValueError(
                f"{prefix} must be a list of floats, got {type(values).__name__}."
            )
        for i, v in enumerate(values):
            if not isinstance(v, (int, float)):
                raise ValueError(
                    f"{prefix}[{i}] must be a finite float, got {type(v).__name__}."
                )
            if not math.isfinite(v):
                raise ValueError(f"{prefix}[{i}] must be finite, got {v}.")

    @cached_property
    def effective_prefill_steering(
        self,
    ) -> dict[str, dict[int, np.ndarray]] | None:
        """Resolved prefill steering: base + prefill-specific, pre-scaled.

        Returns 1-D ``np.ndarray`` per (hook, layer).  When the request
        was packed by the client (``_effective_prefill_steering_packed``
        is set) those arrays are already in the model's compute dtype;
        otherwise a fresh resolve over the original list-of-floats
        fields produces ``np.float64`` arrays.  ``hash_steering_config``
        casts to ``float32`` at the SHA boundary in either case, so the
        hash is stable within a deployment (cross-pipeline reuse — i.e.,
        switching a workload between packed and unpacked — is a one-time
        cache miss).
        """
        if self._effective_prefill_steering_packed is not None:
            return self._effective_prefill_steering_packed
        return resolve_effective_vectors(
            self.steering_vectors, self.prefill_steering_vectors
        )

    @cached_property
    def effective_decode_steering(
        self,
    ) -> dict[str, dict[int, np.ndarray]] | None:
        """Resolved decode steering: base + decode-specific, pre-scaled."""
        if self._effective_decode_steering_packed is not None:
            return self._effective_decode_steering_packed
        return resolve_effective_vectors(
            self.steering_vectors, self.decode_steering_vectors
        )

    @cached_property
    def patch_site_demand(self) -> dict[tuple[int, str], int]:
        """Per-``(layer, hook)`` patch-slot demand for this request.

        Counts the distinct patched ``dest_position``s at each site. A single
        forward step can compute all of a request's positions at a site at once
        (a prefill chunk), so this count is the request's worst-case slot draw
        at that site — what the scheduler reserves to keep the per-site pool
        from overflowing (see ``Scheduler``). Empty when no patching."""
        if not self.patch:
            return {}
        demand: dict[tuple[int, str], int] = {}
        for entry in self.patch:
            key = (int(entry["layer"]), str(entry["hook"]))
            seen = demand.setdefault(key, 0)
            demand[key] = seen + 1
        return demand

    @cached_property
    def patch_kv_taint(self) -> tuple[int, int] | None:
        """``(min_dest_position, spec_hash)`` for patch-aware prefix caching.

        A patched activation at position ``p`` changes the KV written at ``p``
        and (via attention in later layers) at every subsequent position, so
        blocks containing any position ``>= min_dest_position`` must not share
        cache entries with unpatched runs. ``spec_hash`` is a deterministic
        digest of the full spec (stable across processes, unlike ``hash()``),
        folded into those blocks' hashes: distinct specs get distinct KV
        chains, while blocks strictly below the patch floor stay shareable.
        ``None`` when the request patches nothing."""
        if not self.patch:
            return None
        import hashlib

        # Include every source kind + mask so distinct sources get distinct KV
        # chains; a client-provided value's identity lives in patch_vectors, so
        # fold the packed payload in too (different rows -> different KV).
        entries = sorted(
            (
                int(e["layer"]),
                str(e["hook"]),
                int(e["dest_position"]),
                str(e.get("source_run") or ""),
                (
                    int(e["source_position"])
                    if e.get("source_position") is not None
                    else -1
                ),
                str(e.get("source_module") or ""),
                (
                    int(e["source_inline"])
                    if e.get("source_inline") is not None
                    else -1
                ),
                repr(e.get("mask")),
                float(e.get("alpha", 1.0)),
            )
            for e in self.patch
        )
        payload = repr(entries)
        if self.patch_vectors is not None:
            payload += repr(self.patch_vectors.get("data"))
        digest = hashlib.sha256(payload.encode()).digest()
        return min(e[2] for e in entries), int.from_bytes(digest[:8], "big")

        self._validate_steering_vectors()
        self._validate_capture()

    def _validate_capture(self) -> None:
        """Structural check on ``capture``.

        Only verifies the shape at construction time (``dict[str, Any]``
        with string keys). Per-consumer validation — against the active
        consumer registry, with access to the request context — happens
        in the OpenAI entrypoint (``_admit_capture``). Leaving full
        validation out of ``SamplingParams`` keeps the module free of
        any capture-framework imports.
        """
        capture = self.capture
        if capture is None:
            return
        if not isinstance(capture, dict):
            raise ValueError(
                "capture must be a dict keyed by consumer name, got "
                f"{type(capture).__name__}"
            )
        for key in capture:
            if not isinstance(key, str):
                raise ValueError(
                    "capture keys must be strings (consumer names), got "
                    f"{type(key).__name__} ({key!r})"
                )

    def _validate_steering_vectors(self) -> None:
        """Validate all steering vector fields if provided.

        Expected format per field:
        ``{hook_point: {layer_idx: SteeringLayerEntry}}``
        where ``SteeringLayerEntry`` is either ``list[float]`` (scale=1.0)
        or ``{"vector": list[float], "scale": float}``.
        """
        if self.steering_module_ref is not None:
            ref = self.steering_module_ref
            # Accept tuple or list (msgspec / JSON round-trips may emit
            # the latter); coerce to tuple post-validation.
            if (
                not isinstance(ref, (tuple, list))
                or len(ref) != 2
                or not isinstance(ref[0], str)
                or isinstance(ref[1], bool)
                or not isinstance(ref[1], (int, float))
                or not math.isfinite(float(ref[1]))
            ):
                raise ValueError(
                    "steering_module_ref must be a "
                    "(name: str, scale: finite float) tuple, got "
                    f"{ref!r}."
                )
            if not isinstance(ref, tuple):
                self.steering_module_ref = (ref[0], float(ref[1]))

        fields_to_check: list[tuple[str, SteeringVectorSpec | None]] = [
            ("steering_vectors", self.steering_vectors),
            ("prefill_steering_vectors", self.prefill_steering_vectors),
            ("decode_steering_vectors", self.decode_steering_vectors),
        ]
        for field_name, spec in fields_to_check:
            if spec is not None:
                self._validate_single_steering_spec(field_name, spec)

        # Cross-validate overlapping dimensions between base and phase specs.
        if self.steering_vectors:
            for phase_name, phase_spec in [
                ("prefill_steering_vectors", self.prefill_steering_vectors),
                ("decode_steering_vectors", self.decode_steering_vectors),
            ]:
                if phase_spec is None:
                    continue
                for hook, layers in self.steering_vectors.items():
                    if hook not in phase_spec:
                        continue
                    for layer_idx, base_entry in layers.items():
                        if layer_idx not in phase_spec[hook]:
                            continue
                        base_vec, _ = normalize_layer_entry(base_entry)
                        phase_vec, _ = normalize_layer_entry(
                            phase_spec[hook][layer_idx]
                        )
                        if len(base_vec) != len(phase_vec):
                            raise ValueError(
                                f"steering_vectors[{hook!r}]"
                                f"[{layer_idx}] has "
                                f"dimension {len(base_vec)} but "
                                f"{phase_name}[{hook!r}]"
                                f"[{layer_idx}] has "
                                f"dimension {len(phase_vec)}. "
                                f"Overlapping entries must have "
                                f"matching dimensions."
                            )

        # Normalize SAE clamp specs.  Direct ``SamplingParams(...)`` callers
        # may pass raw JSON-shaped dicts; coerce them into validated
        # ``SAEClampSpec`` tuples here so downstream code only ever sees
        # the typed form.  ``coerce_sae_clamp_specs`` is idempotent for
        # already-typed input (no allocation when the field is already a
        # tuple of SAEClampSpec).
        if self.sae_clamp_specs is not None:
            self.sae_clamp_specs = coerce_sae_clamp_specs(self.sae_clamp_specs)
        if self.sae_full_reconstruction_specs is not None:
            self.sae_full_reconstruction_specs = coerce_sae_full_reconstruction_specs(
                self.sae_full_reconstruction_specs
            )

        # Cross-validate overlapping dimensions between prefill and decode
        # phase specs (caught even when no base ``steering_vectors`` is set).
        if self.prefill_steering_vectors and self.decode_steering_vectors:
            for hook, prefill_layers in self.prefill_steering_vectors.items():
                if hook not in self.decode_steering_vectors:
                    continue
                decode_layers = self.decode_steering_vectors[hook]
                for layer_idx, prefill_entry in prefill_layers.items():
                    if layer_idx not in decode_layers:
                        continue
                    prefill_vec, _ = normalize_layer_entry(prefill_entry)
                    decode_vec, _ = normalize_layer_entry(decode_layers[layer_idx])
                    if len(prefill_vec) != len(decode_vec):
                        raise ValueError(
                            f"prefill_steering_vectors[{hook!r}]"
                            f"[{layer_idx}] has "
                            f"dimension {len(prefill_vec)} but "
                            f"decode_steering_vectors[{hook!r}]"
                            f"[{layer_idx}] has "
                            f"dimension {len(decode_vec)}. "
                            f"Overlapping entries must have "
                            f"matching dimensions."
                        )

    def _validate_single_steering_spec(
        self, field_name: str, spec: SteeringVectorSpec
    ) -> None:
        """Validate a single steering vector spec."""
        if not isinstance(spec, dict):
            raise ValueError(
                f"{field_name} must be a dict mapping hook point "
                "names to dicts of layer vectors."
            )
        for hook_name, layer_vecs in spec.items():
            if hook_name not in VALID_HOOK_POINT_NAMES:
                raise ValueError(
                    f"{field_name} key {hook_name!r} is not a "
                    f"valid hook point. Valid values: "
                    f"{sorted(VALID_HOOK_POINT_NAMES)}."
                )
            if not isinstance(layer_vecs, dict):
                raise ValueError(
                    f"{field_name}[{hook_name!r}] must be a dict "
                    f"mapping layer indices to layer entries."
                )
            for key, value in layer_vecs.items():
                layer_idx = validate_steering_index(
                    key, f"{field_name}[{hook_name!r}] key"
                )
                self._validate_layer_entry(field_name, hook_name, layer_idx, value)

    def _validate_layer_entry(
        self,
        field_name: str,
        hook_name: str,
        layer_idx: int,
        entry: SteeringLayerEntry,
    ) -> None:
        """Validate a single layer entry (bare list or dict with scale)."""
        prefix = f"{field_name}[{hook_name!r}][{layer_idx}]"
        if isinstance(entry, dict):
            allowed = {"vector", "scale"}
            extra = set(entry.keys()) - allowed
            if extra:
                raise ValueError(
                    f"{prefix} dict entry has unexpected keys: {sorted(extra)}; "
                    f"allowed keys: ['scale', 'vector']"
                )
            if "vector" not in entry or "scale" not in entry:
                raise ValueError(
                    f"{prefix} dict entries must have 'vector' "
                    f"and 'scale' keys, got {sorted(entry.keys())}."
                )
            if isinstance(entry["scale"], bool) or not isinstance(
                entry["scale"], (int, float)
            ):
                raise ValueError(
                    f"{prefix}['scale'] must be a finite float, got "
                    f"{type(entry['scale']).__name__}."
                )
            if not math.isfinite(entry["scale"]):
                raise ValueError(
                    f"{prefix}['scale'] must be finite, got {entry['scale']}."
                )
            self._validate_float_list(prefix + "['vector']", entry["vector"])
        elif isinstance(entry, list):
            self._validate_float_list(prefix, entry)
        elif isinstance(entry, np.ndarray):
            # ndarray entries arrive from the binary-wire decode path
            # (``unpack_steering_vectors``).  The downstream resolver
            # already accepts ndarrays — ``np.asarray`` is a no-op on
            # them — so we just sanity-check shape/dtype here rather
            # than rejecting outright.
            if entry.ndim != 1:
                raise ValueError(
                    f"{prefix} ndarray must be 1-D, got shape {entry.shape}."
                )
            if entry.dtype.kind != "f":
                raise ValueError(
                    f"{prefix} ndarray must be a floating dtype, got {entry.dtype}."
                )
        else:
            raise ValueError(
                f"{prefix} must be a list of floats or a dict with "
                f"'vector' and 'scale' keys, got "
                f"{type(entry).__name__}."
            )

    @staticmethod
    def _validate_float_list(prefix: str, values: Any) -> None:
        """Validate that *values* is a list of finite floats."""
        if not isinstance(values, list):
            raise ValueError(
                f"{prefix} must be a list of floats, got {type(values).__name__}."
            )
        for i, v in enumerate(values):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"{prefix}[{i}] must be a finite float, got {type(v).__name__}."
                )
            if not math.isfinite(v):
                raise ValueError(f"{prefix}[{i}] must be finite, got {v}.")

    def _phase_filtered_sae_specs(
        self, want_phase: str
    ) -> tuple[SAEClampSpec, ...] | None:
        """Filter ``sae_clamp_specs`` to those active in *want_phase*.

        A spec with ``phase="both"`` enters both prefill and decode
        digests; a spec with ``phase="prefill"`` only enters the
        prefill digest; ``phase="decode"`` only the decode digest.
        Returns ``None`` when no spec applies — letting
        :func:`hash_steering_config` skip the SAE block entirely so
        the resulting digest is bit-for-bit identical to a request
        without ``sae_clamp_specs``.
        """
        if not self.sae_clamp_specs:
            return None
        filtered = tuple(
            s for s in self.sae_clamp_specs if s.phase in ("both", want_phase)
        )
        return filtered if filtered else None

    def _phase_filtered_sae_full_recon_specs(
        self, want_phase: str
    ) -> tuple[SAEFullReconstructionSpec, ...] | None:
        """Filter ``sae_full_reconstruction_specs`` to those active in *want_phase*.

        Same shape as :meth:`_phase_filtered_sae_specs`: ``"both"``
        always matches; ``"prefill"`` / ``"decode"`` match only their
        own phase.  Returns ``None`` when no spec applies so
        :func:`hash_steering_config` can skip the full-recon block.
        """
        if not self.sae_full_reconstruction_specs:
            return None
        filtered = tuple(
            s
            for s in self.sae_full_reconstruction_specs
            if s.phase in ("both", want_phase)
        )
        return filtered if filtered else None

    @cached_property
    def prefill_steering_config_hash(self) -> int:
        """Cached hash of ``effective_prefill_steering`` plus
        ``steering_module_ref`` plus prefill-active ``sae_clamp_specs``.

        Lives on ``SamplingParams`` (not ``Request``) so that many requests
        sharing the same ``SamplingParams`` object — the common case for
        batched ``llm.generate(prompts, [sp]*N)`` — only pay the hashing
        cost once across the whole batch instead of once per request.

        When ``steering_module_ref`` and ``sae_clamp_specs`` are both
        ``None`` this reduces to the original inline-only hash
        bit-for-bit, preserving prefix-cache reuse for requests that
        don't reference a named module or SAE clamp.  When set, those
        fields are folded into the digest so two requests with the
        same reference + identical inline overrides + identical SAE
        clamps produce the same hash regardless of when the named
        modules were registered worker-side.
        """
        return hash_steering_config(
            self.effective_prefill_steering,
            module_ref=self.steering_module_ref,
            sae_clamp_specs=self._phase_filtered_sae_specs("prefill"),
            sae_full_reconstruction_specs=(
                self._phase_filtered_sae_full_recon_specs("prefill")
            ),
        )

    @cached_property
    def decode_steering_config_hash(self) -> int:
        """Cached hash of ``effective_decode_steering`` plus
        ``steering_module_ref`` plus decode-active ``sae_clamp_specs``.
        See ``prefill_steering_config_hash``."""
        return hash_steering_config(
            self.effective_decode_steering,
            module_ref=self.steering_module_ref,
            sae_clamp_specs=self._phase_filtered_sae_specs("decode"),
            sae_full_reconstruction_specs=(
                self._phase_filtered_sae_full_recon_specs("decode")
            ),
        )

    @cached_property
    def prefill_additive_steering_config_hash(self) -> int:
        """Cached hash of only the additive prefill steering identity."""
        return hash_steering_config(
            self.effective_prefill_steering,
            module_ref=self.steering_module_ref,
        )

    @cached_property
    def decode_additive_steering_config_hash(self) -> int:
        """Cached hash of only the additive decode steering identity."""
        return hash_steering_config(
            self.effective_decode_steering,
            module_ref=self.steering_module_ref,
        )

    @cached_property
    def prefill_sae_clamp_config_hash(self) -> int:
        """Cached hash of only the prefill-active SAE clamp identity."""
        return hash_sae_clamp_specs_for_phase(
            self._phase_filtered_sae_specs("prefill"),
            "prefill",
        )

    @cached_property
    def decode_sae_clamp_config_hash(self) -> int:
        """Cached hash of only the decode-active SAE clamp identity."""
        return hash_sae_clamp_specs_for_phase(
            self._phase_filtered_sae_specs("decode"),
            "decode",
        )

    @cached_property
    def prefill_sae_full_recon_config_hash(self) -> int:
        """Cached hash of only the prefill-active SAE full-reconstruction identity."""
        return hash_sae_full_reconstruction_specs_for_phase(
            self._phase_filtered_sae_full_recon_specs("prefill"),
            "prefill",
        )

    @cached_property
    def decode_sae_full_recon_config_hash(self) -> int:
        """Cached hash of only the decode-active SAE full-reconstruction identity."""
        return hash_sae_full_reconstruction_specs_for_phase(
            self._phase_filtered_sae_full_recon_specs("decode"),
            "decode",
        )

    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise ValueError(f"n must be 1 when using greedy sampling, got {self.n}.")

    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
        if not self.ignore_eos:
            self._eos_token_id = eos_token_id

        if eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    @property
    def num_logprobs(self) -> int | None:
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None

    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy.

        Steering vector dicts are shared by reference between the original
        and the clone (via the deepcopy memo) instead of deep-copied. They
        can be ~hundreds of KB of floats per request and the clone's
        downstream consumers do not mutate them, so deep-copying them is
        ~10-20ms of pure waste per request submission.

        Cached steering hash values are also explicitly carried over to the
        clone — they are a deterministic function of the steering vector
        contents, which are identical between the original and the clone,
        so recomputing them would only burn CPU time.
        """
        if self.skip_clone:
            return copy.copy(self)

        # Pre-populate the deepcopy memo with the top-level steering vector
        # dicts. ``copy.deepcopy`` checks the memo before recursing, so any
        # dict found there is returned by reference instead of deep-copied.
        memo: dict[int, Any] = {}
        for attr in (
            self.steering_vectors,
            self.prefill_steering_vectors,
            self.decode_steering_vectors,
            self._effective_prefill_steering_packed,
            self._effective_decode_steering_packed,
            self.sae_clamp_specs,
            self.sae_full_reconstruction_specs,
        ):
            if attr is not None:
                memo[id(attr)] = attr

        new_sp = copy.deepcopy(self, memo)
        if self.sae_clamp_specs is not None:
            new_sp.sae_clamp_specs = self.sae_clamp_specs
        if self.sae_full_reconstruction_specs is not None:
            new_sp.sae_full_reconstruction_specs = self.sae_full_reconstruction_specs

        # Carry over cached @cached_property values so the clone doesn't
        # re-hash the same steering vectors. cached_property stores its
        # values in the instance ``__dict__``.
        for key in (
            "prefill_steering_config_hash",
            "decode_steering_config_hash",
            "prefill_additive_steering_config_hash",
            "decode_additive_steering_config_hash",
            "prefill_sae_clamp_config_hash",
            "decode_sae_clamp_config_hash",
            "prefill_sae_full_recon_config_hash",
            "decode_sae_full_recon_config_hash",
            "effective_prefill_steering",
            "effective_decode_steering",
        ):
            if key in self.__dict__:
                new_sp.__dict__[key] = self.__dict__[key]

        return new_sp

    def verify(
        self,
        model_config: ModelConfig,
        speculative_config: SpeculativeConfig | None,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        self._validate_logprobs(model_config)
        self._validate_logit_bias(model_config)
        self._validate_logits_processors(model_config)
        self._validate_allowed_token_ids(tokenizer)
        self._validate_spec_decode(speculative_config)
        self._validate_structured_outputs(
            model_config, structured_outputs_config, tokenizer
        )

    def _validate_logprobs(self, model_config: ModelConfig) -> None:
        max_logprobs = model_config.max_logprobs
        if max_logprobs == -1:
            max_logprobs = model_config.get_vocab_size()

        # Validate sample logprobs.
        if num_logprobs := self.logprobs:
            if num_logprobs == -1:
                num_logprobs = model_config.get_vocab_size()
            if num_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="logprobs",
                    value=num_logprobs,
                )

        # Validate logprob_token_ids.
        if self.logprob_token_ids is not None:
            n = len(self.logprob_token_ids)
            if n > MAX_LOGPROB_TOKEN_IDS:
                raise VLLMValidationError(
                    f"Requested logprob_token_ids of length {n}, "
                    f"which is greater than max allowed: {MAX_LOGPROB_TOKEN_IDS}",
                    parameter="logprob_token_ids",
                    value=n,
                )
            vocab_size = model_config.get_vocab_size()
            invalid_token_ids = [
                token_id
                for token_id in self.logprob_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    f"token_id(s) {invalid_token_ids} in logprob_token_ids "
                    f"contain out-of-vocab token ids. Vocabulary size: "
                    f"{vocab_size}",
                    parameter="logprob_token_ids",
                    value=invalid_token_ids,
                )
            if self.logprobs is not None and self.logprobs != n:
                raise VLLMValidationError(
                    f"When both logprobs and logprob_token_ids are set, "
                    f"logprobs must equal len(logprob_token_ids). Got "
                    f"logprobs={self.logprobs}, len(logprob_token_ids)={n}.",
                    parameter="logprob_token_ids",
                    value=n,
                )

        # Validate prompt logprobs.
        if num_prompt_logprobs := self.prompt_logprobs:
            if num_prompt_logprobs == -1:
                num_prompt_logprobs = model_config.get_vocab_size()
            if num_prompt_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="prompt_logprobs",
                    value=num_prompt_logprobs,
                )

    def _validate_logit_bias(self, model_config: ModelConfig) -> None:
        """Validate logit_bias token IDs are within vocabulary range."""
        if not self.logit_bias:
            return

        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.logit_bias
            if token_id < 0 or token_id >= vocab_size
        ]

        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="logit_bias",
                value=invalid_token_ids,
            )

    def _validate_logits_processors(self, model_config: ModelConfig) -> None:
        from vllm.v1.sample.logits_processor import (
            validate_logits_processors_parameters,
        )

        validate_logits_processors_parameters(model_config.logits_processors, self)

    def _validate_allowed_token_ids(self, tokenizer: TokenizerLike | None) -> None:
        allowed_token_ids = self.allowed_token_ids
        if allowed_token_ids is None:
            return

        if len(allowed_token_ids) == 0:
            raise VLLMValidationError(
                "allowed_token_ids is not None and empty!",
                parameter="allowed_token_ids",
                value=allowed_token_ids,
            )

        if tokenizer is not None:
            vocab_size = len(tokenizer)
            invalid_token_ids = [
                token_id
                for token_id in allowed_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    "allowed_token_ids contains out-of-vocab token id!",
                    parameter="allowed_token_ids",
                    value=invalid_token_ids,
                )

    def _validate_spec_decode(
        self,
        speculative_config: SpeculativeConfig | None,
    ) -> None:
        if speculative_config is None:
            return

        # Some sampling parameters are not yet compatible with spec decoding.
        if self.min_p > _SAMPLING_EPS or self.logit_bias:
            raise ValueError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )

    def _validate_structured_outputs(
        self,
        model_config: ModelConfig,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        if model_config.is_diffusion:
            # Diffusion LLMs denoise a whole canvas of tokens in parallel
            # rather than sampling left-to-right, which the grammar FSM
            # requires. Without this check, requests fail mid-generation
            # with an FSM rejection (HTTP 500). See issue #45436.
            raise ValueError(
                "Structured outputs are not yet supported for diffusion "
                "language models. Remove the structured output constraint "
                "(e.g. `response_format`, `structured_outputs`) from the "
                "request."
            )

        if tokenizer is None:
            raise ValueError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        backend = structured_outputs_config.backend
        if _backend := self.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            if backend != _backend and not (
                backend == "auto" and self.structured_outputs._backend_was_auto
            ):
                raise ValueError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        if (
            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise ValueError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise ValueError("structured_outputs.grammar cannot be an empty string")

        from vllm.v1.structured_output.backend_guidance import (
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        from vllm.v1.structured_output.backend_lm_format_enforcer import (
            validate_structured_output_request_lm_format_enforcer,
        )
        from vllm.v1.structured_output.backend_outlines import (
            validate_structured_output_request_outlines,
        )
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        elif backend.startswith("guidance"):
            if _is_non_tekken_mistral(tokenizer=tokenizer):
                raise ValueError(
                    "Non-tekken Mistral tokenizers are not supported for the 'guidance'"
                    " structured output backend. Please either use a more recent "
                    "Mistral model, the ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(
                self,
                tokenizer=_get_llg_tokenizer(tokenizer),
            )
        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(self)
        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            if is_mistral_tokenizer(tokenizer):
                raise ValueError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(self)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except ValueError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                skip_guidance = _is_non_tekken_mistral(tokenizer)

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs
                if not skip_guidance and so_params.json:
                    if isinstance(so_params.json, str):
                        schema = json_mod.loads(so_params.json)
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                if skip_guidance:
                    # Fall back to outlines if the tokenizer is non-tekken Mistral or
                    # the schema contains features unsupported by guidance
                    validate_structured_output_request_outlines(self)
                    self.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(
                        self,
                        tokenizer=_get_llg_tokenizer(tokenizer),
                    )
                    self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        self.structured_outputs.__post_init__()

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"thinking_token_budget={self.thinking_token_budget}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
            f"extra_args={self.extra_args})"
        )

    @staticmethod
    def for_sampler_warmup() -> "SamplingParams":
        """Set parameters to exercise all sampler logic."""
        return SamplingParams(
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            repetition_penalty=1.2,
            min_tokens=2,
            logit_bias={0: -1.0, 1: 0.5},
            _bad_words_token_ids=[[0], [1, 2]],
            logprobs=5,
            prompt_logprobs=1,
        )


class BeamSearchParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    structured_outputs: StructuredOutputsParams | None = None
