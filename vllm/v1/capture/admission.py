# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared capture-spec admission.

Both the OpenAI serving layer (``OpenAIServing._admit_capture``) and the
offline ``InputProcessor`` resolve the per-request ``capture`` dict into the
same prefix-cache reuse flags on ``SamplingParams`` *before* the request
reaches the scheduler. The resolution is pure — a deterministic function of
the raw spec, the config-built consumer validators, and the request's
:class:`CaptureContext` — and the worker re-runs the identical resolution at
registration. Centralizing it here keeps the two entry points exactly in
step so the scheduler's prefix-cache decision is the same on both paths.

See ``docs/design/capture_consumers.md`` (§Prefix-Cache Interaction) for how
the stamped flags drive the B/C/A reuse layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.v1.capture.errors import (
    CaptureValidationError,
    UnknownCaptureConsumerError,
)
from vllm.v1.capture.types import (
    CaptureContext,
    VllmInternalRequestId,
    capture_expert_parallel_size,
    captured_prompt_positions,
    min_captured_prompt_position,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.capture.consumer import CaptureConsumer


def build_capture_context(
    vllm_config: VllmConfig,
    num_prompt_tokens: int,
    request_id: str,
) -> CaptureContext:
    """Build the admission-time :class:`CaptureContext` for a request.

    Mirrors the worker's registration context but with
    ``num_computed_tokens=0`` — admission classifies the spec against the
    full prompt, before any prefix-cache reuse is decided. Uses the *global*
    layer count because client specs reference global layer indices.
    """
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config

    num_hidden_layers = model_config.get_total_num_hidden_layers()
    hidden_size = model_config.get_hidden_size()
    dt = model_config.dtype
    element_size_bytes = getattr(dt, "itemsize", None)
    if element_size_bytes is None:
        import torch

        element_size_bytes = torch.tensor([], dtype=dt).element_size()

    return CaptureContext(
        vllm_internal_request_id=VllmInternalRequestId(request_id),
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        element_size_bytes=int(element_size_bytes),
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        expert_parallel_size=capture_expert_parallel_size(parallel_config),
        data_parallel_size=parallel_config.data_parallel_size,
    )


def resolve_capture_prefix_flags(
    consumers_by_name: dict[str, CaptureConsumer],
    sampling_params: SamplingParams,
    ctx: CaptureContext,
) -> None:
    """Resolve ``sampling_params.capture`` and stamp the prefix-cache flags.

    For each ``(name, raw_spec)`` entry, looks up the config-built validator
    and resolves the raw payload into a ``CaptureSpec``. From the resolved
    specs it computes the request-wide re-forward floor and the
    activation-store serve set, then writes them onto ``sampling_params``
    (``capture_touches_prompt``, ``capture_min_prompt_position``,
    ``capture_store_hook_layers``, ``capture_store_positions``):

    - Generated-only captures (no prompt-range tap) classify as
      ``capture_touches_prompt=False`` and keep full prefix caching.
    - Prompt-touching captures record the lowest tapped prompt position as
      the re-forward floor; the scheduler reuses cache only up to the floor
      (C clamp) or, if the whole captured prefix is store-resident, serves
      it from the activation store (A).

    The raw ``capture`` dict is left untouched — the worker re-validates from
    it after scheduling (``CaptureSpec`` is not IPC-serializable, so the raw
    payload, not the resolved spec, must cross the engine boundary).

    Raises:
        UnknownCaptureConsumerError: a named consumer is not configured. The
            offending key is attached as ``capture_param`` for error mapping.
        CaptureValidationError: a raw spec is invalid. The offending key is
            attached as ``capture_param``.
    """
    if sampling_params.capture is None:
        return

    num_prompt_tokens = ctx.num_prompt_tokens

    validated: dict[str, Any] = {}
    for name, raw_spec in sampling_params.capture.items():
        consumer = consumers_by_name.get(name)
        if consumer is None:
            available = sorted(consumers_by_name.keys())
            err = UnknownCaptureConsumerError(
                f"capture: no consumer named {name!r} is registered. "
                f"Available consumers: {available}."
            )
            err.capture_param = f"capture.{name}"  # type: ignore[attr-defined]
            raise err
        try:
            validated[name] = consumer.validate_client_spec(raw_spec, ctx)
        except CaptureValidationError as exc:
            if not getattr(exc, "capture_param", None):
                exc.capture_param = f"capture.{name}"  # type: ignore[attr-defined]
            raise

    # The request-wide re-forward floor is the lowest prompt position any
    # consumer taps. ``None`` from every consumer means generated-only.
    floors = [
        pos
        for pos in (
            min_captured_prompt_position(spec, num_prompt_tokens)
            for spec in validated.values()
        )
        if pos is not None
    ]
    if floors:
        sampling_params.capture_touches_prompt = True
        sampling_params.capture_min_prompt_position = min(floors)
        # Union (hook, layer) and prompt positions for activation-store serve
        # (A): the scheduler tests whether this whole set is store-resident
        # and, if so, serves it instead of re-forwarding.
        hook_layers: set[tuple[str, int]] = set()
        store_positions: set[int] = set()
        for spec in validated.values():
            for hook, layers in spec.hooks.items():
                for layer in layers:
                    hook_layers.add((hook, layer))
            store_positions.update(
                captured_prompt_positions(spec, num_prompt_tokens)
            )
        sampling_params.capture_store_hook_layers = sorted(hook_layers)
        sampling_params.capture_store_positions = sorted(store_positions)
    else:
        sampling_params.capture_touches_prompt = False
        sampling_params.capture_min_prompt_position = None
        sampling_params.capture_store_hook_layers = None
        sampling_params.capture_store_positions = None
