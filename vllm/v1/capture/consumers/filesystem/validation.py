# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation bridge between ``FilesystemCaptureRequest`` and the
existing ``validate_activation_storing`` admission validator.

Converts the consumer-framework types into the legacy activation-storing
types, delegates to the existing validator, then converts the result back
into a ``CaptureSpec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config.activation_storing_types import ActivationStoringSpec
from vllm.entrypoints.openai.activation_storing_validation import (
    ActivationStoringContext,
    ActivationStoringValidationError,
    validate_activation_storing,
)
from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import CaptureContext, CaptureSpec, HookName

from .types import FilesystemCaptureRequest

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def validate_filesystem_request(
    raw: FilesystemCaptureRequest,
    vllm_config: VllmConfig,
    ctx: CaptureContext,
) -> CaptureSpec:
    """Validate a ``FilesystemCaptureRequest`` and return a ``CaptureSpec``.

    Converts the consumer-framework ``FilesystemCaptureRequest`` into an
    ``ActivationStoringSpec``, delegates to
    ``validate_activation_storing``, and translates the resolved result
    back into the capture-framework's ``CaptureSpec``.

    Raises ``CaptureValidationError`` on any validation failure.
    """
    # Build the legacy spec from the consumer-framework request.
    try:
        legacy_spec = ActivationStoringSpec(
            request_id=raw.request_id,
            tag=raw.tag,
            hooks=raw.hooks,
            positions=raw.positions,
        )
    except ValueError as exc:
        raise CaptureValidationError(str(exc)) from exc

    # Build the legacy context from the capture-framework context.
    legacy_ctx = ActivationStoringContext(
        num_prompt_tokens=ctx.num_prompt_tokens,
        num_computed_tokens=ctx.num_computed_tokens,
        tensor_parallel_size=ctx.tensor_parallel_size,
        pipeline_parallel_size=ctx.pipeline_parallel_size,
        num_hidden_layers=ctx.num_hidden_layers,
        hidden_size=ctx.hidden_size,
        element_size_bytes=ctx.element_size_bytes,
    )

    # Delegate to the existing admission validator.
    try:
        resolved = validate_activation_storing(legacy_spec, vllm_config, legacy_ctx)
    except ActivationStoringValidationError as exc:
        raise CaptureValidationError(str(exc)) from exc

    # Convert resolved hooks back to CaptureSpec format.
    # The resolved hooks use plain string keys; CaptureSpec expects
    # HookName literals. The validator already checked that every key is
    # a valid hook name.
    typed_hooks: dict[HookName, list[int]] = {}
    for hook_name, layers in resolved.hooks.items():
        typed_hooks[hook_name] = layers  # type: ignore[literal-required]

    # Convert positions: the resolved spec has either a list[int] or a
    # string kind. CaptureSpec.positions uses the same union via
    # PositionSelector.
    positions = resolved.positions

    return CaptureSpec(hooks=typed_hooks, positions=positions)
