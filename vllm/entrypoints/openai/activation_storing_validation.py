# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-time validation for per-request activation storing.

This module runs at the OpenAI-compatible entrypoint boundary, before the
request is handed to the engine. It takes a raw
:class:`ActivationStoringSpec` + a :class:`VllmConfig` + a small request
context and either:

- returns a :class:`ResolvedActivationStoringSpec` that downstream code
  can consume without re-doing any of this work, or
- raises :class:`ActivationStoringValidationError` with a descriptive
  message that the entrypoint surfaces as HTTP 400.

The checks performed here are the ones that *can't* be done at
``SamplingParams`` construction time because they need a
:class:`VllmConfig` or request context:

1. The feature is enabled (``vllm_config.activation_storing_config is
   not None``).
2. ``tensor_parallel_size == 1`` and ``pipeline_parallel_size == 1``.
   Multi-rank residual collection is out of scope for v1.
3. Every layer referenced by every hook is in
   ``[0, num_hidden_layers)``.
4. Tag and request_id slug cleanly (reject ``..``, leading ``/``, > 256
   chars, empty).
5. Explicit position indices are valid and not below
   ``num_computed_tokens`` (prefix-cache hits that were never forwarded).
6. The estimated capture byte count does not exceed
   ``max_bytes_per_request`` when that cap is set.

Runtime concerns (writer pool, plan building, finalization) are *not*
this module's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config.activation_storing_types import (
    ActivationStoringSlugError,
    ActivationStoringSpec,
    expand_hook_layers,
    resolve_positions,
    slug,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class ActivationStoringValidationError(ValueError):
    """Raised by :func:`validate_activation_storing` on admission failure.

    The message is user-facing: the entrypoint should surface it verbatim
    in an HTTP 400 response. Inherits from :class:`ValueError` so callers
    can ``except ValueError`` generically if they prefer, but the typed
    class lets the entrypoint distinguish activation-storing failures
    from other sampling-param issues.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


@dataclass
class ActivationStoringContext:
    """Per-request context needed by the admission validator.

    Separated from :class:`VllmConfig` because these values come from the
    request (num_prompt_tokens, num_computed_tokens) or from the model
    runner's knowledge of residual shape (hidden_size, element_size_bytes)
    — ``VllmConfig`` alone is not enough.
    """

    num_prompt_tokens: int
    num_computed_tokens: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    num_hidden_layers: int
    hidden_size: int
    element_size_bytes: int


@dataclass
class ResolvedActivationStoringSpec:
    """A :class:`ActivationStoringSpec` with selectors pre-expanded.

    Produced by :func:`validate_activation_storing` on success. Downstream
    code (runner-side capture manager, writer pool) consumes this instead
    of the raw spec so expansion cost is paid exactly once.
    """

    request_id_slug: str
    tag_slug: str
    hooks: dict[str, list[int]]
    # ``positions`` may be a resolved list (``last_prompt``, ``all_prompt``,
    # explicit list) or the original kind string when the position count
    # cannot be known at admission time (``all_generated``, ``all``).
    positions: list[int] | str
    position_kind: str
    # Byte-estimate we computed during validation, useful for metrics and
    # warnings at the runner. ``0`` when the capture is symbolic (kind is
    # ``"all"``/``"all_generated"``).
    estimated_bytes: int
    # Echo of the original spec for downstream components that want to
    # re-read raw fields.
    raw: ActivationStoringSpec


def _estimate_bytes(
    positions: list[int] | None,
    hooks: dict[str, list[int]],
    hidden_size: int,
    element_size_bytes: int,
) -> int:
    """Estimate the total bytes one request will write.

    Returns ``0`` when ``positions`` is ``None`` (symbolic, resolution
    deferred to the runner). Otherwise:

        len(positions) * sum(len(layers) for layers in hooks.values())
                       * hidden_size * element_size_bytes

    The byte-budget cap is enforced against this number at admission time.
    """
    if positions is None:
        return 0
    total_layers = sum(len(layers) for layers in hooks.values())
    return (
        len(positions) * total_layers * hidden_size * element_size_bytes
    )


def validate_activation_storing(
    spec: ActivationStoringSpec,
    vllm_config: "VllmConfig",
    ctx: ActivationStoringContext,
) -> ResolvedActivationStoringSpec:
    """Resolve a per-request spec + reject invalid configurations.

    See module docstring for the exact list of checks performed. Raises
    :class:`ActivationStoringValidationError` with a descriptive message
    on failure; returns a :class:`ResolvedActivationStoringSpec` on
    success.
    """
    # 1. Feature enabled?
    ast_config = getattr(vllm_config, "activation_storing_config", None)
    if ast_config is None or ast_config.root_path is None:
        raise ActivationStoringValidationError(
            "activation_storing was provided on the request but the "
            "server does not have activation storing enabled. Start the "
            "server with --activation-storing /path/to/root to enable.",
            field="activation_storing",
        )

    # 2. TP/PP > 1 rejected.
    if ctx.tensor_parallel_size != 1:
        raise ActivationStoringValidationError(
            f"activation storing is only supported with "
            f"tensor_parallel_size=1; got {ctx.tensor_parallel_size}. "
            "Multi-rank residual collection is out of scope for v1.",
            field="tensor_parallel_size",
        )
    if ctx.pipeline_parallel_size != 1:
        raise ActivationStoringValidationError(
            f"activation storing is only supported with "
            f"pipeline_parallel_size=1; got {ctx.pipeline_parallel_size}. "
            "Multi-rank residual collection is out of scope for v1.",
            field="pipeline_parallel_size",
        )

    # 3. Slug tag + request_id so we surface a clean error (rather than
    # failing much later on a filesystem write).
    try:
        request_id_slug = slug(spec.request_id)
    except ActivationStoringSlugError as exc:
        raise ActivationStoringValidationError(
            f"activation_storing.request_id is invalid: {exc}",
            field="activation_storing.request_id",
        ) from exc
    try:
        tag_slug = slug(spec.tag)
    except ActivationStoringSlugError as exc:
        raise ActivationStoringValidationError(
            f"activation_storing.tag is invalid: {exc}",
            field="activation_storing.tag",
        ) from exc

    # 4. Expand hooks and validate layer indices.
    if ctx.num_hidden_layers <= 0:
        raise ActivationStoringValidationError(
            f"activation storing requires num_hidden_layers > 0; got "
            f"{ctx.num_hidden_layers}",
            field="num_hidden_layers",
        )
    resolved_hooks: dict[str, list[int]] = {}
    for hook_name, selector in spec.hooks.items():
        try:
            resolved_hooks[hook_name] = expand_hook_layers(
                selector,
                ctx.num_hidden_layers,
                where=f"activation_storing.hooks[{hook_name!r}]",
            )
        except ValueError as exc:
            raise ActivationStoringValidationError(
                str(exc),
                field=f"activation_storing.hooks[{hook_name!r}]",
            ) from exc
        if not resolved_hooks[hook_name]:
            raise ActivationStoringValidationError(
                f"activation_storing.hooks[{hook_name!r}] expanded to an "
                "empty layer list",
                field=f"activation_storing.hooks[{hook_name!r}]",
            )

    # 5. Resolve positions.
    #
    # Two kinds of selectors we can fully resolve at admission time:
    #
    #   - "last_prompt"  → [num_prompt_tokens - 1]
    #   - "all_prompt"   → [0, num_prompt_tokens)
    #   - explicit list  → bounds-checked against [0, num_prompt_tokens)
    #
    # "all_generated" and "all" stay symbolic here because the total
    # generated count is unknown at admission time; the runner's capture
    # manager materializes them per-step.
    position_kind: str
    resolved_positions: list[int] | None
    if isinstance(spec.positions, str):
        position_kind = spec.positions
        if spec.positions in ("last_prompt", "all_prompt"):
            try:
                resolved_positions = resolve_positions(
                    spec.positions,
                    ctx.num_prompt_tokens,
                    num_generated_tokens=0,
                    where="activation_storing.positions",
                )
            except ValueError as exc:
                raise ActivationStoringValidationError(
                    str(exc),
                    field="activation_storing.positions",
                ) from exc
        else:
            # "all_generated" / "all" — defer.
            resolved_positions = None
    elif isinstance(spec.positions, list):
        position_kind = "explicit"
        # Explicit positions are validated against the prompt window. We
        # don't know num_generated_tokens at admission time, so we treat
        # it as 0 and bound explicit indices to the prompt.
        try:
            resolved_positions = resolve_positions(
                spec.positions,
                ctx.num_prompt_tokens,
                num_generated_tokens=0,
                where="activation_storing.positions",
            )
        except ValueError as exc:
            raise ActivationStoringValidationError(
                str(exc),
                field="activation_storing.positions",
            ) from exc
    else:
        raise ActivationStoringValidationError(
            "activation_storing.positions must be a string or list of ints",
            field="activation_storing.positions",
        )

    # 6. Prefix-cache position rejection.
    # Any explicit position below num_computed_tokens was not forwarded
    # through the model (it was served from the prefix cache), so we can
    # never capture a residual for it. Reject rather than silently drop.
    if resolved_positions is not None:
        bad = [p for p in resolved_positions if p < ctx.num_computed_tokens]
        if bad:
            raise ActivationStoringValidationError(
                f"activation_storing.positions contains indices that were "
                f"served from the prefix cache and will never be forwarded "
                f"through the model: {bad}. The first uncached position "
                f"for this request is {ctx.num_computed_tokens}.",
                field="activation_storing.positions",
            )

    # 7. Byte budget.
    estimated_bytes = _estimate_bytes(
        resolved_positions,
        resolved_hooks,
        ctx.hidden_size,
        ctx.element_size_bytes,
    )
    cap = ast_config.max_bytes_per_request
    if cap > 0 and estimated_bytes > cap:
        raise ActivationStoringValidationError(
            f"activation storing estimated size {estimated_bytes} bytes "
            f"exceeds --activation-storing-max-bytes-per-request={cap}. "
            f"Reduce the number of layers, hooks, or positions, or raise "
            f"the server cap.",
            field="activation_storing",
        )

    return ResolvedActivationStoringSpec(
        request_id_slug=request_id_slug,
        tag_slug=tag_slug,
        hooks=resolved_hooks,
        positions=resolved_positions
        if resolved_positions is not None
        else position_kind,
        position_kind=position_kind,
        estimated_bytes=estimated_bytes,
        raw=spec,
    )


__all__: list[Any] = [
    "ActivationStoringContext",
    "ActivationStoringValidationError",
    "ResolvedActivationStoringSpec",
    "validate_activation_storing",
]
