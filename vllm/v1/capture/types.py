# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core types for the capture-consumer framework.

This module is torch-aware — ``CaptureChunk`` carries a ``torch.Tensor``
so consumers can dispatch on captured activations without further
round-tripping. Unit tests exercising the dataclasses stay cheap by
using small CPU tensors.

See ``docs/design/capture_consumers.md`` § "Core Types" for the
authoritative field-by-field spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NewType

import torch

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

# ---------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------

# The unique identifier vLLM assigns internally to a request. Always
# available; never client-controlled; opaque string. To correlate a
# capture back to the original client request, consumers can read the
# always-present ``client_request_id`` sidecar field (the id the API
# returned; falls back to the internal id when request id randomization is
# disabled). Other external-identity fields (e.g. ``tag``) remain opt-in
# via ``required_sidecar_fields``.
VllmInternalRequestId = NewType("VllmInternalRequestId", str)


# ---------------------------------------------------------------------------
# Hook points and position selector
# ---------------------------------------------------------------------------

# Mirrors ``_HOOK_NAME_TO_ID`` in
# ``vllm/model_executor/layers/activation_capture.py``. Any change to the
# set of hook points must be reflected there as well.
HookName = Literal[
    "pre_attn",
    "post_attn",
    "post_block",
    "mlp_in",
    "mlp_out",
]

PositionSelector = (
    Literal["last_prompt", "all_prompt", "all_generated", "all"] | list[int]
)


# ---------------------------------------------------------------------------
# CaptureKey
# ---------------------------------------------------------------------------

# ``(vllm_internal_request_id, layer_idx, hook_name)``.
CaptureKey = tuple[VllmInternalRequestId, int, str]


# ---------------------------------------------------------------------------
# CaptureSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureSpec:
    """Describes which activations to capture for a request.

    ``hooks`` maps each hook point to the layer indices at which the hook
    fires. An empty list disables the hook. ``positions`` selects which
    token positions are captured at every ``(hook, layer)`` pair.

    ``CaptureSpec`` is the in-framework representation produced by a
    consumer's ``global_capture_spec()`` or its per-request
    ``validate_client_spec()``. It is not directly serializable across
    the process boundary; consumers that ship specs via IPC should go
    through their own validator.
    """

    hooks: dict[HookName, list[int]]
    positions: PositionSelector


def min_captured_prompt_position(
    spec: CaptureSpec, num_prompt_tokens: int
) -> int | None:
    """Lowest prompt-range position ``spec`` captures, or ``None``.

    A capture tap only produces a residual for a token position that is
    actually forwarded through the model. Prefix-cache reuse skips the
    forward pass for cached prompt positions, so a request must re-forward
    from its lowest captured prompt position onward; everything strictly
    below that position can still be served from cache. This function
    returns that floor — the value prefix-cache hits are clamped to
    (:meth:`vllm.v1.request.Request.get_capture_prefix_cache_limit`) — or
    ``None`` when the spec captures no prompt position (generated-only),
    which never conflicts with prefix caching.

    Called at admission (the OpenAI entrypoint's ``_admit_capture``) on
    each consumer's resolved :class:`CaptureSpec`. By that point
    ``positions`` is consumer-resolved: ``"last_prompt"`` / ``"all_prompt"``
    and explicit lists have become concrete index lists, while
    ``"all_generated"`` and ``"all"`` stay symbolic. Unrecognized symbols
    are treated conservatively as tapping the whole prompt (floor ``0``).
    """
    positions = spec.positions
    if positions == "all_generated":
        return None
    if positions == "all" or positions == "all_prompt":
        return 0
    if positions == "last_prompt":
        return num_prompt_tokens - 1 if num_prompt_tokens > 0 else None
    if isinstance(positions, list):
        prompt_positions = [p for p in positions if 0 <= p < num_prompt_tokens]
        return min(prompt_positions) if prompt_positions else None
    # Unrecognized symbolic selector → conservative: re-forward whole prompt.
    return 0


def spec_touches_prompt(spec: CaptureSpec, num_prompt_tokens: int) -> bool:
    """Whether ``spec`` captures any position in the prompt range.

    Thin predicate over :func:`min_captured_prompt_position`: a spec taps
    the prompt iff it has a captured prompt position to re-forward from.
    """
    return min_captured_prompt_position(spec, num_prompt_tokens) is not None


def captured_prompt_positions(spec: CaptureSpec, num_prompt_tokens: int) -> list[int]:
    """The prompt-range positions ``spec`` captures (sorted, deduped).

    Empty for generated-only specs. Used at admission to record which
    positions a whole-prefix activation-store serve must cover, and by the
    worker to assemble served chunks per consumer. ``"all"`` is treated as
    covering the whole prompt here (its generated half is captured by the
    normal forward path, not served).
    """
    positions = spec.positions
    if num_prompt_tokens <= 0 or positions == "all_generated":
        return []
    if positions == "all" or positions == "all_prompt":
        return list(range(num_prompt_tokens))
    if positions == "last_prompt":
        return [num_prompt_tokens - 1]
    if isinstance(positions, list):
        return sorted({p for p in positions if 0 <= p < num_prompt_tokens})
    # Unrecognized symbolic selector → conservative: whole prompt.
    return list(range(num_prompt_tokens))


# ---------------------------------------------------------------------------
# CaptureChunk and CaptureFinalize
# ---------------------------------------------------------------------------


@dataclass
class CaptureChunk:
    """One batch of captured rows for a ``CaptureKey``.

    Emitted by the manager after every forward step that produced rows
    for this key. For a single key, chunks arrive in ``row_offset``
    order; different keys have no ordering relationship.
    """

    key: CaptureKey
    # CPU tensor, shape ``(num_rows, hidden_size)``.
    tensor: torch.Tensor
    # Explicit dtype to avoid ``tensor.dtype`` dispatch in consumers.
    dtype: torch.dtype
    # Cumulative row index within this key's sequence.
    row_offset: int
    # Which forward step produced this chunk.
    step_index: int
    # Per-chunk context — see ``docs/design/capture_consumers.md``
    # § "Manager Runtime" for what the manager populates and
    # § "Known Limitations" for the gaps.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptureFinalize:
    """Request-completion signal for a ``CaptureKey``.

    Emitted by the manager when the owning request finishes (any finish
    reason). Arrives after all ``CaptureChunk``s for the key. On receipt
    the sink should flush any buffered state for this key and produce a
    terminal ``CaptureResult`` accessible via ``get_result(key)``.
    """

    key: CaptureKey
    sidecar: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CaptureResult
# ---------------------------------------------------------------------------

CaptureStatus = Literal[
    "pending",
    "ok",
    "partial_error",
    "error",
    "not_requested",
]


@dataclass
class CaptureResult:
    """Terminal per-key result from a consumer.

    Attached to ``RequestOutput.capture_results[consumer_name]`` on
    request completion. The ``payload`` field is consumer-specific and
    opaque to the framework — filesystem returns ``list[Path]``, a
    dashboard might return ``dict[str, str]``, a silent consumer returns
    ``None``.
    """

    key: CaptureKey
    status: CaptureStatus
    error: str | None = None
    payload: Any = None


# ---------------------------------------------------------------------------
# CaptureContext
# ---------------------------------------------------------------------------


@dataclass
class CaptureContext:
    """Per-request context passed to ``validate_client_spec``.

    Contains everything a validator needs to check a client spec against
    the request's actual shape. Fields are deliberately narrow —
    validators should not poke at ``vllm_config`` beyond these. If a
    validator needs more, add it here explicitly.
    """

    vllm_internal_request_id: VllmInternalRequestId
    num_prompt_tokens: int
    # Prefix-cache hits; positions below this index are already in the
    # KV cache and cannot be re-captured.
    num_computed_tokens: int
    # Global layer count (across all pipeline stages), so client specs
    # validate against the model's full layer space regardless of which
    # pipeline-parallel rank performs admission.
    num_hidden_layers: int
    hidden_size: int
    element_size_bytes: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    # Size of the expert-parallel plane (ranks experts are sharded over);
    # 1 when expert parallelism is disabled. Replicated residual hooks are
    # identical across this plane; only sharded-expert capture (Phase 4)
    # depends on it.
    expert_parallel_size: int = 1
    # Number of data-parallel replicas. Each replica is an independent
    # engine core over disjoint requests, so capture never aggregates
    # across this axis; carried for completeness / validator messaging.
    data_parallel_size: int = 1


def capture_expert_parallel_size(parallel_config: ParallelConfig) -> int:
    """EP-plane size for a :class:`CaptureContext`.

    vLLM has no standalone ``expert_parallel_size`` field. When
    ``enable_expert_parallel`` is set, experts shard across the
    ``tensor_parallel_size * data_parallel_size`` plane; otherwise the
    plane is a single rank. Shared by every ``CaptureContext``
    construction site so the derivation stays in one place.
    """
    if getattr(parallel_config, "enable_expert_parallel", False):
        return parallel_config.tensor_parallel_size * parallel_config.data_parallel_size
    return 1
