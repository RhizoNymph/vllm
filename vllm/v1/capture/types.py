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
# available; never client-controlled; opaque string. Consumers that want
# to correlate with external identity should declare the appropriate
# optional sidecar field (e.g., ``client_request_id``, ``tag``).
VllmInternalRequestId = NewType("VllmInternalRequestId", str)


# ---------------------------------------------------------------------------
# Hook points and position selector
# ---------------------------------------------------------------------------

# Mirrors ``_HOOK_NAME_TO_ID`` in
# ``vllm/model_executor/layers/activation_capture.py``. Any change to the
# set of hook points must be reflected there as well.
#
# The ``mhc_*`` hooks are DeepSeek-V4 manifold-hyperconnection targets.
# Unlike the standard residual hooks (always ``(hidden_size,)`` in the
# model dtype) they vary in width and dtype, so each carries a
# :class:`HookSchema` describing its row width, dtype, and logical row
# shape. See :func:`build_hook_schema`.
HookName = Literal[
    "pre_attn",
    "post_attn",
    "post_mlp",
    "mlp_in",
    "mlp_out",
    # DeepSeek-V4 mHC: multi-stream residual (hc_mult * hidden, bf16).
    "mhc_streams_pre_attn",
    "mhc_streams_pre_mlp",
    "mhc_streams_final",
    # DeepSeek-V4 mHC: stream-mixing coefficients (fp32).
    "mhc_attn_post_mix",
    "mhc_ffn_post_mix",
    "mhc_attn_res_mix",
    "mhc_ffn_res_mix",
]

# The five standard residual hook *names* (mirrors the hook-id table in
# ``activation_capture.py``); each, when tapped, captures the full
# ``(hidden_size,)`` residual in the model dtype.
STANDARD_HOOKS: tuple[str, ...] = (
    "pre_attn",
    "post_attn",
    "post_mlp",
    "mlp_in",
    "mlp_out",
)

# The hooks ``apply_layer_steering`` actually taps on every standard model
# today (``mlp_in`` / ``mlp_out`` are reserved names but not wired into any
# standard model forward). A model's hook schema lists exactly the hooks it
# taps, so admission can reject hooks that would yield empty captures.
WIRED_STANDARD_HOOKS: tuple[str, ...] = (
    "pre_attn",
    "post_attn",
    "post_mlp",
)

# Hooks that fire once per request at the *model tail*, not per decoder
# layer (DeepSeek-V4's final pre-``hc_head`` streams). They are keyed to the
# last layer index (``num_hidden_layers - 1``), where the tap fires on the
# last pipeline stage. Their layer selector is meaningless, so admission
# ignores it and normalizes to that single index — a caller writes
# ``{"mhc_streams_final": "all"}`` without needing to know the index.
MODEL_LEVEL_HOOKS: frozenset[str] = frozenset({"mhc_streams_final"})

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


# ---------------------------------------------------------------------------
# HookSchema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HookSchema:
    """Per-hook row geometry for a captured activation.

    The framework stores each captured row as a 2D ``(num_rows, width)``
    tensor.  ``width`` is the flattened per-row element count and
    ``dtype`` the element type.  ``logical_shape`` is the per-row shape a
    consumer should reshape a row back to — e.g. ``(hidden_size,)`` for a
    standard residual hook, ``(hc_mult, hidden_size)`` for an mHC stream
    hook, ``(hc_mult, hc_mult)`` for an mHC Sinkhorn ``res_mix`` hook.
    ``width`` must equal ``prod(logical_shape)``.
    """

    width: int
    dtype: torch.dtype
    logical_shape: tuple[int, ...]


def default_hook_schema(hidden_size: int, dtype: torch.dtype) -> dict[str, HookSchema]:
    """Schema for the standard residual hooks a normal model taps.

    Lists exactly :data:`WIRED_STANDARD_HOOKS` — the hooks
    ``apply_layer_steering`` fires on every standard model — so a request
    for an unwired hook (``mlp_in`` / ``mlp_out`` / ``mhc_*``) is rejected
    at admission (the validator checks the hook is present in the schema).
    """
    return {
        hook: HookSchema(hidden_size, dtype, (hidden_size,))
        for hook in WIRED_STANDARD_HOOKS
    }


def build_hook_schema(
    hidden_size: int,
    dtype: torch.dtype,
    hc_mult: int | None = None,
) -> dict[str, HookSchema]:
    """Hook schema for a model — the hooks it taps, with their geometry.

    Without ``hc_mult`` this is the standard wired residual hooks
    (:func:`default_hook_schema`). When ``hc_mult`` is provided (DeepSeek-V4
    manifold-hyperconnection models expose it as ``hf_config.hc_mult``) the
    schema instead describes the V4 decoder's tapped hooks: the
    single-stream attention/FFN in-out residuals (reusing the standard
    ``pre_attn`` / ``post_attn`` / ``mlp_in`` / ``mlp_out`` names — note V4
    has no single-stream ``post_mlp``, its end-of-layer residual is the
    multi-stream ``mhc_streams_*``) plus the mHC stream and coefficient
    hooks.
    """
    if hc_mult is None:
        return default_hook_schema(hidden_size, dtype)

    single = HookSchema(hidden_size, dtype, (hidden_size,))
    stream_width = hc_mult * hidden_size
    stream_shape = (hc_mult, hidden_size)
    return {
        "pre_attn": single,
        "post_attn": single,
        "mlp_in": single,
        "mlp_out": single,
        "mhc_streams_pre_attn": HookSchema(stream_width, dtype, stream_shape),
        "mhc_streams_pre_mlp": HookSchema(stream_width, dtype, stream_shape),
        "mhc_streams_final": HookSchema(stream_width, dtype, stream_shape),
        "mhc_attn_post_mix": HookSchema(hc_mult, torch.float32, (hc_mult,)),
        "mhc_ffn_post_mix": HookSchema(hc_mult, torch.float32, (hc_mult,)),
        "mhc_attn_res_mix": HookSchema(
            hc_mult * hc_mult, torch.float32, (hc_mult, hc_mult)
        ),
        "mhc_ffn_res_mix": HookSchema(
            hc_mult * hc_mult, torch.float32, (hc_mult, hc_mult)
        ),
    }


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
    # Per-hook row geometry (width, dtype, logical shape). Validators use
    # it to size byte estimates per hook and to reject ``mhc_*`` hooks on
    # models that do not expose them (the hook is simply absent here). When
    # omitted (older construction sites) it falls back to the five standard
    # residual hooks sized from ``hidden_size``/``element_size_bytes``.
    hook_schema: dict[str, HookSchema] = field(default_factory=dict)


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
