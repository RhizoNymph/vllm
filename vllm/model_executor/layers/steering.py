# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation steering custom op and hook-point definitions.

Registered as ``torch.ops.vllm.apply_steering`` so that torch.compile
treats the operation as an opaque splitting point.  The real Python
implementation executes at runtime between compiled graph segments,
reading the live buffer values rather than baked-in constants.
"""

from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.layers.activation_capture import maybe_capture_residual
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class SteeringHookPoint(str, Enum):
    """Positions in a decoder layer where steering can be applied.

    All hook points operate on the residual skip tensor carried through
    the decoder layer, not on the post-norm sublayer input tensor.
    The names identify approximate regions of the layer where the
    residual skip tensor is steered.
    """

    PRE_ATTN = "pre_attn"
    """Steer the residual skip tensor in the pre-attention region."""

    POST_ATTN = "post_attn"
    """Steer the residual skip tensor in the post-attention region."""

    POST_MLP = "post_mlp"
    """Steer the residual skip tensor in the post-MLP region."""

    # --- mHC (manifold-constrained hyper-connection) hook points ---
    # These are only registered on mHC models (e.g. DeepSeek-V4) via the
    # ``hook_widths`` argument to :func:`register_steering_buffers`; standard
    # decoder layers register only the three single-stream hooks above
    # (see ``STANDARD_STEERING_HOOKS``).  The string values match the capture
    # framework's hook names so a tensor can be both captured and steered
    # under one identifier.

    MLP_IN = "mlp_in"
    """Steer the single-stream pre-mixed FFN input (mHC models)."""

    MLP_OUT = "mlp_out"
    """Steer the single-stream FFN output (mHC models)."""

    MHC_STREAMS_PRE_ATTN = "mhc_streams_pre_attn"
    """Steer the multi-stream residual entering the attention sublayer."""

    MHC_STREAMS_PRE_MLP = "mhc_streams_pre_mlp"
    """Steer the multi-stream residual entering the FFN sublayer."""

    MHC_STREAMS_FINAL = "mhc_streams_final"
    """Steer the final multi-stream residual before the head fold."""


# The single-stream residual hook points wired into every standard decoder
# architecture. :func:`register_steering_buffers` registers exactly these
# (each at the model ``hidden_size``) unless a model passes an explicit
# ``hook_widths`` map. Adding new ``SteeringHookPoint`` members above does
# NOT change what standard models register.
STANDARD_STEERING_HOOKS: tuple[SteeringHookPoint, ...] = (
    SteeringHookPoint.PRE_ATTN,
    SteeringHookPoint.POST_ATTN,
    SteeringHookPoint.POST_MLP,
)


# Buffer attribute names on decoder layer modules, keyed by hook point.
HOOK_POINT_TABLE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "steering_table_pre_attn",
    SteeringHookPoint.POST_ATTN: "steering_table_post_attn",
    SteeringHookPoint.POST_MLP: "steering_table_post_mlp",
    SteeringHookPoint.MLP_IN: "steering_table_mlp_in",
    SteeringHookPoint.MLP_OUT: "steering_table_mlp_out",
    SteeringHookPoint.MHC_STREAMS_PRE_ATTN: "steering_table_mhc_streams_pre_attn",
    SteeringHookPoint.MHC_STREAMS_PRE_MLP: "steering_table_mhc_streams_pre_mlp",
    SteeringHookPoint.MHC_STREAMS_FINAL: "steering_table_mhc_streams_final",
}

# Per-hook ``any-active`` flag attribute names. The flag is a single-element
# bool tensor co-located with each hook point's table buffer; the apply
# kernel reads it at launch and short-circuits the gather + add when no row
# is currently active for that hook point. The attribute name is derived
# from the table attribute so the two are always discoverable together.
HOOK_POINT_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_any_active" for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}

# Valid hook point string values for validation.
VALID_HOOK_POINT_NAMES: frozenset[str] = frozenset(hp.value for hp in SteeringHookPoint)

DEFAULT_HOOK_POINT = SteeringHookPoint.POST_MLP


def register_steering_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    max_steering_tokens: int,
    max_steering_configs: int,
    dtype: torch.dtype | None = None,
    hook_widths: dict[SteeringHookPoint, int] | None = None,
) -> None:
    """Attach per-hook steering buffers to a decoder layer.

    ``dtype`` controls the storage dtype of the steering table buffers.
    When ``None`` (the default), the buffers fall back to fp32 to
    preserve historical behaviour.  Callers in vLLM models pass the
    model's compute dtype (typically bf16) so the indexed gather in
    :func:`apply_steering` returns rows already aligned with the residual
    tensor and no dtype cast is required at the gather site.

    ``hook_widths`` selects which hook points get a table and, for each,
    the per-row width of that table. When ``None`` (the default for every
    standard decoder architecture) exactly the three
    :data:`STANDARD_STEERING_HOOKS` are registered, each ``hidden_size``
    wide — historical behaviour. mHC models (e.g. DeepSeek-V4) pass an
    explicit map so multi-stream residual hooks get ``num_streams *
    hidden_size``-wide tables while single-stream hooks stay
    ``hidden_size`` wide. The per-row gather in :func:`apply_steering`
    reads each table's own width, so mixed widths coexist on one layer.

    When ``max_steering_configs == 0`` (steering disabled at the engine
    level — ``vllm_config.steering_config is None``), this is a no-op.
    No buffers are attached to ``module``, which causes
    :func:`apply_layer_steering` to short-circuit so the steering
    kernel never launches.  This keeps disabled-mode forwards free of
    steering overhead.
    """
    if max_steering_configs == 0:
        return
    table_dtype = dtype if dtype is not None else torch.float32
    if hook_widths is None:
        hook_widths = {hp: hidden_size for hp in STANDARD_STEERING_HOOKS}
    for hp, width in hook_widths.items():
        module.register_buffer(
            HOOK_POINT_TABLE_ATTR[hp],
            torch.zeros(max_steering_configs + 3, width, dtype=table_dtype),
            persistent=False,
        )
        # Per-hook activity flag.  A single-element bool tensor that the
        # ``apply_steering`` kernel reads at launch and uses to skip the
        # gather + add when no rows are currently active for this hook
        # point.  The flag is a tensor (not a Python bool) so the
        # ``torch.compile`` graph topology stays stable across batches
        # with different active-hook sets — only the data in the tensor
        # changes between forward passes.
        module.register_buffer(
            HOOK_POINT_ANY_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )

    module.register_buffer(
        "steering_index",
        torch.zeros(max_steering_tokens, dtype=torch.long),
        persistent=False,
    )


def get_steering_buffer_config(vllm_config: "VllmConfig") -> tuple[int, int]:
    """Return ``(max_tokens, max_configs)`` for steering buffers."""
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    steering_config = getattr(vllm_config, "steering_config", None)
    max_configs = steering_config.max_steering_configs if steering_config else 0
    return max_tokens, max_configs


def get_steering_buffer_dtype(vllm_config: "VllmConfig") -> torch.dtype:
    """Return the dtype that steering table buffers should be allocated in.

    Mirrors :func:`get_steering_buffer_config`. Returns the resolved
    ``torch.dtype`` that the model was loaded with so steering table
    rows can be gathered into ``hidden_states`` without an extra cast.
    """
    return vllm_config.model_config.dtype


def share_steering_index_across_layers(layers: list[nn.Module]) -> None:
    """Reuse one ``steering_index`` tensor across all steerable layers."""
    shared_index: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_index"):
            continue
        if shared_index is None:
            shared_index = layer.steering_index
            continue
        layer.steering_index = shared_index


def apply_layer_steering(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Apply the steering table for ``hook_point`` to ``hidden_states``.

    Capture consumers (when configured) see the pre-steering residual via
    :func:`maybe_capture_residual`.

    When the layer has no steering table buffer registered (engine
    started with ``enable_steering=False``, so
    :func:`register_steering_buffers` was a no-op), this short-circuits
    and returns ``hidden_states`` unchanged.  The ``hasattr`` check is
    decided once at module ``__init__`` and is constant for the rest
    of the layer's lifetime, so ``torch.compile`` traces it as a static
    branch and the disabled path emits no steering kernel at all.
    """
    maybe_capture_residual(hidden_states, module.layer_idx, hook_point.value)
    table_attr = HOOK_POINT_TABLE_ATTR[hook_point]
    if not hasattr(module, table_attr):
        return hidden_states
    return torch.ops.vllm.apply_steering(
        hidden_states,
        getattr(module, table_attr),
        module.steering_index,
        getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hook_point]),
    )


def apply_layer_steering_streams(
    module: nn.Module,
    streams: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Apply the steering table for ``hook_point`` to a multi-stream residual.

    ``streams`` is ``(num_tokens, num_streams, hidden)`` — the mHC residual
    carried as ``num_streams`` parallel hidden-size streams. The per-stream
    steering vector is stored flattened as a ``(num_streams * hidden,)``
    table row, so the tensor is flattened to ``(num_tokens, num_streams *
    hidden)`` for both the capture tap and the indexed gather/add, then
    reshaped back to the original 3-D shape.

    This mirrors :func:`apply_layer_steering` exactly — capture-then-steer,
    with the disabled path (no table buffer registered for this hook)
    decided once at ``__init__`` so ``torch.compile`` traces it as a static
    branch. When steering is disabled the flattened view is dead and folds
    out of the compiled graph, leaving the capture tap's own ``None`` gate
    as the only residual cost.
    """
    flat = streams.flatten(1)
    maybe_capture_residual(flat, module.layer_idx, hook_point.value)
    table_attr = HOOK_POINT_TABLE_ATTR[hook_point]
    if not hasattr(module, table_attr):
        return streams
    steered = torch.ops.vllm.apply_steering(
        flat,
        getattr(module, table_attr),
        module.steering_index,
        getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hook_point]),
    )
    return steered.view_as(streams)


def apply_steering(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Apply per-request activation steering via indexed gather.

    ``steering_table`` is a per-layer buffer of shape
    ``(max_configs + 3, hidden_size)`` where row 0 is always zeros
    (no-steering sentinel), row 1 holds the global prefill effective
    vector, row 2 holds the global decode effective vector, and rows
    3+ hold combined phase-appropriate global + per-request vectors.

    ``steering_index`` is a shared buffer of shape ``(max_tokens,)``
    mapping each token position to its steering table row.  Updated
    in-place by the model runner before each forward pass.

    ``any_active`` is a single-element bool tensor co-located with the
    ``steering_table`` buffer that the model runner sets to ``True``
    whenever at least one non-zero row exists for this hook point in
    the current batch and ``False`` otherwise.  When ``False``, the
    kernel skips the gather + add and emits a copy of
    ``hidden_states`` so the output value is unchanged.  The flag is a
    tensor (not a Python bool) so the ``torch.compile`` graph topology
    stays stable across batches whose active-hook set differs — only
    the data in the tensor changes between forward passes.

    The compute path dispatches to a fused Triton kernel on CUDA which
    folds the gather and add into a single pass over ``hidden_states``.
    The CPU path is a plain eager add. ``steering_table`` is allocated in
    the model's compute dtype via :func:`register_steering_buffers`, so
    the gather already matches ``hidden_states.dtype`` and no cast is
    needed in either path. The output is always a freshly allocated
    tensor so the ``torch.compile`` graph keeps value semantics — never
    in place.

    Note: even with ``any_active`` False the kernel still launches and
    writes ``hidden_states`` into a fresh output tensor (a memcpy).  A
    full no-op skip — the kernel returns immediately without touching
    output memory — requires combining with the in-place sibling branch
    (``mutates_args=["hidden_states"]``) so the op can elide the output
    copy entirely.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.steering_kernel import (
            apply_steering_triton,
        )

        return apply_steering_triton(
            hidden_states, steering_table, steering_index, any_active
        )
    # CPU eager: short-circuit on the host so we don't even materialize
    # the gather. ``.item()`` synchronizes against the device producer
    # for the flag tensor — irrelevant for the CPU path (the flag is
    # always written from the same thread before this op runs).
    if not bool(any_active.item()):
        # Match the freshly-allocated-output contract of the CUDA path so
        # callers never see an alias of ``hidden_states``.
        return hidden_states.clone()
    return hidden_states + steering_table[steering_index[: hidden_states.shape[0]]]


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
    mutates_args=[],
)
