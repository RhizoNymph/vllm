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


# Buffer attribute names on decoder layer modules, keyed by hook point.
HOOK_POINT_TABLE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "steering_table_pre_attn",
    SteeringHookPoint.POST_ATTN: "steering_table_post_attn",
    SteeringHookPoint.POST_MLP: "steering_table_post_mlp",
}

# Per-hook ``any-active`` flag attribute names. The flag is a single-element
# bool tensor co-located with each hook point's table buffer; the apply
# kernel reads it at launch and short-circuits the gather + add when no row
# is currently active for that hook point. The attribute name is derived
# from the table attribute so the two are always discoverable together.
HOOK_POINT_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_any_active" for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}

# Per-hook dedicated dynamic-tier vector attribute names (§5.4). A single
# fp32 ``(hidden,)`` buffer per hook point holding that hook's global
# dynamic-tier vector; the apply kernel adds ``dynamic_vec * token_scales``
# on top of the row gather, gated per token (decode-only). Derived from the
# table attribute so the two are discoverable together.
HOOK_POINT_DYNVEC_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_dynvec" for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
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
) -> None:
    """Attach per-hook steering buffers to a decoder layer.

    ``dtype`` controls the storage dtype of the steering table buffers.
    When ``None`` (the default), the buffers fall back to fp32 to
    preserve historical behaviour.  Callers in vLLM models pass the
    model's compute dtype (typically bf16) so the indexed gather in
    :func:`apply_steering` returns rows already aligned with the residual
    tensor and no dtype cast is required at the gather site.

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
    for hp in SteeringHookPoint:
        module.register_buffer(
            HOOK_POINT_TABLE_ATTR[hp],
            torch.zeros(max_steering_configs + 3, hidden_size, dtype=table_dtype),
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
        # Per-hook dedicated dynamic-tier vector (§5.4): fp32 (hidden,),
        # default 0 ⇒ no tier. The manager writes it from
        # ``dynamic_tier_vectors`` in populate; the kernel adds
        # ``dynamic_vec * token_scales`` on top of the row gather.
        module.register_buffer(
            HOOK_POINT_DYNVEC_ATTR[hp],
            torch.zeros(hidden_size, dtype=torch.float32),
            persistent=False,
        )

    module.register_buffer(
        "steering_index",
        torch.zeros(max_steering_tokens, dtype=torch.long),
        persistent=False,
    )

    # Per-token dynamic-tier gate (§5.4): fp32 (max_tokens,), default 0,
    # shared across layers like ``steering_index`` (shared in
    # ``_init_steering_state``). The runner writes it each step —
    # ``dynamic_tier_gain`` for decode tokens of a tier-active state, 0
    # otherwise (so the tier stays decode-only). Phase 2 replaces the
    # runner write with an in-graph monitor.
    module.register_buffer(
        "steering_token_scales",
        torch.zeros(max_steering_tokens, dtype=torch.float32),
        persistent=False,
    )

    # Per-row strength scale (the §5.3 "how much" knob): one fp32 buffer
    # per layer, shared across that layer's hook points (the row index is
    # hook-independent — row 3 = config X for every hook). Default 1.0 ⇒
    # unscaled steering; the kernel multiplies the gathered row by
    # ``scales[row]``. The manager writes the same per-row values into
    # every layer's buffer during populate.
    module.register_buffer(
        "steering_scales",
        torch.ones(max_steering_configs + 3, dtype=torch.float32),
        persistent=False,
    )


def get_steering_buffer_config(vllm_config: "VllmConfig") -> tuple[int, int]:
    """Return ``(max_tokens, max_configs)`` for steering buffers.

    ``max_configs`` is the total row budget above the three reserved
    rows: the scheduler-admitted per-request pool
    (``max_steering_configs``) plus the dynamic-override pool
    (``max_dynamic_steering_configs`` — rows allocated at runtime by
    dynamic steering; see docs/design/dynamic_steering.md §5.2). Every
    model sizes its tables through this single function, so the pool
    split needs no model-file knowledge.
    """
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    steering_config = getattr(vllm_config, "steering_config", None)
    if steering_config is None:
        return max_tokens, 0
    max_configs = steering_config.max_steering_configs + getattr(
        steering_config, "max_dynamic_steering_configs", 0
    )
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


def share_steering_token_scales_across_layers(layers) -> None:
    """Reuse one ``steering_token_scales`` tensor across all steerable layers.

    The per-token dynamic-tier gate (§5.4) is layer-independent and
    ``max_tokens``-sized, so it is shared (one per-step H2D, not per
    layer) exactly like ``steering_index``. Called once from the mixin's
    ``_init_steering_state`` (not per model file) since it has every
    steerable layer in hand at that point.
    """
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_token_scales"):
            continue
        if shared is None:
            shared = layer.steering_token_scales
            continue
        layer.steering_token_scales = shared


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
        module.steering_scales,
        getattr(module, HOOK_POINT_DYNVEC_ATTR[hook_point]),
        module.steering_token_scales,
    )


def apply_steering(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_scales: torch.Tensor,
    steering_dynamic_vec: torch.Tensor,
    steering_token_scales: torch.Tensor,
) -> torch.Tensor:
    """Apply per-request activation steering via indexed gather.

    Two additive terms: the per-row gather
    ``table[index[i]] * scales[index[i]]`` plus the **dedicated dynamic
    tier** ``dynamic_vec * token_scales[i]`` (§5.4).

    ``steering_scales`` is a shared buffer of shape ``(max_configs + 3,)``
    (fp32, default 1.0) holding a per-row strength multiplier — the
    runtime "how much" knob. The gathered row is scaled by
    ``scales[index[i]]`` before the add, so changing strength needs only
    a cheap scales write, no vector re-upload (see §5.3). A scale of 1.0
    reproduces the unscaled steering.

    ``steering_dynamic_vec`` is this (layer, hook)'s dynamic-tier vector
    (fp32, ``(hidden,)``, default 0); ``steering_token_scales`` is a
    shared per-token gate (fp32, ``(max_tokens,)``, default 0, and 0 for
    prefill tokens). The tier add ``dynamic_vec * token_scales[i]`` lets
    global dynamic steering be modulated per token and stay decode-only;
    with the default-0 gate it contributes nothing.

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
            hidden_states,
            steering_table,
            steering_index,
            any_active,
            steering_scales,
            steering_dynamic_vec,
            steering_token_scales,
        )
    # CPU eager: short-circuit on the host so we don't even materialize
    # the gather. ``.item()`` synchronizes against the device producer
    # for the flag tensor — irrelevant for the CPU path (the flag is
    # always written from the same thread before this op runs).
    if not bool(any_active.item()):
        # Match the freshly-allocated-output contract of the CUDA path so
        # callers never see an alias of ``hidden_states``.
        return hidden_states.clone()
    n = hidden_states.shape[0]
    rows = steering_index[:n]
    # Per-row scale (fp32, default 1.0); broadcast over hidden dim.
    scale = steering_scales[rows].unsqueeze(-1).to(steering_table.dtype)
    out = hidden_states + steering_table[rows] * scale
    # Dedicated dynamic tier: dvec * per-token gate (0 ⇒ no-op).
    tier = steering_dynamic_vec.unsqueeze(0) * steering_token_scales[:n].unsqueeze(-1)
    return out + tier.to(out.dtype)


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_scales: torch.Tensor,
    steering_dynamic_vec: torch.Tensor,
    steering_token_scales: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
    mutates_args=[],
)
