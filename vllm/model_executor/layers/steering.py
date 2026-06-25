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

# Kept as plain strings here so the no-SAE hot path can short-circuit
# without importing ``sae_steering`` on every decoder-layer hook call.
HOOK_POINT_SAE_CLAMP_KIND_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_clamp_kind_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_clamp_kind_post_attn",
    SteeringHookPoint.POST_MLP: "sae_clamp_kind_post_mlp",
}

# Full-reconstruction (Phase 4) site marker.  Same pattern as the delta
# marker above — kept as plain strings so the no-FR hot path doesn't
# import ``sae_full_reconstruction`` on every decoder-layer hook call.
HOOK_POINT_SAE_FR_CLAMP_KIND_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_clamp_kind_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_clamp_kind_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_clamp_kind_post_mlp",
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

    Composition order at this hook (per ``docs/features/sae_steering.md``):

    1. Additive steering (the existing precomputed-vector path).
    2. SAE feature-surgery delta (encode → clamp → decoder-direction
       add).  Runs on the additively-steered residual so the additive
       behaviour is unchanged when SAE is added on top.
    3. SAE full reconstruction (replacement) — replaces the residual
       at tokens whose ``sae_recon_index != 0``.  Runs last so any
       additive / delta perturbations *upstream* of the replacement
       are still observable on tokens that don't opt into
       reconstruction; tokens that *do* opt in have their residual
       replaced wholesale, discarding the upstream perturbations
       (which matches the design-doc replacement semantics).

    Each stage short-circuits independently via a static ``hasattr``
    check; ``torch.compile`` decides the branch once at module
    instantiation, so a disabled-mode forward emits zero steering
    kernels.

    Capture consumers (when configured) see the pre-steering residual
    via :func:`maybe_capture_residual`, which runs unconditionally —
    capture is independent of whether steering is enabled.
    """
    maybe_capture_residual(hidden_states, module.layer_idx, hook_point.value)
    table_attr = HOOK_POINT_TABLE_ATTR[hook_point]
    buffers = module._buffers
    steering_table = buffers.get(table_attr)
    if steering_table is not None:
        hidden_states = torch.ops.vllm.apply_steering(
            hidden_states,
            steering_table,
            module.steering_index,
            buffers[HOOK_POINT_ANY_ACTIVE_ATTR[hook_point]],
        )
    has_sae_delta = HOOK_POINT_SAE_CLAMP_KIND_ATTR[hook_point] in buffers
    has_sae_fr = HOOK_POINT_SAE_FR_CLAMP_KIND_ATTR[hook_point] in buffers
    if not has_sae_delta and not has_sae_fr:
        return hidden_states

    # SAE feature-surgery delta runs after the additive op on the same
    # hook.  Local import keeps ``steering.py`` free of any dependency
    # on ``sae_steering.py`` at module load time (``sae_steering.py``
    # imports ``SteeringHookPoint`` from here, so a top-level import
    # would form a cycle).  The guard above keeps disabled/no-SAE
    # forwards on the additive-only path.
    if has_sae_delta:
        from vllm.model_executor.layers.sae_steering import apply_layer_sae_delta

        hidden_states = apply_layer_sae_delta(module, hidden_states, hook_point)
    if has_sae_fr:
        from vllm.model_executor.layers.sae_full_reconstruction import (
            apply_layer_sae_full_reconstruction,
        )

        hidden_states = apply_layer_sae_full_reconstruction(
            module, hidden_states, hook_point
        )
    return hidden_states


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
