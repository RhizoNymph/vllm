# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime hook sites called from rewritten Transformers decoder layers.

`rewrite.rewrite_forward` replaces each residual add of a decoder layer with a
call to one of these functions, so the native intervention stack
(`apply_layer_steering` / `apply_block_steering`: capture -> patch -> steer ->
SAE -> clamp) runs at the same tensors it runs at on native decoder layers.

Transformers layers carry a batch dimension of one (`[1, tokens, hidden]`)
while the intervention ops index tokens on dim 0, so each site works on a
token-major `[tokens, hidden]` view and returns the original tensor object
untouched when nothing is attached at that site. Nothing here allocates
buffers or branches on data, so the rewritten forward stays traceable by
`torch.compile` and replayable by CUDA graphs exactly like the native hooks.

Under tensor parallelism every tensor that reaches these sites is already
fully reduced: the backend's row-parallel output projections and fused MoE
layers all-reduce their own results, and no fusion defers a reduction past a
decoder-layer residual add.
"""

import torch
from torch import nn

from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_block_steering,
    apply_layer_steering,
)


def _site(layer: nn.Module, x: torch.Tensor, hook: SteeringHookPoint) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    out = apply_layer_steering(layer, flat, hook)
    return x if out is flat else out.view(x.shape)


def _block(layer: nn.Module, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """`residual + branch` with the `post_block` site applied to the sum."""
    if residual.shape != branch.shape:
        raise ValueError(
            f"post_block expects the residual {tuple(residual.shape)} and the MLP "
            f"branch {tuple(branch.shape)} to have the same shape"
        )
    flat_branch = branch.reshape(-1, branch.shape[-1])
    flat_residual = residual.reshape(-1, residual.shape[-1])
    flat_branch, flat_residual = apply_block_steering(layer, flat_branch, flat_residual)
    return (flat_residual + flat_branch).view(residual.shape)


def mlp_input(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """`mlp_in`: the normed hidden states entering the MLP sublayer."""
    return _site(layer, hidden_states, SteeringHookPoint.MLP_IN)


def attn_residual_add(
    layer: nn.Module, residual: torch.Tensor, attn: torch.Tensor
) -> torch.Tensor:
    """The attention residual add of a sequential block.

    `pre_attn` acts on the skip operand. The attention input was computed from
    the same residual before this add, so it is unaffected, as on native
    fused-residual layers; the steered skip flows on into the MLP block."""
    return _site(layer, residual, SteeringHookPoint.PRE_ATTN) + attn


def mlp_residual_add(
    layer: nn.Module, residual: torch.Tensor, mlp: torch.Tensor
) -> torch.Tensor:
    """The MLP residual add of a sequential block.

    `post_attn` acts on the skip operand, whose MLP input has already been
    taken; `mlp_out` on the whole MLP branch; `post_block` on their sum."""
    residual = _site(layer, residual, SteeringHookPoint.POST_ATTN)
    mlp = _site(layer, mlp, SteeringHookPoint.MLP_OUT)
    return _block(layer, residual, mlp)


def parallel_residual_add(
    layer: nn.Module, residual: torch.Tensor, attn: torch.Tensor, mlp: torch.Tensor
) -> torch.Tensor:
    """The single residual add of a parallel block, `residual + attn + mlp`.

    Evaluated as `(residual + attn) + mlp` so `post_attn` observes the
    post-attention residual, as the native parallel (Phi) layer did. That
    re-associates the original sum, so outputs can differ from the unhooked
    model by floating-point rounding."""
    residual = _site(layer, residual, SteeringHookPoint.PRE_ATTN)
    post_attn = _site(layer, residual + attn, SteeringHookPoint.POST_ATTN)
    mlp = _site(layer, mlp, SteeringHookPoint.MLP_OUT)
    return _block(layer, post_attn, mlp)
