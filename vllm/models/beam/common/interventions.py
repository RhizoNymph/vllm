# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering/capture wiring shared by the NVIDIA and ROCm Beam decoder layers.

Beam keeps the full residual stream between layers and rescales it at every
sublayer join (``h = x_scale * post_norm(branch) + r_scale * h``). Upstream
fuses that join with the *next* sublayer's pre-norm (and, on NVIDIA, with an
MNNVL all-reduce of the branch), so no kernel ever materializes the branch
that joins the stream, and a hook on the joined stream would be ignored by the
already-normalized next input. With any intervention configured the layers
run :func:`beam_steerable_layer_forward` instead: same arithmetic, one tensor
per hook site.
"""

from collections.abc import Callable

import torch
from torch import nn

from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_block_steering,
    apply_layer_steering,
)

RmsNorm = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def _scaled_join(
    rms_norm: RmsNorm,
    branch: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    x_scale: float,
) -> torch.Tensor:
    """``x_scale * post_norm(branch)``, rounded like the fused boundary kernel."""
    return rms_norm(branch, weight, eps) * x_scale


def beam_steerable_layer_forward(
    layer: nn.Module,
    positions: torch.Tensor,
    residual: torch.Tensor,
    rms_norm: RmsNorm,
) -> tuple[torch.Tensor, None]:
    """Unfused Beam decoder layer emitting every intervention hook.

    ``layer`` must have been built with all-reduces inside its sublayers
    (``reduce_results=True``), so every branch here is complete. Returns the
    stream leaving the layer and ``None`` for the next normalized input: the
    next layer (or the final norm) normalizes the possibly-steered stream.
    """
    eps = layer.rms_norm_eps

    # Attention: the norm reads the incoming stream; pre_attn edits the stream
    # carried past attention (as in the fused-add models).
    x = rms_norm(residual, layer.input_layernorm.weight, eps)
    residual = apply_layer_steering(layer, residual, SteeringHookPoint.PRE_ATTN)
    x = layer.self_attn(positions, x)
    x_scale, r_scale = layer.attn_scales
    residual = residual * r_scale + _scaled_join(
        rms_norm, x, layer.post_attention_layernorm.weight, eps, x_scale
    )

    # MLP / MoE.
    x = rms_norm(residual, layer.pre_feedforward_layernorm.weight, eps)
    residual = apply_layer_steering(layer, residual, SteeringHookPoint.POST_ATTN)
    x = apply_layer_steering(layer, x, SteeringHookPoint.MLP_IN)
    x = layer.mlp(x)
    x_scale, r_scale = layer.mlp_scales
    branch = _scaled_join(
        rms_norm, x, layer.post_feedforward_layernorm.weight, eps, x_scale
    )
    branch = apply_layer_steering(layer, branch, SteeringHookPoint.MLP_OUT)
    # post_block observes ``r_scale * post_attn + mlp_out``, the stream leaving
    # the layer; steering rides the carried stream.
    branch, carried = apply_block_steering(layer, branch, residual * r_scale)
    return carried + branch, None
