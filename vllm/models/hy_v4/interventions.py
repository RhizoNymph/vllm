# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering/capture wiring for the HY V4 decoder layer (NVIDIA, ROCm, Flash).

``HYV4DecoderLayer`` runs in one of two shapes:

* **iHC** (``config.enable_ihc``): ``hc_mult`` residual channels, reduced by a
  sigmoid-gated pre op before each sub-block and re-scattered by a post op
  (no Sinkhorn comb mixing). These layers take the DeepSeek-V4 mHC hook set
  (see ``vllm/models/deepseek_v4/common/interventions.py`` and the "mHC
  Steering" section of ``docs/features/steering.md``): single-stream
  ``pre_attn`` / ``post_attn`` / ``mlp_in`` / ``mlp_out`` plus the
  multi-stream ``mhc_streams_pre_attn`` / ``mhc_streams_pre_mlp`` and the
  model-level ``mhc_streams_final`` (owned by the last layer). There is no
  single-stream ``post_block``.
* **single residual** (no iHC): the standard five hooks.

Every hook call goes through this module so draft blocks that reuse the
decoder class (``HYV4MTP``'s ``mtp_block``, the DFly drafter's layers, whose
layer indices restart at 0 and would collide with the target's) set
``intervention_hooks = False`` and neither register buffers nor fire capture
taps.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation_capture import (
    get_active_capture_manager,
    maybe_capture_residual,
)
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_block_steering,
    apply_layer_steering,
    apply_layer_steering_streams,
)
from vllm.models.common.interventions import (
    interventions_configured,
    register_layer_steering_buffers,
)

__all__ = [
    "interventions_configured",
    "register_hy_v4_steering_buffers",
    "steer",
    "steer_and_capture_ihc",
    "steer_block",
    "steer_streams",
]


def register_hy_v4_steering_buffers(
    layer: nn.Module,
    vllm_config: VllmConfig,
    *,
    ihc: bool,
    is_last_layer: bool,
) -> None:
    """Register the per-layer intervention buffers on a target decoder layer.

    ``layer`` must carry ``layer_idx`` and ``hidden_size`` and, when ``ihc``,
    ``hc_mult``. iHC layers get the DeepSeek-V4 mHC widths: ``hidden`` for
    the single-stream hooks, ``hc_mult * hidden`` for the channel hooks; only
    the last layer owns the model-level ``mhc_streams_final`` table.

    Every steering / patch / clamp buffer keys off the intervention configs,
    so with none configured there is nothing to register.
    """
    if not interventions_configured(vllm_config):
        return
    if not ihc:
        register_layer_steering_buffers(layer, vllm_config, layer.hidden_size)
        return
    hidden = layer.hidden_size
    hc_dim = layer.hc_mult * hidden
    hook_widths = {
        SteeringHookPoint.PRE_ATTN: hidden,
        SteeringHookPoint.POST_ATTN: hidden,
        SteeringHookPoint.MLP_IN: hidden,
        SteeringHookPoint.MLP_OUT: hidden,
        SteeringHookPoint.MHC_STREAMS_PRE_ATTN: hc_dim,
        SteeringHookPoint.MHC_STREAMS_PRE_MLP: hc_dim,
    }
    if is_last_layer:
        hook_widths[SteeringHookPoint.MHC_STREAMS_FINAL] = hc_dim
    register_layer_steering_buffers(layer, vllm_config, hidden, hook_widths=hook_widths)


def steer(
    layer: nn.Module, hidden_states: torch.Tensor, hook_point: SteeringHookPoint
) -> torch.Tensor:
    """``apply_layer_steering`` for target layers; identity on draft blocks."""
    if not layer.intervention_hooks:
        return hidden_states
    return apply_layer_steering(layer, hidden_states, hook_point)


def steer_streams(
    layer: nn.Module, streams: torch.Tensor, hook_point: SteeringHookPoint
) -> torch.Tensor:
    """``apply_layer_steering_streams`` for target layers; identity otherwise."""
    if not layer.intervention_hooks:
        return streams
    return apply_layer_steering_streams(layer, streams, hook_point)


def steer_block(
    layer: nn.Module, hidden_states: torch.Tensor, residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``apply_block_steering`` for target layers; identity on draft blocks."""
    if not layer.intervention_hooks:
        return hidden_states, residual
    return apply_block_steering(layer, hidden_states, residual)


def steer_and_capture_ihc(
    layer: nn.Module,
    residual: torch.Tensor,
    post_gates: torch.Tensor,
    layer_input: torch.Tensor,
    sublayer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Steer and capture one sub-block's iHC activations.

    The DeepSeek-V4 ``steer_and_capture_mhc`` contract: ``residual``
    (``(tokens, hc_mult, hidden)``) holds the channels the iHC pre op just
    reduced, so a channel vector reaches the post op's residual add and the
    layers after it, not this sub-block's own input; ``layer_input`` is the
    normed single-stream sub-block input. The fp32 post gates
    (``(tokens, hc_mult)``) are captured as ``mhc_*_post_mix``; iHC has no comb
    matrix, so ``mhc_*_res_mix`` never fires. Returns the possibly steered
    ``(residual, layer_input)``.
    """
    if not layer.intervention_hooks:
        return residual, layer_input
    if sublayer == "attn":
        stream_hp, in_hp = (
            SteeringHookPoint.MHC_STREAMS_PRE_ATTN,
            SteeringHookPoint.PRE_ATTN,
        )
        post_hook = "mhc_attn_post_mix"
    else:
        stream_hp, in_hp = (
            SteeringHookPoint.MHC_STREAMS_PRE_MLP,
            SteeringHookPoint.MLP_IN,
        )
        post_hook = "mhc_ffn_post_mix"
    residual = apply_layer_steering_streams(layer, residual, stream_hp)
    if get_active_capture_manager() is not None:
        maybe_capture_residual(post_gates.flatten(1), layer.layer_idx, post_hook)
    layer_input = apply_layer_steering(layer, layer_input, in_hp)
    return residual, layer_input
