# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering/capture wiring for the GLM-5.3-Flash (``glm5next``) decoder layer.

The decoder layer is shared by the NVIDIA and ROCm paths (``common/model.py``)
and runs in one of two shapes:

* **mHC** (``config.mhc``): ``hc_mult`` residual streams mixed per token by
  the same TileLang / FlashInfer mHC pre/post ops as DeepSeek-V4. These
  layers take the DeepSeek-V4 mHC hook set (see
  ``vllm/models/deepseek_v4/common/interventions.py`` and the "mHC Steering"
  section of ``docs/features/steering.md``): single-stream ``pre_attn`` /
  ``post_attn`` / ``mlp_in`` / ``mlp_out`` plus the multi-stream
  ``mhc_streams_pre_attn`` / ``mhc_streams_pre_mlp`` and, on the last layer,
  ``mhc_streams_final``. There is no single-stream ``post_block``.
* **single residual** (no mHC): the standard five hooks.

The helpers mirror DeepSeek-V4's ``register_mhc_steering_buffers`` /
``steer_and_capture_mhc`` one-for-one; they are kept family-local so this
package does not import the DeepSeek-V4 platform package.
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
    apply_layer_steering,
    apply_layer_steering_streams,
)
from vllm.models.common.interventions import (
    interventions_configured,
    register_layer_steering_buffers,
)

__all__ = [
    "interventions_configured",
    "register_glm5next_steering_buffers",
    "steer_and_capture_mhc",
]


def register_glm5next_steering_buffers(
    layer: nn.Module,
    vllm_config: VllmConfig,
    *,
    mhc: bool,
    is_last_layer: bool,
) -> None:
    """Register the per-layer intervention buffers on a decoder layer.

    ``layer`` must carry ``layer_idx`` and ``hidden_size`` and, when ``mhc``,
    ``hc_mult``. mHC layers get the DeepSeek-V4 widths: ``hidden`` for the
    single-stream hooks, ``hc_mult * hidden`` for the stream hooks; only the
    last layer owns the model-level ``mhc_streams_final`` table.

    Every steering / patch / clamp buffer keys off the intervention configs,
    so with none configured there is nothing to register and the scheduler
    and model configs are not read.
    """
    if not interventions_configured(vllm_config):
        return
    if not mhc:
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


def steer_and_capture_mhc(
    layer: nn.Module,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    layer_input: torch.Tensor,
    sublayer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Steer and capture one sublayer's mHC activations.

    Same contract as DeepSeek-V4's helper: ``residual`` (``(tokens, hc_mult,
    hidden)``) is the multi-stream residual the mHC pre op just consumed, so a
    stream vector reaches the sublayer's mix-back and the carried residual,
    not this sublayer's own input; ``layer_input`` is the normed single-stream
    sublayer input (the pre op fuses the RMSNorm). The fp32 mixing
    coefficients are capture-only. Returns the possibly steered
    ``(residual, layer_input)``.
    """
    if sublayer == "attn":
        stream_hp, in_hp = (
            SteeringHookPoint.MHC_STREAMS_PRE_ATTN,
            SteeringHookPoint.PRE_ATTN,
        )
        post_hook, res_hook = "mhc_attn_post_mix", "mhc_attn_res_mix"
    else:
        stream_hp, in_hp = (
            SteeringHookPoint.MHC_STREAMS_PRE_MLP,
            SteeringHookPoint.MLP_IN,
        )
        post_hook, res_hook = "mhc_ffn_post_mix", "mhc_ffn_res_mix"
    residual = apply_layer_steering_streams(layer, residual, stream_hp)
    if get_active_capture_manager() is not None:
        maybe_capture_residual(post_mix.flatten(1), layer.layer_idx, post_hook)
        maybe_capture_residual(res_mix.flatten(1), layer.layer_idx, res_hook)
    layer_input = apply_layer_steering(layer, layer_input, in_hp)
    return residual, layer_input
