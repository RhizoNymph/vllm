# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared steering/capture wiring for the hardware-split Qwen3.5 models."""

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.intervention_common import interventions_configured
from vllm.model_executor.layers.steering import (
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
)

# Re-exported for the platform layers. The NVIDIA and ROCm decoder layers defer
# each MLP/MoE all-reduce into the next norm, so under TP>1 the branch output
# at ``mlp_out`` / ``post_block`` is an un-reduced partial; the NVIDIA layer
# also quantizes the normed MLP input for ``gate_up_proj``, leaving ``mlp_in``
# no full-precision tensor to steer. Those fusions are turned off when
# ``interventions_configured`` holds so every hook point observes and edits the
# complete activation.
__all__ = ["interventions_configured", "register_qwen3_5_steering_buffers"]


def register_qwen3_5_steering_buffers(
    layer: nn.Module, vllm_config: VllmConfig, hidden_size: int
) -> None:
    """Register per-request steering buffers on a Qwen3.5 decoder layer.

    These layers skip ``Qwen3NextDecoderLayer.__init__`` (which would
    otherwise register them), so each platform's layer calls this directly.
    ``layer.layer_idx`` must already be set.
    """
    max_steering_tokens, max_steering_configs = get_steering_buffer_config(vllm_config)
    register_steering_buffers(
        layer,
        hidden_size,
        max_steering_tokens=max_steering_tokens,
        max_steering_configs=max_steering_configs,
        dtype=get_steering_buffer_dtype(vllm_config),
    )
