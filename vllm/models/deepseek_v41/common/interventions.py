# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering/capture wiring shared by the DeepSeek-V4.1 nvidia and amd decoders.

V4.1 carries DeepSeek-V4's mHC multi-stream residual, so it reuses V4's hook
set and per-sublayer helper (``deepseek_v4/common/interventions.py``). This
module adds the two V4.1-specific pieces both platforms need, so they cannot
drift:

* buffer registration that skips the DSpark draft layers (DSpark builds the
  same decoder layer class at global indices past the target's), and
* the model-level ``mhc_streams_final`` tap, which only the last PP rank's
  last layer owns.
"""

from collections.abc import Sequence

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_layer_steering_streams,
)
from vllm.models.deepseek_v4.common.interventions import (
    register_mhc_steering_buffers,
)


def register_v41_steering_buffers(
    layer: nn.Module, vllm_config: VllmConfig, num_hidden_layers: int
) -> None:
    """Register the mHC steering buffers on a target decoder layer.

    ``layer`` must already carry ``layer_idx`` (global), ``hidden_size`` and
    ``hc_mult``. Draft layers (``layer_idx >= num_hidden_layers``) register
    nothing: speculative drafters are not steered, and their indices are not
    target layers the runner or capture plans know.
    """
    if layer.layer_idx >= num_hidden_layers:
        return
    register_mhc_steering_buffers(
        layer,
        vllm_config,
        is_last_layer=layer.layer_idx == num_hidden_layers - 1,
    )


def steer_final_streams(
    layers: Sequence[nn.Module],
    start_layer: int,
    end_layer: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Steer and capture the final multi-stream residual before the collapse.

    Call only on the last PP rank, where ``layers[end_layer - 1]`` is the
    global last layer — the only one holding the ``mhc_streams_final`` table,
    matching the capture framework's tail attribution. A rank without local
    layers returns ``hidden_states`` unchanged.
    """
    if end_layer <= start_layer:
        return hidden_states
    return apply_layer_steering_streams(
        layers[end_layer - 1], hidden_states, SteeringHookPoint.MHC_STREAMS_FINAL
    )
