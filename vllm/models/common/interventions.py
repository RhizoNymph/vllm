# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared steering/capture/patch wiring for natively implemented decoders.

Every intervention family (additive steering, clamps, dynamic steering,
patching, activation capture, SAE steering) dispatches through the same
``apply_layer_steering`` / ``apply_block_steering`` call sites, so a decoder
layer only needs two things to take part:

* per-layer buffers, registered once in ``__init__`` (this module's
  :func:`register_layer_steering_buffers`), and
* one shared ``steering_index`` across the model's layers
  (``share_steering_index_across_layers`` in the model ``__init__``).

:func:`interventions_configured` is the switch the hardware-split models use
to give up fusions that hide or split a hook site (deferred all-reduces,
fused residual+norm tails, whole-layer decode megakernels). Servers with no
intervention configured keep the fused path unchanged.
"""

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
)

_INTERVENTION_CONFIG_FIELDS = (
    "steering_config",
    "patch_config",
    "capture_consumers_config",
)


def interventions_configured(vllm_config: VllmConfig | None) -> bool:
    """Whether steering, patching, or activation capture can touch the model.

    SAE steering, clamps, and dynamic steering all ride the steering config,
    so these three fields cover every intervention family.
    """
    if vllm_config is None:
        return False
    return any(
        getattr(vllm_config, name, None) is not None
        for name in _INTERVENTION_CONFIG_FIELDS
    )


def register_layer_steering_buffers(
    layer: nn.Module,
    vllm_config: VllmConfig,
    hidden_size: int,
    *,
    hook_widths: dict[SteeringHookPoint, int] | None = None,
) -> None:
    """Register the per-layer intervention buffers on ``layer``.

    ``layer.layer_idx`` must already be set: the runner keys steerable layers
    by it. A no-op when steering is not configured (``max_configs == 0``).
    """
    if not hasattr(layer, "layer_idx"):
        raise AttributeError(
            f"{type(layer).__name__}.layer_idx must be set before registering "
            "steering buffers"
        )
    max_tokens, max_configs = get_steering_buffer_config(vllm_config)
    register_steering_buffers(
        layer,
        hidden_size,
        max_steering_tokens=max_tokens,
        max_steering_configs=max_configs,
        dtype=get_steering_buffer_dtype(vllm_config),
        hook_widths=hook_widths,
    )
