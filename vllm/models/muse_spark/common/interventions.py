# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering/capture/patch wiring shared by the NVIDIA and ROCm Muse Spark layers.

Muse Spark carries an FP32 residual stream blended by a gated, norm-preserving
connection, ``new = alpha * stream + beta * post_norm(branch)``. Both stacks
defer each layer's FFN blend into the next layer's fused
post-norm/blend/pre-norm boundary kernel, so a layer normally returns an
unblended ``MuseSparkResidualStream`` and no tensor ever holds its output.

When an intervention is configured the target decoder layers take the hooked
path instead, which materializes every hook site:

* ``pre_attn``: the FP32 stream entering attention (after the input boundary).
* ``post_attn``: the FP32 stream after the gated attention blend.
* ``mlp_in``: the normed FFN/MoE input.
* ``mlp_out``: the post-FFN-normed branch (FP32), the term the blend adds.
* ``post_block``: the blended stream, which the layer then returns as a plain
  tensor (the next layer's input boundary is then a single pre-norm). The
  layer output is fully materialized, so this site uses
  ``apply_layer_steering(..., POST_BLOCK)`` like other materialized-output
  layers rather than the deferred-add ``apply_block_steering``.

Because the blend is gated, a vector added at ``pre_attn`` / ``post_attn``
reaches the block output scaled by the downstream ``alpha`` coefficients;
``post_block`` edits land exactly.

Drafter layers (DSpark) reuse the target decoder layer class but are not
target layers: they register no buffers and keep their own forward.
"""

from collections.abc import Callable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.models.common.interventions import (
    interventions_configured,
    register_layer_steering_buffers,
)

HookFn = Callable[[torch.Tensor, SteeringHookPoint], torch.Tensor]


class MuseSparkInterventionsUnsupportedError(ValueError):
    """A configured backend runs the decoder outside the hooked Python forward."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(
            f"Muse Spark {backend} runs the decoder layers in fused kernels that "
            "bypass the steering/patching/capture hooks; it cannot be combined "
            "with a configured intervention. Drop it or the intervention config."
        )


def wire_decoder_layer_interventions(
    layer: nn.Module, vllm_config: VllmConfig, *, is_draft: bool
) -> bool:
    """Register ``layer``'s buffers; return whether it takes the hooked path.

    ``layer.layer_idx`` and ``layer.hidden_size`` must already be set. Draft
    layers are skipped so their indices never collide with target layers.
    """
    if is_draft:
        return False
    register_layer_steering_buffers(layer, vllm_config, layer.hidden_size)
    return interventions_configured(vllm_config)


def refuse_hook_bypassing_backend(
    vllm_config: VllmConfig, *, enabled: bool, backend: str
) -> None:
    """Raise if ``backend`` (a whole-layer fused decode) meets an intervention."""
    if enabled and interventions_configured(vllm_config):
        raise MuseSparkInterventionsUnsupportedError(backend)


def blend_ffn_with_hooks(
    layer: nn.Module, stream: torch.Tensor, branch: torch.Tensor, hook: HookFn
) -> torch.Tensor:
    """Post-FFN norm, ``mlp_out`` hook, gated blend and ``post_block`` hook.

    Returns the blended FP32 stream (the layer output).
    """
    normed = layer.post_ffn_norm(branch, out_dtype=stream.dtype)
    normed = hook(normed, SteeringHookPoint.MLP_OUT)
    stream = layer.add_ffn(stream=stream, branch=normed)
    return hook(stream, SteeringHookPoint.POST_BLOCK)
