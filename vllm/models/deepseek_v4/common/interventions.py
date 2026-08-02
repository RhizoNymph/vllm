# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared steering/capture wiring for the DeepSeek-V4 mHC decoder layer.

The nvidia and amd model files implement the same interp contract at the
same hook points; this module holds that wiring once so the platforms
cannot drift. The hook string names match the capture framework's mHC hook
names, so one identifier captures and steers the same tensor.
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
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
)


def register_mhc_steering_buffers(
    layer: nn.Module,
    vllm_config: VllmConfig,
    *,
    is_last_layer: bool,
) -> None:
    """Register per-request steering buffers on a V4 decoder layer.

    The single-stream sublayer in/out tensors are steered at ``hidden``
    width; the multi-stream residual hooks at ``hc_mult * hidden`` (one
    flattened per-stream vector per row). ``mhc_streams_final`` is a
    model-level hook keyed to the last layer, so only that layer registers
    its table — matching the capture framework's tail attribution.

    ``layer`` must already carry ``hidden_size`` and ``hc_mult``.
    """
    hc_dim = layer.hc_mult * layer.hidden_size
    max_steering_tokens, max_steering_configs = get_steering_buffer_config(vllm_config)
    hook_widths = {
        SteeringHookPoint.PRE_ATTN: layer.hidden_size,
        SteeringHookPoint.POST_ATTN: layer.hidden_size,
        SteeringHookPoint.MLP_IN: layer.hidden_size,
        SteeringHookPoint.MLP_OUT: layer.hidden_size,
        SteeringHookPoint.MHC_STREAMS_PRE_ATTN: hc_dim,
        SteeringHookPoint.MHC_STREAMS_PRE_MLP: hc_dim,
    }
    if is_last_layer:
        hook_widths[SteeringHookPoint.MHC_STREAMS_FINAL] = hc_dim
    register_steering_buffers(
        layer,
        layer.hidden_size,
        max_steering_tokens=max_steering_tokens,
        max_steering_configs=max_steering_configs,
        dtype=get_steering_buffer_dtype(vllm_config),
        hook_widths=hook_widths,
    )


def steer_and_capture_mhc(
    layer: nn.Module,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    layer_input: torch.Tensor,
    sublayer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Steer and capture one sublayer's mHC activations.

    Returns the (possibly steered) ``(residual, layer_input)`` pair.

    The multi-stream residual and the single-stream sublayer input are
    routed through the steering helpers, which capture the pre-steering
    value via the capture framework and then add any registered steering
    vector. When neither steering nor capture is active for a hook the
    helpers short-circuit (a static branch decided at ``__init__``), so
    the work constant-folds out of the compiled graph.

    Both ``layer_input`` and the mixing coefficients were computed from the
    *pre-steering* streams (the mHC pre op has already consumed them), so a
    ``mhc_streams_*`` steer takes effect through the sublayer's mix-back
    and the residual carried forward — not through this sublayer's own
    input; steer the single-stream hook for that.

    The fp32 mixing coefficients are capture-only — they are routing
    weights, not a residual, and carry no steering semantics — so they
    stay behind the capture-manager gate, leaving their flattens to
    constant-fold out when capture is disabled.

    ``layer_input`` is the *normed* single-stream sublayer input on both
    platforms (the nvidia mHC pre kernels fuse ``attn_norm`` / ``ffn_norm``
    so the pre-norm tensor is never materialized; the amd path applies the
    norm explicitly just before this tap), matching the ``mlp_in`` contract
    in :class:`SteeringHookPoint`.

    ``residual`` is ``(num_tokens, hc_mult, hidden)``; ``post_mix`` is
    ``(num_tokens, hc_mult, 1)``; ``res_mix`` is ``(num_tokens, hc_mult,
    hc_mult)``; ``layer_input`` is ``(num_tokens, hidden)``.
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
