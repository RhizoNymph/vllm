# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering / clamp / patch / SAE / capture sites of the Inkling2 decoders.

Inkling2 has no single residual stream: block attention residuals keep a
stack of completed blocks plus the running block, and every sublayer reads
a learned softmax mix of them. The *residual stream* at a hook is the
running block's state `h`, the tensor the decoder carries from sublayer
to sublayer (`stack[num_blocks]`):

- `pre_attn`: `h` entering the layer (at the first layer of a block, the
  block just completed, which later mixes read);
- `post_attn`: `h + attn_out` (just `attn_out` at the first layer of a
  block, which starts a new block), `attn_out` being the attention output
  after its short conv;
- `post_block`: `h` leaving the layer, `(h +) attn_out + conv(moe_out)`;
- `mlp_in`: the normalized MoE input (AttnRes mix + RMSNorm, then the loop
  message blend in passes after the first);
- `mlp_out`: the MoE output, before the MLP short conv.

`pre_attn` / `post_attn` / `post_block` live in one space, so a captured
vector is a valid steering vector at any of them.

Every layer visit has its own site (`Inkling2InterventionSite`) whose
`layer_idx` is the visit's layer key (`vllm.config.loop_layers`): the
layer index, or for a loop core layer in pass `p`
`layer + (p + 1) * num_hidden_layers`. Per-request steering rows, clamp
directions, patch slots and SAE slots are per site; the per-token index /
gate / scale buffers are shared by every site.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.loop_layers import LoopLayerSpace
from vllm.model_executor.layers.steering import (
    SteeringHookPoint,
    apply_layer_steering,
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
    share_steering_index_across_layers,
)


def interventions_configured(vllm_config: VllmConfig | None) -> bool:
    """Whether steering, patching or capture can touch this model. The
    decoders then run the unfused short convs and AttnRes mixes, so every
    hook observes and edits the tensor the next sublayer reads (the fused
    conv/mix kernels compute the next mix before a hook could run)."""
    return vllm_config is not None and any(
        getattr(vllm_config, name, None) is not None
        for name in ("steering_config", "patch_config", "capture_consumers_config")
    )


class Inkling2InterventionSite(nn.Module):
    """The intervention buffers of one layer visit (`layer_idx` is its
    layer key)."""

    def __init__(
        self, layer_key: int, hidden_size: int, vllm_config: VllmConfig | None
    ) -> None:
        super().__init__()
        self.layer_idx = layer_key
        if vllm_config is None:
            return
        max_tokens, max_configs = get_steering_buffer_config(vllm_config)
        register_steering_buffers(
            self,
            hidden_size,
            max_steering_tokens=max_tokens,
            max_steering_configs=max_configs,
            dtype=get_steering_buffer_dtype(vllm_config),
        )


def visit_layer_keys(config) -> list[int]:
    """The layer key of each decoder-layer visit, in execution order."""
    space = LoopLayerSpace.from_text_config(config)
    if space is None:
        return list(range(config.num_hidden_layers))
    return space.visit_keys()


def make_intervention_sites(config, vllm_config: VllmConfig | None) -> nn.ModuleList:
    """One site per layer visit, in execution order."""
    sites = nn.ModuleList(
        Inkling2InterventionSite(key, config.hidden_size, vllm_config)
        for key in visit_layer_keys(config)
    )
    share_steering_index_across_layers(list(sites))
    return sites


def hook_state(
    site: nn.Module, state: torch.Tensor, hook_point: SteeringHookPoint
) -> None:
    """Run the intervention stack on a materialized residual state, in
    place (`pre_attn`, `post_block`)."""
    new = apply_layer_steering(site, state, hook_point)
    if new is not state:
        state.copy_(new)


def hook_branch(
    site: nn.Module,
    hook_point: SteeringHookPoint,
    branch: torch.Tensor,
    base: torch.Tensor | None,
) -> torch.Tensor:
    """Run the intervention stack on the state `base + branch` (`branch`
    alone without `base`) and return the branch that yields the new state.

    The edit is written back as a delta (`branch + (new - state)`), so a
    hook that leaves the state unchanged (no buffers, no active row, a zero
    vector) leaves `branch` bit-identical and the computation downstream of
    the fused `(base + branch) + ...` updates exactly as without it."""
    if base is None:
        return apply_layer_steering(site, branch, hook_point)
    state = base + branch
    new = apply_layer_steering(site, state, hook_point)
    if new is state:
        return branch
    return branch + (new - state)
