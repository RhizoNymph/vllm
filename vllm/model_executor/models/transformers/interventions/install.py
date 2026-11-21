# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach intervention buffers and hooked forwards to decoder layers."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
    share_steering_index_across_layers,
)
from vllm.model_executor.models.transformers.interventions.analysis import (
    analyze_layer,
    classify_children,
    is_vllm_attention,
)
from vllm.model_executor.models.transformers.interventions.errors import (
    UnsupportedInterventionLayout,
)
from vllm.model_executor.models.transformers.interventions.rewrite import (
    rewrite_forward,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_INTERVENTION_CONFIGS = ("steering_config", "patch_config", "capture_consumers_config")


def interventions_requested(vllm_config: "VllmConfig") -> bool:
    """Whether steering, patching, or activation capture is configured.

    Decoder layers are only rewritten when one is, so servers without any keep
    the upstream forward bit-for-bit."""
    return any(
        getattr(vllm_config, name, None) is not None for name in _INTERVENTION_CONFIGS
    )


@dataclass(frozen=True)
class BufferSizes:
    """Sizes of the per-layer intervention buffers (see
    `register_steering_buffers`). `max_configs == 0` registers no steering
    tables, leaving capture and patching only."""

    max_tokens: int
    max_configs: int
    dtype: torch.dtype | None

    @classmethod
    def from_vllm_config(cls, vllm_config: "VllmConfig") -> "BufferSizes":
        max_tokens, max_configs = get_steering_buffer_config(vllm_config)
        return cls(max_tokens, max_configs, get_steering_buffer_dtype(vllm_config))


_HOOKED_FORWARDS: dict[tuple, Callable] = {}
"""Rewritten forward per layer class and child structure."""


def hooked_forward(
    layer: nn.Module, is_attention: Callable[[nn.Module], bool] = is_vllm_attention
) -> Callable:
    """The forward of `layer`'s class with every hook site wired (cached).

    The key includes the child structure, because optional children (e.g.
    StableLM's `post_attention_layernorm`) select which paths are feasible."""
    attention, heavy = classify_children(layer, is_attention)
    key = (type(layer), tuple(n for n, _ in layer.named_children()), attention, heavy)
    if (forward := _HOOKED_FORWARDS.get(key)) is None:
        forward = rewrite_forward(analyze_layer(layer, is_attention))
        _HOOKED_FORWARDS[key] = forward
    return forward


def install_interventions(
    layers: Sequence[nn.Module],
    indices: Iterable[int],
    *,
    hidden_size: int,
    sizes: BufferSizes,
    is_attention: Callable[[nn.Module], bool] = is_vllm_attention,
) -> None:
    """Make `layers[i]` for each `i` in `indices` an intervention site.

    Each layer gets `layer_idx = i`, the steering/patch/clamp buffers native
    decoder layers register, and a forward with the native hook points wired
    in. `indices` should be the decoder layers local to this pipeline rank.
    Every layer is analysed before any is modified, so an unsupported layer
    leaves the model untouched."""
    indices = list(indices)
    forwards = {i: hooked_forward(layers[i], is_attention) for i in indices}
    for i in indices:
        existing = getattr(layers[i], "layer_idx", None)
        if existing is not None and existing != i:
            raise UnsupportedInterventionLayout(
                type(layers[i]).__name__,
                f"the layer at index {i} already has layer_idx={existing}",
            )
    for i in indices:
        layer = layers[i]
        layer.layer_idx = i
        register_steering_buffers(
            layer,
            hidden_size,
            max_steering_tokens=sizes.max_tokens,
            max_steering_configs=sizes.max_configs,
            dtype=sizes.dtype,
        )
        layer.forward = MethodType(forwards[i], layer)
    share_steering_index_across_layers([layers[i] for i in indices])
    if indices:
        logger.info(
            "Hooked Transformers decoder layers for interventions: "
            "layer_classes=%s first_layer=%d last_layer=%d",
            sorted({type(layers[i]).__name__ for i in indices}),
            indices[0],
            indices[-1],
        )
