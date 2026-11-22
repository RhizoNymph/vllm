# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request MLP neuron clamp: buffers, custom op and model hook.

The clamp overrides individual coordinates of the SiLU-gated MLP
intermediate activation (the input of ``down_proj``) per token, selected
by the same shared per-token ``steering_index`` that drives
``apply_steering``. Each clamp-capable MLP owns three tables of shape
``(steering_rows, max_clamps_per_layer)``:

- ``neuron_clamp_index`` (int32): TP-local neuron index, ``-1`` = empty slot
- ``neuron_clamp_mul`` / ``neuron_clamp_add`` (fp32): the clamp as an
  affine map ``x*mul + add`` (``set`` is ``mul=0, add=value`` and is
  written exactly, ``add`` is ``(1, value)``, ``scale`` is ``(value, 0)``)

plus a single-element bool ``neuron_clamp_any_active`` flag. Row 0 (and the
global rows 1/2) never carry clamps. The op is registered as
``torch.ops.vllm.apply_neuron_clamp`` so ``torch.compile`` treats it as
opaque and CUDA-graph replay reads the live tables; it mutates the
activation in place (a fresh intermediate, so no aliasing hazard) and is a
no-op launch when the flag is clear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig

NEURON_CLAMP_INDEX_ATTR = "neuron_clamp_index"
NEURON_CLAMP_MUL_ATTR = "neuron_clamp_mul"
NEURON_CLAMP_ADD_ATTR = "neuron_clamp_add"
NEURON_CLAMP_ANY_ACTIVE_ATTR = "neuron_clamp_any_active"
# Plain (non-buffer) int attributes describing the TP shard.
NEURON_CLAMP_TP_RANK_ATTR = "neuron_clamp_tp_rank"
NEURON_CLAMP_LOCAL_SIZE_ATTR = "neuron_clamp_local_size"


def register_neuron_clamp_buffers(
    mlp: nn.Module,
    *,
    steering_index: torch.Tensor | None,
    max_steering_configs: int,
    max_clamps_per_layer: int,
    intermediate_size_local: int,
    tp_rank: int,
) -> None:
    """Attach neuron-clamp tables to a clamp-capable MLP module.

    No-op (no buffers ⇒ the MLP forward emits no clamp op) when steering is
    disabled (``max_steering_configs == 0`` / no ``steering_index``) or
    clamps are disabled (``max_clamps_per_layer == 0``).

    ``steering_index`` must be the decoder layer's ``steering_index``
    buffer; it is registered on the MLP too so the op can read it.
    :func:`vllm.model_executor.layers.steering.\\
share_steering_index_across_layers` re-points it at the shared tensor.
    """
    if max_steering_configs == 0 or max_clamps_per_layer == 0:
        return
    if steering_index is None:
        return
    rows = max_steering_configs + 3
    mlp.register_buffer(
        NEURON_CLAMP_INDEX_ATTR,
        torch.full((rows, max_clamps_per_layer), -1, dtype=torch.int32),
        persistent=False,
    )
    mlp.register_buffer(
        NEURON_CLAMP_MUL_ATTR,
        torch.ones(rows, max_clamps_per_layer, dtype=torch.float32),
        persistent=False,
    )
    mlp.register_buffer(
        NEURON_CLAMP_ADD_ATTR,
        torch.zeros(rows, max_clamps_per_layer, dtype=torch.float32),
        persistent=False,
    )
    mlp.register_buffer(
        NEURON_CLAMP_ANY_ACTIVE_ATTR,
        torch.zeros(1, dtype=torch.bool),
        persistent=False,
    )
    mlp.register_buffer("steering_index", steering_index, persistent=False)
    setattr(mlp, NEURON_CLAMP_TP_RANK_ATTR, int(tp_rank))
    setattr(mlp, NEURON_CLAMP_LOCAL_SIZE_ATTR, int(intermediate_size_local))


def get_neuron_clamp_buffer_config(vllm_config: VllmConfig) -> int:
    """Return ``max_clamps_per_layer`` (``0`` ⇒ clamps disabled)."""
    steering_config = getattr(vllm_config, "steering_config", None)
    if steering_config is None:
        return 0
    return int(getattr(steering_config, "max_neuron_clamps_per_layer", 0))


def register_decoder_mlp_neuron_clamp(
    layer: nn.Module,
    *,
    max_steering_configs: int,
    max_clamps_per_layer: int,
) -> None:
    """Wire clamp tables onto ``layer.mlp`` (gate_up -> act -> down_proj).

    The TP shard geometry is read from the row-parallel ``down_proj``:
    its ``input_size_per_partition`` is the local intermediate width and
    ``tp_rank`` its shard index. Must run after ``register_steering_buffers``
    (needs ``layer.steering_index``).
    """
    if max_steering_configs == 0 or max_clamps_per_layer == 0:
        return
    mlp = layer.mlp
    down_proj = mlp.down_proj
    register_neuron_clamp_buffers(
        mlp,
        steering_index=getattr(layer, "steering_index", None),
        max_steering_configs=max_steering_configs,
        max_clamps_per_layer=max_clamps_per_layer,
        intermediate_size_local=int(down_proj.input_size_per_partition),
        tp_rank=int(getattr(down_proj, "tp_rank", 0)),
    )


def get_neuron_clamp_module(layer: nn.Module) -> nn.Module | None:
    """Return the clamp-capable MLP of a decoder layer, if any."""
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, NEURON_CLAMP_INDEX_ATTR):
        return mlp
    return None


def apply_mlp_neuron_clamp(mlp: nn.Module, act: torch.Tensor) -> torch.Tensor:
    """Apply this layer's neuron clamps to the SiLU-gated activation.

    Static ``hasattr`` gate (decided at ``__init__``) so the disabled path
    traces to nothing under ``torch.compile``.
    """
    if not hasattr(mlp, NEURON_CLAMP_INDEX_ATTR):
        return act
    torch.ops.vllm.apply_neuron_clamp(
        act,
        getattr(mlp, NEURON_CLAMP_INDEX_ATTR),
        getattr(mlp, NEURON_CLAMP_MUL_ATTR),
        getattr(mlp, NEURON_CLAMP_ADD_ATTR),
        mlp.steering_index,
        getattr(mlp, NEURON_CLAMP_ANY_ACTIVE_ATTR),
    )
    return act


def neuron_clamp_reference(
    act: torch.Tensor,
    clamp_index: torch.Tensor,
    clamp_mul: torch.Tensor,
    clamp_add: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: return a clamped copy of ``act`` (never in place)."""
    out = act.clone()
    _neuron_clamp_eager_(
        out, clamp_index, clamp_mul, clamp_add, steering_index, any_active
    )
    return out


def _neuron_clamp_eager_(
    act: torch.Tensor,
    clamp_index: torch.Tensor,
    clamp_mul: torch.Tensor,
    clamp_add: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> None:
    if not bool(any_active.item()):
        return
    n = act.shape[0]
    if n == 0:
        return
    rows = steering_index[:n].to(torch.long)
    idx = clamp_index[rows].to(torch.long)  # (n, K)
    valid = idx >= 0
    if not bool(valid.any()):
        return
    tok = torch.arange(n, device=act.device).unsqueeze(1).expand_as(idx)[valid]
    nid = idx[valid]
    mul = clamp_mul[rows][valid]
    add = clamp_add[rows][valid]
    cur = act[tok, nid].to(torch.float32)
    new = torch.where(mul == 0.0, add, cur * mul + add)
    act[tok, nid] = new.to(act.dtype)


def apply_neuron_clamp(
    act: torch.Tensor,
    clamp_index: torch.Tensor,
    clamp_mul: torch.Tensor,
    clamp_add: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> None:
    """Clamp ``act[t, clamp_index[r, k]]`` in place for ``r = index[t]``.

    CUDA: Triton kernel, one program per token, early-exit on the device
    flag (graph-capturable, no host sync). CPU: eager reference.
    """
    if act.is_cuda:
        from vllm.model_executor.layers.neuron_clamp_kernel import (
            apply_neuron_clamp_triton,
        )

        apply_neuron_clamp_triton(
            act, clamp_index, clamp_mul, clamp_add, steering_index, any_active
        )
        return
    _neuron_clamp_eager_(
        act, clamp_index, clamp_mul, clamp_add, steering_index, any_active
    )


def apply_neuron_clamp_fake(
    act: torch.Tensor,
    clamp_index: torch.Tensor,
    clamp_mul: torch.Tensor,
    clamp_add: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
) -> None:
    """FX-tracing fake — declares the in-place mutation, computes nothing."""
    return None


direct_register_custom_op(
    op_name="apply_neuron_clamp",
    op_func=apply_neuron_clamp,
    fake_impl=apply_neuron_clamp_fake,
    mutates_args=["act"],
)
