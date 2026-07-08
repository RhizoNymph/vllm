# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation-patching custom ops and buffer registration.

Activation patching overwrites (``alpha == 1``) or interpolates toward
(``0 < alpha < 1``) the residual-stream activation at selected token rows with
a source vector captured from a prior "clean" run. It reuses the steering hook
points (``pre_attn``, ``post_attn``, ``post_block``) and folds into the same
``apply_layer_steering`` / ``apply_block_steering`` call sites, so no model file
changes. The intervention order at each hook is::

    capture (pristine) -> patch (replace/lerp) -> steer (add)

Two ops are registered so ``torch.compile`` treats them as opaque, graph-safe
split points (``mutates_args=[]``, fresh output): :func:`apply_patch` for the
single-tensor hooks and :func:`apply_patch_block` for ``post_block`` (see
:mod:`vllm.model_executor.layers.patch_kernel` for why the latter needs both
``hidden_states`` and ``residual``).

Per-layer buffers mirror steering's, with two differences: the index is
**per-(layer, hook)** (a request patches different positions at different
layers, so a shared index is unrepresentable), and the table holds one row per
*patched position in a step* (slot 0 = passthrough sentinel) rather than one
row per config. Buffer registration piggybacks on
:func:`register_steering_buffers` via :func:`maybe_register_patch_buffers`,
gated by a process-global slot count the runner sets before model load — so
patch buffers land on exactly the decoder layers (PP-local) that register
steering buffers, with zero model-file edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Buffer attribute names on decoder layer modules, keyed by hook point.
PATCH_TABLE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"patch_table_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
PATCH_ALPHA_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"patch_alpha_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
PATCH_INDEX_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"patch_index_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
PATCH_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"patch_any_active_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}

DEFAULT_MAX_PATCH_SLOTS = 64

# Process-global patch slot count, set by the runner (per worker process)
# before model construction so :func:`register_steering_buffers` can attach
# patch buffers to every decoder layer it sees.  ``0`` => patching disabled,
# no buffers attached, and the folded apply path constant-folds out.
_PATCH_MAX_SLOTS: int = 0


def set_patch_buffer_slots(max_patch_slots: int) -> None:
    """TEST-ONLY override of the patch slot count (0 disables patching).

    At runtime the slot count is resolved from the current ``VllmConfig``
    (see :func:`_resolve_patch_buffer_slots`); this global is consulted only
    when no config context exists — unit tests constructing layers directly.
    """
    global _PATCH_MAX_SLOTS
    _PATCH_MAX_SLOTS = int(max_patch_slots)


def get_patch_buffer_slots() -> int:
    """Return the process-global patch slot count."""
    return _PATCH_MAX_SLOTS


def get_patch_buffer_config(vllm_config: VllmConfig) -> int:
    """Return ``max_patch_slots`` for patch buffers, or 0 when disabled.

    Reads ``vllm_config.patch_config.max_patch_slots`` when present; the full
    ``PatchConfig`` is wired in a later phase, so this is getattr-tolerant.
    """
    patch_config = getattr(vllm_config, "patch_config", None)
    if patch_config is None:
        return 0
    return int(getattr(patch_config, "max_patch_slots", 0))


def get_patch_source_cache_bytes(vllm_config: VllmConfig) -> int:
    """Return the patch source-store byte budget, or 0 when disabled."""
    patch_config = getattr(vllm_config, "patch_config", None)
    if patch_config is None:
        return 0
    return int(getattr(patch_config, "patch_source_cache_bytes", 0))


def register_patch_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    max_patch_tokens: int,
    max_patch_slots: int,
    dtype: torch.dtype | None = None,
) -> None:
    """Attach per-hook patch buffers to a decoder layer.

    No-op when ``max_patch_slots <= 0`` (patching disabled), matching the
    disabled-mode discipline of :func:`register_steering_buffers`: with no
    buffers attached, :func:`maybe_apply_patch` short-circuits on a ``hasattr``
    check that ``torch.compile`` traces as a constant branch.

    ``patch_table`` rows are in the model compute dtype (``dtype``) so the
    gather needs no cast. Slot 0 is the passthrough sentinel and ``alpha`` row 0
    stays all-zeros for its whole lifetime (the passthrough invariant the CPU
    fallback relies on). The index is per-(layer, hook) and int32.

    ``patch_alpha`` is a per-**dimension** weight: ``(max_slots, hidden_size)``
    fp32. A per-dim alpha is exactly ``alpha * mask`` folded into one buffer, so
    masked patches (restrict to a subset of dims) and graded masks need no
    separate kernel path — an unmasked entry stages a constant ``alpha`` row.
    """
    if max_patch_slots <= 0:
        return
    table_dtype = dtype if dtype is not None else torch.float32
    for hp in HOOK_POINT_TABLE_ATTR:
        module.register_buffer(
            PATCH_TABLE_ATTR[hp],
            torch.zeros(max_patch_slots, hidden_size, dtype=table_dtype),
            persistent=False,
        )
        # alpha row 0 == 0 (passthrough invariant); zeros() satisfies it.
        module.register_buffer(
            PATCH_ALPHA_ATTR[hp],
            torch.zeros(max_patch_slots, hidden_size, dtype=torch.float32),
            persistent=False,
        )
        module.register_buffer(
            PATCH_INDEX_ATTR[hp],
            torch.zeros(max_patch_tokens, dtype=torch.int32),
            persistent=False,
        )
        module.register_buffer(
            PATCH_ANY_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )


def _resolve_patch_buffer_slots() -> int:
    """Resolve the patch slot count at buffer-registration time.

    Primary source is the *current* ``VllmConfig`` (models are always built
    under ``set_current_vllm_config``, on every runner), so no runner has to
    remember to set anything before the model build — the v1 runner once
    silently shipped patching as a no-op precisely because it didn't set the
    old process-global. The global (``set_patch_buffer_slots``) is consulted
    only when no config context exists, i.e. unit tests constructing layers
    directly.
    """
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return get_patch_buffer_config(vllm_config)
    return _PATCH_MAX_SLOTS


def maybe_register_patch_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    max_patch_tokens: int,
    dtype: torch.dtype | None = None,
) -> None:
    """Register patch buffers iff patching is enabled in the vllm config.

    Called from :func:`register_steering_buffers` so patch buffers attach to
    the same decoder layers as steering buffers, independent of whether
    steering itself is enabled. ``max_patch_tokens`` mirrors the steering index
    length (``max_num_batched_tokens``).
    """
    max_patch_slots = _resolve_patch_buffer_slots()
    if max_patch_slots <= 0:
        return
    register_patch_buffers(
        module,
        hidden_size,
        max_patch_tokens=max_patch_tokens,
        max_patch_slots=max_patch_slots,
        dtype=dtype,
    )


def maybe_apply_patch(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Apply single-tensor patching at ``hook_point`` if buffers are present.

    Short-circuits (returns ``hidden_states`` unchanged) when the layer has no
    patch buffers — the ``hasattr`` check is constant for the layer's lifetime,
    so ``torch.compile`` traces the disabled path with no patch kernel.
    """
    table_attr = PATCH_TABLE_ATTR[hook_point]
    if not hasattr(module, table_attr):
        return hidden_states
    return torch.ops.vllm.apply_patch(
        hidden_states,
        getattr(module, table_attr),
        getattr(module, PATCH_INDEX_ATTR[hook_point]),
        getattr(module, PATCH_ALPHA_ATTR[hook_point]),
        getattr(module, PATCH_ANY_ACTIVE_ATTR[hook_point]),
    )


def maybe_apply_patch_block(
    module: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Apply two-tensor ``post_block`` patching; returns patched ``residual``.

    See :func:`apply_patch_block` for the deferred-MLP-add reconstruction.
    """
    hp = SteeringHookPoint.POST_BLOCK
    table_attr = PATCH_TABLE_ATTR[hp]
    if not hasattr(module, table_attr):
        return residual
    return torch.ops.vllm.apply_patch_block(
        hidden_states,
        residual,
        getattr(module, table_attr),
        getattr(module, PATCH_INDEX_ATTR[hp]),
        getattr(module, PATCH_ALPHA_ATTR[hp]),
        getattr(module, PATCH_ANY_ACTIVE_ATTR[hp]),
    )


def apply_patch(
    hidden_states: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Single-tensor lerp patch: ``lerp(hs, table[idx], alpha[idx])``.

    ``patch_table`` is ``(max_slots, hidden)`` with slot 0 the passthrough
    sentinel. ``patch_index`` is ``(max_tokens,)`` int32 mapping each token row
    to a slot (0 = passthrough); only the first ``N`` entries are read.
    ``patch_alpha`` is ``(max_slots, hidden)`` fp32 (per-dim ``alpha * mask``)
    with row 0 all-zeros. ``any_active`` is a single-element bool; ``False``
    skips the gather and emits a copy. Output is always a fresh tensor (graph
    value semantics).
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.patch_kernel import apply_patch_triton

        return apply_patch_triton(
            hidden_states, patch_table, patch_index, patch_alpha, any_active
        )
    if not bool(any_active.item()):
        return hidden_states.clone()
    n = hidden_states.shape[0]
    slots = patch_index[:n].long()
    # alpha row 0 == 0 makes slot-0 rows passthrough without a branch. Precise
    # lerp form ``(1-a)*h + a*t`` is exact at the endpoints (a==1 -> table).
    alpha = patch_alpha[slots].to(hidden_states.dtype)
    gathered = patch_table[slots].to(hidden_states.dtype)
    return (1 - alpha) * hidden_states + alpha * gathered


def apply_patch_fake(
    hidden_states: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


def apply_patch_block(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """Two-tensor ``post_block`` patch; returns a fresh patched ``residual``.

    vLLM defers the MLP-branch add, so the true block output is
    ``residual + hidden_states``. Replace/lerp does not commute through that
    deferred add, so this op reconstructs the block output, lerps it toward the
    source, and folds the delta back into ``residual`` (leaving
    ``hidden_states`` untouched)::

        out_res = residual + alpha[idx] * (table[idx] - (residual + hidden))
        # block_out = out_res + hidden == lerp(residual + hidden, table, alpha)
    """
    if residual.is_cuda:
        from vllm.model_executor.layers.patch_kernel import apply_patch_block_triton

        return apply_patch_block_triton(
            hidden_states,
            residual,
            patch_table,
            patch_index,
            patch_alpha,
            any_active,
        )
    if not bool(any_active.item()):
        return residual.clone()
    n = residual.shape[0]
    slots = patch_index[:n].long()
    alpha = patch_alpha[slots].to(residual.dtype)
    gathered = patch_table[slots].to(residual.dtype)
    # out_res + hidden == lerp(residual + hidden, table, alpha); written so
    # alpha==0 yields residual exactly (passthrough).
    return (1 - alpha) * residual + alpha * (gathered - hidden_states)


def apply_patch_block_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    patch_table: torch.Tensor,
    patch_index: torch.Tensor,
    patch_alpha: torch.Tensor,
    any_active: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="apply_patch",
    op_func=apply_patch,
    fake_impl=apply_patch_fake,
    mutates_args=[],
)

direct_register_custom_op(
    op_name="apply_patch_block",
    op_func=apply_patch_block,
    fake_impl=apply_patch_block_fake,
    mutates_args=[],
)
