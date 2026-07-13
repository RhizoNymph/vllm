# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Directional-clamp custom ops and buffer registration.

Clamping constrains the hidden state's scalar projection along up to K
unit directions per steering row to per-direction bounds::

    p     = h @ v_hat                       # fp32 accumulate
    delta = strength * (clip(p, lo, hi) - p)
    h'    = h + delta * v_hat               # h_perp untouched

``lo == hi`` pins the feature (constant expression), ``lo == hi == 0`` is
directional ablation, one-sided bounds suppress only over/under-expression,
and ``strength < 1`` applies a partial correction.  Unlike additive steering
the delta depends on ``h`` itself — a token already inside the bounds is not
touched at all.

Clamps ride the steering row machinery: the per-layer buffers are sized to
the steering table rows (row 0 = no-steering sentinel with all-zero dirs,
rows 1/2 = global prefill/decode, rows 3+ = per-request) and the kernel
gathers with the SAME shared ``steering_index`` token->row buffer, so no new
per-token bookkeeping exists.  Buffer registration piggybacks on
:func:`vllm.model_executor.layers.steering.register_steering_buffers`
(clamp rows ARE steering rows, so unlike patch they are gated on steering
being enabled) with the K knob resolved from the current ``VllmConfig``
(``steering_config.max_clamp_directions``) — zero model-file edits.

The intervention order at each hook is::

    capture (pristine) -> patch (replace/lerp) -> steer (add) -> clamp

Clamp runs LAST: it is a constraint on whatever leaves the site, so an
additive vector cannot push the projection back out of bounds.

The in-graph monitor modulates clamps through the SHARED
``steering_row_gate`` buffer: with ``gate_active`` set, the per-row clamp
strength is scaled by ``row_gate[token]`` — the same per-token gate the
additive steering row term reads (row-level, all K entries of a token's row
scaled uniformly). This only works when the gate is *materialized* into
``steering_row_gate`` by the GLOBAL cross-layer monitor
(``enable_cross_layer_monitor=True``); in the default fused mode the gate is
recomputed inside the steering kernel and never written to the shared
buffer, so clamps run ungated. Per-request declarative clamp gates are
rejected at admission in every mode (see
``vllm.v1.steering_schema.ClampApply``) — the per-row monitor is fused and
cannot materialize gate values for this op to read.

Two ops are registered so ``torch.compile`` treats them as opaque,
graph-safe split points (``mutates_args=[]``, fresh output):
:func:`apply_clamp` for the single-tensor hooks and
:func:`apply_clamp_block` for ``post_block`` — vLLM defers the MLP-branch
add, so the true block output is ``residual + hidden_states`` and clamping
(like replace) does not commute through the deferred add.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

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
CLAMP_DIRS_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"steering_clamp_dirs_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
CLAMP_BOUNDS_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"steering_clamp_bounds_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
CLAMP_STRENGTH_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"steering_clamp_strength_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
CLAMP_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"steering_clamp_any_active_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}
# Per-hook clamp-gate activity flag. When set, the clamp op folds the shared
# per-token ``steering_row_gate`` into the per-row clamp strength (effective
# strength = strength * row_gate[token]), exactly the gate the additive
# steering row term reads. Default False ⇒ clamps ignore the gate entirely
# (the ungated behaviour, bit-for-bit). The runner sets it True only when the
# gate is *materialized* into ``steering_row_gate`` (cross-layer monitor mode),
# so a fused-mode gate — which the steering kernel recomputes in registers and
# never writes to the shared buffer — never silently under-gates clamps.
CLAMP_GATE_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"steering_clamp_gate_active_{hp.value}" for hp in HOOK_POINT_TABLE_ATTR
}

# Process-global direction count, consulted only when no VllmConfig context
# exists (unit tests constructing layers directly) — mirrors the patch-slot
# global in :mod:`vllm.model_executor.layers.patch`.  ``0`` => clamping
# disabled, no buffers attached, and the folded apply path constant-folds out.
_CLAMP_MAX_DIRECTIONS: int = 0


def set_clamp_buffer_directions(max_directions: int) -> None:
    """TEST-ONLY override of the clamp direction count (0 disables)."""
    global _CLAMP_MAX_DIRECTIONS
    _CLAMP_MAX_DIRECTIONS = int(max_directions)


def get_clamp_directions_config(vllm_config: VllmConfig) -> int:
    """Return ``max_clamp_directions`` (K), or 0 when steering is disabled."""
    steering_config = getattr(vllm_config, "steering_config", None)
    if steering_config is None:
        return 0
    return int(getattr(steering_config, "max_clamp_directions", 0))


def _resolve_clamp_directions() -> int:
    """Resolve K at buffer-registration time.

    Primary source is the *current* ``VllmConfig`` (models are always built
    under ``set_current_vllm_config``); the process global is consulted only
    when no config context exists — unit tests constructing layers directly.
    Mirrors ``_resolve_patch_buffer_slots``.
    """
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        return get_clamp_directions_config(vllm_config)
    return _CLAMP_MAX_DIRECTIONS


def register_clamp_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    num_rows: int,
    max_directions: int,
    dtype: torch.dtype | None = None,
) -> None:
    """Attach per-hook clamp buffers to a decoder layer.

    ``num_rows`` must equal the steering table row count for this layer
    (``max_steering_configs + 3`` as seen by ``register_steering_buffers``,
    which already includes the dynamic pool) — the clamp kernel gathers
    with the shared ``steering_index``, so the row spaces must be
    congruent.

    Buffer defaults are all no-ops: zero dirs (a zero direction
    contributes ``delta * 0 == 0`` regardless of bounds), ``[-inf, +inf]``
    bounds, strength 1.0, inactive flag.  ``dirs`` rows are in the model
    compute dtype so the gather needs no cast; ``bounds``/``strength``
    are fp32 (small, precision-bearing).

    Memory: ``num_rows x K x hidden`` per hook per layer in the model
    dtype — e.g. 39 rows x K=4 x 2560 x 2 B ~= 0.76 MB per (hook, layer),
    ~78 MB total on a gemma-3-4b-class model at defaults.
    """
    if max_directions <= 0:
        return
    table_dtype = dtype if dtype is not None else torch.float32
    for hp in HOOK_POINT_TABLE_ATTR:
        module.register_buffer(
            CLAMP_DIRS_ATTR[hp],
            torch.zeros(num_rows, max_directions, hidden_size, dtype=table_dtype),
            persistent=False,
        )
        bounds = torch.empty(num_rows, max_directions, 2, dtype=torch.float32)
        bounds[..., 0] = -float("inf")
        bounds[..., 1] = float("inf")
        module.register_buffer(CLAMP_BOUNDS_ATTR[hp], bounds, persistent=False)
        module.register_buffer(
            CLAMP_STRENGTH_ATTR[hp],
            torch.ones(num_rows, max_directions, dtype=torch.float32),
            persistent=False,
        )
        module.register_buffer(
            CLAMP_ANY_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )
        # Clamp-gate activity flag (default False ⇒ ungated). See
        # ``CLAMP_GATE_ACTIVE_ATTR``. A tensor (not a Python bool) so the
        # compiled graph topology stays stable across steps that do/don't
        # gate — only the flag's data changes between forward passes.
        module.register_buffer(
            CLAMP_GATE_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )


def maybe_register_clamp_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    num_rows: int,
    dtype: torch.dtype | None = None,
) -> None:
    """Register clamp buffers iff ``max_clamp_directions > 0``.

    Called from ``register_steering_buffers`` after its steering-disabled
    early return, so clamp buffers attach to exactly the (PP-local)
    decoder layers that carry steering tables.
    """
    max_directions = _resolve_clamp_directions()
    if max_directions <= 0:
        return
    register_clamp_buffers(
        module,
        hidden_size,
        num_rows=num_rows,
        max_directions=max_directions,
        dtype=dtype,
    )


class ClampOpArgs(NamedTuple):
    """Canonical positional order of the ``apply_clamp`` op's tensors.

    Single source of truth for the op's positional order — emission,
    warmup, and tests build calls by splatting it; a schema-lock test
    asserts ``_fields`` matches the registered op's argument names.
    """

    hidden_states: torch.Tensor
    clamp_dirs: torch.Tensor
    clamp_bounds: torch.Tensor
    clamp_strength: torch.Tensor
    steering_index: torch.Tensor
    any_active: torch.Tensor
    steering_row_gate: torch.Tensor
    gate_active: torch.Tensor


class ClampBlockOpArgs(NamedTuple):
    """Canonical positional order of the ``apply_clamp_block`` op's tensors."""

    hidden_states: torch.Tensor
    residual: torch.Tensor
    clamp_dirs: torch.Tensor
    clamp_bounds: torch.Tensor
    clamp_strength: torch.Tensor
    steering_index: torch.Tensor
    any_active: torch.Tensor
    steering_row_gate: torch.Tensor
    gate_active: torch.Tensor


def maybe_apply_clamp(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Apply single-tensor clamping at ``hook_point`` if buffers are present.

    Short-circuits (returns ``hidden_states`` unchanged) when the layer has
    no clamp buffers — the ``hasattr`` check is constant for the layer's
    lifetime, so ``torch.compile`` traces the disabled path with no clamp op.
    """
    dirs_attr = CLAMP_DIRS_ATTR[hook_point]
    if not hasattr(module, dirs_attr):
        return hidden_states
    return torch.ops.vllm.apply_clamp(
        *ClampOpArgs(
            hidden_states=hidden_states,
            clamp_dirs=getattr(module, dirs_attr),
            clamp_bounds=getattr(module, CLAMP_BOUNDS_ATTR[hook_point]),
            clamp_strength=getattr(module, CLAMP_STRENGTH_ATTR[hook_point]),
            steering_index=module.steering_index,
            any_active=getattr(module, CLAMP_ANY_ACTIVE_ATTR[hook_point]),
            steering_row_gate=module.steering_row_gate,
            gate_active=getattr(module, CLAMP_GATE_ACTIVE_ATTR[hook_point]),
        )
    )


def maybe_apply_clamp_block(
    module: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Apply two-tensor ``post_block`` clamping; returns clamped ``residual``.

    See :func:`apply_clamp_block` for the deferred-MLP-add reconstruction.
    """
    hp = SteeringHookPoint.POST_BLOCK
    dirs_attr = CLAMP_DIRS_ATTR[hp]
    if not hasattr(module, dirs_attr):
        return residual
    return torch.ops.vllm.apply_clamp_block(
        *ClampBlockOpArgs(
            hidden_states=hidden_states,
            residual=residual,
            clamp_dirs=getattr(module, dirs_attr),
            clamp_bounds=getattr(module, CLAMP_BOUNDS_ATTR[hp]),
            clamp_strength=getattr(module, CLAMP_STRENGTH_ATTR[hp]),
            steering_index=module.steering_index,
            any_active=getattr(module, CLAMP_ANY_ACTIVE_ATTR[hp]),
            steering_row_gate=module.steering_row_gate,
            gate_active=getattr(module, CLAMP_GATE_ACTIVE_ATTR[hp]),
        )
    )


def _clamp_correction(
    projected_onto: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    rows: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shared eager correction term: ``sum_k delta_k * v_hat_k`` in fp32.

    ``projected_onto`` is the tensor whose projection is constrained (the
    hidden state for single-tensor hooks, the reconstructed block output
    for ``post_block``).  Returns the fp32 ``(N, H)`` correction; callers
    cast once to the output dtype and add — the frozen cast-order contract
    the Triton kernels mirror.

    ``gate`` is an optional per-token ``(N,)`` fp32 multiplier folded into
    the per-row strength (``strength * gate[token]``) — the shared
    ``steering_row_gate``, gating all K entries of a token's row uniformly,
    congruent with the additive steering row term. ``None`` ⇒ ungated. A
    gate of 0 yields ``delta == 0`` exactly (no correction for that token).
    """
    dirs = clamp_dirs[rows].to(torch.float32)  # (N, K, H)
    bounds = clamp_bounds[rows]  # (N, K, 2)
    strength = clamp_strength[rows]  # (N, K)
    if gate is not None:
        # Row-level gate: fold the per-token scalar into every K entry.
        strength = strength * gate.to(torch.float32).unsqueeze(-1)
    x32 = projected_onto.to(torch.float32)
    p = (x32.unsqueeze(1) * dirs).sum(dim=-1)  # (N, K)
    p_clamped = torch.clamp(p, min=bounds[..., 0], max=bounds[..., 1])
    delta = strength * (p_clamped - p)  # (N, K); zero dirs -> p == 0 but
    # the correction below multiplies by the zero direction, so unused
    # (zero-padded) K slots contribute nothing regardless of their bounds.
    return (delta.unsqueeze(-1) * dirs).sum(dim=1)  # (N, H) fp32


def apply_clamp(
    hidden_states: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_row_gate: torch.Tensor | None = None,
    gate_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directional projection clamp via the shared steering row gather.

    ``clamp_dirs`` is ``(rows, K, hidden)`` in the model compute dtype with
    row 0 (the no-steering sentinel) all-zero forever; ``clamp_bounds`` is
    ``(rows, K, 2)`` fp32 ``[lo, hi]`` (defaults ``[-inf, +inf]``);
    ``clamp_strength`` is ``(rows, K)`` fp32 (default 1.0).
    ``steering_index`` is the SHARED steering token->row buffer
    (``(max_tokens,)`` int64); only the first ``N`` entries are read.
    ``any_active`` is a single-element bool; ``False`` skips the gather and
    emits a copy.  Output is always a fresh tensor (graph value semantics).

    ``steering_row_gate`` is the SHARED per-token row gate (fp32,
    ``(max_tokens,)``, default 1.0) — the SAME buffer the additive steering
    row term reads. ``gate_active`` is a single-element bool: when set the
    per-row clamp strength is scaled by ``row_gate[token]`` (effective
    strength = ``strength * gate``), so the gate multiplies all K entries of
    a token's row uniformly (row-level gating). When ``False`` the gate is
    ignored (bit-for-bit the ungated behaviour). A gate of 0 yields a delta
    of exactly 0 for that token, so a fully-closed gate leaves the token
    untouched — the same in-bounds-token-untouched invariant.

    Projections accumulate in fp32; the summed correction is cast to the
    hidden dtype exactly once before the add.  This eager path is the
    frozen reference the Triton kernel matches.

    ``steering_row_gate`` / ``gate_active`` default to ``None`` (ungated) so a
    direct eager call may omit them; the in-graph emission sites always pass
    the real shared buffer + flag (the always-present, cudagraph-safe form).
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.clamp_kernel import apply_clamp_triton

        return apply_clamp_triton(
            hidden_states,
            clamp_dirs,
            clamp_bounds,
            clamp_strength,
            steering_index,
            any_active,
            steering_row_gate,
            gate_active,
        )
    if not bool(any_active.item()):
        return hidden_states.clone()
    n = hidden_states.shape[0]
    rows = steering_index[:n]
    gate = (
        steering_row_gate[:n]
        if steering_row_gate is not None
        and gate_active is not None
        and bool(gate_active.item())
        else None
    )
    corr = _clamp_correction(
        hidden_states, clamp_dirs, clamp_bounds, clamp_strength, rows, gate
    )
    return hidden_states + corr.to(hidden_states.dtype)


def apply_clamp_fake(
    hidden_states: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_row_gate: torch.Tensor | None = None,
    gate_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


def apply_clamp_block(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_row_gate: torch.Tensor | None = None,
    gate_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-tensor ``post_block`` clamp; returns a fresh clamped ``residual``.

    vLLM defers the MLP-branch add, so the true block output is
    ``residual + hidden_states``.  The projection constraint must hold on
    the block output, and clamping does not commute through the deferred
    add, so this op reconstructs the block output, measures/clips its
    projection, and folds the correction back into ``residual`` (leaving
    ``hidden_states`` untouched)::

        out_res = residual + strength * (clip(b @ v, lo, hi) - b @ v) * v
        # b = residual + hidden; out_res + hidden == clamp(b)

    ``steering_row_gate`` / ``gate_active`` gate the per-row strength exactly
    as in :func:`apply_clamp` (row-level, shared with additive steering).
    """
    if residual.is_cuda:
        from vllm.model_executor.layers.clamp_kernel import (
            apply_clamp_block_triton,
        )

        return apply_clamp_block_triton(
            hidden_states,
            residual,
            clamp_dirs,
            clamp_bounds,
            clamp_strength,
            steering_index,
            any_active,
            steering_row_gate,
            gate_active,
        )
    if not bool(any_active.item()):
        return residual.clone()
    n = residual.shape[0]
    rows = steering_index[:n]
    gate = (
        steering_row_gate[:n]
        if steering_row_gate is not None
        and gate_active is not None
        and bool(gate_active.item())
        else None
    )
    block_out = residual.to(torch.float32) + hidden_states.to(torch.float32)
    corr = _clamp_correction(
        block_out, clamp_dirs, clamp_bounds, clamp_strength, rows, gate
    )
    return residual + corr.to(residual.dtype)


def apply_clamp_block_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    clamp_dirs: torch.Tensor,
    clamp_bounds: torch.Tensor,
    clamp_strength: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_row_gate: torch.Tensor | None = None,
    gate_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="apply_clamp",
    op_func=apply_clamp,
    fake_impl=apply_clamp_fake,
    mutates_args=[],
)

direct_register_custom_op(
    op_name="apply_clamp_block",
    op_func=apply_clamp_block,
    fake_impl=apply_clamp_block_fake,
    mutates_args=[],
)
