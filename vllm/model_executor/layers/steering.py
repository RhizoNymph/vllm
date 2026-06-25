# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation steering custom op and hook-point definitions.

Registered as ``torch.ops.vllm.apply_steering`` so that torch.compile
treats the operation as an opaque splitting point.  The real Python
implementation executes at runtime between compiled graph segments,
reading the live buffer values rather than baked-in constants.
"""

from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.layers.activation_capture import maybe_capture_residual
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class SteeringHookPoint(str, Enum):
    """Positions in a decoder layer where steering can be applied.

    All hook points operate on the residual skip tensor carried through
    the decoder layer, not on the post-norm sublayer input tensor.
    The names identify approximate regions of the layer where the
    residual skip tensor is steered.
    """

    PRE_ATTN = "pre_attn"
    """Steer the residual skip tensor in the pre-attention region."""

    POST_ATTN = "post_attn"
    """Steer the residual skip tensor in the post-attention region."""

    POST_BLOCK = "post_block"  # block output: residual + mlp branch (true hs[L+1])
    """Steer the block-output residual stream (``residual + mlp_branch``), the
    true ``hidden_states[L + 1]`` -- see :func:`apply_block_steering`."""


# Buffer attribute names on decoder layer modules, keyed by hook point.
HOOK_POINT_TABLE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "steering_table_pre_attn",
    SteeringHookPoint.POST_ATTN: "steering_table_post_attn",
    SteeringHookPoint.POST_BLOCK: "steering_table_post_block",
}

# Per-hook ``any-active`` flag attribute names. The flag is a single-element
# bool tensor co-located with each hook point's table buffer; the apply
# kernel reads it at launch and short-circuits the gather + add when no row
# is currently active for that hook point. The attribute name is derived
# from the table attribute so the two are always discoverable together.
HOOK_POINT_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_any_active" for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}

# Per-hook dedicated dynamic-tier vector attribute names (§5.4). A single
# fp32 ``(hidden,)`` buffer per hook point holding that hook's global
# dynamic-tier vector; the apply kernel adds ``dynamic_vec * token_scales``
# on top of the row gather, gated per token (decode-only). Derived from the
# table attribute so the two are discoverable together.
HOOK_POINT_DYNVEC_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_dynvec" for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}

# Per-hook in-graph monitor buffers (Phase 2, §8). At a probe site one
# ``(layer, hook)`` carries a probe vector ``(hidden,)``, a ``(2,)`` param
# buffer ``[threshold, sharpness]``, and a single-element bool ``active``
# flag. The monitor op reads the residual at this hook and writes a
# per-token gate into the shared ``steering_token_scales`` buffer, which
# the §5.4 dynamic tier then multiplies by. Derived from the table
# attribute so they are discoverable together.
HOOK_POINT_MONITOR_PROBE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_monitor_probe"
    for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}
HOOK_POINT_MONITOR_PARAMS_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_monitor_params"
    for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}
HOOK_POINT_MONITOR_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    hp: f"{table_attr}_monitor_active"
    for hp, table_attr in HOOK_POINT_TABLE_ATTR.items()
}

# Valid hook point string values for validation.
VALID_HOOK_POINT_NAMES: frozenset[str] = frozenset(hp.value for hp in SteeringHookPoint)

DEFAULT_HOOK_POINT = SteeringHookPoint.POST_BLOCK


def register_steering_buffers(
    module: nn.Module,
    hidden_size: int,
    *,
    max_steering_tokens: int,
    max_steering_configs: int,
    dtype: torch.dtype | None = None,
) -> None:
    """Attach per-hook steering buffers to a decoder layer.

    ``dtype`` controls the storage dtype of the steering table buffers.
    When ``None`` (the default), the buffers fall back to fp32 to
    preserve historical behaviour.  Callers in vLLM models pass the
    model's compute dtype (typically bf16) so the indexed gather in
    :func:`apply_steering` returns rows already aligned with the residual
    tensor and no dtype cast is required at the gather site.

    When ``max_steering_configs == 0`` (steering disabled at the engine
    level — ``vllm_config.steering_config is None``), this is a no-op.
    No buffers are attached to ``module``, which causes
    :func:`apply_layer_steering` to short-circuit so the steering
    kernel never launches.  This keeps disabled-mode forwards free of
    steering overhead.
    """
    if max_steering_configs == 0:
        return
    table_dtype = dtype if dtype is not None else torch.float32
    for hp in SteeringHookPoint:
        module.register_buffer(
            HOOK_POINT_TABLE_ATTR[hp],
            torch.zeros(max_steering_configs + 3, hidden_size, dtype=table_dtype),
            persistent=False,
        )
        # Per-hook activity flag.  A single-element bool tensor that the
        # ``apply_steering`` kernel reads at launch and uses to skip the
        # gather + add when no rows are currently active for this hook
        # point.  The flag is a tensor (not a Python bool) so the
        # ``torch.compile`` graph topology stays stable across batches
        # with different active-hook sets — only the data in the tensor
        # changes between forward passes.
        module.register_buffer(
            HOOK_POINT_ANY_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )
        # Per-hook dedicated dynamic-tier vector (§5.4): fp32 (hidden,),
        # default 0 ⇒ no tier. The manager writes it from
        # ``dynamic_tier_vectors`` in populate; the kernel adds
        # ``dynamic_vec * token_scales`` on top of the row gather.
        module.register_buffer(
            HOOK_POINT_DYNVEC_ATTR[hp],
            torch.zeros(hidden_size, dtype=torch.float32),
            persistent=False,
        )
        # Per-hook in-graph monitor buffers (Phase 2, §8). The probe is a
        # fp32 (hidden,) detector vector; params is [threshold, sharpness];
        # active is a bool flag the monitor op reads at launch and uses to
        # short-circuit (a tensor, not a Python bool, so the compiled graph
        # topology stays stable). All default to the inactive/no-op state —
        # ``register_steering_buffers`` registers them on every layer, but
        # the monitor only runs where the manager set a probe and flipped
        # ``active`` (one site). Sharpness default 1.0 keeps the sigmoid
        # finite if ever read while inactive.
        module.register_buffer(
            HOOK_POINT_MONITOR_PROBE_ATTR[hp],
            torch.zeros(hidden_size, dtype=torch.float32),
            persistent=False,
        )
        # [threshold, sharpness, gate_rows]. gate_rows (0/1) ⇒ the monitor
        # also gates the per-request row term (not just the §5.4 tier).
        module.register_buffer(
            HOOK_POINT_MONITOR_PARAMS_ATTR[hp],
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_MONITOR_ACTIVE_ATTR[hp],
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )

    module.register_buffer(
        "steering_index",
        torch.zeros(max_steering_tokens, dtype=torch.long),
        persistent=False,
    )

    # Per-token dynamic-tier gate (§5.4): fp32 (max_tokens,), default 0,
    # shared across layers like ``steering_index`` (shared in
    # ``_init_steering_state``). The runner writes it each step —
    # ``dynamic_tier_gain`` for decode tokens of a tier-active state, 0
    # otherwise (so the tier stays decode-only). Phase 2 replaces the
    # runner write with an in-graph monitor.
    module.register_buffer(
        "steering_token_scales",
        torch.zeros(max_steering_tokens, dtype=torch.float32),
        persistent=False,
    )

    # Per-token row gate (Phase 2 row gating): fp32 (max_tokens,), default
    # 1.0 ⇒ rows apply at full strength. Shared across layers like
    # ``steering_index``. The kernel multiplies the gathered row by
    # ``row_gate[token]``; the runner resets it to 1.0 each step and the
    # in-graph monitor reduces it for decode tokens (prefill stays 1.0, so
    # prefill rows — which feed prefix-cache keys — are never gated). See
    # docs/design/dynamic_steering_row_gating.md.
    module.register_buffer(
        "steering_row_gate",
        torch.ones(max_steering_tokens, dtype=torch.float32),
        persistent=False,
    )

    # Per-token decode mask (Phase 2 row gating): fp32 (max_tokens,),
    # default 0.0. Shared across layers. The runner writes 1.0 for decode
    # tokens / 0.0 for prefill; the in-graph monitor reads it so it only
    # gates DECODE rows, never prefill rows (which feed prefix-cache keys).
    module.register_buffer(
        "steering_decode_mask",
        torch.zeros(max_steering_tokens, dtype=torch.float32),
        persistent=False,
    )

    # Always-False monitor flag. When the cross-layer monitor is enabled
    # (``enable_cross_layer_monitor``), ``apply_layer_steering`` emits the
    # standalone mutating ``steering_monitor`` op and passes THIS flag (never
    # written) as ``apply_steering``'s monitor-active arg, so the same-hook
    # fused gate is bypassed and the per-token gate is applied exactly once via
    # the shared ``token_scales``/``row_gate`` buffers the standalone op writes
    # (avoids double-gating at the probe site). See docs/design/dynamic_steering.md §8.
    module.register_buffer(
        "steering_monitor_off",
        torch.zeros(1, dtype=torch.bool),
        persistent=False,
    )

    # Per-row strength scale (the §5.3 "how much" knob): one fp32 buffer
    # per layer, shared across that layer's hook points (the row index is
    # hook-independent — row 3 = config X for every hook). Default 1.0 ⇒
    # unscaled steering; the kernel multiplies the gathered row by
    # ``scales[row]``. The manager writes the same per-row values into
    # every layer's buffer during populate.
    module.register_buffer(
        "steering_scales",
        torch.ones(max_steering_configs + 3, dtype=torch.float32),
        persistent=False,
    )


def get_steering_buffer_config(vllm_config: "VllmConfig") -> tuple[int, int]:
    """Return ``(max_tokens, max_configs)`` for steering buffers.

    ``max_configs`` is the total row budget above the three reserved
    rows: the scheduler-admitted per-request pool
    (``max_steering_configs``) plus the dynamic-override pool
    (``max_dynamic_steering_configs`` — rows allocated at runtime by
    dynamic steering; see docs/design/dynamic_steering.md §5.2). Every
    model sizes its tables through this single function, so the pool
    split needs no model-file knowledge.
    """
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    steering_config = getattr(vllm_config, "steering_config", None)
    if steering_config is None:
        return max_tokens, 0
    max_configs = steering_config.max_steering_configs + getattr(
        steering_config, "max_dynamic_steering_configs", 0
    )
    return max_tokens, max_configs


def get_steering_buffer_dtype(vllm_config: "VllmConfig") -> torch.dtype:
    """Return the dtype that steering table buffers should be allocated in.

    Mirrors :func:`get_steering_buffer_config`. Returns the resolved
    ``torch.dtype`` that the model was loaded with so steering table
    rows can be gathered into ``hidden_states`` without an extra cast.
    """
    return vllm_config.model_config.dtype


def share_steering_index_across_layers(layers: list[nn.Module]) -> None:
    """Reuse one ``steering_index`` tensor across all steerable layers."""
    shared_index: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_index"):
            continue
        if shared_index is None:
            shared_index = layer.steering_index
            continue
        layer.steering_index = shared_index


def share_steering_token_scales_across_layers(layers) -> None:
    """Reuse one ``steering_token_scales`` tensor across all steerable layers.

    The per-token dynamic-tier gate (§5.4) is layer-independent and
    ``max_tokens``-sized, so it is shared (one per-step H2D, not per
    layer) exactly like ``steering_index``. Called once from the mixin's
    ``_init_steering_state`` (not per model file) since it has every
    steerable layer in hand at that point.
    """
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_token_scales"):
            continue
        if shared is None:
            shared = layer.steering_token_scales
            continue
        layer.steering_token_scales = shared


def share_steering_row_gate_across_layers(layers) -> None:
    """Reuse one ``steering_row_gate`` tensor across all steerable layers.

    Per-token row gate (Phase 2 row gating), ``max_tokens``-sized and
    layer-independent — shared like ``steering_token_scales`` (one per-step
    H2D). Called once from the mixin's ``_init_steering_state``.
    """
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_row_gate"):
            continue
        if shared is None:
            shared = layer.steering_row_gate
            continue
        layer.steering_row_gate = shared


def share_steering_decode_mask_across_layers(layers) -> None:
    """Reuse one ``steering_decode_mask`` tensor across all steerable layers.

    Per-token decode mask for Phase 2 row gating, shared like
    ``steering_row_gate``. Called once from ``_init_steering_state``.
    """
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "steering_decode_mask"):
            continue
        if shared is None:
            shared = layer.steering_decode_mask
            continue
        layer.steering_decode_mask = shared


def _emit_steering_op(
    module: nn.Module,
    x: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Emit the in-graph monitor (when enabled) and the ``apply_steering`` op
    for ``hook_point`` on ``x``.

    Shared by :func:`apply_layer_steering` and :func:`apply_block_steering` so
    the 12-argument op signature and the monitor wiring live in exactly one
    place; the two call sites previously drifted apart.
    """
    monitor_active = getattr(module, HOOK_POINT_MONITOR_ACTIVE_ATTR[hook_point])
    # Cross-layer monitor (Phase 2, §8): when enabled, emit the standalone
    # *mutating* ``steering_monitor`` op at this hook — it reads the
    # pre-steering residual and writes a per-token gate into the SHARED
    # ``steering_token_scales``/``steering_row_gate`` buffers, which every
    # later layer/hook's ``apply_steering`` then reads ("detect at L, gate at
    # layers ≥ L"). The op is a no-op unless the manager activated a probe at
    # this site, so the compiled topology stays stable and unconfigured sites
    # cost nothing (the mutating op is cudagraph-free — measured, see
    # docs/design/dynamic_steering.md §8). The same-hook fused gate is then
    # bypassed (pass the always-False ``steering_monitor_off`` flag) so the
    # gate is applied exactly once via the shared buffers, not twice at L.
    # ``module._cross_layer_monitor`` is stamped once at steering init, so
    # torch.compile traces this as a static branch (stable per process).
    if getattr(module, "_cross_layer_monitor", False):
        torch.ops.vllm.steering_monitor(
            x,
            getattr(module, HOOK_POINT_MONITOR_PROBE_ATTR[hook_point]),
            getattr(module, HOOK_POINT_MONITOR_PARAMS_ATTR[hook_point]),
            monitor_active,
            module.steering_token_scales,
            module.steering_decode_mask,
            module.steering_row_gate,
        )
        fused_active = module.steering_monitor_off
    else:
        # Default: the in-graph monitor is **fused into** ``apply_steering`` —
        # the kernel computes the per-token gate from the pre-steering residual
        # and folds it into the tier/row terms in registers, never writing a
        # shared buffer. Non-mutating (cudagraph-fusable) and same-hook (it
        # gates only this ``(layer, hook)``). Buffers are passed
        # unconditionally; the kernel skips the gate reduction unless the
        # ``active`` flag is set, so the topology stays stable and an
        # unconfigured monitor costs nothing.
        fused_active = monitor_active
    return torch.ops.vllm.apply_steering(
        x,
        getattr(module, HOOK_POINT_TABLE_ATTR[hook_point]),
        module.steering_index,
        getattr(module, HOOK_POINT_ANY_ACTIVE_ATTR[hook_point]),
        module.steering_scales,
        getattr(module, HOOK_POINT_DYNVEC_ATTR[hook_point]),
        module.steering_token_scales,
        module.steering_row_gate,
        getattr(module, HOOK_POINT_MONITOR_PROBE_ATTR[hook_point]),
        getattr(module, HOOK_POINT_MONITOR_PARAMS_ATTR[hook_point]),
        fused_active,
        module.steering_decode_mask,
    )


def apply_layer_steering(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Apply the steering table for ``hook_point`` to ``hidden_states``.

    Capture consumers (when configured) see the pre-steering residual via
    :func:`maybe_capture_residual`.

    When the layer has no steering table buffer registered (engine
    started with ``enable_steering=False``, so
    :func:`register_steering_buffers` was a no-op), this short-circuits
    and returns ``hidden_states`` unchanged.  The ``hasattr`` check is
    decided once at module ``__init__`` and is constant for the rest
    of the layer's lifetime, so ``torch.compile`` traces it as a static
    branch and the disabled path emits no steering kernel at all.
    """
    maybe_capture_residual(hidden_states, module.layer_idx, hook_point.value)
    table_attr = HOOK_POINT_TABLE_ATTR[hook_point]
    if not hasattr(module, table_attr):
        return hidden_states
    return _emit_steering_op(module, hidden_states, hook_point)


def apply_block_steering(
    module: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-output hook (``post_block``), replacing the old ``post_mlp``.

    vLLM defers each branch-add into the *next* fused add+norm, so at the end
    of a decoder layer ``residual`` does NOT yet include this layer's MLP
    output -- the true block output (what HF exposes as
    ``hidden_states[L + 1]``) is ``residual + hidden_states``. The old
    ``post_mlp`` hook captured bare ``residual`` (post-attention, pre-MLP-add),
    which is also byte-identical to what ``post_attn`` captures -- a footgun
    for anyone reading the residual stream.

    Capture consumers therefore see ``residual + hidden_states`` here. The
    sum is computed only when a capture manager is installed on this rank
    (a static property for the process lifetime, so ``torch.compile`` traces
    it as a constant branch; non-capture servers pay nothing).

    Steering is still applied to ``residual`` -- identical propagation to the
    old ``post_mlp`` behavior, because the steering delta rides the residual
    stream into the next layer's fused add either way.
    """
    from vllm.model_executor.layers.activation_capture import (
        get_active_capture_manager,
    )

    if get_active_capture_manager() is not None:
        maybe_capture_residual(
            residual + hidden_states,
            module.layer_idx,
            SteeringHookPoint.POST_BLOCK.value,
        )
    table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint.POST_BLOCK]
    if not hasattr(module, table_attr):
        return hidden_states, residual
    residual = _emit_steering_op(module, residual, SteeringHookPoint.POST_BLOCK)
    return hidden_states, residual


def apply_steering(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_scales: torch.Tensor,
    steering_dynamic_vec: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_row_gate: torch.Tensor,
    steering_monitor_probe: torch.Tensor,
    steering_monitor_params: torch.Tensor,
    steering_monitor_active: torch.Tensor,
    steering_decode_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply per-request activation steering via indexed gather, with the
    in-graph monitor gate fused in (non-mutating, same-hook).

    Two additive terms: the per-row gather
    ``table[index[i]] * scales[index[i]] * row_gate[i]`` plus the
    **dedicated dynamic tier** ``dynamic_vec * token_scales[i]`` (§5.4).

    ``steering_row_gate`` is a shared per-token gate (fp32,
    ``(max_tokens,)``, default 1.0) on the row term — the Phase 2 row-gating
    knob. 1.0 ⇒ the row applies at full strength (and prefill tokens stay
    1.0, so prefill rows are never gated); the in-graph monitor reduces it
    for decode tokens to make per-request steering token-conditional. See
    docs/design/dynamic_steering_row_gating.md.

    ``steering_scales`` is a shared buffer of shape ``(max_configs + 3,)``
    (fp32, default 1.0) holding a per-row strength multiplier — the
    runtime "how much" knob. The gathered row is scaled by
    ``scales[index[i]]`` before the add, so changing strength needs only
    a cheap scales write, no vector re-upload (see §5.3). A scale of 1.0
    reproduces the unscaled steering.

    ``steering_dynamic_vec`` is this (layer, hook)'s dynamic-tier vector
    (fp32, ``(hidden,)``, default 0); ``steering_token_scales`` is a
    shared per-token gate (fp32, ``(max_tokens,)``, default 0, and 0 for
    prefill tokens). The tier add ``dynamic_vec * token_scales[i]`` lets
    global dynamic steering be modulated per token and stay decode-only;
    with the default-0 gate it contributes nothing.

    ``steering_table`` is a per-layer buffer of shape
    ``(max_configs + 3, hidden_size)`` where row 0 is always zeros
    (no-steering sentinel), row 1 holds the global prefill effective
    vector, row 2 holds the global decode effective vector, and rows
    3+ hold combined phase-appropriate global + per-request vectors.

    ``steering_index`` is a shared buffer of shape ``(max_tokens,)``
    mapping each token position to its steering table row.  Updated
    in-place by the model runner before each forward pass.

    ``any_active`` is a single-element bool tensor co-located with the
    ``steering_table`` buffer that the model runner sets to ``True``
    whenever at least one non-zero row exists for this hook point in
    the current batch and ``False`` otherwise.  When ``False``, the
    kernel skips the gather + add and emits a copy of
    ``hidden_states`` so the output value is unchanged.  The flag is a
    tensor (not a Python bool) so the ``torch.compile`` graph topology
    stays stable across batches whose active-hook set differs — only
    the data in the tensor changes between forward passes.

    The compute path dispatches to a fused Triton kernel on CUDA which
    folds the gather and add into a single pass over ``hidden_states``.
    The CPU path is a plain eager add. ``steering_table`` is allocated in
    the model's compute dtype via :func:`register_steering_buffers`, so
    the gather already matches ``hidden_states.dtype`` and no cast is
    needed in either path. The output is always a freshly allocated
    tensor so the ``torch.compile`` graph keeps value semantics — never
    in place.

    Note: even with ``any_active`` False the kernel still launches and
    writes ``hidden_states`` into a fresh output tensor (a memcpy).  A
    full no-op skip — the kernel returns immediately without touching
    output memory — requires combining with the in-place sibling branch
    (``mutates_args=["hidden_states"]``) so the op can elide the output
    copy entirely.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.steering_kernel import (
            apply_steering_triton,
        )

        return apply_steering_triton(
            hidden_states,
            steering_table,
            steering_index,
            any_active,
            steering_scales,
            steering_dynamic_vec,
            steering_token_scales,
            steering_row_gate,
            steering_monitor_probe,
            steering_monitor_params,
            steering_monitor_active,
            steering_decode_mask,
        )
    # CPU eager: short-circuit on the host so we don't even materialize
    # the gather. ``.item()`` synchronizes against the device producer
    # for the flag tensor — irrelevant for the CPU path (the flag is
    # always written from the same thread before this op runs).
    if not bool(any_active.item()):
        # Match the freshly-allocated-output contract of the CUDA path so
        # callers never see an alias of ``hidden_states``.
        return hidden_states.clone()
    n = hidden_states.shape[0]
    rows = steering_index[:n]
    tscale = steering_token_scales[:n]
    rgate_t = steering_row_gate[:n]
    # Fused in-graph monitor gate (same-hook): fold the per-token gate
    # ``sigmoid(sharp·(hidden@probe − thr))`` into tscale (tier) and, when
    # gate_rows, rgate (decode-only via the mask) — locally, no buffer write.
    if bool(steering_monitor_active.item()):
        score = hidden_states.to(torch.float32) @ steering_monitor_probe.to(
            torch.float32
        )
        threshold = steering_monitor_params[0]
        sharpness = steering_monitor_params[1]
        gate_rows = steering_monitor_params[2]
        gate = torch.sigmoid(sharpness * (score - threshold))
        tscale = tscale * gate
        if bool(gate_rows.item() != 0.0):
            dm = steering_decode_mask[:n]
            rgate_t = rgate_t * (dm * gate + (1.0 - dm))
    # Per-row scale (fp32, default 1.0) × per-token row gate (default 1.0);
    # both broadcast over hidden dim. The row gate keeps prefill rows at
    # full strength (1.0) and lets the monitor gate decode rows per token.
    scale = steering_scales[rows].unsqueeze(-1).to(steering_table.dtype)
    rgate = rgate_t.unsqueeze(-1).to(steering_table.dtype)
    out = hidden_states + steering_table[rows] * scale * rgate
    # Dedicated dynamic tier: dvec * per-token gate (0 ⇒ no-op).
    tier = steering_dynamic_vec.unsqueeze(0) * tscale.unsqueeze(-1)
    return out + tier.to(out.dtype)


def apply_steering_fake(
    hidden_states: torch.Tensor,
    steering_table: torch.Tensor,
    steering_index: torch.Tensor,
    any_active: torch.Tensor,
    steering_scales: torch.Tensor,
    steering_dynamic_vec: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_row_gate: torch.Tensor,
    steering_monitor_probe: torch.Tensor,
    steering_monitor_params: torch.Tensor,
    steering_monitor_active: torch.Tensor,
    steering_decode_mask: torch.Tensor,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_steering",
    op_func=apply_steering,
    fake_impl=apply_steering_fake,
    mutates_args=[],
)


def steering_monitor(
    hidden_states: torch.Tensor,
    probe: torch.Tensor,
    params: torch.Tensor,
    monitor_active: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_decode_mask: torch.Tensor,
    steering_row_gate: torch.Tensor,
) -> None:
    """In-graph monitor (Phase 2, §8): per-token gate into the tier and,
    optionally, the per-request row term.

    Reads the pre-steering residual at a probe site, computes a per-token
    score against ``probe``, maps it through the fixed elementwise policy
    ``gate = sigmoid(sharpness * (score - threshold))`` (``params =
    [threshold, sharpness, gate_rows]``), and **multiplies** the gate into
    ``steering_token_scales[:n]`` (the §5.4 tier gate) in place. When
    ``gate_rows`` is set it ALSO gates the per-request row term by
    multiplying ``steering_row_gate[t]`` by ``mask[t]·gate[t] + (1 −
    mask[t])`` — i.e. decode tokens (``decode_mask=1``) get ``·gate``,
    prefill tokens (``decode_mask=0``) are left at full strength so prefill
    rows (which feed prefix-cache keys) are never gated. See
    docs/design/dynamic_steering_row_gating.md.

    The multiply (not overwrite) is deliberate: the runner writes
    ``token_scales``/``row_gate`` fresh each step (the per-step reset) and
    the monitor modulates within the step. Tier prefill-safety holds
    because ``token_scales=0`` for prefill; row prefill-safety holds
    because ``decode_mask=0`` for prefill ⇒ ``row_gate *= 1``.

    ``monitor_active`` is a single-element bool tensor; when ``False`` this
    is a no-op. The op mutates ``steering_token_scales`` and
    ``steering_row_gate`` and returns ``None``.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.steering_monitor_kernel import (
            steering_monitor_triton,
        )

        steering_monitor_triton(
            hidden_states,
            probe,
            params,
            monitor_active,
            steering_token_scales,
            steering_decode_mask,
            steering_row_gate,
        )
        return
    # CPU eager: short-circuit on the host (the flag is written from the
    # same thread before this op runs, so ``.item()`` never blocks).
    if not bool(monitor_active.item()):
        return
    n = hidden_states.shape[0]
    if n == 0:
        return
    threshold = params[0]
    sharpness = params[1]
    gate_rows = bool(params[2].item()) if params.shape[0] > 2 else False
    score = hidden_states.to(torch.float32) @ probe.to(torch.float32)
    gate = torch.sigmoid(sharpness * (score - threshold))
    steering_token_scales[:n] = steering_token_scales[:n] * gate.to(
        steering_token_scales.dtype
    )
    if gate_rows:
        mask = steering_decode_mask[:n]
        # decode → ·gate ; prefill → ·1 (untouched)
        row_factor = mask * gate.to(mask.dtype) + (1.0 - mask)
        steering_row_gate[:n] = steering_row_gate[:n] * row_factor.to(
            steering_row_gate.dtype
        )


def steering_monitor_fake(
    hidden_states: torch.Tensor,
    probe: torch.Tensor,
    params: torch.Tensor,
    monitor_active: torch.Tensor,
    steering_token_scales: torch.Tensor,
    steering_decode_mask: torch.Tensor,
    steering_row_gate: torch.Tensor,
) -> None:
    """FX-tracing fake — declares the mutation, computes nothing."""
    return None


direct_register_custom_op(
    op_name="steering_monitor",
    op_func=steering_monitor,
    fake_impl=steering_monitor_fake,
    mutates_args=["steering_token_scales", "steering_row_gate"],
)
