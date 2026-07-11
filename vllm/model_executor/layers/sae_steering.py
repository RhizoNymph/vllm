# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAE feature-surgery (delta) custom op + per-layer dispatch glue.

The op math, per token ``t`` with clamp set ``I``:

    pre_act_i  = W_enc[i, :] · h_t + b_enc[i]
    f_i        = activation(pre_act_i)
    new_f_i    = clamp_kind == ABSOLUTE  ? clamp_value
                 clamp_kind == ADDITIVE  ? f_i + clamp_value
                 (kind == NONE)         : f_i
    delta_i    = (new_f_i - f_i) gated by only_if_active when set
    h_t_new    = h_t + Σ_{i ∈ I} delta_i · W_dec[i, :]

The compute path dispatches via :func:`apply_sae_delta_op`, which is
registered as ``torch.ops.vllm.apply_sae_delta`` so :mod:`torch.compile`
treats the call as an opaque splitting point (mirroring
``apply_steering`` in :mod:`vllm.model_executor.layers.steering`).
Inside the op, CUDA tensors are routed to a fused Triton kernel
(:mod:`sae_steering_kernel`) and CPU tensors fall back to a vectorised
PyTorch eager body kept in this module.  The eager body remains the
ground truth for tests and for environments without Triton.

The torch-op signature uses an integer ``activation_code`` and a single
``float activation_param`` so :func:`torch.library.infer_schema` accepts
it without bespoke type adapters.  The public Python API
:func:`apply_sae_delta` keeps the original
``(SAEActivation, dict[str, float])`` shape and translates internally
before calling the op directly (i.e. *not* through ``torch.ops``) so
CPU-only test environments don't hit dispatch-key mismatches when the
op is registered for CUDA.

Numeric dtype contract (matches ``docs/features/sae_steering.md``):

* Encoder GEMM and decoder GEMM run in the model's compute dtype.
* The ``(n_tokens, n_clamp)`` activation tensor is promoted to fp32
  for the activation function and the ``delta = clamp(f, target) − f``
  subtraction; results are cast back to compute dtype before the
  decoder GEMM.

Activation support: ``ReLU``, ``JumpReLU``
(``activation_params['threshold']``), and ``TopK``
(``activation_params['k']``).  TopK selects k largest pre-activations
across the **encoder rows passed in** — for a partial encoder this is
"TopK among the clampable subset".  Operators who need full-d_sae
TopK semantics must load the full encoder.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.utils.torch_utils import direct_register_custom_op

# Integer codes for ``clamp_kind`` tensors.  Kept explicit so the
# Triton kernel (Phase 2) and any other consumer can use the same
# values without a Python enum lookup.
CLAMP_KIND_NONE = 0
CLAMP_KIND_ABSOLUTE = 1
CLAMP_KIND_ADDITIVE = 2
_VALID_CLAMP_KIND_MIN = CLAMP_KIND_NONE
_VALID_CLAMP_KIND_MAX = CLAMP_KIND_ADDITIVE

# Integer encoding of :class:`SAEActivation`.  The Triton kernel (and
# the registered torch op) take an ``int`` activation code so the
# schema stays primitive-typed; the eager body and the kernel switch
# on these codes to pick the right activation.  Values must stay in
# sync with ``ACTIVATION_CODE_*`` in :mod:`sae_steering_kernel`.
ACTIVATION_CODE_RELU = 0
ACTIVATION_CODE_JUMPRELU = 1
ACTIVATION_CODE_TOPK = 2

_ACTIVATION_TO_CODE: dict[SAEActivation, int] = {
    SAEActivation.RELU: ACTIVATION_CODE_RELU,
    SAEActivation.JUMPRELU: ACTIVATION_CODE_JUMPRELU,
    SAEActivation.TOPK: ACTIVATION_CODE_TOPK,
}
_CODE_TO_ACTIVATION: dict[int, SAEActivation] = {
    code: act for act, code in _ACTIVATION_TO_CODE.items()
}


def _validate_clamp_kind_values(clamp_kind: torch.Tensor) -> None:
    """Reject unknown clamp-kind enum values for CPU validation paths.

    CUDA layer dispatch uses worker-populated tables whose values are
    produced from validated ``SAEClampEntry.kind`` strings. Avoid a
    per-call device reduction / host sync there, while still making the
    public eager APIs fail closed for malformed CPU tensors.
    """
    if clamp_kind.is_cuda or clamp_kind.numel() == 0:
        return
    invalid = (clamp_kind < _VALID_CLAMP_KIND_MIN) | (
        clamp_kind > _VALID_CLAMP_KIND_MAX
    )
    if bool(torch.any(invalid).item()):
        bad = int(clamp_kind[invalid][0].item())
        raise ValueError(
            "clamp_kind entries must be one of "
            f"{CLAMP_KIND_NONE}, {CLAMP_KIND_ABSOLUTE}, "
            f"{CLAMP_KIND_ADDITIVE}; got {bad}."
        )


def _activation_to_scalar(
    activation: SAEActivation, activation_params: Mapping[str, float]
) -> float:
    """Pack ``activation_params`` into a single ``float`` for the op.

    The custom-op schema only supports primitive scalars, so the
    activation-specific parameter (``threshold`` for JumpReLU, ``k``
    for TopK) is collapsed into one ``float`` argument.  ReLU has no
    parameter; ``0.0`` is passed and ignored.
    """
    if activation is SAEActivation.JUMPRELU:
        return float(activation_params["threshold"])
    if activation is SAEActivation.TOPK:
        return float(activation_params["k"])
    return 0.0


def _scalar_to_activation_params(
    activation: SAEActivation, activation_param: float
) -> dict[str, float]:
    """Inverse of :func:`_activation_to_scalar` for the eager body."""
    if activation is SAEActivation.JUMPRELU:
        return {"threshold": float(activation_param)}
    if activation is SAEActivation.TOPK:
        return {"k": float(activation_param)}
    return {}


# Per-(layer, hook) buffer attribute names.  Using flat per-hook
# attributes (rather than e.g. a sub-Module wrapper) means
# ``torch.compile`` traces them as concrete buffer references rather
# than introspecting a Python container, mirroring how the additive
# steering buffers are attached.
HOOK_POINT_SAE_CLAMP_KIND_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_clamp_kind_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_clamp_kind_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_clamp_kind_post_block",
}
HOOK_POINT_SAE_CLAMP_VALUE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_clamp_value_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_clamp_value_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_clamp_value_post_block",
}
HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_clamp_only_if_active_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_clamp_only_if_active_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_clamp_only_if_active_post_block",
}
HOOK_POINT_SAE_ANY_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_any_active_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_any_active_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_any_active_post_block",
}
HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_encoder_weight_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_encoder_weight_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_encoder_weight_post_block",
}
HOOK_POINT_SAE_ENCODER_BIAS_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_encoder_bias_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_encoder_bias_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_encoder_bias_post_block",
}
HOOK_POINT_SAE_DECODER_WEIGHT_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_decoder_weight_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_decoder_weight_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_decoder_weight_post_block",
}
HOOK_POINT_SAE_MODULE_NAME_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_module_name_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_module_name_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_module_name_post_block",
}
HOOK_POINT_SAE_ACTIVATION_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_activation_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_activation_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_activation_post_block",
}
HOOK_POINT_SAE_ACTIVATION_PARAMS_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_activation_params_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_activation_params_post_attn",
    SteeringHookPoint.POST_BLOCK: "sae_activation_params_post_block",
}

# Per-hook buffer attribute names that need to be cleaned up on
# unregister.  Module name / activation / activation-params live as
# Python attributes (not buffers) and have their own cleanup path.
_SAE_BUFFER_ATTR_TABLES: tuple[dict[SteeringHookPoint, str], ...] = (
    HOOK_POINT_SAE_CLAMP_KIND_ATTR,
    HOOK_POINT_SAE_CLAMP_VALUE_ATTR,
    HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR,
    HOOK_POINT_SAE_ANY_ACTIVE_ATTR,
    HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_SAE_ENCODER_BIAS_ATTR,
    HOOK_POINT_SAE_DECODER_WEIGHT_ATTR,
)
_SAE_PYATTR_TABLES: tuple[dict[SteeringHookPoint, str], ...] = (
    HOOK_POINT_SAE_MODULE_NAME_ATTR,
    HOOK_POINT_SAE_ACTIVATION_ATTR,
    HOOK_POINT_SAE_ACTIVATION_PARAMS_ATTR,
)


def register_sae_buffers(
    module: nn.Module,
    *,
    hook_point: SteeringHookPoint,
    module_name: str,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    n_clamp: int,
    hidden_size: int,
    max_sae_configs: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    """Attach SAE buffers for one ``(layer, hook)`` site to ``module``.

    Phase-1B constrains at most one SAE module per ``(layer, hook)``
    site.  Calling this twice for the same ``hook_point`` on the same
    module raises ``ValueError``.

    Args:
        module: the decoder-layer module to attach buffers to.
        hook_point: which hook point this site sits at.
        module_name: name of the SAE module that owns this site.
        activation: encoder activation function.
        activation_params: parameters for ``activation`` (see
            :func:`sae_encode`).  Stored as a Python attribute so the
            kernel reads them as constants per-site.
        n_clamp: number of clampable encoder/decoder rows.
        hidden_size: model's hidden size (``d_model``).
        max_sae_configs: maximum number of per-request clamp table rows.
            The table also reserves row 0 for the no-op sentinel and
            rows 1/2 for prefill/decode globals.  When ``0``, registration is a no-op
            (SAE disabled engine-wide), mirroring
            ``register_steering_buffers``.
        dtype: compute dtype for the encoder/decoder weight tensors.
        device: device for the runtime buffers.  Runtime registrations
            happen after the model layer has already moved to its worker
            device, so callers should pass the device of an existing layer
            buffer to avoid attaching CPU SAE buffers to CUDA layers.
    """
    if max_sae_configs == 0:
        return
    kind_attr = HOOK_POINT_SAE_CLAMP_KIND_ATTR[hook_point]
    if hasattr(module, kind_attr):
        existing = getattr(
            module, HOOK_POINT_SAE_MODULE_NAME_ATTR[hook_point], "<unknown>"
        )
        raise ValueError(
            f"Layer module already has SAE buffers for hook "
            f"{hook_point.value!r} (owning module={existing!r}).  "
            "Phase-1B constrains at most one SAE module per "
            "(layer, hook) site; unregister the existing module first."
        )
    # Row 0 = hard no-op sentinel; row 1 = prefill globals; row 2 =
    # decode globals; rows 3..max_sae_configs+2 = per-request.  The
    # dispatch shim routes no-per-request tokens to row 1 or 2 based
    # on phase.  The populator writes global clamps into the global
    # rows and merges the phase's globals into per-request rows.
    n_rows = max_sae_configs + 3
    try:
        # Clamp tables: per-(row, feature) clamp state.
        module.register_buffer(
            kind_attr,
            torch.zeros(n_rows, n_clamp, dtype=torch.int8, device=device),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_SAE_CLAMP_VALUE_ATTR[hook_point],
            torch.zeros(n_rows, n_clamp, dtype=torch.float32, device=device),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR[hook_point],
            torch.zeros(n_rows, n_clamp, dtype=torch.bool, device=device),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_SAE_ANY_ACTIVE_ATTR[hook_point],
            torch.zeros(1, dtype=torch.bool, device=device),
            persistent=False,
        )
        # Encoder / decoder weights for the clampable subset.  Worker code
        # writes the actual values via ``copy_`` after manifest-driven
        # weight loading; defaulting to zeros means an unloaded site
        # produces zero delta (safe default, fail-quiet).
        module.register_buffer(
            HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[hook_point],
            torch.zeros(n_clamp, hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_SAE_ENCODER_BIAS_ATTR[hook_point],
            torch.zeros(n_clamp, dtype=dtype, device=device),
            persistent=False,
        )
        module.register_buffer(
            HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[hook_point],
            torch.zeros(n_clamp, hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        # Python attributes — read as per-site constants by the kernel.
        setattr(module, HOOK_POINT_SAE_MODULE_NAME_ATTR[hook_point], module_name)
        setattr(module, HOOK_POINT_SAE_ACTIVATION_ATTR[hook_point], activation)
        setattr(
            module,
            HOOK_POINT_SAE_ACTIVATION_PARAMS_ATTR[hook_point],
            dict(activation_params),
        )
    except Exception:
        unregister_sae_buffers(module, hook_point=hook_point)
        raise


def unregister_sae_buffers(
    module: nn.Module,
    *,
    hook_point: SteeringHookPoint,
) -> None:
    """Detach SAE buffers from ``module`` for ``hook_point``.

    Idempotent: no-op when buffers aren't attached.  Called when the
    owning SAE module is unregistered from the worker.
    """
    for table in _SAE_BUFFER_ATTR_TABLES:
        attr = table[hook_point]
        if hasattr(module, attr):
            # ``register_buffer`` puts the entry into ``_buffers`` *and*
            # makes it accessible via the descriptor; ``delattr`` removes
            # both consistently.
            delattr(module, attr)
    for pytable in _SAE_PYATTR_TABLES:
        attr = pytable[hook_point]
        if hasattr(module, attr):
            delattr(module, attr)


def sae_buffers_attached(module: nn.Module, hook_point: SteeringHookPoint) -> bool:
    """Constant-time check used by the layer-hook dispatch shim.

    ``torch.compile`` traces ``hasattr`` as a static branch (decided
    once at module instantiation), so the disabled path emits zero
    SAE kernel code.
    """
    return hasattr(module, HOOK_POINT_SAE_CLAMP_KIND_ATTR[hook_point])


def register_sae_index_buffer(
    module: nn.Module, max_tokens: int, device: torch.device | None = None
) -> None:
    """Attach the shared per-token ``sae_index`` buffer to ``module``.

    Mirrors the additive ``steering_index`` buffer: a single
    ``(max_tokens,)`` int64 tensor, expected to be
    :func:`share_sae_index_across_layers` so all SAE-covered layers
    point at the same physical tensor.  When ``max_tokens == 0`` (no
    SAE-bearing batches expected), registration is a no-op.

    Idempotent per layer: a manifest covering multiple hooks on the
    same decoder layer — or a second SAE module attaching to another
    hook on an already-covered layer — calls this helper again for
    the same module.  ``register_buffer`` raises when the attribute
    already exists, so we short-circuit when ``sae_index`` is present
    to keep multi-hook registrations working.
    """
    if max_tokens == 0:
        return
    if hasattr(module, "sae_index"):
        return
    module.register_buffer(
        "sae_index",
        torch.zeros(max_tokens, dtype=torch.long, device=device),
        persistent=False,
    )


def share_sae_index_across_layers(layers: list[nn.Module]) -> None:
    """Reuse one ``sae_index`` tensor across all SAE-covered layers."""
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "sae_index"):
            continue
        if shared is None:
            shared = layer.sae_index
            continue
        layer.sae_index = shared


def _topk_mask_lowest_indices(pre_act: torch.Tensor, k: int) -> torch.Tensor:
    """Return a TopK mask with ties broken by lower feature index.

    ``torch.topk`` finds the kth-value cutoff efficiently, but does not
    define which equal-valued features it returns.  We use it only for
    the cutoff, then select all greater values plus the first
    ``k - count(greater)`` tied columns in natural feature-index order.
    This keeps exact-k deterministic semantics without a full argsort.
    """
    if k >= pre_act.shape[1]:
        return torch.ones_like(pre_act, dtype=torch.bool)
    cutoff = torch.topk(pre_act, k, dim=1, largest=True, sorted=False).values.min(
        dim=1, keepdim=True
    ).values
    greater = pre_act > cutoff
    remaining = k - greater.sum(dim=1, keepdim=True)
    ties = pre_act == cutoff
    tie_rank = torch.cumsum(ties.to(torch.int64), dim=1)
    return greater | (ties & (tie_rank <= remaining))


def sae_encode(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
) -> torch.Tensor:
    """Project ``hidden_states`` through the (partial) encoder and apply activation.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.
        encoder_weight: ``(n_clamp, d_model)``.  These are the encoder
            rows for the clampable feature subset.
        encoder_bias: ``(n_clamp,)``.
        activation: encoder activation function.
        activation_params: ``{"threshold": float}`` for JumpReLU,
            ``{"k": float}`` for TopK (cast to int internally), ignored
            for ReLU.

    Returns:
        ``(n_tokens, n_clamp)`` activation tensor in fp32.  Callers
        cast back to compute dtype after applying clamps.
    """
    h_fp32 = hidden_states.to(torch.float32)
    enc_w_fp32 = encoder_weight.to(torch.float32)
    enc_b_fp32 = encoder_bias.to(torch.float32)
    pre_act = h_fp32 @ enc_w_fp32.t() + enc_b_fp32
    if activation is SAEActivation.RELU:
        return torch.clamp(pre_act, min=0.0)
    if activation is SAEActivation.JUMPRELU:
        threshold = float(activation_params["threshold"])
        return torch.where(pre_act > threshold, pre_act, torch.zeros_like(pre_act))
    if activation is SAEActivation.TOPK:
        k = int(activation_params["k"])
        n_clamp = pre_act.shape[1]
        if k >= n_clamp:
            return pre_act
        mask = _topk_mask_lowest_indices(pre_act, k)
        return torch.where(mask, pre_act, torch.zeros_like(pre_act))
    raise ValueError(f"Unsupported SAE activation: {activation!r}")


def _apply_sae_delta_eager(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    any_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Vectorized PyTorch eager body for the SAE feature-surgery op.

    Same numerics as the Triton kernel; this path is the CPU fallback
    and the test ground truth.  Inputs are tensor-only (the activation
    enum is encoded as ``activation_code`` / ``activation_param`` for
    the registered torch op).
    """
    activation = _CODE_TO_ACTIVATION[int(activation_code)]
    activation_params = _scalar_to_activation_params(activation, activation_param)

    # (n_tokens, n_clamp) fp32 — encoder pass.
    if not any_active.is_cuda and (
        not bool(any_active.item())
        or not bool(torch.any(clamp_kind != CLAMP_KIND_NONE))
    ):
        return hidden_states.clone()

    f = sae_encode(
        hidden_states, encoder_weight, encoder_bias, activation, activation_params
    )

    kind = clamp_kind.to(torch.int8)
    value = clamp_value.to(torch.float32)
    gated = clamp_only_if_active.to(torch.bool)
    active = f != 0.0 if activation is SAEActivation.TOPK else f > 0.0

    new_f_absolute = value
    new_f_additive = f + value
    new_f = torch.where(
        kind == CLAMP_KIND_ABSOLUTE,
        new_f_absolute,
        torch.where(kind == CLAMP_KIND_ADDITIVE, new_f_additive, f),
    )
    apply_clamp = (kind != CLAMP_KIND_NONE) & (~gated | active)
    if any_active.is_cuda:
        # Preserve the Triton kernels' device-side any_active gate in
        # CUDA fallback paths without synchronizing to inspect the bool.
        apply_clamp = apply_clamp & any_active.to(torch.bool).view(1, 1)
    delta = torch.where(apply_clamp, new_f - f, torch.zeros_like(f))

    delta_compute = delta.to(hidden_states.dtype)
    decoder_compute = decoder_weight.to(hidden_states.dtype)
    residual_delta = delta_compute @ decoder_compute
    return hidden_states + residual_delta


def apply_sae_delta_op(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    any_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Tensor-only entry point registered as ``torch.ops.vllm.apply_sae_delta``.

    On CUDA, dispatches to the fused Triton kernel from
    :mod:`sae_steering_kernel`.  On CPU, falls back to
    :func:`_apply_sae_delta_eager`.  The output is always a freshly
    allocated tensor with the same shape and dtype as
    ``hidden_states`` so the ``torch.compile`` graph keeps value
    semantics — never in place.

    No shape validation is performed here: callers in this module
    (:func:`apply_sae_delta`, :func:`apply_layer_sae_delta`) validate
    shapes before calling.  The custom-op schema is intentionally
    primitive-typed so :func:`torch.library.infer_schema` can produce
    a valid signature without bespoke type handling.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.sae_steering_kernel import (
            apply_sae_delta_triton,
        )

        return apply_sae_delta_triton(
            hidden_states,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            clamp_kind,
            clamp_value,
            clamp_only_if_active,
            any_active,
            int(activation_code),
            float(activation_param),
        )
    return _apply_sae_delta_eager(
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        clamp_kind,
        clamp_value,
        clamp_only_if_active,
        any_active,
        int(activation_code),
        float(activation_param),
    )


def apply_sae_delta_op_fake(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    any_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_sae_delta",
    op_func=apply_sae_delta_op,
    fake_impl=apply_sae_delta_op_fake,
    mutates_args=[],
)


def apply_sae_delta_indexed_op(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind_table: torch.Tensor,
    clamp_value_table: torch.Tensor,
    clamp_only_if_active_table: torch.Tensor,
    sae_index: torch.Tensor,
    any_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Layer-hook op that indexes clamp tables inside the backend.

    Keeping the row-index gather inside the custom op lets the CUDA kernel
    check ``any_active`` before loading clamp tables, so registered-but-idle
    SAE modules avoid per-token gather work.
    """
    if hidden_states.is_cuda:
        from vllm.model_executor.layers.sae_steering_kernel import (
            apply_sae_delta_indexed_triton,
        )

        return apply_sae_delta_indexed_triton(
            hidden_states,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            clamp_kind_table,
            clamp_value_table,
            clamp_only_if_active_table,
            sae_index,
            any_active,
            int(activation_code),
            float(activation_param),
        )
    if not any_active.is_cuda and not bool(any_active.item()):
        return hidden_states.clone()
    n_tokens = hidden_states.shape[0]
    idx = sae_index[:n_tokens]
    return _apply_sae_delta_eager(
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        clamp_kind_table[idx],
        clamp_value_table[idx],
        clamp_only_if_active_table[idx],
        any_active,
        int(activation_code),
        float(activation_param),
    )


def apply_sae_delta_indexed_op_fake(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    clamp_kind_table: torch.Tensor,
    clamp_value_table: torch.Tensor,
    clamp_only_if_active_table: torch.Tensor,
    sae_index: torch.Tensor,
    any_active: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_sae_delta_indexed",
    op_func=apply_sae_delta_indexed_op,
    fake_impl=apply_sae_delta_indexed_op_fake,
    mutates_args=[],
)


def apply_sae_delta(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    any_active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Public Python API for the SAE feature-surgery delta op.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.  Output
            is the same shape and dtype.
        encoder_weight: ``(n_clamp, d_model)`` encoder rows for the
            clampable feature subset.
        encoder_bias: ``(n_clamp,)``.
        decoder_weight: ``(n_clamp, d_model)`` decoder rows aligned
            with ``encoder_weight``: row ``i`` is the decoder direction
            for the same feature whose encoder row is at index ``i``.
        activation: encoder activation function.
        activation_params: parameters for ``activation`` (see
            :func:`sae_encode`).
        clamp_kind: ``(n_tokens, n_clamp)`` int8.  Per-token, per-
            feature clamp kind: ``CLAMP_KIND_NONE`` (skip),
            ``CLAMP_KIND_ABSOLUTE`` (set ``f := value``), or
            ``CLAMP_KIND_ADDITIVE`` (set ``f := f + value``).
        clamp_value: ``(n_tokens, n_clamp)`` float.  Target value for
            absolute clamps; offset for additive clamps.  Ignored where
            ``clamp_kind == CLAMP_KIND_NONE``.
        clamp_only_if_active: ``(n_tokens, n_clamp)`` bool.  When True,
            the clamp is suppressed at positions where ``f <= 0`` in
            the live encoder pass — "amplify when present" semantics.
        any_active: Optional single-element bool tensor.  When False,
            the op skips the table gather / encoder / decoder work and
            returns a fresh no-op copy.  Layer-hook dispatch passes a
            device tensor here so compiled CUDA graphs keep a stable
            topology while inactive SAE sites avoid the expensive path.

    Returns:
        ``hidden_states + Σ_i delta_i · W_dec[i]`` in the same dtype
        as ``hidden_states``.

    The compute path goes through :func:`apply_sae_delta_op` (the
    registered torch custom op).  Calling it directly here — rather
    than via ``torch.ops.vllm.apply_sae_delta`` — keeps CPU-only test
    environments insulated from the registered dispatch key (which
    follows the platform: "CPU" on CPU-only, "CUDA" on a CUDA build).
    The internal branch on ``hidden_states.is_cuda`` picks the right
    backend without going through the dispatcher.
    """
    n_tokens, d_model = hidden_states.shape
    n_clamp = encoder_weight.shape[0]

    if encoder_weight.shape != (n_clamp, d_model):
        raise ValueError(
            "encoder_weight must be (n_clamp, d_model) matching hidden_states; "
            f"got {tuple(encoder_weight.shape)} vs d_model={d_model}."
        )
    if encoder_bias.shape != (n_clamp,):
        raise ValueError(
            "encoder_bias must be (n_clamp,); "
            f"got {tuple(encoder_bias.shape)} vs n_clamp={n_clamp}."
        )
    if decoder_weight.shape != (n_clamp, d_model):
        raise ValueError(
            "decoder_weight must be (n_clamp, d_model) aligned with encoder; "
            f"got {tuple(decoder_weight.shape)} vs (n_clamp={n_clamp}, "
            f"d_model={d_model})."
        )
    expected_clamp_shape = (n_tokens, n_clamp)
    for name, t in (
        ("clamp_kind", clamp_kind),
        ("clamp_value", clamp_value),
        ("clamp_only_if_active", clamp_only_if_active),
    ):
        if tuple(t.shape) != expected_clamp_shape:
            raise ValueError(
                f"{name} must be {expected_clamp_shape}; got {tuple(t.shape)}."
            )
    if clamp_kind.dtype != torch.int8:
        raise ValueError(f"clamp_kind must be torch.int8; got {clamp_kind.dtype}.")
    _validate_clamp_kind_values(clamp_kind)
    if not clamp_value.dtype.is_floating_point:
        raise ValueError(
            f"clamp_value must be a floating dtype; got {clamp_value.dtype}."
        )
    if clamp_only_if_active.dtype != torch.bool:
        raise ValueError(
            "clamp_only_if_active must be torch.bool; "
            f"got {clamp_only_if_active.dtype}."
        )

    # n_clamp == 0 short-circuit: no features to clamp, no work to do.
    if n_clamp == 0:
        return hidden_states.clone()
    if any_active is None:
        any_active = torch.ones(1, dtype=torch.bool, device=hidden_states.device)

    code = _ACTIVATION_TO_CODE[activation]
    param = _activation_to_scalar(activation, activation_params)
    return apply_sae_delta_op(
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        clamp_kind,
        clamp_value,
        clamp_only_if_active,
        any_active,
        code,
        param,
    )


def apply_layer_sae_delta(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Layer-hook dispatch: pull buffer state, hand it to the math primitive.

    When the layer has no SAE buffers attached for ``hook_point``
    (engine started with SAE disabled or the site isn't covered by
    any registered module), this short-circuits and returns
    ``hidden_states`` unchanged.  The ``hasattr`` check is decided
    once at module instantiation and is constant for the rest of the
    layer's lifetime, so ``torch.compile`` traces it as a static
    branch and the disabled path emits no SAE kernel at all —
    mirroring :func:`apply_layer_steering`.

    The shim passes the row table plus ``sae_index`` to
    ``torch.ops.vllm.apply_sae_delta_indexed``.  The backend performs
    the per-token gather after checking ``any_active``, so an attached
    but idle SAE module avoids table-gather work.  The torch-op
    indirection is what makes :mod:`torch.compile` treat the SAE call
    as an opaque splitting point (mirroring
    :func:`apply_layer_steering` → ``torch.ops.vllm.apply_steering``);
    under CUDA the op routes to a fused Triton kernel.  ``n_clamp == 0``
    short-circuits before the op call so we never launch a degenerate
    kernel.
    """
    if not sae_buffers_attached(module, hook_point):
        return hidden_states
    n_tokens = hidden_states.shape[0]
    sae_index = module.sae_index[:n_tokens]  # type: ignore[union-attr]
    kind_table = getattr(module, HOOK_POINT_SAE_CLAMP_KIND_ATTR[hook_point])
    value_table = getattr(module, HOOK_POINT_SAE_CLAMP_VALUE_ATTR[hook_point])
    only_table = getattr(module, HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR[hook_point])
    enc_w = getattr(module, HOOK_POINT_SAE_ENCODER_WEIGHT_ATTR[hook_point])
    enc_b = getattr(module, HOOK_POINT_SAE_ENCODER_BIAS_ATTR[hook_point])
    dec_w = getattr(module, HOOK_POINT_SAE_DECODER_WEIGHT_ATTR[hook_point])
    activation: SAEActivation = getattr(
        module, HOOK_POINT_SAE_ACTIVATION_ATTR[hook_point]
    )
    activation_params: dict[str, float] = getattr(
        module, HOOK_POINT_SAE_ACTIVATION_PARAMS_ATTR[hook_point]
    )

    if enc_w.shape[0] == 0:
        return hidden_states
    any_active = getattr(module, HOOK_POINT_SAE_ANY_ACTIVE_ATTR[hook_point], None)
    if any_active is None:
        any_active = torch.ones(1, dtype=torch.bool, device=hidden_states.device)

    code = _ACTIVATION_TO_CODE[activation]
    param = _activation_to_scalar(activation, activation_params)
    return torch.ops.vllm.apply_sae_delta_indexed(
        hidden_states,
        enc_w,
        enc_b,
        dec_w,
        kind_table,
        value_table,
        only_table,
        sae_index,
        any_active,
        code,
        param,
    )


def populate_sae_clamp_table(
    *,
    manager: SAEClampManager,  # noqa: F821 — forward ref to avoid import cycle
    module: nn.Module,
    hook_point: SteeringHookPoint,
    module_name: str,
    clampable_features: tuple[int, ...],
    layer_idx: int | None = None,
    worker_phase: str | None = None,
) -> None:
    """Project active manager rows into the per-(layer, hook) clamp tables.

    Walks every active row in ``manager`` and writes its clamp content
    into the corresponding row of ``module``'s clamp tables for this
    ``hook_point``, gated by:

    * **Module match** — the spec's ``module_name`` must equal this
      site's ``module_name``.  Specs from other modules are skipped
      at this site.
    * **Phase match** — the spec's ``phase`` field (``"both"`` /
      ``"prefill"`` / ``"decode"``) must be compatible with the
      row's ``row_phase`` (the worker phase the row was admitted
      under, recorded by
      :meth:`SAEClampManager.register_clamp_spec`).  Each row is
      written under its own phase in a single pass so prefill and
      decode rows coexist without overwriting each other.
    * **Layer/hook match** — when ``layer_idx`` is given, only entries
      under ``spec.clamps[hook_point.value][layer_idx]`` are projected.

    Feature indices are resolved against ``clampable_features`` (the
    module's clampable-set order); a feature in the spec that is not
    in ``clampable_features`` raises ``ValueError``.

    Row 0 (the no-op sentinel) is always reset to all-zero so it stays
    a true no-op. Rows 1 and 2 hold global prefill/decode clamps,
    respectively; they remain zero when no globals apply for that
    phase.

    The optional ``worker_phase`` argument is retained for tests that
    want to assert phase-gating behaviour explicitly: when given, the
    populator only writes rows whose ``row_phase`` matches it, leaving
    other rows' existing buffer content untouched.  Production
    callers omit it so every row is populated under its own phase.

    The populator runs only when manager state is dirty, not on every
    token. The per-token work happens in the custom op after the row
    table and ``sae_index`` have been materialized.

    Args:
        manager: the :class:`SAEClampManager` holding active rows.
        module: the layer module with SAE buffers attached.
        hook_point: which hook this site sits at.
        module_name: name of the SAE module that owns this site;
            specs from other modules are skipped at this site.
        clampable_features: ordered tuple of feature indices that
            define the (feature_idx → row position) mapping for this
            module.
        layer_idx: layer index this site sits at; when ``None`` (e.g.
            for tests that don't care about layer specificity), the
            populator scans all layers in the spec under
            ``hook_point``.
        worker_phase: optional phase filter — when set, only rows with
            matching ``row_phase`` are touched.
    """
    if not sae_buffers_attached(module, hook_point):
        return
    if worker_phase is not None and worker_phase not in ("prefill", "decode"):
        raise ValueError(
            f"worker_phase must be 'prefill' or 'decode' or None; got {worker_phase!r}."
        )
    kind_table = getattr(module, HOOK_POINT_SAE_CLAMP_KIND_ATTR[hook_point])
    value_table = getattr(module, HOOK_POINT_SAE_CLAMP_VALUE_ATTR[hook_point])
    only_table = getattr(module, HOOK_POINT_SAE_CLAMP_ONLY_IF_ACTIVE_ATTR[hook_point])
    any_active = getattr(module, HOOK_POINT_SAE_ANY_ACTIVE_ATTR[hook_point], None)
    n_clamp = kind_table.shape[1]
    if len(clampable_features) != n_clamp:
        raise ValueError(
            "clampable_features length must equal n_clamp; got "
            f"{len(clampable_features)} vs {n_clamp}."
        )
    feature_to_pos: dict[int, int] = {f: i for i, f in enumerate(clampable_features)}
    # Row 0: hard sentinel. Rows 1 and 2 are phase-specific global
    # rows. Per-request rows below also stack globals so a request that
    # opts in still gets globals applied.
    kind_table[0].zero_()
    value_table[0].zero_()
    only_table[0].zero_()
    if worker_phase is None or worker_phase == "prefill":
        kind_table[1].zero_()
        value_table[1].zero_()
        only_table[1].zero_()
    if worker_phase is None or worker_phase == "decode":
        kind_table[2].zero_()
        value_table[2].zero_()
        only_table[2].zero_()
    if any_active is not None and worker_phase is None:
        any_active.zero_()
    hook_name = hook_point.value

    def _mark_any_active() -> None:
        if any_active is not None:
            any_active.fill_(True)

    def _write_entries_to_row(
        row: int,
        entries_for_site: list,
        *,
        reject_existing: bool = False,
    ) -> bool:
        """Write a list of clamp entries into ``row`` at this site.

        Returns True if any entry was written.  The caller is
        responsible for zeroing the row beforehand.  Feature indices
        already populated at this row can be rejected by setting
        ``reject_existing=True``; the manager enforces the same
        no-overlap invariant at admission time for global/per-request
        combinations.
        """
        any_written = False
        for entry in entries_for_site:
            pos = feature_to_pos.get(entry.feature_idx)
            if pos is None:
                raise ValueError(
                    f"SAEClampSpec entry feature_idx={entry.feature_idx} for "
                    f"module {module_name!r} is not in clampable_features "
                    f"{list(clampable_features)} for site "
                    f"(layer={layer_idx}, hook={hook_name})."
                )
            if reject_existing and kind_table[row, pos].item() != CLAMP_KIND_NONE:
                raise ValueError(
                    "SAE global and per-request clamps overlap for "
                    f"module {module_name!r}, hook={hook_name!r}, "
                    f"layer={layer_idx}, feature_idx={entry.feature_idx}."
                )
            kind_table[row, pos] = (
                CLAMP_KIND_ABSOLUTE
                if entry.kind == "absolute"
                else CLAMP_KIND_ADDITIVE
            )
            value_table[row, pos] = float(entry.value)
            only_table[row, pos] = bool(entry.only_if_active)
            any_written = True
        return any_written

    def _gather_entries_for_specs(
        specs_iter,
        row_phase: str,
    ) -> list:
        """Gather clamp entries for this site from the given specs tuple."""
        out: list = []
        for spec in specs_iter:
            if spec.module_name != module_name:
                continue
            if spec.phase != "both" and spec.phase != row_phase:
                continue
            layer_map = spec.clamps.get(hook_name)
            if layer_map is None:
                continue
            if layer_idx is None:
                for entries in layer_map.values():
                    out.extend(entries)
            else:
                out.extend(layer_map.get(layer_idx, ()))
        return out

    def _apply_globals_to_row(row: int, row_phase: str) -> bool:
        """Write the global clamps for ``row_phase`` into ``row``."""
        global_specs = manager.global_specs_for_phase(row_phase)
        if not global_specs:
            return False
        entries = _gather_entries_for_specs(global_specs, row_phase)
        if not entries:
            return False
        return _write_entries_to_row(row, entries)

    global_rows = {"prefill": 1, "decode": 2}
    for global_phase, global_row in global_rows.items():
        if worker_phase is not None and worker_phase != global_phase:
            continue
        if _apply_globals_to_row(global_row, global_phase):
            _mark_any_active()

    for row, _config_hash, row_phase, specs in manager.active_rows():
        if worker_phase is not None and row_phase != worker_phase:
            continue
        # Default: zero this row at this site.  Then accumulate
        # globals first, followed by the request's own clamps.  Any
        # global/request collision is rejected instead of overwritten.
        kind_table[row].zero_()
        value_table[row].zero_()
        only_table[row].zero_()
        if _apply_globals_to_row(row, row_phase):
            _mark_any_active()
        entries = _gather_entries_for_specs(specs, row_phase)
        if entries and _write_entries_to_row(row, entries, reject_existing=True):
            _mark_any_active()


# Note: an earlier draft had a ``_phase_applies(spec_phase, worker_phase,
# row_phase)`` helper that gated content on a *worker_phase* argument.
# That broke when the populator was called twice (once per worker phase)
# because the second call zeroed rows the first call had populated.  The
# populator now writes each row under its own ``row_phase`` in a single
# pass; the residual phase check ``spec.phase == "both" or
# spec.phase == row_phase`` lives inline above.


# Forward-reference import is resolved at call time; importing at module
# level would create a cycle (sae_clamp_manager imports
# SAEClampSpec from vllm.config.sae_steering_types, which is fine, but
# importing SAEClampManager here from a worker module would tie this
# layer module to the worker hierarchy).  This local import lifts the
# type into scope without taking the cycle.
from vllm.v1.worker.sae_clamp_manager import SAEClampManager  # noqa: E402,F401
