# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SAE feature-surgery (full reconstruction) custom op + per-layer dispatch glue.

Sibling to :mod:`sae_steering` (the *delta* path).  The math, per
token ``t`` whose ``recon_mask[t]`` is ``True``:

    pre_act     = h_t @ W_enc.T + b_enc          # (d_sae,)
    f           = activation(pre_act)            # (d_sae,)
    f' [c]      = apply_clamp(f[clampable_features[c]],
                              clamp_kind[t, c],
                              clamp_value[t, c],
                              clamp_only_if_active[t, c])
                  # for c ∈ [0, n_clamp); other features are unchanged
    h_t_new     = f' @ W_dec + b_dec             # (d_model,)

Tokens where ``recon_mask[t]`` is ``False`` pass through unchanged
(``out[t] = h[t]``).  This per-token gate is what makes per-request
opt-in tractable: a request that does not register a full-
reconstruction spec keeps its residual stream identical to a build
without the SAE module attached, the same per-token-row-indexing
contract the additive and delta paths use.

Compared to the delta variant (:func:`apply_sae_delta`):

* Encoder / decoder use the **full** ``d_sae`` rows, not just a
  ``clampable_features`` subset.  The reconstructed residual carries
  the SAE's reconstruction error along with whatever clamp
  modifications the caller requested, replacing the original
  residual entirely (Anthropic Scaling Monosemanticity / Golden Gate
  Claude semantics).
* The decoder bias ``b_dec`` participates — it does not in the delta
  path, where the result is a perturbation around ``h``.
* Per-token opt-in is via ``recon_mask`` (a bool tensor) rather than
  via row 0 of an indirection table — the math primitive is
  layer-buffer-agnostic so the eventual worker integration can wire
  ``recon_mask`` from any source (an explicit per-token flag, a
  derived ``recon_index != 0`` mask, etc.).

The compute path dispatches via :func:`apply_sae_full_reconstruction_op`,
registered as ``torch.ops.vllm.apply_sae_full_reconstruction`` so
:mod:`torch.compile` treats the call as an opaque splitting point
(mirroring ``apply_steering`` and ``apply_sae_delta``).  CPU is
served by the eager body; CUDA dispatches to the fused Triton
kernel — wired in Stage 4.  Until then the CUDA path also routes to
the eager body.

Numeric dtype contract (matches ``docs/features/sae_steering.md``):

* Encoder GEMM and decoder GEMM run in the model's compute dtype.
* The activation tensor and clamp arithmetic are promoted to fp32 in
  the eager body to match the contract the kernel will use; results
  are cast back to compute dtype before the decoder GEMM.

Activation support: ``ReLU``, ``JumpReLU``
(``activation_params['threshold']``), and ``TopK``
(``activation_params['k']``).  Unlike the delta path, TopK runs over
the **full** ``d_sae`` here — true Golden-Gate-style semantics —
because we have the full encoder available.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from vllm.config.sae_steering_types import SAEActivation
from vllm.model_executor.layers.sae_steering import (
    _ACTIVATION_TO_CODE,
    _CODE_TO_ACTIVATION,
    ACTIVATION_CODE_RELU,
    CLAMP_KIND_ABSOLUTE,
    CLAMP_KIND_ADDITIVE,
    CLAMP_KIND_NONE,
    _activation_to_scalar,
    _scalar_to_activation_params,
)
from vllm.model_executor.layers.steering import SteeringHookPoint
from vllm.utils.torch_utils import direct_register_custom_op

# Per-(layer, hook) buffer attribute names.  Flat per-hook attrs (not
# a sub-Module wrapper) so ``torch.compile`` traces them as concrete
# buffer references rather than container introspection — the same
# rationale as :mod:`sae_steering`.

# Full SAE encoder / decoder weight tensors, shared across all rows.
HOOK_POINT_FR_ENCODER_WEIGHT_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_encoder_weight_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_encoder_weight_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_encoder_weight_post_mlp",
}
HOOK_POINT_FR_ENCODER_BIAS_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_encoder_bias_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_encoder_bias_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_encoder_bias_post_mlp",
}
HOOK_POINT_FR_DECODER_WEIGHT_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_decoder_weight_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_decoder_weight_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_decoder_weight_post_mlp",
}
HOOK_POINT_FR_DECODER_BIAS_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_decoder_bias_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_decoder_bias_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_decoder_bias_post_mlp",
}
# Per-row clamp tables: row ``r`` carries the clamp state for tokens
# whose ``sae_recon_index`` selects ``r``.  Row 0 is reserved as the
# "no reconstruction" sentinel — a token that maps to row 0 passes
# through unchanged because :func:`apply_layer_sae_full_reconstruction`
# derives ``recon_mask`` as ``recon_index != 0``.
HOOK_POINT_FR_CLAMP_KIND_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_clamp_kind_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_clamp_kind_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_clamp_kind_post_mlp",
}
HOOK_POINT_FR_CLAMP_VALUE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_clamp_value_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_clamp_value_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_clamp_value_post_mlp",
}
HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_clamp_only_if_active_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_clamp_only_if_active_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_clamp_only_if_active_post_mlp",
}
# Clampable global feature indices for this site (constant per
# manifest registration).
HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_clampable_features_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_clampable_features_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_clampable_features_post_mlp",
}
# Module name + activation are Python attributes — read as per-site
# constants by the kernel.
HOOK_POINT_FR_MODULE_NAME_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_module_name_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_module_name_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_module_name_post_mlp",
}
HOOK_POINT_FR_ACTIVATION_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_activation_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_activation_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_activation_post_mlp",
}
HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR: dict[SteeringHookPoint, str] = {
    SteeringHookPoint.PRE_ATTN: "sae_fr_activation_params_pre_attn",
    SteeringHookPoint.POST_ATTN: "sae_fr_activation_params_post_attn",
    SteeringHookPoint.POST_MLP: "sae_fr_activation_params_post_mlp",
}

# Buffer attribute tables — used by :func:`unregister_sae_full_recon_buffers`
# to delete every per-(hook) buffer in one pass.
_FR_BUFFER_ATTR_TABLES: tuple[dict[SteeringHookPoint, str], ...] = (
    HOOK_POINT_FR_ENCODER_WEIGHT_ATTR,
    HOOK_POINT_FR_ENCODER_BIAS_ATTR,
    HOOK_POINT_FR_DECODER_WEIGHT_ATTR,
    HOOK_POINT_FR_DECODER_BIAS_ATTR,
    HOOK_POINT_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_FR_CLAMP_VALUE_ATTR,
    HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR,
    HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR,
)
_FR_PYATTR_TABLES: tuple[dict[SteeringHookPoint, str], ...] = (
    HOOK_POINT_FR_MODULE_NAME_ATTR,
    HOOK_POINT_FR_ACTIVATION_ATTR,
    HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR,
)


def register_sae_full_recon_buffers(
    module: nn.Module,
    *,
    hook_point: SteeringHookPoint,
    module_name: str,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    d_sae: int,
    n_clamp: int,
    hidden_size: int,
    max_recon_configs: int,
    clampable_features: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Attach full-reconstruction SAE buffers for one ``(layer, hook)`` site.

    Phase-4 constrains at most one full-reconstruction SAE module per
    ``(layer, hook)`` site, mirroring the delta path's invariant.
    Calling this twice for the same ``hook_point`` on the same module
    raises ``ValueError``.

    The clamp tables are sized ``(max_recon_configs + 1, n_clamp)``
    where row 0 is the no-reconstruction sentinel — never written
    by the populator and gated out by the layer-hook dispatch shim
    via the ``recon_index != 0`` derivation.

    Args:
        module: the decoder-layer module to attach buffers to.
        hook_point: which hook point this site sits at.
        module_name: name of the SAE module that owns this site.
        activation: encoder activation function.
        activation_params: parameters for ``activation``.
        d_sae: SAE feature count.  Encoder / decoder weight buffers
            are sized ``(d_sae, hidden_size)``.
        n_clamp: number of clampable features (length of
            ``clampable_features``).  Clamp tables are sized
            ``(max_recon_configs + 1, n_clamp)``.
        hidden_size: model's hidden size (``d_model``).
        max_recon_configs: per-step row capacity (analog of
            ``max_steering_configs``).  When ``0``, registration is
            a no-op (full-reconstruction disabled engine-wide).
        clampable_features: ``(n_clamp,)`` int64 tensor of *global*
            feature indices that may be clamped at this site.
        dtype: compute dtype for the encoder / decoder weight tensors.
    """
    if max_recon_configs == 0:
        return
    if d_sae <= 0:
        raise ValueError(f"d_sae must be positive; got {d_sae}.")
    if n_clamp < 0:
        raise ValueError(f"n_clamp must be non-negative; got {n_clamp}.")
    if clampable_features.dtype != torch.int64:
        raise ValueError(
            f"clampable_features must be int64; got dtype={clampable_features.dtype}."
        )
    if tuple(clampable_features.shape) != (n_clamp,):
        raise ValueError(
            f"clampable_features must have shape ({n_clamp},); "
            f"got {tuple(clampable_features.shape)}."
        )
    enc_w_attr = HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[hook_point]
    if hasattr(module, enc_w_attr):
        existing = getattr(
            module, HOOK_POINT_FR_MODULE_NAME_ATTR[hook_point], "<unknown>"
        )
        raise ValueError(
            f"Layer module already has full-reconstruction SAE buffers for "
            f"hook {hook_point.value!r} (owning module={existing!r}).  "
            "Phase-4 constrains at most one full-reconstruction SAE module "
            "per (layer, hook) site; unregister the existing module first."
        )
    n_rows = max_recon_configs + 1
    module.register_buffer(
        enc_w_attr,
        torch.zeros(d_sae, hidden_size, dtype=dtype),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_ENCODER_BIAS_ATTR[hook_point],
        torch.zeros(d_sae, dtype=dtype),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_DECODER_WEIGHT_ATTR[hook_point],
        torch.zeros(d_sae, hidden_size, dtype=dtype),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_DECODER_BIAS_ATTR[hook_point],
        torch.zeros(hidden_size, dtype=dtype),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_CLAMP_KIND_ATTR[hook_point],
        torch.zeros(n_rows, n_clamp, dtype=torch.int8),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_CLAMP_VALUE_ATTR[hook_point],
        torch.zeros(n_rows, n_clamp, dtype=torch.float32),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR[hook_point],
        torch.zeros(n_rows, n_clamp, dtype=torch.bool),
        persistent=False,
    )
    module.register_buffer(
        HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR[hook_point],
        clampable_features.detach().to(torch.int64).clone(),
        persistent=False,
    )
    setattr(module, HOOK_POINT_FR_MODULE_NAME_ATTR[hook_point], module_name)
    setattr(module, HOOK_POINT_FR_ACTIVATION_ATTR[hook_point], activation)
    setattr(
        module,
        HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR[hook_point],
        dict(activation_params),
    )


def unregister_sae_full_recon_buffers(
    module: nn.Module,
    *,
    hook_point: SteeringHookPoint,
) -> None:
    """Detach full-reconstruction SAE buffers from ``module`` for ``hook_point``.

    Idempotent: no-op when buffers aren't attached.  Called when the
    owning SAE module is unregistered from the worker.
    """
    for table in _FR_BUFFER_ATTR_TABLES:
        attr = table[hook_point]
        if hasattr(module, attr):
            delattr(module, attr)
    for pytable in _FR_PYATTR_TABLES:
        attr = pytable[hook_point]
        if hasattr(module, attr):
            delattr(module, attr)


def sae_full_recon_buffers_attached(
    module: nn.Module, hook_point: SteeringHookPoint
) -> bool:
    """Constant-time check used by the layer-hook dispatch shim.

    ``torch.compile`` traces ``hasattr`` as a static branch (decided
    once at module instantiation), so the disabled path emits zero
    full-reconstruction kernel code.
    """
    return hasattr(module, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[hook_point])


def register_sae_recon_index_buffer(module: nn.Module, max_tokens: int) -> None:
    """Attach the shared per-token ``sae_recon_index`` buffer to ``module``.

    Mirrors the additive ``steering_index`` and the delta path's
    ``sae_index``: a single ``(max_tokens,)`` int64 tensor, expected
    to be shared across all SAE-full-reconstruction-covered layers
    via :func:`share_sae_recon_index_across_layers`.  Row 0 is the
    no-reconstruction sentinel; the layer-hook dispatch shim derives
    ``recon_mask = (recon_index != 0)``.
    """
    if max_tokens == 0:
        return
    module.register_buffer(
        "sae_recon_index",
        torch.zeros(max_tokens, dtype=torch.long),
        persistent=False,
    )


def share_sae_recon_index_across_layers(layers: list[nn.Module]) -> None:
    """Reuse one ``sae_recon_index`` tensor across all covered layers."""
    shared: torch.Tensor | None = None
    for layer in layers:
        if not hasattr(layer, "sae_recon_index"):
            continue
        if shared is None:
            shared = layer.sae_recon_index  # type: ignore[union-attr]
            continue
        layer.sae_recon_index = shared  # type: ignore[union-attr]


def sae_encode_full(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
) -> torch.Tensor:
    """Project ``hidden_states`` through the full SAE encoder.

    Mirrors :func:`sae_steering.sae_encode` but operates on the full
    ``d_sae`` encoder rather than a clampable-subset slice; the
    reconstruction op needs every feature's activation to recompute
    the residual.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.
        encoder_weight: ``(d_sae, d_model)``.
        encoder_bias: ``(d_sae,)``.
        activation: encoder activation function.
        activation_params: ``{"threshold": float}`` for JumpReLU,
            ``{"k": float}`` for TopK (cast to int internally), ignored
            for ReLU.

    Returns:
        ``(n_tokens, d_sae)`` activation tensor in fp32.  The full-
        reconstruction caller modifies this in place at the
        clampable-feature positions before the decoder pass.
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
        d_sae = pre_act.shape[1]
        if k >= d_sae:
            return pre_act
        _, top_idx = torch.topk(pre_act, k=k, dim=1, largest=True)
        mask = torch.zeros_like(pre_act, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
        return torch.where(mask, pre_act, torch.zeros_like(pre_act))
    raise ValueError(f"Unsupported SAE activation: {activation!r}")


def _apply_sae_full_reconstruction_eager(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Vectorized PyTorch eager body for the full-reconstruction op.

    Same numerics as the eventual Triton kernel; this path is the
    CPU fallback and the test ground truth.  Inputs are tensor-only
    (the activation enum is encoded as ``activation_code`` /
    ``activation_param`` for the registered torch op).
    """
    n_tokens = hidden_states.shape[0]
    n_clamp = clampable_features.shape[0]
    if n_tokens == 0:
        return hidden_states.clone()

    activation = _CODE_TO_ACTIVATION[int(activation_code)]
    activation_params = _scalar_to_activation_params(activation, activation_param)

    f = sae_encode_full(
        hidden_states,
        encoder_weight,
        encoder_bias,
        activation,
        activation_params,
    )

    if n_clamp > 0:
        idx_2d = clampable_features.unsqueeze(0).expand(n_tokens, -1)
        f_subset = f.gather(1, idx_2d)

        kind = clamp_kind.to(torch.int8)
        value = clamp_value.to(torch.float32)
        gated = clamp_only_if_active.to(torch.bool)
        active = f_subset > 0.0

        new_f_absolute = value
        new_f_additive = f_subset + value
        new_f = torch.where(
            kind == CLAMP_KIND_ABSOLUTE,
            new_f_absolute,
            torch.where(kind == CLAMP_KIND_ADDITIVE, new_f_additive, f_subset),
        )
        apply_clamp = (kind != CLAMP_KIND_NONE) & (~gated | active)
        new_f_subset = torch.where(apply_clamp, new_f, f_subset)

        f = f.scatter(1, idx_2d, new_f_subset)

    f_compute = f.to(hidden_states.dtype)
    decoder_compute = decoder_weight.to(hidden_states.dtype)
    decoder_bias_compute = decoder_bias.to(hidden_states.dtype)
    reconstruction = f_compute @ decoder_compute + decoder_bias_compute
    return torch.where(recon_mask.unsqueeze(1), reconstruction, hidden_states)


def apply_sae_full_reconstruction_op(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """Tensor-only entry registered as ``torch.ops.vllm.apply_sae_full_reconstruction``.

    On CUDA, will dispatch to the fused Triton kernel once Stage 4
    lands; until then both CPU and CUDA route through
    :func:`_apply_sae_full_reconstruction_eager`.  The output is
    always a freshly allocated tensor with the same shape and dtype
    as ``hidden_states``.

    No shape validation is performed here — callers in this module
    (:func:`apply_sae_full_reconstruction`,
    :func:`apply_layer_sae_full_reconstruction`) validate before
    calling.  The schema is intentionally primitive-typed so
    :func:`torch.library.infer_schema` produces a valid signature
    without bespoke type adapters, mirroring the delta path.
    """
    if hidden_states.is_cuda:
        # Stage 4 will swap in the Triton kernel here.  Until then
        # the CUDA path runs the eager body so the surface is stable
        # and CUDA-graph capture / torch.compile fences already work
        # against the registered op.
        return _apply_sae_full_reconstruction_eager(
            hidden_states,
            encoder_weight,
            encoder_bias,
            decoder_weight,
            decoder_bias,
            clampable_features,
            clamp_kind,
            clamp_value,
            clamp_only_if_active,
            recon_mask,
            int(activation_code),
            float(activation_param),
        )
    return _apply_sae_full_reconstruction_eager(
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        decoder_bias,
        clampable_features,
        clamp_kind,
        clamp_value,
        clamp_only_if_active,
        recon_mask,
        int(activation_code),
        float(activation_param),
    )


def apply_sae_full_reconstruction_op_fake(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
    activation_code: int,
    activation_param: float,
) -> torch.Tensor:
    """FX-tracing fake — correct shape, no computation."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="apply_sae_full_reconstruction",
    op_func=apply_sae_full_reconstruction_op,
    fake_impl=apply_sae_full_reconstruction_op_fake,
    mutates_args=[],
)


def apply_sae_full_reconstruction(
    hidden_states: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
    decoder_bias: torch.Tensor,
    activation: SAEActivation,
    activation_params: Mapping[str, float],
    clampable_features: torch.Tensor,
    clamp_kind: torch.Tensor,
    clamp_value: torch.Tensor,
    clamp_only_if_active: torch.Tensor,
    recon_mask: torch.Tensor,
) -> torch.Tensor:
    """Public Python API for the SAE full-reconstruction op.

    Args:
        hidden_states: ``(n_tokens, d_model)``, compute dtype.  Output
            is the same shape and dtype.
        encoder_weight: ``(d_sae, d_model)`` — *full* encoder rows.
        encoder_bias: ``(d_sae,)``.
        decoder_weight: ``(d_sae, d_model)`` — *full* decoder rows.
        decoder_bias: ``(d_model,)``.
        activation: encoder activation function.
        activation_params: parameters for ``activation`` (see
            :func:`sae_encode_full`).
        clampable_features: ``(n_clamp,)`` int64 tensor of *global*
            feature indices that may be clamped per-token.  Indexes
            into the ``d_sae`` activation tensor; values must lie in
            ``[0, d_sae)`` and be unique.
        clamp_kind: ``(n_tokens, n_clamp)`` int8 — clamp kind per
            (token, clampable-position).
        clamp_value: ``(n_tokens, n_clamp)`` float — target / offset.
        clamp_only_if_active: ``(n_tokens, n_clamp)`` bool — when
            True, suppress the clamp where the live ``f`` for that
            feature is ≤ 0.
        recon_mask: ``(n_tokens,)`` bool — per-token gate.  Tokens
            with ``recon_mask[t] == False`` pass through unchanged;
            tokens with ``recon_mask[t] == True`` have their residual
            replaced by ``decode(clamp(encode(h_t)))``.

    Returns:
        ``(n_tokens, d_model)`` tensor in the same dtype as
        ``hidden_states``.

    The compute path goes through :func:`apply_sae_full_reconstruction_op`
    (the registered torch custom op).  Calling it directly here —
    rather than via ``torch.ops.vllm.apply_sae_full_reconstruction``
    — keeps CPU-only test environments insulated from the registered
    dispatch key (which follows the platform: "CPU" on CPU-only,
    "CUDA" on a CUDA build).  The op-func branches on
    ``hidden_states.is_cuda`` to pick the right backend without
    going through the dispatcher.
    """
    n_tokens, d_model = hidden_states.shape
    d_sae = encoder_weight.shape[0]
    n_clamp = clampable_features.shape[0]

    if encoder_weight.shape != (d_sae, d_model):
        raise ValueError(
            "encoder_weight must be (d_sae, d_model); "
            f"got {tuple(encoder_weight.shape)} vs d_model={d_model}."
        )
    if encoder_bias.shape != (d_sae,):
        raise ValueError(
            "encoder_bias must be (d_sae,); "
            f"got {tuple(encoder_bias.shape)} vs d_sae={d_sae}."
        )
    if decoder_weight.shape != (d_sae, d_model):
        raise ValueError(
            "decoder_weight must be (d_sae, d_model) aligned with encoder; "
            f"got {tuple(decoder_weight.shape)} vs (d_sae={d_sae}, "
            f"d_model={d_model})."
        )
    if decoder_bias.shape != (d_model,):
        raise ValueError(
            "decoder_bias must be (d_model,); "
            f"got {tuple(decoder_bias.shape)} vs d_model={d_model}."
        )
    if clampable_features.dtype != torch.int64:
        raise ValueError(
            f"clampable_features must be int64; got dtype={clampable_features.dtype}."
        )
    if clampable_features.ndim != 1:
        raise ValueError(
            "clampable_features must be 1-D; "
            f"got shape={tuple(clampable_features.shape)}."
        )
    if recon_mask.shape != (n_tokens,):
        raise ValueError(
            "recon_mask must be (n_tokens,); "
            f"got {tuple(recon_mask.shape)} vs n_tokens={n_tokens}."
        )
    if recon_mask.dtype != torch.bool:
        raise ValueError(
            f"recon_mask must be torch.bool; got dtype={recon_mask.dtype}."
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
    if (
        n_tokens > 0
        and n_clamp > 0
        and (
            int(clampable_features.min().item()) < 0
            or int(clampable_features.max().item()) >= d_sae
        )
    ):
        raise ValueError(
            f"clampable_features must lie in [0, d_sae={d_sae}); "
            f"got min={int(clampable_features.min().item())}, "
            f"max={int(clampable_features.max().item())}."
        )

    code = _ACTIVATION_TO_CODE[activation]
    param = _activation_to_scalar(activation, activation_params)
    return apply_sae_full_reconstruction_op(
        hidden_states,
        encoder_weight,
        encoder_bias,
        decoder_weight,
        decoder_bias,
        clampable_features,
        clamp_kind,
        clamp_value,
        clamp_only_if_active,
        recon_mask,
        code,
        param,
    )


def apply_layer_sae_full_reconstruction(
    module: nn.Module,
    hidden_states: torch.Tensor,
    hook_point: SteeringHookPoint,
) -> torch.Tensor:
    """Layer-hook dispatch: pull buffer state, hand it to the registered op.

    When the layer has no full-reconstruction SAE buffers attached
    for ``hook_point`` (engine started with full-reconstruction
    disabled or this site isn't covered by any registered module),
    this short-circuits and returns ``hidden_states`` unchanged.
    The ``hasattr`` check is decided once at module instantiation
    and stays constant for the rest of the layer's lifetime, so
    ``torch.compile`` traces it as a static branch — the disabled
    path emits no kernel at all, mirroring
    :func:`apply_layer_steering` and :func:`apply_layer_sae_delta`.

    The shim performs a per-token gather from the row tables (keyed
    by the shared ``sae_recon_index`` buffer), derives
    ``recon_mask = (recon_index != 0)`` so row 0 is the no-op
    sentinel, and dispatches to
    ``torch.ops.vllm.apply_sae_full_reconstruction``.  The torch-op
    indirection is what makes :mod:`torch.compile` treat the call as
    an opaque splitting point.
    """
    if not sae_full_recon_buffers_attached(module, hook_point):
        return hidden_states
    n_tokens = hidden_states.shape[0]
    if n_tokens == 0:
        return hidden_states
    recon_index_full: torch.Tensor = module.sae_recon_index  # type: ignore[assignment]
    recon_index = recon_index_full[:n_tokens]
    recon_mask = recon_index != 0

    enc_w = getattr(module, HOOK_POINT_FR_ENCODER_WEIGHT_ATTR[hook_point])
    enc_b = getattr(module, HOOK_POINT_FR_ENCODER_BIAS_ATTR[hook_point])
    dec_w = getattr(module, HOOK_POINT_FR_DECODER_WEIGHT_ATTR[hook_point])
    dec_b = getattr(module, HOOK_POINT_FR_DECODER_BIAS_ATTR[hook_point])
    kind_table = getattr(module, HOOK_POINT_FR_CLAMP_KIND_ATTR[hook_point])
    value_table = getattr(module, HOOK_POINT_FR_CLAMP_VALUE_ATTR[hook_point])
    only_table = getattr(module, HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR[hook_point])
    clampable_features = getattr(
        module, HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR[hook_point]
    )
    activation: SAEActivation = getattr(
        module, HOOK_POINT_FR_ACTIVATION_ATTR[hook_point]
    )
    activation_params: dict[str, float] = getattr(
        module, HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR[hook_point]
    )

    # Per-token gather: (n_rows, n_clamp) → (n_tokens, n_clamp).
    clamp_kind = kind_table[recon_index]
    clamp_value = value_table[recon_index]
    clamp_only_if_active = only_table[recon_index]

    code = _ACTIVATION_TO_CODE[activation]
    param = _activation_to_scalar(activation, activation_params)
    return torch.ops.vllm.apply_sae_full_reconstruction(
        hidden_states,
        enc_w,
        enc_b,
        dec_w,
        dec_b,
        clampable_features,
        clamp_kind,
        clamp_value,
        clamp_only_if_active,
        recon_mask,
        code,
        param,
    )


# ``ACTIVATION_CODE_RELU`` re-exported so callers (tests, future
# kernel module) have a single import surface for the activation
# encoding without touching :mod:`sae_steering` directly.
__all__ = [
    "ACTIVATION_CODE_RELU",
    "CLAMP_KIND_ABSOLUTE",
    "CLAMP_KIND_ADDITIVE",
    "CLAMP_KIND_NONE",
    "HOOK_POINT_FR_ACTIVATION_ATTR",
    "HOOK_POINT_FR_ACTIVATION_PARAMS_ATTR",
    "HOOK_POINT_FR_CLAMPABLE_FEATURES_ATTR",
    "HOOK_POINT_FR_CLAMP_KIND_ATTR",
    "HOOK_POINT_FR_CLAMP_ONLY_IF_ACTIVE_ATTR",
    "HOOK_POINT_FR_CLAMP_VALUE_ATTR",
    "HOOK_POINT_FR_DECODER_BIAS_ATTR",
    "HOOK_POINT_FR_DECODER_WEIGHT_ATTR",
    "HOOK_POINT_FR_ENCODER_BIAS_ATTR",
    "HOOK_POINT_FR_ENCODER_WEIGHT_ATTR",
    "HOOK_POINT_FR_MODULE_NAME_ATTR",
    "apply_layer_sae_full_reconstruction",
    "apply_sae_full_reconstruction",
    "apply_sae_full_reconstruction_op",
    "apply_sae_full_reconstruction_op_fake",
    "register_sae_full_recon_buffers",
    "register_sae_recon_index_buffer",
    "sae_encode_full",
    "sae_full_recon_buffers_attached",
    "share_sae_recon_index_across_layers",
    "unregister_sae_full_recon_buffers",
]
