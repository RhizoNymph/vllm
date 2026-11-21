# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steering, patching, clamps, SAE and capture for the Qwen4Exp decoder.

Qwen4Exp has no single residual stream. It carries ``hc_count``
hyper-connection (HC) streams of ``hidden_size`` each, flattened stream-major
as ``[T, hc_count * hidden_size]``. Each sublayer reads a gated mean of the
per-stream-normalized streams (the HC *mix*) and writes its output back to
every stream through a learned per-stream injection weight (the HC
*combine*). The NVIDIA and ROCm layers defer each combine into the next mix,
so a layer returns ``(streams, pending_branch, injection_logits)``.

The hook points are defined as follows. The NVIDIA and ROCm files call the
same helpers, so the two platforms cannot drift.

* ``pre_attn`` / ``post_attn`` / ``post_block`` (``hidden_size`` wide) take a
  *stream-mean view*. Capture and the editing families (patch, additive,
  SAE, clamp) see the mean over streams of the carried state. Whatever edit
  they make to that view is added, as the same delta, to every stream. An
  additive vector ``v`` therefore adds ``v`` to each stream and moves the
  captured mean by exactly ``v``, so a captured difference of means can be
  fed back as a steering vector. At ``hc_count == 1`` this reduces to
  ordinary residual steering.
* ``mlp_in`` / ``mlp_out`` are the single-stream MoE input (the HC mix
  output) and the MoE output before the HC combine. These are the same
  branch tensors as on standard models.
* ``mhc_streams_pre_attn`` / ``mhc_streams_pre_mlp`` / ``mhc_streams_final``
  (``hc_count * hidden_size`` wide) are the full multi-stream state. Names,
  widths and layout match DeepSeek-V4, so a vector made of ``v`` repeated
  ``hc_count`` times at ``mhc_streams_pre_attn`` steers the same as ``v`` at
  ``pre_attn``. ``mhc_streams_final`` is model-level and registered only on
  the last decoder layer.

On timing, the residual and stream hooks before each sublayer run after
that sublayer's HC mix has consumed the streams. This matches standard
models, where ``pre_attn`` steers the carried residual after
``input_layernorm`` has read it. An edit there reaches the residual carried
past the sublayer, not the sublayer's own input. To steer what the MoE
reads, use ``mlp_in``.

``post_block`` and ``mhc_streams_final`` observe the block output
``combine(streams, branch, injection)``. The NVIDIA and ROCm layers never
materialize it here: the next layer's fused combine+norm does. These hooks
compute it only when something reads or edits it, and fold edits back into
the pre-combine ``streams``. The combine is affine in ``streams``, so
``combine(streams + d) == combine(streams) + d`` and the fused kernels stay
in place.

Derived views (the stream mean and the block output) are captured through
``qwen4_exp_capture_hc_view``. That op forms the view inside the op, anchored
on the live ``streams`` tensor. A view formed outside the op would be dead in
the ROCm ``torch.compile`` graph whenever nothing edits it, and the capture
would be dropped, the same hazard ``maybe_capture_residual_add`` guards
against for ``post_block`` on standard models.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation_capture import (
    capture_hook_id,
    capture_hook_name,
    get_active_capture_manager,
    maybe_capture_residual,
)
from vllm.model_executor.layers.clamp import CLAMP_DIRS_ATTR
from vllm.model_executor.layers.intervention_common import interventions_configured
from vllm.model_executor.layers.patch import PATCH_TABLE_ATTR
from vllm.model_executor.layers.steering import (
    HOOK_POINT_SAE_FR_CLAMP_KIND_ATTR,
    HOOK_POINT_SAE_SLOTS_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
    apply_layer_interventions,
    apply_layer_steering,
    get_steering_buffer_config,
    get_steering_buffer_dtype,
    register_steering_buffers,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .hyperconnection import is_hc_sequence_parallel_enabled

_HP = SteeringHookPoint

# Hidden-width hooks whose view is the mean over HC streams of the carried
# multi-stream state; edits broadcast to every stream.
STREAM_MEAN_HOOKS: frozenset[SteeringHookPoint] = frozenset(
    {_HP.PRE_ATTN, _HP.POST_ATTN, _HP.POST_BLOCK}
)
# Hidden-width hooks on the single-stream MoE branch.
BRANCH_HOOKS: frozenset[SteeringHookPoint] = frozenset({_HP.MLP_IN, _HP.MLP_OUT})
# Full-width hooks on the multi-stream state, laid out stream-major.
STREAM_HOOKS: frozenset[SteeringHookPoint] = frozenset(
    {
        _HP.MHC_STREAMS_PRE_ATTN,
        _HP.MHC_STREAMS_PRE_MLP,
        _HP.MHC_STREAMS_FINAL,
    }
)


class Qwen4ExpInterventionConfigError(ValueError):
    """An intervention is configured on a Qwen4Exp layout that cannot honour it."""


@dataclass(frozen=True, slots=True)
class HCStreamGeometry:
    """Hyper-connection geometry of a decoder layer whose hooks are live.

    A layer stores this as ``hc_interventions``, or ``None`` when its hooks
    are inert: MTP drafter layers, or a server with no intervention
    configured. ``None`` means every helper below returns its input object
    unchanged, without capturing.
    """

    hc_count: int
    hidden_size: int

    def __post_init__(self) -> None:
        if self.hc_count < 1 or self.hidden_size < 1:
            raise ValueError(
                f"invalid HC geometry: hc_count={self.hc_count}, "
                f"hidden_size={self.hidden_size}"
            )

    @property
    def stream_width(self) -> int:
        return self.hc_count * self.hidden_size


def is_qwen4_exp_mtp_layer(layer_idx: int, num_hidden_layers: int) -> bool:
    """MTP drafter layers are numbered after the backbone's layers."""
    return layer_idx >= num_hidden_layers


def qwen4_exp_hook_widths(
    geometry: HCStreamGeometry, *, is_last_layer: bool
) -> dict[SteeringHookPoint, int]:
    """Per-hook buffer widths for one Qwen4Exp decoder layer."""
    widths = {hp: geometry.hidden_size for hp in STREAM_MEAN_HOOKS | BRANCH_HOOKS}
    widths[_HP.MHC_STREAMS_PRE_ATTN] = geometry.stream_width
    widths[_HP.MHC_STREAMS_PRE_MLP] = geometry.stream_width
    if is_last_layer:
        widths[_HP.MHC_STREAMS_FINAL] = geometry.stream_width
    return widths


def init_qwen4_exp_interventions(
    layer: nn.Module,
    vllm_config: VllmConfig,
    *,
    layer_idx: int,
    num_hidden_layers: int,
    hidden_size: int,
    hc_count: int,
) -> HCStreamGeometry | None:
    """Register a decoder layer's buffers and return its hook geometry.

    Returns ``None`` (and registers nothing) for MTP drafter layers. Other
    MTP drafters in this fork carry no hooks of their own either. Also
    returns ``None`` when no intervention is configured, so those servers
    run the upstream forward unchanged.

    Raises :class:`Qwen4ExpInterventionConfigError` under HC sequence
    parallelism. There the carried streams and ``mlp_out`` hold a rank-local
    shard of the token axis, while ``steering_index``, patch slots and
    capture rows are indexed by global token position.
    """
    if is_qwen4_exp_mtp_layer(layer_idx, num_hidden_layers):
        return None
    if not interventions_configured(vllm_config):
        return None
    if is_hc_sequence_parallel_enabled(vllm_config):
        raise Qwen4ExpInterventionConfigError(
            "Qwen4Exp steering, patching and capture need the full token axis "
            "on every rank, but HC sequence parallelism shards it "
            "(enable_hc_sp or sequence-parallel MoE with TP>1). Disable both, "
            "or drop the steering/patch/capture config."
        )
    geometry = HCStreamGeometry(hc_count=hc_count, hidden_size=hidden_size)
    max_steering_tokens, max_steering_configs = get_steering_buffer_config(vllm_config)
    register_steering_buffers(
        layer,
        hidden_size,
        max_steering_tokens=max_steering_tokens,
        max_steering_configs=max_steering_configs,
        dtype=get_steering_buffer_dtype(vllm_config),
        hook_widths=qwen4_exp_hook_widths(
            geometry, is_last_layer=layer_idx == num_hidden_layers - 1
        ),
    )
    return geometry


def has_layer_interventions(module: nn.Module, hook_point: SteeringHookPoint) -> bool:
    """Whether any editing family has buffers at ``hook_point`` on ``module``.

    Mirrors the gates of :func:`apply_layer_interventions` so derived views
    are formed only when something edits them. SAE buffers can attach after
    model construction, so this is evaluated per call.
    """
    if (
        hasattr(module, HOOK_POINT_TABLE_ATTR[hook_point])
        or hasattr(module, PATCH_TABLE_ATTR[hook_point])
        or hasattr(module, CLAMP_DIRS_ATTR[hook_point])
    ):
        return True
    slots_attr = HOOK_POINT_SAE_SLOTS_ATTR.get(hook_point)
    if slots_attr is not None and hasattr(module, slots_attr):
        return True
    fr_attr = HOOK_POINT_SAE_FR_CLAMP_KIND_ATTR.get(hook_point)
    return fr_attr is not None and fr_attr in module._buffers


def hc_residual_view(
    streams: torch.Tensor,
    block_output: torch.Tensor | None,
    injection_logits: torch.Tensor | None,
    hc_count: int,
    *,
    reduce_mean: bool,
) -> torch.Tensor:
    """The tensor a Qwen4Exp residual or stream hook observes.

    ``streams`` is the carried ``[T, hc_count * H]`` state. With
    ``block_output`` (``[T, H]``) the view is the block output
    ``streams + block_output * 2 * sigmoid(injection_logits / hc_count)`` per
    stream, which is the HC combine the next fused combine+norm will compute.
    A ``None`` injection means unit weight, as in the kernels. With
    ``reduce_mean`` the view is then averaged over streams to ``[T, H]``.

    Arithmetic is fp32 and the result is cast to ``streams.dtype``. With
    ``block_output=None`` and ``reduce_mean=False`` the view is ``streams``
    itself, not a copy.
    """
    if block_output is None and not reduce_mean:
        return streams
    per_stream = streams.unflatten(-1, (hc_count, -1)).float()
    if block_output is None:
        return per_stream.mean(dim=-2).to(streams.dtype)
    block = block_output.float()
    if injection_logits is None:
        weight = torch.ones(
            (*per_stream.shape[:-1],), dtype=torch.float32, device=streams.device
        )
    else:
        weight = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    if reduce_mean:
        mean = per_stream.mean(dim=-2) + block * weight.mean(dim=-1, keepdim=True)
        return mean.to(streams.dtype)
    combined = per_stream + block.unsqueeze(-2) * weight.unsqueeze(-1)
    return combined.flatten(-2).to(streams.dtype)


def _capture_hc_view_impl(
    streams: torch.Tensor,
    block_output: torch.Tensor | None,
    injection_logits: torch.Tensor | None,
    hc_count: int,
    reduce_mean: bool,
    layer_idx: int,
    hook_id: int,
) -> None:
    """Real impl of ``torch.ops.vllm.qwen4_exp_capture_hc_view``.

    Forms :func:`hc_residual_view` inside the op and hands it to the active
    capture manager. ``streams`` is never written; see the registration
    below for the ``mutates_args`` anchor.
    """
    mgr = get_active_capture_manager()
    if mgr is None:
        return
    view = hc_residual_view(
        streams, block_output, injection_logits, hc_count, reduce_mean=reduce_mean
    )
    mgr.on_hook(layer_idx, capture_hook_name(hook_id), view)


def _capture_hc_view_fake(
    streams: torch.Tensor,
    block_output: torch.Tensor | None,
    injection_logits: torch.Tensor | None,
    hc_count: int,
    reduce_mean: bool,
    layer_idx: int,
    hook_id: int,
) -> None:
    return None


# ``mutates_args=["streams"]`` is the same deliberate white lie as
# ``capture_residual``: the impl reads ``streams`` but never writes it. The
# annotation marks a side effect so ``torch.compile`` keeps the op even though
# it returns nothing. ``streams`` flows on into the next layer, so the op is
# anchored on a live value.
direct_register_custom_op(
    op_name="qwen4_exp_capture_hc_view",
    op_func=_capture_hc_view_impl,
    fake_impl=_capture_hc_view_fake,
    mutates_args=["streams"],
)


def intervene_hc_residual(
    layer: nn.Module,
    streams: torch.Tensor,
    hook_point: SteeringHookPoint,
    *,
    block_output: torch.Tensor | None = None,
    injection_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capture, then edit, a residual or stream hook; return the new ``streams``.

    ``hook_point`` must be in :data:`STREAM_MEAN_HOOKS` or
    :data:`STREAM_HOOKS`. ``block_output`` / ``injection_logits`` name a
    pending HC combine; the hook then observes the block output (see
    :func:`hc_residual_view`). The edit ``edited - view`` is added to
    ``streams``, broadcast over streams for the mean hooks. When the view is
    ``streams`` itself, the edited tensor is returned directly.
    """
    geometry: HCStreamGeometry | None = layer.hc_interventions
    if geometry is None:
        return streams
    if hook_point in STREAM_MEAN_HOOKS:
        reduce_mean = True
    elif hook_point in STREAM_HOOKS:
        reduce_mean = False
    else:
        raise ValueError(f"{hook_point.value} is not a Qwen4Exp residual hook")
    identity_view = block_output is None and not reduce_mean
    if get_active_capture_manager() is not None:
        if identity_view:
            maybe_capture_residual(streams, layer.layer_idx, hook_point.value)
        else:
            torch.ops.vllm.qwen4_exp_capture_hc_view(
                streams,
                block_output,
                injection_logits,
                geometry.hc_count,
                reduce_mean,
                layer.layer_idx,
                capture_hook_id(hook_point.value),
            )
    if not has_layer_interventions(layer, hook_point):
        return streams
    view = hc_residual_view(
        streams,
        block_output,
        injection_logits,
        geometry.hc_count,
        reduce_mean=reduce_mean,
    )
    edited = apply_layer_interventions(layer, view, hook_point)
    if identity_view:
        return edited
    if edited is view:
        return streams
    delta = edited.float() - view.float()
    if reduce_mean:
        per_stream = streams.unflatten(-1, (geometry.hc_count, -1)).float()
        return (per_stream + delta.unsqueeze(-2)).flatten(-2).to(streams.dtype)
    return (streams.float() + delta).to(streams.dtype)


def intervene_sublayer_input(
    layer: nn.Module, streams: torch.Tensor, sublayer: str
) -> torch.Tensor:
    """The two hooks on the streams a sublayer's HC mix just consumed.

    ``sublayer`` is ``"attn"`` (``mhc_streams_pre_attn`` then ``pre_attn``)
    or ``"mlp"`` (``mhc_streams_pre_mlp`` then ``post_attn``). The full-width
    hook runs first, so the mean hook observes its edit.
    """
    if sublayer == "attn":
        stream_hp, mean_hp = _HP.MHC_STREAMS_PRE_ATTN, _HP.PRE_ATTN
    elif sublayer == "mlp":
        stream_hp, mean_hp = _HP.MHC_STREAMS_PRE_MLP, _HP.POST_ATTN
    else:
        raise ValueError(f"unknown Qwen4Exp sublayer {sublayer!r}")
    streams = intervene_hc_residual(layer, streams, stream_hp)
    return intervene_hc_residual(layer, streams, mean_hp)


def intervene_branch(
    layer: nn.Module, hidden_states: torch.Tensor, hook_point: SteeringHookPoint
) -> torch.Tensor:
    """``mlp_in`` / ``mlp_out`` on the single-stream MoE branch."""
    if layer.hc_interventions is None:
        return hidden_states
    if hook_point not in BRANCH_HOOKS:
        raise ValueError(f"{hook_point.value} is not a Qwen4Exp branch hook")
    return apply_layer_steering(layer, hidden_states, hook_point)


def intervene_block_output(
    layer: nn.Module,
    streams: torch.Tensor,
    mlp_out: torch.Tensor,
    injection_logits: torch.Tensor | None,
) -> torch.Tensor:
    """``post_block``: the layer's output streams, still pending their combine."""
    return intervene_hc_residual(
        layer,
        streams,
        _HP.POST_BLOCK,
        block_output=mlp_out,
        injection_logits=injection_logits,
    )


def intervene_final_streams(
    last_layer: nn.Module,
    streams: torch.Tensor,
    block_output: torch.Tensor | None,
    injection_logits: torch.Tensor | None,
) -> torch.Tensor:
    """``mhc_streams_final``: the multi-stream state the final HC mixer reads.

    Called on the last PP rank with the module of the last decoder layer,
    which owns the hook's buffers.
    """
    return intervene_hc_residual(
        last_layer,
        streams,
        _HP.MHC_STREAMS_FINAL,
        block_output=block_output,
        injection_logits=injection_logits,
    )
