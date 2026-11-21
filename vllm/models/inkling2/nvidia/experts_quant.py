# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on a vLLM quantization method (`QUANT_METHOD`).

The expert weights live in a standard `RoutedExperts`, so the quantization
config (GGUF, or none for BF16) supplies `create_weights` / `apply` as for
any FusedMoE model. Inkling keeps its own router (sigmoid + gate bias top-k,
renormalized and scaled by `route_scale`) and passes the selection to
`quant_method.apply`; the experts' softcapped SwiGLU is vLLM's `SITU`
activation with `beta = mlp_gate_softcap` and
`linear_beta = mlp_up_softcap`.

Contract: `notes/inkling2-expert-interface.md`.
"""

from collections.abc import Sequence

import regex as re
import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.layer import make_parallel_config
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .moe import ExpertsLoRA, Inkling2Gate, Inkling2RoutingCapture

# Per-expert checkpoint names: `...mlp.experts.<e>.<proj>.<suffix>`.
_PER_EXPERT = re.compile(
    r"^(?P<base>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>\w+)$"
)
# The official fused names, accepted with floating weights (BF16 reference).
_FUSED = re.compile(r"^(?P<base>.*\.mlp\.experts)\.(?P<proj>w13|w2)_weight$")
_SHARDS = {
    "gate_proj": ("w13", "w1"),
    "up_proj": ("w13", "w3"),
    "down_proj": ("w2", "w2"),
}


def situ_reference(
    gate_up: torch.Tensor, gate_cap: float, up_cap: float
) -> torch.Tensor:
    """fp32 reference of the experts' activation on `[gate | up]` rows:
    `softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap)` (a cap of 0
    disables it), which is vLLM's `SITU` with `beta = gate_cap` and
    `linear_beta = up_cap`."""
    g, u = gate_up.float().chunk(2, dim=-1)
    sig = torch.sigmoid(g)
    if gate_cap > 0:
        g = gate_cap * torch.tanh(g / gate_cap)
    if up_cap > 0:
        u = up_cap * torch.tanh(u / up_cap)
    return g * sig * u


def route_reference(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    topk: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference of the Inkling2 router (`ops.moe._select_experts`):
    int32 ids `[T, topk]` (top-k of `sigmoid(logits) + bias`) and fp32
    weights `[T, topk]` (their sigmoids, renormalized, times
    `route_scale`)."""
    logits = x.float() @ gate_weight.float().t()
    probs = torch.sigmoid(logits)
    ids = torch.topk(probs + gate_bias.float(), topk, dim=-1).indices
    selected = probs.gather(1, ids)
    weights = selected / selected.sum(-1, keepdim=True) * route_scale
    return ids.to(torch.int32), weights


def make_routed_experts(
    config, quant_config: QuantizationConfig | None, prefix: str
) -> RoutedExperts:
    """A `RoutedExperts` holding the experts of one Inkling2 MoE layer, with
    its quantization method resolved from `quant_config` (unquantized if
    None). Built without a `MoERunner`: Inkling routes and calls
    `quant_method.apply` itself."""
    vllm_config = get_current_vllm_config()
    parallel = make_parallel_config(
        tp_size=None,
        dp_size=None,
        pcp_size=None,
        is_sequence_parallel=False,
        parallel_config=vllm_config.parallel_config,
    )
    num_experts, topk = config.n_routed_experts, config.num_experts_per_tok
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    expert_map = ExpertMapManager(
        max_num_batched_tokens=max_tokens,
        top_k=topk,
        global_num_experts=num_experts,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy=vllm_config.parallel_config.expert_placement_strategy,
        enable_eplb=False,
    )
    model_config = vllm_config.model_config
    dtype = model_config.dtype if model_config is not None else torch.bfloat16
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_local_experts=expert_map.local_num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SITU,
        device=vllm_config.device_config.device,
        routing_method=RoutingMethodType.Custom,
        moe_parallel_config=parallel,
        in_dtype=dtype,
        moe_backend=vllm_config.kernel_config.moe_backend,
        max_num_tokens=max_tokens,
        is_lora_enabled=vllm_config.lora_config is not None,
        require_decomposed_backend=True,
        activation_situ_beta=config.mlp_gate_softcap,
        activation_situ_linear_beta=config.mlp_up_softcap,
        max_capture_size=vllm_config.compilation_config.max_cudagraph_capture_size or 0,
    )
    return RoutedExperts(
        prefix,
        dtype,
        moe_config,
        quant_config,
        expert_map_manager=expert_map,
        # The router weights are already scaled by route_scale.
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
    )


class Inkling2QuantMoE(nn.Module):
    """`Inkling2MoE` for the `QUANT_METHOD` backend (TP = 1): the Inkling2
    gate and router, and the experts of a vLLM quantization method."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
        visit_ids: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        if config.n_shared_experts != 0:
            raise ValueError("Inkling2 has no shared experts")
        if config.mlp_gate_softcap <= 0:
            # vLLM's SITU needs a gate softcap; without one the activation
            # is plain SiLU-and-mul with a capped up.
            raise ValueError("Inkling2 quant-method experts need mlp_gate_softcap > 0")
        self.topk = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate = Inkling2Gate(config)
        self.experts = make_routed_experts(config, quant_config, f"{prefix}.experts")
        self.routing_captures = nn.ModuleList(
            Inkling2RoutingCapture(visit) for visit in visit_ids
        )
        self.register_buffer(
            "all_experts",
            torch.tensor([0, config.n_routed_experts], dtype=torch.int32),
            persistent=False,
        )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """int32 expert ids and fp32 routing weights `[T, topk]`."""
        gate = self.gate
        if x.is_cuda:
            from .ops.moe_sm90 import select_experts

            ids, weights, _ = select_experts(
                x, gate.weight, gate.bias, self.topk, self.route_scale, self.all_experts
            )
            return ids, weights.view(-1, self.topk)
        return route_reference(x, gate.weight, gate.bias, self.topk, self.route_scale)

    def routed_expert_ids(self, x: torch.Tensor) -> torch.Tensor:
        return self.route(x.contiguous())[0]

    def forward(
        self,
        x: torch.Tensor,
        reduce: bool = True,
        push=None,
        lora: ExpertsLoRA | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """bf16 `[T, H]` -> bf16 `[T, H]` (TP = 1, so always reduced)."""
        if push is not None or not reduce:
            raise NotImplementedError(
                "Inkling2 quant-method experts return the reduced output only"
            )
        if lora is not None:
            raise NotImplementedError("Inkling2 expert LoRA is not supported here")
        x = x.contiguous()
        ids, weights = self.route(x)
        capture_fn = self.routing_captures[pass_idx].capture_fn
        if capture_fn is not None:
            capture_fn(ids)
        experts = self.experts
        out = experts.quant_method.apply(
            layer=experts,
            x=x,
            topk_weights=weights,
            topk_ids=ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"{type(experts.quant_method).__name__}.apply returned "
                f"{type(out).__name__}; Inkling2 needs the finalized tensor"
            )
        return out


def load_quant_expert_weight(
    params: dict[str, nn.Parameter], name: str, weight: torch.Tensor
) -> str | None:
    """Load one routed-expert checkpoint tensor of the `QUANT_METHOD`
    backend into its `RoutedExperts` parameter. Returns the parameter name,
    or None if `name` is not a routed-expert tensor of this backend."""
    m = _PER_EXPERT.match(name)
    if m is not None:
        fused, shard_id = _SHARDS[m["proj"]]
        param_name = f"{m['base']}.{fused}_{m['suffix']}"
        if param_name not in params:
            return None
        param = params[param_name]
        param.weight_loader(
            param, weight, name, shard_id=shard_id, expert_id=int(m["expert"])
        )
        return param_name
    m = _FUSED.match(name)
    if m is None or not weight.is_floating_point():
        return None
    param_name = name
    if param_name not in params:
        return None
    param = params[param_name]
    if m["proj"] == "w13":
        # Official layout: interleaved (gate, up) rows.
        for e in range(weight.shape[0]):
            for shard_id, rows in (("w1", weight[e, 0::2]), ("w3", weight[e, 1::2])):
                param.weight_loader(param, rows, name, shard_id=shard_id, expert_id=e)
    else:
        for e in range(weight.shape[0]):
            param.weight_loader(param, weight[e], name, shard_id="w2", expert_id=e)
    return param_name


def is_unquantized(experts: RoutedExperts) -> bool:
    return isinstance(experts.quant_method, UnquantizedFusedMoEMethod)
