# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp steering and capture through the real HC kernels (single GPU).

The CPU wiring tests in
``tests/model_executor/models/test_qwen4_exp_interventions.py`` use
stand-in HC modules. This test builds the production
``Qwen4ExpDecoderLayer``, including its Triton HC mix/combine kernels and
its MoE block, with a tiny config and a steering config. It then checks the
hook semantics against the layer's own HC modules:

* registered but inactive steering is a bitwise no-op;
* a ``pre_attn`` vector lands on every carried stream;
* a ``post_block`` vector survives the deferred HC combine;
* the ``pre_attn`` / ``post_block`` captures are the stream means.

Linear attention is replaced by a plain projection, as in ``test_tpsp.py``,
so no attention metadata is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Qwen4Exp requires CUDA or ROCm", allow_module_level=True)

from vllm.config import (
    EngramConfig,
    ParallelConfig,
    SteeringConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers.steering import (
    HOOK_POINT_ANY_ACTIVE_ATTR,
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

if current_platform.is_rocm():
    from vllm.models.qwen4_exp.amd import model as qwen4_model
else:
    from vllm.models.qwen4_exp.nvidia import (  # type: ignore[no-redef]
        model as qwen4_model,
    )

HP = SteeringHookPoint
HC, H, T = 2, 128, 5


class _Projection(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class _CaptureRecorder:
    def __init__(self) -> None:
        self.seen: dict[str, torch.Tensor] = {}

    def on_hook(self, layer_idx, hook_name, hidden_states):
        self.seen[hook_name] = hidden_states.clone()


def _make_config() -> VllmConfig:
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=1))
    config.model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        hf_text_config=Qwen4ExpTextConfig(
            vocab_size=128,
            eos_token_id=0,
            hidden_size=H,
            hc_count=HC,
            hc_lowrank=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            ple_layer_ids=[],
        ),
        dtype=torch.bfloat16,
    )
    config.engram_config = EngramConfig(cpu_offload=False)
    config.steering_config = SteeringConfig(max_steering_configs=1)
    return config


def _make_layer(config: VllmConfig):
    with (
        set_current_vllm_config(config),
        patch.object(qwen4_model, "QwenGatedDeltaNetAttention", _Projection),
    ):
        layer = qwen4_model.Qwen4ExpDecoderLayer(
            config, "linear_attention", "model.layers.0"
        )
        torch.manual_seed(7)
        for param in layer.parameters():
            param.normal_(std=0.08)
        experts = layer.mlp.experts.routed_experts
        experts.quant_method.process_weights_after_loading(experts)
    return layer


def _set_vector(layer, hook, vec) -> None:
    table = getattr(layer, HOOK_POINT_TABLE_ATTR[hook])
    table[3] = vec.to(table.dtype)
    layer.steering_index[:T] = 3
    getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hook]).fill_(True)


def _clear(layer) -> None:
    for hook, attr in HOOK_POINT_TABLE_ATTR.items():
        if hasattr(layer, attr):
            getattr(layer, attr).zero_()
            getattr(layer, HOOK_POINT_ANY_ACTIVE_ATTR[hook]).fill_(False)


@pytest.fixture(scope="module")
def runtime():
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
    )
    init_workspace_manager(torch.device("cuda:0"))
    config = _make_config()
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=1)
        with torch.device("cuda:0"), torch.inference_mode():
            torch.set_default_dtype(torch.bfloat16)
            yield config, _make_layer(config)
    torch.set_default_dtype(torch.float32)
    reset_workspace_manager()
    cleanup_dist_env_and_memory()


def _run(config, layer, streams, block, logits):
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config, num_tokens=T),
    ):
        out = layer(
            streams,
            block,
            logits,
            torch.arange(T),
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        # A later MoE call may reuse workspace-backed output storage.
        return tuple(x.clone() for x in out)


def _inputs():
    torch.manual_seed(11)
    return (
        torch.randn(T, HC * H),
        torch.randn(T, H),
        torch.randn(T, HC),
    )


def test_registers_qwen4_exp_hook_widths(runtime):
    _, layer = runtime
    assert layer.hc_interventions is not None
    widths = {
        hp: getattr(layer, attr).shape[1]
        for hp, attr in HOOK_POINT_TABLE_ATTR.items()
        if hasattr(layer, attr)
    }
    assert widths[HP.PRE_ATTN] == H and widths[HP.MLP_OUT] == H
    assert widths[HP.MHC_STREAMS_PRE_ATTN] == HC * H
    # num_hidden_layers == 1, so layer 0 also owns the final-streams hook.
    assert widths[HP.MHC_STREAMS_FINAL] == HC * H


def test_inactive_and_zero_steering_is_bitwise_noop(runtime):
    config, layer = runtime
    _clear(layer)
    inputs = _inputs()
    geometry = layer.hc_interventions
    layer.hc_interventions = None
    try:
        baseline = _run(config, layer, *inputs)
    finally:
        layer.hc_interventions = geometry
    inactive = _run(config, layer, *inputs)
    _set_vector(layer, HP.PRE_ATTN, torch.zeros(H))
    _set_vector(layer, HP.POST_BLOCK, torch.zeros(H))
    zero = _run(config, layer, *inputs)
    _clear(layer)
    for got in (inactive, zero):
        for g, b in zip(got, baseline):
            assert torch.equal(g, b)


def test_pre_attn_and_post_block_vectors_through_real_kernels(runtime):
    config, layer = runtime
    _clear(layer)
    inputs = _inputs()
    base_streams, base_mlp, base_inj = _run(config, layer, *inputs)

    vec = torch.randn(H) * 0.5
    _set_vector(layer, HP.PRE_ATTN, vec)
    streams, _, _ = _run(config, layer, *inputs)
    _clear(layer)
    shift = (streams.float() - base_streams.float()).unflatten(-1, (HC, H))
    torch.testing.assert_close(shift, vec.float().expand(T, HC, H), atol=0.1, rtol=0.02)

    _set_vector(layer, HP.POST_BLOCK, vec)
    streams, mlp_out, inj = _run(config, layer, *inputs)
    _clear(layer)
    torch.testing.assert_close(mlp_out, base_mlp, atol=0, rtol=0)
    combine = layer.mlp_hyper_connection.combine
    delta = (
        combine(streams, mlp_out, inj).float()
        - combine(base_streams, base_mlp, base_inj).float()
    )
    torch.testing.assert_close(
        delta.unflatten(-1, (HC, H)),
        vec.float().expand(T, HC, H),
        atol=0.1,
        rtol=0.02,
    )


def test_residual_captures_are_stream_means(runtime):
    config, layer = runtime
    _clear(layer)
    streams, block, logits = _inputs()
    rec = _CaptureRecorder()
    cap_mod.set_active_capture_manager(rec)
    try:
        out_streams, mlp_out, inj = _run(config, layer, streams, block, logits)
    finally:
        cap_mod.set_active_capture_manager(None)

    with set_current_vllm_config(config):
        pre_attn, _, _ = layer.attn_hyper_connection.combine_and_mix(
            streams, block, logits
        )
        block_out = layer.mlp_hyper_connection.combine(out_streams, mlp_out, inj)
    torch.testing.assert_close(rec.seen["mhc_streams_pre_attn"], pre_attn)
    torch.testing.assert_close(
        rec.seen["pre_attn"].float(),
        pre_attn.float().unflatten(-1, (HC, H)).mean(dim=-2),
        atol=0.02,
        rtol=0.02,
    )
    torch.testing.assert_close(
        rec.seen["post_block"].float(),
        block_out.float().unflatten(-1, (HC, H)).mean(dim=-2),
        atol=0.05,
        rtol=0.02,
    )
    torch.testing.assert_close(rec.seen["mlp_out"], mlp_out)
