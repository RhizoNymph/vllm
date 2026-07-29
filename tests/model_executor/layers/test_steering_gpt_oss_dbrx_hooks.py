# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring tests for steering/capture hooks on gpt_oss and dbrx.

Both models previously lacked the residual-stream taps. These tests bypass the
(distributed-init) ``__init__``, stub the decoder's submodules with
deterministic callables, and drive the *real* ``forward`` to assert that
``pre_attn``/``post_attn``/``post_block`` fire at the right place with the right
tensor. Capture rides the steering helpers, so intercepting
``maybe_capture_residual`` (with no steering tables registered, the helpers are
pass-throughs) records exactly what each hook observes.
"""

import torch
import torch.nn as nn

from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers import steering as steering_mod
from vllm.model_executor.models import dbrx as dbrx_mod
from vllm.model_executor.models import gpt_oss as gpt_oss_mod

T, H = 4, 8
LAYER_IDX = 3


def _install_recorder(monkeypatch):
    rec: dict[str, tuple[torch.Tensor, int]] = {}

    def _record(tensor, layer_idx, hook_name):
        rec[hook_name] = (tensor.detach().clone(), layer_idx)

    def _record_add(a, b, layer_idx, hook_name):
        # ``apply_block_steering`` captures the block output via the
        # summand-preserving helper (the sum is formed inside the op to
        # survive torch.compile DCE); record the semantic value it captures.
        rec[hook_name] = ((a + b).detach().clone(), layer_idx)

    monkeypatch.setattr(steering_mod, "maybe_capture_residual", _record)
    monkeypatch.setattr(steering_mod, "maybe_capture_residual_add", _record_add)
    monkeypatch.setattr(cap_mod, "get_active_capture_manager", lambda: object())
    return rec


def _new(cls):
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer.layer_idx = LAYER_IDX
    return layer


def _record_input(sink, key, ret):
    def _fn(*args, **kwargs):
        sink[key] = kwargs.get("hidden_states", args[0] if args else None)
        return ret

    return _fn


def test_gpt_oss_steering_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch)
    layer = _new(gpt_oss_mod.TransformerBlock)

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.attn = lambda h, positions: attn_out
    layer.post_attention_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", mlp_out)

    out, res = layer.forward(hidden_states=x, positions=torch.arange(T), residual=None)

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    assert rec["pre_attn"][1] == LAYER_IDX
    torch.testing.assert_close(rec["pre_attn"][0], x)
    torch.testing.assert_close(rec["post_attn"][0], residual_after)
    torch.testing.assert_close(rec["post_block"][0], residual_after + mlp_out)
    # No tables registered -> tensors flow through unchanged.
    torch.testing.assert_close(out, mlp_out)
    torch.testing.assert_close(res, residual_after)


def test_dbrx_steering_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch)
    layer = _new(dbrx_mod.DbrxBlock)

    x = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    ffn_out = torch.randn(T, H)

    seen: dict = {}
    layer.norm_attn_norm = lambda position_ids, hidden_states: (mlp_in, residual_after)
    layer.ffn = _record_input(seen, "ffn_arg", ffn_out)

    out = layer.forward(position_ids=torch.arange(T), hidden_states=x)

    torch.testing.assert_close(seen["ffn_arg"], mlp_in)
    assert rec["pre_attn"][1] == LAYER_IDX
    torch.testing.assert_close(rec["pre_attn"][0], x)
    torch.testing.assert_close(rec["post_attn"][0], residual_after)
    torch.testing.assert_close(rec["post_block"][0], residual_after + ffn_out)
    # DBRX adds the branch explicitly: block output == ffn_out + residual.
    torch.testing.assert_close(out, ffn_out + residual_after)
