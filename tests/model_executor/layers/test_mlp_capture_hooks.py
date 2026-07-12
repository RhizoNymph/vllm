# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring tests for the ``mlp_in``/``mlp_out`` capture hooks.

These taps feed transcoder training: ``mlp_in`` is the normed activation fed
into the MLP/MoE sublayer and ``mlp_out`` is the branch that sublayer writes
back to the residual stream. They are wired into the gemma3, gemma4, and
qwen3-family decoder layers.

Each test bypasses the (heavy, distributed-init) ``__init__`` and stubs the
decoder's submodules with deterministic callables, then drives the *real*
``forward`` to assert the taps fire at the right place with the right tensor.
``maybe_capture_residual`` (reached via the steering helpers, plus gemma4's
direct calls) is intercepted into a per-hook recorder; with no steering
tables registered the steering helpers are pass-throughs.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers import activation_capture as cap_mod
from vllm.model_executor.layers import steering as steering_mod
from vllm.model_executor.models import gemma3 as gemma3_mod
from vllm.model_executor.models import gemma4 as gemma4_mod
from vllm.model_executor.models import qwen3 as qwen3_mod
from vllm.model_executor.models import qwen3_moe as qwen3_moe_mod

try:
    # qwen3_next backs Qwen3.5 (Qwen3_5DecoderLayer subclasses it). On some
    # checkouts it is unimportable due to an unrelated mamba-refactor import
    # breakage; skip rather than fail the whole module in that case.
    from vllm.model_executor.models import qwen3_next as qwen3_next_mod

    _QWEN3_NEXT_IMPORT_ERR: str | None = None
except Exception as exc:  # noqa: BLE001 - surfaced as a skip reason
    qwen3_next_mod = None
    _QWEN3_NEXT_IMPORT_ERR = repr(exc)

T, H = 4, 8
LAYER_IDX = 3


def _install_recorder(monkeypatch, model_mod):
    """Record every ``maybe_capture_residual`` call keyed by hook name.

    Patches the name in both the model module (direct ``mlp_in``/``mlp_out``
    calls) and the steering module (``pre_attn``/``post_attn``/``post_block``
    via the steering helpers), and forces the capture gate on so
    ``apply_block_steering`` takes its capturing branch.
    """
    rec: dict[str, tuple[torch.Tensor, int]] = {}

    def _record(tensor, layer_idx, hook_name):
        rec[hook_name] = (tensor.detach().clone(), layer_idx)

    def _record_add(a, b, layer_idx, hook_name):
        # apply_block_steering captures post_block as separate summands
        # (DCE-safe under CUDA graphs); the recorded value is their sum.
        rec[hook_name] = ((a + b).detach().clone(), layer_idx)

    # Most models route every hook (incl. mlp_in/mlp_out) through the
    # steering helpers; gemma4 keeps direct model-module capture calls, so
    # patch the model-module name only where it exists.
    monkeypatch.setattr(model_mod, "maybe_capture_residual", _record, raising=False)
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


def _assert_common(rec, expected_mlp_in, expected_mlp_out, expected_post_block):
    assert "mlp_in" in rec and "mlp_out" in rec
    in_t, in_idx = rec["mlp_in"]
    out_t, out_idx = rec["mlp_out"]
    assert in_idx == LAYER_IDX and out_idx == LAYER_IDX
    torch.testing.assert_close(in_t, expected_mlp_in)
    torch.testing.assert_close(out_t, expected_mlp_out)
    # post_block must be the residual stream *after* the MLP branch is added.
    torch.testing.assert_close(rec["post_block"][0], expected_post_block)


def test_qwen3_mlp_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch, qwen3_mod)
    layer = _new(qwen3_mod.Qwen3DecoderLayer)

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = lambda **kw: attn_out
    layer.post_attention_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", mlp_out)

    layer.forward(positions=torch.arange(T), hidden_states=x, residual=None)

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    _assert_common(rec, mlp_in, mlp_out, residual_after + mlp_out)


def test_qwen3_mlp_hook_steering_injects(monkeypatch):
    """MLP_IN/MLP_OUT are injectable branch-tensor hooks.

    A steering row at MLP_IN shifts the tensor entering ``self.mlp``; a row
    at MLP_OUT shifts the branch returned to the residual stream (and hence
    the downstream post_block capture). Captures at the steered site itself
    stay pristine (capture runs before the steer inside the helper).
    """
    rec = _install_recorder(monkeypatch, qwen3_mod)
    layer = _new(qwen3_mod.Qwen3DecoderLayer)
    steering_mod.register_steering_buffers(
        layer,
        H,
        max_steering_tokens=T,
        max_steering_configs=1,
        dtype=torch.float32,
    )
    hp = steering_mod.SteeringHookPoint
    vec_in, vec_out = torch.randn(H), torch.randn(H)
    for hook, vec in ((hp.MLP_IN, vec_in), (hp.MLP_OUT, vec_out)):
        getattr(layer, steering_mod.HOOK_POINT_TABLE_ATTR[hook])[3] = vec
        getattr(layer, steering_mod.HOOK_POINT_ANY_ACTIVE_ATTR[hook]).fill_(True)
    layer.steering_index[:T] = 3

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = lambda **kw: attn_out
    layer.post_attention_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", mlp_out)

    hidden, _ = layer.forward(positions=torch.arange(T), hidden_states=x, residual=None)

    # Steered MLP input; pristine capture at the site.
    torch.testing.assert_close(seen["mlp_arg"], mlp_in + vec_in)
    torch.testing.assert_close(rec["mlp_in"][0], mlp_in)
    # Steered branch output; pristine capture at the site; post_block
    # (downstream of the intervention) observes the steered branch.
    torch.testing.assert_close(hidden, mlp_out + vec_out)
    torch.testing.assert_close(rec["mlp_out"][0], mlp_out)
    torch.testing.assert_close(rec["post_block"][0], residual_after + mlp_out + vec_out)


def test_qwen3_moe_mlp_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch, qwen3_moe_mod)
    layer = _new(qwen3_moe_mod.Qwen3MoeDecoderLayer)

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = lambda **kw: attn_out
    layer.post_attention_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", mlp_out)

    layer.forward(positions=torch.arange(T), hidden_states=x, residual=None)

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    _assert_common(rec, mlp_in, mlp_out, residual_after + mlp_out)


@pytest.mark.skipif(
    qwen3_next_mod is None,
    reason=f"qwen3_next unimportable: {_QWEN3_NEXT_IMPORT_ERR}",
)
def test_qwen3_next_mlp_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch, qwen3_next_mod)
    layer = _new(qwen3_next_mod.Qwen3NextDecoderLayer)
    layer.layer_type = "full_attention"
    layer.layer_scale = False

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    def _attn(hidden_states, output, positions=None):
        output.copy_(attn_out)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = _attn
    layer.post_attention_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", mlp_out)

    layer.forward(hidden_states=x, residual=None, positions=torch.arange(T))

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    _assert_common(rec, mlp_in, mlp_out, residual_after + mlp_out)


def test_gemma3_mlp_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch, gemma3_mod)
    layer = _new(gemma3_mod.Gemma3DecoderLayer)

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    normed_attn = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    residual_after = torch.randn(T, H)
    raw_mlp = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = lambda **kw: attn_out
    layer.post_attention_layernorm = lambda h: normed_attn
    layer.pre_feedforward_layernorm = lambda h, r: (mlp_in, residual_after)
    layer.mlp = _record_input(seen, "mlp_arg", raw_mlp)
    layer.post_feedforward_layernorm = lambda h: mlp_out

    layer.forward(positions=torch.arange(T), hidden_states=x, residual=None)

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    # gemma post-FFN norm is part of the branch, so mlp_out is the normed output.
    _assert_common(rec, mlp_in, mlp_out, residual_after + mlp_out)


def test_gemma4_mlp_hooks(monkeypatch):
    rec = _install_recorder(monkeypatch, gemma4_mod)
    layer = _new(gemma4_mod.Gemma4DecoderLayer)
    layer.enable_moe_block = False
    layer.per_layer_input_gate = None
    layer.layer_scalar = 1.0

    x = torch.randn(T, H)
    attn_out = torch.randn(T, H)
    normed_attn = torch.randn(T, H)
    mlp_in = torch.randn(T, H)
    raw_mlp = torch.randn(T, H)
    mlp_out = torch.randn(T, H)

    seen: dict = {}
    layer.input_layernorm = lambda h: h
    layer.self_attn = lambda **kw: attn_out
    layer.post_attention_layernorm = lambda h: normed_attn
    layer.pre_feedforward_layernorm = lambda h: mlp_in
    layer.mlp = _record_input(seen, "mlp_arg", raw_mlp)
    layer.post_feedforward_layernorm = lambda h: mlp_out

    layer.forward(positions=torch.arange(T), hidden_states=x, residual=None)

    torch.testing.assert_close(seen["mlp_arg"], mlp_in)
    # gemma4 adds the branch to the post-attention residual (normed_attn + x).
    post_attn = normed_attn + x
    _assert_common(rec, mlp_in, mlp_out, mlp_out + post_attn)
    # Residual-stream decomposition holds for gemma4: the post-attn residual is
    # captured *after* the deferred add, so post_block - post_attn == mlp_out.
    torch.testing.assert_close(rec["post_block"][0] - rec["post_attn"][0], mlp_out)
