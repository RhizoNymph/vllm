# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intervention hooks of the Inkling2 decoders (CPU, stubbed sublayers).

Pins the semantics documented in `vllm/models/inkling2/interventions.py`:
one site per layer visit keyed by its layer key, the hook order inside a
visit, what tensor each hook sees, that a pass restarts from the prelude
state, and that an unchanged state is an exact no-op.

The decoder layers are built with `object.__new__` and their attention /
convs / MoE replaced by small torch functions; the AttnRes mixes run a
torch reference. Real kernels are covered by the GPU tests.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.loop_layers import LoopLayerSpace
from vllm.model_executor.layers.steering import SteeringHookPoint as HP
from vllm.models.inkling2 import interventions
from vllm.models.inkling2.interventions import Inkling2InterventionSite
from vllm.models.inkling2.nvidia import loop as loop_mod
from vllm.models.inkling2.nvidia import model as model_mod

pytestmark = pytest.mark.skip_global_cleanup

D, T = 16, 5
HOOK_ORDER = [HP.PRE_ATTN, HP.POST_ATTN, HP.MLP_IN, HP.MLP_OUT, HP.POST_BLOCK]


def _attn_res(stack, last, addend, queries, norm_weight, norm_eps, normalize, eps=1e-6):
    last = last.float() if addend is None else last.float() + addend.float()
    src = torch.cat([stack.float(), last[None]])
    keys = src * torch.rsqrt(src.square().mean(-1, keepdim=True) + eps)
    weights = torch.einsum("ntd,qd->qnt", keys if normalize else src, queries).softmax(
        1
    )
    mix = torch.einsum("qnt,ntd->qtd", weights, src)
    if norm_weight is None:
        return mix
    normed = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + norm_eps)
    return (normed * norm_weight[:, None].float()).to(stack.dtype)


class _Norm:
    def __init__(self, n):
        self.weight = torch.ones(n, D)
        self.eps = 1e-6

    def __call__(self, x, index=0):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)


def _layer(i, is_local, looped, num_passes, g):
    cls = (
        loop_mod.Inkling2LoopDecoderLayer if looped else model_mod.Inkling2DecoderLayer
    )
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer.is_local = is_local
    layer.num_attn_inputs = 1 if is_local else 3
    layer.attn_norm = _Norm(layer.num_attn_inputs)
    layer.mlp_norm = _Norm(1)
    layer.conv_states = tuple(f"conv{i}.pass{p}" for p in range(num_passes))
    w_attn, w_mlp = (
        torch.randn(D, D, generator=g) / 4,
        torch.randn(D, D, generator=g) / 4,
    )
    owners = layer.owners = []
    layer.attention = lambda positions, inputs, tau, pass_idx=0: inputs[0] @ w_attn
    layer.attn_sconv = lambda x, positions, owner=None: (owners.append(owner), x * 0.5)[
        1
    ]

    def mlp(x, pass_idx=0):
        layer.mlp_inputs.append(x.clone())
        return torch.tanh(x @ w_mlp)

    layer.mlp_inputs = []
    layer.mlp = mlp

    def mlp_sconv(x, positions, addend=None, residual=None, out=None, owner=None):
        base = addend if residual is None else residual + addend
        return out.copy_(base + 0.5 * x)

    layer.mlp_sconv = mlp_sconv
    layer.blend = lambda streams, dest, message, w, eps: _Norm(1)(dest + message)
    return layer


# (num_layers, core) per AttnRes block size, shaped like Inkling 2 Mini: the
# core spans two blocks, so a pass reads a two-block message bank.
GEOMETRY = {1: (4, (1, 3)), 2: (8, (2, 6))}


def _loop_model(block_size, g, passes=3):
    num_layers, core = GEOMETRY[block_size]
    model = object.__new__(loop_mod.Inkling2LoopModel)
    nn.Module.__init__(model)
    model.block_size = block_size
    model.normalize_values = True
    model.loop_core = range(*core)
    model.config = SimpleNamespace(num_loop_passes=passes)
    model.layers = nn.ModuleList(
        _layer(i, i % 2 == 0, i in model.loop_core, passes, g)
        for i in range(num_layers)
    )
    model.attn_res = SimpleNamespace(
        attn_queries=[
            torch.randn(model.layers[i].num_attn_inputs, D, generator=g)
            for i in range(1, num_layers)
        ],
        mlp_queries=[torch.randn(1, D, generator=g) for _ in range(num_layers)],
        output_query=torch.randn(1, D, generator=g),
    )
    model.norm = _Norm(1)
    model.message_queries = torch.randn(len(model.loop_core), 4, D, generator=g)
    model.aux_hidden_state_layers = ()
    model._log_scaling_tau = lambda positions: None
    model.interventions = True
    space = LoopLayerSpace(num_layers, core[0], core[1], passes)
    model.intervention_sites = nn.ModuleList(
        Inkling2InterventionSite(key, D, None) for key in space.visit_keys()
    )
    return model, space


@pytest.fixture
def hooks(monkeypatch):
    """Record every hook call; `deltas[(key, hook)]` is added to the state."""
    state = SimpleNamespace(calls=[], deltas={})

    def apply(site, hidden, hook_point):
        state.calls.append((site.layer_idx, hook_point, hidden.detach().clone()))
        delta = state.deltas.get((site.layer_idx, hook_point))
        return hidden if delta is None else hidden + delta

    for mod in (model_mod, interventions):
        monkeypatch.setattr(mod, "apply_layer_steering", apply)
    for mod in (model_mod, loop_mod):
        monkeypatch.setattr(mod, "attn_res", _attn_res)
    monkeypatch.setattr(model_mod, "tensor_model_parallel_all_reduce", lambda x: x)
    return state


def _run(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    return model(torch.arange(T), torch.randn(T, D, generator=g))


@pytest.mark.parametrize("block_size", [1, 2])
def test_one_site_per_visit_in_execution_order(hooks, block_size):
    model, space = _loop_model(block_size, torch.Generator().manual_seed(0))
    _run(model)
    keys = [key for key, _, _ in hooks.calls[:: len(HOOK_ORDER)]]
    assert keys == space.visit_keys()
    if block_size == 1:
        # prelude 0, core (1, 2) x 3 passes, coda 3.
        assert keys == [0, 5, 6, 9, 10, 13, 14, 3]
    assert [hp for _, hp, _ in hooks.calls] == HOOK_ORDER * len(keys)
    # Each pass of a core layer runs on its own conv-state cache.
    first_core = space.core_start
    assert model.layers[first_core].owners == [
        f"conv{first_core}.pass{p}" for p in range(3)
    ]
    assert model.layers[-1].owners == [f"conv{len(model.layers) - 1}.pass0"]


def _states(calls):
    return {(key, hp): t for key, hp, t in calls}


@pytest.mark.parametrize("block_size", [1, 2])
def test_post_block_is_post_attn_state_plus_mlp_branch(hooks, block_size):
    """`post_attn` is the running block state after attention (`h +
    attn_out`, or `attn_out` at a block's first layer) and `post_block`
    that state plus the MLP branch: all three residual hooks share one
    space."""
    model, space = _loop_model(block_size, torch.Generator().manual_seed(1))
    delta = torch.full((T, D), 0.25)
    hooks.deltas[(space.key(2, 1), HP.POST_ATTN)] = delta
    _run(model)
    states = _states(hooks.calls)
    for key in space.visit_keys():
        post_attn = states[(key, HP.POST_ATTN)]
        post_attn = post_attn + hooks.deltas.get((key, HP.POST_ATTN), 0)
        branch = states[(key, HP.MLP_OUT)]
        torch.testing.assert_close(
            states[(key, HP.POST_BLOCK)], post_attn + 0.5 * branch
        )


def test_post_attn_of_a_continuing_block_includes_the_running_state(hooks):
    """With two-layer blocks, core layer 3's `post_attn` is `pre_attn +
    attn_out` (its `pre_attn` state continues the block); a `pre_attn` edit
    flows into it."""
    model, space = _loop_model(2, torch.Generator().manual_seed(2))
    key = space.key(3, 1)
    _run(model)
    base = _states(hooks.calls)
    hooks.calls.clear()
    hooks.deltas[(key, HP.PRE_ATTN)] = torch.full((T, D), 3.0)
    _run(model)
    steered = _states(hooks.calls)
    torch.testing.assert_close(steered[(key, HP.PRE_ATTN)], base[(key, HP.PRE_ATTN)])
    # Layer 1 is not the first of its block: the edit carries into its
    # post-attention state (its attention input changed too, so only the
    # carried part is pinned: the difference is >= the edit in every slot
    # where attn_out is unchanged-sign; check the mean).
    diff = steered[(key, HP.POST_ATTN)] - base[(key, HP.POST_ATTN)]
    assert diff.mean() > 2.0


def test_each_pass_restarts_from_the_prelude_state(hooks):
    """An edit late in pass 0 reaches pass 1 only through its messages: the
    pass-1 entry state of the first core layer is the prelude's again."""
    model, space = _loop_model(1, torch.Generator().manual_seed(3))
    _run(model)
    base = _states(hooks.calls)
    hooks.calls.clear()
    hooks.deltas[(space.key(2, 0), HP.POST_BLOCK)] = torch.full((T, D), 5.0)
    _run(model)
    steered = _states(hooks.calls)
    for p in range(3):
        entry = (space.key(1, p), HP.PRE_ATTN)
        torch.testing.assert_close(steered[entry], base[entry])
        # Every pass starts from the prelude's post-block state.
        torch.testing.assert_close(steered[entry], base[(0, HP.POST_BLOCK)])
    # Pass 0 before the edit is untouched; pass 1 differs (messages).
    torch.testing.assert_close(
        steered[(space.key(2, 0), HP.PRE_ATTN)], base[(space.key(2, 0), HP.PRE_ATTN)]
    )
    assert not torch.allclose(
        steered[(space.key(1, 1), HP.MLP_IN)], base[(space.key(1, 1), HP.MLP_IN)]
    )


@pytest.mark.parametrize("pass_idx", [0, 1, 2])
def test_a_pass_specific_edit_only_changes_later_visits(hooks, pass_idx):
    model, space = _loop_model(1, torch.Generator().manual_seed(4))
    out_base = _run(model)
    base = list(hooks.calls)
    hooks.calls.clear()
    key = space.key(1, pass_idx)
    hooks.deltas[(key, HP.MLP_IN)] = torch.full((T, D), 0.5)
    out = _run(model)
    order = space.visit_keys()
    edited = order.index(key)
    per_visit = len(HOOK_ORDER)
    for n, ((k0, hp0, t0), (k1, hp1, t1)) in enumerate(zip(base, hooks.calls)):
        assert (k0, hp0) == (k1, hp1)
        visit, slot = divmod(n, per_visit)
        before = visit < edited or (
            visit == edited and slot <= HOOK_ORDER.index(HP.MLP_IN)
        )
        if before:
            torch.testing.assert_close(t1, t0)
    # The MoE of the edited visit read the edited input (layer 1's MoE runs
    # once per pass: three calls per forward, the base run's first).
    torch.testing.assert_close(
        model.layers[1].mlp_inputs[3 + pass_idx],
        base[edited * per_visit + HOOK_ORDER.index(HP.MLP_IN)][2] + 0.5,
    )
    assert not torch.allclose(out, out_base)


def test_unchanged_states_are_an_exact_no_op(hooks):
    """Hooks that return a new but equal tensor (inactive buffers, a zero
    vector) leave the output bit-identical."""
    model, space = _loop_model(2, torch.Generator().manual_seed(5))
    out_base = _run(model)
    for key in space.visit_keys():
        for hp in HOOK_ORDER:
            hooks.deltas[(key, hp)] = torch.zeros(T, D)
    out = _run(model)
    assert torch.equal(out, out_base)


def test_hook_branch_writes_the_edit_back_as_a_delta():
    calls = []

    def apply(site, hidden, hook_point):
        calls.append(hidden.clone())
        return hidden * 2

    base, branch = torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.25])
    original = interventions.apply_layer_steering
    interventions.apply_layer_steering = apply
    try:
        out = interventions.hook_branch(None, HP.POST_ATTN, branch, base)
        assert torch.equal(calls[0], base + branch)
        # new state 2 * (base + branch) == base + out
        torch.testing.assert_close(base + out, 2 * (base + branch))
        solo = interventions.hook_branch(None, HP.POST_ATTN, branch, None)
        torch.testing.assert_close(solo, 2 * branch)
    finally:
        interventions.apply_layer_steering = original


def test_small_model_sites_are_the_layer_indices(hooks):
    model = object.__new__(model_mod.Inkling2Model)
    nn.Module.__init__(model)
    g = torch.Generator().manual_seed(6)
    model.block_size = 2
    model.normalize_values = True
    model.layers = nn.ModuleList(_layer(i, i == 0, False, 1, g) for i in range(3))
    model.attn_res = SimpleNamespace(
        attn_queries=[torch.randn(model.layers[i].num_attn_inputs, D) for i in (1, 2)],
        mlp_queries=[torch.randn(1, D) for _ in range(3)],
        output_query=torch.randn(1, D),
    )
    model.norm = _Norm(1)
    model.aux_hidden_state_layers = ()
    model._log_scaling_tau = lambda positions: None
    model.interventions = True
    model.intervention_sites = nn.ModuleList(
        Inkling2InterventionSite(i, D, None) for i in range(3)
    )
    _run(model)
    assert [k for k, _, _ in hooks.calls[:: len(HOOK_ORDER)]] == [0, 1, 2]
