# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook-point semantics of rewritten Transformers decoder layers (CPU only).

The native fused-residual layers (Llama, Qwen3, ...) fix what each hook point
means, and the rewritten HF layers must match:

- `pre_attn` / `post_attn` act on the *carried residual* (the skip operand of
  the attention / MLP residual add). Steering there does not change the input
  of the sublayer the residual skips, only what flows on.
- `mlp_in` is the normed input entering the MLP sublayer.
- `mlp_out` is the whole MLP branch as added to the residual.
- `post_block` observes the block output `residual + mlp_branch`.

Parallel blocks (Phi, parallel GPT-NeoX) observe `post_attn` at
`residual + attn_branch`, as the native Phi layer did.
"""

import pytest
import torch

from tests.models.transformers.interventions.helpers import (
    HOOKS,
    activate,
    intercept_capture,
)
from tests.models.transformers.interventions.layers import (
    SUPPORTED_LAYERS,
    LlamaLikeLayer,
    NeoXLikeLayer,
    ParallelLayer,
    is_stub_attention,
)
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    STANDARD_STEERING_HOOKS,
    SteeringHookPoint,
)
from vllm.model_executor.models.transformers.interventions import (
    BufferSizes,
    UnsupportedInterventionLayout,
    install_interventions,
)

HIDDEN = 8
TOKENS = 5
SIZES = BufferSizes(max_tokens=16, max_configs=2, dtype=torch.float32)


def _layer_variants():
    for cls in SUPPORTED_LAYERS:
        if cls is NeoXLikeLayer:
            yield pytest.param(lambda: NeoXLikeLayer(HIDDEN, True), id="neox-parallel")
            yield pytest.param(
                lambda: NeoXLikeLayer(HIDDEN, False), id="neox-sequential"
            )
        else:
            yield pytest.param(lambda cls=cls: cls(HIDDEN), id=cls.__name__)


LAYER_VARIANTS = list(_layer_variants())


def _is_parallel(layer) -> bool:
    return isinstance(layer, ParallelLayer) or (
        isinstance(layer, NeoXLikeLayer) and layer.use_parallel_residual
    )


def _build(make, sizes=SIZES):
    torch.manual_seed(0)
    layer = make().eval()
    original = type(layer).forward
    install_interventions(
        [layer], [0], hidden_size=HIDDEN, sizes=sizes, is_attention=is_stub_attention
    )
    return layer, original


def _input():
    torch.manual_seed(1)
    # Transformers decoder layers see a batch of one: [1, tokens, hidden].
    return torch.randn(1, TOKENS, HIDDEN)


def _out(result):
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("make", LAYER_VARIANTS)
def test_inactive_interventions_are_a_no_op(make):
    layer, original = _build(make)
    x = _input()
    with torch.no_grad():
        expected = _out(original(layer, x))
        actual = _out(layer(x))
    if _is_parallel(layer):
        # The parallel rewrite forms `residual + attn` before adding the MLP
        # branch, which re-associates the original three-way sum.
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    else:
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("make", LAYER_VARIANTS)
def test_capture_observes_native_hook_tensors(make, monkeypatch):
    layer, _ = _build(make)
    seen = intercept_capture(monkeypatch)
    x = _input()
    with torch.no_grad():
        out = _out(layer(x))
        reference = layer.reference(x)

    observed = {hook: t for _, hook, t in seen}
    assert sorted(observed) == sorted(h.value for h in HOOKS)
    assert all(layer_idx == 0 for layer_idx, _, _ in seen)
    for hook in HOOKS:
        tensor = observed[hook.value]
        # Hooks see the token-major 2-D view, as on native models.
        assert tensor.shape == (TOKENS, HIDDEN)
        torch.testing.assert_close(
            tensor, reference[hook.value][0], rtol=0, atol=1e-6, msg=hook.value
        )
    torch.testing.assert_close(observed["post_block"], out[0], rtol=0, atol=1e-6)


@pytest.mark.parametrize("make", LAYER_VARIANTS)
@pytest.mark.parametrize("hook", HOOKS, ids=lambda h: h.value)
def test_steering_lands_on_the_native_site(make, hook):
    layer, _ = _build(make)
    x = _input()
    vec = torch.linspace(-1.0, 1.0, HIDDEN)
    with torch.no_grad():
        if hook is SteeringHookPoint.PRE_ATTN:
            # The skip carries the vector into the rest of the block, but this
            # layer's attention input is computed from the unsteered residual.
            expected = layer.reference(x, d_pre=vec)["post_block"]
        elif hook is SteeringHookPoint.MLP_IN:
            expected = layer.reference(x, d_mlp_in=vec)["post_block"]
        else:
            # post_attn steers the skip of the MLP add (not the MLP input), and
            # mlp_out / post_block ride the block output: all three add `vec`.
            expected = layer.reference(x)["post_block"] + vec
        activate(layer, hook, vec, TOKENS)
        actual = _out(layer(x))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
    assert not torch.allclose(actual, layer.reference(x)["post_block"])


@pytest.mark.parametrize("make", LAYER_VARIANTS)
def test_steering_only_touches_indexed_tokens(make):
    layer, _ = _build(make)
    x = _input()
    with torch.no_grad():
        baseline = _out(layer(x)).clone()
        activate(layer, SteeringHookPoint.POST_BLOCK, torch.ones(HIDDEN), TOKENS)
        layer.steering_index[: TOKENS - 1].fill_(0)
        steered = _out(layer(x))
    assert torch.equal(steered[0, :-1], baseline[0, :-1])
    torch.testing.assert_close(steered[0, -1], baseline[0, -1] + 1.0)


def test_buffers_and_layer_identity():
    torch.manual_seed(0)
    layers = [LlamaLikeLayer(HIDDEN), LlamaLikeLayer(HIDDEN), LlamaLikeLayer(HIDDEN)]
    install_interventions(
        layers, [1, 2], hidden_size=HIDDEN, sizes=SIZES, is_attention=is_stub_attention
    )
    # Only the listed (pipeline-local) layers are instrumented.
    assert not hasattr(layers[0], "steering_index")
    assert "forward" not in vars(layers[0])
    assert [layers[1].layer_idx, layers[2].layer_idx] == [1, 2]
    for layer in layers[1:]:
        for hook in STANDARD_STEERING_HOOKS:
            table = getattr(layer, HOOK_POINT_TABLE_ATTR[hook])
            assert table.shape[1] == HIDDEN
            assert table.dtype == torch.float32
    assert layers[1].steering_index is layers[2].steering_index


def test_capture_without_steering_tables(monkeypatch):
    """Capture-only servers register no steering tables but keep every tap."""
    layer, original = _build(
        lambda: LlamaLikeLayer(HIDDEN),
        BufferSizes(max_tokens=16, max_configs=0, dtype=torch.float32),
    )
    assert not hasattr(layer, HOOK_POINT_TABLE_ATTR[SteeringHookPoint.PRE_ATTN])
    seen = intercept_capture(monkeypatch)
    x = _input()
    with torch.no_grad():
        assert torch.equal(_out(layer(x)), _out(original(layer, x)))
    assert sorted(hook for _, hook, _ in seen) == sorted(h.value for h in HOOKS)


def test_existing_layer_idx_must_agree():
    layer = LlamaLikeLayer(HIDDEN)
    layer.layer_idx = 3
    with pytest.raises(UnsupportedInterventionLayout, match="layer_idx"):
        install_interventions(
            [layer],
            [0],
            hidden_size=HIDDEN,
            sizes=SIZES,
            is_attention=is_stub_attention,
        )


def test_rewritten_forward_is_shared_per_class():
    torch.manual_seed(0)
    layers = [LlamaLikeLayer(HIDDEN), LlamaLikeLayer(HIDDEN)]
    install_interventions(
        layers, [0, 1], hidden_size=HIDDEN, sizes=SIZES, is_attention=is_stub_attention
    )
    assert layers[0].forward.__func__ is layers[1].forward.__func__


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: LlamaLikeLayer(HIDDEN), id="sequential"),
        pytest.param(lambda: ParallelLayer(HIDDEN), id="parallel"),
    ],
)
def test_torch_compile_fullgraph(make):
    """The rewritten forward traces without graph breaks, and the steering op
    reads its buffers at run time (a later table write changes the output
    without recompiling)."""
    layer, _ = _build(make)
    x = _input()
    compiled = torch.compile(layer, backend="eager", fullgraph=True, dynamic=False)
    vec = torch.linspace(-1.0, 1.0, HIDDEN)
    with torch.no_grad():
        baseline = _out(compiled(x)).clone()
        torch.testing.assert_close(baseline, _out(layer(x)), rtol=0, atol=0)
        for hook in (SteeringHookPoint.MLP_IN, SteeringHookPoint.POST_BLOCK):
            activate(layer, hook, vec, TOKENS)
        steered = _out(compiled(x))
        torch.testing.assert_close(steered, _out(layer(x)), rtol=0, atol=1e-6)
    assert not torch.allclose(steered, baseline)
