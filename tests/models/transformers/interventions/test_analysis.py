# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residual-layout analysis of Transformers decoder layers (CPU only)."""

import pytest
from torch import nn

from tests.models.transformers.interventions.layers import (
    UNSUPPORTED_LAYERS,
    InlineMlpLayer,
    LlamaLikeLayer,
    NeoXLikeLayer,
    OptionalCrossAttentionLayer,
    ParallelLayer,
    PostNormLayer,
    ScaledDropoutLayer,
    is_stub_attention,
)
from vllm.model_executor.models.transformers.interventions import (
    ParallelResidual,
    SequentialResidual,
    UnsupportedInterventionLayout,
    analyze_layer,
)


@pytest.mark.parametrize(
    ("layer_cls", "expected"),
    [
        (LlamaLikeLayer, [SequentialResidual]),
        (PostNormLayer, [SequentialResidual]),
        (ScaledDropoutLayer, [SequentialResidual]),
        (InlineMlpLayer, [SequentialResidual]),
        (ParallelLayer, [ParallelResidual]),
        # Both arms of `if self.use_parallel_residual:`, in source order.
        (NeoXLikeLayer, [ParallelResidual, SequentialResidual]),
        # The cross-attention arm calls modules the instance does not have.
        (OptionalCrossAttentionLayer, [SequentialResidual]),
    ],
)
def test_layouts(layer_cls, expected):
    plan = analyze_layer(layer_cls(), is_stub_attention)
    assert [type(layout) for layout in plan.layouts] == expected
    for layout in plan.layouts:
        assert layout.mlp_input.child == "mlp"


def test_sequential_layout_names_the_residual_operands():
    plan = analyze_layer(LlamaLikeLayer(), is_stub_attention)
    (layout,) = plan.layouts
    assert isinstance(layout, SequentialResidual)
    # `hidden_states = residual + hidden_states` twice: the residual operand is
    # the skip, the other operand is the sublayer branch.
    assert layout.attn_stream.id == "residual"
    assert layout.attn_branch.id == "hidden_states"
    assert layout.mlp_stream.id == "residual"
    assert layout.mlp_branch.id == "hidden_states"
    assert layout.attn_add.lineno < layout.mlp_add.lineno
    assert layout.mlp_input.keyword is None


def test_parallel_layout_separates_the_branches():
    plan = analyze_layer(ParallelLayer(), is_stub_attention)
    (layout,) = plan.layouts
    assert isinstance(layout, ParallelResidual)
    assert layout.stream.id == "residual"
    assert layout.attn_branch.id == "attn_outputs"
    assert layout.mlp_branch.id == "feed_forward_hidden_states"


def test_plans_are_independent_trees():
    """Each analysis parses a fresh tree, so rewriting one cannot corrupt another."""
    first = analyze_layer(LlamaLikeLayer(), is_stub_attention)
    second = analyze_layer(LlamaLikeLayer(), is_stub_attention)
    assert first.funcdef is not second.funcdef


@pytest.mark.parametrize("layer_cls", UNSUPPORTED_LAYERS)
def test_unsupported_layouts_fail_loudly(layer_cls):
    with pytest.raises(UnsupportedInterventionLayout) as info:
        analyze_layer(layer_cls(), is_stub_attention)
    assert info.value.layer_cls == layer_cls.__name__
    assert info.value.reason
    assert layer_cls.__name__ in str(info.value)


def test_requires_an_attention_child():
    """With the default predicate, a module without a vLLM attention layer in
    its subtree is not attention, so the layer has no attention sublayer."""
    with pytest.raises(UnsupportedInterventionLayout, match="dispatches attention"):
        analyze_layer(LlamaLikeLayer())


def test_refuses_an_instance_forward_override():
    layer = LlamaLikeLayer()
    layer.forward = lambda hidden_states: hidden_states
    with pytest.raises(UnsupportedInterventionLayout, match="already replaced"):
        analyze_layer(layer, is_stub_attention)


def test_norms_and_dropouts_are_not_sublayers():
    """Only children with projections count as sublayers; a norm wrapping the
    attention output stays part of the attention branch (OLMo 2)."""
    plan = analyze_layer(PostNormLayer(), is_stub_attention)
    (layout,) = plan.layouts
    assert isinstance(layout, SequentialResidual)
    assert layout.mlp_input.child == "mlp"


class _Uninspectable(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = LlamaLikeLayer().self_attn


_Uninspectable.forward = eval("lambda self, hidden_states: hidden_states")


def test_unrecoverable_source_fails_loudly():
    with pytest.raises(UnsupportedInterventionLayout, match="source"):
        analyze_layer(_Uninspectable(), is_stub_attention)
