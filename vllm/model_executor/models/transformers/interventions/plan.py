# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The residual layouts a Transformers decoder layer can be hooked through.

`analysis.analyze_layer` reads a decoder layer's `forward` source and returns a
`LayerPlan`: for every control-flow path through the forward, the AST nodes of
its residual adds and of its MLP call. `rewrite.rewrite_forward` turns those
nodes into calls to `hooks`. Every field is an AST node of the plan's own
`funcdef`, so a plan is only meaningful together with that tree.
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class MlpInputSite:
    """The hidden-states argument of the one MLP sublayer call.

    `keyword` is `None` when the argument is the call's first positional
    argument, else the `hidden_states=` keyword carrying it."""

    call: ast.Call
    child: str
    """Attribute name of the MLP child on the decoder layer (e.g. `"mlp"`)."""
    keyword: ast.keyword | None

    @property
    def value(self) -> ast.expr:
        return self.call.args[0] if self.keyword is None else self.keyword.value


@dataclass(frozen=True, eq=False)
class SequentialResidual:
    """`h = x + attn(...)` then `out = h + mlp(...)`: two residual adds.

    `*_stream` is the residual operand of each add and `*_branch` the sublayer
    operand (everything between the sublayer and the add, e.g. dropout, a
    post-sublayer norm, or a residual multiplier, is part of the branch)."""

    attn_add: ast.BinOp
    attn_stream: ast.expr
    attn_branch: ast.expr
    mlp_add: ast.BinOp
    mlp_stream: ast.expr
    mlp_branch: ast.expr
    mlp_input: MlpInputSite


@dataclass(frozen=True, eq=False)
class ParallelResidual:
    """`out = x + attn(...) + mlp(...)`: one residual add over both branches,
    as in GPT-J/NeoX-style parallel blocks (Phi, parallel GPT-NeoX/StableLM)."""

    add: ast.BinOp
    stream: ast.expr
    attn_branch: ast.expr
    mlp_branch: ast.expr
    mlp_input: MlpInputSite


ResidualLayout = SequentialResidual | ParallelResidual


@dataclass(frozen=True, eq=False)
class LayerPlan:
    """Where to hook one decoder layer class, and the tree to rewrite."""

    layer_cls: type
    funcdef: ast.FunctionDef
    """The parsed forward the layout nodes belong to (rewritten in place)."""
    fn: Callable
    """The original, undecorated forward the source was parsed from."""
    self_name: str
    """Name of the forward's first parameter (`self`)."""
    layouts: tuple[ResidualLayout, ...]
    """One layout per distinct control-flow path (e.g. both arms of
    `if self.use_parallel_residual:`), in source order."""
