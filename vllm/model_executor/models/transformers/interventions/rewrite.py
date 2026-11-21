# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rewrite a decoder layer's forward so its residual adds call `hooks`.

Like the backend's `RewriteFuser`s, this edits the forward's *source* (AST):
only the nodes `analysis` located change, and the rest stays the original
Python, compiled in the original module's namespace so tracebacks and globals
resolve as before. The hooks are reached through one module-level name,
`HOOKS_NAME`, added to that namespace.
"""

import ast
from collections.abc import Callable

from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    replace_expr,
)
from vllm.model_executor.models.transformers.interventions import hooks
from vllm.model_executor.models.transformers.interventions.errors import (
    UnsupportedInterventionLayout,
)
from vllm.model_executor.models.transformers.interventions.plan import (
    LayerPlan,
    MlpInputSite,
    ParallelResidual,
    SequentialResidual,
)

HOOKS_NAME = "_vllm_intervention_hooks"
"""Global through which a rewritten forward reaches `hooks`."""


def _hook_call(plan: LayerPlan, name: str, *args: ast.expr) -> ast.Call:
    func = ast.Attribute(
        value=ast.Name(id=HOOKS_NAME, ctx=ast.Load()), attr=name, ctx=ast.Load()
    )
    layer = ast.Name(id=plan.self_name, ctx=ast.Load())
    return ast.Call(func=func, args=[layer, *args], keywords=[])


def _wrap_mlp_input(plan: LayerPlan, site: MlpInputSite, done: set[int]) -> None:
    if id(site.call) in done:
        return
    done.add(id(site.call))
    wrapped = _hook_call(plan, "mlp_input", site.value)
    ast.copy_location(wrapped, site.value)
    if site.keyword is None:
        site.call.args[0] = wrapped
    else:
        site.keyword.value = wrapped


def _replace(plan: LayerPlan, node: ast.expr, new: ast.Call, done: set[int]) -> None:
    if id(node) in done:
        return
    done.add(id(node))
    ast.copy_location(new, node)
    replace_expr(plan.funcdef, node, new)


def _check_namespace(plan: LayerPlan, namespace: dict) -> None:
    names = {n.id for n in ast.walk(plan.funcdef) if isinstance(n, ast.Name)}
    if HOOKS_NAME in names:
        raise UnsupportedInterventionLayout(
            plan.layer_cls.__name__, f"its forward already uses the name {HOOKS_NAME}"
        )
    existing = namespace.get(HOOKS_NAME, hooks)
    if existing is not hooks:
        raise UnsupportedInterventionLayout(
            plan.layer_cls.__name__,
            f"its module defines {HOOKS_NAME}, which the rewrite needs",
        )


def rewrite_forward(plan: LayerPlan) -> Callable:
    """Compile `plan`'s forward with every located site hooked.

    Mutates `plan.funcdef`, so each plan is rewritten at most once."""
    namespace = plan.fn.__globals__
    _check_namespace(plan, namespace)
    done: set[int] = set()
    # MLP calls first: they can sit inside a residual add's branch operand,
    # which the add's replacement then carries along already wrapped.
    for layout in plan.layouts:
        _wrap_mlp_input(plan, layout.mlp_input, done)
    for layout in plan.layouts:
        if isinstance(layout, SequentialResidual):
            _replace(
                plan,
                layout.attn_add,
                _hook_call(
                    plan, "attn_residual_add", layout.attn_stream, layout.attn_branch
                ),
                done,
            )
            _replace(
                plan,
                layout.mlp_add,
                _hook_call(
                    plan, "mlp_residual_add", layout.mlp_stream, layout.mlp_branch
                ),
                done,
            )
        elif isinstance(layout, ParallelResidual):
            _replace(
                plan,
                layout.add,
                _hook_call(
                    plan,
                    "parallel_residual_add",
                    layout.stream,
                    layout.attn_branch,
                    layout.mlp_branch,
                ),
                done,
            )
    namespace.setdefault(HOOKS_NAME, hooks)
    return compile_forward(plan.funcdef, plan.fn)
