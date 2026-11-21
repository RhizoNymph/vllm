# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find the residual adds of a Transformers decoder layer from its source.

Hugging Face decoder layers do their residual adds inline in `forward`
(`hidden_states = residual + hidden_states`), so module hooks cannot observe
"the residual after attention". This module reads the forward's AST and runs a
small abstract interpretation over it, tracking for every local name whether it
holds the residual stream (and after how many residual adds) or a value
computed from the layer's sublayers. An add with exactly one residual-stream
operand and only sublayer-derived other operands is a residual add.

The interpretation is path-sensitive: each `if` forks the analysis, so a layer
like GPT-NeoX, whose `use_parallel_residual` branch selects a parallel or a
sequential block, yields one layout per arm. A path that calls a `self.<attr>`
the instance lacks (or holds as `None`) would raise at run time and is dropped,
like a path ending in `raise`. Anything else the interpreter cannot follow
(loops or `try` over residual values, in-place updates of them, calls to
non-module methods with them) raises `UnsupportedInterventionLayout` rather
than guessing, since a wrong guess would silently hook the wrong tensor.

Only the layer's *direct* children matter. A child is an *attention* sublayer
when its subtree holds a module for which `is_attention` is true (by default a
vLLM `AttentionLayerBase`, which `Base._create_attention_instances` attaches to
the module that dispatches attention), and *heavy* when it holds a projection
(`nn.Linear`, a vLLM linear or MoE layer) or any matrix-shaped parameter. A
heavy child that is not attention is an MLP sublayer; norms and dropouts are
neither and are treated as part of whichever branch they sit on.
"""

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from torch import nn
from torch.nn.parameter import UninitializedParameter

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.transformers.fx_utils import recover_forward
from vllm.model_executor.models.transformers.interventions.errors import (
    UnsupportedInterventionLayout,
)
from vllm.model_executor.models.transformers.interventions.plan import (
    LayerPlan,
    MlpInputSite,
    ParallelResidual,
    ResidualLayout,
    SequentialResidual,
)

MAX_PATHS = 64
"""Upper bound on control-flow paths explored through one forward."""

_PROJECTION_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    LinearBase,
    MoERunner,
    AttentionLayerBase,
)


def is_vllm_attention(module: nn.Module) -> bool:
    """Default attention predicate: the module is a vLLM attention layer."""
    return isinstance(module, AttentionLayerBase)


def _has_matrix_parameter(module: nn.Module) -> bool:
    return any(
        isinstance(p, UninitializedParameter) or p.ndim >= 2
        for p in module.parameters()
    )


def _is_heavy(module: nn.Module) -> bool:
    """Whether `module` computes a sublayer (projections), not a norm/dropout."""
    return any(
        isinstance(m, _PROJECTION_TYPES) for m in module.modules()
    ) or _has_matrix_parameter(module)


@dataclass(frozen=True)
class _Val:
    """Abstract value of an expression.

    `stream` is the number of residual adds behind the value when it *is* the
    residual stream, else `None`. `calls` are the layer children whose output
    the value depends on, and `mlp_calls` the call nodes among them that invoke
    an MLP sublayer. A value computed from the stream (e.g. a norm of it) is
    not the stream: only plain name copies and residual adds preserve it."""

    stream: int | None = None
    calls: frozenset[str] = frozenset()
    mlp_calls: frozenset[ast.Call] = frozenset()

    @property
    def tracked(self) -> bool:
        return self.stream is not None or bool(self.calls)


_OTHER = _Val()


def _join(*vals: _Val) -> _Val:
    """A value computed from `vals` (not itself the residual stream)."""
    calls: frozenset[str] = frozenset().union(*(v.calls for v in vals))
    mlp_calls: frozenset[ast.Call] = frozenset().union(*(v.mlp_calls for v in vals))
    if not calls and all(v.stream is None for v in vals):
        return _OTHER
    return _Val(None, calls, mlp_calls)


class _Kind(Enum):
    STREAM = auto()
    ATTN = auto()
    MLP = auto()
    MIXED = auto()
    INERT = auto()


@dataclass(frozen=True, eq=False)
class _ResidualAdd:
    node: ast.BinOp
    stream: ast.expr
    attn: tuple[tuple[ast.expr, _Val], ...]
    mlp: tuple[tuple[ast.expr, _Val], ...]


@dataclass
class _State:
    """Per-path facts collected while interpreting."""

    adds: list[_ResidualAdd] = field(default_factory=list)
    plain_adds: set[ast.BinOp] = field(default_factory=set)

    def copy(self) -> "_State":
        return _State(list(self.adds), set(self.plain_adds))


@dataclass(frozen=True)
class _Path:
    adds: tuple[_ResidualAdd, ...]
    plain_adds: frozenset[ast.BinOp]
    returned: _Val | None


def _flatten_add(node: ast.expr) -> list[ast.expr]:
    """The terms of a chain of `+` (`a + b + c` -> `[a, b, c]`)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add(node.left) + _flatten_add(node.right)
    return [node]


_IGNORED_STATEMENTS = (
    ast.Pass,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)
_OPAQUE_STATEMENTS: tuple[type[ast.stmt], ...] = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
    *((ast.TryStar,) if hasattr(ast, "TryStar") else ()),
)


class _InfeasiblePath(Exception):
    """The path calls `self.<name>` for an attribute the instance lacks or sets
    to `None`, so it would raise at run time (e.g. GPTBigCode's
    `ln_cross_attn`, built only for cross-attention configs, or StableLM's
    `post_attention_layernorm` in parallel mode). Like a path ending in
    `raise`, it is dropped."""


class _Interpreter:
    def __init__(
        self,
        layer_cls: str,
        self_name: str,
        children: Iterable[str],
        attention: frozenset[str],
        heavy: frozenset[str],
        has_attribute: Callable[[str], bool],
    ):
        self.layer_cls = layer_cls
        self.self_name = self_name
        self.children = frozenset(children)
        self.has_attribute = has_attribute
        self.attention = attention
        self.mlp = heavy - attention
        self.num_paths = 1

    def fail(self, reason: str) -> UnsupportedInterventionLayout:
        return UnsupportedInterventionLayout(self.layer_cls, reason)

    # -- classification ------------------------------------------------------

    def kind(self, val: _Val) -> _Kind:
        if val.stream is not None:
            return _Kind.STREAM
        attn = bool(val.calls & self.attention)
        mlp = bool(val.calls & self.mlp)
        if attn and mlp:
            return _Kind.MIXED
        if attn:
            return _Kind.ATTN
        if mlp:
            return _Kind.MLP
        return _Kind.INERT

    def is_self_child_call(self, func: ast.expr) -> str | None:
        """`child` if `func` is `self.<child>` for a direct child module."""
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == self.self_name
            and func.attr in self.children
        ):
            return func.attr
        return None

    # -- expressions ---------------------------------------------------------

    def eval(self, node: ast.expr | None, env: dict[str, _Val], st: _State) -> _Val:
        if node is None:
            return _OTHER
        if isinstance(node, ast.Name):
            return env.get(node.id, _OTHER)
        if isinstance(node, ast.Constant):
            return _OTHER
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return self.eval_add(node, env, st)
            return _join(self.eval(node.left, env, st), self.eval(node.right, env, st))
        if isinstance(node, ast.UnaryOp):
            return _join(self.eval(node.operand, env, st))
        if isinstance(node, ast.BoolOp):
            return _join(*(self.eval(v, env, st) for v in node.values))
        if isinstance(node, ast.Compare):
            parts = [node.left, *node.comparators]
            return _join(*(self.eval(p, env, st) for p in parts))
        if isinstance(node, ast.Call):
            return self.eval_call(node, env, st)
        if isinstance(node, ast.Attribute):
            return _join(self.eval(node.value, env, st))
        if isinstance(node, ast.Subscript):
            return _join(self.eval(node.value, env, st), self.eval(node.slice, env, st))
        if isinstance(node, ast.Slice):
            parts = (node.lower, node.upper, node.step)
            return _join(*(self.eval(p, env, st) for p in parts))
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return _join(*(self.eval(e, env, st) for e in node.elts))
        if isinstance(node, ast.Dict):
            return _join(*(self.eval(v, env, st) for v in node.values))
        if isinstance(node, ast.Starred):
            return self.eval(node.value, env, st)
        if isinstance(node, ast.IfExp):
            parts = (node.test, node.body, node.orelse)
            return _join(*(self.eval(p, env, st) for p in parts))
        if isinstance(node, ast.NamedExpr):
            val = self.eval(node.value, env, st)
            env[node.target.id] = val
            return val
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            return _OTHER
        if not self.touches_tracked(node, env):
            # Lambdas, comprehensions and the like over untracked values only.
            return _OTHER
        raise self.fail(
            f"line {getattr(node, 'lineno', '?')} uses a residual-stream value in "
            f"a {type(node).__name__} expression, which cannot be followed"
        )

    def eval_add(self, node: ast.BinOp, env: dict[str, _Val], st: _State) -> _Val:
        terms = _flatten_add(node)
        vals = [self.eval(t, env, st) for t in terms]
        kinds = [self.kind(v) for v in vals]
        streams = [i for i, k in enumerate(kinds) if k is _Kind.STREAM]
        others = [k for k in kinds if k is not _Kind.STREAM]
        if len(streams) == 1 and others and set(others) <= {_Kind.ATTN, _Kind.MLP}:
            (index,) = streams
            depth = vals[index].stream
            assert depth is not None
            branches = [
                (t, v, k)
                for j, (t, v, k) in enumerate(zip(terms, vals, kinds))
                if j != index
            ]
            st.adds.append(
                _ResidualAdd(
                    node=node,
                    stream=terms[index],
                    attn=tuple((t, v) for t, v, k in branches if k is _Kind.ATTN),
                    mlp=tuple((t, v) for t, v, k in branches if k is _Kind.MLP),
                )
            )
            return _Val(stream=depth + 1)
        st.plain_adds.add(node)
        return _join(*vals)

    def eval_call(self, node: ast.Call, env: dict[str, _Val], st: _State) -> _Val:
        func = node.func
        args = [self.eval(a, env, st) for a in node.args]
        args += [self.eval(k.value, env, st) for k in node.keywords]
        if (child := self.is_self_child_call(func)) is not None:
            inputs = _join(*args)
            mlp_calls = inputs.mlp_calls
            if child in self.mlp:
                mlp_calls = mlp_calls | {node}
            return _Val(None, inputs.calls | {child}, mlp_calls)
        tracked_args = any(a.tracked for a in args)
        if isinstance(func, ast.Attribute):
            if (
                isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"
            ):
                raise self.fail("its forward delegates to super(), which is not analysed")
            is_self_attr = (
                isinstance(func.value, ast.Name) and func.value.id == self.self_name
            )
            if is_self_attr and not self.has_attribute(func.attr):
                raise _InfeasiblePath
            if is_self_attr and tracked_args:
                raise self.fail(
                    f"line {node.lineno} passes a residual-stream value to "
                    f"self.{func.attr}(), which is not a submodule, so the "
                    "sublayers it may run cannot be seen"
                )
            receiver = self.eval(func.value, env, st)
            name = func.attr
            if receiver.tracked and name.endswith("_") and not name.endswith("__"):
                raise self.fail(
                    f"line {node.lineno} updates a residual-stream value in place "
                    f"with .{name}()"
                )
            args.append(receiver)
        if tracked_args and self.writes_through_args(node):
            raise self.fail(f"line {node.lineno} writes a residual-stream value in place")
        return _join(*args)

    @staticmethod
    def writes_through_args(node: ast.Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "out":
                return True
            if keyword.arg == "inplace" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                return True
        return False

    def touches_tracked(self, node: ast.AST, env: dict[str, _Val]) -> bool:
        """Whether `node` reads or writes a tracked name or calls a sublayer."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and env.get(child.id, _OTHER).tracked:
                return True
            if isinstance(child, ast.Call) and self.is_self_child_call(child.func):
                return True
        return False

    # -- statements ----------------------------------------------------------

    def bind(self, target: ast.expr, val: _Val, env: dict[str, _Val]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = val
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self.bind(elt, val, env)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, val, env)
        elif isinstance(target, ast.Subscript):
            base = target.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and env.get(base.id, _OTHER).tracked:
                raise self.fail(
                    f"line {target.lineno} writes into a residual-stream value in place"
                )
        # Attribute stores (`self.x = ...`) do not affect local dataflow.

    def assign(
        self,
        targets: list[ast.expr],
        value: ast.expr,
        env: dict[str, _Val],
        st: _State,
    ) -> None:
        if (
            len(targets) == 1
            and isinstance(targets[0], (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(targets[0].elts) == len(value.elts)
            and not any(isinstance(e, ast.Starred) for e in targets[0].elts)
        ):
            vals = [self.eval(e, env, st) for e in value.elts]
            for elt, val in zip(targets[0].elts, vals):
                self.bind(elt, val, env)
            return
        val = self.eval(value, env, st)
        for target in targets:
            self.bind(target, val, env)

    def returned(self, node: ast.expr | None, env: dict[str, _Val], st: _State) -> _Val:
        if isinstance(node, ast.Tuple) and node.elts:
            vals = [self.eval(e, env, st) for e in node.elts]
            return vals[0]
        return self.eval(node, env, st)

    def fork(self) -> None:
        self.num_paths += 1
        if self.num_paths > MAX_PATHS:
            raise self.fail(f"its forward has more than {MAX_PATHS} control-flow paths")

    def run_feasible(
        self, stmts: list[ast.stmt], env: dict[str, _Val], st: _State
    ) -> list[_Path]:
        try:
            return self.run(stmts, env, st)
        except _InfeasiblePath:
            return []

    def run(self, stmts: list[ast.stmt], env: dict[str, _Val], st: _State) -> list[_Path]:
        for i, stmt in enumerate(stmts):
            rest = stmts[i + 1 :]
            if isinstance(stmt, ast.If):
                self.eval(stmt.test, env, st)
                self.fork()
                return self.run_feasible(
                    stmt.body + rest, dict(env), st.copy()
                ) + self.run_feasible(stmt.orelse + rest, dict(env), st.copy())
            if isinstance(stmt, ast.Return):
                val = self.returned(stmt.value, env, st)
                return [_Path(tuple(st.adds), frozenset(st.plain_adds), val)]
            if isinstance(stmt, ast.Assign):
                self.assign(stmt.targets, stmt.value, env, st)
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None:
                    self.assign([stmt.target], stmt.value, env, st)
            elif isinstance(stmt, ast.AugAssign):
                target_val = (
                    self.eval(stmt.target, env, st)
                    if isinstance(stmt.target, ast.Name)
                    else _OTHER
                )
                value_val = self.eval(stmt.value, env, st)
                if target_val.tracked or value_val.tracked:
                    raise self.fail(
                        f"line {stmt.lineno} uses an augmented assignment on a "
                        "residual-stream value, which updates it in place"
                    )
                self.bind(stmt.target, _OTHER, env)
            elif isinstance(stmt, ast.Expr):
                self.eval(stmt.value, env, st)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    self.eval(item.context_expr, env, st)
                return self.run(stmt.body + rest, env, st)
            elif isinstance(stmt, ast.Assert):
                self.eval(stmt.test, env, st)
            elif isinstance(stmt, ast.Raise):
                return []
            elif isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env.pop(target.id, None)
            elif isinstance(stmt, _IGNORED_STATEMENTS):
                continue
            elif isinstance(stmt, _OPAQUE_STATEMENTS):
                if self.touches_tracked(stmt, env):
                    raise self.fail(
                        f"line {stmt.lineno} handles residual-stream values in a "
                        f"{type(stmt).__name__} statement, which cannot be followed"
                    )
            else:
                raise self.fail(
                    f"line {stmt.lineno} is an unsupported {type(stmt).__name__} "
                    "statement"
                )
        return [_Path(tuple(st.adds), frozenset(st.plain_adds), None)]

    # -- layouts -------------------------------------------------------------

    def mlp_input(self, val: _Val) -> MlpInputSite:
        calls = sorted(val.mlp_calls, key=lambda c: (c.lineno, c.col_offset))
        if len(calls) != 1:
            raise self.fail(
                f"its MLP branch calls {len(calls)} MLP sublayers, expected exactly one"
            )
        (call,) = calls
        assert isinstance(call.func, ast.Attribute)
        child = call.func.attr
        if call.args and not isinstance(call.args[0], ast.Starred):
            return MlpInputSite(call=call, child=child, keyword=None)
        keywords = [k for k in call.keywords if k.arg in ("hidden_states", "x")]
        if not call.args and len(keywords) == 1:
            return MlpInputSite(call=call, child=child, keyword=keywords[0])
        raise self.fail(
            f"line {call.lineno} calls self.{child} without an identifiable "
            "hidden-states argument"
        )

    def layout(self, path: _Path) -> ResidualLayout:
        adds = path.adds
        returned = path.returned
        if returned is None or returned.stream is None or returned.stream != len(adds):
            raise self.fail(
                "a control-flow path through its forward does not return the "
                f"residual stream (found {len(adds)} residual add(s) on it)"
            )

        def describe() -> str:
            return ", ".join(
                f"line {a.node.lineno}: {len(a.attn)} attention + {len(a.mlp)} MLP "
                "branch(es)"
                for a in adds
            )

        if len(adds) == 2:
            attn, mlp = adds
            if len(attn.attn) == 1 and not attn.mlp and len(mlp.mlp) == 1 and not mlp.attn:
                (attn_branch, _), (mlp_branch, mlp_val) = attn.attn[0], mlp.mlp[0]
                return SequentialResidual(
                    attn_add=attn.node,
                    attn_stream=attn.stream,
                    attn_branch=attn_branch,
                    mlp_add=mlp.node,
                    mlp_stream=mlp.stream,
                    mlp_branch=mlp_branch,
                    mlp_input=self.mlp_input(mlp_val),
                )
        if len(adds) == 1:
            (add,) = adds
            if len(add.attn) == 1 and len(add.mlp) == 1:
                (attn_branch, _), (mlp_branch, mlp_val) = add.attn[0], add.mlp[0]
                return ParallelResidual(
                    add=add.node,
                    stream=add.stream,
                    attn_branch=attn_branch,
                    mlp_branch=mlp_branch,
                    mlp_input=self.mlp_input(mlp_val),
                )
        raise self.fail(
            "expected an attention residual add followed by an MLP residual add, "
            "or one parallel residual add over both branches; found "
            f"{describe() or 'no residual adds'}"
        )

    def plan(
        self, funcdef: ast.FunctionDef, stream_param: str
    ) -> tuple[ResidualLayout, ...]:
        env = {stream_param: _Val(stream=0)}
        paths = self.run_feasible(funcdef.body, env, _State())
        if not paths:
            raise self.fail("every control-flow path through its forward raises")
        layouts: dict[tuple[int, ...], ResidualLayout] = {}
        residual_nodes: set[ast.BinOp] = set()
        plain_nodes: set[ast.BinOp] = set()
        for path in paths:
            layout = self.layout(path)
            residual_nodes.update(a.node for a in path.adds)
            plain_nodes.update(path.plain_adds)
            key = tuple(id(a.node) for a in path.adds)
            layouts.setdefault(key, layout)
        if residual_nodes & plain_nodes:
            raise self.fail(
                "an add is a residual add on one control-flow path but not on another"
            )
        roles: dict[ast.AST, str] = {}
        for layout in layouts.values():
            for node, role in _roles(layout):
                if roles.setdefault(node, role) != role:
                    raise self.fail(
                        f"line {node.lineno} plays different residual roles on "
                        "different control-flow paths"
                    )
        return tuple(layouts.values())


def _roles(layout: ResidualLayout) -> list[tuple[ast.AST, str]]:
    if isinstance(layout, SequentialResidual):
        return [
            (layout.attn_add, "attn_add"),
            (layout.mlp_add, "mlp_add"),
            (layout.mlp_input.call, "mlp_call"),
        ]
    return [(layout.add, "parallel_add"), (layout.mlp_input.call, "mlp_call")]


def classify_children(
    layer: nn.Module, is_attention: Callable[[nn.Module], bool]
) -> tuple[frozenset[str], frozenset[str]]:
    """`(attention, heavy)` direct-child names of `layer`."""
    children = dict(layer.named_children())
    attention = frozenset(
        name
        for name, child in children.items()
        if any(is_attention(m) for m in child.modules())
    )
    heavy = frozenset(name for name, child in children.items() if _is_heavy(child))
    return attention, heavy | attention


def analyze_layer(
    layer: nn.Module,
    is_attention: Callable[[nn.Module], bool] = is_vllm_attention,
) -> LayerPlan:
    """Locate the residual adds and MLP call of `layer`'s forward.

    Raises `UnsupportedInterventionLayout` if they cannot be located with
    certainty. The returned plan owns a freshly parsed tree, so it can be
    rewritten without affecting other plans."""
    layer_cls = type(layer)
    name = layer_cls.__name__
    if "forward" in vars(layer):
        raise UnsupportedInterventionLayout(
            name, "its forward was already replaced on the instance by another rewrite"
        )
    try:
        funcdef, fn = recover_forward(layer_cls)
    except (OSError, TypeError, ValueError, SyntaxError) as exc:
        raise UnsupportedInterventionLayout(
            name, f"its forward source cannot be recovered ({exc})"
        ) from exc
    params = [*funcdef.args.posonlyargs, *funcdef.args.args]
    if len(params) < 2:
        raise UnsupportedInterventionLayout(
            name, "its forward takes no positional hidden-states parameter"
        )
    attention, heavy = classify_children(layer, is_attention)
    if not attention:
        raise UnsupportedInterventionLayout(
            name, "none of its child modules dispatches attention"
        )
    interpreter = _Interpreter(
        name,
        self_name=params[0].arg,
        children=dict(layer.named_children()),
        attention=attention,
        heavy=heavy,
        has_attribute=lambda attr: getattr(layer, attr, None) is not None,
    )
    layouts = interpreter.plan(funcdef, stream_param=params[1].arg)
    return LayerPlan(
        layer_cls=layer_cls,
        funcdef=funcdef,
        fn=fn,
        self_name=params[0].arg,
        layouts=layouts,
    )
