# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU helpers for pinning where a decoder layer emits its intervention hooks.

Steering, clamps, dynamic steering, patching, activation capture, and SAE
steering all dispatch through ``apply_layer_steering`` / ``apply_block_steering``
at the same call sites, so recording what those two functions receive pins the
contract for every intervention family at once.

Decoder layers are built with ``object.__new__`` and stubbed collaborators (the
real constructors need a device, a distributed group, and an attention backend),
following ``test_commandr_parallel_block_reduce.py``.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

POST_BLOCK = "post_block"
ALL_HOOKS = ["pre_attn", "post_attn", "mlp_in", "mlp_out", POST_BLOCK]
REGISTRATION_FNS = frozenset(
    {
        "register_steering_buffers",
        "register_layer_steering_buffers",
        "register_qwen3_5_steering_buffers",
    }
)


@dataclass
class HookRecorder:
    """Ordered record of (hook point name, observed tensor) pairs."""

    calls: list[tuple[str, torch.Tensor]] = field(default_factory=list)

    @property
    def order(self) -> list[str]:
        return [name for name, _ in self.calls]

    def seen(self, name: str, occurrence: int = 0) -> torch.Tensor:
        matches = [t for n, t in self.calls if n == name]
        return matches[occurrence]


def record_hooks(monkeypatch, *modules: ModuleType) -> HookRecorder:
    """Replace the hook entry points in ``modules`` with pass-through recorders.

    ``post_block`` records ``residual + hidden_states``, the true block output
    that capture observes in ``apply_block_steering``.
    """
    rec = HookRecorder()

    def _layer(module, hidden_states, hook_point):
        rec.calls.append((hook_point.value, hidden_states.detach().clone()))
        return hidden_states

    def _block(module, hidden_states, residual):
        rec.calls.append((POST_BLOCK, (residual + hidden_states).detach().clone()))
        return hidden_states, residual

    for mod in modules:
        if hasattr(mod, "apply_layer_steering"):
            monkeypatch.setattr(mod, "apply_layer_steering", _layer)
        if hasattr(mod, "apply_block_steering"):
            monkeypatch.setattr(mod, "apply_block_steering", _block)
    return rec


class Const(nn.Module):
    """Sublayer stub returning a constant-filled tensor of the input's shape."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        x = next(a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor))
        if "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        elif len(args) >= 2 and isinstance(args[1], torch.Tensor):
            # (positions, hidden_states, ...) signatures
            x = args[1]
        return torch.full_like(x, self.value)


class Scale(nn.Module):
    """Norm stub: plain call scales; ``(x, residual)`` call is a fused add.

    The fused form mirrors vLLM's ``RMSNorm(x, residual)``: the residual is
    updated to ``x + residual`` and the scaled sum is returned as the branch.
    """

    def __init__(self, factor: float = 1.0):
        super().__init__()
        self.factor = factor

    def forward(self, x, residual=None):
        if residual is None:
            return x * self.factor
        s = x + residual
        return s * self.factor, s


def bare(cls: type, **attrs) -> nn.Module:
    """Instantiate ``cls`` without running its ``__init__``."""
    obj = object.__new__(cls)
    nn.Module.__init__(obj)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# Static wiring checks
# ---------------------------------------------------------------------------


def _called_names(fn) -> list[str]:
    """Names called in ``fn``; ``fn`` may be a function or ``(cls, "method")``.

    The class form reads the method from the class source, so it sees through
    decorators such as ``support_torch_compile`` that replace ``__init__``.
    """
    if isinstance(fn, tuple):
        cls, method = fn
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        root = next(
            n
            for n in cls_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == method
        )
    else:
        root = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    out: list[str] = []
    for node in ast.walk(root):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.append(f.id)
            elif isinstance(f, ast.Attribute):
                out.append(f.attr)
    return out


def calls(fn, name: str) -> bool:
    """Whether ``fn``'s source contains a call to ``name``."""
    return name in _called_names(fn)


def registers_buffers(fn) -> bool:
    """Whether ``fn`` calls one of the steering-buffer registration helpers."""
    return bool(REGISTRATION_FNS.intersection(_called_names(fn)))


# ---------------------------------------------------------------------------
# Real steering buffers on CPU
# ---------------------------------------------------------------------------


def fake_vllm_config(
    max_num_batched_tokens: int = 8, *, steering: bool = True, dtype=torch.float32
):
    """The slice of ``VllmConfig`` that steering-buffer sizing reads.

    A real ``VllmConfig`` cannot be built on a Triton-less x86 CPU host, and
    buffer registration only reads these fields.
    """
    from vllm.config.steering import SteeringConfig

    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        steering_config=SteeringConfig() if steering else None,
        patch_config=None,
        capture_consumers_config=None,
        model_config=SimpleNamespace(dtype=dtype),
    )


def set_global_vector(
    layer: nn.Module, hook: str, vector: torch.Tensor, num_tokens: int
) -> None:
    """Point the first ``num_tokens`` tokens at table row 1 holding ``vector``."""
    from vllm.model_executor.layers.steering import (
        HOOK_POINT_TABLE_ATTR,
        SteeringHookPoint,
    )

    attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook)]
    getattr(layer, attr)[1] = vector
    getattr(layer, f"{attr}_any_active").fill_(bool(vector.abs().sum() > 0))
    layer.steering_index[:num_tokens] = 1
