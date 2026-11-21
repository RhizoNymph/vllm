# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer keys of looped models: one intervention site per (layer, pass).

A looped model (Inkling 2 Mini: prelude layers 0-5, core layers 6-17 run 3
passes, coda 18-23) visits each core layer once per pass. Steering,
clamping, patching, SAE and capture all key their per-site state by one
integer *layer key*; on a looped model every visit gets its own key:

- a prelude / coda layer `L` (one visit): key `L`;
- a core layer `L` in pass `p` (0-based): key `L + (p + 1) * N`, with `N`
  the number of decoder layers (`num_hidden_layers`).

User-facing addressing is `(layer, pass)`:

- a plain layer index `L` targets **every** visit of `L` (all passes);
- `"L@p"` (a string key, or `layer` + `pass` fields in list-shaped specs)
  targets pass `p` only; on a prelude / coda layer only `p = 0` exists.

Ingestion (`SamplingParams.__post_init__`, the steering / clamp / patch /
capture / SAE coercers) rewrites user keys to layer keys with
`expand_layer_key`. The rewrite is idempotent: a plain core-layer index
always expands, and keys `>= N` are already layer keys, so specs can be
re-validated in every process they cross. Non-looped models keep plain
indices and reject a pass index.

The space is process-global (`install_loop_layer_space`), set from the
model config by the input processor, the engine core and the worker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

_PASS_SEPARATOR = "@"

T = TypeVar("T")


class LoopLayerKeyError(ValueError):
    """A layer / pass reference that does not name a site of the model."""


@dataclass(frozen=True)
class LoopLayerSpace:
    """The layer keys of a looped model."""

    num_layers: int
    core_start: int
    core_end: int
    num_passes: int

    def __post_init__(self) -> None:
        if not 0 <= self.core_start <= self.core_end <= self.num_layers:
            raise ValueError(f"invalid loop core [{self.core_start}, {self.core_end})")
        if self.num_passes < 1:
            raise ValueError("a loop needs at least one pass")

    @classmethod
    def from_text_config(cls, config: Any) -> LoopLayerSpace | None:
        """The space of an HF text config with a `loop_config` (Inkling2
        loop checkpoints), else None."""
        loop = getattr(config, "loop_config", None)
        if not loop:
            return None
        passes = getattr(config, "num_loop_passes", None) or loop["num_passes"]
        return cls(
            num_layers=int(config.num_hidden_layers),
            core_start=int(loop["core_start"]),
            core_end=int(loop["core_end"]),
            num_passes=int(passes),
        )

    @property
    def core(self) -> range:
        return range(self.core_start, self.core_end)

    @property
    def num_keys(self) -> int:
        """An exclusive upper bound of the layer keys."""
        return (self.num_passes + 1) * self.num_layers

    def key(self, layer: int, pass_idx: int) -> int:
        """The layer key of `layer`'s visit in pass `pass_idx`."""
        if not 0 <= layer < self.num_layers:
            raise LoopLayerKeyError(
                f"layer {layer} out of range [0, {self.num_layers})"
            )
        if layer not in self.core:
            if pass_idx != 0:
                raise LoopLayerKeyError(
                    f"layer {layer} is outside the loop core "
                    f"[{self.core_start}, {self.core_end}) and runs once; "
                    f"only pass 0 exists, got pass {pass_idx}"
                )
            return layer
        if not 0 <= pass_idx < self.num_passes:
            raise LoopLayerKeyError(
                f"pass {pass_idx} out of range [0, {self.num_passes}) for "
                f"loop core layer {layer}"
            )
        return layer + (pass_idx + 1) * self.num_layers

    def keys(self, layer: int, pass_idx: int | None = None) -> list[int]:
        """The keys of `layer` in pass `pass_idx`, or of every pass."""
        if pass_idx is not None:
            return [self.key(layer, pass_idx)]
        if layer in self.core:
            return [self.key(layer, p) for p in range(self.num_passes)]
        return [self.key(layer, 0)]

    def is_key(self, key: int) -> bool:
        """Whether `key` is a layer key (a site) of this space."""
        if 0 <= key < self.num_layers:
            return key not in self.core
        layer, rest = key % self.num_layers, key // self.num_layers - 1
        return layer in self.core and 0 <= rest < self.num_passes

    def decode(self, key: int) -> tuple[int, int]:
        """`(layer, pass)` of a layer key."""
        if not self.is_key(key):
            raise LoopLayerKeyError(f"{key} is not a layer key of this model")
        if key < self.num_layers:
            return key, 0
        return key % self.num_layers, key // self.num_layers - 1

    def label(self, key: int) -> str:
        """`"L@p"` for a core visit, `"L"` otherwise."""
        layer, pass_idx = self.decode(key)
        return (
            f"{layer}{_PASS_SEPARATOR}{pass_idx}" if layer in self.core else str(layer)
        )

    def visit_keys(self) -> list[int]:
        """Every site's key in execution order (prelude, each pass of the
        core, coda)."""
        keys = list(range(self.core_start))
        for p in range(self.num_passes):
            keys += [self.key(layer, p) for layer in self.core]
        keys += list(range(self.core_end, self.num_layers))
        return keys


_ACTIVE: LoopLayerSpace | None = None
_INSTALLED = False


def install_loop_layer_space(space: LoopLayerSpace | None) -> None:
    """Set this process's layer-key space (None: a non-looped model)."""
    global _ACTIVE, _INSTALLED
    _ACTIVE, _INSTALLED = space, True


def install_loop_layer_space_from_config(model_config: Any) -> None:
    """`install_loop_layer_space` for a vLLM `ModelConfig`."""
    text_config = getattr(model_config, "hf_text_config", None)
    install_loop_layer_space(
        None if text_config is None else LoopLayerSpace.from_text_config(text_config)
    )


def get_loop_layer_space() -> LoopLayerSpace | None:
    """This process's layer-key space; before an explicit install, the one of
    the current vLLM config (if any)."""
    if _INSTALLED:
        return _ACTIVE
    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None:
        return None
    return LoopLayerSpace.from_text_config(text_config)


def parse_layer_key(key: object, label: str = "layer") -> tuple[int, int | None]:
    """`(layer, pass)` of a user layer key: an int (or its string) is a
    layer, all passes; `"L@p"` is layer `L` in pass `p`."""
    if type(key) is int:
        if key < 0:
            raise LoopLayerKeyError(f"{label} must be non-negative, got {key!r}")
        return key, None
    if isinstance(key, str):
        text = key.strip()
        layer_text, sep, pass_text = text.partition(_PASS_SEPARATOR)
        if layer_text.isdigit() and (not sep or pass_text.isdigit()):
            return int(layer_text), (int(pass_text) if sep else None)
    raise LoopLayerKeyError(
        f"{label} must be a non-negative integer or 'layer@pass', got {key!r}"
    )


def expand_layer_key(
    key: object,
    pass_idx: int | None = None,
    space: LoopLayerSpace | None = None,
    label: str = "layer",
) -> list[int]:
    """The layer keys a user reference targets (see the module docstring).

    `pass_idx` is an explicit pass (list-shaped specs' `pass` field); it may
    not be combined with an `"L@p"` key. `space` defaults to the process's.
    """
    layer, key_pass = parse_layer_key(key, label)
    if pass_idx is not None:
        if key_pass is not None:
            raise LoopLayerKeyError(f"{label} {key!r} already names a pass")
        if type(pass_idx) is not int or pass_idx < 0:
            raise LoopLayerKeyError(
                f"{label} pass must be a non-negative integer, got {pass_idx!r}"
            )
        key_pass = pass_idx
    if space is None:
        space = get_loop_layer_space()
    if space is None:
        if key_pass is not None:
            raise LoopLayerKeyError(
                f"{label} {key!r}: a pass index needs a looped model; this "
                "model runs each layer once"
            )
        return [layer]
    if key_pass is None and layer >= space.num_layers:
        # Already a layer key (specs are re-validated in every process).
        if not space.is_key(layer):
            raise LoopLayerKeyError(
                f"{label} {layer} out of range [0, {space.num_layers})"
            )
        return [layer]
    try:
        return space.keys(layer, key_pass)
    except LoopLayerKeyError as exc:
        raise LoopLayerKeyError(f"{label} {key!r}: {exc}") from None


def expand_layer_map(
    layers: Mapping[Any, T],
    combine: Callable[[T, T], T],
    space: LoopLayerSpace | None = None,
    label: str = "layer",
) -> dict[int, T]:
    """`{user key: value}` -> `{layer key: value}`; values landing on one
    key (e.g. `"7"` and `"7@1"`) are merged with `combine(earlier,
    later)`."""
    if space is None:
        space = get_loop_layer_space()
    out: dict[int, T] = {}
    for user_key, value in layers.items():
        for key in expand_layer_key(user_key, space=space, label=label):
            out[key] = combine(out[key], value) if key in out else value
    return out


def expand_layer_list(
    layers: Iterable[object],
    space: LoopLayerSpace | None = None,
    label: str = "layer",
) -> list[int]:
    """Ordered, de-duplicated layer keys of user layer references."""
    if space is None:
        space = get_loop_layer_space()
    seen: dict[int, None] = {}
    for user_key in layers:
        for key in expand_layer_key(user_key, space=space, label=label):
            seen.setdefault(key, None)
    return list(seen)


def num_layer_keys(model_config: Any) -> int:
    """An exclusive bound of the model's layer keys (`num_hidden_layers`
    for non-looped models)."""
    text_config = getattr(model_config, "hf_text_config", None)
    space = (
        None if text_config is None else LoopLayerSpace.from_text_config(text_config)
    )
    if space is not None:
        return space.num_keys
    return model_config.get_total_num_hidden_layers()
