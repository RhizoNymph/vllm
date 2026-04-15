# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request activation storing types + pure helpers.

This module is intentionally torch-free so unit tests for selector
expansion, slugging, and position resolution run fast and don't pull in
CUDA initialization. Anything that needs a GPU lives in later phases
(capture manager, writer pool, runner integration).

The spec is :class:`ActivationStoringSpec`. A request opts into capture
by constructing one and attaching it to
:class:`vllm.SamplingParams.activation_storing`. The server-side
admission validator resolves the layer and position selectors against
``num_hidden_layers`` and the request's prompt length; the runner-side
manager uses the resolved values to build per-step capture plans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Hook-point literal + valid names
# ---------------------------------------------------------------------------

# Hook names are mirrored from ``vllm.model_executor.layers.steering``. We
# redeclare them here (rather than importing) to keep this module torch-free;
# the capture feature piggybacks on the existing steering hook-point enum, so
# whenever steering grows a new hook point, this tuple must be updated in
# lockstep. A regression test in the admission validator pins the set so the
# drift can't go unnoticed.
HookPointName = Literal["pre_attn", "post_attn", "post_mlp"]

VALID_ACTIVATION_HOOK_NAMES: frozenset[str] = frozenset(
    ("pre_attn", "post_attn", "post_mlp")
)


# ---------------------------------------------------------------------------
# Layer and position selector types
# ---------------------------------------------------------------------------

# ``HookLayerSelector`` describes which layers to capture under a given hook.
# Three shapes are allowed at the wire level:
#
#   "all"                                       — every layer
#   [0, 12, 24]                                 — explicit list
#   {"layers": [1, 31], "ranges": [[10, 20]]}   — mixed form, union of both
#
# The dict form's ``ranges`` are inclusive on both ends to match human
# intuition ("layers 10 through 20" = ``range(10, 21)``), explicitly
# diverging from Python's half-open ``range``. See :func:`expand_hook_layers`.
HookLayerSelector = str | list[int] | dict[str, Any]

# ``PositionSelector`` describes which logical token positions to capture.
PositionSelector = str | list[int]

# Literal position-kind strings that the admission validator accepts.
VALID_POSITION_KINDS: frozenset[str] = frozenset(
    ("last_prompt", "all_prompt", "all_generated", "all")
)


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------

# The spec uses this exact regex so a single canonical form covers every
# path segment. Any character outside ``[A-Za-z0-9._-]`` becomes ``_``.
_SLUG_REGEX = re.compile(r"[^a-zA-Z0-9._-]")

# Hard cap on any slugged segment. Matches the admission validator.
SLUG_MAX_LEN = 256


class ActivationStoringSlugError(ValueError):
    """Raised when ``tag`` or ``request_id`` cannot be slugged safely.

    Mirrors the shape of :class:`vllm.exceptions.VLLMValidationError` at
    the ``ValueError`` level so callers can ``except ValueError`` without
    needing to import the exception class.
    """


def slug(name: str) -> str:
    """Slug a ``tag`` or ``request_id`` into a safe path segment.

    Applies ``re.sub(r'[^a-zA-Z0-9._-]', '_', name)`` and rejects:

    - empty input,
    - any input containing ``..`` (path traversal),
    - any input starting with ``/`` (absolute path escape),
    - any input whose pre-slug form exceeds 256 characters.

    Rejection raises :class:`ActivationStoringSlugError` with a descriptive
    message so the OpenAI entrypoint can surface a useful HTTP 400.
    """
    if not isinstance(name, str):
        raise ActivationStoringSlugError(
            f"activation storing name must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ActivationStoringSlugError("activation storing name must be non-empty")
    if len(name) > SLUG_MAX_LEN:
        raise ActivationStoringSlugError(
            f"activation storing name must be at most {SLUG_MAX_LEN} characters, "
            f"got {len(name)}"
        )
    if ".." in name:
        raise ActivationStoringSlugError(
            f"activation storing name must not contain '..': {name!r}"
        )
    if name.startswith("/"):
        raise ActivationStoringSlugError(
            f"activation storing name must not start with '/': {name!r}"
        )
    return _SLUG_REGEX.sub("_", name)


# ---------------------------------------------------------------------------
# Layer selector expansion
# ---------------------------------------------------------------------------


def _expand_layer_list(
    raw: list[int],
    *,
    num_hidden_layers: int,
    where: str,
) -> list[int]:
    """Validate and return a copy of an explicit layer list."""
    result: list[int] = []
    for i, value in enumerate(raw):
        # ``bool`` is a subclass of ``int`` — reject it so ``[True]`` doesn't
        # silently become ``[1]``.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{where}[{i}] must be an int, got {type(value).__name__}"
            )
        if value < 0 or value >= num_hidden_layers:
            raise ValueError(
                f"{where}[{i}] = {value} is out of range for a model with "
                f"num_hidden_layers={num_hidden_layers}"
            )
        result.append(value)
    return result


def _expand_ranges(
    raw: list[Any],
    *,
    num_hidden_layers: int,
    where: str,
) -> list[int]:
    """Expand an inclusive-range list into individual layer indices."""
    result: list[int] = []
    for i, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"{where}[{i}] must be a 2-element [start, end] pair, "
                f"got {pair!r}"
            )
        start, end = pair
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(
                f"{where}[{i}] must be a pair of ints, got {pair!r}"
            )
        if start > end:
            raise ValueError(
                f"{where}[{i}] has start > end ({start} > {end})"
            )
        if start < 0 or end >= num_hidden_layers:
            raise ValueError(
                f"{where}[{i}] = [{start}, {end}] is out of range for a "
                f"model with num_hidden_layers={num_hidden_layers}"
            )
        # Inclusive on both ends.
        result.extend(range(start, end + 1))
    return result


def expand_hook_layers(
    selector: HookLayerSelector,
    num_hidden_layers: int,
    *,
    where: str = "hooks[<hook>]",
) -> list[int]:
    """Resolve a :data:`HookLayerSelector` into a sorted, deduped list of
    layer indices.

    Accepts the three wire shapes described in the spec:

    - ``"all"`` → every layer in ``[0, num_hidden_layers)``.
    - ``list[int]`` → the literal list, validated in-range.
    - ``{"layers": list[int] | None, "ranges": list[[int, int]] | None}``
      → union of ``layers`` and the expanded (inclusive) ranges.

    Every index in the output is guaranteed to satisfy
    ``0 <= idx < num_hidden_layers``. The output is sorted and deduped.

    :param where: Human-readable breadcrumb used in error messages
        (``"hooks['post_mlp']"``, etc.).
    """
    if num_hidden_layers <= 0:
        raise ValueError(
            f"num_hidden_layers must be positive, got {num_hidden_layers}"
        )

    if isinstance(selector, str):
        if selector != "all":
            raise ValueError(
                f"{where} string form must be 'all', got {selector!r}"
            )
        return list(range(num_hidden_layers))

    if isinstance(selector, list):
        resolved = _expand_layer_list(
            selector, num_hidden_layers=num_hidden_layers, where=where
        )
        return sorted(set(resolved))

    if isinstance(selector, dict):
        allowed = {"layers", "ranges"}
        extra = set(selector.keys()) - allowed
        if extra:
            raise ValueError(
                f"{where} dict form has unexpected keys {sorted(extra)}; "
                f"allowed keys: {sorted(allowed)}"
            )
        layers_raw = selector.get("layers")
        ranges_raw = selector.get("ranges")
        if layers_raw is None and ranges_raw is None:
            raise ValueError(
                f"{where} dict form must set 'layers' and/or 'ranges'; got empty"
            )

        collected: list[int] = []
        if layers_raw is not None:
            if not isinstance(layers_raw, list):
                raise ValueError(
                    f"{where}.layers must be a list of ints, got "
                    f"{type(layers_raw).__name__}"
                )
            collected.extend(
                _expand_layer_list(
                    layers_raw,
                    num_hidden_layers=num_hidden_layers,
                    where=f"{where}.layers",
                )
            )
        if ranges_raw is not None:
            if not isinstance(ranges_raw, list):
                raise ValueError(
                    f"{where}.ranges must be a list of [start, end] pairs, "
                    f"got {type(ranges_raw).__name__}"
                )
            collected.extend(
                _expand_ranges(
                    ranges_raw,
                    num_hidden_layers=num_hidden_layers,
                    where=f"{where}.ranges",
                )
            )
        return sorted(set(collected))

    raise ValueError(
        f"{where} must be 'all', list[int], or {{'layers': ..., 'ranges': ...}}; "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# Position selector resolution
# ---------------------------------------------------------------------------


def resolve_positions(
    selector: PositionSelector,
    num_prompt_tokens: int,
    num_generated_tokens: int,
    *,
    where: str = "positions",
) -> list[int]:
    """Resolve a :data:`PositionSelector` into absolute token indices.

    Only handles the cases whose size is known from static request
    metadata. ``"all_generated"`` and ``"all"`` require
    ``num_generated_tokens`` to be provided up front; at admission time
    callers should pass the request's ``max_tokens`` as an upper bound or
    defer resolution to the runner. The returned list is sorted and
    deduped; explicit indices that are out of range raise ``ValueError``.

    Semantics:

    - ``"last_prompt"`` → ``[num_prompt_tokens - 1]``.
    - ``"all_prompt"``  → ``range(0, num_prompt_tokens)``.
    - ``"all_generated"`` →
      ``range(num_prompt_tokens, num_prompt_tokens + num_generated_tokens)``.
    - ``"all"`` →
      ``range(0, num_prompt_tokens + num_generated_tokens)``.
    - ``list[int]`` → the literal list, validated against
      ``[0, num_prompt_tokens + num_generated_tokens)``.
    """
    if num_prompt_tokens <= 0:
        raise ValueError(
            f"num_prompt_tokens must be positive, got {num_prompt_tokens}"
        )
    if num_generated_tokens < 0:
        raise ValueError(
            f"num_generated_tokens must be non-negative, got {num_generated_tokens}"
        )

    if isinstance(selector, str):
        if selector not in VALID_POSITION_KINDS:
            raise ValueError(
                f"{where} must be one of {sorted(VALID_POSITION_KINDS)} "
                f"or a list of ints, got {selector!r}"
            )
        if selector == "last_prompt":
            return [num_prompt_tokens - 1]
        if selector == "all_prompt":
            return list(range(0, num_prompt_tokens))
        if selector == "all_generated":
            return list(
                range(num_prompt_tokens, num_prompt_tokens + num_generated_tokens)
            )
        # "all"
        return list(range(0, num_prompt_tokens + num_generated_tokens))

    if isinstance(selector, list):
        total = num_prompt_tokens + num_generated_tokens
        result: list[int] = []
        for i, value in enumerate(selector):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{where}[{i}] must be an int, got {type(value).__name__}"
                )
            if value < 0 or value >= total:
                raise ValueError(
                    f"{where}[{i}] = {value} is out of range for a request "
                    f"with num_prompt_tokens={num_prompt_tokens}, "
                    f"num_generated_tokens={num_generated_tokens}"
                )
            result.append(value)
        return sorted(set(result))

    raise ValueError(
        f"{where} must be one of {sorted(VALID_POSITION_KINDS)} or list[int], "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# ActivationStoringSpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class ActivationStoringSpec:
    """Per-request capture spec.

    A request opts into activation storing by constructing one of these
    and attaching it to ``SamplingParams.activation_storing``. Fields:

    - ``request_id``: client-chosen stem for the ``.bin`` / ``.json``
      filenames. Slugged server-side via :func:`slug`.
    - ``tag``: client-chosen grouping segment in the path. Also slugged.
    - ``hooks``: mapping ``{hook_point_name: layer_selector}``. Must be
      non-empty. Each value is a :data:`HookLayerSelector`; the admission
      validator expands it via :func:`expand_hook_layers`.
    - ``positions``: a :data:`PositionSelector` describing which absolute
      logical positions to capture. The admission validator resolves
      ``"last_prompt"`` / ``"all_prompt"`` / explicit lists immediately;
      ``"all_generated"`` / ``"all"`` stay symbolic until the runner
      observes each step.

    Structural validation runs in ``__post_init__``: non-empty ``hooks``,
    known hook-point names, known position-kind string. Full admission
    validation (in-range layers, prefix-cache collisions, byte budget)
    happens in
    :mod:`vllm.entrypoints.openai.activation_storing_validation`.
    """

    request_id: str
    tag: str
    hooks: dict[str, HookLayerSelector]
    positions: PositionSelector

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError(
                f"activation_storing.request_id must be a non-empty string, "
                f"got {self.request_id!r}"
            )
        if not isinstance(self.tag, str) or not self.tag:
            raise ValueError(
                f"activation_storing.tag must be a non-empty string, "
                f"got {self.tag!r}"
            )
        if not isinstance(self.hooks, dict) or not self.hooks:
            raise ValueError(
                "activation_storing.hooks must be a non-empty dict mapping "
                "hook point name to a layer selector"
            )
        for hook_name in self.hooks:
            if not isinstance(hook_name, str):
                raise ValueError(
                    f"activation_storing.hooks key must be a string, "
                    f"got {type(hook_name).__name__}"
                )
            if hook_name not in VALID_ACTIVATION_HOOK_NAMES:
                raise ValueError(
                    f"activation_storing.hooks key {hook_name!r} is not a "
                    f"valid hook point; valid names: "
                    f"{sorted(VALID_ACTIVATION_HOOK_NAMES)}"
                )
        # Structural check on positions: either a known kind or a list of ints.
        if isinstance(self.positions, str):
            if self.positions not in VALID_POSITION_KINDS:
                raise ValueError(
                    f"activation_storing.positions string form must be one "
                    f"of {sorted(VALID_POSITION_KINDS)}, got "
                    f"{self.positions!r}"
                )
        elif isinstance(self.positions, list):
            if not self.positions:
                raise ValueError(
                    "activation_storing.positions list form must be non-empty"
                )
            for i, value in enumerate(self.positions):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"activation_storing.positions[{i}] must be an int, "
                        f"got {type(value).__name__}"
                    )
                if value < 0:
                    raise ValueError(
                        f"activation_storing.positions[{i}] = {value} "
                        "must be non-negative"
                    )
        else:
            raise ValueError(
                "activation_storing.positions must be a string "
                f"({sorted(VALID_POSITION_KINDS)}) or a list of ints, "
                f"got {type(self.positions).__name__}"
            )
