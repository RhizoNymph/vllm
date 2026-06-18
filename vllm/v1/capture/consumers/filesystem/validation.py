# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-time validation for filesystem capture requests.

Runs at the OpenAI-compatible entrypoint boundary (or at
``LLM(...)`` admission) before the request is handed to the engine.
Takes a :class:`FilesystemCaptureRequest` + :class:`VllmConfig` +
:class:`CaptureContext` and either returns a :class:`CaptureSpec`
that downstream code can consume, or raises
:class:`CaptureValidationError` with a descriptive message that the
entrypoint surfaces as HTTP 400.

The checks performed here are the ones that can't be done at
``SamplingParams`` construction time because they need a
``VllmConfig`` or request context:

1. Every layer referenced by every hook is in
   ``[0, num_hidden_layers)`` (the *global* layer count under PP).
2. Tag and request_id slug cleanly (reject ``..``, leading ``/``,
   > 256 chars, empty).
3. Explicit position indices are valid and not below
   ``num_computed_tokens`` (prefix-cache hits that were never
   forwarded).

Tensor / pipeline / expert / data parallelism are all supported for
the replicated residual hooks; see the parallelism note at the
rejection site (now removed) and ``docs/design/capture_parallelism.md``.

Runtime concerns (writer pool, plan building, finalization) are not
this module's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import regex as re

from vllm.v1.capture.errors import CaptureValidationError
from vllm.v1.capture.types import (
    MODEL_LEVEL_HOOKS,
    CaptureContext,
    CaptureSpec,
    HookName,
    PositionSelector,
)

from .types import FilesystemCaptureRequest

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# ---------------------------------------------------------------------------
# Hook-point + position-kind literal sets
# ---------------------------------------------------------------------------

# Fallback valid-hook set, used only when the request's ``CaptureContext``
# carries no hook schema (older construction sites / unit tests). In
# production the model's hook schema (``ctx.hook_schema``) is the source of
# truth for which hooks are tapped — see ``validate_filesystem_request``.
# These three are the hooks ``apply_layer_steering`` taps on every standard
# model; ``mlp_in`` / ``mlp_out`` are reserved names not wired into a
# standard model forward (DeepSeek-V4 wires them via its own schema).
_VALID_HOOK_NAMES: frozenset[str] = frozenset(("pre_attn", "post_attn", "post_mlp"))

_VALID_POSITION_KINDS: frozenset[str] = frozenset(
    ("last_prompt", "all_prompt", "all_generated", "all")
)


# ---------------------------------------------------------------------------
# Slugging
# ---------------------------------------------------------------------------

_SLUG_REGEX = re.compile(r"[^a-zA-Z0-9._-]")
_SLUG_MAX_LEN = 256


def _slug(name: str, *, field: str) -> str:
    """Slug a ``tag`` or ``request_id`` into a safe path segment.

    Rejects empty input, ``..`` (path traversal), leading ``/``, and
    anything over 256 chars.
    """
    if not isinstance(name, str):
        raise CaptureValidationError(
            f"{field} must be a string, got {type(name).__name__}"
        )
    if not name:
        raise CaptureValidationError(f"{field} must be non-empty")
    if len(name) > _SLUG_MAX_LEN:
        raise CaptureValidationError(
            f"{field} must be at most {_SLUG_MAX_LEN} characters, got {len(name)}"
        )
    if ".." in name:
        raise CaptureValidationError(f"{field} must not contain '..': {name!r}")
    if name.startswith("/"):
        raise CaptureValidationError(f"{field} must not start with '/': {name!r}")
    return _SLUG_REGEX.sub("_", name)


# ---------------------------------------------------------------------------
# Layer selector expansion
# ---------------------------------------------------------------------------


def _expand_layer_list(
    raw: list[int], *, num_hidden_layers: int, where: str
) -> list[int]:
    result: list[int] = []
    for i, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CaptureValidationError(
                f"{where}[{i}] must be an int, got {type(value).__name__}"
            )
        if value < 0 or value >= num_hidden_layers:
            raise CaptureValidationError(
                f"{where}[{i}] = {value} is out of range for a model with "
                f"num_hidden_layers={num_hidden_layers}"
            )
        result.append(value)
    return result


def _expand_ranges(raw: list[Any], *, num_hidden_layers: int, where: str) -> list[int]:
    result: list[int] = []
    for i, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise CaptureValidationError(
                f"{where}[{i}] must be a 2-element [start, end] pair, got {pair!r}"
            )
        start, end = pair
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise CaptureValidationError(
                f"{where}[{i}] must be a pair of ints, got {pair!r}"
            )
        if start > end:
            raise CaptureValidationError(
                f"{where}[{i}] has start > end ({start} > {end})"
            )
        if start < 0 or end >= num_hidden_layers:
            raise CaptureValidationError(
                f"{where}[{i}] = [{start}, {end}] is out of range for a "
                f"model with num_hidden_layers={num_hidden_layers}"
            )
        result.extend(range(start, end + 1))
    return result


def _expand_hook_layers(
    selector: Any, num_hidden_layers: int, *, where: str
) -> list[int]:
    """Resolve a layer selector into a sorted, deduped list of indices.

    Accepts ``"all"``, ``list[int]``, or ``{"layers": [...], "ranges": [[a,b], ...]}``.
    """
    if num_hidden_layers <= 0:
        raise CaptureValidationError(
            f"num_hidden_layers must be positive, got {num_hidden_layers}"
        )

    if isinstance(selector, str):
        if selector != "all":
            raise CaptureValidationError(
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
            raise CaptureValidationError(
                f"{where} dict form has unexpected keys {sorted(extra)}; "
                f"allowed keys: {sorted(allowed)}"
            )
        layers_raw = selector.get("layers")
        ranges_raw = selector.get("ranges")
        if layers_raw is None and ranges_raw is None:
            raise CaptureValidationError(
                f"{where} dict form must set 'layers' and/or 'ranges'; got empty"
            )

        collected: list[int] = []
        if layers_raw is not None:
            if not isinstance(layers_raw, list):
                raise CaptureValidationError(
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
                raise CaptureValidationError(
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

    raise CaptureValidationError(
        f"{where} must be 'all', list[int], or {{'layers': ..., 'ranges': ...}}; "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# Position selector resolution
# ---------------------------------------------------------------------------


def _resolve_positions(
    selector: Any,
    num_prompt_tokens: int,
    num_generated_tokens: int,
    *,
    where: str,
) -> list[int]:
    """Resolve a position selector into absolute token indices.

    Only handles the cases whose size is known from static request
    metadata. ``all_generated`` and ``all`` are symbolic — callers
    should defer those to the runner.
    """
    if num_prompt_tokens <= 0:
        raise CaptureValidationError(
            f"num_prompt_tokens must be positive, got {num_prompt_tokens}"
        )
    if num_generated_tokens < 0:
        raise CaptureValidationError(
            f"num_generated_tokens must be non-negative, got {num_generated_tokens}"
        )

    if isinstance(selector, str):
        if selector not in _VALID_POSITION_KINDS:
            raise CaptureValidationError(
                f"{where} must be one of {sorted(_VALID_POSITION_KINDS)} "
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
                raise CaptureValidationError(
                    f"{where}[{i}] must be an int, got {type(value).__name__}"
                )
            if value < 0 or value >= total:
                raise CaptureValidationError(
                    f"{where}[{i}] = {value} is out of range for a request "
                    f"with num_prompt_tokens={num_prompt_tokens}, "
                    f"num_generated_tokens={num_generated_tokens}"
                )
            result.append(value)
        return sorted(set(result))

    raise CaptureValidationError(
        f"{where} must be one of {sorted(_VALID_POSITION_KINDS)} or list[int], "
        f"got {type(selector).__name__}"
    )


# ---------------------------------------------------------------------------
# Public API: validate_filesystem_request
# ---------------------------------------------------------------------------


def _structural_validate(
    raw: FilesystemCaptureRequest, valid_hooks: frozenset[str]
) -> None:
    """Structural validation of the raw request before resolving.

    ``valid_hooks`` is the set of hook points the target model actually
    taps (its hook schema). A request for any other hook is rejected here
    so it fails fast rather than producing an empty, zero-byte capture.
    """
    if not isinstance(raw.request_id, str) or not raw.request_id:
        raise CaptureValidationError(
            f"capture.request_id must be a non-empty string, got {raw.request_id!r}"
        )
    if not isinstance(raw.tag, str) or not raw.tag:
        raise CaptureValidationError(
            f"capture.tag must be a non-empty string, got {raw.tag!r}"
        )
    if not isinstance(raw.hooks, dict) or not raw.hooks:
        raise CaptureValidationError(
            "capture.hooks must be a non-empty dict mapping hook point "
            "name to a layer selector"
        )
    for hook_name in raw.hooks:
        if not isinstance(hook_name, str):
            raise CaptureValidationError(
                f"capture.hooks key must be a string, got {type(hook_name).__name__}"
            )
        if hook_name not in valid_hooks:
            raise CaptureValidationError(
                f"capture.hooks key {hook_name!r} is not a hook point this "
                f"model taps; valid names: {sorted(valid_hooks)}"
            )
    if isinstance(raw.positions, str):
        if raw.positions not in _VALID_POSITION_KINDS:
            raise CaptureValidationError(
                f"capture.positions string form must be one of "
                f"{sorted(_VALID_POSITION_KINDS)}, got {raw.positions!r}"
            )
    elif isinstance(raw.positions, list):
        if not raw.positions:
            raise CaptureValidationError(
                "capture.positions list form must be non-empty"
            )
        for i, value in enumerate(raw.positions):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CaptureValidationError(
                    f"capture.positions[{i}] must be an int, got {type(value).__name__}"
                )
            if value < 0:
                raise CaptureValidationError(
                    f"capture.positions[{i}] = {value} must be non-negative"
                )
    else:
        raise CaptureValidationError(
            "capture.positions must be a string "
            f"({sorted(_VALID_POSITION_KINDS)}) or a list of ints, "
            f"got {type(raw.positions).__name__}"
        )


def validate_filesystem_request(
    raw: FilesystemCaptureRequest,
    vllm_config: VllmConfig,
    ctx: CaptureContext,
) -> CaptureSpec:
    """Validate a ``FilesystemCaptureRequest`` and return a ``CaptureSpec``.

    Raises :class:`CaptureValidationError` on any validation failure.
    The returned spec's ``positions`` is either a resolved ``list[int]``
    (``last_prompt`` / ``all_prompt`` / explicit list) or a symbolic
    kind string (``all_generated`` / ``all``) which the runner
    materializes per-step.
    """
    # 1. Structural validation. A model's hook schema lists exactly the
    # hooks it taps; use it as the valid-hook set so per-model hooks
    # (``mlp_in`` / ``mlp_out`` / ``mhc_*`` on DeepSeek-V4) are accepted
    # only where wired. Fall back to the standard wired residual hooks when
    # no schema was provided (older construction sites / unit tests).
    valid_hooks = frozenset(ctx.hook_schema) if ctx.hook_schema else _VALID_HOOK_NAMES
    _structural_validate(raw, valid_hooks)

    # 2. Parallelism. The residual hooks captured today (pre_attn /
    # post_attn / post_mlp) read the residual stream after the
    # tensor-parallel all-reduce / MoE combine, so it is replicated and
    # full-width across the TP and EP planes; data parallelism partitions
    # requests across independent engine cores. All four axes are
    # therefore supported: TP rank 0 of each pipeline stage captures that
    # stage's (global-indexed) layers to the shared mount, and the engine
    # merges the per-stage results. No accept/reject branch on TP / PP /
    # EP / DP size is needed here.
    #
    # Phase 4 (sharded-activation capture: MLP intermediate / per-expert
    # outputs) will reintroduce a rejection *for sharded hooks only*,
    # since those tensors are partitioned across the TP / EP plane and
    # need a gather. The residual hooks above remain unconditional.

    # 3. Slug tag + request_id so we surface a clean error (rather
    # than failing much later on a filesystem write).
    _slug(raw.request_id, field="capture.request_id")
    _slug(raw.tag, field="capture.tag")

    # 4. Expand hooks and validate layer indices.
    if ctx.num_hidden_layers <= 0:
        raise CaptureValidationError(
            f"filesystem capture requires num_hidden_layers > 0; got "
            f"{ctx.num_hidden_layers}"
        )
    resolved_hooks: dict[HookName, list[int]] = {}
    for hook_name, selector in raw.hooks.items():
        if hook_name in MODEL_LEVEL_HOOKS:
            # Model-level hook: fires once at the model tail, keyed to the
            # last layer. The layer selector is meaningless here, so accept
            # any selector and normalize to that single index — callers need
            # not know it (e.g. ``{"mhc_streams_final": "all"}``).
            resolved_hooks[cast(HookName, hook_name)] = [ctx.num_hidden_layers - 1]
            continue
        resolved = _expand_hook_layers(
            selector,
            ctx.num_hidden_layers,
            where=f"capture.hooks[{hook_name!r}]",
        )
        if not resolved:
            raise CaptureValidationError(
                f"capture.hooks[{hook_name!r}] expanded to an empty layer list"
            )
        # hook_name already validated as one of _VALID_HOOK_NAMES.
        resolved_hooks[cast(HookName, hook_name)] = resolved

    # 5. Resolve positions.
    # "last_prompt", "all_prompt", and explicit lists resolve at
    # admission time. "all_generated" and "all" stay symbolic — the
    # runner materializes them per-step once the generated token
    # count is known.
    resolved_positions: PositionSelector
    if isinstance(raw.positions, str):
        if raw.positions in ("last_prompt", "all_prompt"):
            resolved_positions = _resolve_positions(
                raw.positions,
                ctx.num_prompt_tokens,
                num_generated_tokens=0,
                where="capture.positions",
            )
        else:
            # "all_generated" / "all" — defer to the runner. Already
            # validated against _VALID_POSITION_KINDS above.
            resolved_positions = cast(PositionSelector, raw.positions)
    elif isinstance(raw.positions, list):
        resolved_positions = _resolve_positions(
            raw.positions,
            ctx.num_prompt_tokens,
            num_generated_tokens=0,
            where="capture.positions",
        )
    else:
        raise CaptureValidationError(
            "capture.positions must be a string or list of ints"
        )

    # 6. Prefix-cache position rejection. Any explicit position below
    # num_computed_tokens was not forwarded through the model (served
    # from the prefix cache), so we can never capture its residual.
    if isinstance(resolved_positions, list):
        bad = [p for p in resolved_positions if p < ctx.num_computed_tokens]
        if bad:
            raise CaptureValidationError(
                f"capture.positions contains indices that were served "
                f"from the prefix cache and will never be forwarded "
                f"through the model: {bad}. The first uncached position "
                f"for this request is {ctx.num_computed_tokens}."
            )

    return CaptureSpec(hooks=resolved_hooks, positions=resolved_positions)


def slug_path_components(raw: FilesystemCaptureRequest) -> tuple[str, str]:
    """Slug ``tag`` and ``request_id`` for filesystem path use.

    Exposed alongside ``validate_filesystem_request`` so the consumer can
    record the slugs at admission time (when the raw ``FilesystemCaptureRequest``
    is still in hand) for later use at submit/finalize time, since the
    framework's :class:`CaptureSpec` only carries ``hooks`` and ``positions``.
    Raises :class:`CaptureValidationError` on invalid slugs.
    """
    tag_slug = _slug(raw.tag, field="capture.tag")
    request_id_slug = _slug(raw.request_id, field="capture.request_id")
    return tag_slug, request_id_slug


__all__: list[str] = ["validate_filesystem_request", "slug_path_components"]
