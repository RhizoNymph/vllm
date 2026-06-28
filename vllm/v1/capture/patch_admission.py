# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-time validation + prefix-cache classification for patch specs.

Runs at the OpenAI entrypoint (``_admit_patch``). Unlike capture, patch entries
carry explicit ``dest_position`` ints, so no consumer validator is needed — the
prefix floor is read straight off the spec. Validates hook/layer/position and
the strict single-request pool capacity, then stamps ``patch_touches_prompt`` /
``patch_min_prompt_position`` (mirrors ``capture_*``) so prompt-range patches
re-forward from their lowest patched position and the injection hook fires.

Source-run existence is not checked here (it would need an engine RPC to the
per-worker source store); a missing source surfaces at worker resolution as a
logged, skipped entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.capture.types import CaptureContext

_INJECTABLE_HOOKS = frozenset(h.value for h in HOOK_POINT_TABLE_ATTR)


class PatchValidationError(ValueError):
    """Raised when a patch spec fails admission validation (-> HTTP 400)."""


def resolve_patch_prefix_flags(
    sampling_params: SamplingParams,
    ctx: CaptureContext,
    *,
    max_patch_slots: int,
) -> None:
    """Validate ``sampling_params.patch`` and stamp prefix-cache flags in place.

    Raises :class:`PatchValidationError` on an invalid hook, out-of-range
    layer, negative position, or a single-request per-site demand exceeding the
    pool (strict policy).
    """
    spec = sampling_params.patch
    if not spec:
        return

    prompt_floors: list[int] = []
    site_counts: dict[tuple[int, str], int] = {}
    for i, entry in enumerate(spec):
        hook = entry["hook"]
        if hook not in _INJECTABLE_HOOKS:
            raise PatchValidationError(
                f"patch[{i}]: hook {hook!r} is not injectable; valid: "
                f"{sorted(_INJECTABLE_HOOKS)}"
            )
        layer = int(entry["layer"])
        if not (0 <= layer < ctx.num_hidden_layers):
            raise PatchValidationError(
                f"patch[{i}]: layer {layer} out of range "
                f"[0, {ctx.num_hidden_layers})"
            )
        dest = int(entry["dest_position"])
        if dest < 0:
            raise PatchValidationError(
                f"patch[{i}]: dest_position {dest} must be >= 0"
            )
        if int(entry["source_position"]) < 0:
            raise PatchValidationError(
                f"patch[{i}]: source_position must be >= 0"
            )
        key = (layer, hook)
        site_counts[key] = site_counts.get(key, 0) + 1
        if max_patch_slots and site_counts[key] > max_patch_slots - 1:
            raise PatchValidationError(
                f"patch needs more than {max_patch_slots - 1} slots at "
                f"layer={layer} hook={hook}; raise --max-patch-slots"
            )
        if dest < ctx.num_prompt_tokens:
            prompt_floors.append(dest)

    if prompt_floors:
        sampling_params.patch_touches_prompt = True
        sampling_params.patch_min_prompt_position = min(prompt_floors)
    else:
        sampling_params.patch_touches_prompt = False
        sampling_params.patch_min_prompt_position = None
