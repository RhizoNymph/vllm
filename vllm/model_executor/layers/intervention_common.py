# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared layer-side scaffolding for residual-stream intervention tiers.

Every intervention tier (steering adds, patching overwrites, clamping bounds)
attaches per-hook persistent buffers to decoder layers and resolves its buffer
sizing the same way; only the payload semantics differ. This module holds the
payload-agnostic pieces:

- :func:`hook_attrs` / :func:`derived_attrs` — the ``{hook: attr-name}`` dicts
  that bind Python buffer names to what the registered custom ops read via
  ``getattr``-by-string. A tier's attr-name *values* are a runtime contract:
  changing one breaks buffer lookup silently (no import error), so derive them
  in one place.
- :class:`BufferKnob` — the config-first / TEST-ONLY-process-global pattern
  for resolving a tier's buffer size at registration time.

This module must stay import-light and free of tier imports: ``steering.py``
and ``patch.py`` (which already carry a deliberately broken import cycle)
both sit above it in the import graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.steering import SteeringHookPoint


def hook_attrs(
    prefix: str, hooks: Iterable[SteeringHookPoint]
) -> dict[SteeringHookPoint, str]:
    """Buffer attribute names keyed by hook point: ``{hp: f"{prefix}_{hp}"}``."""
    return {hp: f"{prefix}_{hp.value}" for hp in hooks}


def derived_attrs(
    base: Mapping[SteeringHookPoint, str], suffix: str
) -> dict[SteeringHookPoint, str]:
    """Attr names derived from a base dict by suffix, so the two families are
    always discoverable together: ``{hp: f"{base[hp]}{suffix}"}``."""
    return {hp: f"{attr}{suffix}" for hp, attr in base.items()}


class BufferKnob:
    """Config-first buffer-sizing knob with a TEST-ONLY process-global fallback.

    ``resolve()`` is meant for buffer-registration time: the primary source is
    the *current* ``VllmConfig`` (models are always built under
    ``set_current_vllm_config``, on every runner), so no runner has to
    remember to set anything before the model build. The process-global test
    value is consulted only when no config context exists — unit tests
    constructing layers directly. ``0`` conventionally means "tier disabled":
    no buffers attached, and the folded apply path constant-folds out.
    """

    def __init__(self, config_getter: Callable[[VllmConfig], int]) -> None:
        self._config_getter = config_getter
        self._test_value = 0

    def set_for_tests(self, value: int) -> None:
        """TEST-ONLY override, read only when no VllmConfig context exists."""
        self._test_value = int(value)

    def get_test_value(self) -> int:
        return self._test_value

    def resolve(self) -> int:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            return self._config_getter(vllm_config)
        return self._test_value
