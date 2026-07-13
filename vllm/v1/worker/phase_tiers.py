# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic base/prefill/decode phase-tier container.

The steering control plane stores several kinds of global state split into
the same three phase tiers — base (both phases), prefill-specific, and
decode-specific — and every tier family needs the same phase-string dispatch,
clear-all, and any-set checks. This container replaces the per-family
if/elif ladders; ``SteeringManager``'s global vectors use it today, and
future intervention tiers (e.g. clamps) adopt it instead of adding twins.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

Phase = Literal["base", "prefill", "decode"]

T = TypeVar("T")


@dataclass
class PhaseTiers(Generic[T]):
    """Three same-typed tiers keyed by phase.

    ``label`` names the tier family in the ``for_phase`` error message
    (e.g. ``"global vector"`` -> ``"Invalid global vector phase: ..."``).
    ``clear_all`` requires ``T`` to expose ``.clear()`` (dict-shaped tiers).
    """

    base: T
    prefill: T
    decode: T
    label: str = ""

    def for_phase(self, phase: str) -> T:
        if phase not in ("base", "prefill", "decode"):
            family = f"{self.label} " if self.label else ""
            raise ValueError(
                f"Invalid {family}phase: {phase!r}. "
                f"Must be 'base', 'prefill', or 'decode'."
            )
        return getattr(self, phase)

    def items(self) -> Iterator[tuple[Phase, T]]:
        yield "base", self.base
        yield "prefill", self.prefill
        yield "decode", self.decode

    def clear_all(self) -> None:
        for _, tier in self.items():
            tier.clear()  # type: ignore[attr-defined]

    def __bool__(self) -> bool:
        return bool(self.base) or bool(self.prefill) or bool(self.decode)
