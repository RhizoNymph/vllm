# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-position alignment between a clean and a corrupt prompt.

Patch specs pair a destination position (in the corrupt run) with a source
position (in the clean run). When the two prompts tokenize to different
lengths, ``source = dest`` silently patches the WRONG position for everything
after the length divergence — a shifted-but-plausible heatmap. This module
computes the safe correspondence:

- equal token length: identity everywhere (patching the corresponding position
  of the differing tokens is exactly the causal-tracing operation);
- unequal: the longest common token prefix maps by identity, the longest common
  token suffix maps by the length delta, and the differing middle has no
  positional correspondence — those destination positions are unaligned and
  must be skipped (loudly), not guessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PositionAlignment:
    """Mapping from corrupt (dest) positions to clean (source) positions."""

    mapping: dict[int, int]
    """dest_position -> source_position, for every alignable dest position."""
    unaligned: list[int] = field(default_factory=list)
    """Dest positions inside the differing span (no safe source position)."""
    prefix_len: int = 0
    suffix_len: int = 0
    n_clean: int = 0
    n_corrupt: int = 0

    @property
    def is_identity(self) -> bool:
        return self.n_clean == self.n_corrupt and not self.unaligned

    def source_for(self, dest_position: int) -> int | None:
        return self.mapping.get(dest_position)

    def summary(self) -> dict:
        return {
            "n_clean": self.n_clean,
            "n_corrupt": self.n_corrupt,
            "prefix_len": self.prefix_len,
            "suffix_len": self.suffix_len,
            "unaligned_positions": list(self.unaligned),
        }


def align_token_positions(
    clean_ids: Sequence[int], corrupt_ids: Sequence[int]
) -> PositionAlignment:
    """Align corrupt-prompt token positions to clean-prompt token positions."""
    n_clean, n_corrupt = len(clean_ids), len(corrupt_ids)

    if n_clean == n_corrupt:
        # Positionally corresponding even where tokens differ — the standard
        # causal-tracing setup.
        return PositionAlignment(
            mapping={i: i for i in range(n_corrupt)},
            prefix_len=n_corrupt,
            suffix_len=0,
            n_clean=n_clean,
            n_corrupt=n_corrupt,
        )

    limit = min(n_clean, n_corrupt)
    prefix = 0
    while prefix < limit and clean_ids[prefix] == corrupt_ids[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and clean_ids[n_clean - 1 - suffix] == corrupt_ids[n_corrupt - 1 - suffix]
    ):
        suffix += 1

    shift = n_clean - n_corrupt
    mapping: dict[int, int] = {i: i for i in range(prefix)}
    mapping.update({i: i + shift for i in range(n_corrupt - suffix, n_corrupt)})
    unaligned = list(range(prefix, n_corrupt - suffix))
    return PositionAlignment(
        mapping=mapping,
        unaligned=unaligned,
        prefix_len=prefix,
        suffix_len=suffix,
        n_clean=n_clean,
        n_corrupt=n_corrupt,
    )
