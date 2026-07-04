# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Substring -> token-position resolution shared by the sweep endpoint and the
:class:`PatchStudy` client.

A sweep position may be given as a substring of the prompt instead of a token
index. Resolving one means: tokenize the prompt exactly as the sweep does, map
each token to its half-open character span, and select the tokens whose span
overlaps the substring. The pure math here is tokenizer-free (unit-testable);
the character offsets are supplied either by a fast tokenizer's offset mapping
(server, in-process) or by incremental detokenization (client, over HTTP).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence


def resolve_span_positions(
    token_offsets: Sequence[tuple[int, int]],
    text: str,
    span: str,
    occurrence: int = 0,
) -> list[int]:
    """Token positions whose character span overlaps ``span`` within ``text``.

    Args:
        token_offsets: ``(char_start, char_end)`` per token position, indexing
            into ``text`` (half-open).
        text: The prompt text the offsets index into.
        span: The substring to cover.
        occurrence: Which match to select when ``span`` appears multiple times.

    Returns:
        The ascending token positions overlapping the chosen match.

    Raises:
        ValueError: ``span`` is empty, not found, or ``occurrence`` is out of
            range for the number of matches.
    """
    if not span:
        raise ValueError("span must be a non-empty substring")
    starts: list[int] = []
    idx = text.find(span)
    while idx != -1:
        starts.append(idx)
        idx = text.find(span, idx + 1)
    if not starts:
        raise ValueError(f"span {span!r} not found in prompt {text!r}")
    if not 0 <= occurrence < len(starts):
        raise ValueError(
            f"span {span!r} occurs {len(starts)} time(s) in the prompt; "
            f"occurrence={occurrence} is out of range"
        )
    char_start = starts[occurrence]
    char_end = char_start + len(span)
    positions = [
        i
        for i, (start, end) in enumerate(token_offsets)
        if start < char_end and end > char_start
    ]
    if not positions:
        raise ValueError(
            f"span {span!r} matched chars [{char_start}, {char_end}) but "
            f"covers no token position (empty/whitespace-only token span)"
        )
    return positions


def dedup_positions(resolved: Iterable[Iterable[int]]) -> list[int]:
    """Flatten already-resolved position lists, order-preserving with dedup.

    A span expanding to ``[1, 2, 3]`` mixed with an explicit ``2`` yields no
    duplicate; first occurrence wins.
    """
    out: list[int] = []
    seen: set[int] = set()
    for group in resolved:
        for pos in group:
            if pos not in seen:
                seen.add(pos)
                out.append(pos)
    return out


def incremental_char_offsets(
    decode: Callable[[Sequence[int]], str | None],
    ids: Sequence[int],
) -> tuple[str, list[tuple[int, int]]]:
    """Per-token char offsets from detokenizing growing id prefixes.

    Used when no offset mapping is available (the HTTP client, or a slow
    tokenizer): detokenizing ``ids[:k]`` yields offsets in the same detokenized
    space the substring search runs in, for any tokenizer. Special tokens
    (e.g. BOS) detokenize to an empty span and are never selected.

    Args:
        decode: Maps a prefix of ``ids`` to its detokenization, or ``None`` on
            failure.
        ids: The exact token ids of the prompt.

    Returns:
        ``(text, offsets)`` where ``offsets[k]`` is token ``k``'s half-open
        ``(start, end)`` char span in ``text``.

    Raises:
        RuntimeError: ``decode`` returned ``None`` (detokenizer unavailable).
    """
    offsets: list[tuple[int, int]] = []
    prev_len = 0
    text = ""
    for k in range(1, len(ids) + 1):
        text = decode(ids[:k])
        if text is None:
            raise RuntimeError("detokenize unavailable; cannot map span")
        offsets.append((prev_len, len(text)))
        prev_len = len(text)
    return text, offsets


def prompt_char_offsets(
    tokenizer, prompt: str
) -> tuple[str, list[tuple[int, int]]]:
    """``(text, per-token char offsets)`` for ``prompt``, in-process.

    Tokenizes exactly as the sweep does (``add_special_tokens=True``, matching
    both the engine's prompt tokenization and the alignment path). Prefers the
    fast tokenizer's offset mapping (offsets index the original ``prompt``);
    falls back to incremental detokenization of growing id prefixes (offsets
    index the detokenized text). Special tokens map to an empty span either way
    and are never selected.
    """
    try:
        enc = tokenizer(
            prompt, add_special_tokens=True, return_offsets_mapping=True
        )
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        return prompt, offsets
    except (TypeError, KeyError, ValueError, NotImplementedError, AttributeError):
        ids = tokenizer.encode(prompt)
        return incremental_char_offsets(lambda i: tokenizer.decode(i), ids)
