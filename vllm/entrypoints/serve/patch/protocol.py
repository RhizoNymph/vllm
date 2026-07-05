# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response models for the server-side patch-sweep endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LayerRange(BaseModel):
    """Inclusive-exclusive layer range, like ``range(start, stop, step)``."""

    start: int
    stop: int
    step: int = Field(default=1, ge=1)


class SpanPosition(BaseModel):
    """A sweep position given as a substring of the corrupt prompt.

    Resolved server-side to the token positions covering the substring — the
    prompt is tokenized exactly as the sweep tokenizes it, each token mapped to
    its character span, and the tokens overlapping ``span`` are selected. May
    be mixed with plain integer positions.
    """

    span: str
    occurrence: int = Field(default=0, ge=0)
    """Which match to use when ``span`` appears more than once (0 = first)."""


class PatchSweepRequest(BaseModel):
    """A whole ``(layers x positions)`` activation-patching sweep in one call.

    The server expands the grid into one patched variant per cell, runs them
    through the continuously-batched engine (the common corrupt-prompt prefix
    is shared via prefix caching; each variant is patched at its own site via
    per-row gating), and returns the assembled metric grid.
    """

    model: str | None = None
    prompt: str
    """The corrupted/destination prompt swept over."""
    source_run: str
    """Clean-run handle whose stored activations are patched in."""
    clean_prompt: str | None = None
    """The clean prompt ``source_run`` was captured from.

    Serves two roles. (1) Position alignment: required when it tokenizes to a
    different length than ``prompt`` — positions are then aligned (common token
    prefix by identity, common suffix by the length delta) and the unalignable
    middle is skipped loudly (assuming ``source == dest`` across a length
    divergence silently patches the wrong positions). (2) One-call auto-capture:
    when ``source_run`` does not yet exist, the server captures this clean
    prompt itself (hook + swept layers, ``all_prompt`` positions) with
    capture-wait durability before running the grid, collapsing the
    capture-then-sweep dance to a single request. Without it a missing
    ``source_run`` is a 400 (capture the clean run explicitly first)."""
    hook: str = "post_block"
    layers: list[int] | LayerRange
    positions: list[int | SpanPosition] | Literal["all_prompt"] = "all_prompt"
    """Token indices, and/or ``{"span": str, "occurrence": int}`` substring
    markers resolved server-side against ``prompt``. The response's
    ``positions`` is the resolved integer axis."""
    alpha: float = 1.0
    # Grade by this answer token. Prefer the id (exact); else the string is
    # matched against the decoded top-k tokens (whitespace-tolerant).
    answer_token: str | None = None
    answer_token_id: int | None = None
    # Optional foil for the logit_diff metric.
    foil_token: str | None = None
    foil_token_id: int | None = None
    metric: Literal["logprob", "logit_diff", "recovered"] = "logprob"
    logprobs: int = Field(default=20, ge=1)
    stream: bool = False
    """Stream per-cell results over SSE (``text/event-stream``) as they land,
    ending with a ``summary`` event carrying the full non-streaming response and
    a ``[DONE]`` terminator. Pre-fan-out errors (bad hook/layers, span/alignment
    failure, missing source) still return a plain JSON 400 — the stream only
    starts once the grid fan-out begins. Off by default (single JSON response)."""
    # Clean baseline answer metric (from an explicit capture_clean) for the
    # recovered metric; the corrupt baseline is computed in-endpoint. Ignored
    # when the server auto-captures (the clean baseline is then graded from the
    # internal clean generation).
    clean_baseline: float | None = None


class PatchSweepResponse(BaseModel):
    layers: list[int]
    positions: list[int]
    hook: str
    metric: str
    grid: list[list[float | None]]
    """grid[i][j] is the metric for layers[i] patched at positions[j]."""
    clean: float | None = None
    corrupt: float | None = None
    argmax: dict | None = None
    skipped: list[dict] = Field(default_factory=list)
    alignment: dict | None = None
    """Position-alignment summary when ``clean_prompt`` was provided
    (prefix/suffix lengths and any unaligned positions that were skipped)."""
    noise_floor: float | None = None
    """|metric(baseline re-run inside the cell batch) - metric(solo baseline)|.
    vLLM is not batch-invariant by default, so identical requests in different
    batch compositions return slightly different logprobs; grid differences at
    or below this floor are not meaningful. For exact reproducibility start the
    server with batch-invariant mode (see docs/features/batch_invariance.md)."""
    auto_captured: bool = False
    """True when ``source_run`` was missing and the server captured
    ``clean_prompt`` itself (one-call sweep) before running the grid, rather
    than reusing a pre-existing capture run."""
    captured_source_run: str | None = None
    """The run handle the auto-capture wrote under (equals ``source_run``), or
    ``None`` when no auto-capture happened."""
