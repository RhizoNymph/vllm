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
    """The clean prompt ``source_run`` was captured from. Required when it
    tokenizes to a different length than ``prompt``: positions are then
    aligned (common token prefix by identity, common suffix by the length
    delta) and the unalignable middle is skipped loudly — assuming
    ``source == dest`` across a length divergence silently patches the wrong
    positions."""
    hook: str = "post_block"
    layers: list[int] | LayerRange
    positions: list[int] | Literal["all_prompt"] = "all_prompt"
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
    # Clean baseline answer metric (from capture_clean) for the recovered
    # metric; the corrupt baseline is computed in-endpoint.
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
