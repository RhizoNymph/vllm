# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-patching sweeps through the OpenAI-compatible server.

The "walk" to find which sites carry a behavior:

1. Capture a CLEAN run once into the server-side source store under a run
   handle (``capture_clean``).
2. Sweep a CORRUPTED prompt, patching one (layer, position) site per request
   with the clean run's stored activation, grading each by the answer token's
   logprob (``sweep_layers_positions``). The clean capture is paid once and
   referenced cheaply across the whole grid.
3. Zoom into the hot region and re-sweep at finer granularity (``zoom``).

The server does all tensor work; this client only orchestrates requests and
assembles a heatmap. Patching uses the precise-lerp op, so ``alpha=1`` is an
exact activation replacement (classic denoising).

Start the server with patching + the clean-run source consumer enabled::

    vllm serve google/gemma-3-4b-it \\
        --enable-patching \\
        --max-patch-slots 64 \\
        --patch-source-cache-bytes 2000000000 \\
        --capture-consumers '[{"type": "patch_source"}]'

Then, e.g.::

    import asyncio
    from openai_patch_client import PatchStudy

    study = PatchStudy(model="google/gemma-3-4b-it")
    clean = study.capture_clean(
        "The Eiffel Tower is in the city of",
        run="paris", answer_token=" Paris",
    )
    result = asyncio.run(study.sweep_layers_positions(
        "The Colosseum is in the city of",
        run="paris", layers=range(0, 34, 2),
        positions=range(0, clean.num_prompt_tokens),
        answer_token=" Paris",
    ))
    print(result.argmax_cell(), result.top(5))
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, OpenAI


@dataclass
class CleanRun:
    """Handle to a captured clean run + its graded baseline."""

    run_id: str
    num_prompt_tokens: int
    hook: str
    answer_token: str | None = None
    clean_logprob: float | None = None
    clean_logit_diff: float | None = None


@dataclass
class SweepResult:
    """A (layers x positions) grid of patching metrics."""

    layers: list[int]
    positions: list[int]
    hook: str
    metric_name: str
    grid: list[list[float]]  # grid[i][j] for layers[i], positions[j]
    clean: float | None = None
    corrupt: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_numpy(self):
        import numpy as np

        return np.asarray(self.grid, dtype=float)

    def argmax_cell(self) -> tuple[int, int]:
        """Return the ``(layer, position)`` of the maximum-metric cell."""
        best = None
        best_val = -math.inf
        for i, layer in enumerate(self.layers):
            for j, pos in enumerate(self.positions):
                v = self.grid[i][j]
                if v is not None and v > best_val:
                    best_val = v
                    best = (layer, pos)
        return best  # type: ignore[return-value]

    def top(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Top-``k`` ``(layer, position, metric)`` cells by metric."""
        cells = [
            (layer, pos, self.grid[i][j])
            for i, layer in enumerate(self.layers)
            for j, pos in enumerate(self.positions)
            if self.grid[i][j] is not None
        ]
        cells.sort(key=lambda c: c[2], reverse=True)
        return cells[:k]

    def heatmap(self, ax=None, **imshow_kwargs):
        """Render the grid with matplotlib (rows=layers, cols=positions)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots()
        im = ax.imshow(self.to_numpy(), aspect="auto", origin="lower", **imshow_kwargs)
        ax.set_xlabel("position")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(self.positions)), self.positions)
        ax.set_yticks(range(len(self.layers)), self.layers)
        ax.set_title(f"{self.metric_name} (hook={self.hook})")
        ax.figure.colorbar(im, ax=ax)
        return ax


def _answer_logprob(choice: Any, answer_token: str) -> float | None:
    """Read the logprob of ``answer_token`` from a completion choice.

    Uses the top-logprobs of the single generated token. Returns ``None`` when
    the token is outside the requested top-k.
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    top = getattr(lp, "top_logprobs", None)
    if top:
        first = top[0]
        if answer_token in first:
            return float(first[answer_token])
        # Tolerate leading-space tokenization differences.
        alt = answer_token.lstrip()
        for k, v in first.items():
            if k.strip() == alt:
                return float(v)
    tokens = getattr(lp, "tokens", None)
    token_lps = getattr(lp, "token_logprobs", None)
    if tokens and token_lps and tokens[0].strip() == answer_token.strip():
        return float(token_lps[0])
    return None


class PatchStudy:
    """Thin orchestration over the OpenAI HTTP API for patching sweeps."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "unused",
        concurrency: int = 16,
        hook: str = "post_block",
        logprobs: int = 20,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency = concurrency
        self.hook = hook
        self.logprobs = logprobs
        self._sync = OpenAI(base_url=base_url, api_key=api_key)

    # ---- step 1: capture the clean run once --------------------------------

    def capture_clean(
        self,
        prompt: str,
        *,
        run: str,
        layers: Sequence[int] | str = "all",
        hook: str | None = None,
        positions: str = "all_prompt",
        answer_token: str | None = None,
        foil_token: str | None = None,
    ) -> CleanRun:
        """Run the clean prompt once, capturing residuals into the source store.

        Records the clean baseline logprob (and logit-diff if ``foil_token`` is
        given) of ``answer_token`` for ``recovered``-metric normalization.
        """
        hook = hook or self.hook
        hooks_layers = list(layers) if layers != "all" else "all"
        capture = {
            "patch_source": {
                "run": run,
                "hooks": {hook: hooks_layers},
                "positions": positions,
            }
        }
        # capture_wait holds the response until the capture has finalized, so
        # the source store is durably populated before any patch request
        # references this run (capture write-through is otherwise async — a
        # patch issued too early would resolve to a missing source).
        resp = self._sync.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=self.logprobs,
            extra_body={"capture": capture, "capture_wait": True},
        )
        choice = resp.choices[0]
        clean_lp = (
            _answer_logprob(choice, answer_token)
            if answer_token is not None
            else None
        )
        clean_diff = None
        if answer_token is not None and foil_token is not None:
            foil_lp = _answer_logprob(choice, foil_token)
            if clean_lp is not None and foil_lp is not None:
                clean_diff = clean_lp - foil_lp
        num_prompt = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
        return CleanRun(
            run_id=run,
            num_prompt_tokens=num_prompt,
            hook=hook,
            answer_token=answer_token,
            clean_logprob=clean_lp,
            clean_logit_diff=clean_diff,
        )

    # ---- step 2: sweep the corrupted prompt --------------------------------

    async def sweep_layers_positions(
        self,
        corrupt_prompt: str,
        *,
        run: str,
        layers: Sequence[int],
        positions: Sequence[int],
        hook: str | None = None,
        alpha: float = 1.0,
        answer_token: str,
        foil_token: str | None = None,
        metric: str = "logprob",
        clean: CleanRun | None = None,
    ) -> SweepResult:
        """Fan out one patched request per (layer, position) cell.

        ``metric``: ``"logprob"`` (P(answer)), ``"logit_diff"`` (answer - foil),
        or ``"recovered"`` ((patched - corrupt) / (clean - corrupt)).
        """
        hook = hook or self.hook
        layers = list(layers)
        positions = list(positions)
        notes: list[str] = []

        async with AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key
        ) as aclient:
            sem = asyncio.Semaphore(self.concurrency)

            async def grade(patch: list[dict] | None) -> float | None:
                async with sem:
                    resp = await aclient.completions.create(
                        model=self.model,
                        prompt=corrupt_prompt,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=self.logprobs,
                        extra_body=({"patch": patch} if patch else {}),
                    )
                return self._metric(resp.choices[0], answer_token, foil_token, metric)

            # Corrupt baseline (no patch) once, for recovered-metric.
            corrupt_val = await grade(None)

            async def cell(layer: int, pos: int) -> float | None:
                patch = [
                    {
                        "layer": layer,
                        "hook": hook,
                        "dest_position": pos,
                        "source_run": run,
                        "source_position": pos,
                        "alpha": alpha,
                    }
                ]
                return await grade(patch)

            tasks = [
                [asyncio.create_task(cell(layer, pos)) for pos in positions]
                for layer in layers
            ]
            grid = [[await t for t in row] for row in tasks]

        clean_val = None
        if clean is not None:
            clean_val = (
                clean.clean_logit_diff
                if metric == "logit_diff"
                else clean.clean_logprob
            )

        if metric == "recovered":
            if clean_val is None or corrupt_val is None:
                notes.append("recovered needs clean+corrupt baselines; raw metric kept")
            else:
                denom = clean_val - corrupt_val
                if abs(denom) < 1e-9:
                    notes.append("clean==corrupt baseline; recovered undefined")
                else:
                    grid = [
                        [
                            None if v is None else (v - corrupt_val) / denom
                            for v in row
                        ]
                        for row in grid
                    ]

        return SweepResult(
            layers=layers,
            positions=positions,
            hook=hook,
            metric_name=metric,
            grid=grid,
            clean=clean_val,
            corrupt=corrupt_val,
            notes=notes,
        )

    @staticmethod
    def _metric(
        choice: Any,
        answer_token: str,
        foil_token: str | None,
        metric: str,
    ) -> float | None:
        ans = _answer_logprob(choice, answer_token)
        if metric == "logit_diff":
            if foil_token is None:
                raise ValueError("logit_diff metric requires foil_token")
            foil = _answer_logprob(choice, foil_token)
            if ans is None or foil is None:
                return None
            return ans - foil
        # "logprob" and "recovered" (recovered normalizes logprob downstream).
        return ans

    # ---- step 3: coarse -> fine walk ---------------------------------------

    async def zoom(
        self,
        result: SweepResult,
        *,
        corrupt_prompt: str,
        run: str,
        around: tuple[int, int] | None = None,
        layer_radius: int = 3,
        position_radius: int = 3,
        answer_token: str,
        foil_token: str | None = None,
        alpha: float = 1.0,
        metric: str | None = None,
        clean: CleanRun | None = None,
    ) -> SweepResult:
        """Re-sweep a dense neighborhood around the peak (or ``around``)."""
        center = around or result.argmax_cell()
        c_layer, c_pos = center
        layers = list(
            range(max(0, c_layer - layer_radius), c_layer + layer_radius + 1)
        )
        positions = list(
            range(max(0, c_pos - position_radius), c_pos + position_radius + 1)
        )
        return await self.sweep_layers_positions(
            corrupt_prompt,
            run=run,
            layers=layers,
            positions=positions,
            hook=result.hook,
            alpha=alpha,
            answer_token=answer_token,
            foil_token=foil_token,
            metric=metric or result.metric_name,
            clean=clean,
        )

    def drop_run(self, run: str) -> None:
        """Best-effort source-run drop (no-op if the server lacks the route)."""
        # A dedicated DELETE route is a server-side follow-up; until then the
        # source store evicts whole runs by LRU/budget automatically.
        pass


def _demo() -> None:
    study = PatchStudy(model="google/gemma-3-4b-it")
    clean = study.capture_clean(
        "The Eiffel Tower is in the city of",
        run="paris",
        answer_token=" Paris",
    )
    print("clean:", clean)
    result = asyncio.run(
        study.sweep_layers_positions(
            "The Colosseum is in the city of",
            run="paris",
            layers=range(0, 34, 4),
            positions=range(0, max(clean.num_prompt_tokens, 1)),
            answer_token=" Paris",
            metric="logprob",
        )
    )
    print("peak:", result.argmax_cell())
    print("top 5:", result.top(5))


if __name__ == "__main__":
    _demo()
