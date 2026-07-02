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
    prompt: str | None = None
    """The clean prompt text — lets sweeps align positions automatically when
    the corrupt prompt tokenizes to a different length."""


def _align_token_positions(
    clean_ids: list[int], corrupt_ids: list[int]
) -> tuple[dict[int, int], list[int]]:
    """(mapping dest->source, unaligned dest positions).

    Equal lengths map identity everywhere (corresponding positions, the
    standard causal-tracing setup). Unequal lengths map the common token
    prefix by identity and the common suffix by the length delta; the
    differing middle has no positional correspondence and is unaligned.
    Mirrors ``vllm/entrypoints/serve/patch/alignment.py``.
    """
    n_clean, n_corrupt = len(clean_ids), len(corrupt_ids)
    if n_clean == n_corrupt:
        return {i: i for i in range(n_corrupt)}, []
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
    mapping = {i: i for i in range(prefix)}
    mapping.update({i: i + shift for i in range(n_corrupt - suffix, n_corrupt)})
    return mapping, list(range(prefix, n_corrupt - suffix))


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
        self._token_id_cache: dict[str, int | None] = {}

    def _grade_token_id(self, token: str | None) -> int | None:
        """Resolve a grading token to its single token id via ``/tokenize``.

        The id is passed as ``logprob_token_ids`` so the server scores it
        exactly on every request — without this, an answer token outside the
        generated top-k grades as ``None`` (top-k boundary flicker). Returns
        ``None`` (with a warning) when the string is not a single token or the
        endpoint is unavailable; grading then falls back to top-k matching.
        """
        if token is None:
            return None
        if token in self._token_id_cache:
            return self._token_id_cache[token]
        import httpx

        token_id: int | None = None
        try:
            root = self.base_url.rstrip("/")
            root = root[: -len("/v1")] if root.endswith("/v1") else root
            r = httpx.post(
                f"{root}/tokenize",
                json={"model": self.model, "prompt": token,
                      "add_special_tokens": False},
                timeout=10.0,
            )
            r.raise_for_status()
            ids = r.json().get("tokens", [])
            if len(ids) == 1:
                token_id = int(ids[0])
            else:
                print(f"warning: {token!r} tokenizes to {len(ids)} tokens; "
                      f"grading falls back to top-k matching")
        except Exception as exc:  # noqa: BLE001 - fallback is functional
            print(f"warning: /tokenize failed ({exc}); grading falls back "
                  f"to top-k matching")
        self._token_id_cache[token] = token_id
        return token_id

    def _grade_ids(self, answer_token: str | None,
                   foil_token: str | None = None) -> list[int] | None:
        ids = [i for i in (self._grade_token_id(answer_token),
                           self._grade_token_id(foil_token)) if i is not None]
        return ids or None

    def _tokenize(self, text: str) -> list[int] | None:
        """Tokenize ``text`` via the server's ``/tokenize`` (None on failure)."""
        import httpx

        try:
            root = self.base_url.rstrip("/")
            root = root[: -len("/v1")] if root.endswith("/v1") else root
            r = httpx.post(
                f"{root}/tokenize",
                json={"model": self.model, "prompt": text,
                      "add_special_tokens": True},
                timeout=10.0,
            )
            r.raise_for_status()
            return [int(t) for t in r.json().get("tokens", [])]
        except Exception as exc:  # noqa: BLE001
            print(f"warning: /tokenize failed ({exc})")
            return None

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
        extra: dict = {"capture": capture, "capture_wait": True}
        grade_ids = self._grade_ids(answer_token, foil_token)
        if grade_ids:
            extra["logprob_token_ids"] = grade_ids
        resp = self._sync.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            # the engine requires logprobs == len(logprob_token_ids) when ids
            # are given (exact scoring replaces top-k)
            logprobs=len(grade_ids) if grade_ids else self.logprobs,
            extra_body=extra,
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
            prompt=prompt,
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
        server_side: bool = False,
    ) -> SweepResult:
        """Fan out one patched request per (layer, position) cell.

        ``metric``: ``"logprob"`` (P(answer)), ``"logit_diff"`` (answer - foil),
        or ``"recovered"`` ((patched - corrupt) / (clean - corrupt)).

        With ``server_side=True`` the whole grid is sent as a single
        ``/v1/patch_sweep`` request — the server expands and batches the cells
        internally (no per-cell HTTP round trips). Returns the same
        :class:`SweepResult`.
        """
        hook = hook or self.hook
        layers = list(layers)
        positions = list(positions)
        notes: list[str] = []

        if server_side:
            return await self._sweep_server_side(
                corrupt_prompt,
                run=run,
                layers=layers,
                positions=positions,
                hook=hook,
                alpha=alpha,
                answer_token=answer_token,
                foil_token=foil_token,
                metric=metric,
                clean=clean,
            )

        grade_ids = self._grade_ids(answer_token, foil_token)

        # Position alignment: when the clean prompt is known, map each dest
        # position to its clean source position. Equal lengths map identity;
        # unequal lengths align prefix/suffix and skip the differing middle
        # (source == dest across a length divergence patches the WRONG
        # positions — a shifted-but-plausible heatmap).
        source_map: dict[int, int] | None = None
        if clean is not None and clean.prompt is not None:
            ids_clean = self._tokenize(clean.prompt)
            ids_corrupt = self._tokenize(corrupt_prompt)
            if ids_clean is not None and ids_corrupt is not None:
                mapping, unaligned = _align_token_positions(ids_clean, ids_corrupt)
                source_map = mapping
                dropped = [p for p in positions if p in set(unaligned)]
                if dropped:
                    positions = [p for p in positions if p not in set(dropped)]
                    notes.append(
                        f"skipped unaligned positions {dropped} (clean/corrupt "
                        f"token spans differ there; no positional "
                        f"correspondence)"
                    )
            elif len(corrupt_prompt) != len(clean.prompt):
                notes.append(
                    "warning: /tokenize unavailable; positions assumed aligned"
                )

        async with AsyncOpenAI(
            base_url=self.base_url, api_key=self.api_key
        ) as aclient:
            sem = asyncio.Semaphore(self.concurrency)

            async def grade(patch: list[dict] | None) -> float | None:
                extra: dict = {"patch": patch} if patch else {}
                if grade_ids:
                    # Exact scoring: the server always reports these ids'
                    # logprobs, so grading never depends on top-k rank.
                    extra["logprob_token_ids"] = grade_ids
                async with sem:
                    resp = await aclient.completions.create(
                        model=self.model,
                        prompt=corrupt_prompt,
                        max_tokens=1,
                        temperature=0.0,
                        # engine requires logprobs == len(ids) with exact ids
                        logprobs=len(grade_ids) if grade_ids else self.logprobs,
                        extra_body=extra,
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
                        "source_position": (
                            source_map[pos] if source_map is not None else pos
                        ),
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

    async def _sweep_server_side(
        self,
        corrupt_prompt: str,
        *,
        run: str,
        layers: list[int],
        positions: list[int],
        hook: str,
        alpha: float,
        answer_token: str,
        foil_token: str | None,
        metric: str,
        clean: CleanRun | None,
    ) -> SweepResult:
        """One POST to /v1/patch_sweep; the server expands + batches the grid."""
        import httpx

        clean_baseline = None
        if clean is not None:
            clean_baseline = (
                clean.clean_logit_diff
                if metric == "logit_diff"
                else clean.clean_logprob
            )
        payload = {
            "model": self.model,
            "prompt": corrupt_prompt,
            "source_run": run,
            # The server aligns positions when the prompts tokenize to
            # different lengths (and 400s a mismatch without clean_prompt).
            "clean_prompt": clean.prompt if clean is not None else None,
            "hook": hook,
            "layers": layers,
            "positions": positions,
            "alpha": alpha,
            "answer_token": answer_token,
            "foil_token": foil_token,
            "metric": metric,
            "logprobs": self.logprobs,
            "clean_baseline": clean_baseline,
        }
        url = f"{self.base_url}/patch_sweep"
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
        notes = []
        if data.get("skipped"):
            notes.append(f"skipped={data['skipped']}")
        if data.get("alignment"):
            notes.append(f"alignment={data['alignment']}")
        return SweepResult(
            layers=data["layers"],
            positions=data["positions"],
            hook=data["hook"],
            metric_name=data["metric"],
            grid=data["grid"],
            clean=data.get("clean"),
            corrupt=data.get("corrupt"),
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
