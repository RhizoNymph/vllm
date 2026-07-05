# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-patching sweep client for the OpenAI-compatible server.

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

Import it from the installed package::

    from vllm.entrypoints.serve.patch.client import PatchStudy, Span

See ``examples/online_serving/openai_patch_client.py`` for a runnable demo.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI, OpenAI

from vllm.entrypoints.serve.patch.alignment import align_token_positions
from vllm.entrypoints.serve.patch.spans import (
    dedup_positions,
    incremental_char_offsets,
    resolve_span_positions as _resolve_span_positions,
)


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


@dataclass(frozen=True)
class Span:
    """A sweep-position marker resolved to the token positions covering a
    substring of the prompt.

    Pass anywhere a position index is accepted in ``positions``; the sweep
    resolves it against the corrupt prompt (the destination run) via the
    server's ``/tokenize`` endpoint, exactly as the sweep tokenizes it.
    """

    text: str
    occurrence: int = 0
    """Which match to use when ``text`` appears more than once (0 = first)."""


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
    auto_captured: bool = False
    """True when a server-side sweep auto-captured the clean run in one call
    (``clean_prompt`` given, ``source_run`` was missing) rather than reusing a
    prior capture."""
    captured_source_run: str | None = None
    """The run handle a one-call auto-capture wrote under, else ``None``."""

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

    def _root_url(self) -> str:
        """Base URL with the trailing ``/v1`` stripped (for ``/tokenize`` etc.)."""
        root = self.base_url.rstrip("/")
        return root[: -len("/v1")] if root.endswith("/v1") else root

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
            r = httpx.post(
                f"{self._root_url()}/tokenize",
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
            r = httpx.post(
                f"{self._root_url()}/tokenize",
                json={"model": self.model, "prompt": text,
                      "add_special_tokens": True},
                timeout=10.0,
            )
            r.raise_for_status()
            return [int(t) for t in r.json().get("tokens", [])]
        except Exception as exc:  # noqa: BLE001
            print(f"warning: /tokenize failed ({exc})")
            return None

    def _detokenize(self, ids: Sequence[int]) -> str | None:
        """Detokenize ``ids`` via the server's ``/detokenize`` (None on fail)."""
        import httpx

        try:
            r = httpx.post(
                f"{self._root_url()}/detokenize",
                json={"model": self.model, "tokens": list(ids)},
                timeout=10.0,
            )
            r.raise_for_status()
            return str(r.json().get("prompt", ""))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: /detokenize failed ({exc})")
            return None

    def _token_char_offsets(
        self, ids: Sequence[int]
    ) -> tuple[str, list[tuple[int, int]]]:
        """Per-token character offsets, by incremental detokenization.

        The server exposes no offset mapping and ``token_strs`` carry raw
        subword markers (``Ġ``/``▁``/byte fallbacks) whose reconstruction is
        tokenizer-family-specific. Detokenizing growing prefixes of the exact
        ``ids`` instead yields offsets in the same detokenized space the search
        runs in, for any tokenizer. Special tokens (e.g. BOS) detokenize to an
        empty span and are simply never selected.

        Returns:
            ``(text, offsets)`` where ``text`` is the full detokenization and
            ``offsets[k]`` is the half-open ``(start, end)`` char span of token
            ``k`` in ``text``.
        """
        return incremental_char_offsets(self._detokenize, ids)

    async def positions_for(
        self, prompt: str, span: str, *, occurrence: int = 0
    ) -> list[int]:
        """Token positions in ``prompt`` covering the substring ``span``.

        Tokenizes ``prompt`` with the same ``/tokenize`` semantics the sweep
        uses, so the returned positions index the tokenized prompt exactly as
        the server sees it. Character offsets are reconstructed by incremental
        detokenization.

        Args:
            prompt: The prompt to search (typically the corrupt prompt).
            span: The substring whose covering token positions are wanted.
            occurrence: Which match to use when ``span`` repeats (0 = first).

        Returns:
            Ascending token positions overlapping the chosen match.

        Raises:
            ValueError: ``span`` is empty, not found, or ``occurrence`` is out
                of range.
            RuntimeError: ``/tokenize`` or ``/detokenize`` is unavailable.
        """
        ids = self._tokenize(prompt)
        if ids is None:
            raise RuntimeError("/tokenize unavailable; cannot resolve span")
        text, offsets = self._token_char_offsets(ids)
        return _resolve_span_positions(offsets, text, span, occurrence)

    async def _resolve_positions(
        self, positions: Sequence[int] | Span, prompt: str
    ) -> list[int]:
        """Expand any :class:`Span` markers in ``positions`` against ``prompt``.

        Plain ints pass through; spans resolve to their covering positions.
        Order is preserved and duplicates dropped.
        """
        items: Sequence[Any] = [positions] if isinstance(positions, Span) else positions
        groups: list[list[int]] = []
        for item in items:
            if isinstance(item, Span):
                groups.append(
                    await self.positions_for(
                        prompt, item.text, occurrence=item.occurrence
                    )
                )
            else:
                groups.append([int(item)])
        return dedup_positions(groups)

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
        run: str | None = None,
        layers: Sequence[int],
        positions: Sequence[int | Span] | Span,
        hook: str | None = None,
        alpha: float = 1.0,
        answer_token: str,
        foil_token: str | None = None,
        metric: str = "logprob",
        clean: CleanRun | None = None,
        clean_prompt: str | None = None,
        server_side: bool = False,
    ) -> SweepResult:
        """Fan out one patched request per (layer, position) cell.

        ``positions`` accepts token indices and/or :class:`Span` markers; each
        span resolves to the corrupt-prompt token positions covering its text.
        With ``server_side=True`` spans are forwarded to the server (resolved
        there); the per-cell path resolves them client-side.

        ``metric``: ``"logprob"`` (P(answer)), ``"logit_diff"`` (answer - foil),
        or ``"recovered"`` ((patched - corrupt) / (clean - corrupt)).

        The clean run to patch from is chosen as: explicit ``run=``, else the
        ``clean`` handle's ``run_id``, else — for ``server_side=True`` with
        ``clean_prompt`` — a fresh per-call run the server auto-captures (fresh
        because the auto-capture taps only the swept layers, so reusing a name
        across grids with different layers would 400).

        ``clean_prompt``: the clean prompt text. With ``server_side=True`` and
        no existing run it drives one-call auto-capture (the server captures it
        before running the grid, and ``SweepResult.auto_captured`` /
        ``captured_source_run`` report it); with an existing run it only aligns
        positions. The per-cell path (``server_side=False``) cannot auto-capture
        — pass a captured ``clean`` handle there instead.

        With ``server_side=True`` the whole grid is sent as a single
        ``/v1/patch_sweep`` request — the server expands and batches the cells
        internally (no per-cell HTTP round trips). Returns the same
        :class:`SweepResult`.
        """
        hook = hook or self.hook
        layers = list(layers)
        notes: list[str] = []

        # Effective source run: explicit run wins, else the clean handle's run.
        source_run = run if run is not None else (
            clean.run_id if clean is not None else None
        )

        if not server_side and clean_prompt is not None and clean is None:
            raise ValueError(
                "clean_prompt drives server-side auto-capture; the per-cell "
                "path (server_side=False) has no capture endpoint — call "
                "capture_clean() and pass clean=<CleanRun>, or set "
                "server_side=True."
            )

        if server_side:
            # clean_prompt (explicit) wins over the clean handle's prompt.
            send_clean_prompt = (
                clean_prompt if clean_prompt is not None
                else (clean.prompt if clean is not None else None)
            )
            if source_run is None:
                if send_clean_prompt is None:
                    raise ValueError(
                        "server-side sweep needs run=, a captured clean=, or "
                        "clean_prompt= for one-call auto-capture."
                    )
                # Fresh name per call: the server auto-captures only the swept
                # layers, so a name reused across differing-layer grids 400s.
                source_run = uuid4().hex
            return await self._sweep_server_side(
                corrupt_prompt,
                run=source_run,
                layers=layers,
                positions=positions,
                hook=hook,
                alpha=alpha,
                answer_token=answer_token,
                foil_token=foil_token,
                metric=metric,
                clean=clean,
                clean_prompt=send_clean_prompt,
            )

        if source_run is None:
            raise ValueError(
                "sweep needs run= (capture_clean() first) or server_side=True."
            )
        run = source_run
        positions = await self._resolve_positions(positions, corrupt_prompt)

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
                alignment = align_token_positions(ids_clean, ids_corrupt)
                source_map = alignment.mapping
                unaligned = set(alignment.unaligned)
                dropped = [p for p in positions if p in unaligned]
                if dropped:
                    positions = [p for p in positions if p not in unaligned]
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

    @staticmethod
    def _encode_positions(
        positions: Sequence[int | Span] | Span,
    ) -> list[int | dict]:
        """Forward positions to the server, spans as ``{span, occurrence}``."""
        items = [positions] if isinstance(positions, Span) else positions
        out: list[int | dict] = []
        for item in items:
            if isinstance(item, Span):
                out.append({"span": item.text, "occurrence": item.occurrence})
            else:
                out.append(int(item))
        return out

    async def _sweep_server_side(
        self,
        corrupt_prompt: str,
        *,
        run: str,
        layers: list[int],
        positions: Sequence[int | Span] | Span,
        hook: str,
        alpha: float,
        answer_token: str,
        foil_token: str | None,
        metric: str,
        clean: CleanRun | None,
        clean_prompt: str | None,
    ) -> SweepResult:
        """One POST to /v1/patch_sweep; the server expands + batches the grid.

        Spans in ``positions`` are forwarded as objects (the server resolves
        them against the corrupt prompt). ``clean_prompt`` drives server-side
        alignment and, when ``run`` is missing, one-call auto-capture.
        """
        import httpx

        # An auto-capturing sweep grades the clean baseline itself; only send a
        # precomputed baseline when reusing an explicit clean handle.
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
            # different lengths (and 400s a mismatch without clean_prompt);
            # a missing run + clean_prompt triggers one-call auto-capture.
            "clean_prompt": clean_prompt,
            "hook": hook,
            "layers": layers,
            "positions": self._encode_positions(positions),
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
            auto_captured=data.get("auto_captured", False),
            captured_source_run=data.get("captured_source_run"),
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
