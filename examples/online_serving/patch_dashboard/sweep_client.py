# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming client for ``POST /v1/patch_sweep`` used by the Dash dashboard.

Dash-free by design: everything here (payload building, SSE parsing, the
thread-safe grid accumulator) is plain Python over ``requests``, so it is unit
testable without a server and reusable outside the dashboard. The wire
protocol is documented in ``docs/features/activation_patching.md`` (Streaming).
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

DONE = "[DONE]"
INJECTABLE_HOOKS = ("pre_attn", "post_attn", "post_block", "mlp_in", "mlp_out")
METRICS = ("recovered", "logit_diff", "logprob")

# (connect, read) — auto-capture of the clean run happens before the start
# event, so the first byte can take a while on a cold model.
_DEFAULT_TIMEOUT = (10, 600)


class SweepRequestError(ValueError):
    """A sweep request that cannot be built from the given form inputs."""


def visible_token(token: str) -> str:
    """Make a token safe and legible as a plotly tick label.

    Plotly renders labels as pseudo-HTML, so raw special tokens like
    ``<s>`` are parsed as (unclosed) tags and corrupt the axis — escape
    first, then make whitespace visible.
    """
    escaped = token.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return escaped.replace(" ", "␣").replace("\n", "⏎")


def parse_int_list(text: str | None, *, what: str) -> list[int]:
    """Parse a comma-separated int list; empty/None means an empty list."""
    if not text or not text.strip():
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise SweepRequestError(f"{what}: {part!r} is not an integer") from None
    return out


def build_payload(
    *,
    corrupt_prompt: str,
    clean_prompt: str | None,
    source: str,
    mask_indices: list[int] | None,
    hooks: list[str],
    layer_start: int | None,
    layer_stop: int | None,
    layer_step: int | None,
    positions_mode: str,
    span_text: str | None,
    span_occurrence: int,
    position_indices: list[int] | None,
    metric: str,
    answer_token: str | None,
    foil_token: str | None,
    alpha: float,
    keep_source: bool,
    model: str | None,
) -> dict[str, Any]:
    """Validate form inputs and assemble a streaming ``PatchSweepRequest`` body.

    Args:
        source: ``"clean"`` for a capture-sourced (denoising) sweep — the
            server auto-captures ``clean_prompt`` under a fresh run name — or
            ``"zeros"`` for a vector-sourced ablation sweep.
        positions_mode: ``"all_prompt"``, ``"span"`` (substring of the corrupt
            prompt, resolved server-side), or ``"indices"``.

    Raises:
        SweepRequestError: on any invalid or inconsistent input.
    """
    if not corrupt_prompt or not corrupt_prompt.strip():
        raise SweepRequestError("corrupt prompt is required")
    if not hooks or any(h not in INJECTABLE_HOOKS for h in hooks):
        raise SweepRequestError(
            f"hooks must be a non-empty subset of {INJECTABLE_HOOKS}"
        )
    if layer_start is None or layer_stop is None or layer_step is None:
        raise SweepRequestError("layer start/stop/step are required")
    if layer_step < 1 or layer_stop <= layer_start:
        raise SweepRequestError("layer range must have stop > start and step >= 1")
    if metric not in METRICS:
        raise SweepRequestError(f"metric must be one of {METRICS}")
    if not answer_token:
        raise SweepRequestError("answer token is required (grades every cell)")
    if metric == "logit_diff" and not foil_token:
        raise SweepRequestError("logit_diff needs a foil token")

    positions: Any
    if positions_mode == "all_prompt":
        positions = "all_prompt"
    elif positions_mode == "span":
        if not span_text or not span_text.strip():
            raise SweepRequestError("span mode needs a substring")
        positions = [{"span": span_text, "occurrence": span_occurrence or 0}]
    elif positions_mode == "indices":
        if not position_indices:
            raise SweepRequestError("indices mode needs at least one position")
        positions = list(position_indices)
    else:
        raise SweepRequestError(f"unknown positions mode {positions_mode!r}")

    payload: dict[str, Any] = {
        "prompt": corrupt_prompt,
        "hooks": list(hooks),
        "layers": {"start": layer_start, "stop": layer_stop, "step": layer_step},
        "positions": positions,
        "metric": metric,
        "answer_token": answer_token,
        "alpha": alpha,
        "stream": True,
    }
    if model:
        payload["model"] = model
    if foil_token:
        payload["foil_token"] = foil_token

    if source == "clean":
        if not clean_prompt or not clean_prompt.strip():
            raise SweepRequestError(
                "clean prompt is required for a capture-sourced sweep"
            )
        # Fresh run name: the server auto-captures the clean prompt under it
        # (and drops it after the sweep unless keep_source).
        payload["source_run"] = f"dash-{uuid.uuid4().hex[:12]}"
        payload["clean_prompt"] = clean_prompt
        payload["keep_source"] = keep_source
    elif source == "zeros":
        if metric == "recovered":
            raise SweepRequestError(
                "recovered needs a clean baseline; ablation sweeps use "
                "logprob or logit_diff"
            )
        payload["source_module"] = "zeros"
        if mask_indices:
            payload["mask"] = {"indices": list(mask_indices)}
    else:
        raise SweepRequestError(f"unknown source {source!r}")
    return payload


def iter_sse_events(lines: Iterable[bytes | str]) -> Iterator[dict | str]:
    """Yield parsed ``data:`` events from SSE lines, ending at ``[DONE]``."""
    for raw in lines:
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == DONE:
            yield DONE
            return
        yield json.loads(data)


@dataclass
class SweepState:
    """Thread-safe accumulator for one streamed sweep.

    The worker thread applies events; the UI polls :meth:`snapshot`.
    """

    status: str = "connecting"
    error: str | None = None
    hooks: list[str] = field(default_factory=list)
    layers: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    metric: str = ""
    grids: dict[str, list[list[float | None]]] = field(default_factory=dict)
    token_labels: dict[int, str] = field(default_factory=dict)
    cell_errors: list[dict] = field(default_factory=list)
    done_cells: int = 0
    total_cells: int = 0
    auto_captured: bool = False
    captured_source_run: str | None = None
    summary: dict | None = None

    def __post_init__(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._layer_idx: dict[int, int] = {}
        self._pos_idx: dict[int, int] = {}
        self._response: Any = None

    # -- worker side -------------------------------------------------------

    def apply_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "start":
            self._apply_start(event)
        elif kind == "cell":
            self._apply_cell(event)
        elif kind == "summary":
            self._apply_summary(event)

    def _apply_start(self, event: dict) -> None:
        with self._lock:
            self.hooks = list(event.get("hooks") or [event["hook"]])
            self.layers = list(event["layers"])
            self.positions = list(event["positions"])
            self.metric = event.get("metric", "")
            self.auto_captured = bool(event.get("auto_captured"))
            self.captured_source_run = event.get("captured_source_run")
            self._layer_idx = {la: i for i, la in enumerate(self.layers)}
            self._pos_idx = {p: j for j, p in enumerate(self.positions)}
            self.grids = {
                h: [[None] * len(self.positions) for _ in self.layers]
                for h in self.hooks
            }
            self.total_cells = len(self.hooks) * len(self.layers) * len(self.positions)
            self.status = "streaming"

    def _apply_cell(self, event: dict) -> None:
        with self._lock:
            grid = self.grids.get(event.get("hook"))
            i = self._layer_idx.get(event.get("layer"))
            j = self._pos_idx.get(event.get("position"))
            if grid is None or i is None or j is None:
                return
            if event.get("value") is None:
                # A voided cell re-emits as null after its first value landed;
                # record the reason and blank the cell without double-counting.
                grid[i][j] = None
                self.cell_errors.append(
                    {
                        "hook": event["hook"],
                        "layer": event["layer"],
                        "position": event["position"],
                        "error": event.get("error", "voided"),
                    }
                )
                return
            grid[i][j] = event["value"]
            self.done_cells += 1

    def _apply_summary(self, event: dict) -> None:
        with self._lock:
            self.summary = event
            # The summary grids are authoritative (voided cells nulled).
            for entry in event.get("hook_grids") or [event]:
                hook = entry.get("hook")
                if hook in self.grids:
                    self.grids[hook] = [list(r) for r in entry["grid"]]
            self.status = "done"

    def set_token_labels(self, labels: dict[int, str]) -> None:
        with self._lock:
            self.token_labels = dict(labels)

    def attach_response(self, response: Any) -> None:
        with self._lock:
            self._response = response

    def fail(self, message: str) -> None:
        with self._lock:
            if self.status not in ("done", "cancelled"):
                self.status = "error"
                self.error = message

    def mark_cancelled(self) -> None:
        with self._lock:
            if self.status != "done":
                self.status = "cancelled"

    # -- UI side -----------------------------------------------------------

    def cancel(self) -> None:
        """Stop the stream; the server aborts outstanding cells on disconnect."""
        self._cancel.set()
        with self._lock:
            response = self._response
        if response is not None:
            response.close()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def snapshot(self) -> dict[str, Any]:
        """A UI-safe copy of the current state."""
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "hooks": list(self.hooks),
                "layers": list(self.layers),
                "positions": list(self.positions),
                "metric": self.metric,
                "grids": {h: [row[:] for row in g] for h, g in self.grids.items()},
                "token_labels": dict(self.token_labels),
                "cell_errors": [dict(e) for e in self.cell_errors],
                "done_cells": self.done_cells,
                "total_cells": self.total_cells,
                "auto_captured": self.auto_captured,
                "captured_source_run": self.captured_source_run,
                "summary": dict(self.summary) if self.summary else None,
            }


def fetch_token_labels(
    server_url: str,
    model: str | None,
    prompt: str,
    post: Callable[..., Any] = requests.post,
) -> dict[int, str]:
    """Best-effort per-token text labels via ``/tokenize`` + ``/detokenize``.

    Detokenizes each prefix and diffs consecutive prefixes, so multi-byte and
    leading-space tokens render as the tokenizer actually splits them.

    Raises:
        requests.HTTPError: on a non-2xx response from either route.
    """
    root = server_url.rstrip("/")
    body: dict[str, Any] = {"prompt": prompt}
    if model:
        body["model"] = model
    resp = post(f"{root}/tokenize", json=body, timeout=(10, 30))
    _raise_for_status(resp)
    tokens = resp.json()["tokens"]
    labels: dict[int, str] = {}
    previous = ""
    for i in range(1, len(tokens) + 1):
        detok_body: dict[str, Any] = {"tokens": tokens[:i]}
        if model:
            detok_body["model"] = model
        detok = post(f"{root}/detokenize", json=detok_body, timeout=(10, 30))
        _raise_for_status(detok)
        text = detok.json()["prompt"]
        labels[i - 1] = text[len(previous) :] or str(tokens[i - 1])
        previous = text
    return labels


def _raise_for_status(resp: Any) -> None:
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")


def _error_text(resp: Any) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:500]}"
    error = payload.get("error", payload)
    message = error.get("message") if isinstance(error, dict) else None
    return f"HTTP {resp.status_code}: {message or json.dumps(error)[:500]}"


def run_sweep(
    state: SweepState,
    server_url: str,
    payload: dict[str, Any],
    *,
    post: Callable[..., Any] = requests.post,
    label_fn: Callable[..., dict[int, str]] | None = fetch_token_labels,
    timeout: tuple[float, float] = _DEFAULT_TIMEOUT,
) -> None:
    """Worker-thread entry: stream one sweep into ``state`` until done.

    Never raises — every failure mode lands in ``state`` (``error`` or
    ``cancelled``) for the UI to render.
    """
    url = server_url.rstrip("/") + "/v1/patch_sweep"
    try:
        response = post(url, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as exc:
        state.fail(f"request failed: {exc}")
        return
    with response:
        state.attach_response(response)
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            # Pre-fan-out errors (bad hook/layers, span/alignment failure,
            # missing source) come back as plain JSON.
            state.fail(_error_text(response))
            return
        saw_done = False
        try:
            for event in iter_sse_events(response.iter_lines()):
                if state.cancelled:
                    break
                if event == DONE:
                    saw_done = True
                    break
                state.apply_event(event)
                if event.get("type") == "start" and label_fn is not None:
                    _fetch_labels_into(state, server_url, payload, label_fn, post)
        except (requests.RequestException, ValueError) as exc:
            if not state.cancelled:
                state.fail(f"stream broke: {exc}")
                return
    if state.cancelled:
        state.mark_cancelled()
    elif not saw_done and state.snapshot()["status"] != "done":
        state.fail("stream ended before the summary event")


def _fetch_labels_into(
    state: SweepState,
    server_url: str,
    payload: dict[str, Any],
    label_fn: Callable[..., dict[int, str]],
    post: Callable[..., Any],
) -> None:
    """Token labels are cosmetic — a failure must never kill the sweep."""
    try:
        labels = label_fn(
            server_url, payload.get("model"), payload.get("prompt", ""), post=post
        )
    except (requests.RequestException, ValueError, KeyError):
        return
    state.set_token_labels(labels)
