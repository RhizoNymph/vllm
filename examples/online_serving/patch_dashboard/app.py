# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dash dashboard for live activation-patching sweeps.

Drives ``POST /v1/patch_sweep`` with ``stream: true`` and renders the SSE
cell events as a live-updating (layers x positions) heatmap, one panel per
hook. Point it at a patching-enabled server::

    vllm serve Qwen/Qwen3-0.6B --enable-patching

    python app.py            # http://127.0.0.1:8050

The sweep runs in a background thread (`sweep_client.run_sweep`); a 500 ms
poll re-renders the figure from the accumulated grid. Cancelling closes the
HTTP stream, which makes the server abort the outstanding cells.
"""

from __future__ import annotations

import argparse
import threading
import uuid

import plotly.graph_objects as go
import sweep_client as sc
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ theme
# Reference palette (light mode) — see the repo's dataviz conventions.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "rgba(11,11,11,0.10)"
GOOD = "#006300"
CRITICAL = "#d03b3b"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Sequential blue ramp (steps 100..700): magnitude metrics.
_SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQ_BLUES = [[i / (len(_SEQ) - 1), c] for i, c in enumerate(_SEQ)]

# Diverging red <- neutral gray -> blue: polarity metrics (logit_diff).
_DIV = [
    "#7f2120",
    "#b13435",
    "#d95958",
    "#ec8c86",
    "#f7c4bd",
    "#f0efec",
    "#b7d3f6",
    "#86b6ef",
    "#5598e7",
    "#2a78d6",
    "#184f95",
]
DIVERGING = [[i / (len(_DIV) - 1), c] for i, c in enumerate(_DIV)]

# ------------------------------------------------------------------ sweeps

SWEEPS: dict[str, sc.SweepState] = {}
_SWEEPS_LOCK = threading.Lock()


def _start_sweep(server_url: str, payload: dict) -> str:
    """Register a new sweep, cancel any running one, spawn the worker."""
    state = sc.SweepState()
    sweep_id = uuid.uuid4().hex[:8]
    with _SWEEPS_LOCK:
        for old in SWEEPS.values():
            old.cancel()
        SWEEPS.clear()
        SWEEPS[sweep_id] = state
    threading.Thread(
        target=sc.run_sweep, args=(state, server_url, payload), daemon=True
    ).start()
    return sweep_id


def _get_sweep(sweep_id: str | None) -> sc.SweepState | None:
    with _SWEEPS_LOCK:
        return SWEEPS.get(sweep_id) if sweep_id else None


# ------------------------------------------------------------------ figure


def _axis_ticks(snap: dict) -> tuple[list[int], list[str]]:
    labels = snap["token_labels"]
    ticktext = [
        sc.visible_token(labels[p]) if p in labels else str(p)
        for p in snap["positions"]
    ]
    return snap["positions"], ticktext


def _metric_scale(metric: str) -> dict:
    if metric == "recovered":
        return {"colorscale": SEQ_BLUES, "zmin": 0.0, "zmax": 1.0}
    if metric == "logit_diff":
        return {"colorscale": DIVERGING, "zmid": 0.0}
    return {"colorscale": SEQ_BLUES}


def _placeholder(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=text, showarrow=False, font={"color": MUTED, "size": 14, "family": FONT}
    )
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=320,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return fig


def build_figure(snap: dict, sweep_id: str | None) -> go.Figure:
    """Render the accumulated grids: one heatmap panel per hook."""
    if not snap or not snap.get("hooks"):
        return _placeholder("run a sweep — cells stream in live")
    hooks, layers = snap["hooks"], snap["layers"]
    tickvals, ticktext = _axis_ticks(snap)
    y = [f"L{la}" for la in layers]
    scale = _metric_scale(snap["metric"])

    fig = make_subplots(
        rows=len(hooks),
        cols=1,
        shared_xaxes=True,
        subplot_titles=hooks if len(hooks) > 1 else None,
        vertical_spacing=min(0.3, 0.12 / max(1, len(hooks) - 1))
        if len(hooks) > 1
        else 0.0,
    )
    for row, hook in enumerate(hooks, start=1):
        fig.add_trace(
            go.Heatmap(
                z=snap["grids"][hook],
                x=tickvals,
                y=y,
                name=hook,
                xgap=2,
                ygap=2,
                **scale,
                showscale=(row == 1),
                colorbar={
                    "title": {"text": snap["metric"], "font": {"size": 12}},
                    "thickness": 12,
                    "outlinewidth": 0,
                    "tickfont": {"size": 11, "color": INK_2},
                },
                hovertemplate=(
                    "%{y} · pos %{x}<br>"
                    f"{snap['metric']} %{{z:.4f}}<extra>{hook}</extra>"
                ),
                hoverongaps=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=y,
            row=row,
            col=1,
            tickfont={"size": 11, "color": INK_2},
            ticks="",
        )
    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        ticks="",
        tickangle=-45 if snap["token_labels"] else 0,
        tickfont={"size": 11, "color": INK_2},
    )
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK},
        height=max(340, len(hooks) * (110 + 22 * len(layers))),
        margin={"l": 56, "r": 24, "t": 32 if len(hooks) > 1 else 16, "b": 72},
        uirevision=sweep_id or "none",
    )
    for annotation in fig.layout.annotations or ():
        annotation.font = {"family": FONT, "size": 13, "color": INK_2}
    return fig


# ------------------------------------------------------------------ status


def _status_children(snap: dict) -> list:
    status = snap["status"]
    color = {"error": CRITICAL, "done": GOOD}.get(status, INK_2)
    bits = [html.Span(status, style={"color": color, "fontWeight": 600})]
    if snap["total_cells"]:
        bits.append(
            html.Span(
                f" · {snap['done_cells']} / {snap['total_cells']} cells",
                style={"color": INK_2},
            )
        )
    if snap["auto_captured"] and snap["captured_source_run"]:
        bits.append(
            html.Span(
                f" · auto-captured {snap['captured_source_run']}",
                style={"color": MUTED},
            )
        )
    if snap["error"]:
        bits.append(
            html.Div(
                snap["error"],
                style={
                    "color": CRITICAL,
                    "marginTop": "4px",
                    "whiteSpace": "pre-wrap",
                },
            )
        )
    return bits


def _summary_children(snap: dict) -> list:
    summary = snap.get("summary")
    if not summary:
        return []

    def item(label, value):
        return html.Div(
            [
                html.Span(f"{label} ", style={"color": MUTED}),
                html.Span(value, style={"color": INK}),
            ],
            style={"marginRight": "20px", "display": "inline-block"},
        )

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    items = [
        item("clean", fmt(summary.get("clean"))),
        item("corrupt", fmt(summary.get("corrupt"))),
        item("noise floor", fmt(summary.get("noise_floor"))),
    ]
    for entry in summary.get("hook_grids") or [summary]:
        argmax = entry.get("argmax")
        if argmax:
            items.append(
                item(
                    f"peak {entry.get('hook', snap['metric'])}",
                    f"L{argmax.get('layer')} @ pos {argmax.get('position')} "
                    f"({fmt(argmax.get('value'))})",
                )
            )
    skipped = summary.get("skipped") or []
    if skipped or snap["cell_errors"]:
        items.append(
            item("skipped/voided", str(len(skipped) + len(snap["cell_errors"])))
        )
    return items


# ------------------------------------------------------------------ layout

_INPUT = {
    "width": "100%",
    "boxSizing": "border-box",
    "padding": "6px 8px",
    "border": f"1px solid {HAIRLINE}",
    "borderRadius": "6px",
    "background": SURFACE,
    "color": INK,
    "fontFamily": FONT,
    "fontSize": "13px",
}
_LABEL = {"fontSize": "12px", "color": INK_2, "marginBottom": "2px", "display": "block"}
_CARD = {
    "background": SURFACE,
    "border": f"1px solid {HAIRLINE}",
    "borderRadius": "10px",
    "padding": "16px",
    "marginBottom": "14px",
}


def _field(label: str, control, flex="1 1 160px") -> html.Div:
    return html.Div(
        [html.Label(label, style=_LABEL), control],
        style={"flex": flex, "minWidth": "120px"},
    )


def _row(*children) -> html.Div:
    return html.Div(
        list(children),
        style={
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "marginBottom": "10px",
            "alignItems": "flex-end",
        },
    )


def build_layout() -> html.Div:
    return html.Div(
        style={
            "background": PAGE,
            "minHeight": "100vh",
            "padding": "20px",
            "fontFamily": FONT,
            "color": INK,
            "maxWidth": "1100px",
            "margin": "0 auto",
        },
        children=[
            html.H2(
                "Activation patching — live sweep",
                style={"fontWeight": 600, "fontSize": "18px", "margin": "0 0 14px"},
            ),
            html.Div(
                style=_CARD,
                children=[
                    _row(
                        _field(
                            "Server URL",
                            dcc.Input(
                                id="server-url",
                                value="http://localhost:8000",
                                style=_INPUT,
                            ),
                        ),
                        _field(
                            "Model (optional)",
                            dcc.Input(
                                id="model",
                                value="",
                                placeholder="served model",
                                style=_INPUT,
                            ),
                        ),
                        _field(
                            "Metric",
                            dcc.Dropdown(
                                id="metric",
                                options=[{"label": m, "value": m} for m in sc.METRICS],
                                value="recovered",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ),
                        _field(
                            "Source",
                            dcc.Dropdown(
                                id="source",
                                options=[
                                    {
                                        "label": "clean run (denoising)",
                                        "value": "clean",
                                    },
                                    {"label": "zeros (ablation)", "value": "zeros"},
                                ],
                                value="clean",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ),
                    ),
                    _row(
                        _field(
                            "Clean prompt",
                            dcc.Textarea(
                                id="clean-prompt",
                                value="The Eiffel Tower is in the city of",
                                style={**_INPUT, "height": "54px"},
                            ),
                            flex="1 1 320px",
                        ),
                        _field(
                            "Corrupt prompt",
                            dcc.Textarea(
                                id="corrupt-prompt",
                                value="The Colosseum is in the city of",
                                style={**_INPUT, "height": "54px"},
                            ),
                            flex="1 1 320px",
                        ),
                    ),
                    _row(
                        _field(
                            "Answer token",
                            dcc.Input(id="answer-token", value=" Paris", style=_INPUT),
                        ),
                        _field(
                            "Foil token (logit_diff)",
                            dcc.Input(
                                id="foil-token",
                                value="",
                                placeholder=" Rome",
                                style=_INPUT,
                            ),
                        ),
                        _field(
                            "Alpha",
                            dcc.Input(
                                id="alpha",
                                type="number",
                                value=1.0,
                                min=0,
                                max=1,
                                step=0.05,
                                style=_INPUT,
                            ),
                            flex="0 1 90px",
                        ),
                        _field(
                            "Layers start / stop / step",
                            html.Div(
                                [
                                    dcc.Input(
                                        id="layer-start",
                                        type="number",
                                        value=0,
                                        style={**_INPUT, "width": "31%"},
                                    ),
                                    dcc.Input(
                                        id="layer-stop",
                                        type="number",
                                        value=28,
                                        style={
                                            **_INPUT,
                                            "width": "31%",
                                            "marginLeft": "3%",
                                        },
                                    ),
                                    dcc.Input(
                                        id="layer-step",
                                        type="number",
                                        value=2,
                                        min=1,
                                        style={
                                            **_INPUT,
                                            "width": "31%",
                                            "marginLeft": "3%",
                                        },
                                    ),
                                ]
                            ),
                            flex="1 1 220px",
                        ),
                    ),
                    _row(
                        _field(
                            "Positions",
                            dcc.Dropdown(
                                id="positions-mode",
                                options=[
                                    {
                                        "label": "all prompt tokens",
                                        "value": "all_prompt",
                                    },
                                    {"label": "substring span", "value": "span"},
                                    {"label": "token indices", "value": "indices"},
                                ],
                                value="all_prompt",
                                clearable=False,
                                style={"fontSize": "13px"},
                            ),
                        ),
                        _field(
                            "Span text",
                            dcc.Input(id="span-text", value="Colosseum", style=_INPUT),
                        ),
                        _field(
                            "Span occurrence",
                            dcc.Input(
                                id="span-occurrence",
                                type="number",
                                value=0,
                                min=0,
                                style=_INPUT,
                            ),
                            flex="0 1 110px",
                        ),
                        _field(
                            "Indices (comma-sep)",
                            dcc.Input(
                                id="position-indices",
                                value="",
                                placeholder="0, 4, 7",
                                style=_INPUT,
                            ),
                        ),
                    ),
                    _row(
                        _field(
                            "Hooks",
                            dcc.Checklist(
                                id="hooks",
                                options=[
                                    {"label": h, "value": h}
                                    for h in sc.INJECTABLE_HOOKS
                                ],
                                value=["post_block"],
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={
                                    "marginRight": "14px",
                                    "fontSize": "13px",
                                    "color": INK_2,
                                },
                            ),
                            flex="2 1 420px",
                        ),
                        _field(
                            "Mask dims (zeros source)",
                            dcc.Input(
                                id="mask-indices",
                                value="",
                                placeholder="12, 40, 815",
                                style=_INPUT,
                            ),
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="keep-source",
                                options=[
                                    {
                                        "label": " keep captured source run",
                                        "value": "keep",
                                    }
                                ],
                                value=[],
                                style={"fontSize": "13px", "color": INK_2},
                            ),
                            style={"flex": "0 0 auto", "paddingBottom": "4px"},
                        ),
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Run sweep",
                                id="run",
                                n_clicks=0,
                                style={
                                    "background": "#2a78d6",
                                    "color": "#ffffff",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "padding": "8px 18px",
                                    "fontSize": "13px",
                                    "fontWeight": 600,
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "Cancel",
                                id="cancel",
                                n_clicks=0,
                                style={
                                    "background": SURFACE,
                                    "color": INK_2,
                                    "border": f"1px solid {HAIRLINE}",
                                    "borderRadius": "6px",
                                    "padding": "8px 18px",
                                    "fontSize": "13px",
                                    "marginLeft": "10px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Div(
                                id="status",
                                style={
                                    "display": "inline-block",
                                    "marginLeft": "16px",
                                    "fontSize": "13px",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        style={
                            "height": "4px",
                            "background": "#f0efec",
                            "borderRadius": "2px",
                            "marginTop": "12px",
                            "overflow": "hidden",
                        },
                        children=html.Div(
                            id="progress-fill",
                            style={
                                "height": "100%",
                                "width": "0%",
                                "background": "#2a78d6",
                                "transition": "width 300ms",
                            },
                        ),
                    ),
                ],
            ),
            html.Div(
                style=_CARD,
                children=[
                    dcc.Graph(
                        id="heatmap",
                        figure=_placeholder("run a sweep — cells stream in live"),
                        config={"displaylogo": False},
                    ),
                    html.Div(
                        id="summary", style={"fontSize": "13px", "marginTop": "6px"}
                    ),
                ],
            ),
            dcc.Store(id="sweep-id"),
            dcc.Interval(id="poll", interval=500, disabled=True),
        ],
    )


# ------------------------------------------------------------------ wiring

app = Dash(__name__, title="vLLM patch sweeps")
app.layout = build_layout()


@app.callback(
    Output("sweep-id", "data"),
    Output("poll", "disabled"),
    Output("status", "children"),
    Input("run", "n_clicks"),
    State("server-url", "value"),
    State("model", "value"),
    State("clean-prompt", "value"),
    State("corrupt-prompt", "value"),
    State("source", "value"),
    State("mask-indices", "value"),
    State("hooks", "value"),
    State("layer-start", "value"),
    State("layer-stop", "value"),
    State("layer-step", "value"),
    State("positions-mode", "value"),
    State("span-text", "value"),
    State("span-occurrence", "value"),
    State("position-indices", "value"),
    State("metric", "value"),
    State("answer-token", "value"),
    State("foil-token", "value"),
    State("alpha", "value"),
    State("keep-source", "value"),
    prevent_initial_call=True,
)
def on_run(
    _clicks,
    server_url,
    model,
    clean_prompt,
    corrupt_prompt,
    source,
    mask_indices,
    hooks,
    layer_start,
    layer_stop,
    layer_step,
    positions_mode,
    span_text,
    span_occurrence,
    position_indices,
    metric,
    answer_token,
    foil_token,
    alpha,
    keep_source,
):
    try:
        payload = sc.build_payload(
            corrupt_prompt=corrupt_prompt,
            clean_prompt=clean_prompt,
            source=source,
            mask_indices=sc.parse_int_list(mask_indices, what="mask dims"),
            hooks=hooks or [],
            layer_start=layer_start,
            layer_stop=layer_stop,
            layer_step=layer_step,
            positions_mode=positions_mode,
            span_text=span_text,
            span_occurrence=span_occurrence or 0,
            position_indices=sc.parse_int_list(
                position_indices, what="position indices"
            ),
            metric=metric,
            answer_token=answer_token,
            foil_token=foil_token,
            alpha=alpha if alpha is not None else 1.0,
            keep_source=bool(keep_source),
            model=model or None,
        )
    except sc.SweepRequestError as exc:
        return no_update, no_update, html.Span(str(exc), style={"color": CRITICAL})
    sweep_id = _start_sweep(server_url or "http://localhost:8000", payload)
    return sweep_id, False, html.Span("connecting…", style={"color": INK_2})


@app.callback(
    Output("status", "children", allow_duplicate=True),
    Input("cancel", "n_clicks"),
    State("sweep-id", "data"),
    prevent_initial_call=True,
)
def on_cancel(_clicks, sweep_id):
    state = _get_sweep(sweep_id)
    if state is None:
        return no_update
    state.cancel()
    return html.Span("cancelling…", style={"color": INK_2})


@app.callback(
    Output("heatmap", "figure"),
    Output("status", "children", allow_duplicate=True),
    Output("progress-fill", "style"),
    Output("summary", "children"),
    Output("poll", "disabled", allow_duplicate=True),
    Input("poll", "n_intervals"),
    State("sweep-id", "data"),
    prevent_initial_call=True,
)
def on_tick(_n, sweep_id):
    state = _get_sweep(sweep_id)
    if state is None:
        return no_update, no_update, no_update, no_update, True
    snap = state.snapshot()
    finished = snap["status"] in ("done", "error", "cancelled")
    pct = (
        100.0 * snap["done_cells"] / snap["total_cells"] if snap["total_cells"] else 0.0
    )
    fill = {
        "height": "100%",
        "width": f"{pct:.1f}%",
        "background": GOOD if snap["status"] == "done" else "#2a78d6",
        "transition": "width 300ms",
    }
    figure = (
        build_figure(snap, sweep_id)
        if snap["grids"]
        else _placeholder(f"{snap['status']}…")
    )
    return (figure, _status_children(snap), fill, _summary_children(snap), finished)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
