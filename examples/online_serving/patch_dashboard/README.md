# Activation-Patching Dashboard (Dash)

A minimal [Dash](https://dash.plotly.com/) dashboard for interactive
activation-patching studies against a patching-enabled vLLM server. It drives
`POST /v1/patch_sweep` with `stream: true` and renders the SSE cell events as
a **live-updating (layers × positions) heatmap** — one panel per hook — so a
1000-cell causal-tracing sweep fills in as the server finishes each cell
instead of blocking on one long request.

See [docs/features/activation_patching.md](../../../docs/features/activation_patching.md)
for the sweep API this consumes (grid expansion, metrics, auto-capture,
streaming protocol).

## What it does

- **Denoising (capture-sourced) sweeps** — enter a clean and a corrupt prompt;
  the server auto-captures the clean run in the same call (no explicit
  capture step) and patches each grid cell from it. Metrics: `recovered`,
  `logit_diff`, `logprob`.
- **Ablation (vector-sourced) sweeps** — source `zeros` with optional mask
  dims zero-ablates residual dims per cell; no clean run involved.
- **Multi-hook grids** — tick several hooks (`pre_attn`, `post_attn`,
  `post_block`, `mlp_in`, `mlp_out`) for the classic attention-vs-MLP
  decomposition; each hook renders as its own heatmap panel sharing one
  color scale.
- **Positions** — all prompt tokens, a substring span (resolved server-side to
  token positions), or explicit token indices. The x-axis is labeled with the
  actual tokens (fetched best-effort via `/tokenize` + `/detokenize`).
- **Live progress + cancel** — a progress bar tracks cells landed; Cancel
  closes the HTTP stream, which makes the server abort the outstanding cells.
- **Summary** — clean/corrupt baselines, empirical noise floor, per-hook peak
  cell, and any skipped/voided cells once the sweep completes.

## Running

Start a patching-enabled server (any steering/capture-wired model works;
MLP hooks are wired on the gemma3/gemma4 and qwen3 families):

```bash
vllm serve Qwen/Qwen3-0.6B --enable-patching
```

Install the dashboard's dependencies (isolated from the vLLM env is fine —
the app talks HTTP only, it does not import vllm):

```bash
uv venv --python 3.12 .venv-dash
.venv-dash/bin/pip install -r requirements.txt
.venv-dash/bin/python app.py    # http://127.0.0.1:8050
```

Fill in the clean/corrupt prompts, the answer token (grades every cell), a
layer range, and hit **Run sweep**. Cells stream in as they land.

## Files

- `app.py` — Dash layout, callbacks, and the plotly heatmap. A background
  thread streams the sweep; a 500 ms `dcc.Interval` polls the accumulated
  grid into the figure.
- `sweep_client.py` — Dash-free client: request-body building/validation,
  SSE event parsing, and the thread-safe `SweepState` grid accumulator.
  Reusable outside the dashboard.
- `test_sweep_client.py` — unit tests for the client (no server needed):
  `pytest test_sweep_client.py`.

## Notes

- Single-user example: starting a new sweep cancels the previous one, and
  state lives in the Dash process (run with the default single worker).
- Pre-fan-out errors (bad hook/layers, span not found, missing source run)
  come back as plain JSON 400s and are shown in the status line; the SSE
  stream only starts once the grid fan-out begins.
- Grid differences at or below the reported `noise floor` are not meaningful
  (vLLM is not batch-invariant by default — see
  `docs/features/batch_invariance.md`).
