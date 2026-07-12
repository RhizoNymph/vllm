# Activation Patching

Activation patching overwrites (or interpolates toward) the residual-stream
activation at specific `(layer, hook, position)` sites of a *destination* run
with activations captured from a prior *clean/source* run. It is the core
causal-tracing primitive of mechanistic interpretability: run a clean prompt
once, then re-run a corrupted prompt while splicing in the clean run's
activations at chosen sites to measure which sites restore the clean behavior.

Patching reuses the [activation capture](capture_consumers.md) and
[activation steering](steering.md) machinery. It *is* steering with three
changes: replace/lerp instead of add, per-`(request, layer, hook, position)`
values instead of one vector per config, and values sourced from a prior
capture run.

## Supported Scope

- **Hook points:** `pre_attn`, `post_attn`, `post_block` (residual stream) and
  `mlp_in`, `mlp_out` (MLP branch) — all five injection-capable
  `SteeringHookPoint`s. `post_block` uses a dedicated two-tensor op because
  vLLM defers each layer's MLP-branch add — see [Invariants](#invariants). The
  MLP hooks patch the branch tensor directly (single-tensor op, no
  reconstruction): `mlp_in` is the normed MLP input, `mlp_out` the branch
  before its residual add — replacing `mlp_out` is the classic
  attention-vs-MLP causal decomposition. The MLP taps are wired on gemma3,
  gemma4, and the qwen3 family; on unwired models an mlp patch stages but no
  emission site reads it (a silent no-op), so use the wired models for
  MLP-hook studies. After an
  `mlp_out` intervention, downstream captures (`post_block`) see the
  intervened branch — `post_block == post_attn + mlp_out` holds for pristine
  runs only.
- **Models:** every architecture wired for steering/capture (the patch buffers
  are folded into `register_steering_buffers`, so patching attaches wherever
  steering does — zero per-model changes). **Text-only prompts:** multimodal
  prompts are rejected at admission — prompt positions include image
  placeholder tokens, so patch positions would target placeholder activations
  (semantically undefined and unvalidated).
- **Runners:** both the v1 and v2 GPU model runners.
- **Parallelism:** TP and PP (source vectors are resolved rank-locally; TP rank
  0 resolves and broadcasts, PP stores are partitioned by owned layers).
- **Execution:** eager and CUDA graph (patch buffers are persistent, written in
  place before the forward, so a FULL cudagraph replay reads the step's values —
  no force-eager seam, unlike capture's dynamic gather).
- **Server-side sweeps:** `POST /v1/patch_sweep` expands a `(layers × positions)`
  grid (optionally × hooks) into one densely-batched call.

End-to-end GPU-validated with real weights (eager + cudagraph, including the
multi-rank configs):

- **Qwen3-0.6B** (v2 runner, default): TP1/PP1, TP2/PP1, TP1/PP2.
- **gemma3-4b** (v1 runner): TP1/PP1, TP2/PP1, TP1/PP2.

Both prove no-op / self-identity are bit-exact, cross-run replace reproduces the
clean run within bf16 accumulation, and single-site denoising surfaces the clean
answer.

The MLP branch hooks are GPU-validated on **Qwen3-0.6B** TP1/PP1 (eager +
cudagraph): no-op and self-identity at `mlp_in`/`mlp_out` are bit-exact,
zero-ablating `mlp_out` lands (output changes), additive steering at
`mlp_out` matches `post_block` steering within fp addition-order noise, and
single-site `mlp_out` denoising recovers a majority share of the same site's
`post_block` shift (`tests/gpu_patch_validate.py` checks G–K). Multi-rank
mlp-hook validation has not been run yet (the hooks share the residual
hooks' replicated-tensor contract, so the same TP/PP machinery applies).

## Frontends

- **Python OpenAI server** (`vllm.entrypoints.openai.api_server`): the full
  surface — per-request `patch=`, admission-time source-existence `400`s, the
  multimodal guard, and the server-side `/v1/patch_sweep` grid sweeps (with
  auto-capture, substring positions, multi-hook grids, SSE streaming, and the
  source-run lifecycle route).
- **Offline `LLM()`** and the **Rust frontend**: per-request `patch=` with
  engine-side admission (`vllm/v1/engine/input_processor.py`), which now stamps
  the *precise* prefix-cache floor (APC position-windowing) rather than failing
  safe to no-reuse as it did before. Source existence is not pre-checked at
  admission on these paths — a missing source is caught by the worker
  resolution-failure registry backstop.
- **Rust frontend, server-side sweeps:** the `/v1/patch_sweep` surface is served
  through the Rust frontend by an auto-spawned Python **patch sidecar**. When
  vLLM launches with the Rust frontend and `--enable-patching`, the driver also
  starts one loopback Python `api_server` (`--patch-sidecar-port`, default
  auto-picked) attached to the *same* engines as a second engine client — one
  weight set, one KV cache, one shared worker-side `PatchSourceStore` — and the
  Rust server reverse-proxies `POST /v1/patch_sweep` and
  `DELETE /v1/patch_source/{run_id}` to it, streaming SSE chunks through
  unbuffered and propagating client disconnect upstream. Set
  `VLLM_RUST_PATCH_SIDECAR=0` to skip the sidecar; the Rust frontend then
  returns HTTP 501 for those routes. See the Rust frontend's README for details.

## Request API

A patch spec is a list of site entries on `SamplingParams`:

```python
patch = [
    {
        "layer": 14,
        "hook": "post_block",
        "dest_position": 6,       # position in the destination (corrupt) run
        "source_run": "clean",    # a prior capture run handle
        "source_position": 6,     # position in the source run
        "alpha": 1.0,             # 1.0 = replace; (1-a)*h + a*source otherwise
    },
]
```

Each entry overwrites (`alpha == 1`) or interpolates the destination activation
at `(layer, hook, dest_position)` toward `source_run`'s activation at
`source_position`. The interpolation is the precise form `(1-a)*h + a*source`
(endpoint-exact at `alpha == 1`).

Positions are integer token indices — only the `/v1/patch_sweep` endpoint
resolves substring spans server-side. To target a substring in a per-request
patch without counting tokens by hand, resolve it first with the client helper
(the same span math the sweep uses — see
[Substring positions](#substring-positions)) and expand the resolved positions
into entries:

```python
from vllm.entrypoints.serve.patch.client import PatchStudy

study = PatchStudy(model=MODEL, hook="post_block")
positions = await study.positions_for(corrupt_prompt, "Germany")
patch = [
    {"layer": 14, "hook": "post_block", "dest_position": p,
     "source_run": "clean", "source_position": p, "alpha": 1.0}
    for p in positions
]
```

### Standalone per-request use (outside a study)

`patch` (and `patch_vectors`) are ordinary per-request sampling fields, not a
study concept: any `/v1/completions` or `/v1/chat/completions` request — or an
offline `SamplingParams` — can carry them, with no sweep endpoint, capture run,
or client library involved. Only the carrying request is intervened on (per-row
gating in a mixed continuous batch); sampling, streaming, stop handling, and
everything else behave normally. `dest_position` is the 0-based logical token
position (prompt positions first; generated positions continue the count and
are patched on the step that computes them).

Zero-ablate two residual dims for a single completion and grade the answer —
no prior capture needed:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "The capital of France is",
  "max_tokens": 1, "temperature": 0.0, "logprobs": 5,
  "patch": [{"layer": 14, "hook": "post_block", "dest_position": 4,
             "source_module": "zeros", "mask": {"indices": [12, 815]}}]
}'
```

The same `patch` field works on chat requests (text-only: patch + multimodal
content is rejected before rendering). For exact answer-token grading use the
completions-only `logprob_token_ids` field (the requested ids replace top-k);
chat exposes top-k `logprobs` only.

`patch` is a top-level extra field in the OpenAI request body, alongside the
standard params. The same body posts to `POST /v1/chat/completions` (top-k
`logprobs` only — no `logprob_token_ids`):

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "The capital of France is"}],
  "max_tokens": 1, "temperature": 0.0, "logprobs": true, "top_logprobs": 5,
  "patch": [
    {"layer": 14, "hook": "post_block", "dest_position": 4,
     "source_module": "zeros", "mask": {"indices": [12, 815]}}
  ]
}
```

Capture-sourced (`source_run`/`source_position`) and named-module
(`source_module: "<name>"`) entries take the same shape — only the contents of
the `patch` entries change. `patch_vectors` (the packed table for
`source_inline`/mask `inline`) is a sibling top-level field the same way.

To patch from a *named* vector, register a steering module once (a steering
mutation — send the `Authorization: Bearer <key>` header if the server sets
`--steering-api-key`) and reference it — the same handle can be steered with
(add) or patched in (replace):

```bash
curl -s localhost:8000/v1/steering/modules/register \
  -H 'Content-Type: application/json' -d '{
  "name": "dataset_mean",
  "vectors": {"post_block": {"14": {"vector": [/* hidden_size floats */]}}}
}'
# then, on any request:
#   "patch": [{"layer": 14, "hook": "post_block", "dest_position": 4,
#              "source_module": "dataset_mean"}]
```

Offline, the same spec rides `SamplingParams` directly:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enable_patching=True)
out = llm.generate(
    "The capital of France is",
    SamplingParams(
        max_tokens=1, temperature=0.0, logprobs=5,
        patch=[{"layer": 14, "hook": "post_block", "dest_position": 4,
                "source_module": "zeros", "mask": {"indices": [12, 815]}}],
    ),
)
```

Capture-sourced entries (`source_run`/`source_position`) work per-request the
same way — run the clean prompt once with the `patch_source` capture spec and
`capture_wait=True` (see below), then reference the run from any later request.

Per-request safety properties, all automatic: patched KV is taint-keyed
(`patch_kv_taint`) so it can never be served to, or poison, other requests via
the prefix cache; prompt positions at or above the lowest patched position are
recomputed while the prefix below still reuses cache; a single request whose
per-site demand exceeds the usable pool (`--max-patch-slots` minus the
passthrough slot) is rejected with a 400, and concurrent patched requests are
queued by scheduler backpressure rather than failed.

### Patch value sources

The source vector need not come from a capture run. Each entry sets **exactly
one** source kind:

- **`source_run` + `source_position`** — a prior server-side capture run's
  stored activation (the classic causal-tracing path above).
- **`source_module: "<name>"`** — resolved worker-side against the existing
  *named steering module* registry (no new registry). The value is the module's
  BASE `vectors` tier row at the same `(hook, layer)` the entry patches; a
  `{"vector": [...], "scale": s}` registry entry resolves to `s * vector`.
- **`source_module: "zeros"`** — a reserved built-in: a zero row of the hook's
  width. Needs no registry and no store, so it works everywhere including
  offline (`LLM(...)`). Zero-ablate residual dims by pairing it with a `mask`.
- **`source_inline: <row_index>`** — an index into a request-level packed table
  `patch_vectors` (see below). Packed-only (no raw float-list form — inline
  float lists caused a large E2EL regression; the base64 packed form avoids it).

`alpha` defaults `1.0` and applies to every kind.

#### Per-dim masks

An optional **`mask`** (composes with **any** source kind, including
`source_run`) restricts the patch to a subset of dims:

```
out_d = (1 - alpha * m_d) * hs_d + alpha * m_d * src_d
```

Because `alpha * mask` is just a per-dimension `alpha`, a mask needs no separate
kernel path — it is folded into the per-dim alpha row at staging time (unmasked
entries stage a constant `alpha` fill). Wire forms per entry:

- `{"indices": [int, ...]}` — sparse; expanded worker-side to a 0/1 mask.
- `{"inline": <row_index>}` — a row of `patch_vectors` (values in `[0, 1]`,
  graded/soft masks allowed).

#### Packed `patch_vectors` table

`source_inline` and mask `inline` index a request-level field carried verbatim
on `SamplingParams` and decoded **once per request** at worker-side resolution.
It uses the same binary wire encoding as `SteeringHookPacked`
(`vllm/config/steering_types.py`) minus `layer_indices`/`scales`:

```json
{"dtype": "float32|float16|bfloat16", "shape": [n_rows, width],
 "data": "<base64 contiguous bytes>"}
```

`width` must equal the hook width (`hidden_size` for all five injectable
hooks). Structural errors (bad base64, shape/length mismatch, out-of-range row
index, wrong width) are rejected at admission (HTTP 400); malformed structure on
the direct `SamplingParams` path raises `ValueError`.

#### Neuron / dim clamping

To pin specific residual dims to zero across a run, combine `source_module:
"zeros"` with a `mask`:

```python
patch = [{"layer": 14, "hook": "post_block", "dest_position": 6,
          "source_module": "zeros", "mask": {"indices": [12, 40, 815]}}]
```

The residual-stream sites (`pre_attn`/`post_attn`/`post_block`) clamp
residual dims. `mlp_in`/`mlp_out` are also injectable, but they are
`hidden_size`-wide branch tensors — clamping *MLP-width* (intermediate)
neurons remains out of scope; zeroing `mlp_out` dims ablates the MLP
branch's contribution to those residual dims for that token.

#### Offline named-module caveat

Named-module *name* existence is validated at admission only when a frontend
steering registry is present (the OpenAI server). Offline (`LLM(...)`) has no
such registry, so a `source_module` name other than `"zeros"` cannot be
admission-checked; a bad name surfaces loudly via the worker resolution-failure
registry (the request runs unpatched and its output is not trusted). Structural
validation (source-kind exclusivity, mask shape, packed table) still runs
offline.

#### Registry reuse rationale

`source_module` deliberately reuses the steering module registry rather than
introducing a patch-specific one: named modules already broadcast to every
worker (so TP rank 0 can resolve and broadcast like any other source), already
carry the `{hook: {layer: entry}}` shape patching addresses by `(hook, layer)`,
and already have a frontend existence check to mirror.

The clean run is captured once via the `patch_source` capture consumer:

```python
capture = {
    "patch_source": {
        "run": "clean",
        "hooks": {"post_block": "all"},
        "positions": "all_prompt",
    }
}
```

Capture writes are asynchronous; set `capture_wait=True` on the clean request
(HTTP) so the source store is durable before any patch references it. This
explicit pre-capture is still required for probe / single-cell patch requests
(`patch` on `SamplingParams`). For grid sweeps it is now *optional*: passing
`clean_prompt` to `/v1/patch_sweep` lets the server auto-capture the clean run
in the same call (see [Server-side sweeps](#server-side-sweeps)).

## Server-side sweeps

`POST /v1/patch_sweep` runs a whole `(layers × positions)` grid in one call: the
server expands the grid into one patched variant per cell, fans them to the
continuously-batched engine (the shared corrupt-prompt prefix is reused via
prefix caching; each variant is patched at its own site via per-row gating), and
returns the assembled metric grid. Metrics: `logprob`, `logit_diff`
(answer − foil), `recovered` `(patched − corrupt) / (clean − corrupt)`.

### Vector-sourced (ablation) sweeps

A sweep is either **capture-sourced** (the default — `source_run` names a stored
clean run, with optional one-call auto-capture) or **vector-sourced**: every
cell is patched from the *same* client-provided value (`source_module` — a named
module or `"zeros"` — or `source_inline` into a request-level `patch_vectors`
table), optionally through a shared `mask`. A vector-sourced sweep sets exactly
one of `source_module` / `source_inline` and leaves `source_run` / `clean_prompt`
unset; the server skips auto-capture, the source-manifest existence check, and
the source-run lifecycle (nothing to drop). The unpatched corrupt run is still
measured as the baseline. The `recovered` metric needs a clean baseline and is
**400**'d for vector-sourced sweeps; the raw metrics (`logprob`, `logit_diff`)
work unchanged, and streaming (`stream: true`) behaves identically. Named
`source_module` names are validated against the server's steering registry
(400 with the available names on a miss).

`PatchStudy.ablation_sweep(prompt, source="zeros", mask={"indices": [...]},
layers=..., positions=...)` drives this from the client;
`PatchStudy.pack_vectors(array)` packs a numpy array into the `patch_vectors`
wire dict.

The `PatchStudy` client
(`vllm.entrypoints.serve.patch.client`) wraps capture + sweep for the
coarse→fine "walk" of finding the causal sites:

```python
from vllm.entrypoints.serve.patch.client import PatchStudy, Span
```

`examples/online_serving/openai_patch_client.py` is a runnable demo that imports
from that module.

### Multi-hook sweeps

The classic attention-vs-MLP decomposition sweeps the *same* `(layers ×
positions)` grid at several hooks. Two ways:

- **One request (`hooks` field).** Pass `hooks: [..]` (each injectable —
  `pre_attn`, `post_attn`, `post_block`, `mlp_in`, `mlp_out`; a bad or empty
  list is a 400) and the
  grid runs at every hook. `hooks` wins over the single `hook` field; the
  corrupt baseline and noise floor are computed **once** and shared across
  hooks (only the patched cells fan out per hook, concurrently). The response
  gains `hook_grids: [{hook, grid, argmax}, ..]` (one entry per hook, same
  order); the top-level `grid`/`hook`/`argmax` mirror the first hook so
  single-hook clients keep working. `positions`, `layers`, `clean`, `corrupt`,
  `noise_floor`, `alignment`, and `skipped` stay top-level and shared (each
  skipped cell carries its `hook`). `hooks=["post_block"]` matches
  `hook="post_block"` bar the extra `hook_grids` entry.

  `PatchStudy.sweep_layers_positions(..., server_side=True, hooks=[..])`
  returns `dict[str, SweepResult]` keyed by hook name (a single-hook sweep
  still returns one `SweepResult`). Passing `hooks` on the per-cell path
  (`server_side=False`) raises `ValueError` — run one hook at a time or set
  `server_side=True`.

- **Sequential (reuse one hook-complete run).** Auto-capture taps **all** three
  injectable hooks at the swept layers in one forward (see [One-call
  auto-capture](#one-call-auto-capture)), so a run kept with `keep_source=True`
  is reusable across follow-up single-hook sweeps at a *different* hook. Capture
  once, then sweep it at each hook in turn, dropping it at the end with the
  [DELETE route](#source-run-lifecycle).

### Substring positions

Sweep positions are token indices, but tokenization is easy to get wrong by
hand. A substring of the prompt can be given instead, resolved to the token
positions covering it — the prompt is tokenized exactly as the sweep tokenizes
it and each token is mapped to its character span, so the positions index the
prompt as the server sees it. A missing substring raises, and repeated matches
must be disambiguated with `occurrence` (default 0 = first).

This works from both surfaces:

- **Client:** pass a `Span("text", occurrence=0)` marker directly in a sweep's
  `positions` (mixed with plain indices); each span resolves against the corrupt
  prompt. `await study.positions_for(prompt, span, occurrence=0)` resolves one
  explicitly. Client-side resolution uses `/tokenize` + incremental
  `/detokenize` (the HTTP API exposes no char offsets). For `server_side=True`
  sweeps the spans are forwarded to the server and resolved there.
- **Raw HTTP:** `positions` accepts span objects mixed with integers, resolved
  server-side against `prompt` (the destination run):

  ```json
  {"positions": [{"span": "Germany", "occurrence": 0}, 4]}
  ```

  The response's `positions` is the resolved integer axis (grid columns index
  it). Server-side offsets come from the fast tokenizer's offset mapping (or
  incremental detokenization as a fallback), tokenized identically to the sweep
  so special tokens (e.g. BOS) map to an empty span and are never selected.
  Expansion is order-preserving with dedup across the whole list. Empty spans,
  missing substrings, and out-of-range occurrences are clean 400s.

The pure resolution math lives in `vllm/entrypoints/serve/patch/spans.py`,
shared by the endpoint and the client (no duplicated copy).

### One-call auto-capture

The common case — capture the clean run, then sweep the corrupt run against it —
collapses to a single request. When a sweep references a `source_run` that does
**not** yet exist in the source manifests *and* supplies `clean_prompt`, the
server captures the clean run itself before running the grid:

- The capture spec is derived from the grid: **all five injectable hooks**
  (`pre_attn`, `post_attn`, `post_block`, `mlp_in`, `mlp_out`) at the swept
  layer set, positions =
  `all_prompt` (one forward; the extra cost is only source-store bytes, which
  the [lifecycle](#source-run-lifecycle) makes reclaimable). Tapping every hook
  makes a *kept* run reusable for hook-comparison follow-up sweeps. It runs
  through the normal `patch_source` capture consumer, so `--capture-consumers
  patch_source` must be enabled.
- It uses `capture_wait` durability semantics internally, so the per-worker
  source store is populated before any cell resolves against it — the classic
  "forgot `capture_wait`, sweep 400s / silently no-ops" race is unrepresentable.
- The clean baseline for the `recovered` metric is graded from that same
  internal clean generation (exactly like the corrupt baseline, including exact
  `logprob_token_ids` grading), so the caller needn't pass `clean_baseline`.
- The response reports `auto_captured: true` and `captured_source_run` so
  callers can distinguish a one-call sweep from one reusing a prior run.

Existing runs are reused unchanged (no re-capture). A missing run captures under
the requested `source_run` name; the store is run-id keyed, so two concurrent
sweeps auto-capturing the same name are last-writer-wins (existence is
re-checked after capture via the manifest refresh-on-miss). If a referenced run
is missing and `clean_prompt` is **not** provided, the sweep still 400s
(capture the clean run explicitly first). Explicit pre-capture (client
`capture_clean` + `capture_wait`) keeps working unchanged.

`PatchStudy` wires this into one call: pass `clean_prompt` (the clean text) with
`server_side=True` and no captured `clean`/`run`, and the client skips capture
entirely — it generates a fresh per-call run name (auto-capture covers only the
swept layers, so a name reused across differing-layer grids would 400), sends
the sweep, and lets the server auto-capture. The fresh run is auto-dropped when
the sweep completes unless `keep_source=True` (see [lifecycle](#source-run-lifecycle)).
Spans are forwarded and resolved server-side. `SweepResult.auto_captured` /
`captured_source_run` report it.

```python
study = PatchStudy(model=MODEL, hook="post_block")
result = await study.sweep_layers_positions(
    corrupt_prompt,
    clean_prompt=clean_prompt,          # one call: server captures the clean run
    layers=range(20),
    positions=[Span("Germany")],        # resolved server-side, no token indices
    answer_token=" Paris",              # grade by the clean run's answer
    metric="recovered",
    server_side=True,
)
assert result.auto_captured  # captured_source_run is the fresh run name
```

An existing run wins: passing a captured `clean` handle (or explicit `run=`)
alongside `clean_prompt` reuses the run (the server re-captures nothing) and
`clean_prompt` then only drives alignment. The per-cell fan-out path
(`server_side=False`) has no capture endpoint, so `clean_prompt` there without a
captured `clean` handle raises — call `capture_clean` first.

### Source-run lifecycle

Clean runs live in the per-worker `PatchSourceStore` (whole-run LRU, budgeted by
`--patch-source-cache-bytes`). One-call sweeps mint a **fresh uuid run per
call**, so without cleanup they accumulate — and worse, a just-captured uuid run
sits at the LRU *most-recently-used* end, so budget pressure evicts a user's
older deliberate captures first. Three mechanisms manage this:

- **Auto-drop (default).** After a sweep that auto-captured completes (response
  assembled), the server drops the run it captured. `keep_source: true` retains
  it instead — the response's `captured_source_run` then names a reusable,
  hook-complete run (the sequential way to do a [multi-hook
  study](#multi-hook-sweeps)). A **pre-existing** run is never dropped (only a
  run this request auto-captured). Drop failures are logged, not request
  failures.
- **`DELETE /v1/patch_source/{run_id}`.** Frees a run from every worker's store
  (`collective_rpc("drop_patch_source_run", ..)`, unioned across PP ranks):
  `200 {"dropped": true}` if any rank held it, else `404`. An explicit owner
  drop succeeds **even if the run is leased** (unlike LRU eviction, which never
  touches a leased run); the lease is cleared with it. `PatchStudy.drop_run`
  wraps the route and returns a `bool` (`False` on 404 / failure).
- **Manifest-cache coherence.** The admission-side `_PatchSourceCache` caches
  run manifests; both drop paths invalidate it (`invalidate_patch_source_run`)
  so a dropped run is not reported as still-existing — otherwise the next sweep
  would skip auto-capture and 400 on the now-absent run.

### Streaming

A large grid (all_prompt × many layers on a real prompt = 1000+ cells) holds one
HTTP response open for minutes with no progress signal; a proxy/client timeout
then loses a nearly-finished sweep. Passing `"stream": true` opts into an
`text/event-stream` (SSE) response that emits results as cells land:

- `data: {"type": "start", "layers": [...], "positions": [...], "hook": ...,
  "metric": ..., "auto_captured": ..., "captured_source_run": ...}` — first, so
  a consumer can size a live heatmap before any cell arrives (a multi-hook
  sweep adds its `hooks` list).
- `data: {"type": "cell", "hook": "post_block", "layer": 14, "position": 3,
  "value": -0.37}` per cell as it completes (completion order — no ordering
  promise). Cell values are in the sweep metric's units — identical to the
  summary grid (for `recovered`, already normalized). `hook` is always present
  and, for a [multi-hook sweep](#multi-hook-sweeps), labels which hook's grid
  the cell belongs to.
  A cell that voids mid-sweep (its patch failed to resolve on the workers)
  re-emits as `{"type": "cell", ..., "value": null, "error": "..."}`, mirroring
  how voided cells land in `skipped`.
- `data: {"type": "summary", ...}` — the exact same payload as the non-streaming
  `PatchSweepResponse` (assembled by the same code path), then `data: [DONE]`.

**Pre-fan-out errors stay plain JSON.** Everything that 400s before the grid
fan-out begins — bad hook/layers, span-resolution failure, alignment failure,
missing source with no `clean_prompt` — still returns a normal JSON error
response; the stream only starts once the fan-out actually runs. So a client
must check the response `Content-Type` / status before parsing SSE.

Cells run concurrently regardless of streaming; if the client disconnects
mid-stream the server cancels the outstanding cell tasks (best-effort abort of
their engine requests) rather than grinding through a dead sweep.

`examples/online_serving/patch_dashboard/` is a runnable Dash dashboard built
on this stream: it drives a sweep from a browser form and renders the cell
events as a live-updating per-hook heatmap (with progress, cancel-on-close,
and the summary panel), consuming the SSE protocol above over plain
`requests` — no vllm import.

`PatchStudy.sweep_layers_positions(..., server_side=True, on_cell=fn)` wires this
in: passing an `on_cell(event_dict)` callback sends `stream: true`, invokes the
callback per cell event, and builds the returned `SweepResult` from the summary
event — identical to the non-streaming result. Without `on_cell` the sweep is
non-streaming (unchanged). `on_cell` is server-side only. It composes with
multi-hook sweeps: each cell event carries its own `hook`, and the streamed
summary's `hook_grids` matches the non-streaming multi-hook response.

### Position alignment

Patch entries pair a destination position (corrupt run) with a source position
(clean run). When the two prompts tokenize to **equal lengths**, corresponding
positions are the pairing (standard causal tracing) and nothing more is needed.
When they tokenize to **different lengths**, `source = dest` silently patches
shifted positions — so the sweep endpoint refuses a length mismatch unless
`clean_prompt` is provided, in which case positions are aligned automatically:
the common token prefix maps by identity, the common token suffix maps by the
length delta, and the differing middle (no positional correspondence) is
skipped with per-position entries in `skipped` plus an `alignment` summary in
the response. `PatchStudy` does the same client-side — importing the shared
`vllm.entrypoints.serve.patch.alignment` module (no duplicated copy) — when the
`CleanRun` handle (which records its prompt) is passed to the sweep.

### Reproducibility

vLLM is not batch-invariant by default: identical requests in different batch
compositions return slightly different logprobs, so sweep grids reproduce only
within a small tolerance. Each sweep response reports an empirical
`noise_floor` — the metric delta between the corrupt baseline run solo and
re-run inside the cell batch; grid differences at or below it are not
meaningful. For exact reproducibility, start the server in batch-invariant mode
(see [Batch Invariance](batch_invariance.md)); causal-tracing signal is
typically orders of magnitude above the default noise floor.

## Data / control flow

1. **Clean capture.** The `patch_source` consumer (`location="worker"`) taps the
   residual at each requested `(hook, layer)` and writes rows into a per-worker
   run-id-keyed `PatchSourceStore` (whole-run LRU eviction, budgeted by
   `--patch-source-cache-bytes`). Reuses the capture pipeline wholesale.
2. **Admission.** `_admit_patch` validates layer-in-range, hook injectability,
   and source existence (against a cached `get_patch_source_manifests()` engine
   RPC → HTTP 400 on a genuinely-missing run/site), and sets the prefix-cache
   floor so patched prompt positions (and after) are re-forwarded.
3. **Scheduler backpressure.** Per-`(layer, hook)` reserved-slot counts over the
   running batch admit a waiting request only if every touched site stays within
   the *usable* pool (`max_patch_slots - 1`; slot 0 is the passthrough
   sentinel — `PatchConfig.usable_slots`), so a step can never overflow the
   pool.
4. **Resolution.** On admission each worker resolves its request's spec against
   its local `PatchSourceStore`, keeping only locally-owned layers (PP);
   TP rank 0 broadcasts the resolved vectors to its peers.
5. **Per-step injection.** Before the forward, `_update_patch_buffers` projects
   the batch into `(abs_row = token_offset + (dest_pos - num_computed))` slots,
   stages source rows + alphas + the index scatter into the persistent
   per-`(layer, hook)` buffers. Ephemeral slots (reset each step, no refcount).
6. **Apply.** Folded into `apply_layer_steering` / `apply_block_steering`:
   `capture(pristine) → patch(replace/lerp) → steer(add)`. Disabled mode
   constant-folds out of the forward.

## Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --enable-patching \
  --max-patch-slots 64 \
  --patch-source-cache-bytes 2000000000
```

- `--enable-patching`: attach patch buffers + install the source store, and
  imply the `patch_source` capture consumer (below).
- `--max-patch-slots`: per-`(layer, hook)` patch pool size (default 64). Slot 0
  is the passthrough sentinel, so one fewer is usable per step
  (`PatchConfig.usable_slots`).
- `--patch-source-cache-bytes`: source-store byte budget.
- `--capture-consumers patch_source`: register the clean-capture consumer.
  Optional — `--enable-patching` implies it (identically, via the shared config
  finalization point, so both the server and offline `LLM(...)` get it). Passing
  it explicitly is still valid and never double-registers.

## Invariants

- **`post_block` reconstruction.** vLLM defers each layer's MLP-branch add into
  the next layer's fused add+norm, so at `post_block` the true block output is
  `residual + hidden_states`. Addition commutes through the deferred add (steering
  is correct on `residual` alone), but replace/lerp does not — so
  `apply_patch_block` reconstructs the block output, patches it, and writes back
  a residual that yields the intended block output while leaving `hidden_states`
  untouched. This matches what the `patch_source` consumer captures.
- **Precise lerp.** `(1-alpha)*h + alpha*source`, exact at `alpha == 1` (the
  naive `h + alpha*(source - h)` flakes from fp cancellation).
- **Per-`(layer, hook)` index.** Unlike steering, the patch index is *not* shared
  across layers — a request patches different positions at different layers.
- **Slot 0 = passthrough.** `alpha[0] ≡ 0`; token rows mapping to slot 0 are a
  no-op, keeping the kernel branch-free.
- **Strict capacity.** Overflow is prevented at admission + scheduler
  backpressure; the kernel never sees an out-of-range slot.
- **Buffer registration.** Patch buffers self-register during the model build by
  reading the slot count from the ambient `VllmConfig` context
  (`get_current_vllm_config_or_none`), so any runner that builds a model gets
  them for free; the `set_patch_buffer_slots` global is a test-only fallback.
  With slots 0 the apply path constant-folds out.

## Related files

- Data plane: `vllm/model_executor/layers/patch.py`, `patch_kernel.py`
  (two ops + Triton kernels), folded into `steering.py`.
- Injection plane: `vllm/v1/worker/patch_runner_mixin.py` (runner-agnostic) +
  `vllm/v1/worker/gpu/patch_runner_mixin.py` (v2) + v1 wiring in
  `vllm/v1/worker/gpu_model_runner.py`.
- Source store: `vllm/v1/capture/source_store.py`,
  `vllm/v1/capture/consumers/patch_source.py`,
  `vllm/v1/worker/gpu/patch_resolve.py`.
- Config / admission: `vllm/config/patch.py`, `vllm/sampling_params.py`
  (`patch` field), `vllm/v1/capture/patch_admission.py`,
  `vllm/v1/engine/input_processor.py` (`_resolve_patch_prefix_flags` — offline /
  Rust / non-OpenAI engine-side admission), `vllm/v1/request.py` (prefix floor),
  `vllm/v1/core/sched/scheduler.py` (backpressure).
- Endpoint / client: `vllm/entrypoints/serve/patch/` (endpoint:
  `api_router.py`; request/response: `protocol.py`; client: `client.py`;
  shared substring→position math: `spans.py`; position alignment:
  `alignment.py`; runnable demo:
  `examples/online_serving/openai_patch_client.py`; live-heatmap Dash
  dashboard: `examples/online_serving/patch_dashboard/`).
- GPU validation: `tests/gpu_patch_validate.py`.
