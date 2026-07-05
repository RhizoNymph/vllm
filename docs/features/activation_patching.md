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

- **Hook points:** `pre_attn`, `post_attn`, `post_block` (the injection-capable
  `SteeringHookPoint`s). `post_block` uses a dedicated two-tensor op because
  vLLM defers each layer's MLP-branch add — see [Invariants](#invariants).
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

End-to-end GPU-validated with real weights (TP1/PP1 eager + cudagraph; the
multi-rank configs eager):

- **Qwen3-0.6B** (v2 runner, default): TP1/PP1, TP2/PP1, TP1/PP2.
- **gemma3-4b** (v1 runner): TP1/PP1, TP2/PP1, TP1/PP2.

Both prove no-op / self-identity are bit-exact, cross-run replace reproduces the
clean run within bf16 accumulation, and single-site denoising surfaces the clean
answer.

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
  `pre_attn`, `post_attn`, `post_block`; a bad or empty list is a 400) and the
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

- The capture spec is derived from the grid: **all three injectable hooks**
  (`pre_attn`, `post_attn`, `post_block`) at the swept layer set, positions =
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
  (`patch` field), `vllm/v1/capture/patch_admission.py`, `vllm/v1/request.py`
  (prefix floor), `vllm/v1/core/sched/scheduler.py` (backpressure).
- Endpoint / client: `vllm/entrypoints/serve/patch/` (endpoint:
  `api_router.py`; request/response: `protocol.py`; client: `client.py`;
  shared substring→position math: `spans.py`; position alignment:
  `alignment.py`; runnable demo:
  `examples/online_serving/openai_patch_client.py`).
- GPU validation: `tests/gpu_patch_validate.py`.
