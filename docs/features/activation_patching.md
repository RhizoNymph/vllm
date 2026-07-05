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
  grid into one densely-batched call.

End-to-end GPU-validated with real weights (eager + cudagraph):

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

#### One-call auto-capture

The common case — capture the clean run, then sweep the corrupt run against it —
collapses to a single request. When a sweep references a `source_run` that does
**not** yet exist in the source manifests *and* supplies `clean_prompt`, the
server captures the clean run itself before running the grid:

- The capture spec is derived from the grid: hook = the sweep's `hook`, layers =
  the swept layer set, positions = `all_prompt` (mirrors the client's
  `capture_clean`). It runs through the normal `patch_source` capture consumer,
  so `--capture-consumers patch_source` must be enabled.
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
entirely — it generates a fresh per-call run name (the auto-capture taps only
the swept layers, so a name reused across differing-layer grids would 400), sends
the sweep, and lets the server auto-capture. Spans are forwarded and resolved
server-side. `SweepResult.auto_captured` / `captured_source_run` report it.

```python
study = PatchStudy(model=MODEL, hook="post_block")
result = await study.sweep_layers_positions(
    corrupt_prompt,
    clean_prompt=clean_prompt,          # one call: server captures the clean run
    layers=range(20),
    positions=[Span("Germany")],        # resolved server-side, no token indices
    answer_token=" Berlin",
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

### Streaming

A large grid (all_prompt × many layers on a real prompt = 1000+ cells) holds one
HTTP response open for minutes with no progress signal; a proxy/client timeout
then loses a nearly-finished sweep. Passing `"stream": true` opts into an
`text/event-stream` (SSE) response that emits results as cells land:

- `data: {"type": "start", "layers": [...], "positions": [...], "hook": ...,
  "metric": ..., "auto_captured": ..., "captured_source_run": ...}` — first, so
  a consumer can size a live heatmap before any cell arrives.
- `data: {"type": "cell", "hook": "post_block", "layer": 14, "position": 3,
  "value": -0.37}` per cell as it completes (completion order — no ordering
  promise). `hook` is always present (a sibling branch adds multi-hook sweeps).
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
non-streaming (unchanged). `on_cell` is server-side only.

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
   running batch admit a waiting request only if every touched site stays ≤
   `--max-patch-slots`, so a step can never overflow the pool.
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
- `--max-patch-slots`: per-`(layer, hook)` patch pool size (default 64).
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
