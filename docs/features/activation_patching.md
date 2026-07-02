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
  steering does — zero per-model changes).
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
- **gemma3-4b** (v1 runner): TP1/PP1.

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
(HTTP) so the source store is durable before any patch references it.

## Server-side sweeps

`POST /v1/patch_sweep` runs a whole `(layers × positions)` grid in one call: the
server expands the grid into one patched variant per cell, fans them to the
continuously-batched engine (the shared corrupt-prompt prefix is reused via
prefix caching; each variant is patched at its own site via per-row gating), and
returns the assembled metric grid. Metrics: `logprob`, `logit_diff`
(answer − foil), `recovered` `(patched − corrupt) / (clean − corrupt)`.

The `PatchStudy` client
(`examples/online_serving/openai_patch_client.py`) wraps capture + sweep for the
coarse→fine "walk" of finding the causal sites.

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
the response. `PatchStudy` does the same client-side when the `CleanRun` handle
(which records its prompt) is passed to the sweep.

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
  --patch-source-cache-bytes 2000000000 \
  --capture-consumers patch_source
```

- `--enable-patching`: attach patch buffers + install the source store.
- `--max-patch-slots`: per-`(layer, hook)` patch pool size (default 64).
- `--patch-source-cache-bytes`: source-store byte budget.
- `--capture-consumers patch_source`: register the clean-capture consumer.

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
- **Buffer registration.** Patch buffers attach at model build via the
  process-global slot count (set by `set_patch_buffer_slots` before the model is
  built, on both runners). With slots 0 the apply path constant-folds out.

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
- Endpoint / client: `vllm/entrypoints/serve/patch/`,
  `examples/online_serving/openai_patch_client.py`.
- GPU validation: `tests/gpu_patch_validate.py`.
